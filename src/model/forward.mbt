///|
/// What one forward pass produces for one question.
pub struct Outputs {
  /// Raw scorer logits, one per option slot. Always at least two long: a
  /// one-option choice is padded with a `-1e4` slot, matching the reference
  /// runtime, because the action head's top-two feature needs two slots.
  logits : Array[Double]
  /// How many of those slots are real options.
  options : Int
  /// Raw action-head logits.
  action : Array[Double]
} derive(Debug)

///|
/// Value written into padded option slots, matching the reference runtime.
/// After the softmax shift this is indistinguishable from `-inf`.
const MASKED_LOGIT : Double = -1.0e4

///|
/// Value used to mask out-of-window attention scores. Large enough that
/// `exp(x - rowmax)` underflows to exactly zero, so it behaves as `-inf`
/// without risking a NaN if a row were ever fully masked.
const MASKED_SCORE : Float = -1.0e30

///|
/// Scratch buffers for a single sequence.
///
/// One question is run per forward pass rather than a padded batch. The
/// reference runtime pads a batch and then masks the padding out of every
/// attention row, so results for real tokens are identical either way — and
/// without padding there is no attention mask to carry around, and peak memory
/// stays proportional to the actual sequence length.
priv struct Scratch {
  hidden : FixedArray[Float]
  normed : FixedArray[Float]
  qkv : FixedArray[Float]
  q : FixedArray[Float]
  k : FixedArray[Float]
  v : FixedArray[Float]
  scores : FixedArray[Float]
  context : FixedArray[Float]
  packed : FixedArray[Float]
  projected : FixedArray[Float]
  wide : FixedArray[Float]
  gated : FixedArray[Float]
}

///|
fn Scratch::new(model : Model, length : Int) -> Scratch {
  let hidden = model.encoder.hidden_size
  let inner = model.encoder.intermediate_size
  // The decision head's feed-forward is 4x hidden, wider than the encoder's
  // 2x intermediate on these checkpoints, so `wide` is sized for whichever is
  // larger and shared between them.
  let widest = @cmp.maximum(2 * inner, 4 * hidden)
  let heads = @cmp.maximum(model.encoder.num_attention_heads, model.head_heads)
  {
    hidden: @tensor.zeros(length * hidden),
    normed: @tensor.zeros(length * hidden),
    qkv: @tensor.zeros(length * 3 * hidden),
    q: @tensor.zeros(length * hidden),
    k: @tensor.zeros(length * hidden),
    v: @tensor.zeros(length * hidden),
    scores: @tensor.zeros(heads * length * length),
    context: @tensor.zeros(length * hidden),
    packed: @tensor.zeros(length * hidden),
    projected: @tensor.zeros(length * hidden),
    wide: @tensor.zeros(length * widest),
    gated: @tensor.zeros(length * @cmp.maximum(inner, 4 * hidden)),
  }
}

///|
/// Run one question through the encoder, the decision head, the scorer and the
/// action head.
///
/// `markers` holds the position of each option's `[MASK]` token in `ids`, and
/// `qtype` selects the learned type embedding (0 choice, 1 score, 2 noul).
///
/// Passing `trace` records each stage's activations under a descriptive key;
/// it exists for the parity tests and costs a copy per stage when supplied.
pub fn Model::forward(
  self : Model,
  ids : Array[Int],
  markers : Array[Int],
  qtype : Int,
  trace? : Map[String, Array[Double]],
) -> Outputs raise @error.LayaError {
  self.check_inputs(ids, markers, qtype)
  let options = markers.length()
  let slots = @cmp.maximum(options, 2)
  let (logits, cls) = match self.metal {
    Some(state) => self.forward_metal(state, ids, markers, qtype, trace)
    None => self.forward_cpu(ids, markers, qtype, trace)
  }
  let action = self.action_logits(cls, logits, options, slots)
  { logits, options, action, }
}

///|
/// Reject inputs neither backend can run.
fn Model::check_inputs(
  self : Model,
  ids : Array[Int],
  markers : Array[Int],
  qtype : Int,
) -> Unit raise @error.LayaError {
  let length = ids.length()
  @error.expect(length > 0, "cannot run an empty sequence")
  @error.expect(
    length <= self.encoder.max_position_embeddings,
    "sequence of \{length} tokens exceeds the encoder's \{self.encoder.max_position_embeddings}-position limit",
  )
  @error.expect(
    qtype >= 0 && qtype < 3,
    "question type must be 0, 1 or 2, got \{qtype}",
  )
  @error.expect(markers.length() > 0, "a question needs at least one option")
  for marker in markers {
    @error.expect(
      marker >= 0 && marker < length,
      "option marker \{marker} falls outside a \{length}-token sequence",
    )
  }
}

///|
/// Run one sequence on the CPU, returning the option logits and the pooled CLS
/// row the action head needs.
fn Model::forward_cpu(
  self : Model,
  ids : Array[Int],
  markers : Array[Int],
  qtype : Int,
  trace : Map[String, Array[Double]]?,
) -> (Array[Double], FixedArray[Float]) raise @error.LayaError {
  let length = ids.length()
  let hidden = self.encoder.hidden_size
  let scratch = Scratch::new(self, length)

  // Embeddings: a row gather straight out of the mapped checkpoint, then a
  // norm. The embedding matrix is never converted to float32 wholesale.
  let rows = FixedArray::make(length, 0)
  for index, id in ids {
    @error.expect(
      id >= 0 && id < self.encoder.vocab_size,
      "token id \{id} is outside the \{self.encoder.vocab_size}-entry vocabulary",
    )
    rows[index] = id
  }
  self.file.gather_rows(self.tok_embeddings, rows, scratch.normed)
  @tensor.layer_norm(
    scratch.normed,
    scratch.hidden,
    rows=length,
    cols=hidden,
    self.embed_norm.values,
    None,
    eps=self.encoder.norm_eps,
  )
  record(trace, "embeddings", scratch.hidden, length * hidden)
  for index, layer in self.layers {
    self.encoder_layer(scratch, layer, length)
    record(trace, "encoder_layer_\{index}", scratch.hidden, length * hidden)
  }
  @tensor.layer_norm(
    scratch.hidden,
    scratch.normed,
    rows=length,
    cols=hidden,
    self.final_norm.values,
    None,
    eps=self.encoder.norm_eps,
  )
  scratch.normed.blit_to(scratch.hidden, len=length * hidden)
  record(trace, "encoder_out", scratch.hidden, length * hidden)

  // The question type is a single learned vector added to every position.
  for position in 0..<length {
    let base = position * hidden
    for i in 0..<hidden {
      scratch.hidden[base + i] = scratch.hidden[base + i] +
        self.type_emb.values[qtype * hidden + i]
    }
  }
  record(trace, "with_type_emb", scratch.hidden, length * hidden)
  for index, layer in self.head {
    self.head_layer(scratch, layer, length)
    record(trace, "head_layer_\{index}", scratch.hidden, length * hidden)
  }

  // Score the option markers. Slots are padded to two so the action head's
  // top-two features are always defined.
  let options = markers.length()
  let slots = @cmp.maximum(options, 2)
  let marker_rows = @tensor.zeros(options * hidden)
  for slot, marker in markers {
    scratch.hidden.blit_to(
      marker_rows,
      len=hidden,
      src_offset=marker * hidden,
      dst_offset=slot * hidden,
    )
  }
  record(trace, "marker_rows", marker_rows, options * hidden)
  let logits = self.score_markers(marker_rows, options, slots)
  record_doubles(trace, "scorer_logits", logits[:options])
  let cls = @tensor.zeros(hidden)
  scratch.hidden.blit_to(cls, len=hidden)
  (logits, cls)
}

///|
/// `x = x + attn(attn_norm(x))` then `x = x + mlp(mlp_norm(x))`.
fn Model::encoder_layer(
  self : Model,
  scratch : Scratch,
  layer : EncoderLayer,
  length : Int,
) -> Unit {
  let hidden = self.encoder.hidden_size
  let inner = self.encoder.intermediate_size
  let heads = self.encoder.num_attention_heads
  let dim = self.encoder.head_dim
  match layer.attn_norm {
    Some(norm) =>
      @tensor.layer_norm(
        scratch.hidden,
        scratch.normed,
        rows=length,
        cols=hidden,
        norm.values,
        None,
        eps=self.encoder.norm_eps,
      )
    // Layer 0 substitutes an identity for the pre-attention norm.
    None => scratch.hidden.blit_to(scratch.normed, len=length * hidden)
  }
  @tensor.linear(
    scratch.normed,
    self.arena,
    layer.wqkv,
    -1L,
    scratch.qkv,
    m=length,
    k=hidden,
    n=3 * hidden,
  )
  split_qkv(scratch, length, heads, dim)
  @tensor.rope_inplace(
    scratch.q,
    heads~,
    len=length,
    dim~,
    base=layer.rope_base,
  )
  @tensor.rope_inplace(
    scratch.k,
    heads~,
    len=length,
    dim~,
    base=layer.rope_base,
  )
  attend(
    scratch,
    length,
    heads,
    dim,
    match layer.attention {
      Global => -1
      Sliding => self.encoder.local_attention / 2
    },
  )
  @tensor.linear(
    scratch.packed,
    self.arena,
    layer.attn_wo,
    -1L,
    scratch.projected,
    m=length,
    k=hidden,
    n=hidden,
  )
  @tensor.add_inplace(scratch.hidden, scratch.projected, len=length * hidden)

  // GeGLU feed-forward: Wi emits value and gate side by side.
  @tensor.layer_norm(
    scratch.hidden,
    scratch.normed,
    rows=length,
    cols=hidden,
    layer.mlp_norm.values,
    None,
    eps=self.encoder.norm_eps,
  )
  @tensor.linear(
    scratch.normed,
    self.arena,
    layer.mlp_wi,
    -1L,
    scratch.wide,
    m=length,
    k=hidden,
    n=2 * inner,
  )
  @tensor.geglu(scratch.wide, scratch.gated, rows=length, inner~)
  @tensor.linear(
    scratch.gated,
    self.arena,
    layer.mlp_wo,
    -1L,
    scratch.projected,
    m=length,
    k=inner,
    n=hidden,
  )
  @tensor.add_inplace(scratch.hidden, scratch.projected, len=length * hidden)
}

///|
/// `x = x + attn(norm1(x))` then `x = x + linear2(relu(linear1(norm2(x))))`.
fn Model::head_layer(
  self : Model,
  scratch : Scratch,
  layer : HeadLayer,
  length : Int,
) -> Unit {
  let hidden = self.encoder.hidden_size
  let heads = self.head_heads
  let dim = self.head_dim
  @tensor.layer_norm(
    scratch.hidden,
    scratch.normed,
    rows=length,
    cols=hidden,
    layer.norm1_weight.values,
    Some(layer.norm1_bias.values),
    eps=1.0e-5,
  )
  @tensor.linear(
    scratch.normed,
    self.arena,
    layer.in_proj,
    layer.in_proj_bias,
    scratch.qkv,
    m=length,
    k=hidden,
    n=3 * hidden,
  )
  split_qkv(scratch, length, heads, dim)
  // No rotary embedding and no masking here: the head attends over the whole
  // sequence, which for an unpadded batch of one means no mask at all.
  attend(scratch, length, heads, dim, -1)
  @tensor.linear(
    scratch.packed,
    self.arena,
    layer.out_proj,
    layer.out_proj_bias,
    scratch.projected,
    m=length,
    k=hidden,
    n=hidden,
  )
  @tensor.add_inplace(scratch.hidden, scratch.projected, len=length * hidden)
  @tensor.layer_norm(
    scratch.hidden,
    scratch.normed,
    rows=length,
    cols=hidden,
    layer.norm2_weight.values,
    Some(layer.norm2_bias.values),
    eps=1.0e-5,
  )
  @tensor.linear(
    scratch.normed,
    self.arena,
    layer.linear1,
    layer.linear1_bias,
    scratch.wide,
    m=length,
    k=hidden,
    n=4 * hidden,
  )
  @tensor.relu_inplace(scratch.wide, len=length * 4 * hidden)
  @tensor.linear(
    scratch.wide,
    self.arena,
    layer.linear2,
    layer.linear2_bias,
    scratch.projected,
    m=length,
    k=4 * hidden,
    n=hidden,
  )
  @tensor.add_inplace(scratch.hidden, scratch.projected, len=length * hidden)
}

///|
/// Unpack a `[len, 3, heads, dim]` projection into three `[heads, len, dim]`
/// buffers, which is the layout the per-head GEMMs want.
fn split_qkv(scratch : Scratch, length : Int, heads : Int, dim : Int) -> Unit {
  let stride = 3 * heads * dim
  for position in 0..<length {
    let src = position * stride
    for head in 0..<heads {
      let dst = (head * length + position) * dim
      let offset = head * dim
      for i in 0..<dim {
        scratch.q[dst + i] = scratch.qkv[src + offset + i]
        scratch.k[dst + i] = scratch.qkv[src + heads * dim + offset + i]
        scratch.v[dst + i] = scratch.qkv[src + 2 * heads * dim + offset + i]
      }
    }
  }
}

///|
/// Scaled dot-product attention over the packed per-head buffers, leaving the
/// result in `scratch.packed` in `[len, heads · dim]` layout.
///
/// `window` is the inclusive distance a position may attend over, or `-1` for
/// unrestricted attention.
fn attend(
  scratch : Scratch,
  length : Int,
  heads : Int,
  dim : Int,
  window : Int,
) -> Unit {
  let scale = Float::from_double(1.0 / Double::sqrt(dim.to_double()))
  for head in 0..<heads {
    @tensor.gemm(
      scratch.q,
      scratch.k,
      scratch.scores,
      m=length,
      n=length,
      k=dim,
      a_offset=head * length * dim,
      b_offset=head * length * dim,
      c_offset=head * length * length,
      lda=dim,
      ldb=dim,
      ldc=length,
      trans_b=true,
      alpha=scale,
    )
  }
  if window >= 0 {
    for head in 0..<heads {
      let base = head * length * length
      for i in 0..<length {
        let row = base + i * length
        for j in 0..<length {
          if (i - j).abs() > window {
            scratch.scores[row + j] = MASKED_SCORE
          }
        }
      }
    }
  }
  @tensor.softmax_rows(scratch.scores, rows=heads * length, cols=length)
  for head in 0..<heads {
    @tensor.gemm(
      scratch.scores,
      scratch.v,
      scratch.context,
      m=length,
      n=dim,
      k=length,
      a_offset=head * length * length,
      b_offset=head * length * dim,
      c_offset=head * length * dim,
      lda=length,
      ldb=dim,
      ldc=dim,
    )
  }
  for head in 0..<heads {
    for position in 0..<length {
      let src = (head * length + position) * dim
      let dst = position * heads * dim + head * dim
      for i in 0..<dim {
        scratch.packed[dst + i] = scratch.context[src + i]
      }
    }
  }
}

///|
/// LayerNorm, Linear, GELU, Linear-to-one over the option marker rows.
fn Model::score_markers(
  self : Model,
  marker_rows : FixedArray[Float],
  options : Int,
  slots : Int,
) -> Array[Double] {
  let hidden = self.encoder.hidden_size
  let normed = @tensor.zeros(options * hidden)
  @tensor.layer_norm(
    marker_rows,
    normed,
    rows=options,
    cols=hidden,
    self.scorer_norm_weight.values,
    Some(self.scorer_norm_bias.values),
    eps=1.0e-5,
  )
  let dense = @tensor.zeros(options * hidden)
  @tensor.linear(
    normed,
    self.arena,
    self.scorer_dense,
    self.scorer_dense_bias,
    dense,
    m=options,
    k=hidden,
    n=hidden,
  )
  @tensor.gelu_inplace(dense, len=options * hidden)
  let scores = @tensor.zeros(options)
  @tensor.linear(
    dense,
    self.arena,
    self.scorer_out,
    self.scorer_out_bias,
    scores,
    m=options,
    k=hidden,
    n=1,
  )
  let logits = Array::make(slots, MASKED_LOGIT)
  for slot in 0..<options {
    logits[slot] = scores[slot].to_double()
  }
  logits
}

///|
/// Run the action head over the pooled CLS row plus four calibration features.
///
/// This stays on the CPU on both backends: it is two multiplies against a
/// single row, and the features in between come from a softmax over the option
/// logits, which would need a round trip anyway.
///
/// The features are derived from the *uncalibrated* option distribution: the
/// top probability, its margin over the runner-up, the entropy normalised by
/// `log(max(options, 2))`, and the option count scaled by 255.
fn Model::action_logits(
  self : Model,
  cls : FixedArray[Float],
  logits : Array[Double],
  options : Int,
  slots : Int,
) -> Array[Double] {
  let hidden = self.encoder.hidden_size
  let probabilities = @tensor.softmax(logits[:])
  let effective = @cmp.maximum(options, 2).to_double()
  let mut entropy = 0.0
  for probability in probabilities {
    entropy -= probability * @math.ln(@cmp.maximum(probability, 1.0e-9))
  }
  entropy /= @math.ln(effective)
  let sorted = probabilities.copy()
  sorted.sort()
  let best = sorted[slots - 1]
  let runner_up = sorted[slots - 2]
  let pooled = @tensor.zeros(hidden + 4)
  cls.blit_to(pooled, len=hidden)
  pooled[hidden] = Float::from_double(best)
  pooled[hidden + 1] = Float::from_double(best - runner_up)
  pooled[hidden + 2] = Float::from_double(entropy)
  pooled[hidden + 3] = Float::from_double(effective / 255.0)
  let projected = @tensor.zeros(self.act_hidden)
  @tensor.linear(
    pooled,
    self.arena,
    self.act_in,
    self.act_in_bias,
    projected,
    m=1,
    k=hidden + 4,
    n=self.act_hidden,
  )
  @tensor.gelu_inplace(projected, len=self.act_hidden)
  let out = @tensor.zeros(self.act_outputs)
  @tensor.linear(
    projected,
    self.arena,
    self.act_out,
    self.act_out_bias,
    out,
    m=1,
    k=self.act_hidden,
    n=self.act_outputs,
  )
  let action = Array::make(self.act_outputs, 0.0)
  for index in 0..<self.act_outputs {
    action[index] = out[index].to_double()
  }
  action
}

///|
fn record(
  trace : Map[String, Array[Double]]?,
  key : String,
  values : FixedArray[Float],
  length : Int,
) -> Unit {
  guard trace is Some(trace) else { return }
  let out = Array::make(length, 0.0)
  for index in 0..<length {
    out[index] = values[index].to_double()
  }
  trace[key] = out
}

///|
fn record_doubles(
  trace : Map[String, Array[Double]]?,
  key : String,
  values : ArrayView[Double],
) -> Unit {
  guard trace is Some(trace) else { return }
  trace[key] = values.to_owned()
}
