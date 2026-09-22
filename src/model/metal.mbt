///|
/// GPU-resident state: the context, the weight arena exposed to Metal, and a
/// set of scratch buffers sized for the longest sequence seen so far.
priv struct MetalState {
  context : @tensor.MetalContext
  weights : @tensor.MetalBuffer
  mut scratch : MetalScratch?
}

///|
/// The GPU mirror of `Scratch`, plus the buffers the scorer needs.
///
/// Reallocating sixteen buffers per call costs more than the pass itself, so
/// the set is kept and only ever grows: `capacity` and `options` record the
/// high-water marks it has been sized for.
priv struct MetalScratch {
  capacity : Int
  options : Int
  hidden : @tensor.MetalBuffer
  normed : @tensor.MetalBuffer
  qkv : @tensor.MetalBuffer
  q : @tensor.MetalBuffer
  k : @tensor.MetalBuffer
  v : @tensor.MetalBuffer
  scores : @tensor.MetalBuffer
  context_buffer : @tensor.MetalBuffer
  packed : @tensor.MetalBuffer
  projected : @tensor.MetalBuffer
  wide : @tensor.MetalBuffer
  gated : @tensor.MetalBuffer
  marker_indices : @tensor.MetalBuffer
  marker_rows : @tensor.MetalBuffer
  scorer_normed : @tensor.MetalBuffer
  scorer_dense : @tensor.MetalBuffer
  staging : FixedArray[Float]
}

///|
/// Attach a Metal context, moving subsequent passes onto the GPU.
///
/// The weight arena is exposed to the GPU without being copied, so this is
/// cheap: it compiles the kernels and wraps memory that already exists.
pub fn Model::use_metal(
  self : Model,
  context : @tensor.MetalContext,
) -> Unit raise @error.LayaError {
  let weights = context.wrap_arena(self.arena)
  self.metal = Some({ context, weights, scratch: None, })
}

///|
/// Whether passes currently run on the GPU.
pub fn Model::on_metal(self : Model) -> Bool {
  self.metal is Some(_)
}

///|
/// Fall back to the CPU backend, releasing the GPU buffers.
pub fn Model::use_cpu(self : Model) -> Unit {
  self.metal = None
}

///|
/// Grow the scratch set if this sequence needs more room than the last one.
fn MetalState::ensure(
  self : MetalState,
  model : Model,
  length : Int,
  options : Int,
) -> MetalScratch raise @error.LayaError {
  match self.scratch {
    Some(scratch) if scratch.capacity >= length && scratch.options >= options =>
      scratch
    _ => {
      // Grow to the high-water mark on both axes. Sizing to exactly what this
      // call needs would thrash: a batch of questions of differing lengths and
      // option counts would shrink one axis while growing the other, and
      // reallocate again on the next question.
      let length = match self.scratch {
        Some(scratch) => @cmp.maximum(scratch.capacity, length)
        None => length
      }
      let options = match self.scratch {
        Some(scratch) => @cmp.maximum(scratch.options, options)
        None => options
      }
      let hidden = model.encoder.hidden_size
      let inner = model.encoder.intermediate_size
      let widest = @cmp.maximum(2 * inner, 4 * hidden)
      let heads = @cmp.maximum(
        model.encoder.num_attention_heads,
        model.head_heads,
      )
      let gpu = self.context
      let scratch = {
        capacity: length,
        options,
        hidden: gpu.buffer(length * hidden),
        normed: gpu.buffer(length * hidden),
        qkv: gpu.buffer(length * 3 * hidden),
        q: gpu.buffer(length * hidden),
        k: gpu.buffer(length * hidden),
        v: gpu.buffer(length * hidden),
        scores: gpu.buffer(heads * length * length),
        context_buffer: gpu.buffer(length * hidden),
        packed: gpu.buffer(length * hidden),
        projected: gpu.buffer(length * hidden),
        wide: gpu.buffer(length * widest),
        gated: gpu.buffer(length * @cmp.maximum(inner, 4 * hidden)),
        marker_indices: gpu.buffer(options),
        marker_rows: gpu.buffer(options * hidden),
        scorer_normed: gpu.buffer(options * hidden),
        scorer_dense: gpu.buffer(options * hidden),
        staging: @tensor.zeros(length * hidden),
      }
      self.scratch = Some(scratch)
      scratch
    }
  }
}

///|
/// Encode and run one sequence on the GPU.
///
/// The whole encoder, the decision head and the scorer's first three stages go
/// into a single command buffer, so there is exactly one submission and one
/// wait per question. What comes back is the scorer's pre-output activations
/// and the pooled CLS row; the last projection and the action head are a
/// handful of multiply-accumulates and stay on the CPU, which also avoids a
/// second round trip to compute the action head's calibration features.
fn Model::forward_metal(
  self : Model,
  state : MetalState,
  ids : Array[Int],
  markers : Array[Int],
  qtype : Int,
  trace : Map[String, Array[Double]]?,
) -> (Array[Double], FixedArray[Float]) raise @error.LayaError {
  let length = ids.length()
  let hidden = self.encoder.hidden_size
  let inner = self.encoder.intermediate_size
  let options = markers.length()
  let slots = @cmp.maximum(options, 2)
  let scratch = state.ensure(self, length, options)
  let gpu = state.context
  let weights = state.weights

  // Embeddings are gathered on the CPU: the token embedding is the largest
  // tensor in the checkpoint and is only ever indexed, so it stays on disk
  // rather than being uploaded.
  let rows = FixedArray::make(length, 0)
  for index, id in ids {
    @error.expect(
      id >= 0 && id < self.encoder.vocab_size,
      "token id \{id} is outside the \{self.encoder.vocab_size}-entry vocabulary",
    )
    rows[index] = id
  }
  self.file.gather_rows(self.tok_embeddings, rows, scratch.staging)
  scratch.normed.upload(scratch.staging, count=length * hidden)
  let marker_indices = FixedArray::make(options, 0)
  for slot, marker in markers {
    marker_indices[slot] = marker
  }
  scratch.marker_indices.upload_ints(marker_indices, count=options)
  gpu.begin()
  gpu.layer_norm(
    scratch.normed,
    scratch.hidden,
    weights,
    weights,
    rows=length,
    cols=hidden,
    gamma_offset=self.embed_norm.offset,
    beta_offset=-1L,
    eps=self.encoder.norm_eps,
  )
  for layer in self.layers {
    // Layer 0 has an identity where the pre-attention norm would be, so the
    // residual stream feeds the projection directly.
    let source = match layer.attn_norm {
      Some(norm) => {
        gpu.layer_norm(
          scratch.hidden,
          scratch.normed,
          weights,
          weights,
          rows=length,
          cols=hidden,
          gamma_offset=norm.offset,
          beta_offset=-1L,
          eps=self.encoder.norm_eps,
        )
        scratch.normed
      }
      None => scratch.hidden
    }
    gpu.gemm(
      source,
      weights,
      scratch.qkv,
      m=length,
      n=3 * hidden,
      k=hidden,
      lda=hidden,
      ldb=hidden,
      ldc=3 * hidden,
      b_offset=layer.wqkv,
      trans_b=true,
    )
    gpu.split_qkv(
      scratch.qkv,
      scratch.q,
      scratch.k,
      scratch.v,
      heads=self.encoder.num_attention_heads,
      len=length,
      dim=self.encoder.head_dim,
    )
    for rotated in [scratch.q, scratch.k] {
      gpu.rope(
        rotated,
        heads=self.encoder.num_attention_heads,
        len=length,
        dim=self.encoder.head_dim,
        base=layer.rope_base,
      )
    }
    self.attend_metal(
      gpu,
      scratch,
      length,
      self.encoder.num_attention_heads,
      self.encoder.head_dim,
      match layer.attention {
        Global => -1
        Sliding => self.encoder.local_attention / 2
      },
    )
    gpu.gemm(
      scratch.packed,
      weights,
      scratch.projected,
      m=length,
      n=hidden,
      k=hidden,
      lda=hidden,
      ldb=hidden,
      ldc=hidden,
      b_offset=layer.attn_wo,
      trans_b=true,
    )
    gpu.add(scratch.hidden, scratch.projected, count=length * hidden)
    gpu.layer_norm(
      scratch.hidden,
      scratch.normed,
      weights,
      weights,
      rows=length,
      cols=hidden,
      gamma_offset=layer.mlp_norm.offset,
      beta_offset=-1L,
      eps=self.encoder.norm_eps,
    )
    gpu.gemm(
      scratch.normed,
      weights,
      scratch.wide,
      m=length,
      n=2 * inner,
      k=hidden,
      lda=hidden,
      ldb=hidden,
      ldc=2 * inner,
      b_offset=layer.mlp_wi,
      trans_b=true,
    )
    gpu.geglu(scratch.wide, scratch.gated, rows=length, inner~)
    gpu.gemm(
      scratch.gated,
      weights,
      scratch.projected,
      m=length,
      n=hidden,
      k=inner,
      lda=inner,
      ldb=inner,
      ldc=hidden,
      b_offset=layer.mlp_wo,
      trans_b=true,
    )
    gpu.add(scratch.hidden, scratch.projected, count=length * hidden)
  }
  gpu.layer_norm(
    scratch.hidden,
    scratch.normed,
    weights,
    weights,
    rows=length,
    cols=hidden,
    gamma_offset=self.final_norm.offset,
    beta_offset=-1L,
    eps=self.encoder.norm_eps,
  )

  // `normed` now holds the encoder output. Rather than copy it back, the type
  // embedding is added into it and it becomes the residual stream for the head.
  gpu.add_broadcast(
    scratch.normed,
    weights,
    rows=length,
    cols=hidden,
    v_offset=self.type_emb.offset + (qtype * hidden).to_int64(),
  )
  let stream = scratch.normed
  let staging = scratch.hidden
  for layer in self.head {
    gpu.layer_norm(
      stream,
      staging,
      weights,
      weights,
      rows=length,
      cols=hidden,
      gamma_offset=layer.norm1_weight.offset,
      beta_offset=layer.norm1_bias.offset,
      eps=1.0e-5,
    )
    gpu.fill_bias(
      scratch.qkv,
      weights,
      rows=length,
      cols=3 * hidden,
      bias_offset=layer.in_proj_bias,
    )
    gpu.gemm(
      staging,
      weights,
      scratch.qkv,
      m=length,
      n=3 * hidden,
      k=hidden,
      lda=hidden,
      ldb=hidden,
      ldc=3 * hidden,
      b_offset=layer.in_proj,
      trans_b=true,
      beta=1.0,
    )
    gpu.split_qkv(
      scratch.qkv,
      scratch.q,
      scratch.k,
      scratch.v,
      heads=self.head_heads,
      len=length,
      dim=self.head_dim,
    )
    // The head has no rotary embedding, and with one unpadded sequence there
    // is nothing to mask.
    self.attend_metal(gpu, scratch, length, self.head_heads, self.head_dim, -1)
    gpu.fill_bias(
      scratch.projected,
      weights,
      rows=length,
      cols=hidden,
      bias_offset=layer.out_proj_bias,
    )
    gpu.gemm(
      scratch.packed,
      weights,
      scratch.projected,
      m=length,
      n=hidden,
      k=hidden,
      lda=hidden,
      ldb=hidden,
      ldc=hidden,
      b_offset=layer.out_proj,
      trans_b=true,
      beta=1.0,
    )
    gpu.add(stream, scratch.projected, count=length * hidden)
    gpu.layer_norm(
      stream,
      staging,
      weights,
      weights,
      rows=length,
      cols=hidden,
      gamma_offset=layer.norm2_weight.offset,
      beta_offset=layer.norm2_bias.offset,
      eps=1.0e-5,
    )
    gpu.fill_bias(
      scratch.wide,
      weights,
      rows=length,
      cols=4 * hidden,
      bias_offset=layer.linear1_bias,
    )
    gpu.gemm(
      staging,
      weights,
      scratch.wide,
      m=length,
      n=4 * hidden,
      k=hidden,
      lda=hidden,
      ldb=hidden,
      ldc=4 * hidden,
      b_offset=layer.linear1,
      trans_b=true,
      beta=1.0,
    )
    gpu.relu(scratch.wide, count=length * 4 * hidden)
    gpu.fill_bias(
      scratch.projected,
      weights,
      rows=length,
      cols=hidden,
      bias_offset=layer.linear2_bias,
    )
    gpu.gemm(
      scratch.wide,
      weights,
      scratch.projected,
      m=length,
      n=hidden,
      k=4 * hidden,
      lda=4 * hidden,
      ldb=4 * hidden,
      ldc=hidden,
      b_offset=layer.linear2,
      trans_b=true,
      beta=1.0,
    )
    gpu.add(stream, scratch.projected, count=length * hidden)
  }
  gpu.gather_rows(
    stream,
    scratch.marker_indices,
    scratch.marker_rows,
    rows=options,
    cols=hidden,
  )
  gpu.layer_norm(
    scratch.marker_rows,
    scratch.scorer_normed,
    weights,
    weights,
    rows=options,
    cols=hidden,
    gamma_offset=self.scorer_norm_weight.offset,
    beta_offset=self.scorer_norm_bias.offset,
    eps=1.0e-5,
  )
  gpu.fill_bias(
    scratch.scorer_dense,
    weights,
    rows=options,
    cols=hidden,
    bias_offset=self.scorer_dense_bias,
  )
  gpu.gemm(
    scratch.scorer_normed,
    weights,
    scratch.scorer_dense,
    m=options,
    n=hidden,
    k=hidden,
    lda=hidden,
    ldb=hidden,
    ldc=hidden,
    b_offset=self.scorer_dense,
    trans_b=true,
    beta=1.0,
  )
  gpu.gelu(scratch.scorer_dense, count=options * hidden)
  gpu.commit()

  // Back on the CPU: the scorer's final projection to one logit per option.
  // A [options, hidden] x [1, hidden]^T multiply is far too small to be worth
  // a second command buffer, and the action head needs a CPU round trip for
  // its calibration features regardless.
  let scored = @tensor.zeros(options * hidden)
  scratch.scorer_dense.download(scored, count=options * hidden)
  let scores = @tensor.zeros(options)
  @tensor.linear(
    scored,
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
  let cls = @tensor.zeros(hidden)
  stream.download(cls, count=hidden)
  if trace is Some(_) {
    record(trace, "marker_rows_gpu", scored, options * hidden)
    record_doubles(trace, "scorer_logits", logits[:options])
  }
  (logits, cls)
}

///|
/// Scaled dot-product attention on the GPU, leaving `[len, heads · dim]` in
/// `scratch.packed`.
///
/// Both multiplies are batched over heads in one MPS call each, since every
/// head is the same shape at a uniform stride.
fn Model::attend_metal(
  self : Model,
  gpu : @tensor.MetalContext,
  scratch : MetalScratch,
  length : Int,
  heads : Int,
  dim : Int,
  window : Int,
) -> Unit {
  ignore(self)
  let scale = Float::from_double(1.0 / Double::sqrt(dim.to_double()))
  gpu.gemm(
    scratch.q,
    scratch.k,
    scratch.scores,
    m=length,
    n=length,
    k=dim,
    lda=dim,
    ldb=dim,
    ldc=length,
    a_stride=(length * dim).to_int64(),
    b_stride=(length * dim).to_int64(),
    c_stride=(length * length).to_int64(),
    batch=heads,
    trans_b=true,
    alpha=scale,
  )
  gpu.softmax(scratch.scores, rows=heads * length, len=length, window~)
  gpu.gemm(
    scratch.scores,
    scratch.v,
    scratch.context_buffer,
    m=length,
    n=dim,
    k=length,
    lda=length,
    ldb=dim,
    ldc=dim,
    a_stride=(length * length).to_int64(),
    b_stride=(length * dim).to_int64(),
    c_stride=(length * dim).to_int64(),
    batch=heads,
  )
  gpu.unpack_heads(
    scratch.context_buffer,
    scratch.packed,
    heads~,
    len=length,
    dim~,
  )
}
