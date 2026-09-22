///|
/// A small parameter that both backends need to reach.
///
/// The CPU kernels read `values` directly — going through the arena would mean
/// an FFI call per element — while the Metal kernels address the same numbers
/// in the arena at `offset`. Norm gains, biases and the type embedding total a
/// few hundred kilobytes, so holding both is cheaper than either alternative.
priv struct Param {
  values : FixedArray[Float]
  offset : Int64
}

///|
/// Plans the float32 arena, then fills it.
///
/// Every parameter is reserved up front so the arena can be allocated as one
/// block; `finish` then decodes each tensor from float16 into its slot.
priv struct Loader {
  file : @safetensors.File
  planned : Array[@safetensors.TensorInfo]
  targets : Array[Int64]
  mut total : Int64
}

///|
fn Loader::new(file : @safetensors.File) -> Loader {
  { file, planned: [], targets: [], total: 0L, }
}

///|
/// Reserve arena space for a tensor, returning the element offset it will live
/// at. The shape is checked here so a mismatched checkpoint fails before any
/// allocation happens.
fn Loader::reserve(
  self : Loader,
  name : String,
  shape : Array[Int],
) -> Int64 raise @error.LayaError {
  let info = self.file.info_shaped(name, shape)
  let offset = self.total
  self.planned.push(info)
  self.targets.push(offset)
  self.total += info.elements()
  offset
}

///|
/// Reserve a vector and also read it into a MoonBit array for the CPU path.
fn Loader::param(
  self : Loader,
  name : String,
  length : Int,
) -> Param raise @error.LayaError {
  let offset = self.reserve(name, [length])
  let values = @tensor.zeros(length)
  self.file.read_into_array(self.file.info(name), values)
  { values, offset, }
}

///|
/// Same, for a matrix the CPU path also reads elementwise.
fn Loader::param_matrix(
  self : Loader,
  name : String,
  rows : Int,
  cols : Int,
) -> Param raise @error.LayaError {
  let offset = self.reserve(name, [rows, cols])
  let values = @tensor.zeros(rows * cols)
  self.file.read_into_array(self.file.info(name), values)
  { values, offset, }
}

///|
fn Loader::finish(self : Loader) -> @tensor.Arena raise @error.LayaError {
  let arena = @tensor.Arena::new(self.total)
  for index, info in self.planned {
    self.file.read_into_arena(info, arena, self.targets[index])
  }
  arena
}

///|
/// The parameter blocks of one ModernBERT encoder layer, plus the attention
/// pattern it was configured with.
priv struct EncoderLayer {
  /// `None` on layer 0, where ModernBERT uses an identity in place of the
  /// pre-attention norm because the embedding norm already ran.
  attn_norm : Param?
  wqkv : Int64
  attn_wo : Int64
  mlp_norm : Param
  mlp_wi : Int64
  mlp_wo : Int64
  attention : Attention
  rope_base : Double
}

///|
/// One layer of the decision head — a `torch.nn.TransformerEncoderLayer` in the
/// original, so norms carry biases and the feed-forward uses ReLU.
priv struct HeadLayer {
  norm1_weight : Param
  norm1_bias : Param
  norm2_weight : Param
  norm2_bias : Param
  in_proj : Int64
  in_proj_bias : Int64
  out_proj : Int64
  out_proj_bias : Int64
  linear1 : Int64
  linear1_bias : Int64
  linear2 : Int64
  linear2_bias : Int64
}

///|
/// A loaded Laya checkpoint, ready to run.
struct Model {
  encoder : EncoderConfig
  agent : AgentConfig
  arena : @tensor.Arena
  file : @safetensors.File
  tok_embeddings : @safetensors.TensorInfo
  embed_norm : Param
  layers : Array[EncoderLayer]
  final_norm : Param
  /// `[3, hidden]`: one learned offset per question type, added to every
  /// position before the decision head runs.
  type_emb : Param
  head : Array[HeadLayer]
  head_heads : Int
  head_dim : Int
  scorer_norm_weight : Param
  scorer_norm_bias : Param
  scorer_dense : Int64
  scorer_dense_bias : Int64
  scorer_out : Int64
  scorer_out_bias : Int64
  act_hidden : Int
  act_in : Int64
  act_in_bias : Int64
  act_out : Int64
  act_out_bias : Int64
  act_outputs : Int
  /// Set by `Model::use_metal`; `None` means every pass runs on the CPU.
  mut metal : MetalState?
}

///|
/// The encoder configuration this model was built from.
pub fn Model::encoder_config(self : Model) -> EncoderConfig {
  self.encoder
}

///|
/// The calibration and prompt-budget configuration.
pub fn Model::agent_config(self : Model) -> AgentConfig {
  self.agent
}

///|
/// Load a checkpoint from an already-parsed pair of configs plus its weights.
pub fn Model::load(
  weights_path : String,
  encoder : EncoderConfig,
  agent : AgentConfig,
) -> Model raise @error.LayaError {
  let file = @safetensors.File::open(weights_path)
  let hidden = encoder.hidden_size
  let inner = encoder.intermediate_size
  let loader = Loader::new(file)

  // Encoder body. Wqkv packs query, key and value into one [3H, H] matrix and
  // Wi packs the GeGLU value and gate halves into one [2I, H] matrix.
  let layers = Array(capacity=encoder.num_hidden_layers)
  for index in 0..<encoder.num_hidden_layers {
    let prefix = "encoder.layers.\{index}"
    let attn_norm = if index == 0 {
      None
    } else {
      Some(loader.param("\{prefix}.attn_norm.weight", hidden))
    }
    layers.push({
      attn_norm,
      wqkv: loader.reserve("\{prefix}.attn.Wqkv.weight", [3 * hidden, hidden]),
      attn_wo: loader.reserve("\{prefix}.attn.Wo.weight", [hidden, hidden]),
      mlp_norm: loader.param("\{prefix}.mlp_norm.weight", hidden),
      mlp_wi: loader.reserve("\{prefix}.mlp.Wi.weight", [2 * inner, hidden]),
      mlp_wo: loader.reserve("\{prefix}.mlp.Wo.weight", [hidden, inner]),
      attention: encoder.layer_types[index],
      rope_base: encoder.layer_rope_base[index],
    })
  }

  // Decision head. Its attention head count follows the original's
  // `max(1, dims // 64)` rule rather than the encoder's.
  let head_heads = @cmp.maximum(1, hidden / 64)
  @error.expect(
    hidden % head_heads == 0,
    "hidden size \{hidden} is not divisible by the decision head's \{head_heads} heads",
  )
  let head = Array(capacity=agent.head_layers)
  for index in 0..<agent.head_layers {
    let prefix = "head.layers.\{index}"
    head.push({
      norm1_weight: loader.param("\{prefix}.norm1.weight", hidden),
      norm1_bias: loader.param("\{prefix}.norm1.bias", hidden),
      norm2_weight: loader.param("\{prefix}.norm2.weight", hidden),
      norm2_bias: loader.param("\{prefix}.norm2.bias", hidden),
      in_proj: loader.reserve("\{prefix}.self_attn.in_proj.weight", [
        3 * hidden,
        hidden,
      ]),
      in_proj_bias: loader.reserve("\{prefix}.self_attn.in_proj.bias", [
        3 * hidden,
      ]),
      out_proj: loader.reserve("\{prefix}.self_attn.out_proj.weight", [
        hidden, hidden,
      ]),
      out_proj_bias: loader.reserve("\{prefix}.self_attn.out_proj.bias", [
        hidden,
      ]),
      linear1: loader.reserve("\{prefix}.linear1.weight", [4 * hidden, hidden]),
      linear1_bias: loader.reserve("\{prefix}.linear1.bias", [4 * hidden]),
      linear2: loader.reserve("\{prefix}.linear2.weight", [hidden, 4 * hidden]),
      linear2_bias: loader.reserve("\{prefix}.linear2.bias", [hidden]),
    })
  }

  // Scorer: LayerNorm, Linear, GELU, Linear-to-one. The `nn.Sequential` index
  // gap at 2 is the activation, which has no parameters.
  let scorer_norm_weight = loader.param("scorer.layers.0.weight", hidden)
  let scorer_norm_bias = loader.param("scorer.layers.0.bias", hidden)
  let scorer_dense = loader.reserve("scorer.layers.1.weight", [hidden, hidden])
  let scorer_dense_bias = loader.reserve("scorer.layers.1.bias", [hidden])
  let scorer_out = loader.reserve("scorer.layers.3.weight", [1, hidden])
  let scorer_out_bias = loader.reserve("scorer.layers.3.bias", [1])

  // Action head: the four calibration features are concatenated onto the
  // pooled CLS row, and its output width follows the checkpoint rather than the
  // config, since it is `len(act_costs) + 1`.
  let act_in_info = file.info("act_head.layers.0.weight")
  @error.expect(
    act_in_info.shape.length() == 2 && act_in_info.shape[1] == hidden + 4,
    "act_head input expects \{hidden + 4} features, checkpoint has \{Repr(act_in_info.shape)}",
  )
  let act_hidden = act_in_info.shape[0]
  let act_out_info = file.info("act_head.layers.2.weight")
  @error.expect(
    act_out_info.shape.length() == 2 && act_out_info.shape[1] == act_hidden,
    "act_head output layer does not match its \{act_hidden}-wide hidden layer",
  )
  let act_outputs = act_out_info.shape[0]
  @error.expect(
    act_outputs >= 1,
    "act_head must emit at least one action logit, checkpoint emits \{act_outputs}",
  )
  let act_in = loader.reserve("act_head.layers.0.weight", [
    act_hidden,
    hidden + 4,
  ])
  let act_in_bias = loader.reserve("act_head.layers.0.bias", [act_hidden])
  let act_out = loader.reserve("act_head.layers.2.weight", [
    act_outputs, act_hidden,
  ])
  let act_out_bias = loader.reserve("act_head.layers.2.bias", [act_outputs])
  let embed_norm = loader.param("encoder.embeddings.norm.weight", hidden)
  let final_norm = loader.param("encoder.final_norm.weight", hidden)
  let type_emb = loader.param_matrix("type_emb.weight", 3, hidden)
  let tok_embeddings = file.info_shaped(
    "encoder.embeddings.tok_embeddings.weight",
    [encoder.vocab_size, hidden],
  )
  {
    encoder,
    agent,
    arena: loader.finish(),
    file,
    tok_embeddings,
    embed_norm,
    layers,
    final_norm,
    type_emb,
    head,
    head_heads,
    head_dim: hidden / head_heads,
    scorer_norm_weight,
    scorer_norm_bias,
    scorer_dense,
    scorer_dense_bias,
    scorer_out,
    scorer_out_bias,
    act_hidden,
    act_in,
    act_in_bias,
    act_out,
    act_out_bias,
    act_outputs,
    metal: None,
  }
}

///|
/// Number of action logits the checkpoint's action head emits.
pub fn Model::action_outputs(self : Model) -> Int {
  self.act_outputs
}
