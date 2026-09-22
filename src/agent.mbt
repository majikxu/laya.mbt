///|
/// `LayaError` is re-exported so callers need only depend on this package.
pub using @error {type LayaError}

///|
/// Which device runs the forward pass.
pub(all) enum Backend {
  /// BLAS on the CPU. Available everywhere.
  Cpu
  /// Metal on the GPU, silently falling back to the CPU when no Metal device
  /// is available — every non-Apple platform, and Apple machines where the
  /// kernels fail to compile.
  Metal
  /// Metal, failing to load rather than falling back.
  MetalRequired
} derive(Eq, Debug)

///|
/// A loaded Laya checkpoint: weights, tokenizer and calibration.
///
/// Construct one with `Agent::load` and keep it: loading decodes several
/// hundred megabytes of float16 weights into float32, and `predict` is cheap by
/// comparison.
struct Agent {
  model : @model.Model
  vocabulary : Vocabulary
  device : String
}

///|
/// The tokenizer this checkpoint was trained with.
pub fn Agent::vocabulary(self : Agent) -> Vocabulary {
  self.vocabulary
}

///|
/// Load a checkpoint directory.
///
/// The directory is the layout the published checkpoints ship with:
///
/// ```text
/// <directory>/model.safetensors
/// <directory>/rl_agent_config.json
/// <directory>/encoder/config.json
/// <directory>/tokenizer/tokenizer.json
/// <directory>/tokenizer/tokenizer_config.json
/// ```
///
/// `backend` defaults to `Metal`, which falls back to the CPU when no Metal
/// device is available; `Agent::device` reports what was actually attached.
pub fn Agent::load(
  directory : String,
  backend? : Backend = Metal,
) -> Agent raise @error.LayaError {
  let encoder = @model.EncoderConfig::parse(
    read_config("\{directory}/encoder/config.json"),
  )
  let agent = @model.AgentConfig::parse(
    read_config("\{directory}/rl_agent_config.json"),
  )
  let vocabulary = Vocabulary::load("\{directory}/tokenizer")
  let model = @model.Model::load(
    "\{directory}/model.safetensors",
    encoder,
    agent,
  )
  let device = attach_backend(model, backend)
  { model, vocabulary, device, }
}

///|
/// Attach the requested backend, returning a human-readable device name.
fn attach_backend(
  model : @model.Model,
  backend : Backend,
) -> String raise @error.LayaError {
  if backend is Cpu {
    return "cpu"
  }
  match @tensor.MetalContext::create() {
    Ok(context) => {
      model.use_metal(context)
      context.device_name()
    }
    Err(reason) =>
      if backend is MetalRequired {
        raise @error.LayaError(
          "Metal was required but is unavailable: \{reason}",
        )
      } else {
        "cpu"
      }
  }
}

///|
/// The device the forward pass actually runs on: `"cpu"`, or the GPU's name.
///
/// Worth checking after `load`: requesting `Metal` falls back silently.
pub fn Agent::device(self : Agent) -> String {
  self.device
}

///|
/// Whether the forward pass is running on the GPU.
pub fn Agent::on_metal(self : Agent) -> Bool {
  self.model.on_metal()
}

///|
fn read_config(path : String) -> String raise @error.LayaError {
  @fs.read_file_to_string(path) catch {
    _ => raise @error.LayaError("cannot read \{path}")
  }
}

///|
/// Answer one question about one state.
pub fn Agent::answer(
  self : Agent,
  state : String,
  question : Question,
) -> Answer raise @error.LayaError {
  let config = self.model.agent_config()
  let prompt = build_prompt(
    self.vocabulary,
    state,
    question,
    max_len=config.max_len,
    head_max_len=config.head_max_len,
  )
  let outputs = self.model.forward(
    prompt.ids,
    prompt.markers,
    prompt.type_index,
  )
  decode(question, outputs, config)
}

///|
/// Answer several questions about one state.
///
/// Each question is a separate forward pass over its own prompt — the questions
/// share the state text but not the sequence, because every question carries its
/// own option markers.
pub fn Agent::predict(
  self : Agent,
  state : String,
  questions : Array[(String, Question)],
) -> Prediction raise @error.LayaError {
  let config = self.model.agent_config()
  let answers = Array(capacity=questions.length())
  let mut input_tokens = 0
  for entry in questions {
    let (id, question) = entry
    let prompt = build_prompt(
      self.vocabulary,
      state,
      question,
      max_len=config.max_len,
      head_max_len=config.head_max_len,
    )
    input_tokens += prompt.length()
    let outputs = self.model.forward(
      prompt.ids,
      prompt.markers,
      prompt.type_index,
    )
    answers.push((id, decode(question, outputs, config)))
  }
  { answers, input_tokens, }
}

///|
/// Build the prompt for one question without running the model.
///
/// Useful for checking the token budget, or for comparing against another Laya
/// runtime's tokenization.
pub fn Agent::prompt(
  self : Agent,
  state : String,
  question : Question,
) -> Prompt raise @error.LayaError {
  let config = self.model.agent_config()
  build_prompt(
    self.vocabulary,
    state,
    question,
    max_len=config.max_len,
    head_max_len=config.head_max_len,
  )
}

///|
/// The encoder configuration of the loaded checkpoint.
pub fn Agent::encoder_config(self : Agent) -> @model.EncoderConfig {
  self.model.encoder_config()
}

///|
/// The calibration and prompt-budget configuration of the loaded checkpoint.
pub fn Agent::agent_config(self : Agent) -> @model.AgentConfig {
  self.model.agent_config()
}
