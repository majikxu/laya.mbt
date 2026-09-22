///|
/// Which attention pattern a ModernBERT layer uses.
pub(all) enum Attention {
  /// Every position attends to every other position.
  Global
  /// Positions attend within `local_attention / 2` of themselves.
  Sliding
} derive(Debug, Eq)

///|
/// The encoder half of a Laya checkpoint, read from `encoder/config.json`.
///
/// Both published checkpoints are ModernBERT-shaped — `laya` wraps
/// ModernBERT-large and `laya-multilingual` wraps mmBERT-base — so one config
/// type covers both; they differ only in width, depth and RoPE bases.
pub struct EncoderConfig {
  vocab_size : Int
  hidden_size : Int
  intermediate_size : Int
  num_hidden_layers : Int
  num_attention_heads : Int
  head_dim : Int
  norm_eps : Float
  local_attention : Int
  max_position_embeddings : Int
  layer_types : Array[Attention]
  layer_rope_base : Array[Double]
} derive(Debug)

///|
/// Calibration and prompt budgets, read from `rl_agent_config.json`.
pub struct AgentConfig {
  head_layers : Int
  max_len : Int
  head_max_len : Int
  /// Per question type: index 0 choice, 1 score, 2 noul.
  temperature : Array[Double]
  /// Finer-grained overrides keyed `"<type>:<option-count-bucket>"`.
  temperature_by_options : Map[String, Double]
} derive(Debug)

///|
/// Parse `encoder/config.json`.
pub fn EncoderConfig::parse(
  source : String,
) -> EncoderConfig raise @error.LayaError {
  let json = parse_json(source, "encoder/config.json")
  let model_type = optional_string(json, "model_type").unwrap_or("modernbert")
  @error.expect(
    model_type == "modernbert",
    "unsupported encoder \{model_type}; laya.mbt implements the ModernBERT encoder",
  )
  let activation = optional_string(json, "hidden_activation").unwrap_or("gelu")
  @error.expect(
    activation == "gelu",
    "unsupported encoder activation \{activation}; only gelu is implemented",
  )
  let hidden_size = required_int(json, "hidden_size")
  let num_attention_heads = required_int(json, "num_attention_heads")
  @error.expect(
    num_attention_heads > 0 && hidden_size % num_attention_heads == 0,
    "hidden_size \{hidden_size} is not divisible by num_attention_heads \{num_attention_heads}",
  )
  let head_dim = hidden_size / num_attention_heads
  @error.expect(
    head_dim % 2 == 0,
    "head dimension \{head_dim} must be even for rotary embeddings",
  )
  let num_hidden_layers = required_int(json, "num_hidden_layers")
  let global_every = optional_int(json, "global_attn_every_n_layers").unwrap_or(
    3,
  )
  let layer_types = parse_layer_types(json, num_hidden_layers, global_every)
  let global_theta = optional_number(json, "global_rope_theta").unwrap_or(
    160000.0,
  )
  let local_theta = optional_number(json, "local_rope_theta").unwrap_or(10000.0)
  let rope_global = rope_theta(json, "full_attention", global_theta)
  let rope_local = rope_theta(json, "sliding_attention", local_theta)
  {
    vocab_size: required_int(json, "vocab_size"),
    hidden_size,
    intermediate_size: required_int(json, "intermediate_size"),
    num_hidden_layers,
    num_attention_heads,
    head_dim,
    norm_eps: Float::from_double(
      optional_number(json, "norm_eps").unwrap_or(
        optional_number(json, "layer_norm_eps").unwrap_or(1.0e-5),
      ),
    ),
    local_attention: optional_int(json, "local_attention").unwrap_or(128),
    max_position_embeddings: optional_int(json, "max_position_embeddings").unwrap_or(
      8192,
    ),
    layer_types,
    layer_rope_base: layer_types.map(kind => {
      match kind {
        Global => rope_global
        Sliding => rope_local
      }
    }),
  }
}

///|
/// An explicit `layer_types` list wins; otherwise ModernBERT's rule is that
/// every `global_attn_every_n_layers`-th layer attends globally.
fn parse_layer_types(
  json : Json,
  layers : Int,
  global_every : Int,
) -> Array[Attention] raise @error.LayaError {
  match json {
    Object({ "layer_types": Array(entries), .. }) => {
      @error.expect(
        entries.length() == layers,
        "layer_types lists \{entries.length()} entries but num_hidden_layers is \{layers}",
      )
      let out = Array(capacity=layers)
      for entry in entries {
        match entry {
          String("full_attention") => out.push(Global)
          String("sliding_attention") => out.push(Sliding)
          other =>
            raise @error.LayaError(
              "unknown layer type \{other.stringify()}; expected full_attention or sliding_attention",
            )
        }
      }
      out
    }
    _ => {
      @error.expect(
        global_every > 0,
        "global_attn_every_n_layers must be positive, got \{global_every}",
      )
      Array::makei(layers, i => {
        if i % global_every == 0 {
          Global
        } else {
          Sliding
        }
      })
    }
  }
}

///|
/// ModernBERT allows per-pattern RoPE settings; only unscaled RoPE is supported.
fn rope_theta(
  json : Json,
  kind : String,
  fallback : Double,
) -> Double raise @error.LayaError {
  guard json is Object({ "rope_parameters": Object(patterns), .. }) else {
    return fallback
  }
  guard patterns.get(kind) is Some(Object(params)) else { return fallback }
  let rope_type = match params.get("rope_type") {
    Some(String(value)) => value
    _ => "default"
  }
  @error.expect(
    rope_type == "default",
    "rope_type \{rope_type} for \{kind} is not supported; only unscaled RoPE is implemented",
  )
  match params.get("rope_theta") {
    Some(Number(value, ..)) => value
    _ => fallback
  }
}

///|
/// Parse `rl_agent_config.json`.
pub fn AgentConfig::parse(
  source : String,
) -> AgentConfig raise @error.LayaError {
  let json = parse_json(source, "rl_agent_config.json")
  let head_layers = required_int(json, "head_layers")
  @error.expect(
    head_layers > 0,
    "head_layers must be positive, got \{head_layers}",
  )
  let max_len = optional_int(json, "max_len").unwrap_or(512)
  let head_max_len = optional_int(json, "head_max_len").unwrap_or(192)
  @error.expect(
    4 < head_max_len && head_max_len < max_len,
    "expected 4 < head_max_len < max_len, got head_max_len=\{head_max_len} max_len=\{max_len}",
  )
  let temperature = match json {
    Object({ "temperature": Array(values), .. }) => {
      @error.expect(
        values.length() == 3,
        "temperature must hold one value per question type, got \{values.length()}",
      )
      let out = Array(capacity=3)
      for value in values {
        out.push(finite_temperature(value))
      }
      out
    }
    _ => [1.0, 1.0, 1.0]
  }
  let temperature_by_options = Map([])
  if json is Object({ "temperature_by_options": Object(buckets), .. }) {
    for bucket, value in buckets {
      temperature_by_options[bucket] = finite_temperature(value)
    }
  }
  { head_layers, max_len, head_max_len, temperature, temperature_by_options, }
}

///|
fn finite_temperature(value : Json) -> Double raise @error.LayaError {
  guard value is Number(value, ..) else {
    raise @error.LayaError("calibration temperatures must be numbers")
  }
  @error.expect(
    value > 0.0 && !value.is_nan() && !value.is_inf(),
    "calibration temperatures must be finite and positive, got \{value}",
  )
  value
}

///|
fn parse_json(source : String, what : String) -> Json raise @error.LayaError {
  let json = @json.parse(source) catch {
    _ => raise @error.LayaError("\{what} is not valid JSON")
  }
  guard json is Object(_) else {
    raise @error.LayaError("\{what} is not a JSON object")
  }
  json
}

///|
fn required_int(json : Json, key : String) -> Int raise @error.LayaError {
  match optional_int(json, key) {
    Some(value) => value
    None => raise @error.LayaError("configuration is missing \{key}")
  }
}

///|
fn optional_int(json : Json, key : String) -> Int? {
  match optional_number(json, key) {
    Some(value) => Some(value.to_int())
    None => None
  }
}

///|
fn optional_number(json : Json, key : String) -> Double? {
  guard json is Object(fields) else { return None }
  match fields.get(key) {
    Some(Number(value, ..)) => Some(value)
    _ => None
  }
}

///|
fn optional_string(json : Json, key : String) -> String? {
  guard json is Object(fields) else { return None }
  match fields.get(key) {
    Some(String(value)) => Some(value)
    _ => None
  }
}
