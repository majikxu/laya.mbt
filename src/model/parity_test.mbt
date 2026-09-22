///|
/// Numerical parity against `laya-mlx` running in float32 on the CPU.
///
/// The fixtures under `fixtures/` are produced by `ref/dump_fixtures.py`; the
/// weights under `models/` are downloaded from the Hugging Face checkpoints.
/// Neither is in version control, so these tests report that they were skipped
/// rather than failing when the data is absent.
const MODEL_DIR : String = "models/laya-multilingual-mlx"

///|
const FIXTURE : String = "fixtures/laya-multilingual.json"

///|
/// Intermediate activations are compared *relative to their own scale*. The
/// residual stream grows large inside a ModernBERT encoder — the last layer of
/// this checkpoint peaks near 1.2e4 — so a fixed absolute bound would be
/// measuring float32 resolution rather than agreement. At 1e-5 relative this
/// is roughly a handful of ULPs per stage.
///
/// For reference, the upstream `validation.json` reports MLX-vs-PyTorch logit
/// agreement of ~9e-6 with probability agreement of ~1e-6.
const STAGE_TOLERANCE : Double = 1.0e-5

///|
const LOGIT_TOLERANCE : Double = 1.0e-4

///|
fn load_fixture() -> Json? {
  if !@fs.path_exists(FIXTURE) {
    return None
  }
  let source = @fs.read_file_to_string(FIXTURE) catch { _ => return None }
  Some(@json.parse(source)) catch {
    _ => None
  }
}

///|
fn load_model() -> @model.Model? raise {
  if !@fs.path_exists("\{MODEL_DIR}/model.safetensors") {
    return None
  }
  let encoder = @model.EncoderConfig::parse(
    @fs.read_file_to_string("\{MODEL_DIR}/encoder/config.json"),
  )
  let agent = @model.AgentConfig::parse(
    @fs.read_file_to_string("\{MODEL_DIR}/rl_agent_config.json"),
  )
  Some(@model.Model::load("\{MODEL_DIR}/model.safetensors", encoder, agent))
}

///|
fn max_abs_diff(
  actual : ArrayView[Double],
  expected : ArrayView[Double],
) -> Double {
  if actual.length() != expected.length() {
    return @double.infinity
  }
  let mut worst = 0.0
  for index, value in actual {
    let diff = (value - expected[index]).abs()
    if diff > worst {
      worst = diff
    }
  }
  worst
}

///|
/// Largest deviation, divided by the scale of the values being compared so the
/// bound means the same thing for a stage peaking at 40 and one peaking at
/// 12000.
fn max_relative_diff(
  actual : ArrayView[Double],
  expected : ArrayView[Double],
) -> Double {
  let absolute = max_abs_diff(actual, expected)
  if absolute == @double.infinity {
    return absolute
  }
  let mut scale = 1.0
  for value in expected {
    if value.abs() > scale {
      scale = value.abs()
    }
  }
  absolute / scale
}

///|
fn numbers(json : Json?) -> Array[Double] {
  guard json is Some(Array(entries)) else { return [] }
  let out = Array(capacity=entries.length())
  for entry in entries {
    guard entry is Number(value, ..) else { return [] }
    out.push(value)
  }
  out
}

///|
fn ints(json : Json?) -> Array[Int] {
  numbers(json).map(value => value.to_int())
}

///|
test "encoder config matches the published checkpoint shapes" {
  guard load_fixture() is Some(Object(fixture)) else {
    println("skipped: \{FIXTURE} is missing, run ref/dump_fixtures.py")
    return
  }
  guard load_model() is Some(model) else {
    println("skipped: \{MODEL_DIR} has no weights")
    return
  }
  let config = model.encoder_config()
  guard fixture.get("encoder") is Some(Object(expected)) else {
    fail("fixture has no encoder section")
  }
  let expect_int = key => {
    match expected.get(key) {
      Some(Number(value, ..)) => value.to_int()
      _ => -1
    }
  }
  inspect(config.hidden_size == expect_int("hidden_size"), content="true")
  inspect(
    config.num_hidden_layers == expect_int("num_hidden_layers"),
    content="true",
  )
  inspect(
    config.num_attention_heads == expect_int("num_attention_heads"),
    content="true",
  )
  inspect(
    config.intermediate_size == expect_int("intermediate_size"),
    content="true",
  )
  inspect(config.vocab_size == expect_int("vocab_size"), content="true")
  inspect(model.action_outputs(), content="2")
}

///|
test "every forward stage tracks the reference within tolerance" {
  guard load_fixture() is Some(Object(fixture)) else {
    println("skipped: \{FIXTURE} is missing, run ref/dump_fixtures.py")
    return
  }
  guard load_model() is Some(model) else {
    println("skipped: \{MODEL_DIR} has no weights")
    return
  }
  guard fixture.get("cases") is Some(Object(cases)) else {
    fail("fixture has no cases")
  }
  guard cases.get("tiny") is Some(Object(tiny)) else {
    fail("fixture has no `tiny` case; regenerate it with --full")
  }
  guard tiny.get("stages") is Some(Object(stages)) else {
    fail("the `tiny` case has no recorded stages; regenerate it with --full")
  }
  guard tiny.get("items") is Some(Array([Object(item)])) else {
    fail("the `tiny` case should hold exactly one question")
  }
  let ids = ints(item.get("ids"))
  let markers = ints(item.get("markers"))
  guard item.get("qtype") is Some(Number(qtype, ..)) else {
    fail("item has no qtype")
  }
  let trace = Map([])
  let outputs = model.forward(ids, markers, qtype.to_int(), trace~)

  // Walk the stages in execution order so the first divergence is the one
  // reported, rather than whichever key the map happens to yield first.
  let ordered = [
    "embeddings", "encoder_layer_0", "encoder_layer_1", "encoder_layer_21", "encoder_out",
    "with_type_emb", "head_layer_0", "head_layer_1", "marker_rows", "scorer_logits",
  ]
  let report = []
  for key in ordered {
    guard stages.get(key) is Some(expected_json) else { continue }
    let expected = numbers(Some(expected_json))
    guard trace.get(key) is Some(actual) else {
      report.push("\{key}: not recorded")
      continue
    }
    let diff = max_relative_diff(actual[:], expected[:])
    if diff > STAGE_TOLERANCE {
      report.push(
        "\{key}: relative diff \{diff} over \{actual.length()} values",
      )
    }
  }
  debug_inspect(report, content="[]")
  guard tiny.get("logits") is Some(Array([first_row, ..])) else {
    fail("the `tiny` case has no logits row")
  }
  let expected_logits = numbers(Some(first_row))
  inspect(
    max_abs_diff(
      outputs.logits[:outputs.options],
      expected_logits[:outputs.options],
    ) <
    LOGIT_TOLERANCE,
    content="true",
  )
}

///|
test "logits and action logits match the reference for every fixture case" {
  guard load_fixture() is Some(Object(fixture)) else {
    println("skipped: \{FIXTURE} is missing, run ref/dump_fixtures.py")
    return
  }
  guard load_model() is Some(model) else {
    println("skipped: \{MODEL_DIR} has no weights")
    return
  }
  guard fixture.get("cases") is Some(Object(cases)) else {
    fail("fixture has no cases")
  }
  let report = []
  for name, case in cases {
    guard case is Object(case) else { continue }
    guard case.get("items") is Some(Array(items)) else { continue }
    guard case.get("logits") is Some(Array(expected_logits)) else { continue }
    guard case.get("action_logits") is Some(Array(expected_action)) else {
      continue
    }
    for index, entry in items {
      guard entry is Object(item) else { continue }
      guard item.get("qtype") is Some(Number(qtype, ..)) else { continue }
      let ids = ints(item.get("ids"))
      let markers = ints(item.get("markers"))
      let outputs = model.forward(ids, markers, qtype.to_int())
      // The reference collates a batch, so its rows are padded out to the
      // widest question in the batch; only the real option slots are
      // comparable. laya.mbt runs one question at a time and pads to two.
      let want_logits = numbers(Some(expected_logits[index]))
      let options = outputs.options
      if want_logits.length() < options {
        report.push("\{name}[\{index}]: reference has fewer option slots")
        continue
      }
      let logit_diff = max_abs_diff(
        outputs.logits[:options],
        want_logits[:options],
      )
      if logit_diff > LOGIT_TOLERANCE {
        report.push("\{name}[\{index}] logits: max abs diff \{logit_diff}")
      }

      // The reference's action logits are the noisiest quantity in the whole
      // pipeline: `validation.json` records MLX diverging from PyTorch by up to
      // 1.3e-2 there. What matters downstream is only the softmax over them, so
      // that is what is compared.
      let want_action = @tensor.softmax(
        numbers(Some(expected_action[index]))[:],
      )
      let got_action = @tensor.softmax(outputs.action[:])
      let action_diff = max_abs_diff(got_action[:], want_action[:])
      if action_diff > LOGIT_TOLERANCE {
        report.push("\{name}[\{index}] action: max abs diff \{action_diff}")
      }
    }
  }
  debug_inspect(report, content="[]")
}
