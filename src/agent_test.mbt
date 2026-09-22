///|
/// End-to-end agreement with the Python runtime: same prompt, same tokens, same
/// public JSON.
///
/// The per-stage numerical checks live in `src/model/parity_test.mbt`; what is
/// checked here is everything around the model — prompt construction, the token
/// budget, calibration, confidence and the rendered output shape.
///
/// Both fixtures and weights are outside version control, so a missing
/// checkpoint reports a skip instead of failing.
struct Checkpoint {
  name : String
  directory : String
  fixture : String
}

///|
let checkpoints : Array[Checkpoint] = [
  {
    name: "laya-multilingual",
    directory: "models/laya-multilingual-mlx",
    fixture: "fixtures/laya-multilingual.json",
  },
  {
    name: "laya",
    directory: "models/laya-mlx",
    fixture: "fixtures/laya-en.json",
  },
]

///|
fn read_json(path : String) -> Json? {
  if !@fs.path_exists(path) {
    return None
  }
  let source = @fs.read_file_to_string(path) catch { _ => return None }
  Some(@json.parse(source)) catch {
    _ => None
  }
}

///|
/// Rebuild a `Question` from the fixture's Laya-format question definition, so
/// the test exercises the same definitions the reference ran.
fn question_from_json(
  id : String,
  definition : Json,
) -> @laya.Question raise @error.LayaError {
  guard definition
    is Object(
      { "type": String(kind), "instructions": String(instructions), .. }
    ) else {
    raise @error.LayaError("fixture question is missing type or instructions")
  }
  let criteria = match definition {
    Object(fields) => fields.get("criteria")
    _ => None
  }
  match kind {
    "choice" =>
      match criteria {
        // A list of labels with no descriptions.
        Some(Array(labels)) => {
          let options = Array(capacity=labels.length())
          for label in labels {
            guard label is String(label) else {
              raise @error.LayaError("choice labels must be strings")
            }
            options.push(@laya.ChoiceOption::bare(label))
          }
          Choice(id~, instructions~, options~)
        }
        // A mapping of label to description; insertion order is the marker
        // order, which is why the fixture's JSON object order matters.
        Some(Object(described)) => {
          let options = Array(capacity=described.length())
          for label, description in described {
            match description {
              String(text) =>
                options.push(@laya.ChoiceOption::described(label, text))
              Null => options.push(@laya.ChoiceOption::bare(label))
              other =>
                raise @error.LayaError(
                  "choice description for \{label} is \{other.stringify()}, which this test does not render",
                )
            }
          }
          Choice(id~, instructions~, options~)
        }
        _ => raise @error.LayaError("choice question has no usable criteria")
      }
    "score" =>
      match criteria {
        Some(Array(entries)) => {
          let levels = Array(capacity=entries.length())
          for entry in entries {
            guard entry is String(level) else {
              raise @error.LayaError("score levels must be strings")
            }
            levels.push(level)
          }
          Score(id~, instructions~, levels~)
        }
        _ => raise @error.LayaError("score question has no usable criteria")
      }
    "noul" => {
      let outcome = key => {
        match criteria {
          Some(Object(fields)) =>
            match fields.get(key) {
              Some(String(text)) => Some(text)
              _ => None
            }
          _ => None
        }
      }
      Noul(
        id~,
        instructions~,
        when_false=outcome("false"),
        when_true=outcome("true"),
      )
    }
    other => raise @error.LayaError("unknown question type \{other}")
  }
}

///|
/// Questions from the fixture's questions object, each carrying its own id.
fn questions_from_json(
  json : Json?,
) -> Array[@laya.Question] raise @error.LayaError {
  guard json is Some(Object(fields)) else { return [] }
  let out = Array(capacity=fields.length())
  for id, definition in fields {
    out.push(question_from_json(id, definition))
  }
  out
}

///|
fn expect_string(json : Json?) -> String? {
  match json {
    Some(String(value)) => Some(value)
    _ => None
  }
}

///|
test "prompts tokenize exactly as the Python runtime does" {
  let report = []
  let checked = []
  for checkpoint in checkpoints {
    guard read_json(checkpoint.fixture) is Some(Object(fixture)) else {
      continue
    }
    if !@fs.path_exists("\{checkpoint.directory}/model.safetensors") {
      continue
    }
    let agent = @laya.Agent::load(checkpoint.directory)
    guard fixture.get("cases") is Some(Object(cases)) else {
      fail("\{checkpoint.fixture} has no cases")
    }
    for name, case in cases {
      guard case is Object(case) else { continue }
      guard expect_string(case.get("state_text")) is Some(state) else {
        fail("case \{name} has no state_text; regenerate the fixtures")
      }
      guard case.get("items") is Some(Array(items)) else { continue }
      let questions = questions_from_json(case.get("questions"))
      if questions.length() != items.length() {
        report.push("\{checkpoint.name}/\{name}: question count mismatch")
        continue
      }
      for index, question in questions {
        guard items[index] is Object(item) else { continue }
        let prompt = agent.prompt(@laya.State(state), question)
        let want_ids = ints(item.get("ids"))
        let want_markers = ints(item.get("markers"))
        let id = question.id()
        if prompt.ids() != want_ids {
          report.push(
            "\{checkpoint.name}/\{name}/\{id}: ids differ (\{prompt.ids().length()} vs \{want_ids.length()} tokens)",
          )
        }
        if prompt.markers() != want_markers {
          report.push(
            "\{checkpoint.name}/\{name}/\{id}: markers \{Repr(prompt.markers())} vs \{Repr(want_markers)}",
          )
        }
        checked.push("\{checkpoint.name}/\{name}/\{id}")
      }
    }
  }
  if checked.is_empty() {
    println("skipped: no checkpoint and fixture pair is available")
    return
  }
  debug_inspect(report, content="[]")
  inspect(checked.length(), content="16")
}

///|
test "predictions render identically to the Python runtime" {
  let report = []
  let checked = []
  for checkpoint in checkpoints {
    guard read_json(checkpoint.fixture) is Some(Object(fixture)) else {
      continue
    }
    if !@fs.path_exists("\{checkpoint.directory}/model.safetensors") {
      continue
    }
    let agent = @laya.Agent::load(checkpoint.directory)
    guard fixture.get("cases") is Some(Object(cases)) else {
      fail("\{checkpoint.fixture} has no cases")
    }
    for name, case in cases {
      guard case is Object(case) else { continue }
      guard expect_string(case.get("state_text")) is Some(state) else {
        continue
      }
      guard case.get("result") is Some(expected) else { continue }
      let questions = questions_from_json(case.get("questions"))
      let prediction = agent.predict(@laya.State(state), questions)
      let produced = prediction.to_json()
      if produced != expected {
        report.push(
          "\{checkpoint.name}/\{name}:\n  got  \{produced.stringify()}\n  want \{expected.stringify()}",
        )
      }
      checked.push("\{checkpoint.name}/\{name}")
    }
  }
  if checked.is_empty() {
    println("skipped: no checkpoint and fixture pair is available")
    return
  }
  debug_inspect(report, content="[]")
  inspect(checked.length(), content="10")
}

///|
fn ints(json : Json?) -> Array[Int] {
  guard json is Some(Array(entries)) else { return [] }
  let out = Array(capacity=entries.length())
  for entry in entries {
    guard entry is Number(value, ..) else { return [] }
    out.push(value.to_int())
  }
  out
}

///|
/// The same end-to-end comparison, on the GPU.
///
/// The two backends share prompt construction and decoding, so what this adds
/// over the CPU test is the Metal kernels and MPS: if a kernel is wrong, the
/// rendered JSON stops matching the Python runtime.
test "Metal predictions render identically to the Python runtime" {
  let report = []
  let checked = []
  for checkpoint in checkpoints {
    guard read_json(checkpoint.fixture) is Some(Object(fixture)) else {
      continue
    }
    if !@fs.path_exists("\{checkpoint.directory}/model.safetensors") {
      continue
    }
    let agent = @laya.Agent::load(checkpoint.directory, backend=Metal)
    if !agent.on_metal() {
      println("skipped: no Metal device (\{agent.device()})")
      return
    }
    guard fixture.get("cases") is Some(Object(cases)) else {
      fail("\{checkpoint.fixture} has no cases")
    }
    for name, case in cases {
      guard case is Object(case) else { continue }
      guard expect_string(case.get("state_text")) is Some(state) else {
        continue
      }
      guard case.get("result") is Some(expected) else { continue }
      let questions = questions_from_json(case.get("questions"))
      let produced = agent.predict(@laya.State(state), questions).to_json()
      if produced != expected {
        report.push(
          "\{checkpoint.name}/\{name} on \{agent.device()}:\n  got  \{produced.stringify()}\n  want \{expected.stringify()}",
        )
      }
      checked.push("\{checkpoint.name}/\{name}")
    }
  }
  if checked.is_empty() {
    println("skipped: no checkpoint and fixture pair is available")
    return
  }
  debug_inspect(report, content="[]")
  inspect(checked.length(), content="10")
}

///|
/// Compare the two backends at full precision.
///
/// The rendered JSON rounds to four decimals, which would hide a kernel that
/// is merely close. This checks the raw probabilities the two paths produce.
test "the Metal and CPU backends agree to float32 precision" {
  let directory = "models/laya-multilingual-mlx"
  if !@fs.path_exists("\{directory}/model.safetensors") {
    println("skipped: \{directory} has no weights")
    return
  }
  let gpu = @laya.Agent::load(directory, backend=Metal)
  if !gpu.on_metal() {
    println("skipped: no Metal device (\{gpu.device()})")
    return
  }
  let cpu = @laya.Agent::load(directory, backend=Cpu)
  let state = @laya.State(
    "I was billed twice this month. Please refund the duplicate charge.",
  )
  let questions = [
    @laya.Question::choice("department", "Who should handle this?", [
      "billing", "technical", "sales",
    ]),
    Score(id="urgency", instructions="How urgent is this?", levels=[
      "not urgent", "soon", "urgent", "critical",
    ]),
    @laya.Question::noul("refund", "The customer is asking for a refund."),
  ]
  let on_gpu = gpu.predict(state, questions)
  let on_cpu = cpu.predict(state, questions)
  let report = []
  for index, gpu_answer in on_gpu.answers {
    let cpu_answer = on_cpu.answers[index]
    let id = gpu_answer.id
    let confidence = (gpu_answer.confidence - cpu_answer.confidence).abs()
    if confidence > 1.0e-5 {
      report.push("\{id}: confidence differs by \{confidence}")
    }
    let action = (gpu_answer.act_probability - cpu_answer.act_probability).abs()
    if action > 1.0e-5 {
      report.push("\{id}: act_probability differs by \{action}")
    }
    match (gpu_answer.decision, cpu_answer.decision) {
      (
        Choice(label~, probabilities~),
        Choice(label=want_label, probabilities=want),
      ) => {
        if label != want_label {
          report.push(
            "\{id}: chose \{label} on the GPU, \{want_label} on the CPU",
          )
        }
        for slot, pair in probabilities {
          let diff = (pair.1 - want[slot].1).abs()
          if diff > 1.0e-5 {
            report.push("\{id}/\{pair.0}: probability differs by \{diff}")
          }
        }
      }
      (Score(value~, ..), Score(value=want, ..)) =>
        if (value - want).abs() > 1.0e-5 {
          report.push("\{id}: score differs by \{(value - want).abs()}")
        }
      (Noul(probability~), Noul(probability=want)) =>
        if (probability - want).abs() > 1.0e-5 {
          report.push("\{id}: noul differs by \{(probability - want).abs()}")
        }
      _ => report.push("\{id}: decision types differ between backends")
    }
  }
  debug_inspect(report, content="[]")
  inspect(on_gpu.input_tokens == on_cpu.input_tokens, content="true")
}
