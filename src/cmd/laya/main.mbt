///|
/// `laya` — run a Laya checkpoint over a state and a set of typed questions.
///
/// Questions are given in the same JSON shape the Python runtime accepts, so an
/// existing question file works unchanged:
///
/// ```json
/// {
///   "department": {
///     "type": "choice",
///     "instructions": "Who should handle this?",
///     "criteria": ["billing", "technical", "sales"]
///   },
///   "refund": { "type": "noul", "instructions": "The customer wants a refund." }
/// }
/// ```
fn main {
  let command = @argparse.Command(
    "laya",
    about="Run a Laya typed-decision checkpoint locally",
    version="0.1.0",
    arg_required_else_help=true,
    options=[
      @argparse.OptionArg(
        "model",
        short='m',
        long="model",
        about="Checkpoint directory (model.safetensors, encoder/, tokenizer/)",
        required=true,
      ),
      @argparse.OptionArg(
        "questions",
        short='q',
        long="questions",
        about="Questions JSON file, or - for stdin",
        required=true,
      ),
      @argparse.OptionArg(
        "state",
        short='s',
        long="state",
        about="State text to evaluate",
      ),
      @argparse.OptionArg(
        "state-file",
        long="state-file",
        about="Read the state from a file instead of --state",
      ),
      @argparse.OptionArg(
        "backend",
        short='b',
        long="backend",
        about="metal (default, falls back to cpu), metal-required, or cpu",
        default_values=["metal"],
      ),
      @argparse.OptionArg(
        "repeat",
        short='r',
        long="repeat",
        about="Run the prediction this many times and report timings on stderr",
        default_values=["1"],
      ),
    ],
  )
  let matches = command.parse() catch {
    err => {
      println("laya: \{err}")
      return
    }
  }
  run(matches) catch {
    err => {
      println("laya: \{err}")
      @sys.exit(1)
    }
  }
}

///|
fn run(matches : @argparse.Matches) -> Unit raise {
  let model_dir = required(matches, "model")
  let questions_path = required(matches, "questions")
  let repeat = @string.parse_int(value(matches, "repeat").unwrap_or("1")) catch {
    _ => raise @error.LayaError("--repeat expects an integer")
  }
  @error.expect(repeat >= 1, "--repeat must be at least 1")
  let backend = match value(matches, "backend").unwrap_or("metal") {
    "cpu" => @lib.Backend::Cpu
    "metal" => @lib.Backend::Metal
    "metal-required" => @lib.Backend::MetalRequired
    other =>
      raise @error.LayaError(
        "unknown backend \{other}; expected cpu, metal or metal-required",
      )
  }
  let state = match (value(matches, "state"), value(matches, "state-file")) {
    (Some(_), Some(_)) =>
      raise @error.LayaError("pass either --state or --state-file, not both")
    (Some(text), None) => text
    (None, Some(path)) => read_input(path)
    (None, None) =>
      raise @error.LayaError(
        "a state is required: pass --state or --state-file",
      )
  }
  let questions = parse_questions(read_input(questions_path))
  @error.expect(
    !questions.is_empty(),
    "the questions file defines no questions",
  )
  let load_started = @env.now()
  let agent = @lib.Agent::load(model_dir, backend~)
  let loaded_in = @env.now() - load_started
  let mut prediction = agent.predict(state, questions)
  let predict_started = @env.now()
  for _ in 0..<repeat {
    prediction = agent.predict(state, questions)
  }
  let predicted_in = @env.now() - predict_started
  println(prediction.to_json().stringify(indent=2))

  // Timings go to stderr so stdout stays a clean JSON document.
  let per_run = predicted_in.to_double() / repeat.to_double()
  let per_question = per_run / questions.length().to_double()
  eprintln(
    "device \{agent.device()}; loaded in \{loaded_in} ms; \{repeat} run(s) of \{questions.length()} question(s) in \{predicted_in} ms (\{format_ms(per_run)} ms/run, \{format_ms(per_question)} ms/question, \{prediction.input_tokens} prompt tokens/run)",
  )
}

///|
fn required(matches : @argparse.Matches, name : String) -> String raise {
  match value(matches, name) {
    Some(found) => found
    None => raise @error.LayaError("--\{name} is required")
  }
}

///|
fn value(matches : @argparse.Matches, name : String) -> String? {
  match matches.values.get(name) {
    Some([first, ..]) => Some(first)
    _ => None
  }
}

///|
/// Read a file, or stdin when the path is `-`.
fn read_input(path : String) -> String raise {
  if path == "-" {
    read_stdin()
  } else {
    @fs.read_file_to_string(path) catch {
      _ => raise @error.LayaError("cannot read \{path}")
    }
  }
}

///|
/// Parse the Python runtime's questions JSON into ordered questions.
fn parse_questions(source : String) -> Array[(String, @lib.Question)] raise {
  let json = @json.parse(source) catch {
    _ => raise @error.LayaError("the questions file is not valid JSON")
  }
  guard json is Object(entries) else {
    raise @error.LayaError(
      "the questions file must be a JSON object keyed by question id",
    )
  }
  let out = Array(capacity=entries.length())
  for id, definition in entries {
    out.push((id, parse_question(id, definition)))
  }
  out
}

///|
fn parse_question(id : String, definition : Json) -> @lib.Question raise {
  guard definition is Object(fields) else {
    raise @error.LayaError("question \{id} must be a JSON object")
  }
  guard fields.get("type") is Some(String(kind)) else {
    raise @error.LayaError("question \{id} has no type")
  }
  let instructions = match fields.get("instructions") {
    Some(String(text)) => text
    // The Python runtime accepts structured instructions and serialises them.
    Some(other) => other.stringify()
    None => raise @error.LayaError("question \{id} has no instructions")
  }
  let criteria = fields.get("criteria")
  match kind {
    "choice" =>
      match criteria {
        Some(Array(labels)) => {
          let options = Array(capacity=labels.length())
          for label in labels {
            guard label is String(label) else {
              raise @error.LayaError(
                "question \{id}: choice labels must be strings",
              )
            }
            options.push(@lib.ChoiceOption::bare(label))
          }
          Choice(instructions~, options~)
        }
        Some(Object(described)) => {
          let options = Array(capacity=described.length())
          for label, description in described {
            options.push(
              match description {
                Null => @lib.ChoiceOption::bare(label)
                String(text) => @lib.ChoiceOption::described(label, text)
                other => @lib.ChoiceOption::described(label, other.stringify())
              },
            )
          }
          Choice(instructions~, options~)
        }
        _ =>
          raise @error.LayaError(
            "question \{id}: a choice needs criteria as a list of labels or an object of label to description",
          )
      }
    "score" =>
      match criteria {
        Some(Array(entries)) => {
          let levels = Array(capacity=entries.length())
          for entry in entries {
            levels.push(
              match entry {
                String(text) => text
                other => other.stringify()
              },
            )
          }
          Score(instructions~, levels~)
        }
        _ =>
          raise @error.LayaError(
            "question \{id}: a score needs criteria as a list of level descriptions",
          )
      }
    "noul" => {
      let outcome = key => {
        match criteria {
          Some(Object(fields)) =>
            match fields.get(key) {
              Some(String(text)) => Some(text)
              Some(Null) | None => None
              Some(other) => Some(other.stringify())
            }
          _ => None
        }
      }
      Noul(
        instructions~,
        when_false=outcome("false"),
        when_true=outcome("true"),
      )
    }
    other =>
      raise @error.LayaError(
        "question \{id}: unknown type \{other}; expected choice, score or noul",
      )
  }
}

///|
/// Two decimal places, without pulling in a formatting library.
fn format_ms(value : Double) -> String {
  let scaled = (value * 100.0).round().to_int()
  "\{scaled / 100}.\{(scaled % 100 + 100) % 100}"
}
