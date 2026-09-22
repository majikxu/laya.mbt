///|
/// What the model decided, shaped by the question that was asked.
pub(all) enum Decision {
  /// The highest-probability label, plus the probability of every label.
  Choice(label~ : String, probabilities~ : Array[(String, Double)])
  /// The probability-weighted level, the probability of each level, and the
  /// level descriptions the question supplied.
  Score(
    value~ : Double,
    probabilities~ : Array[Double],
    legend~ : Array[String]
  )
  /// Calibrated probability that the statement holds.
  Noul(probability~ : Double)
} derive(Debug)

///|
/// One decision with its calibration.
pub(all) struct Answer {
  decision : Decision
  /// `1 - H(p) / log(k)` over the calibrated distribution: 1.0 when the model
  /// is certain, 0.0 when it is uniform. For `Noul`, the distance of the
  /// probability from a coin flip.
  confidence : Double
  /// The action head's probability of acting rather than escalating.
  act_probability : Double
} derive(Debug)

///|
/// Every answer for one `predict` call, in the order the questions were asked.
pub(all) struct Prediction {
  answers : Array[(String, Answer)]
  /// Total prompt tokens across all questions, which is what a Laya call costs.
  input_tokens : Int
} derive(Debug)

///|
/// Look up an answer by question id.
pub fn Prediction::get(self : Prediction, id : String) -> Answer? {
  for entry in self.answers {
    if entry.0 == id {
      return Some(entry.1)
    }
  }
  None
}

///|
/// Decode one forward pass into an answer.
///
/// The scorer's raw logits are divided by a calibration temperature before the
/// softmax. Which temperature applies depends on the question type *and* on how
/// many options it has: checkpoints ship a per-bucket table because a
/// two-option choice and an eleven-option choice are differently overconfident.
fn decode(
  question : Question,
  outputs : @model.Outputs,
  config : @model.AgentConfig,
) -> Answer {
  let options = outputs.options
  let temperature = calibration(config, question.type_name(), options)
  let scaled = Array::make(options, 0.0)
  for index in 0..<options {
    scaled[index] = outputs.logits[index] / temperature
  }
  let probabilities = @tensor.softmax(scaled[:])
  let act_probability = @tensor.softmax(outputs.action[:])[0]
  match question {
    Choice(..) => {
      let labels = question.labels()
      let mut best = 0
      for index in 1..<options {
        if probabilities[index] > probabilities[best] {
          best = index
        }
      }
      let paired = Array(capacity=options)
      for index in 0..<options {
        paired.push((labels[index], probabilities[index]))
      }
      {
        decision: Choice(label=labels[best], probabilities=paired),
        confidence: confidence_from(probabilities),
        act_probability,
      }
    }
    Score(levels~, ..) => {
      let mut value = 0.0
      for index, probability in probabilities {
        value += index.to_double() * probability
      }
      {
        decision: Score(value~, probabilities~, legend=levels),
        confidence: confidence_from(probabilities),
        act_probability,
      }
    }
    Noul(..) => {
      // A noul question always has exactly two options, so index 1 is `true`.
      let probability = probabilities[options - 1]
      {
        decision: Noul(probability~),
        confidence: @cmp.maximum(probability, 1.0 - probability),
        act_probability,
      }
    }
  }
}

///|
/// Pick the calibration temperature, preferring the per-option-count bucket.
fn calibration(
  config : @model.AgentConfig,
  type_name : String,
  options : Int,
) -> Double {
  let bucket = if options <= 2 {
    "2"
  } else if options <= 5 {
    "3-5"
  } else if options <= 10 {
    "6-10"
  } else {
    "11+"
  }
  let scale = match
    config.temperature_by_options.get("\{type_name}:\{bucket}") {
    Some(scale) => scale
    None =>
      match type_name {
        "choice" => config.temperature[0]
        "score" => config.temperature[1]
        _ => config.temperature[2]
      }
  }
  @cmp.maximum(1.0e-3, scale)
}

///|
/// Normalised Shannon entropy turned into a confidence: `1 - H(p) / log(k)`.
///
/// A single-option question is trivially certain, so `k < 2` is 1.0 rather than
/// a division by `log(1)`.
fn confidence_from(probabilities : Array[Double]) -> Double {
  let k = probabilities.length()
  if k < 2 {
    return 1.0
  }
  let mut entropy = 0.0
  for probability in probabilities {
    entropy -= probability *
      @math.ln(@cmp.maximum(1.0e-12, @cmp.minimum(1.0, probability)))
  }
  let confidence = 1.0 - entropy / @math.ln(k.to_double())
  @cmp.maximum(0.0, @cmp.minimum(1.0, confidence))
}

///|
/// Render a prediction in the same JSON shape the Python runtime returns.
///
/// Numbers are rounded to four decimals, as upstream does, so outputs can be
/// diffed directly against it. The in-memory `Answer` keeps full precision.
pub fn Prediction::to_json(self : Prediction) -> Json {
  let answers = Map([])
  for entry in self.answers {
    answers[entry.0] = entry.1.to_json()
  }
  {
    "model": "laya-rl-agent",
    "answers": Json::object(answers),
    "usage": {
      "input_tokens": Json::number(self.input_tokens.to_double()),
      "output_tokens": Json::number(0.0),
    },
  }
}

///|
/// Render one answer in the Python runtime's JSON shape.
pub fn Answer::to_json(self : Answer) -> Json {
  let fields = Map([])
  fields["confidence"] = Json::number(round4(self.confidence))
  fields["action"] = {
    "act_probability": Json::number(round4(self.act_probability)),
  }
  match self.decision {
    Choice(label~, probabilities~) => {
      fields["type"] = Json::string("choice")
      fields["choice"] = Json::string(label)
      let table = Map([])
      for pair in probabilities {
        table[pair.0] = Json::number(round4(pair.1))
      }
      fields["probabilities"] = Json::object(table)
    }
    Score(value~, probabilities~, legend~) => {
      fields["type"] = Json::string("score")
      fields["score"] = Json::number(round4(value))
      let table = Map([])
      for index, probability in probabilities {
        table[index.to_string()] = Json::number(round4(probability))
      }
      fields["probabilities"] = Json::object(table)
      let names = Map([])
      for index, level in legend {
        names[index.to_string()] = Json::string(level)
      }
      fields["legend"] = Json::object(names)
    }
    Noul(probability~) => {
      fields["type"] = Json::string("noul")
      fields["noul"] = Json::number(round4(probability))
    }
  }
  Json::object(fields)
}

///|
/// Round to four decimals, breaking ties towards the even digit as Python's
/// `round` does.
fn round4(value : Double) -> Double {
  let scaled = value * 10000.0
  let lower = scaled.floor()
  let fraction = scaled - lower
  let rounded = if fraction > 0.5 {
    lower + 1.0
  } else if fraction < 0.5 {
    lower
  } else if (lower / 2.0).floor() * 2.0 == lower {
    lower
  } else {
    lower + 1.0
  }
  rounded / 10000.0
}
