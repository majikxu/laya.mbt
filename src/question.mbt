///|
/// One option of a `Choice` question: a label the model can return, with an
/// optional description shown to the model but never echoed back.
pub(all) struct ChoiceOption {
  label : String
  description : String?
} derive(Debug, Eq)

///|
/// A typed question about some state.
///
/// These are the three answer shapes Laya is trained to produce. All of them
/// come out of a single forward pass: the model scores one `[MASK]` marker per
/// option and the distribution over those markers is the answer.
pub(all) enum Question {
  /// Pick one label. Returns the top label and a probability per label.
  Choice(id~ : String, instructions~ : String, options~ : Array[ChoiceOption])
  /// Place the state on an ordinal scale, where `levels[i]` describes level
  /// `i`. Returns the probability-weighted level.
  Score(id~ : String, instructions~ : String, levels~ : Array[String])
  /// A calibrated probability that a statement holds. `when_false` and
  /// `when_true` override the default descriptions of the two outcomes.
  Noul(
    id~ : String,
    instructions~ : String,
    when_false~ : String?,
    when_true~ : String?
  )
} derive(Debug, Eq)

///|
/// The id this question was asked under; the key its `Answer` is returned
/// under.
pub fn Question::id(self : Question) -> String {
  match self {
    Choice(id~, ..) | Score(id~, ..) | Noul(id~, ..) => id
  }
}

///|
/// The question-type index the model's type embedding is keyed by.
pub fn Question::type_index(self : Question) -> Int {
  match self {
    Choice(..) => 0
    Score(..) => 1
    Noul(..) => 2
  }
}

///|
/// The question-type name used in the prompt and in calibration bucket keys.
pub fn Question::type_name(self : Question) -> String {
  match self {
    Choice(..) => "choice"
    Score(..) => "score"
    Noul(..) => "noul"
  }
}

///|
/// The instruction text.
pub fn Question::instructions(self : Question) -> String {
  match self {
    Choice(instructions~, ..)
    | Score(instructions~, ..)
    | Noul(instructions~, ..) => instructions
  }
}

///|
/// The option texts as the model sees them, in the order their markers appear.
///
/// A `Choice` option renders as `"label"` when it has no description and
/// `"label: description"` when it does — only `None` and `""` count as absent,
/// so a description of `"0"` or `"false"` is still rendered. `Score` renders as
/// `"level <i>: <description>"`. `Noul` is always exactly two options.
pub fn Question::rendered_options(self : Question) -> Array[String] {
  match self {
    Choice(options~, ..) =>
      options.map(option => {
        match option.description {
          None | Some("") => option.label
          Some(description) => "\{option.label}: \{description}"
        }
      })
    Score(levels~, ..) =>
      levels.mapi((index, level) => "level \{index}: \{level}")
    Noul(when_false~, when_true~, ..) =>
      [
        "false: " + describe(when_false, "no, the statement does not hold"),
        "true: " + describe(when_true, "yes, the statement holds"),
      ]
  }
}

///|
fn describe(text : String?, fallback : String) -> String {
  match text {
    None | Some("") => fallback
    Some(text) => text
  }
}

///|
/// The labels a `Choice` answer can take, in marker order.
pub fn Question::labels(self : Question) -> Array[String] {
  match self {
    Choice(options~, ..) => options.map(option => option.label)
    Score(levels~, ..) => levels.mapi((index, _) => index.to_string())
    Noul(..) => ["false", "true"]
  }
}

///|
/// Reject a question the model cannot be asked.
pub fn Question::validate(self : Question) -> Unit raise @error.LayaError {
  match self {
    Choice(options~, ..) => {
      @error.expect(
        options.length() > 0,
        "a choice question needs at least one option",
      )
      let seen = Map([])
      for option in options {
        @error.expect(
          !seen.contains(option.label),
          "choice labels must be unique, \{option.label} appears twice",
        )
        seen[option.label] = ()
      }
    }
    Score(levels~, ..) =>
      @error.expect(
        levels.length() > 0,
        "a score question needs at least one level",
      )
    Noul(..) => ()
  }
}
