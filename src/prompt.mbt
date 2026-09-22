///|
/// A tokenized question, ready for the model.
struct Prompt {
  ids : Array[Int]
  /// Position of each option's `[MASK]` marker within `ids`.
  markers : Array[Int]
  type_index : Int
} derive(Debug, Eq)

///|
/// The token ids.
pub fn Prompt::ids(self : Prompt) -> Array[Int] {
  self.ids
}

///|
/// The question-type index (0 choice, 1 score, 2 noul).
pub fn Prompt::type_index(self : Prompt) -> Int {
  self.type_index
}

///|
/// The option marker positions.
pub fn Prompt::markers(self : Prompt) -> Array[Int] {
  self.markers
}

///|
/// Number of tokens.
pub fn Prompt::length(self : Prompt) -> Int {
  self.ids.length()
}

///|
/// Per-option token budget, before the shared budget is applied.
const OPTION_TOKEN_LIMIT : Int = 48

///|
/// Build Laya's input sequence for one question over one state.
///
/// The layout is
/// `[CLS] <type> question: <instructions> [SEP] [MASK] <option 0> [MASK] <option 1> … [SEP] <state> [SEP]`,
/// and the scorer reads exactly the `[MASK]` positions. `head_max_len` bounds
/// the question half so the state always gets room; the state is then truncated
/// from the right to fit `max_len`.
///
/// Any occurrence of the mask token in user text is replaced with a space, so a
/// state or instruction cannot inject an extra option marker.
pub fn build_prompt(
  vocabulary : Vocabulary,
  state : State,
  question : Question,
  max_len~ : Int,
  head_max_len~ : Int,
) -> Prompt raise @error.LayaError {
  question.validate()
  let mask = vocabulary.mask_token()
  let rendered = question.rendered_options()
  let head = vocabulary.encode(
    "\{question.type_name()} question: \{scrub(question.instructions(), mask)}",
  )
  // Each entry is the option's marker token followed by its body, so the
  // shared budget below is applied to marker and body together.
  let options = Array(capacity=rendered.length())
  for option in rendered {
    let body = truncate(
      vocabulary.encode(" " + scrub(option, mask)),
      OPTION_TOKEN_LIMIT,
    )
    let entry = [vocabulary.mask_id()]
    for id in body {
      entry.push(id)
    }
    options.push(entry)
  }

  // The options share `head_max_len` with the instructions. If they have eaten
  // all but 16 tokens of it, they are trimmed to an equal share instead, which
  // keeps every option present rather than letting the last ones vanish.
  let mut option_budget = head_max_len - total_length(options)
  if option_budget < 16 {
    let per_option = @cmp.maximum(
      4,
      (head_max_len - 16) / @cmp.maximum(1, options.length()),
    )
    for index in 0..<options.length() {
      options[index] = truncate(options[index], per_option)
    }
    option_budget = head_max_len - total_length(options)
  }
  let ids = [vocabulary.cls_id()]
  for id in truncate(head, @cmp.maximum(8, option_budget)) {
    ids.push(id)
  }
  ids.push(vocabulary.sep_id())
  let markers = Array(capacity=options.length())
  for option in options {
    markers.push(ids.length())
    for id in option {
      ids.push(id)
    }
  }
  ids.push(vocabulary.sep_id())
  let room = @cmp.maximum(0, max_len - ids.length() - 1)
  for id in truncate(vocabulary.encode(scrub(state.text(), mask)), room) {
    ids.push(id)
  }
  ids.push(vocabulary.sep_id())
  let kept = truncate(ids, max_len)
  let visible = markers.filter(marker => marker < max_len)
  @error.expect(
    visible.length() == markers.length(),
    "question has too many options for a \{max_len}-token budget: \{markers.length() - visible.length()} marker(s) fall outside it",
  )
  { ids: kept, markers: visible, type_index: question.type_index(), }
}

///|
/// Remove the mask token from user-supplied text.
///
/// The marker positions are derived from where the builder placed mask tokens,
/// so a mask token arriving inside an instruction or a state would silently
/// shift every option's score onto the wrong token.
fn scrub(text : String, mask : String) -> String {
  if text.contains(mask) {
    text.replace_all(old=mask, new=" ")
  } else {
    text
  }
}

///|
fn truncate(ids : Array[Int], limit : Int) -> Array[Int] {
  if ids.length() <= limit {
    ids
  } else {
    ids[:limit].to_owned()
  }
}

///|
fn total_length(options : Array[Array[Int]]) -> Int {
  options.fold(init=0, (total, option) => total + option.length())
}
