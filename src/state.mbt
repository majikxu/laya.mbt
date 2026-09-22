///|
/// Text to evaluate: a ticket, an email, a JSON-serialised record.
///
/// A distinct type rather than a bare `String` mainly guards against passing a
/// state where an instructions string was meant, or the reverse. Structured
/// input should be serialised to text before wrapping it — Laya's Python
/// runtime uses `json.dumps(state, ensure_ascii=False)`, and matching that
/// exactly is what keeps token-level parity with it.
pub(all) struct State(String) derive(Eq, Debug)

///|
/// The underlying text.
pub fn State::text(self : State) -> String {
  let State(text) = self
  text
}
