///|
/// A checkpoint's tokenizer plus the four special token ids Laya's prompt
/// format needs.
///
/// Laya never asks the tokenizer to add special tokens: it places `[CLS]`,
/// `[SEP]` and the per-option `[MASK]` markers itself, so the token ids matter
/// but the `post_processor` does not.
struct Vocabulary {
  backend : @tokenizer.Tokenizer
  preprocess : Preprocess
  mask_token : String
  cls_id : Int
  sep_id : Int
  pad_id : Int
  mask_id : Int
}

///|
/// Normalisation `Vocabulary` performs itself, having removed the equivalent
/// stage from the backend tokenizer. See `retune` for why.
priv enum Preprocess {
  /// The backend's own normalizer is used, or there is none.
  Backend
  /// Apply NFC here, with the backend's NFC stage removed.
  Nfc
}

///|
/// The `[MASK]`-equivalent token text, which Laya strips out of user input so a
/// prompt cannot forge an option marker.
pub fn Vocabulary::mask_token(self : Vocabulary) -> String {
  self.mask_token
}

///|
/// Token id of the sequence-start token (`[CLS]` / `<bos>`).
pub fn Vocabulary::cls_id(self : Vocabulary) -> Int {
  self.cls_id
}

///|
/// Token id of the separator token (`[SEP]` / `<eos>`).
pub fn Vocabulary::sep_id(self : Vocabulary) -> Int {
  self.sep_id
}

///|
/// Token id of the padding token.
pub fn Vocabulary::pad_id(self : Vocabulary) -> Int {
  self.pad_id
}

///|
/// Token id of the option-marker token (`[MASK]` / `<mask>`).
pub fn Vocabulary::mask_id(self : Vocabulary) -> Int {
  self.mask_id
}

///|
/// Load `tokenizer/tokenizer.json` and `tokenizer/tokenizer_config.json` from a
/// checkpoint directory.
pub fn Vocabulary::load(
  directory : String,
) -> Vocabulary raise @error.LayaError {
  let tokenizer_path = "\{directory}/tokenizer.json"
  let config_path = "\{directory}/tokenizer_config.json"
  let backend = @tokenizer.from_file(tokenizer_path) catch {
    err =>
      raise @error.LayaError("cannot load \{tokenizer_path}: \{err.message()}")
  }
  let config = @fs.read_file_to_string(config_path) catch {
    _ => raise @error.LayaError("cannot read \{config_path}")
  }
  let config = @json.parse(config) catch {
    _ => raise @error.LayaError("\{config_path} is not valid JSON")
  }
  let backend = backend.no_padding().no_truncation()
  let (backend, preprocess) = retune(backend)
  let resolve = name => {
    let token = special_token(config, name)
    match token {
      None => raise @error.LayaError("\{config_path} does not define \{name}")
      Some(token) =>
        match backend.token_to_id(token) {
          Some(id) => (token, id)
          None =>
            raise @error.LayaError(
              "\{name} is \{token}, which is not in the tokenizer's vocabulary",
            )
        }
    }
  }
  let (_, cls_id) = resolve("cls_token")
  let (_, sep_id) = resolve("sep_token")
  let (_, pad_id) = resolve("pad_token")
  let (mask_token, mask_id) = resolve("mask_token")
  { backend, preprocess, mask_token, cls_id, sep_id, pad_id, mask_id, }
}

///|
/// Replace normalisation stages that `howtomakeaname/tokenizers-moonbit@0.9.1`
/// does not reproduce faithfully, having verified the substitutes against
/// Hugging Face `tokenizers` case by case in `vocabulary_test.mbt`.
///
/// Two stages need this, one per published checkpoint:
///
/// * **`Replace(" " → "▁")` ahead of `Metaspace`** (mmBERT). Hugging Face's
///   Metaspace prepends its replacement character only when the text does not
///   already start with it; the port tests for a literal space, so once the
///   normalizer has rewritten a leading space the marker is prepended twice and
///   `" false: …"` gains a spurious `▁`. Dropping the normalizer and letting
///   Metaspace do the substitution — which is what Metaspace is for — matches
///   Hugging Face exactly, and stays correct if the upstream test is widened.
///
/// * **`NFC`** (ModernBERT). The port decomposes precomposed Hangul syllables
///   into jamo without recomposing them, so `"테스트"` tokenizes as raw jamo
///   bytes. Latin composition is fine, but rather than special-case Hangul this
///   defers to `moonbit-community/normalization`, which normalises Hangul
///   algorithmically as the standard specifies.
fn retune(backend : @tokenizer.Tokenizer) -> (@tokenizer.Tokenizer, Preprocess) {
  match backend.get_normalizer() {
    // `ReplaceString` is the literal-pattern form; `Replace` is the regex form
    // and is deliberately left alone.
    Some(ReplaceString(pattern~, content~)) => {
      guard backend.get_pre_tokenizer() is Some(Metaspace(replacement~, ..)) else {
        return (backend, Backend)
      }
      if pattern == " " &&
        content.get_char(0) == Some(replacement) &&
        content.char_length() == 1 {
        (backend.with_normalizer(None), Backend)
      } else {
        (backend, Backend)
      }
    }
    Some(NFC) => (backend.with_normalizer(None), Nfc)
    _ => (backend, Backend)
  }
}

///|
/// Special tokens in `tokenizer_config.json` are either a bare string or an
/// `AddedToken` object carrying the text under `content`.
fn special_token(config : Json, name : String) -> String? {
  guard config is Object(fields) else { return None }
  match fields.get(name) {
    Some(String(value)) => Some(value)
    Some(Object(added)) =>
      match added.get("content") {
        Some(String(value)) => Some(value)
        _ => None
      }
    _ => None
  }
}

///|
/// Encode `text` without adding any special tokens.
pub fn Vocabulary::encode(
  self : Vocabulary,
  text : String,
) -> Array[Int] raise @error.LayaError {
  if text == "" {
    return []
  }
  let text = match self.preprocess {
    Backend => text
    Nfc => @normalization.nfc(text)
  }
  self.backend.encode(text, add_special_tokens=false).get_ids() catch {
    err => raise @error.LayaError("tokenization failed: \{err.message()}")
  }
}

///|
/// Size of the tokenizer's vocabulary, including added tokens.
pub fn Vocabulary::size(self : Vocabulary) -> Int {
  self.backend.get_vocab_size()
}
