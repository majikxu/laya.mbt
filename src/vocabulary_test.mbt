///|
/// Token-for-token agreement with Hugging Face `tokenizers` over a corpus that
/// deliberately stresses leading and repeated whitespace, since that is where
/// the byte-level and SentencePiece pre-tokenizers behave differently — and
/// where `Vocabulary::load`'s Metaspace correction applies.
///
/// Fixtures come from `ref/dump_tokenizer_fixtures.py`.
test "tokenization matches Hugging Face for both checkpoints" {
  let cases = [
    (
      "models/laya-multilingual-mlx/tokenizer", "fixtures/tokenizer-multilingual.json",
    ),
    ("models/laya-mlx/tokenizer", "fixtures/tokenizer-en.json"),
  ]
  let report = []
  let checked = []
  for entry in cases {
    let (directory, fixture_path) = entry
    guard read_json(fixture_path) is Some(Object(fixture)) else { continue }
    if !@fs.path_exists("\{directory}/tokenizer.json") {
      continue
    }
    let vocabulary = @laya.Vocabulary::load(directory)
    guard fixture.get("vocab_size") is Some(Number(vocab_size, ..)) else {
      fail("\{fixture_path} has no vocab_size")
    }
    if vocabulary.size() != vocab_size.to_int() {
      report.push(
        "\{directory}: vocab size \{vocabulary.size()} vs \{vocab_size.to_int()}",
      )
    }

    // The four special ids drive marker placement, so a mismatch here would
    // silently misalign every option score.
    guard fixture.get("special_tokens") is Some(Object(specials)) else {
      fail("\{fixture_path} has no special_tokens")
    }
    let special_id = name => {
      match specials.get(name) {
        Some(Object(fields)) =>
          match fields.get("id") {
            Some(Number(id, ..)) => Some(id.to_int())
            _ => None
          }
        _ => None
      }
    }
    for
      pair in [
        ("cls_token", vocabulary.cls_id()),
        ("sep_token", vocabulary.sep_id()),
        ("pad_token", vocabulary.pad_id()),
        ("mask_token", vocabulary.mask_id()),
      ] {
      if special_id(pair.0) != Some(pair.1) {
        report.push("\{directory}: \{pair.0} is \{pair.1}, reference disagrees")
      }
    }
    guard fixture.get("cases") is Some(Array(entries)) else {
      fail("\{fixture_path} has no cases")
    }
    for case in entries {
      guard case is Object(case) else { continue }
      guard expect_string(case.get("text")) is Some(text) else { continue }
      let want = ints(case.get("ids"))
      let got = vocabulary.encode(text)
      if got != want {
        report.push(
          "\{directory}: \{Repr(text)} -> \{Repr(got)}, expected \{Repr(want)}",
        )
      }
      checked.push(text)
    }
  }
  if checked.is_empty() {
    println("skipped: no tokenizer and fixture pair is available")
    return
  }
  debug_inspect(report, content="[]")
  inspect(checked.length(), content="76")
}

///|
test "the mask token is stripped out of user text" {
  let directory = "models/laya-multilingual-mlx"
  if !@fs.path_exists("\{directory}/model.safetensors") {
    println("skipped: \{directory} has no weights")
    return
  }
  let agent = @laya.Agent::load(directory)
  let mask = agent.vocabulary().mask_token()

  // A state that tries to plant an extra option marker must not change how many
  // markers the prompt has, nor where they sit.
  let question = @laya.Question::choice("Pick one.", ["a", "b"])
  let clean = agent.prompt("nothing unusual here", question)
  let hostile = agent.prompt("nothing \{mask} unusual \{mask} here", question)
  inspect(clean.markers() == hostile.markers(), content="true")
  inspect(clean.markers().length(), content="2")
  let mask_id = agent.vocabulary().mask_id()
  inspect(hostile.ids().filter(id => id == mask_id).length(), content="2")
}

///|
test "a question whose options overflow the budget is trimmed, not dropped" {
  let directory = "models/laya-multilingual-mlx"
  if !@fs.path_exists("\{directory}/model.safetensors") {
    println("skipped: \{directory} has no weights")
    return
  }
  let agent = @laya.Agent::load(directory)
  let config = agent.agent_config()

  // Thirty long options blow well past head_max_len; every one must still get a
  // marker, because a missing marker means a missing probability.
  let labels = Array::makei(30, index => {
    "option \{index} " + "verbose ".repeat(12)
  })
  let question = @laya.Question::choice("Pick one.", labels)
  let prompt = agent.prompt("some state", question)
  inspect(prompt.markers().length(), content="30")
  inspect(prompt.length() <= config.max_len, content="true")
  let answer = agent.answer("some state", question)
  match answer.decision {
    Choice(probabilities~, ..) => inspect(probabilities.length(), content="30")
    _ => fail("expected a choice decision")
  }
}
