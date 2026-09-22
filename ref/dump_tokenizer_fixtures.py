"""Dump reference token ids so laya.mbt's tokenizer wrapper can be checked.

Run:  uv run python dump_tokenizer_fixtures.py <tokenizer-dir> <out.json>

The corpus deliberately includes the shapes Laya's prompt builder produces —
`"<type> question: ..."` heads and `" <option>"` bodies with their leading
space — because leading whitespace is exactly where SentencePiece-style and
byte-level pre-tokenizers disagree.
"""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

CORPUS = [
    "",
    " ",
    "  ",
    "refund me",
    " refund me",
    "refund me ",
    "  double  space ",
    "\ttab\tseparated\t",
    "line\nbreak",
    "choice question: Who should handle this?",
    "score question: How urgent is this?",
    "noul question: The customer wants money back.",
    " billing",
    " technical: needs an engineer",
    " level 0: not urgent",
    " false: no, the statement does not hold",
    " true: yes, the statement holds",
    " closed",
    "I was billed twice this month. Please refund the duplicate charge.",
    "我这个月被重复扣款了两次，请把多收的那笔退给我。",
    " 账务",
    "これは日本語のテストです。",
    "Это тест на русском языке.",
    "اختبار باللغة العربية",
    "테스트 문장입니다",
    "café résumé naïve",
    "cafe\u0301 re\u0301sume\u0301",  # decomposed accents: NFC must compose them
    "테스트 문장입니다",  # precomposed Hangul: NFC must leave it alone
    "\u1112\u1161\u11ab\u1100\u116e\u11a8",  # Hangul jamo: NFC must compose them
    "\ufb01 ligature and \u216b roman",  # NFC keeps these; only NFKC folds them
    "\u0041\u030a vs \u00c5 vs \u212b",  # three spellings of A-ring
    "emoji 🙂 and 👨‍👩‍👧‍👦 family",
    '{"plan": "pro", "seats": 12, "overdue_days": 41, "region": "EU"}',
    '{"nested": {"a": [1, 2, 3], "b": null}, "flag": true}',
    "trailing punctuation!!! ??? ...",
    "MiXeD CaSe AnD 12345 numbers",
    "a" * 200,
    "word " * 60,
]


def main():
    tokenizer_dir = Path(sys.argv[1])
    out_path = sys.argv[2]
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    config = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())

    specials = {}
    for name in ("cls_token", "sep_token", "pad_token", "mask_token", "unk_token"):
        value = config.get(name)
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str):
            specials[name] = {"token": value, "id": tokenizer.token_to_id(value)}

    cases = [
        {"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids}
        for text in CORPUS
    ]
    with open(out_path, "w") as f:
        json.dump(
            {
                "tokenizer_dir": str(tokenizer_dir),
                "vocab_size": tokenizer.get_vocab_size(),
                "special_tokens": specials,
                "cases": cases,
            },
            f,
            ensure_ascii=False,
        )
    print(f"wrote {out_path}: {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
