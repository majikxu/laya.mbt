name = "majikxu/laya"

version = "0.1.0"

description = "Laya typed-decision inference engine for MoonBit — native ModernBERT/mmBERT encoder, no Python runtime"

readme = "src/README.mbt.md"

repository = "https://github.com/majikxu/laya.mbt"

license = "Apache-2.0"

keywords = [ "laya", "inference", "modernbert", "native", "nlp" ]

import {
  "howtomakeaname/tokenizers-moonbit@0.9.1",
  "moonbitlang/x@0.4.45",
  "moonbit-community/normalization@0.5.0",
}

source = "src"

preferred_target = "native"

options(
  "--moonbit-unstable-prebuild": "build.js",
)
