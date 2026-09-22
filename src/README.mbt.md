# laya.mbt

Laya typed-decision inference for MoonBit, as a native library. No Python, no
PyTorch, no MLX at run time — a `moon` build, a checkpoint directory, and
either the platform's BLAS or its GPU.

[Laya](https://github.com/NandhaKishorM/laya) answers typed questions about a
piece of state — a ticket, an email, a JSON blob — in a single forward pass of a
bidirectional encoder. There is no token-by-token generation, so there is
nothing to hallucinate: every answer is a distribution over options the caller
supplied.

Three question types, all from one pass:

| type | answer |
|---|---|
| `choice` | the top label, plus a probability per label |
| `score` | a probability-weighted level on an ordinal scale |
| `noul` | a calibrated probability that a statement holds |

## Status

Both published checkpoints run on both backends — BLAS on the CPU and Metal on
the GPU — and agree with the Python runtime:

- **Per-stage numerics** match `laya-mlx` in float32 to within 1e-5 *relative*
  at every stage — embeddings, each encoder layer, the decision head, the
  scorer. Final option logits agree to <1e-4 absolute.
- **Tokenization** is identical to Hugging Face `tokenizers` across 76 cases per
  checkpoint, including leading/repeated whitespace, CJK, Hangul, decomposed
  accents and ligatures.
- **Rendered output** is byte-identical to `laya_mlx.Agent.predict`'s JSON for
  every fixture case, on both checkpoints **and on both backends**. The two
  backends are also compared to each other at full precision, since the
  rendered JSON rounds to four decimals.

The checks are real tests (`moon test --target native`), not claims; see
[Verifying against the reference](#verifying-against-the-reference).

## Requirements

- MoonBit toolchain with the native backend (developed against `moon`
  0.1.20260915).
- A BLAS: Apple **Accelerate** on macOS (always present), **OpenBLAS**
  elsewhere (`libopenblas-dev`).
- Node, for `build.js`, which embeds the Metal kernels and picks the
  platform's linker flags.
- ~1.5 GB of RAM for the large checkpoint, ~0.8 GB for the multilingual one.
  The GPU backend adds nothing: on Apple silicon it addresses the same weights.

The Metal backend needs no extra dependencies — Metal and
MetalPerformanceShaders are system frameworks. On non-Apple platforms the GPU
stub compiles to a handful of no-ops and `Backend::Metal` falls back to the
CPU; because that stub is an `.m` file, a non-Apple build wants an
Objective-C-capable compiler (`CC=clang`).

## Getting a checkpoint

Weights are not redistributed here. Any of the published Laya checkpoints work;
the `-mlx` conversions are convenient because they are already float16
safetensors with the tokenizer alongside.

```bash
mkdir -p models/laya-multilingual-mlx
cd models/laya-multilingual-mlx
BASE=https://huggingface.co/aac6fef/laya-multilingual-mlx/resolve/main
curl -sLO $BASE/model.safetensors
curl -sLO $BASE/rl_agent_config.json
mkdir -p encoder tokenizer
curl -sL $BASE/encoder/config.json      -o encoder/config.json
curl -sL $BASE/tokenizer/tokenizer.json -o tokenizer/tokenizer.json
curl -sL $BASE/tokenizer/tokenizer_config.json -o tokenizer/tokenizer_config.json
```

`Agent::load` wants exactly that layout:

```text
<directory>/model.safetensors
<directory>/rl_agent_config.json
<directory>/encoder/config.json
<directory>/tokenizer/tokenizer.json
<directory>/tokenizer/tokenizer_config.json
```

Available checkpoints:

| repository | encoder | params | context |
|---|---|---|---|
| `aac6fef/laya-mlx` | ModernBERT-large | 421M | 512 |
| `aac6fef/laya-multilingual-mlx` | mmBERT-base | 322M | 1024 |
| `aac6fef/laya-typed-decisions-mlx` | ModernBERT-large | 421M | 1024 |

## Using the library

```bash
moon add majikxu/laya
```

```mbt nocheck
///|
fn triage() -> Unit raise {
  // `backend` defaults to Metal, falling back to the CPU when there is no GPU.
  // Pass `backend=Cpu` to force it; `agent.device()` reports what you got.
  let agent = @laya.Agent::load("models/laya-multilingual-mlx")
  let state = @laya.State(
    "I was billed twice this month. Please refund the duplicate charge.",
  )
  let prediction = agent.predict(state, [
    Choice(
      id="department",
      instructions="Who should handle this?",
      options=["billing", "technical", "sales"].map(label => {
        label,
        description: None,
      }),
    ),
    Score(id="urgency", instructions="How urgent is this?", levels=[
      "not urgent", "soon", "urgent", "critical",
    ]),
    Noul(
      id="refund",
      instructions="The customer is asking for a refund.",
      when_false=None,
      when_true=None,
    ),
  ])
  for answer in prediction.answers {
    match answer.decision {
      Choice(label~, ..) =>
        println("\{answer.id}: \{label} (\{answer.confidence})")
      Score(value~, ..) =>
        println("\{answer.id}: \{value} (\{answer.confidence})")
      Noul(probability~) => println("\{answer.id}: \{probability}")
    }
  }
}
```

For one question at a time, `Agent::answer` skips the `Prediction` wrapper.
`Prediction::to_json` renders the same JSON document the Python runtime returns,
rounding to four decimals as it does, which is handy for diffing the two.

Options can carry descriptions, which the model sees but which are never echoed
back in the answer — set `description` on the `ChoiceOption` directly:

```mbt nocheck
Choice(
  id="action",
  instructions="What should the billing system do next?",
  options=[
    { label: "dunning", description: Some("send a payment reminder") },
    { label: "suspend", description: Some("suspend the workspace") },
    { label: "wait", description: Some("take no action yet") },
  ],
)
```

`State` wraps text. If yours is structured, serialize it first — Laya's Python
runtime uses `json.dumps(state, ensure_ascii=False)`, so matching that gives
matching tokens.

### Prompt format

Nothing about the prompt is hidden; `Agent::prompt` returns it without running
the model.

```text
[CLS] <type> question: <instructions> [SEP] [MASK] <option 0> [MASK] <option 1> … [SEP] <state> [SEP]
```

The scorer reads exactly the `[MASK]` positions, one per option. Instructions
and state are shared `head_max_len` / `max_len` budgets from the checkpoint's
`rl_agent_config.json`; the state is truncated from the right, and options are
trimmed to an equal share rather than dropped when they overflow. Any mask token
appearing in caller-supplied text is replaced with a space, so a hostile state
cannot plant an extra option marker.

The option rendering follows upstream exactly, including the corners: a `choice`
option with no description renders as just its label, where only `None` and `""`
count as "no description"; `score` options render as `level <i>: <text>`; `noul`
is always two options with fixed defaults.

## Using the CLI

```bash
moon build --target native --release
./_build/native/release/build/cmd/laya/laya.exe \
  --model models/laya-multilingual-mlx \
  --state "I was billed twice this month. Please refund the duplicate charge." \
  --questions examples/questions.json
```

`--questions` takes the same JSON the Python runtime accepts, so an existing
question file works unchanged:

```json
{
  "department": {
    "type": "choice",
    "instructions": "Who should handle this?",
    "criteria": ["billing", "technical", "sales"]
  },
  "refund": {
    "type": "noul",
    "instructions": "The customer is asking for a refund."
  }
}
```

The prediction goes to stdout as JSON; `--repeat N` puts timings and the chosen
device on stderr so stdout stays a clean document. `--backend` takes `metal`
(default), `metal-required` (fail instead of falling back) or `cpu`.
`--state-file` and `--questions` both accept `-` for stdin.

## Performance

Apple M4 Pro, one question, float32 throughout, 25 runs each measured back to
back. `speedup` is CPU time divided by Metal time.

| checkpoint | prompt tokens | CPU | Metal | speedup |
|---|---|---|---|---|
| multilingual (322M) | 23 | 19.5 ms | 36.3 ms | 0.54× |
| multilingual | 79 | 45.4 ms | 25.6 ms | **1.77×** |
| multilingual | 304 | 219 ms | 51.6 ms | **4.25×** |
| multilingual | 1024 | 1609 ms | 157 ms | **10.3×** |
| large (421M) | 23 | 46.7 ms | 72.2 ms | 0.65× |
| large | 79 | 102 ms | 58.2 ms | **1.76×** |
| large | 304 | 465 ms | 122 ms | **3.82×** |
| large | 512 | 1046 ms | 217 ms | **4.83×** |

The crossover is between 32 and 49 prompt tokens on both checkpoints. Below it
the CPU wins; above it Metal pulls away fast, because the CPU path's cost grows
roughly linearly in tokens while the GPU's barely moves until the matrices get
large enough to matter.

**Pick `Backend::Cpu` if your prompts are consistently under ~50 tokens.** The
default is `Metal`, which suits the inputs Laya is actually for — a ticket, an
email, a serialised record — and `Agent::device` reports what you got.

Peak RSS is 0.76 GB (multilingual) and 1.44 GB (large) on either backend; load
takes 0.2–0.5 s.

### Where the remaining time goes

Two things, both addressable, and neither specific to the GPU:

- **No batching.** Each question is a separate forward pass, so a three-question
  call streams the weights three times. At short prompt lengths this workload is
  bound by exactly that: one multilingual pass reads 441 MB of float32 weights
  regardless of how many tokens it is processing. The reference runtime pads a
  batch of questions into one pass and masks the padding out of every attention
  row, which does not change results — `src/model/forward.mbt` notes where the
  mask would go back.
- **float32 weights.** Halving them to float16 would halve that 441 MB, and MPS
  multiplies float16 directly. This is what `laya-mlx` does by default, and it
  is the single biggest remaining lever on both backends.

For reference, `laya-mlx` on Metal in float16 answers the three-question example
at roughly 4 ms/question against our 37 ms; batching and float16 are most of
that gap.

## How it fits together

```text
src/                      facade: Agent, Question, Answer, Prompt, Vocabulary
src/error/                LayaError, shared by every layer
src/tensor/               the numeric layer, both backends:
  laya_native.c             BLAS, float16 decoding, checkpoint reads
  ops.mbt                   CPU kernels (layer norm, GELU, RoPE, softmax)
  laya_metal.m              Metal device, buffers, dispatch, MPS matmul
  metal/laya_kernels.metal  GPU kernels, one per CPU kernel
src/safetensors/          safetensors header parsing and typed tensor reads
src/model/                config parsing, the weight arena, both forward passes
src/cmd/laya/             the CLI
ref/                      Python reference runtime and fixture generators
```

The native/MoonBit split is deliberate. `src/tensor/laya_native.c` owns only
what has to be native:

- **GEMM**, through `cblas_sgemm`. PyTorch stores `Linear` weights as
  `[out, in]`, so the transpose is expressed to BLAS rather than materialised.
- **The weight arena**: one flat float32 allocation outside the GC heap, since
  ModernBERT-large expands to ~1.4 GB and there is no reason for the collector
  to walk it. MoonBit addresses it by element offset.
- **Reading the checkpoint**, with `pread` into a 256 KiB staging buffer.

Everything above that is MoonBit operating on `FixedArray[Float]`, which the
native backend represents as a plain `float*` — so the kernels are ordinary
readable loops and the same arrays go to BLAS with no copy.

### The Metal backend

MoonBit drives Metal exactly as it drives the CPU: one call per operation, in
the same order, from the same `Model`. The two paths are deliberately
structural mirrors, so a numerical divergence localises to a single kernel —
and `src/tensor/metal_test.mbt` checks each GPU kernel against its CPU twin
before any of them are composed into a forward pass.

What differs is that the Metal calls only *encode*. `begin` opens a command
buffer, the encoder, the decision head and the scorer all queue into it, and
`commit` submits once and waits — one synchronisation per question rather than
one per operation.

- **Matrix multiplication** is `MPSMatrixMultiplication`, cached by shape: a
  pass issues around ninety multiplies but only a handful of distinct shapes,
  and building one is not free. Attention batches all heads into a single call
  using uniform strides.
- **Weights are not copied to the GPU.** The float32 arena is mapped at a page
  boundary specifically so `newBufferWithBytesNoCopy:` can wrap it; on Apple
  silicon the CPU and GPU address the same bytes. ModernBERT-large would
  otherwise need a second 1.4 GB.
- **The token embedding stays on the CPU.** It is the largest tensor in the
  checkpoint and is only ever indexed, so rows are read from the file and
  uploaded as a `[len, hidden]` block.
- **The action head stays on the CPU.** It is two multiplies against a single
  row, and its calibration features come from a softmax over the option logits
  — which would force a round trip anyway.
- **GELU is hand-written.** The Metal shading language has no `erf`, so the
  shader uses Abramowitz & Stegun 7.1.26 (max absolute error 1.5e-7) and a test
  compares it against the CPU path's libm `erff` across 4001 points.

Reference counting in `laya_metal.m` is manual. `moon.pkg` cannot pass
`-fobjc-arc` to a native stub: that flag lives under a `link` block, and
setting `link` on a library package makes moon try to link it as an executable.
Every entry point therefore wraps its body in `@autoreleasepool`, and the
long-lived objects are held at +1 from their constructor.

Small parameters (norm gains, biases, the type embedding) are decoded into
MoonBit arrays instead of the arena, so the elementwise kernels never cross FFI
to read them. The token embedding — the single largest tensor, 393 MiB on the
multilingual checkpoint — is never converted at all; its rows are read and
decoded on demand.

### One question per forward pass

The reference runtime collates questions into a padded batch and then masks the
padding out of every attention row. Running one unpadded sequence at a time
gives identical results for real tokens, and removes the attention mask
entirely: with no padding, global attention attends to everything and sliding
attention is just `|i - j| <= local_attention / 2`. Peak memory follows the
actual sequence length rather than the longest question in the batch.

One padding detail does survive, because the model depends on it: the action
head's features include the margin between the top two option probabilities, so
a one-option `choice` is padded to two slots with a `-1e4` logit, exactly as
upstream does.

## Verifying against the reference

The parity tests need weights under `models/` and fixtures under `fixtures/`,
neither of which is in version control. Without them the tests report a skip
rather than failing, so `moon test` is still useful on a fresh clone.

```bash
cd ref && uv sync                       # installs laya-mlx as the oracle
uv run python dump_fixtures.py ../models/laya-multilingual-mlx \
    ../fixtures/laya-multilingual.json --full
uv run python dump_fixtures.py ../models/laya-mlx ../fixtures/laya-en.json --full
uv run python dump_tokenizer_fixtures.py ../models/laya-multilingual-mlx/tokenizer \
    ../fixtures/tokenizer-multilingual.json
uv run python dump_tokenizer_fixtures.py ../models/laya-mlx/tokenizer \
    ../fixtures/tokenizer-en.json
cd .. && moon test --target native
```

`--full` additionally records per-stage activations for one short case, which is
what you want when a change moves the numbers and you need to know *where*.

The GPU kernels are checked twice over: each one against its CPU twin in
`src/tensor/metal_test.mbt`, and then the composed pass against the Python
runtime and against the CPU backend at full precision. A Metal-less machine
skips the GPU tests rather than failing them.

Intermediate activations are compared relative to their own scale. The residual
stream inside a ModernBERT encoder grows large — the last layer of the
multilingual checkpoint peaks near 1.2e4 — so a fixed absolute bound there would
be measuring float32 resolution rather than agreement.

## Two upstream tokenizer corrections

`Vocabulary::load` rewrites two normalisation stages of
`howtomakeaname/tokenizers-moonbit@0.9.1`, having verified the substitutes
against Hugging Face case by case. If you are reading token ids and wondering
why the pipeline does not look like `tokenizer.json`, this is why.

- **`Replace(" " → "▁")` ahead of `Metaspace`** (mmBERT). Hugging Face's
  Metaspace prepends its replacement character only when the text does not
  already start with it; the port tests for a literal space, so once the
  normalizer has rewritten a leading space the marker is prepended twice.
  `" false: …"` gains a spurious `▁`, which matters because every option body is
  built with a leading space. Dropping the normalizer and letting Metaspace do
  the substitution itself matches Hugging Face exactly, and stays correct if the
  upstream test is widened.
- **`NFC`** (ModernBERT). The port decomposes precomposed Hangul syllables into
  jamo without recomposing them, so `"테스트"` tokenizes as raw jamo bytes.
  Latin composition is fine, but rather than special-case Hangul this defers to
  `moonbit-community/normalization`.

## What is not here

- **Batched inference.** One question per forward pass; see
  [Performance](#performance).
- **float16 compute.** Checkpoints are decoded to float32 on both backends.
- **GPU outside Apple platforms.** No CUDA or Vulkan backend.
- **The router.** Upstream's `Router` picks a checkpoint per request. Load the
  `Agent` you want.
- **Presets, email helpers, shortlisting.** Upstream's convenience layers on top
  of `predict` are not ported.
- **Quantization.** `bfloat16` payloads are recognised by the header parser but
  not yet decoded.

## Licence and attribution

This package is Apache-2.0. It is an independent reimplementation of Laya's
inference path and ships no model weights.

Laya is by Convai Innovations (Apache-2.0). The prompt format, calibration and
output schema follow it; `mizorewww/laya-mlx`, itself an independent MLX port,
was the reference implementation this was checked against and the source of the
architecture details. `moonbit-community/tonyfettes-ds4` and `mizchi/blas` were
the prior art for MoonBit native FFI packaging — including embedding Metal
shader source through a prebuild step — and for BLAS linkage respectively.

## Appendix: option rendering

The rendered option text is what the model actually scores, so it is worth being
able to see it:

```mbt check
///|
test "options render as the model sees them" {
  let choice : @laya.Question = Choice(
    id="route",
    instructions="Who should handle this?",
    options=[
      { label: "billing", description: Some("payments and invoices"), },
      { label: "technical", description: Some("bugs and outages"), },
    ],
  )
  debug_inspect(
    choice.rendered_options(),
    content=(
      #|["billing: payments and invoices", "technical: bugs and outages"]
    ),
  )
  let score : @laya.Question = Score(id="urgency", instructions="How urgent?", levels=[
    "not urgent", "critical",
  ])
  debug_inspect(
    score.rendered_options(),
    content=(
      #|["level 0: not urgent", "level 1: critical"]
    ),
  )
  let noul : @laya.Question = Noul(
    id="refund",
    instructions="The customer wants a refund.",
    when_false=None,
    when_true=None,
  )
  debug_inspect(
    noul.rendered_options(),
    content=(
      #|[
      #|  "false: no, the statement does not hold",
      #|  "true: yes, the statement holds",
      #|]
    ),
  )
  inspect(noul.type_name(), content="noul")
  inspect(noul.type_index(), content="2")
}
```
