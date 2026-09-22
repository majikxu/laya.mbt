///|
/// A flat float32 parameter arena held outside the MoonBit heap.
///
/// ModernBERT-large expands to roughly 1.4 GiB of float32 weights, which is not
/// something you want the collector walking on every cycle, so the arena is a
/// single `calloc` behind an external object with a finalizer.
pub type Arena

///|
extern "c" fn arena_new(len : Int64) -> Arena = "moonbit_laya_arena_new"

///|
#borrow(arena)
extern "c" fn arena_len(arena : Arena) -> Int64 = "moonbit_laya_arena_len"

///|
#borrow(arena)
extern "c" fn arena_get(arena : Arena, index : Int64) -> Float = "moonbit_laya_arena_get"

///|
/// Allocate an arena of `len` float32 slots, zero-initialised.
pub fn Arena::new(len : Int64) -> Arena raise @error.LayaError {
  let arena = arena_new(len)
  if arena_len(arena) != len {
    raise @error.LayaError(
      "failed to allocate \{len * 4L} bytes for the weight arena",
    )
  }
  arena
}

///|
/// Number of float32 slots in the arena.
pub fn Arena::capacity(self : Arena) -> Int64 {
  arena_len(self)
}

///|
/// Read one slot. Out-of-range reads return `0.0`; this exists for tests and
/// diagnostics, not for the hot path — the kernels address the arena natively.
pub fn Arena::get(self : Arena, index : Int64) -> Float {
  arena_get(self, index)
}

///|
/// A read-only handle on a safetensors checkpoint.
///
/// Tensors are read with `pread` rather than mapped. Mapping looks attractive
/// for the token embedding — 393 MiB on the multilingual checkpoint, and only
/// ever indexed — but it keeps the whole file resident once loading has touched
/// it, and Darwin will not hand those pages back. Reading keeps the footprint
/// at the float32 arena plus a staging buffer.
pub type Reader

///|
#borrow(path)
extern "c" fn reader_open(path : Bytes) -> Reader = "moonbit_laya_reader_open"

///|
#borrow(reader)
extern "c" fn reader_header_len(reader : Reader) -> Int64 = "moonbit_laya_reader_header_len"

///|
#borrow(reader)
extern "c" fn reader_size(reader : Reader) -> Int64 = "moonbit_laya_reader_size"

///|
#borrow(reader)
extern "c" fn reader_header(reader : Reader) -> Bytes = "moonbit_laya_reader_header"

///|
#borrow(reader, arena)
extern "c" fn reader_read_f16(
  reader : Reader,
  offset : Int64,
  count : Int64,
  arena : Arena,
  dst : Int64,
) -> Int = "moonbit_laya_reader_read_f16"

///|
#borrow(reader, arena)
extern "c" fn reader_read_f32(
  reader : Reader,
  offset : Int64,
  count : Int64,
  arena : Arena,
  dst : Int64,
) -> Int = "moonbit_laya_reader_read_f32"

///|
#borrow(reader, out)
extern "c" fn reader_read_f16_array(
  reader : Reader,
  offset : Int64,
  count : Int,
  out : FixedArray[Float],
) -> Int = "moonbit_laya_reader_read_f16_array"

///|
#borrow(reader, rows, out)
extern "c" fn reader_gather_f16(
  reader : Reader,
  offset : Int64,
  rows : FixedArray[Int],
  n : Int,
  dim : Int,
  vocab : Int,
  out : FixedArray[Float],
) -> Int = "moonbit_laya_reader_gather_f16"

///|
/// Open a safetensors file and validate its 8-byte header length prefix.
pub fn Reader::open(path : String) -> Reader raise @error.LayaError {
  let reader = reader_open(@utf8.encode(path))
  match reader_header_len(reader) {
    -1L => raise @error.LayaError("cannot open \{path}")
    -2L =>
      raise @error.LayaError("\{path} is too small to be a safetensors file")
    -3L => raise @error.LayaError("cannot read the header of \{path}")
    -4L =>
      raise @error.LayaError(
        "\{path} has an out-of-range safetensors header length",
      )
    _ => reader
  }
}

///|
/// The raw JSON header of the safetensors file.
pub fn Reader::header(self : Reader) -> Bytes {
  reader_header(self)
}

///|
/// Byte offset at which tensor payloads begin.
pub fn Reader::data_start(self : Reader) -> Int64 {
  8L + reader_header_len(self)
}

///|
/// Total file size in bytes.
pub fn Reader::size(self : Reader) -> Int64 {
  reader_size(self)
}

///|
/// Decode `count` float16 values at `offset` into `arena` starting at element
/// index `dst`.
pub fn Reader::read_f16(
  self : Reader,
  offset : Int64,
  count : Int64,
  arena : Arena,
  dst : Int64,
) -> Unit raise @error.LayaError {
  @error.expect(
    reader_read_f16(self, offset, count, arena, dst) == 0,
    "float16 read of \{count} values at \{offset} is out of range",
  )
}

///|
/// Copy `count` float32 values at `offset` into `arena` starting at `dst`.
pub fn Reader::read_f32(
  self : Reader,
  offset : Int64,
  count : Int64,
  arena : Arena,
  dst : Int64,
) -> Unit raise @error.LayaError {
  @error.expect(
    reader_read_f32(self, offset, count, arena, dst) == 0,
    "float32 read of \{count} values at \{offset} is out of range",
  )
}

///|
/// Decode float16 values at `offset` into a MoonBit array, filling it entirely.
pub fn Reader::read_f16_into(
  self : Reader,
  offset : Int64,
  out : FixedArray[Float],
) -> Unit raise @error.LayaError {
  @error.expect(
    reader_read_f16_array(self, offset, out.length(), out) == 0,
    "float16 read of \{out.length()} values at \{offset} is out of range",
  )
}

///|
/// Gather `rows.length()` rows of width `dim` out of a float16 matrix of
/// `vocab` rows located at `offset`, writing them contiguously into `out`.
pub fn Reader::gather_rows_f16(
  self : Reader,
  offset : Int64,
  rows : FixedArray[Int],
  dim : Int,
  vocab : Int,
  out : FixedArray[Float],
) -> Unit raise @error.LayaError {
  @error.expect(
    out.length() >= rows.length() * dim,
    "gather destination holds \{out.length()} floats, need \{rows.length() * dim}",
  )
  @error.expect(
    reader_gather_f16(self, offset, rows, rows.length(), dim, vocab, out) == 0,
    "row gather at \{offset} is out of range",
  )
}

///|
#borrow(x, arena, y)
extern "c" fn laya_linear(
  x : FixedArray[Float],
  arena : Arena,
  weight_offset : Int64,
  bias_offset : Int64,
  y : FixedArray[Float],
  m : Int,
  k : Int,
  n : Int,
) -> Unit = "moonbit_laya_linear"

///|
/// `y[m, n] = x[m, k] · W[n, k]ᵀ (+ bias)`, with `W` and `bias` living in the
/// arena at the given element offsets. Pass `bias_offset = -1` for no bias.
///
/// The transpose matches the PyTorch `Linear` layout the checkpoints use and is
/// handed to BLAS rather than materialised.
pub fn linear(
  x : FixedArray[Float],
  arena : Arena,
  weight_offset : Int64,
  bias_offset : Int64,
  y : FixedArray[Float],
  m~ : Int,
  k~ : Int,
  n~ : Int,
) -> Unit {
  laya_linear(x, arena, weight_offset, bias_offset, y, m, k, n)
}

///|
#borrow(a, b, c)
extern "c" fn laya_gemm(
  a : FixedArray[Float],
  a_offset : Int,
  lda : Int,
  b : FixedArray[Float],
  b_offset : Int,
  ldb : Int,
  c : FixedArray[Float],
  c_offset : Int,
  ldc : Int,
  m : Int,
  n : Int,
  k : Int,
  trans_b : Int,
  alpha : Float,
  beta : Float,
) -> Unit = "moonbit_laya_gemm"

///|
/// `C = alpha · A · op(B) + beta · C` over activation arrays, with explicit
/// offsets and leading dimensions so a single head's slice of a packed
/// `[heads, len, dim]` buffer can be multiplied without copying.
pub fn gemm(
  a : FixedArray[Float],
  b : FixedArray[Float],
  c : FixedArray[Float],
  m~ : Int,
  n~ : Int,
  k~ : Int,
  a_offset? : Int = 0,
  b_offset? : Int = 0,
  c_offset? : Int = 0,
  lda~ : Int,
  ldb~ : Int,
  ldc~ : Int,
  trans_b? : Bool = false,
  alpha? : Float = 1.0,
  beta? : Float = 0.0,
) -> Unit {
  laya_gemm(
    a,
    a_offset,
    lda,
    b,
    b_offset,
    ldb,
    c,
    c_offset,
    ldc,
    m,
    n,
    k,
    if trans_b {
      1
    } else {
      0
    },
    alpha,
    beta,
  )
}
