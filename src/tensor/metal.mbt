///|
/// A Metal device, command queue and the compiled kernel library.
///
/// Operations *encode* into a command buffer rather than running: `begin`
/// opens one, the operations queue work into it, and `commit` submits it and
/// waits. A whole forward pass is therefore one submission and one
/// synchronisation, which is what makes the GPU path worth having — per-op
/// round trips would cost more than they save.
pub type MetalContext

///|
/// A float32 buffer in GPU-visible memory, addressed by element offset.
pub type MetalBuffer

///|
extern "c" fn metal_context_new() -> MetalContext = "moonbit_laya_metal_context_new"

///|
#borrow(context)
extern "c" fn metal_ready(context : MetalContext) -> Int = "moonbit_laya_metal_ready"

///|
#borrow(context)
extern "c" fn metal_error(context : MetalContext) -> Bytes = "moonbit_laya_metal_error"

///|
#borrow(context)
extern "c" fn metal_device_name(context : MetalContext) -> Bytes = "moonbit_laya_metal_device_name"

///|
/// Create a Metal context, or report why it is unavailable.
///
/// Returns `Err` rather than raising on an unsupported platform, because
/// falling back to the CPU backend is a normal outcome, not a failure.
pub fn MetalContext::create() -> Result[MetalContext, String] {
  let context = metal_context_new()
  if metal_ready(context) == 1 {
    Ok(context)
  } else {
    Err(@utf8.decode_lossy(metal_error(context)))
  }
}

///|
/// Name of the GPU this context is bound to.
pub fn MetalContext::device_name(self : MetalContext) -> String {
  @utf8.decode_lossy(metal_device_name(self))
}

///|
#borrow(context)
extern "c" fn metal_buffer_new(
  context : MetalContext,
  count : Int64,
) -> MetalBuffer = "moonbit_laya_metal_buffer_new"

///|
#borrow(context, arena)
extern "c" fn metal_buffer_from_arena(
  context : MetalContext,
  arena : Arena,
) -> MetalBuffer = "moonbit_laya_metal_buffer_from_arena"

///|
#borrow(buffer)
extern "c" fn metal_buffer_ok(buffer : MetalBuffer) -> Int = "moonbit_laya_metal_buffer_ok"

///|
/// Allocate `count` float32 slots of GPU-visible storage.
pub fn MetalContext::buffer(
  self : MetalContext,
  count : Int,
) -> MetalBuffer raise @error.LayaError {
  let buffer = metal_buffer_new(self, count.to_int64())
  @error.expect(
    metal_buffer_ok(buffer) == 1,
    "could not allocate a \{count * 4}-byte Metal buffer",
  )
  buffer
}

///|
/// Expose the float32 weight arena to the GPU without copying it.
///
/// Apple silicon shares memory between CPU and GPU, and the arena is mapped at
/// a page boundary precisely so it can be wrapped here — ModernBERT-large
/// would otherwise need a second 1.4 GB copy. The arena is kept alive for as
/// long as the buffer exists.
pub fn MetalContext::wrap_arena(
  self : MetalContext,
  arena : Arena,
) -> MetalBuffer raise @error.LayaError {
  let buffer = metal_buffer_from_arena(self, arena)
  @error.expect(
    metal_buffer_ok(buffer) == 1,
    "could not expose the weight arena to Metal",
  )
  buffer
}

///|
#borrow(buffer, src)
extern "c" fn metal_upload(
  buffer : MetalBuffer,
  offset : Int64,
  src : FixedArray[Float],
  count : Int,
) -> Unit = "moonbit_laya_metal_upload"

///|
#borrow(buffer, src)
extern "c" fn metal_upload_ints(
  buffer : MetalBuffer,
  offset : Int64,
  src : FixedArray[Int],
  count : Int,
) -> Unit = "moonbit_laya_metal_upload_ints"

///|
#borrow(buffer, dst)
extern "c" fn metal_download(
  buffer : MetalBuffer,
  offset : Int64,
  dst : FixedArray[Float],
  count : Int,
) -> Unit = "moonbit_laya_metal_download"

///|
/// Copy `count` floats into the buffer at element offset `offset`.
pub fn MetalBuffer::upload(
  self : MetalBuffer,
  src : FixedArray[Float],
  count~ : Int,
  offset? : Int = 0,
) -> Unit {
  metal_upload(self, offset.to_int64(), src, count)
}

///|
/// Copy `count` ints into the buffer, for index operands such as the option
/// marker positions.
pub fn MetalBuffer::upload_ints(
  self : MetalBuffer,
  src : FixedArray[Int],
  count~ : Int,
  offset? : Int = 0,
) -> Unit {
  metal_upload_ints(self, offset.to_int64(), src, count)
}

///|
/// Copy `count` floats out of the buffer. Only valid after `commit`.
pub fn MetalBuffer::download(
  self : MetalBuffer,
  dst : FixedArray[Float],
  count~ : Int,
  offset? : Int = 0,
) -> Unit {
  metal_download(self, offset.to_int64(), dst, count)
}

///|
#borrow(context)
extern "c" fn metal_begin(context : MetalContext) -> Unit = "moonbit_laya_metal_begin"

///|
#borrow(context)
extern "c" fn metal_commit(context : MetalContext) -> Int = "moonbit_laya_metal_commit"

///|
/// Open a command buffer. Operations encoded after this run on `commit`.
pub fn MetalContext::begin(self : MetalContext) -> Unit {
  metal_begin(self)
}

///|
/// Submit the command buffer and wait for it. Buffer contents are readable
/// afterwards.
pub fn MetalContext::commit(self : MetalContext) -> Unit raise @error.LayaError {
  if metal_commit(self) != 0 {
    raise @error.LayaError(@utf8.decode_lossy(metal_error(self)))
  }
}

///|
#borrow(context, x, y, gamma, beta)
extern "c" fn metal_layer_norm(
  context : MetalContext,
  x : MetalBuffer,
  x_offset : Int64,
  y : MetalBuffer,
  y_offset : Int64,
  gamma : MetalBuffer,
  gamma_offset : Int64,
  beta : MetalBuffer,
  beta_offset : Int64,
  rows : Int,
  cols : Int,
  eps : Float,
) -> Unit = "moonbit_laya_metal_layer_norm"

///|
/// Layer normalisation over `[rows, cols]`. `beta_offset = -1` means no bias,
/// in which case `beta` is ignored and may be any buffer.
pub fn MetalContext::layer_norm(
  self : MetalContext,
  x : MetalBuffer,
  y : MetalBuffer,
  gamma : MetalBuffer,
  beta : MetalBuffer,
  rows~ : Int,
  cols~ : Int,
  x_offset? : Int = 0,
  y_offset? : Int = 0,
  gamma_offset~ : Int64,
  beta_offset~ : Int64,
  eps~ : Float,
) -> Unit {
  metal_layer_norm(
    self,
    x,
    x_offset.to_int64(),
    y,
    y_offset.to_int64(),
    gamma,
    gamma_offset,
    beta,
    beta_offset,
    rows,
    cols,
    eps,
  )
}

///|
#borrow(context, a, b, c)
extern "c" fn metal_gemm(
  context : MetalContext,
  a : MetalBuffer,
  a_offset : Int64,
  lda : Int,
  a_stride : Int64,
  b : MetalBuffer,
  b_offset : Int64,
  ldb : Int,
  b_stride : Int64,
  c : MetalBuffer,
  c_offset : Int64,
  ldc : Int,
  c_stride : Int64,
  batch : Int,
  m : Int,
  n : Int,
  k : Int,
  trans_b : Int,
  alpha : Float,
  beta : Float,
) -> Unit = "moonbit_laya_metal_gemm"

///|
/// `C = alpha · A · op(B) + beta · C`, over `batch` matrices with uniform
/// strides. `batch = 1` covers every Linear; batching covers attention, where
/// each head is one matrix.
pub fn MetalContext::gemm(
  self : MetalContext,
  a : MetalBuffer,
  b : MetalBuffer,
  c : MetalBuffer,
  m~ : Int,
  n~ : Int,
  k~ : Int,
  lda~ : Int,
  ldb~ : Int,
  ldc~ : Int,
  a_offset? : Int64 = 0L,
  b_offset? : Int64 = 0L,
  c_offset? : Int64 = 0L,
  a_stride? : Int64 = 0L,
  b_stride? : Int64 = 0L,
  c_stride? : Int64 = 0L,
  batch? : Int = 1,
  trans_b? : Bool = false,
  alpha? : Float = 1.0,
  beta? : Float = 0.0,
) -> Unit {
  metal_gemm(
    self,
    a,
    a_offset,
    lda,
    a_stride,
    b,
    b_offset,
    ldb,
    b_stride,
    c,
    c_offset,
    ldc,
    c_stride,
    batch,
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

///|
#borrow(context, y, bias)
extern "c" fn metal_fill_bias(
  context : MetalContext,
  y : MetalBuffer,
  y_offset : Int64,
  bias : MetalBuffer,
  bias_offset : Int64,
  rows : Int,
  cols : Int,
) -> Unit = "moonbit_laya_metal_fill_bias"

///|
/// Broadcast a bias row across `[rows, cols]`, so a following `gemm` with
/// `beta = 1` adds it. MPS has no bias term of its own.
pub fn MetalContext::fill_bias(
  self : MetalContext,
  y : MetalBuffer,
  bias : MetalBuffer,
  rows~ : Int,
  cols~ : Int,
  y_offset? : Int = 0,
  bias_offset~ : Int64,
) -> Unit {
  metal_fill_bias(self, y, y_offset.to_int64(), bias, bias_offset, rows, cols)
}

///|
#borrow(context, x, y)
extern "c" fn metal_geglu(
  context : MetalContext,
  x : MetalBuffer,
  x_offset : Int64,
  y : MetalBuffer,
  y_offset : Int64,
  rows : Int,
  inner : Int,
) -> Unit = "moonbit_laya_metal_geglu"

///|
/// GeGLU over a `[rows, 2 · inner]` buffer.
pub fn MetalContext::geglu(
  self : MetalContext,
  x : MetalBuffer,
  y : MetalBuffer,
  rows~ : Int,
  inner~ : Int,
) -> Unit {
  metal_geglu(self, x, 0L, y, 0L, rows, inner)
}

///|
#borrow(context, x)
extern "c" fn metal_relu(
  context : MetalContext,
  x : MetalBuffer,
  offset : Int64,
  count : Int,
) -> Unit = "moonbit_laya_metal_relu"

///|
/// ReLU in place.
pub fn MetalContext::relu(
  self : MetalContext,
  x : MetalBuffer,
  count~ : Int,
) -> Unit {
  metal_relu(self, x, 0L, count)
}

///|
#borrow(context, x)
extern "c" fn metal_gelu(
  context : MetalContext,
  x : MetalBuffer,
  offset : Int64,
  count : Int,
) -> Unit = "moonbit_laya_metal_gelu"

///|
/// Exact GELU in place.
pub fn MetalContext::gelu(
  self : MetalContext,
  x : MetalBuffer,
  count~ : Int,
) -> Unit {
  metal_gelu(self, x, 0L, count)
}

///|
#borrow(context, a, b)
extern "c" fn metal_add(
  context : MetalContext,
  a : MetalBuffer,
  a_offset : Int64,
  b : MetalBuffer,
  b_offset : Int64,
  count : Int,
) -> Unit = "moonbit_laya_metal_add"

///|
/// `a += b` over `count` elements.
pub fn MetalContext::add(
  self : MetalContext,
  a : MetalBuffer,
  b : MetalBuffer,
  count~ : Int,
) -> Unit {
  metal_add(self, a, 0L, b, 0L, count)
}

///|
#borrow(context, x, v)
extern "c" fn metal_add_broadcast(
  context : MetalContext,
  x : MetalBuffer,
  x_offset : Int64,
  v : MetalBuffer,
  v_offset : Int64,
  rows : Int,
  cols : Int,
) -> Unit = "moonbit_laya_metal_add_broadcast"

///|
/// Add one row vector to every row of `[rows, cols]`.
pub fn MetalContext::add_broadcast(
  self : MetalContext,
  x : MetalBuffer,
  v : MetalBuffer,
  rows~ : Int,
  cols~ : Int,
  v_offset~ : Int64,
) -> Unit {
  metal_add_broadcast(self, x, 0L, v, v_offset, rows, cols)
}

///|
#borrow(context, x)
extern "c" fn metal_rope(
  context : MetalContext,
  x : MetalBuffer,
  offset : Int64,
  heads : Int,
  len : Int,
  dim : Int,
  base : Float,
) -> Unit = "moonbit_laya_metal_rope"

///|
/// Rotary embedding in place over a packed `[heads, len, dim]` buffer.
pub fn MetalContext::rope(
  self : MetalContext,
  x : MetalBuffer,
  heads~ : Int,
  len~ : Int,
  dim~ : Int,
  base~ : Double,
) -> Unit {
  metal_rope(self, x, 0L, heads, len, dim, Float::from_double(base))
}

///|
#borrow(context, qkv, q, k, v)
extern "c" fn metal_split_qkv(
  context : MetalContext,
  qkv : MetalBuffer,
  q : MetalBuffer,
  k : MetalBuffer,
  v : MetalBuffer,
  heads : Int,
  len : Int,
  dim : Int,
) -> Unit = "moonbit_laya_metal_split_qkv"

///|
/// Unpack `[len, 3, heads, dim]` into three `[heads, len, dim]` buffers.
pub fn MetalContext::split_qkv(
  self : MetalContext,
  qkv : MetalBuffer,
  q : MetalBuffer,
  k : MetalBuffer,
  v : MetalBuffer,
  heads~ : Int,
  len~ : Int,
  dim~ : Int,
) -> Unit {
  metal_split_qkv(self, qkv, q, k, v, heads, len, dim)
}

///|
#borrow(context, context_buffer, packed)
extern "c" fn metal_unpack_heads(
  context : MetalContext,
  context_buffer : MetalBuffer,
  packed : MetalBuffer,
  heads : Int,
  len : Int,
  dim : Int,
) -> Unit = "moonbit_laya_metal_unpack_heads"

///|
/// Fold `[heads, len, dim]` back into `[len, heads · dim]`.
pub fn MetalContext::unpack_heads(
  self : MetalContext,
  context_buffer : MetalBuffer,
  packed : MetalBuffer,
  heads~ : Int,
  len~ : Int,
  dim~ : Int,
) -> Unit {
  metal_unpack_heads(self, context_buffer, packed, heads, len, dim)
}

///|
#borrow(context, scores)
extern "c" fn metal_softmax(
  context : MetalContext,
  scores : MetalBuffer,
  rows : Int,
  len : Int,
  window : Int,
) -> Unit = "moonbit_laya_metal_softmax"

///|
/// Row-wise softmax over attention scores, applying the sliding-window mask in
/// the same pass. `window < 0` means unrestricted attention.
pub fn MetalContext::softmax(
  self : MetalContext,
  scores : MetalBuffer,
  rows~ : Int,
  len~ : Int,
  window~ : Int,
) -> Unit {
  metal_softmax(self, scores, rows, len, window)
}

///|
#borrow(context, src, indices, dst)
extern "c" fn metal_gather_rows(
  context : MetalContext,
  src : MetalBuffer,
  indices : MetalBuffer,
  dst : MetalBuffer,
  rows : Int,
  cols : Int,
) -> Unit = "moonbit_laya_metal_gather_rows"

///|
/// Gather whole rows of `src` by index into `dst`.
pub fn MetalContext::gather_rows(
  self : MetalContext,
  src : MetalBuffer,
  indices : MetalBuffer,
  dst : MetalBuffer,
  rows~ : Int,
  cols~ : Int,
) -> Unit {
  metal_gather_rows(self, src, indices, dst, rows, cols)
}
