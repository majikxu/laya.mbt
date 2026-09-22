///|
/// Rebuild the reference `1 + 2 + … + 6` GEMM by hand so a linkage failure or a
/// row/column-major mix-up in the BLAS wrapper shows up immediately.
test "gemm multiplies row-major matrices" {
  let a : FixedArray[Float] = [1, 2, 3, 4, 5, 6] // 2x3
  let b : FixedArray[Float] = [7, 8, 9, 10, 11, 12] // 3x2
  let c = @tensor.zeros(4)
  @tensor.gemm(a, b, c, m=2, n=2, k=3, lda=3, ldb=2, ldc=2)
  debug_inspect(
    c,
    content=(
      #|<FixedArray: [58, 64, 139, 154]>
    ),
  )
}

///|
test "gemm can transpose the right operand" {
  let a : FixedArray[Float] = [1, 2, 3, 4, 5, 6] // 2x3
  let b : FixedArray[Float] = [1, 0, 0, 0, 1, 0] // 2x3, used as (3x2)
  let c = @tensor.zeros(4)
  @tensor.gemm(a, b, c, m=2, n=2, k=3, lda=3, ldb=3, ldc=2, trans_b=true)
  debug_inspect(
    c,
    content=(
      #|<FixedArray: [1, 2, 4, 5]>
    ),
  )
}

///|
test "layer_norm normalises each row and applies the affine transform" {
  let x : FixedArray[Float] = [1, 2, 3, 4, 10, 20, 30, 40]
  let out = @tensor.zeros(8)
  let gamma : FixedArray[Float] = [1, 1, 1, 1]
  @tensor.layer_norm(x, out, rows=2, cols=4, gamma, None, eps=1.0e-5)
  // Both rows are affine images of each other, so they normalise identically.
  let row0 = [out[0], out[1], out[2], out[3]]
  let row1 = [out[4], out[5], out[6], out[7]]
  debug_inspect(
    row0,
    content=(
      #|[
      #|  -1.3416354656219482,
      #|  -0.4472118020057678,
      #|  0.4472118020057678,
      #|  1.3416354656219482,
      #|]
    ),
  )
  inspect((row0[0] - row1[0]).abs() < 1.0e-5, content="true")
}

///|
test "gelu matches the exact erf form at reference points" {
  inspect(@tensor.gelu(0.0), content="0")
  inspect(@tensor.gelu(1.0), content="0.8413447141647339")
  inspect(@tensor.gelu(-1.0), content="-0.15865525603294373")
  inspect(@tensor.gelu(2.0), content="1.9544997215270996")
}

///|
test "softmax_rows produces a distribution per row" {
  let x : FixedArray[Float] = [0, 0, 0, 0, 1000, 0]
  @tensor.softmax_rows(x, rows=2, cols=3)
  inspect(x[0] == x[1] && x[1] == x[2], content="true")
  inspect((x[0] - 1.0 / 3.0).abs() < 1.0e-6, content="true")
  // A large logit must not overflow: the row max is subtracted first.
  inspect(x[4], content="1")
}

///|
test "arena reports the capacity it was asked for" {
  let arena = @tensor.Arena::new(16L)
  inspect(arena.capacity(), content="16")
  inspect(arena.get(0L), content="0")
}
