///|
/// The Metal backend is optional: on a machine without a GPU these report a
/// skip rather than failing.
fn metal_context() -> @tensor.MetalContext? {
  match @tensor.MetalContext::create() {
    Ok(context) => Some(context)
    Err(reason) => {
      println("skipped: Metal unavailable (\{reason})")
      None
    }
  }
}

///|
/// Largest deviation between two float32 buffers, divided by the scale of the
/// values being compared. The shader's own `erf` and the GPU's different
/// summation order both show up here as a few parts in 1e7.
fn relative_diff(
  got : FixedArray[Float],
  want : FixedArray[Float],
  count : Int,
) -> Double {
  let mut worst = 0.0
  let mut scale = 1.0
  for i in 0..<count {
    let diff = (got[i] - want[i]).to_double().abs()
    if diff > worst {
      worst = diff
    }
    let magnitude = want[i].to_double().abs()
    if magnitude > scale {
      scale = magnitude
    }
  }
  worst / scale
}

///|
test "metal gemm agrees with the BLAS path" {
  guard metal_context() is Some(gpu) else { return }
  let a : FixedArray[Float] = [1, 2, 3, 4, 5, 6] // 2x3
  let b : FixedArray[Float] = [7, 8, 9, 10, 11, 12] // 3x2
  let buffer_a = gpu.buffer(6)
  let buffer_b = gpu.buffer(6)
  let buffer_c = gpu.buffer(4)
  buffer_a.upload(a, count=6)
  buffer_b.upload(b, count=6)
  gpu.begin()
  gpu.gemm(buffer_a, buffer_b, buffer_c, m=2, n=2, k=3, lda=3, ldb=2, ldc=2)
  gpu.commit()
  let out = @tensor.zeros(4)
  buffer_c.download(out, count=4)
  debug_inspect(
    out,
    content=(
      #|<FixedArray: [58, 64, 139, 154]>
    ),
  )
}

///|
test "metal layer_norm, geglu and softmax agree with the CPU kernels" {
  guard metal_context() is Some(gpu) else { return }
  let rows = 3
  let cols = 64
  let x = @tensor.zeros(rows * cols)
  let gamma = @tensor.zeros(cols)
  let beta = @tensor.zeros(cols)
  for i in 0..<(rows * cols) {
    x[i] = (Float::from_int(i * 37 % 101) - 50.0) / 7.0
  }
  for i in 0..<cols {
    gamma[i] = 1.0 + Float::from_int(i % 5) / 10.0
    beta[i] = Float::from_int(i % 3) / 4.0
  }

  // CPU reference.
  let want = @tensor.zeros(rows * cols)
  @tensor.layer_norm(x, want, rows~, cols~, gamma, Some(beta), eps=1.0e-5)

  // GPU: gains and bias live side by side in one buffer, as they do in the
  // weight arena.
  let params = gpu.buffer(cols * 2)
  params.upload(gamma, count=cols)
  params.upload(beta, count=cols, offset=cols)
  let input = gpu.buffer(rows * cols)
  let output = gpu.buffer(rows * cols)
  input.upload(x, count=rows * cols)
  gpu.begin()
  gpu.layer_norm(
    input,
    output,
    params,
    params,
    rows~,
    cols~,
    gamma_offset=0L,
    beta_offset=cols.to_int64(),
    eps=1.0e-5,
  )
  gpu.commit()
  let got = @tensor.zeros(rows * cols)
  output.download(got, count=rows * cols)
  inspect(relative_diff(got, want, rows * cols) < 1.0e-6, content="true")

  // GeGLU over the same data read as [rows, 2 * 32].
  let inner = cols / 2
  let want_gate = @tensor.zeros(rows * inner)
  @tensor.geglu(x, want_gate, rows~, inner~)
  let gated = gpu.buffer(rows * inner)
  gpu.begin()
  gpu.geglu(input, gated, rows~, inner~)
  gpu.commit()
  let got_gate = @tensor.zeros(rows * inner)
  gated.download(got_gate, count=rows * inner)
  inspect(
    relative_diff(got_gate, want_gate, rows * inner) < 1.0e-6,
    content="true",
  )

  // Softmax, both unmasked and with a sliding window.
  for window in [-1, 8] {
    let want_scores = @tensor.zeros(rows * cols)
    x.blit_to(want_scores, len=rows * cols)
    if window >= 0 {
      for row in 0..<rows {
        for j in 0..<cols {
          if (row - j).abs() > window {
            want_scores[row * cols + j] = -1.0e30
          }
        }
      }
    }
    @tensor.softmax_rows(want_scores, rows~, cols~)
    let scores = gpu.buffer(rows * cols)
    scores.upload(x, count=rows * cols)
    gpu.begin()
    gpu.softmax(scores, rows~, len=cols, window~)
    gpu.commit()
    let got_scores = @tensor.zeros(rows * cols)
    scores.download(got_scores, count=rows * cols)
    inspect(
      relative_diff(got_scores, want_scores, rows * cols) < 1.0e-6,
      content="true",
    )
  }
}

///|
/// The shader implements its own `erf`, so the two GELUs are compared directly
/// rather than only through whole layers.
test "metal gelu tracks the CPU erf form" {
  guard metal_context() is Some(gpu) else { return }
  let count = 4001
  let x = @tensor.zeros(count)
  for i in 0..<count {
    x[i] = Float::from_int(i - count / 2) / 200.0
  }
  let want = @tensor.zeros(count)
  x.blit_to(want, len=count)
  @tensor.gelu_inplace(want, len=count)
  let buffer = gpu.buffer(count)
  buffer.upload(x, count~)
  gpu.begin()
  gpu.gelu(buffer, count~)
  gpu.commit()
  let got = @tensor.zeros(count)
  buffer.download(got, count~)
  // Absolute, because GELU is exactly zero at the origin and the approximation
  // error does not scale with the output.
  let mut worst = 0.0
  for i in 0..<count {
    let diff = (got[i] - want[i]).to_double().abs()
    if diff > worst {
      worst = diff
    }
  }
  inspect(worst < 2.0e-6, content="true")
}
