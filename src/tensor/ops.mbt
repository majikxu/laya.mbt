///|
/// `erff` from libm. MoonBit's `@math` has no error function, and GELU in both
/// ModernBERT and the Laya heads is the exact `erf` form rather than the `tanh`
/// approximation, so an approximation here would show up as a real parity gap.
extern "c" fn erff(x : Float) -> Float = "erff"

///|
/// Allocate a zeroed activation buffer.
pub fn zeros(len : Int) -> FixedArray[Float] {
  FixedArray::make(len, 0.0)
}

///|
/// Layer normalisation over the last axis of a `[rows, cols]` buffer.
///
/// Means and variances accumulate in `Double` — at `cols = 1024` the float32
/// sums drift enough to be visible against the reference.
pub fn layer_norm(
  x : FixedArray[Float],
  out : FixedArray[Float],
  rows~ : Int,
  cols~ : Int,
  gamma : FixedArray[Float],
  beta : FixedArray[Float]?,
  eps~ : Float,
) -> Unit {
  let eps = eps.to_double()
  for row in 0..<rows {
    let base = row * cols
    let mut sum = 0.0
    for i in 0..<cols {
      sum += x[base + i].to_double()
    }
    let mean = sum / cols.to_double()
    let mut variance = 0.0
    for i in 0..<cols {
      let d = x[base + i].to_double() - mean
      variance += d * d
    }
    let scale = 1.0 / Double::sqrt(variance / cols.to_double() + eps)
    match beta {
      Some(beta) =>
        for i in 0..<cols {
          out[base + i] = Float::from_double(
              (x[base + i].to_double() - mean) * scale,
            ) *
            gamma[i] +
            beta[i]
        }
      None =>
        for i in 0..<cols {
          out[base + i] = Float::from_double(
              (x[base + i].to_double() - mean) * scale,
            ) *
            gamma[i]
        }
    }
  }
}

///|
/// Exact GELU: `0.5 · x · (1 + erf(x / √2))`.
pub fn gelu(x : Float) -> Float {
  0.5 * x * (1.0 + erff(x * Float::from_double(1.0 / Double::sqrt(2.0))))
}

///|
/// GeGLU over a `[rows, 2 · inner]` buffer produced by ModernBERT's `Wi`.
///
/// `Wi` emits the value and gate halves concatenated along the last axis;
/// `out[rows, inner] = gelu(value) · gate`.
pub fn geglu(
  x : FixedArray[Float],
  out : FixedArray[Float],
  rows~ : Int,
  inner~ : Int,
) -> Unit {
  for row in 0..<rows {
    let src = row * inner * 2
    let dst = row * inner
    for i in 0..<inner {
      out[dst + i] = gelu(x[src + i]) * x[src + inner + i]
    }
  }
}

///|
/// Apply exact GELU in place.
pub fn gelu_inplace(x : FixedArray[Float], len~ : Int) -> Unit {
  for i in 0..<len {
    x[i] = gelu(x[i])
  }
}

///|
/// Apply ReLU in place.
///
/// The decision head's feed-forward uses ReLU, not GELU: it is a
/// `torch.nn.TransformerEncoderLayer`, whose default activation is ReLU, while
/// the encoder and scorer around it are GELU.
pub fn relu_inplace(x : FixedArray[Float], len~ : Int) -> Unit {
  for i in 0..<len {
    if x[i] < 0.0 {
      x[i] = 0.0
    }
  }
}

///|
/// `a += b` over the first `len` elements.
pub fn add_inplace(
  a : FixedArray[Float],
  b : FixedArray[Float],
  len~ : Int,
) -> Unit {
  for i in 0..<len {
    a[i] = a[i] + b[i]
  }
}

///|
/// Rotary position embedding in the "split halves" (NeoX / Hugging Face)
/// layout, applied in place to a packed `[heads, len, dim]` buffer.
///
/// For position `p` and pair `i`, the angle is `p · base^(-2i/dim)`, and the
/// pair rotated is `(i, i + dim/2)` rather than `(2i, 2i+1)`.
pub fn rope_inplace(
  x : FixedArray[Float],
  heads~ : Int,
  len~ : Int,
  dim~ : Int,
  base~ : Double,
) -> Unit {
  let half = dim / 2
  let inv_freq = FixedArray::make(half, 0.0)
  for i in 0..<half {
    inv_freq[i] = @math.pow(base, -2.0 * i.to_double() / dim.to_double())
  }
  for position in 0..<len {
    let angle_base = position.to_double()
    for i in 0..<half {
      let theta = angle_base * inv_freq[i]
      let cos_theta = Float::from_double(@math.cos(theta))
      let sin_theta = Float::from_double(@math.sin(theta))
      for head in 0..<heads {
        let row = (head * len + position) * dim
        let lo = x[row + i]
        let hi = x[row + i + half]
        x[row + i] = lo * cos_theta - hi * sin_theta
        x[row + i + half] = lo * sin_theta + hi * cos_theta
      }
    }
  }
}

///|
/// In-place softmax over `[rows, cols]`, row by row, shifted by the row max.
pub fn softmax_rows(
  x : FixedArray[Float],
  rows~ : Int,
  cols~ : Int,
  offset? : Int = 0,
) -> Unit {
  for row in 0..<rows {
    let base = offset + row * cols
    let mut max = x[base]
    for i in 1..<cols {
      if x[base + i] > max {
        max = x[base + i]
      }
    }
    let mut total = 0.0
    for i in 0..<cols {
      let value = @math.exp((x[base + i] - max).to_double())
      x[base + i] = Float::from_double(value)
      total += value
    }
    let scale = Float::from_double(1.0 / total)
    for i in 0..<cols {
      x[base + i] = x[base + i] * scale
    }
  }
}

///|
/// Softmax of `logits` into a fresh `Array[Double]`.
///
/// The decision layer works in `Double` once the encoder is done: the calibrated
/// probabilities, the entropy-based confidence and the rounding that follows all
/// compare against float64 reference values.
pub fn softmax(logits : ArrayView[Double]) -> Array[Double] {
  let mut max = logits[0]
  for value in logits {
    if value > max {
      max = value
    }
  }
  let out = Array::make(logits.length(), 0.0)
  let mut total = 0.0
  for i, value in logits {
    let e = @math.exp(value - max)
    out[i] = e
    total += e
  }
  for i in 0..<out.length() {
    out[i] = out[i] / total
  }
  out
}
