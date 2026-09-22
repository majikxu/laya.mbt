// Compute kernels for the Metal backend of laya.mbt.
//
// These mirror the MoonBit kernels in `ops.mbt` one for one, so a divergence
// between the two backends is always localised to a single operation. Matrix
// multiplication is not here: that goes to MPSMatrixMultiplication.
//
// Every buffer is float32 and addressed by element offset, matching the
// float32 weight arena the CPU path shares.

#include <metal_stdlib>

using namespace metal;

// One threadgroup normalises one row, so this bounds the reduction width.
constant constexpr uint LAYA_THREADS = 256;

struct LayaNormParams {
  uint cols;
  uint has_beta;
  float eps;
};

struct LayaShapeParams {
  uint rows;
  uint cols;
};

struct LayaCountParams {
  uint count;
};

struct LayaRopeParams {
  uint heads;
  uint len;
  uint dim;
  float base;
};

struct LayaHeadParams {
  uint heads;
  uint len;
  uint dim;
};

struct LayaSoftmaxParams {
  uint len;
  // Inclusive distance a position may attend over; negative means no mask.
  int window;
};

struct LayaGatherParams {
  uint cols;
};

// The Metal shading language has no `erf`, and GELU here has to be the exact
// erf form — the encoder, the scorer and the action head all use it, and the
// tanh approximation is visibly different. This is Abramowitz & Stegun 7.1.26,
// whose maximum absolute error is 1.5e-7, an order of magnitude below what
// float32 resolves for the activations it is applied to. `laya_kernels_test`
// checks it against the CPU path's libm `erff` across the range that matters.
static inline float laya_erf(float x) {
  constexpr float a1 = 0.254829592f;
  constexpr float a2 = -0.284496736f;
  constexpr float a3 = 1.421413741f;
  constexpr float a4 = -1.453152027f;
  constexpr float a5 = 1.061405429f;
  constexpr float p = 0.3275911f;
  const float magnitude = fabs(x);
  const float t = 1.0f / (1.0f + p * magnitude);
  const float tail =
      (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
      exp(-magnitude * magnitude);
  return x < 0.0f ? tail - 1.0f : 1.0f - tail;
}

// Exact GELU, matching the `erff` form the encoder and the scorer use.
static inline float laya_gelu(float x) {
  return 0.5f * x * (1.0f + laya_erf(x * 0.70710678118654752f));
}

// Sum `values` across the threadgroup, leaving the total in slot 0.
static inline float laya_reduce_sum(
    threadgroup float *scratch, float value, uint tid) {
  scratch[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = LAYA_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return scratch[0];
}

kernel void laya_layer_norm(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    device const float *gamma [[buffer(2)]],
    device const float *beta [[buffer(3)]],
    constant LayaNormParams &p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[LAYA_THREADS];
  const uint base = row * p.cols;
  float partial = 0.0f;
  for (uint i = tid; i < p.cols; i += LAYA_THREADS) {
    partial += x[base + i];
  }
  const float mean = laya_reduce_sum(scratch, partial, tid) / float(p.cols);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float variance = 0.0f;
  for (uint i = tid; i < p.cols; i += LAYA_THREADS) {
    const float d = x[base + i] - mean;
    variance += d * d;
  }
  const float scale =
      rsqrt(laya_reduce_sum(scratch, variance, tid) / float(p.cols) + p.eps);
  for (uint i = tid; i < p.cols; i += LAYA_THREADS) {
    float value = (x[base + i] - mean) * scale * gamma[i];
    if (p.has_beta != 0) {
      value += beta[i];
    }
    y[base + i] = value;
  }
}

// Broadcast a bias row across every row of `y`.
//
// MPSMatrixMultiplication has no bias term, so a biased Linear is a fill
// followed by a multiply with beta = 1 — the same shape as the CPU path, where
// BLAS is handed a pre-filled C.
kernel void laya_fill_bias(
    device float *y [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant LayaShapeParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.rows * p.cols) {
    return;
  }
  y[gid] = bias[gid % p.cols];
}

kernel void laya_zero(
    device float *y [[buffer(0)]],
    constant LayaCountParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < p.count) {
    y[gid] = 0.0f;
  }
}

// GeGLU over a [rows, 2 * inner] buffer: the value and gate halves sit side by
// side in the last axis.
kernel void laya_geglu(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant LayaShapeParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint inner = p.cols;
  if (gid >= p.rows * inner) {
    return;
  }
  const uint row = gid / inner;
  const uint i = gid % inner;
  const uint src = row * inner * 2;
  y[gid] = laya_gelu(x[src + i]) * x[src + inner + i];
}

kernel void laya_relu(
    device float *x [[buffer(0)]],
    constant LayaCountParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < p.count) {
    x[gid] = fmax(x[gid], 0.0f);
  }
}

kernel void laya_gelu_inplace(
    device float *x [[buffer(0)]],
    constant LayaCountParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < p.count) {
    x[gid] = laya_gelu(x[gid]);
  }
}

kernel void laya_add(
    device float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant LayaCountParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < p.count) {
    a[gid] += b[gid];
  }
}

// Add one vector to every row, used for the question-type embedding.
kernel void laya_add_broadcast(
    device float *x [[buffer(0)]],
    device const float *v [[buffer(1)]],
    constant LayaShapeParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < p.rows * p.cols) {
    x[gid] += v[gid % p.cols];
  }
}

// Rotary embedding in the split-halves (NeoX / Hugging Face) layout, in place
// over a packed [heads, len, dim] buffer.
kernel void laya_rope(
    device float *x [[buffer(0)]],
    constant LayaRopeParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  const uint half_dim = p.dim / 2;
  if (gid >= p.heads * p.len * half_dim) {
    return;
  }
  const uint i = gid % half_dim;
  const uint position = (gid / half_dim) % p.len;
  const uint head = gid / (half_dim * p.len);
  const float theta =
      float(position) * pow(p.base, -2.0f * float(i) / float(p.dim));
  const float c = cos(theta);
  const float s = sin(theta);
  const uint row = (head * p.len + position) * p.dim;
  const float lo = x[row + i];
  const float hi = x[row + i + half_dim];
  x[row + i] = lo * c - hi * s;
  x[row + i + half_dim] = lo * s + hi * c;
}

// Unpack a [len, 3, heads, dim] projection into three [heads, len, dim]
// buffers, the layout the per-head multiplies want.
kernel void laya_split_qkv(
    device const float *qkv [[buffer(0)]],
    device float *q [[buffer(1)]],
    device float *k [[buffer(2)]],
    device float *v [[buffer(3)]],
    constant LayaHeadParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const uint width = p.heads * p.dim;
  if (gid >= p.len * width) {
    return;
  }
  const uint i = gid % p.dim;
  const uint head = (gid / p.dim) % p.heads;
  const uint position = gid / width;
  const uint src = position * 3 * width + head * p.dim + i;
  const uint dst = (head * p.len + position) * p.dim + i;
  q[dst] = qkv[src];
  k[dst] = qkv[src + width];
  v[dst] = qkv[src + 2 * width];
}

// Fold [heads, len, dim] back into [len, heads * dim].
kernel void laya_unpack_heads(
    device const float *context [[buffer(0)]],
    device float *packed [[buffer(1)]],
    constant LayaHeadParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint width = p.heads * p.dim;
  if (gid >= p.len * width) {
    return;
  }
  const uint i = gid % p.dim;
  const uint head = (gid / p.dim) % p.heads;
  const uint position = gid / width;
  packed[position * width + head * p.dim + i] =
      context[(head * p.len + position) * p.dim + i];
}

// Masked softmax over attention scores, one threadgroup per [heads * len] row.
//
// The sliding-window mask is applied here rather than in a separate pass, so
// masked positions never reach memory: a position may attend within `window`
// of itself, and `window < 0` means no restriction.
kernel void laya_softmax(
    device float *scores [[buffer(0)]],
    constant LayaSoftmaxParams &p [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[LAYA_THREADS];
  const uint base = row * p.len;
  const int position = int(row % p.len);
  float local_max = -INFINITY;
  for (uint j = tid; j < p.len; j += LAYA_THREADS) {
    if (p.window >= 0 && abs(position - int(j)) > p.window) {
      continue;
    }
    local_max = fmax(local_max, scores[base + j]);
  }
  scratch[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = LAYA_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmax(scratch[tid], scratch[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float row_max = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float partial = 0.0f;
  for (uint j = tid; j < p.len; j += LAYA_THREADS) {
    float value = 0.0f;
    if (p.window < 0 || abs(position - int(j)) <= p.window) {
      value = exp(scores[base + j] - row_max);
    }
    scores[base + j] = value;
    partial += value;
  }
  const float total = laya_reduce_sum(scratch, partial, tid);
  const float inv = 1.0f / total;
  for (uint j = tid; j < p.len; j += LAYA_THREADS) {
    scores[base + j] *= inv;
  }
}

// Gather whole rows by index, used to pull the option marker positions out of
// the decision head's output.
kernel void laya_gather_rows(
    device const float *src [[buffer(0)]],
    device const int *indices [[buffer(1)]],
    device float *dst [[buffer(2)]],
    constant LayaGatherParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint row = gid / p.cols;
  const uint i = gid % p.cols;
  dst[gid] = src[uint(indices[row]) * p.cols + i];
}
