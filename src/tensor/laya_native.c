/* Native support for laya.mbt: BLAS matmul, float16 decoding, and reading
 * tensors out of a safetensors checkpoint.
 *
 * The split of work between here and MoonBit is deliberate: this file owns only
 * what has to be native — the GEMM calls, the weight arena (too large to want in
 * the GC heap) and the checkpoint reader. Everything else (layer norm,
 * GELU, RoPE, softmax, attention orchestration) lives in MoonBit and operates on
 * FixedArray[Float], which the native backend represents as a plain float*. */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <moonbit.h>

#include "laya_internal.h"

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* ------------------------------------------------------------------ float16 */

#if defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE)
#define LAYA_NATIVE_FP16 1
#endif

static inline float laya_half_to_float(uint16_t h) {
#ifdef LAYA_NATIVE_FP16
  __fp16 v;
  memcpy(&v, &h, sizeof(v));
  return (float)v;
#else
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      /* Subnormal half: renormalise into a float32 normal. */
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp -= 1;
      }
      mant &= 0x3ffu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
#endif
}

static void laya_half_to_float_n(float *dst, const uint16_t *src, int64_t n) {
  for (int64_t i = 0; i < n; i++) {
    dst[i] = laya_half_to_float(src[i]);
  }
}

/* --------------------------------------------------------------- weight arena
 *
 * One flat float32 arena holds every parameter the model matmuls against.
 * MoonBit addresses it by element offset, so the whole model is described by a
 * table of (offset, shape) computed once from the safetensors header. */

static void laya_arena_finalize(void *object) {
  laya_arena *arena = (laya_arena *)object;
  if (arena->data != NULL) {
    munmap(arena->data, (size_t)arena->mapped_bytes);
    arena->data = NULL;
  }
  arena->len = 0;
  arena->mapped_bytes = 0;
}

MOONBIT_FFI_EXPORT
void *moonbit_laya_arena_new(int64_t len) {
  laya_arena *arena = (laya_arena *)moonbit_make_external_object(
      laya_arena_finalize, sizeof(laya_arena));
  arena->data = NULL;
  arena->len = 0;
  arena->mapped_bytes = 0;
  if (len <= 0) {
    return arena;
  }
  long page = sysconf(_SC_PAGESIZE);
  if (page <= 0) {
    page = 4096;
  }
  int64_t bytes = len * (int64_t)sizeof(float);
  bytes = ((bytes + page - 1) / page) * page;
  void *mapped = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
  if (mapped == MAP_FAILED) {
    return arena; /* len stays 0; MoonBit checks capacity() and raises */
  }
  arena->data = (float *)mapped;
  arena->len = len;
  arena->mapped_bytes = bytes;
  return arena;
}

/* Base address and mapped length, for the Metal backend's zero-copy wrap. */
MOONBIT_FFI_EXPORT
void *moonbit_laya_arena_data(void *object) {
  return ((laya_arena *)object)->data;
}

MOONBIT_FFI_EXPORT
int64_t moonbit_laya_arena_mapped_bytes(void *object) {
  return ((laya_arena *)object)->mapped_bytes;
}

MOONBIT_FFI_EXPORT
int64_t moonbit_laya_arena_len(void *object) {
  return ((laya_arena *)object)->len;
}

MOONBIT_FFI_EXPORT
float moonbit_laya_arena_get(void *object, int64_t index) {
  laya_arena *arena = (laya_arena *)object;
  if (index < 0 || index >= arena->len) {
    return 0.0f;
  }
  return arena->data[index];
}

/* -------------------------------------------------------- checkpoint reader
 *
 * Tensors are read with `pread` into a small staging buffer rather than mapped.
 * Mapping is the obvious choice for the 393 MiB token embedding, but it keeps
 * the whole checkpoint resident: everything touched while filling the arena
 * stays in the process's footprint, and Darwin does not honour
 * `madvise(MADV_DONTNEED)` on a private file mapping, so it cannot be given
 * back. Reading bounds peak memory to the arena plus this buffer, and the
 * per-row reads the embedding needs are a rounding error next to the GEMMs. */

#define LAYA_STAGE_BYTES (256 * 1024)

typedef struct {
  int fd;
  int64_t size;
  int64_t header_len;
} laya_reader;

static void laya_reader_finalize(void *object) {
  laya_reader *reader = (laya_reader *)object;
  if (reader->fd >= 0) {
    close(reader->fd);
    reader->fd = -1;
  }
}

/* Read exactly `length` bytes at `offset`, retrying short reads. */
static int laya_pread_exact(int fd, void *dst, int64_t length, int64_t offset) {
  uint8_t *out = (uint8_t *)dst;
  while (length > 0) {
    ssize_t got = pread(fd, out, (size_t)length, (off_t)offset);
    if (got <= 0) {
      if (got < 0 && errno == EINTR) {
        continue;
      }
      return -1;
    }
    out += got;
    offset += got;
    length -= got;
  }
  return 0;
}

/* Returns a reader whose header_len is negative on failure:
 *   -1 open failed, -2 fstat/size failed, -3 header read failed,
 *   -4 bad header length. */
MOONBIT_FFI_EXPORT
void *moonbit_laya_reader_open(moonbit_bytes_t path) {
  laya_reader *reader = (laya_reader *)moonbit_make_external_object(
      laya_reader_finalize, sizeof(laya_reader));
  reader->fd = -1;
  reader->size = 0;
  reader->header_len = -1;

  int fd = open((const char *)path, O_RDONLY);
  if (fd < 0) {
    return reader;
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 8) {
    reader->header_len = -2;
    close(fd);
    return reader;
  }
  uint64_t header_len;
  if (laya_pread_exact(fd, &header_len, 8, 0) != 0) {
    reader->header_len = -3;
    close(fd);
    return reader;
  }
  if (header_len == 0 || header_len > (uint64_t)st.st_size - 8) {
    reader->header_len = -4;
    close(fd);
    return reader;
  }
  reader->fd = fd;
  reader->size = (int64_t)st.st_size;
  reader->header_len = (int64_t)header_len;
  return reader;
}

MOONBIT_FFI_EXPORT
int64_t moonbit_laya_reader_header_len(void *object) {
  return ((laya_reader *)object)->header_len;
}

MOONBIT_FFI_EXPORT
int64_t moonbit_laya_reader_size(void *object) {
  return ((laya_reader *)object)->size;
}

/* Copy the JSON header out so MoonBit can parse it with @json. */
MOONBIT_FFI_EXPORT
moonbit_bytes_t moonbit_laya_reader_header(void *object) {
  laya_reader *reader = (laya_reader *)object;
  int64_t len = reader->header_len > 0 ? reader->header_len : 0;
  moonbit_bytes_t out = moonbit_make_bytes((int32_t)len, 0);
  if (len > 0 && laya_pread_exact(reader->fd, out, len, 8) != 0) {
    memset(out, 0, (size_t)len);
  }
  return out;
}

static int laya_reader_in_range(laya_reader *reader, int64_t offset,
                                int64_t bytes) {
  if (reader->fd < 0 || offset < 0 || bytes < 0) {
    return 0;
  }
  return (uint64_t)offset + (uint64_t)bytes <= (uint64_t)reader->size;
}

/* Decode `count` float16 values at byte offset `offset` into `dst`. */
static int laya_read_f16_into(laya_reader *reader, int64_t offset,
                              int64_t count, float *dst) {
  if (!laya_reader_in_range(reader, offset, count * 2)) {
    return -1;
  }
  uint16_t stage[LAYA_STAGE_BYTES / sizeof(uint16_t)];
  const int64_t per_chunk = (int64_t)(sizeof(stage) / sizeof(stage[0]));
  for (int64_t done = 0; done < count;) {
    int64_t chunk = count - done < per_chunk ? count - done : per_chunk;
    if (laya_pread_exact(reader->fd, stage, chunk * 2, offset + done * 2) != 0) {
      return -1;
    }
    laya_half_to_float_n(dst + done, stage, chunk);
    done += chunk;
  }
  return 0;
}

/* Decode `count` float16 values into the arena at element offset `dst`.
 * Returns 0 on success, -1 if out of range. */
MOONBIT_FFI_EXPORT
int32_t moonbit_laya_reader_read_f16(void *reader_object, int64_t offset,
                                     int64_t count, void *arena_object,
                                     int64_t dst) {
  laya_arena *arena = (laya_arena *)arena_object;
  if (dst < 0 || count < 0 || dst + count > arena->len) {
    return -1;
  }
  return laya_read_f16_into((laya_reader *)reader_object, offset, count,
                            arena->data + dst);
}

/* Same, for float32 payloads. */
MOONBIT_FFI_EXPORT
int32_t moonbit_laya_reader_read_f32(void *reader_object, int64_t offset,
                                     int64_t count, void *arena_object,
                                     int64_t dst) {
  laya_reader *reader = (laya_reader *)reader_object;
  laya_arena *arena = (laya_arena *)arena_object;
  if (dst < 0 || count < 0 || dst + count > arena->len) {
    return -1;
  }
  if (!laya_reader_in_range(reader, offset, count * 4)) {
    return -1;
  }
  return laya_pread_exact(reader->fd, arena->data + dst, count * 4, offset);
}

/* Decode float16 values straight into a MoonBit FixedArray[Float]. Small
 * parameters (norm gains, biases, the type embedding) are kept on the MoonBit
 * side so the elementwise kernels can touch them without crossing FFI. */
MOONBIT_FFI_EXPORT
int32_t moonbit_laya_reader_read_f16_array(void *reader_object, int64_t offset,
                                           int32_t count, float *out) {
  return laya_read_f16_into((laya_reader *)reader_object, offset, count, out);
}

/* Gather embedding rows. `rows` holds `n` row indices into a float16 matrix of
 * `vocab` rows and `dim` columns based at `offset`. */
MOONBIT_FFI_EXPORT
int32_t moonbit_laya_reader_gather_f16(void *reader_object, int64_t offset,
                                       int32_t *rows, int32_t n, int32_t dim,
                                       int32_t vocab, float *out) {
  laya_reader *reader = (laya_reader *)reader_object;
  if (n < 0 || dim <= 0 || vocab <= 0) {
    return -1;
  }
  if (!laya_reader_in_range(reader, offset, (int64_t)vocab * dim * 2)) {
    return -1;
  }
  for (int32_t i = 0; i < n; i++) {
    int32_t row = rows[i];
    if (row < 0 || row >= vocab) {
      return -1;
    }
    if (laya_read_f16_into(reader, offset + (int64_t)row * dim * 2, dim,
                           out + (int64_t)i * dim) != 0) {
      return -1;
    }
  }
  return 0;
}

/* --------------------------------------------------------------------- BLAS */

/* y[m,n] = x[m,k] * W[n,k]^T  (+ bias broadcast over rows)
 *
 * PyTorch/MLX store Linear weights as [out_features, in_features], so the
 * transpose is expressed to BLAS rather than materialised. `bias_offset < 0`
 * means no bias. */
MOONBIT_FFI_EXPORT
void moonbit_laya_linear(float *x, void *arena_object, int64_t weight_offset,
                         int64_t bias_offset, float *y, int32_t m, int32_t k,
                         int32_t n) {
  laya_arena *arena = (laya_arena *)arena_object;
  const float *w = arena->data + weight_offset;
  if (bias_offset >= 0) {
    const float *b = arena->data + bias_offset;
    for (int32_t i = 0; i < m; i++) {
      memcpy(y + (int64_t)i * n, b, (size_t)n * sizeof(float));
    }
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, x, k, w,
              k, bias_offset >= 0 ? 1.0f : 0.0f, y, n);
}

/* c[m,n] = a[m,k] * b (optionally transposed), with explicit strides so that
 * per-head slices of a packed [heads, len, dim] activation can be multiplied in
 * place. Used for QK^T and (softmax P)V. */
MOONBIT_FFI_EXPORT
void moonbit_laya_gemm(float *a, int32_t a_offset, int32_t lda, float *b,
                       int32_t b_offset, int32_t ldb, float *c,
                       int32_t c_offset, int32_t ldc, int32_t m, int32_t n,
                       int32_t k, int32_t trans_b, float alpha, float beta) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
              m, n, k, alpha, a + a_offset, lda, b + b_offset, ldb, beta,
              c + c_offset, ldc);
}
