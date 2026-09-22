/* Metal backend for laya.mbt.
 *
 * MoonBit drives this the same way it drives the CPU path: one call per
 * operation, in the same order, so the two backends stay structurally
 * identical and a numerical divergence localises to one kernel. The difference
 * is that these calls only *encode* work — nothing executes until
 * `moonbit_laya_metal_commit`, so a whole forward pass is one command buffer
 * and one synchronisation.
 *
 * Matrix multiplication goes to MPSMatrixMultiplication; everything else is in
 * metal/laya_kernels.metal.
 *
 * Reference counting is manual. `moon.pkg` cannot pass `-fobjc-arc` to a
 * native stub — that flag lives under a `link` block, and setting `link` on a
 * library package makes moon try to link it as an executable — so every entry
 * point below wraps its body in `@autoreleasepool` and the handful of
 * long-lived objects are held at +1 from their `new`/`alloc` constructor.
 */

#include <stdint.h>
#include <string.h>

#include <moonbit.h>

#include "laya_internal.h"

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "laya_kernels.generated.h"

/* Kernels, in the order `laya_pipeline_names` lists them. */
enum {
  LAYA_PIPELINE_LAYER_NORM = 0,
  LAYA_PIPELINE_FILL_BIAS,
  LAYA_PIPELINE_ZERO,
  LAYA_PIPELINE_GEGLU,
  LAYA_PIPELINE_RELU,
  LAYA_PIPELINE_GELU,
  LAYA_PIPELINE_ADD,
  LAYA_PIPELINE_ADD_BROADCAST,
  LAYA_PIPELINE_ROPE,
  LAYA_PIPELINE_SPLIT_QKV,
  LAYA_PIPELINE_UNPACK_HEADS,
  LAYA_PIPELINE_SOFTMAX,
  LAYA_PIPELINE_GATHER_ROWS,
  LAYA_PIPELINE_COUNT
};

static const char *laya_pipeline_names[LAYA_PIPELINE_COUNT] = {
    "laya_layer_norm", "laya_fill_bias",    "laya_zero",
    "laya_geglu",      "laya_relu",         "laya_gelu_inplace",
    "laya_add",        "laya_add_broadcast", "laya_rope",
    "laya_split_qkv",  "laya_unpack_heads", "laya_softmax",
    "laya_gather_rows"};

/* Threadgroup width for the reduction kernels; must match LAYA_THREADS in the
 * shader, where it sizes the threadgroup scratch array. */
#define LAYA_REDUCE_THREADS 256

/* A forward pass issues one matrix multiply per projection per layer, but only
 * a handful of distinct shapes. Building an MPSMatrixMultiplication is far
 * from free — it resolves and configures a kernel — so they are cached by
 * shape. A ModernBERT pass needs about ten entries. */
#define LAYA_MPS_CACHE 64

typedef struct {
  int32_t m;
  int32_t n;
  int32_t k;
  int32_t trans_b;
  float alpha;
  float beta;
  MPSMatrixMultiplication *kernel;
} laya_mps_entry;

typedef struct {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  id<MTLComputePipelineState> pipelines[LAYA_PIPELINE_COUNT];
  id<MTLCommandBuffer> command_buffer;
  id<MTLComputeCommandEncoder> encoder;
  laya_mps_entry mps_cache[LAYA_MPS_CACHE];
  int32_t mps_count;
  char error[512];
} laya_metal;

typedef struct {
  id<MTLBuffer> buffer;
  /* Non-NULL when this buffer wraps a MoonBit-owned arena, which must then be
   * kept alive for as long as the GPU can address it. */
  void *retained_arena;
} laya_metal_buffer;

/* ------------------------------------------------------------------ context */

static void laya_metal_finalize(void *object) {
  laya_metal *ctx = (laya_metal *)object;
  [ctx->encoder release];
  ctx->encoder = nil;
  [ctx->command_buffer release];
  ctx->command_buffer = nil;
  for (int i = 0; i < LAYA_PIPELINE_COUNT; i++) {
    [ctx->pipelines[i] release];
    ctx->pipelines[i] = nil;
  }
  for (int32_t i = 0; i < ctx->mps_count; i++) {
    [ctx->mps_cache[i].kernel release];
    ctx->mps_cache[i].kernel = nil;
  }
  ctx->mps_count = 0;
  [ctx->library release];
  ctx->library = nil;
  [ctx->queue release];
  ctx->queue = nil;
  [ctx->device release];
  ctx->device = nil;
}

static void laya_metal_fail(laya_metal *ctx, const char *message) {
  snprintf(ctx->error, sizeof(ctx->error), "%s", message);
}

MOONBIT_FFI_EXPORT
void *moonbit_laya_metal_context_new(void) {
  laya_metal *ctx = (laya_metal *)moonbit_make_external_object(
      laya_metal_finalize, sizeof(laya_metal));
  memset(ctx, 0, sizeof(laya_metal));

  @autoreleasepool {
    ctx->device = MTLCreateSystemDefaultDevice();
    if (ctx->device == nil) {
      laya_metal_fail(ctx, "no Metal device is available");
      return ctx;
    }
    ctx->queue = [ctx->device newCommandQueue];
    if (ctx->queue == nil) {
      laya_metal_fail(ctx, "could not create a Metal command queue");
      return ctx;
    }
    NSError *error = nil;
    NSString *source = [[NSString alloc] initWithUTF8String:LAYA_KERNEL_SOURCE];
    MTLCompileOptions *options = [MTLCompileOptions new];
    ctx->library = [ctx->device newLibraryWithSource:source
                                             options:options
                                               error:&error];
    [source release];
    [options release];
    if (ctx->library == nil) {
      snprintf(ctx->error, sizeof(ctx->error), "kernel compilation failed: %s",
               error != nil ? [[error localizedDescription] UTF8String]
                            : "unknown error");
      return ctx;
    }
    for (int i = 0; i < LAYA_PIPELINE_COUNT; i++) {
      NSString *name =
          [[NSString alloc] initWithUTF8String:laya_pipeline_names[i]];
      id<MTLFunction> function = [ctx->library newFunctionWithName:name];
      [name release];
      if (function == nil) {
        snprintf(ctx->error, sizeof(ctx->error), "kernel %s is missing",
                 laya_pipeline_names[i]);
        return ctx;
      }
      ctx->pipelines[i] =
          [ctx->device newComputePipelineStateWithFunction:function
                                                     error:&error];
      [function release];
      if (ctx->pipelines[i] == nil) {
        snprintf(ctx->error, sizeof(ctx->error),
                 "pipeline for %s failed: %s", laya_pipeline_names[i],
                 error != nil ? [[error localizedDescription] UTF8String]
                              : "unknown error");
        return ctx;
      }
    }
  }
  return ctx;
}

MOONBIT_FFI_EXPORT
int32_t moonbit_laya_metal_ready(void *object) {
  laya_metal *ctx = (laya_metal *)object;
  return ctx->error[0] == '\0' && ctx->device != nil ? 1 : 0;
}

MOONBIT_FFI_EXPORT
moonbit_bytes_t moonbit_laya_metal_error(void *object) {
  laya_metal *ctx = (laya_metal *)object;
  int32_t length = (int32_t)strlen(ctx->error);
  moonbit_bytes_t out = moonbit_make_bytes(length, 0);
  memcpy(out, ctx->error, (size_t)length);
  return out;
}

MOONBIT_FFI_EXPORT
moonbit_bytes_t moonbit_laya_metal_device_name(void *object) {
  laya_metal *ctx = (laya_metal *)object;
  const char *name = "";
  @autoreleasepool {
    if (ctx->device != nil) {
      name = [[ctx->device name] UTF8String];
    }
    int32_t length = (int32_t)strlen(name);
    moonbit_bytes_t out = moonbit_make_bytes(length, 0);
    memcpy(out, name, (size_t)length);
    return out;
  }
}

/* ------------------------------------------------------------------ buffers */

static void laya_metal_buffer_finalize(void *object) {
  laya_metal_buffer *wrapper = (laya_metal_buffer *)object;
  [wrapper->buffer release];
  wrapper->buffer = nil;
  if (wrapper->retained_arena != NULL) {
    moonbit_decref(wrapper->retained_arena);
    wrapper->retained_arena = NULL;
  }
}

MOONBIT_FFI_EXPORT
void *moonbit_laya_metal_buffer_new(void *context, int64_t count) {
  laya_metal *ctx = (laya_metal *)context;
  laya_metal_buffer *wrapper = (laya_metal_buffer *)moonbit_make_external_object(
      laya_metal_buffer_finalize, sizeof(laya_metal_buffer));
  wrapper->buffer = nil;
  wrapper->retained_arena = NULL;
  @autoreleasepool {
    if (ctx->device != nil && count > 0) {
      wrapper->buffer =
          [ctx->device newBufferWithLength:(NSUInteger)(count * 4)
                                   options:MTLResourceStorageModeShared];
    }
  }
  return wrapper;
}

/* Wrap the MoonBit-side weight arena without copying it.
 *
 * On Apple silicon the CPU and GPU share memory, so the float32 arena the CPU
 * path already holds can be addressed by the GPU directly. The arena is
 * increfed for the lifetime of the wrapper: the buffer must not outlive the
 * mapping it points into. */
MOONBIT_FFI_EXPORT
void *moonbit_laya_metal_buffer_from_arena(void *context, void *arena_object) {
  laya_metal *ctx = (laya_metal *)context;
  laya_arena *arena = (laya_arena *)arena_object;
  laya_metal_buffer *wrapper = (laya_metal_buffer *)moonbit_make_external_object(
      laya_metal_buffer_finalize, sizeof(laya_metal_buffer));
  wrapper->buffer = nil;
  wrapper->retained_arena = NULL;
  @autoreleasepool {
    if (ctx->device == nil || arena->data == NULL || arena->mapped_bytes <= 0) {
      return wrapper;
    }
    wrapper->buffer =
        [ctx->device newBufferWithBytesNoCopy:arena->data
                                       length:(NSUInteger)arena->mapped_bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
    if (wrapper->buffer != nil) {
      moonbit_incref(arena_object);
      wrapper->retained_arena = arena_object;
    }
  }
  return wrapper;
}

MOONBIT_FFI_EXPORT
int32_t moonbit_laya_metal_buffer_ok(void *object) {
  return ((laya_metal_buffer *)object)->buffer != nil ? 1 : 0;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_upload(void *object, int64_t offset, float *src,
                               int32_t count) {
  laya_metal_buffer *wrapper = (laya_metal_buffer *)object;
  if (wrapper->buffer == nil || count <= 0) {
    return;
  }
  float *base = (float *)[wrapper->buffer contents];
  memcpy(base + offset, src, (size_t)count * sizeof(float));
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_upload_ints(void *object, int64_t offset, int32_t *src,
                                    int32_t count) {
  laya_metal_buffer *wrapper = (laya_metal_buffer *)object;
  if (wrapper->buffer == nil || count <= 0) {
    return;
  }
  int32_t *base = (int32_t *)[wrapper->buffer contents];
  memcpy(base + offset, src, (size_t)count * sizeof(int32_t));
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_download(void *object, int64_t offset, float *dst,
                                 int32_t count) {
  laya_metal_buffer *wrapper = (laya_metal_buffer *)object;
  if (wrapper->buffer == nil || count <= 0) {
    return;
  }
  const float *base = (const float *)[wrapper->buffer contents];
  memcpy(dst, base + offset, (size_t)count * sizeof(float));
}

/* --------------------------------------------------------------- scheduling */

/* MPS brings its own encoder, so the running compute encoder has to be closed
 * before a matrix multiply and reopened after. Callers never see this. */
static void laya_metal_end_encoder(laya_metal *ctx) {
  if (ctx->encoder != nil) {
    [ctx->encoder endEncoding];
    [ctx->encoder release];
    ctx->encoder = nil;
  }
}

static id<MTLComputeCommandEncoder> laya_metal_encoder(laya_metal *ctx) {
  if (ctx->encoder == nil && ctx->command_buffer != nil) {
    ctx->encoder = [[ctx->command_buffer computeCommandEncoder] retain];
  }
  return ctx->encoder;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_begin(void *object) {
  laya_metal *ctx = (laya_metal *)object;
  @autoreleasepool {
    laya_metal_end_encoder(ctx);
    [ctx->command_buffer release];
    ctx->command_buffer = [[ctx->queue commandBuffer] retain];
  }
}

MOONBIT_FFI_EXPORT
int32_t moonbit_laya_metal_commit(void *object) {
  laya_metal *ctx = (laya_metal *)object;
  int32_t status = 0;
  @autoreleasepool {
    laya_metal_end_encoder(ctx);
    if (ctx->command_buffer == nil) {
      return -1;
    }
    [ctx->command_buffer commit];
    [ctx->command_buffer waitUntilCompleted];
    if ([ctx->command_buffer status] == MTLCommandBufferStatusError) {
      NSError *error = [ctx->command_buffer error];
      snprintf(ctx->error, sizeof(ctx->error), "command buffer failed: %s",
               error != nil ? [[error localizedDescription] UTF8String]
                            : "unknown error");
      status = -1;
    }
    [ctx->command_buffer release];
    ctx->command_buffer = nil;
  }
  return status;
}

/* ------------------------------------------------------------------ kernels */

static void laya_dispatch(laya_metal *ctx, int pipeline, NSUInteger threads,
                          NSUInteger group) {
  id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
  if (encoder == nil || threads == 0) {
    return;
  }
  [encoder setComputePipelineState:ctx->pipelines[pipeline]];
  // Apple GPUs support non-uniform threadgroups, so the grid is the exact
  // thread count and the kernels' bounds checks never fire.
  [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}

static id<MTLBuffer> laya_buf(void *object) {
  return ((laya_metal_buffer *)object)->buffer;
}

struct laya_norm_params {
  uint32_t cols;
  uint32_t has_beta;
  float eps;
};

struct laya_shape_params {
  uint32_t rows;
  uint32_t cols;
};

struct laya_count_params {
  uint32_t count;
};

struct laya_rope_params {
  uint32_t heads;
  uint32_t len;
  uint32_t dim;
  float base;
};

struct laya_head_params {
  uint32_t heads;
  uint32_t len;
  uint32_t dim;
};

struct laya_softmax_params {
  uint32_t len;
  int32_t window;
};

struct laya_gather_params {
  uint32_t cols;
};

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_layer_norm(void *context, void *x, int64_t x_offset,
                                   void *y, int64_t y_offset, void *gamma,
                                   int64_t gamma_offset, void *beta,
                                   int64_t beta_offset, int32_t rows,
                                   int32_t cols, float eps) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_norm_params params = {(uint32_t)cols, beta_offset >= 0 ? 1u : 0u,
                                      eps};
    [encoder setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_LAYER_NORM]];
    [encoder setBuffer:laya_buf(x) offset:(NSUInteger)(x_offset * 4) atIndex:0];
    [encoder setBuffer:laya_buf(y) offset:(NSUInteger)(y_offset * 4) atIndex:1];
    [encoder setBuffer:laya_buf(gamma)
                offset:(NSUInteger)(gamma_offset * 4)
               atIndex:2];
    // The shader always binds slot 3; with no bias it points at the gains,
    // which `has_beta` then tells it to ignore.
    [encoder setBuffer:laya_buf(beta != NULL ? beta : gamma)
                offset:(NSUInteger)((beta_offset >= 0 ? beta_offset
                                                      : gamma_offset) *
                                    4)
               atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_fill_bias(void *context, void *y, int64_t y_offset,
                                  void *bias, int64_t bias_offset, int32_t rows,
                                  int32_t cols) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_shape_params params = {(uint32_t)rows, (uint32_t)cols};
    [encoder setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_FILL_BIAS]];
    [encoder setBuffer:laya_buf(y) offset:(NSUInteger)(y_offset * 4) atIndex:0];
    [encoder setBuffer:laya_buf(bias)
                offset:(NSUInteger)(bias_offset * 4)
               atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)rows * cols, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_geglu(void *context, void *x, int64_t x_offset, void *y,
                              int64_t y_offset, int32_t rows, int32_t inner) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_shape_params params = {(uint32_t)rows, (uint32_t)inner};
    [encoder setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_GEGLU]];
    [encoder setBuffer:laya_buf(x) offset:(NSUInteger)(x_offset * 4) atIndex:0];
    [encoder setBuffer:laya_buf(y) offset:(NSUInteger)(y_offset * 4) atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)rows * inner, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

/* relu, gelu and zero share a shape: one buffer, one count. */
static void laya_metal_unary(laya_metal *ctx, int pipeline, void *x,
                             int64_t offset, int32_t count) {
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_count_params params = {(uint32_t)count};
    [encoder setComputePipelineState:ctx->pipelines[pipeline]];
    [encoder setBuffer:laya_buf(x) offset:(NSUInteger)(offset * 4) atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_relu(void *context, void *x, int64_t offset,
                             int32_t count) {
  laya_metal_unary((laya_metal *)context, LAYA_PIPELINE_RELU, x, offset, count);
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_gelu(void *context, void *x, int64_t offset,
                             int32_t count) {
  laya_metal_unary((laya_metal *)context, LAYA_PIPELINE_GELU, x, offset, count);
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_zero(void *context, void *x, int64_t offset,
                             int32_t count) {
  laya_metal_unary((laya_metal *)context, LAYA_PIPELINE_ZERO, x, offset, count);
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_add(void *context, void *a, int64_t a_offset, void *b,
                            int64_t b_offset, int32_t count) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_count_params params = {(uint32_t)count};
    [encoder setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_ADD]];
    [encoder setBuffer:laya_buf(a) offset:(NSUInteger)(a_offset * 4) atIndex:0];
    [encoder setBuffer:laya_buf(b) offset:(NSUInteger)(b_offset * 4) atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_add_broadcast(void *context, void *x, int64_t x_offset,
                                      void *v, int64_t v_offset, int32_t rows,
                                      int32_t cols) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_shape_params params = {(uint32_t)rows, (uint32_t)cols};
    [encoder
        setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_ADD_BROADCAST]];
    [encoder setBuffer:laya_buf(x) offset:(NSUInteger)(x_offset * 4) atIndex:0];
    [encoder setBuffer:laya_buf(v) offset:(NSUInteger)(v_offset * 4) atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)rows * cols, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_rope(void *context, void *x, int64_t offset,
                             int32_t heads, int32_t len, int32_t dim,
                             float base) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_rope_params params = {(uint32_t)heads, (uint32_t)len,
                                      (uint32_t)dim, base};
    [encoder setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_ROPE]];
    [encoder setBuffer:laya_buf(x) offset:(NSUInteger)(offset * 4) atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)heads * len * (dim / 2), 1,
                                         1)
        threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_split_qkv(void *context, void *qkv, void *q, void *k,
                                  void *v, int32_t heads, int32_t len,
                                  int32_t dim) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_head_params params = {(uint32_t)heads, (uint32_t)len,
                                      (uint32_t)dim};
    [encoder setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_SPLIT_QKV]];
    [encoder setBuffer:laya_buf(qkv) offset:0 atIndex:0];
    [encoder setBuffer:laya_buf(q) offset:0 atIndex:1];
    [encoder setBuffer:laya_buf(k) offset:0 atIndex:2];
    [encoder setBuffer:laya_buf(v) offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)len * heads * dim, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_unpack_heads(void *context, void *ctx_buffer,
                                     void *packed, int32_t heads, int32_t len,
                                     int32_t dim) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_head_params params = {(uint32_t)heads, (uint32_t)len,
                                      (uint32_t)dim};
    [encoder setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_UNPACK_HEADS]];
    [encoder setBuffer:laya_buf(ctx_buffer) offset:0 atIndex:0];
    [encoder setBuffer:laya_buf(packed) offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)len * heads * dim, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_softmax(void *context, void *scores, int32_t rows,
                                int32_t len, int32_t window) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_softmax_params params = {(uint32_t)len, window};
    [encoder setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_SOFTMAX]];
    [encoder setBuffer:laya_buf(scores) offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_gather_rows(void *context, void *src, void *indices,
                                    void *dst, int32_t rows, int32_t cols) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = laya_metal_encoder(ctx);
    if (encoder == nil) {
      return;
    }
    struct laya_gather_params params = {(uint32_t)cols};
    [encoder setComputePipelineState:ctx->pipelines[LAYA_PIPELINE_GATHER_ROWS]];
    [encoder setBuffer:laya_buf(src) offset:0 atIndex:0];
    [encoder setBuffer:laya_buf(indices) offset:0 atIndex:1];
    [encoder setBuffer:laya_buf(dst) offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)rows * cols, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LAYA_REDUCE_THREADS, 1, 1)];
  }
}

/* --------------------------------------------------------------------- MPS */

/* `c[batch][m, n] = alpha · a[batch][m, k] · op(b)[batch][k, n] + beta · c`.
 *
 * Batching covers attention, where each head is one matrix with a uniform
 * stride; `batch = 1` covers every Linear. */
/* Look up, or build and remember, the multiply for this shape. Returned
 * autoreleased-free: the cache owns it. */
static MPSMatrixMultiplication *laya_mps_kernel(laya_metal *ctx, int32_t m,
                                                int32_t n, int32_t k,
                                                int32_t trans_b, float alpha,
                                                float beta) {
  for (int32_t i = 0; i < ctx->mps_count; i++) {
    laya_mps_entry *entry = &ctx->mps_cache[i];
    if (entry->m == m && entry->n == n && entry->k == k &&
        entry->trans_b == trans_b && entry->alpha == alpha &&
        entry->beta == beta) {
      return entry->kernel;
    }
  }
  MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc]
      initWithDevice:ctx->device
       transposeLeft:NO
      transposeRight:trans_b ? YES : NO
          resultRows:(NSUInteger)m
       resultColumns:(NSUInteger)n
     interiorColumns:(NSUInteger)k
               alpha:(double)alpha
                beta:(double)beta];
  if (ctx->mps_count < LAYA_MPS_CACHE) {
    laya_mps_entry *entry = &ctx->mps_cache[ctx->mps_count++];
    entry->m = m;
    entry->n = n;
    entry->k = k;
    entry->trans_b = trans_b;
    entry->alpha = alpha;
    entry->beta = beta;
    entry->kernel = kernel;
    return kernel;
  }
  /* Cache full: hand back an autoreleased kernel so it is still freed. */
  return [kernel autorelease];
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_gemm(void *context, void *a, int64_t a_offset,
                             int32_t lda, int64_t a_stride, void *b,
                             int64_t b_offset, int32_t ldb, int64_t b_stride,
                             void *c, int64_t c_offset, int32_t ldc,
                             int64_t c_stride, int32_t batch, int32_t m,
                             int32_t n, int32_t k, int32_t trans_b, float alpha,
                             float beta) {
  laya_metal *ctx = (laya_metal *)context;
  @autoreleasepool {
    laya_metal_end_encoder(ctx);
    if (ctx->command_buffer == nil) {
      return;
    }
    MPSMatrixDescriptor *descriptor_a = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)m
                         columns:(NSUInteger)k
                        matrices:(NSUInteger)batch
                        rowBytes:(NSUInteger)lda * 4
                     matrixBytes:(NSUInteger)a_stride * 4
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descriptor_b = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)(trans_b ? n : k)
                         columns:(NSUInteger)(trans_b ? k : n)
                        matrices:(NSUInteger)batch
                        rowBytes:(NSUInteger)ldb * 4
                     matrixBytes:(NSUInteger)b_stride * 4
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descriptor_c = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)m
                         columns:(NSUInteger)n
                        matrices:(NSUInteger)batch
                        rowBytes:(NSUInteger)ldc * 4
                     matrixBytes:(NSUInteger)c_stride * 4
                        dataType:MPSDataTypeFloat32];
    MPSMatrix *matrix_a = [[MPSMatrix alloc] initWithBuffer:laya_buf(a)
                                                     offset:(NSUInteger)(a_offset * 4)
                                                 descriptor:descriptor_a];
    MPSMatrix *matrix_b = [[MPSMatrix alloc] initWithBuffer:laya_buf(b)
                                                     offset:(NSUInteger)(b_offset * 4)
                                                 descriptor:descriptor_b];
    MPSMatrix *matrix_c = [[MPSMatrix alloc] initWithBuffer:laya_buf(c)
                                                     offset:(NSUInteger)(c_offset * 4)
                                                 descriptor:descriptor_c];
    MPSMatrixMultiplication *multiply =
        laya_mps_kernel(ctx, m, n, k, trans_b, alpha, beta);
    multiply.batchStart = 0;
    multiply.batchSize = (NSUInteger)batch;
    [multiply encodeToCommandBuffer:ctx->command_buffer
                         leftMatrix:matrix_a
                        rightMatrix:matrix_b
                       resultMatrix:matrix_c];
    [matrix_a release];
    [matrix_b release];
    [matrix_c release];
  }
}

#else /* !__APPLE__ */

/* No Metal outside Apple platforms. The MoonBit side checks `ready` and falls
 * back to the CPU path, so these only have to be callable. */

MOONBIT_FFI_EXPORT
void *moonbit_laya_metal_context_new(void) {
  return moonbit_make_external_object(NULL, 1);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_laya_metal_ready(void *object) {
  (void)object;
  return 0;
}

MOONBIT_FFI_EXPORT
moonbit_bytes_t moonbit_laya_metal_error(void *object) {
  (void)object;
  static const char message[] = "Metal is only available on Apple platforms";
  int32_t length = (int32_t)(sizeof(message) - 1);
  moonbit_bytes_t out = moonbit_make_bytes(length, 0);
  memcpy(out, message, (size_t)length);
  return out;
}

MOONBIT_FFI_EXPORT
moonbit_bytes_t moonbit_laya_metal_device_name(void *object) {
  (void)object;
  return moonbit_make_bytes(0, 0);
}

MOONBIT_FFI_EXPORT
void *moonbit_laya_metal_buffer_new(void *context, int64_t count) {
  (void)context;
  (void)count;
  return moonbit_make_external_object(NULL, 1);
}

MOONBIT_FFI_EXPORT
void *moonbit_laya_metal_buffer_from_arena(void *context, void *arena_object) {
  (void)context;
  (void)arena_object;
  return moonbit_make_external_object(NULL, 1);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_laya_metal_buffer_ok(void *object) {
  (void)object;
  return 0;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_upload(void *o, int64_t f, float *s, int32_t c) {
  (void)o; (void)f; (void)s; (void)c;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_upload_ints(void *o, int64_t f, int32_t *s, int32_t c) {
  (void)o; (void)f; (void)s; (void)c;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_download(void *o, int64_t f, float *d, int32_t c) {
  (void)o; (void)f; (void)d; (void)c;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_begin(void *o) { (void)o; }

MOONBIT_FFI_EXPORT
int32_t moonbit_laya_metal_commit(void *o) {
  (void)o;
  return -1;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_layer_norm(void *a, void *b, int64_t c, void *d,
                                   int64_t e, void *f, int64_t g, void *h,
                                   int64_t i, int32_t j, int32_t k, float l) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f;
  (void)g; (void)h; (void)i; (void)j; (void)k; (void)l;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_fill_bias(void *a, void *b, int64_t c, void *d,
                                  int64_t e, int32_t f, int32_t g) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_geglu(void *a, void *b, int64_t c, void *d, int64_t e,
                              int32_t f, int32_t g) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_relu(void *a, void *b, int64_t c, int32_t d) {
  (void)a; (void)b; (void)c; (void)d;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_gelu(void *a, void *b, int64_t c, int32_t d) {
  (void)a; (void)b; (void)c; (void)d;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_zero(void *a, void *b, int64_t c, int32_t d) {
  (void)a; (void)b; (void)c; (void)d;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_add(void *a, void *b, int64_t c, void *d, int64_t e,
                            int32_t f) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_add_broadcast(void *a, void *b, int64_t c, void *d,
                                      int64_t e, int32_t f, int32_t g) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_rope(void *a, void *b, int64_t c, int32_t d, int32_t e,
                             int32_t f, float g) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_split_qkv(void *a, void *b, void *c, void *d, void *e,
                                  int32_t f, int32_t g, int32_t h) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_unpack_heads(void *a, void *b, void *c, int32_t d,
                                     int32_t e, int32_t f) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_softmax(void *a, void *b, int32_t c, int32_t d,
                                int32_t e) {
  (void)a; (void)b; (void)c; (void)d; (void)e;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_gather_rows(void *a, void *b, void *c, void *d,
                                    int32_t e, int32_t f) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f;
}

MOONBIT_FFI_EXPORT
void moonbit_laya_metal_gemm(void *a, void *b, int64_t c, int32_t d, int64_t e,
                             void *f, int64_t g, int32_t h, int64_t i, void *j,
                             int64_t k, int32_t l, int64_t m, int32_t n,
                             int32_t o, int32_t p, int32_t q, int32_t r,
                             float s, float t) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
  (void)h; (void)i; (void)j; (void)k; (void)l; (void)m; (void)n;
  (void)o; (void)p; (void)q; (void)r; (void)s; (void)t;
}

#endif /* __APPLE__ */
