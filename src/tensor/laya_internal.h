/* Shared between laya_native.c and laya_metal.m. */

#ifndef LAYA_INTERNAL_H
#define LAYA_INTERNAL_H

#include <stdint.h>

/* The float32 parameter arena.
 *
 * Allocated with `mmap` rather than `malloc` so the base address and length
 * are page-aligned, which is what `newBufferWithBytesNoCopy:` requires: the
 * Metal backend wraps this allocation directly instead of keeping a second
 * copy of the weights on the GPU side. */
typedef struct {
  float *data;
  int64_t len;
  /* Bytes actually mapped, rounded up to a page boundary. */
  int64_t mapped_bytes;
} laya_arena;

#endif /* LAYA_INTERNAL_H */
