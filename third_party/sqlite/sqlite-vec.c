#include "sqlite-vec.h"

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef SQLITE_VEC_OMIT_FS
#include <stdio.h>
#endif

#ifndef SQLITE_CORE
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1
#else
#include "sqlite3.h"
#endif

#ifndef UINT32_TYPE
#ifdef HAVE_UINT32_T
#define UINT32_TYPE uint32_t
#else
#define UINT32_TYPE unsigned int
#endif
#endif
#ifndef UINT16_TYPE
#ifdef HAVE_UINT16_T
#define UINT16_TYPE uint16_t
#else
#define UINT16_TYPE unsigned short int
#endif
#endif
#ifndef INT16_TYPE
#ifdef HAVE_INT16_T
#define INT16_TYPE int16_t
#else
#define INT16_TYPE short int
#endif
#endif
#ifndef UINT8_TYPE
#ifdef HAVE_UINT8_T
#define UINT8_TYPE uint8_t
#else
#define UINT8_TYPE unsigned char
#endif
#endif
#ifndef INT8_TYPE
#ifdef HAVE_INT8_T
#define INT8_TYPE int8_t
#else
#define INT8_TYPE signed char
#endif
#endif
#ifndef LONGDOUBLE_TYPE
#define LONGDOUBLE_TYPE long double
#endif

#ifndef _WIN32
#ifndef __EMSCRIPTEN__
#ifndef __COSMOPOLITAN__
#ifndef __wasi__
typedef u_int8_t uint8_t;
typedef u_int16_t uint16_t;
typedef u_int64_t uint64_t;
#endif
#endif
#endif
#endif

typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t i16;
typedef int32_t i32;
typedef sqlite3_int64 i64;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef size_t usize;

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(X) (void)(X)
#endif

// sqlite3_vtab_in() was added in SQLite version 3.38 (2022-02-22)
// https://www.sqlite.org/changes.html#version_3_38_0
#if SQLITE_VERSION_NUMBER >= 3038000
#define COMPILER_SUPPORTS_VTAB_IN 1
#endif

#ifndef SQLITE_SUBTYPE
#define SQLITE_SUBTYPE 0x000100000
#endif

#ifndef SQLITE_RESULT_SUBTYPE
#define SQLITE_RESULT_SUBTYPE 0x001000000
#endif

#ifndef SQLITE_INDEX_CONSTRAINT_LIMIT
#define SQLITE_INDEX_CONSTRAINT_LIMIT 73
#endif

#ifndef SQLITE_INDEX_CONSTRAINT_OFFSET
#define SQLITE_INDEX_CONSTRAINT_OFFSET 74
#endif

#define countof(x) (sizeof(x) / sizeof((x)[0]))
#define min(a, b) (((a) <= (b)) ? (a) : (b))

enum VectorElementType {
  // clang-format off
  SQLITE_VEC_ELEMENT_TYPE_FLOAT32 = 223 + 0,
  SQLITE_VEC_ELEMENT_TYPE_BIT     = 223 + 1,
  SQLITE_VEC_ELEMENT_TYPE_INT8    = 223 + 2,
  // clang-format on
};

#ifdef SQLITE_VEC_ENABLE_AVX
#include <immintrin.h>
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

static f32 l2_sqr_float_avx(const void *pVect1v, const void *pVect2v,
                            const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
  f32 PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const f32 *pEnd1 = pVect1 + (qty16 << 4);

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }

  _mm256_store_ps(TmpRes, sum);
  return sqrt(TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] +
              TmpRes[5] + TmpRes[6] + TmpRes[7]);
}
#endif

#ifdef SQLITE_VEC_ENABLE_NEON
#include <arm_neon.h>

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))

// thx https://github.com/nmslib/hnswlib/pull/299/files
static f32 l2_sqr_float_neon(const void *pVect1v, const void *pVect2v,
                             const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
  size_t qty16 = qty >> 4;

  const f32 *pEnd1 = pVect1 + (qty16 << 4);

  float32x4_t diff, v1, v2;
  float32x4_t sum0 = vdupq_n_f32(0);
  float32x4_t sum1 = vdupq_n_f32(0);
  float32x4_t sum2 = vdupq_n_f32(0);
  float32x4_t sum3 = vdupq_n_f32(0);

  while (pVect1 < pEnd1) {
    v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    diff = vsubq_f32(v1, v2);
    sum0 = vfmaq_f32(sum0, diff, diff);

    v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    diff = vsubq_f32(v1, v2);
    sum1 = vfmaq_f32(sum1, diff, diff);

    v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    diff = vsubq_f32(v1, v2);
    sum2 = vfmaq_f32(sum2, diff, diff);

    v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    diff = vsubq_f32(v1, v2);
    sum3 = vfmaq_f32(sum3, diff, diff);
  }

  f32 sum_scalar =
      vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
  const f32 *pEnd2 = pVect1 + (qty - (qty16 << 4));
  while (pVect1 < pEnd2) {
    f32 diff = *pVect1 - *pVect2;
    sum_scalar += diff * diff;
    pVect1++;
    pVect2++;
  }

  return sqrt(sum_scalar);
}

static f32 l2_sqr_int8_neon(const void *pVect1v, const void *pVect2v,
                            const void *qty_ptr) {
  i8 *pVect1 = (i8 *)pVect1v;
  i8 *pVect2 = (i8 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  const i8 *pEnd1 = pVect1 + qty;
  i32 sum_scalar = 0;

  while (pVect1 < pEnd1 - 7) {
    // loading 8 at a time
    int8x8_t v1 = vld1_s8(pVect1);
    int8x8_t v2 = vld1_s8(pVect2);
    pVect1 += 8;
    pVect2 += 8;

    // widen to protect against overflow
    int16x8_t v1_wide = vmovl_s8(v1);
    int16x8_t v2_wide = vmovl_s8(v2);

    int16x8_t diff = vsubq_s16(v1_wide, v2_wide);
    int16x8_t squared_diff = vmulq_s16(diff, diff);
    int32x4_t sum = vpaddlq_s16(squared_diff);

    sum_scalar += vgetq_lane_s32(sum, 0) + vgetq_lane_s32(sum, 1) +
                  vgetq_lane_s32(sum, 2) + vgetq_lane_s32(sum, 3);
  }

  // handle leftovers
  while (pVect1 < pEnd1) {
    i16 diff = (i16)*pVect1 - (i16)*pVect2;
    sum_scalar += diff * diff;
    pVect1++;
    pVect2++;
  }

  return sqrtf(sum_scalar);
}

static i32 l1_int8_neon(const void *pVect1v, const void *pVect2v,
                        const void *qty_ptr) {
  i8 *pVect1 = (i8 *)pVect1v;
  i8 *pVect2 = (i8 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  const int8_t *pEnd1 = pVect1 + qty;

  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);
  int32x4_t acc4 = vdupq_n_s32(0);

  while (pVect1 < pEnd1 - 63) {
    int8x16_t v1 = vld1q_s8(pVect1);
    int8x16_t v2 = vld1q_s8(pVect2);
    int8x16_t diff1 = vabdq_s8(v1, v2);
    acc1 = vaddq_s32(acc1, vpaddlq_u16(vpaddlq_u8(diff1)));

    v1 = vld1q_s8(pVect1 + 16);
    v2 = vld1q_s8(pVect2 + 16);
    int8x16_t diff2 = vabdq_s8(v1, v2);
    acc2 = vaddq_s32(acc2, vpaddlq_u16(vpaddlq_u8(diff2)));

    v1 = vld1q_s8(pVect1 + 32);
    v2 = vld1q_s8(pVect2 + 32);
    int8x16_t diff3 = vabdq_s8(v1, v2);
    acc3 = vaddq_s32(acc3, vpaddlq_u16(vpaddlq_u8(diff3)));

    v1 = vld1q_s8(pVect1 + 48);
    v2 = vld1q_s8(pVect2 + 48);
    int8x16_t diff4 = vabdq_s8(v1, v2);
    acc4 = vaddq_s32(acc4, vpaddlq_u16(vpaddlq_u8(diff4)));

    pVect1 += 64;
    pVect2 += 64;
  }

  while (pVect1 < pEnd1 - 15) {
    int8x16_t v1 = vld1q_s8(pVect1);
    int8x16_t v2 = vld1q_s8(pVect2);
    int8x16_t diff = vabdq_s8(v1, v2);
    acc1 = vaddq_s32(acc1, vpaddlq_u16(vpaddlq_u8(diff)));
    pVect1 += 16;
    pVect2 += 16;
  }

  int32x4_t acc = vaddq_s32(vaddq_s32(acc1, acc2), vaddq_s32(acc3, acc4));

  int32_t sum = 0;
  while (pVect1 < pEnd1) {
    int32_t diff = abs((int32_t)*pVect1 - (int32_t)*pVect2);
    sum += diff;
    pVect1++;
    pVect2++;
  }

  return vaddvq_s32(acc) + sum;
}

static double l1_f32_neon(const void *pVect1v, const void *pVect2v,
                          const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  const f32 *pEnd1 = pVect1 + qty;
  float64x2_t acc = vdupq_n_f64(0);

  while (pVect1 < pEnd1 - 3) {
    float32x4_t v1 = vld1q_f32(pVect1);
    float32x4_t v2 = vld1q_f32(pVect2);
    pVect1 += 4;
    pVect2 += 4;

    // f32x4 -> f64x2 pad for overflow
    float64x2_t low_diff = vabdq_f64(vcvt_f64_f32(vget_low_f32(v1)),
                                     vcvt_f64_f32(vget_low_f32(v2)));
    float64x2_t high_diff =
        vabdq_f64(vcvt_high_f64_f32(v1), vcvt_high_f64_f32(v2));

    acc = vaddq_f64(acc, vaddq_f64(low_diff, high_diff));
  }

  double sum = 0;
  while (pVect1 < pEnd1) {
    sum += fabs((double)*pVect1 - (double)*pVect2);
    pVect1++;
    pVect2++;
  }

  return vaddvq_f64(acc) + sum;
}
#endif

static f32 l2_sqr_float(const void *pVect1v, const void *pVect2v,
                        const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  f32 res = 0;
  for (size_t i = 0; i < qty; i++) {
    f32 t = *pVect1 - *pVect2;
    pVect1++;
    pVect2++;
    res += t * t;
  }
  return sqrt(res);
}

static f32 l2_sqr_int8(const void *pA, const void *pB, const void *pD) {
  i8 *a = (i8 *)pA;
  i8 *b = (i8 *)pB;
  size_t d = *((size_t *)pD);

  f32 res = 0;
  for (size_t i = 0; i < d; i++) {
    f32 t = *a - *b;
    a++;
    b++;
    res += t * t;
  }
  return sqrt(res);
}

static f32 distance_l2_sqr_float(const void *a, const void *b, const void *d) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)d) > 16) {
    return l2_sqr_float_neon(a, b, d);
  }
#endif
#ifdef SQLITE_VEC_ENABLE_AVX
  if (((*(const size_t *)d) % 16 == 0)) {
    return l2_sqr_float_avx(a, b, d);
  }
#endif
  return l2_sqr_float(a, b, d);
}

static f32 distance_l2_sqr_int8(const void *a, const void *b, const void *d) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)d) > 7) {
    return l2_sqr_int8_neon(a, b, d);
  }
#endif
  return l2_sqr_int8(a, b, d);
}

static i32 l1_int8(const void *pA, const void *pB, const void *pD) {
  i8 *a = (i8 *)pA;
  i8 *b = (i8 *)pB;
  size_t d = *((size_t *)pD);

  i32 res = 0;
  for (size_t i = 0; i < d; i++) {
    res += abs(*a - *b);
    a++;
    b++;
  }

  return res;
}

static i32 distance_l1_int8(const void *a, const void *b, const void *d) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)d) > 15) {
    return l1_int8_neon(a, b, d);
  }
#endif
  return l1_int8(a, b, d);
}

static double l1_f32(const void *pA, const void *pB, const void *pD) {
  f32 *a = (f32 *)pA;
  f32 *b = (f32 *)pB;
  size_t d = *((size_t *)pD);

  double res = 0;
  for (size_t i = 0; i < d; i++) {
    res += fabs((double)*a - (double)*b);
    a++;
    b++;
  }

  return res;
}

static double distance_l1_f32(const void *a, const void *b, const void *d) {
#ifdef SQLITE_VEC_ENABLE_NEON
  if ((*(const size_t *)d) > 3) {
    return l1_f32_neon(a, b, d);
  }
#endif
  return l1_f32(a, b, d);
}

static f32 distance_cosine_float(const void *pVect1v, const void *pVect2v,
                                 const void *qty_ptr) {
  f32 *pVect1 = (f32 *)pVect1v;
  f32 *pVect2 = (f32 *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  f32 dot = 0;
  f32 aMag = 0;
  f32 bMag = 0;
  for (size_t i = 0; i < qty; i++) {
    dot += *pVect1 * *pVect2;
    aMag += *pVect1 * *pVect1;
    bMag += *pVect2 * *pVect2;
    pVect1++;
    pVect2++;
  }
  return 1 - (dot / (sqrt(aMag) * sqrt(bMag)));
}
static f32 distance_cosine_int8(const void *pA, const void *pB,
                                const void *pD) {
  i8 *a = (i8 *)pA;
  i8 *b = (i8 *)pB;
  size_t d = *((size_t *)pD);

  f32 dot = 0;
  f32 aMag = 0;
  f32 bMag = 0;
  for (size_t i = 0; i < d; i++) {
    dot += *a * *b;
    aMag += *a * *a;
    bMag += *b * *b;
    a++;
    b++;
  }
  return 1 - (dot / (sqrt(aMag) * sqrt(bMag)));
}

// https://github.com/facebookresearch/faiss/blob/77e2e79cd0a680adc343b9840dd865da724c579e/faiss/utils/hamming_distance/common.h#L34
static u8 hamdist_table[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4,
    2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4,
    2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5,
    3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

static f32 distance_hamming_u8(u8 *a, u8 *b, size_t n) {
  int same = 0;
  for (unsigned long i = 0; i < n; i++) {
    same += hamdist_table[a[i] ^ b[i]];
  }
  return (f32)same;
}

#ifdef _MSC_VER
#if !defined(__clang__) && (defined(_M_ARM) || defined(_M_ARM64))
// From
// https://github.com/ngtcp2/ngtcp2/blob/b64f1e77b5e0d880b93d31f474147fae4a1d17cc/lib/ngtcp2_ringbuf.c,
// line 34-43
static unsigned int __builtin_popcountl(unsigned int x) {
  unsigned int c = 0;
  for (; x; ++c) {
    x &= x - 1;
  }
  return c;
}
#else
#include <intrin.h>
#define __builtin_popcountl __popcnt64
#endif
#endif

static f32 distance_hamming_u64(u64 *a, u64 *b, size_t n) {
  int same = 0;
  for (unsigned long i = 0; i < n; i++) {
    same += __builtin_popcountl(a[i] ^ b[i]);
  }
  return (f32)same;
}

/**
 * @brief Calculate the hamming distance between two bitvectors.
 *
 * @param a - first bitvector, MUST have d dimensions
 * @param b - second bitvector, MUST have d dimensions
 * @param d - pointer to size_t, MUST be divisible by CHAR_BIT
 * @return f32
 */
static f32 distance_hamming(const void *a, const void *b, const void *d) {
  size_t dimensions = *((size_t *)d);

  if ((dimensions % 64) == 0) {
    return distance_hamming_u64((u64 *)a, (u64 *)b, dimensions / 8 / CHAR_BIT);
  }
  return distance_hamming_u8((u8 *)a, (u8 *)b, dimensions / CHAR_BIT);
}

// from SQLite source:
// https://github.com/sqlite/sqlite/blob/a509a90958ddb234d1785ed7801880ccb18b497e/src/json.c#L153
static const char vecJsonIsSpaceX[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

#define vecJsonIsspace(x) (vecJsonIsSpaceX[(unsigned char)x])

typedef void (*vector_cleanup)(void *p);

void vector_cleanup_noop(void *_) { UNUSED_PARAMETER(_); }

#define JSON_SUBTYPE 74

void vtab_set_error(sqlite3_vtab *pVTab, const char *zFormat, ...) {
  va_list args;
  sqlite3_free(pVTab->zErrMsg);
  va_start(args, zFormat);
  pVTab->zErrMsg = sqlite3_vmprintf(zFormat, args);
  va_end(args);
}
struct Array {
  size_t element_size;
  size_t length;
  size_t capacity;
  void *z;
};

/**
 * @brief Initial an array with the given element size and capacity.
 *
 * @param array
 * @param element_size
 * @param init_capacity
 * @return SQLITE_OK on success, error code on failure. Only error is
 * SQLITE_NOMEM
 */
int array_init(struct Array *array, size_t element_size, size_t init_capacity) {
  int sz = element_size * init_capacity;
  void *z = sqlite3_malloc(sz);
  if (!z) {
    return SQLITE_NOMEM;
  }
  memset(z, 0, sz);

  array->element_size = element_size;
  array->length = 0;
  array->capacity = init_capacity;
  array->z = z;
  return SQLITE_OK;
}

int array_append(struct Array *array, const void *element) {
  if (array->length == array->capacity) {
    size_t new_capacity = array->capacity * 2 + 100;
    void *z = sqlite3_realloc64(array->z, array->element_size * new_capacity);
    if (z) {
      array->capacity = new_capacity;
      array->z = z;
    } else {
      return SQLITE_NOMEM;
    }
  }
  memcpy(&((unsigned char *)array->z)[array->length * array->element_size],
         element, array->element_size);
  array->length++;
  return SQLITE_OK;
}

void array_cleanup(struct Array *array) {
  if (!array)
    return;
  array->element_size = 0;
  array->length = 0;
  array->capacity = 0;
  sqlite3_free(array->z);
  array->z = NULL;
}

char *vector_subtype_name(int subtype) {
  switch (subtype) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
    return "float32";
  case SQLITE_VEC_ELEMENT_TYPE_INT8:
    return "int8";
  case SQLITE_VEC_ELEMENT_TYPE_BIT:
    return "bit";
  }
  return "";
}
char *type_name(int type) {
  switch (type) {
  case SQLITE_INTEGER:
    return "INTEGER";
  case SQLITE_BLOB:
    return "BLOB";
  case SQLITE_TEXT:
    return "TEXT";
  case SQLITE_FLOAT:
    return "FLOAT";
  case SQLITE_NULL:
    return "NULL";
  }
  return "";
}

typedef void (*fvec_cleanup)(f32 *vector);

void fvec_cleanup_noop(f32 *_) { UNUSED_PARAMETER(_); }

static int fvec_from_value(sqlite3_value *value, f32 **vector,
                           size_t *dimensions, fvec_cleanup *cleanup,
                           char **pzErr) {
  int value_type = sqlite3_value_type(value);

  if (value_type == SQLITE_BLOB) {
    const void *blob = sqlite3_value_blob(value);
    int bytes = sqlite3_value_bytes(value);
    if (bytes == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }
    if ((bytes % sizeof(f32)) != 0) {
      *pzErr = sqlite3_mprintf("invalid float32 vector BLOB length. Must be "
                               "divisible by %d, found %d",
                               sizeof(f32), bytes);
      return SQLITE_ERROR;
    }
    *vector = (f32 *)blob;
    *dimensions = bytes / sizeof(f32);
    *cleanup = fvec_cleanup_noop;
    return SQLITE_OK;
  }

  if (value_type == SQLITE_TEXT) {
    const char *source = (const char *)sqlite3_value_text(value);
    int source_len = sqlite3_value_bytes(value);
    if (source_len == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }
    int i = 0;

    struct Array x;
    int rc = array_init(&x, sizeof(f32), ceil(source_len / 2.0));
    if (rc != SQLITE_OK) {
      return rc;
    }

    // advance leading whitespace to first '['
    while (i < source_len) {
      if (vecJsonIsspace(source[i])) {
        i++;
        continue;
      }
      if (source[i] == '[') {
        break;
      }

      *pzErr = sqlite3_mprintf(
          "JSON array parsing error: Input does not start with '['");
      array_cleanup(&x);
      return SQLITE_ERROR;
    }
    if (source[i] != '[') {
      *pzErr = sqlite3_mprintf(
          "JSON array parsing error: Input does not start with '['");
      array_cleanup(&x);
      return SQLITE_ERROR;
    }
    int offset = i + 1;

    while (offset < source_len) {
      char *ptr = (char *)&source[offset];
      char *endptr;

      errno = 0;
      double result = strtod(ptr, &endptr);
      if ((errno != 0 && result == 0) // some interval error?
          || (errno == ERANGE &&
              (result == HUGE_VAL || result == -HUGE_VAL)) // too big / smalls
      ) {
        sqlite3_free(x.z);
        *pzErr = sqlite3_mprintf("JSON parsing error");
        return SQLITE_ERROR;
      }

      if (endptr == ptr) {
        if (*ptr != ']') {
          sqlite3_free(x.z);
          *pzErr = sqlite3_mprintf("JSON parsing error");
          return SQLITE_ERROR;
        }
        goto done;
      }

      f32 res = (f32)result;
      array_append(&x, (const void *)&res);

      offset += (endptr - ptr);
      while (offset < source_len) {
        if (vecJsonIsspace(source[offset])) {
          offset++;
          continue;
        }
        if (source[offset] == ',') {
          offset++;
          continue;
        }
        if (source[offset] == ']')
          goto done;
        break;
      }
    }

  done:

    if (x.length > 0) {
      *vector = (f32 *)x.z;
      *dimensions = x.length;
      *cleanup = (fvec_cleanup)sqlite3_free;
      return SQLITE_OK;
    }
    sqlite3_free(x.z);
    *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
    return SQLITE_ERROR;
  }

  *pzErr = sqlite3_mprintf(
      "Input must have type BLOB (compact format) or TEXT (JSON), found %s",
      type_name(value_type));
  return SQLITE_ERROR;
}

static int bitvec_from_value(sqlite3_value *value, u8 **vector,
                             size_t *dimensions, vector_cleanup *cleanup,
                             char **pzErr) {
  int value_type = sqlite3_value_type(value);
  if (value_type == SQLITE_BLOB) {
    const void *blob = sqlite3_value_blob(value);
    int bytes = sqlite3_value_bytes(value);
    if (bytes == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }
    *vector = (u8 *)blob;
    *dimensions = bytes * CHAR_BIT;
    *cleanup = vector_cleanup_noop;
    return SQLITE_OK;
  }
  *pzErr = sqlite3_mprintf("Unknown type for bitvector.");
  return SQLITE_ERROR;
}

static int int8_vec_from_value(sqlite3_value *value, i8 **vector,
                               size_t *dimensions, vector_cleanup *cleanup,
                               char **pzErr) {
  int value_type = sqlite3_value_type(value);
  if (value_type == SQLITE_BLOB) {
    const void *blob = sqlite3_value_blob(value);
    int bytes = sqlite3_value_bytes(value);
    if (bytes == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }
    *vector = (i8 *)blob;
    *dimensions = bytes;
    *cleanup = vector_cleanup_noop;
    return SQLITE_OK;
  }

  if (value_type == SQLITE_TEXT) {
    const char *source = (const char *)sqlite3_value_text(value);
    int source_len = sqlite3_value_bytes(value);
    int i = 0;

    if (source_len == 0) {
      *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
      return SQLITE_ERROR;
    }

    struct Array x;
    int rc = array_init(&x, sizeof(i8), ceil(source_len / 2.0));
    if (rc != SQLITE_OK) {
      return rc;
    }

    // advance leading whitespace to first '['
    while (i < source_len) {
      if (vecJsonIsspace(source[i])) {
        i++;
        continue;
      }
      if (source[i] == '[') {
        break;
      }

      *pzErr = sqlite3_mprintf(
          "JSON array parsing error: Input does not start with '['");
      array_cleanup(&x);
      return SQLITE_ERROR;
    }
    if (source[i] != '[') {
      *pzErr = sqlite3_mprintf(
          "JSON array parsing error: Input does not start with '['");
      array_cleanup(&x);
      return SQLITE_ERROR;
    }
    int offset = i + 1;

    while (offset < source_len) {
      char *ptr = (char *)&source[offset];
      char *endptr;

      errno = 0;
      long result = strtol(ptr, &endptr, 10);
      if ((errno != 0 && result == 0) ||
          (errno == ERANGE && (result == LONG_MAX || result == LONG_MIN))) {
        sqlite3_free(x.z);
        *pzErr = sqlite3_mprintf("JSON parsing error");
        return SQLITE_ERROR;
      }

      if (endptr == ptr) {
        if (*ptr != ']') {
          sqlite3_free(x.z);
          *pzErr = sqlite3_mprintf("JSON parsing error");
          return SQLITE_ERROR;
        }
        goto done;
      }

      if (result < INT8_MIN || result > INT8_MAX) {
        sqlite3_free(x.z);
        *pzErr =
            sqlite3_mprintf("JSON parsing error: value out of range for int8");
        return SQLITE_ERROR;
      }

      i8 res = (i8)result;
      array_append(&x, (const void *)&res);

      offset += (endptr - ptr);
      while (offset < source_len) {
        if (vecJsonIsspace(source[offset])) {
          offset++;
          continue;
        }
        if (source[offset] == ',') {
          offset++;
          continue;
        }
        if (source[offset] == ']')
          goto done;
        break;
      }
    }

  done:

    if (x.length > 0) {
      *vector = (i8 *)x.z;
      *dimensions = x.length;
      *cleanup = (vector_cleanup)sqlite3_free;
      return SQLITE_OK;
    }
    sqlite3_free(x.z);
    *pzErr = sqlite3_mprintf("zero-length vectors are not supported.");
    return SQLITE_ERROR;
  }

  *pzErr = sqlite3_mprintf("Unknown type for int8 vector.");
  return SQLITE_ERROR;
}

/**
 * @brief Extract a vector from a sqlite3_value. Can be a float32, int8, or bit
 * vector.
 *
 * @param value: the sqlite3_value to read from.
 * @param vector: Output pointer to vector data.
 * @param dimensions: Output number of dimensions
 * @param dimensions: Output vector element type
 * @param cleanup
 * @param pzErrorMessage
 * @return int SQLITE_OK on success, error code otherwise
 */
int vector_from_value(sqlite3_value *value, void **vector, size_t *dimensions,
                      enum VectorElementType *element_type,
                      vector_cleanup *cleanup, char **pzErrorMessage) {
  int subtype = sqlite3_value_subtype(value);
  if (!subtype || (subtype == SQLITE_VEC_ELEMENT_TYPE_FLOAT32) ||
      (subtype == JSON_SUBTYPE)) {
    int rc = fvec_from_value(value, (f32 **)vector, dimensions,
                             (fvec_cleanup *)cleanup, pzErrorMessage);
    if (rc == SQLITE_OK) {
      *element_type = SQLITE_VEC_ELEMENT_TYPE_FLOAT32;
    }
    return rc;
  }

  if (subtype == SQLITE_VEC_ELEMENT_TYPE_BIT) {
    int rc = bitvec_from_value(value, (u8 **)vector, dimensions, cleanup,
                               pzErrorMessage);
    if (rc == SQLITE_OK) {
      *element_type = SQLITE_VEC_ELEMENT_TYPE_BIT;
    }
    return rc;
  }
  if (subtype == SQLITE_VEC_ELEMENT_TYPE_INT8) {
    int rc = int8_vec_from_value(value, (i8 **)vector, dimensions, cleanup,
                                 pzErrorMessage);
    if (rc == SQLITE_OK) {
      *element_type = SQLITE_VEC_ELEMENT_TYPE_INT8;
    }
    return rc;
  }
  *pzErrorMessage = sqlite3_mprintf("Unknown subtype: %d", subtype);
  return SQLITE_ERROR;
}

int ensure_vector_match(sqlite3_value *aValue, sqlite3_value *bValue, void **a,
                        void **b, enum VectorElementType *element_type,
                        size_t *dimensions, vector_cleanup *outACleanup,
                        vector_cleanup *outBCleanup, char **outError) {
  int rc;
  enum VectorElementType aType, bType;
  size_t aDims, bDims;
  char *error = NULL;
  vector_cleanup aCleanup, bCleanup;

  rc = vector_from_value(aValue, a, &aDims, &aType, &aCleanup, &error);
  if (rc != SQLITE_OK) {
    *outError = sqlite3_mprintf("Error reading 1st vector: %s", error);
    sqlite3_free(error);
    return SQLITE_ERROR;
  }

  rc = vector_from_value(bValue, b, &bDims, &bType, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    *outError = sqlite3_mprintf("Error reading 2nd vector: %s", error);
    sqlite3_free(error);
    aCleanup(a);
    return SQLITE_ERROR;
  }

  if (aType != bType) {
    *outError =
        sqlite3_mprintf("Vector type mistmatch. First vector has type %s, "
                        "while the second has type %s.",
                        vector_subtype_name(aType), vector_subtype_name(bType));
    aCleanup(*a);
    bCleanup(*b);
    return SQLITE_ERROR;
  }
  if (aDims != bDims) {
    *outError = sqlite3_mprintf(
        "Vector dimension mistmatch. First vector has %ld dimensions, "
        "while the second has %ld dimensions.",
        aDims, bDims);
    aCleanup(*a);
    bCleanup(*b);
    return SQLITE_ERROR;
  }
  *element_type = aType;
  *dimensions = aDims;
  *outACleanup = aCleanup;
  *outBCleanup = bCleanup;
  return SQLITE_OK;
}

int _cmp(const void *a, const void *b) { return (*(i64 *)a - *(i64 *)b); }

struct VecNpyFile {
  char *path;
  size_t pathLength;
};
#define SQLITE_VEC_NPY_FILE_NAME "vec0-npy-file"

#ifndef SQLITE_VEC_OMIT_FS
static void vec_npy_file(sqlite3_context *context, int argc,
                         sqlite3_value **argv) {
  assert(argc == 1);
  char *path = (char *)sqlite3_value_text(argv[0]);
  size_t pathLength = sqlite3_value_bytes(argv[0]);
  struct VecNpyFile *f;

  f = sqlite3_malloc(sizeof(*f));
  if (!f) {
    sqlite3_result_error_nomem(context);
    return;
  }
  memset(f, 0, sizeof(*f));

  f->path = path;
  f->pathLength = pathLength;
  sqlite3_result_pointer(context, f, SQLITE_VEC_NPY_FILE_NAME, sqlite3_free);
}
#endif

#pragma region scalar functions
static void vec_f32(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  int rc;
  f32 *vector = NULL;
  size_t dimensions;
  fvec_cleanup cleanup;
  char *errmsg;
  rc = fvec_from_value(argv[0], &vector, &dimensions, &cleanup, &errmsg);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, errmsg, -1);
    sqlite3_free(errmsg);
    return;
  }
  sqlite3_result_blob(context, vector, dimensions * sizeof(f32),
                      (void (*)(void *))cleanup);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
}

static void vec_bit(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  int rc;
  u8 *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *errmsg;
  rc = bitvec_from_value(argv[0], &vector, &dimensions, &cleanup, &errmsg);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, errmsg, -1);
    sqlite3_free(errmsg);
    return;
  }
  sqlite3_result_blob(context, vector, dimensions / CHAR_BIT, SQLITE_TRANSIENT);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_BIT);
  cleanup(vector);
}
static void vec_int8(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  int rc;
  i8 *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *errmsg;
  rc = int8_vec_from_value(argv[0], &vector, &dimensions, &cleanup, &errmsg);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, errmsg, -1);
    sqlite3_free(errmsg);
    return;
  }
  sqlite3_result_blob(context, vector, dimensions, SQLITE_TRANSIENT);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);
  cleanup(vector);
}

static void vec_length(sqlite3_context *context, int argc,
                       sqlite3_value **argv) {
  assert(argc == 1);
  int rc;
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *errmsg;
  enum VectorElementType elementType;
  rc = vector_from_value(argv[0], &vector, &dimensions, &elementType, &cleanup,
                         &errmsg);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, errmsg, -1);
    sqlite3_free(errmsg);
    return;
  }
  sqlite3_result_int64(context, dimensions);
  cleanup(vector);
}

static void vec_distance_cosine(sqlite3_context *context, int argc,
                                sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(
        context, "Cannot calculate cosine distance between two bitvectors.",
        -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    f32 result = distance_cosine_float(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    f32 result = distance_cosine_int8(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  }

finish:
  aCleanup(a);
  bCleanup(b);
  return;
}

static void vec_distance_l2(sqlite3_context *context, int argc,
                            sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(
        context, "Cannot calculate L2 distance between two bitvectors.", -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    f32 result = distance_l2_sqr_float(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    f32 result = distance_l2_sqr_int8(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  }

finish:
  aCleanup(a);
  bCleanup(b);
  return;
}

static void vec_distance_l1(sqlite3_context *context, int argc,
                            sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a, *b;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(
        context, "Cannot calculate L1 distance between two bitvectors.", -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    double result = distance_l1_f32(a, b, &dimensions);
    sqlite3_result_double(context, result);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    i64 result = distance_l1_int8(a, b, &dimensions);
    sqlite3_result_int(context, result);
    goto finish;
  }
  }

finish:
  aCleanup(a);
  bCleanup(b);
  return;
}

static void vec_distance_hamming(sqlite3_context *context, int argc,
                                 sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_double(context, distance_hamming(a, b, &dimensions));
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    sqlite3_result_error(
        context,
        "Cannot calculate hamming distance between two float32 vectors.", -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    sqlite3_result_error(
        context, "Cannot calculate hamming distance between two int8 vectors.",
        -1);
    goto finish;
  }
  }

finish:
  aCleanup(a);
  bCleanup(b);
  return;
}

char *vec_type_name(enum VectorElementType elementType) {
  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
    return "float32";
  case SQLITE_VEC_ELEMENT_TYPE_INT8:
    return "int8";
  case SQLITE_VEC_ELEMENT_TYPE_BIT:
    return "bit";
  }
  return "";
}

static void vec_type(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *pzError;
  enum VectorElementType elementType;
  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &cleanup, &pzError);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, pzError, -1);
    sqlite3_free(pzError);
    return;
  }
  sqlite3_result_text(context, vec_type_name(elementType), -1, SQLITE_STATIC);
  cleanup(vector);
}
static void vec_quantize_binary(sqlite3_context *context, int argc,
                                sqlite3_value **argv) {
  assert(argc == 1);
  void *vector;
  size_t dimensions;
  vector_cleanup vectorCleanup;
  char *pzError;
  enum VectorElementType elementType;
  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &vectorCleanup, &pzError);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, pzError, -1);
    sqlite3_free(pzError);
    return;
  }

  if (dimensions <= 0) {
    sqlite3_result_error(context, "Zero length vectors are not supported.", -1);
    goto cleanup;
    return;
  }
  if ((dimensions % CHAR_BIT) != 0) {
    sqlite3_result_error(
        context,
        "Binary quantization requires vectors with a length divisible by 8",
        -1);
    goto cleanup;
    return;
  }

  int sz = dimensions / CHAR_BIT;
  u8 *out = sqlite3_malloc(sz);
  if (!out) {
    sqlite3_result_error_code(context, SQLITE_NOMEM);
    goto cleanup;
    return;
  }
  memset(out, 0, sz);

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {

    for (size_t i = 0; i < dimensions; i++) {
      int res = ((f32 *)vector)[i] > 0.0;
      out[i / 8] |= (res << (i % 8));
    }
    break;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    for (size_t i = 0; i < dimensions; i++) {
      int res = ((i8 *)vector)[i] > 0;
      out[i / 8] |= (res << (i % 8));
    }
    break;
  }
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(context,
                         "Can only binary quantize float or int8 vectors", -1);
    sqlite3_free(out);
    return;
  }
  }
  sqlite3_result_blob(context, out, sz, sqlite3_free);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_BIT);

cleanup:
  vectorCleanup(vector);
}

static void vec_quantize_int8(sqlite3_context *context, int argc,
                              sqlite3_value **argv) {
  assert(argc == 2);
  f32 *srcVector;
  size_t dimensions;
  fvec_cleanup srcCleanup;
  char *err;
  i8 *out = NULL;
  int rc = fvec_from_value(argv[0], &srcVector, &dimensions, &srcCleanup, &err);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }

  int sz = dimensions * sizeof(i8);
  out = sqlite3_malloc(sz);
  if (!out) {
    sqlite3_result_error_nomem(context);
    goto cleanup;
  }
  memset(out, 0, sz);

  if ((sqlite3_value_type(argv[1]) != SQLITE_TEXT) ||
      (sqlite3_value_bytes(argv[1]) != strlen("unit")) ||
      (sqlite3_stricmp((const char *)sqlite3_value_text(argv[1]), "unit") !=
       0)) {
    sqlite3_result_error(
        context, "2nd argument to vec_quantize_int8() must be 'unit'.", -1);
    sqlite3_free(out);
    goto cleanup;
  }
  f32 step = (1.0 - (-1.0)) / 255;
  for (size_t i = 0; i < dimensions; i++) {
    out[i] = ((srcVector[i] - (-1.0)) / step) - 128;
  }

  sqlite3_result_blob(context, out, dimensions * sizeof(i8), sqlite3_free);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);

cleanup:
  srcCleanup(srcVector);
}

static void vec_add(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(context, "Cannot add two bitvectors together.", -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    size_t outSize = dimensions * sizeof(f32);
    f32 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto finish;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < dimensions; i++) {
      out[i] = ((f32 *)a)[i] + ((f32 *)b)[i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    size_t outSize = dimensions * sizeof(i8);
    i8 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto finish;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < dimensions; i++) {
      out[i] = ((i8 *)a)[i] + ((i8 *)b)[i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);
    goto finish;
  }
  }
finish:
  aCleanup(a);
  bCleanup(b);
  return;
}
static void vec_sub(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 2);
  int rc;
  void *a = NULL, *b = NULL;
  size_t dimensions;
  vector_cleanup aCleanup, bCleanup;
  char *error;
  enum VectorElementType elementType;
  rc = ensure_vector_match(argv[0], argv[1], &a, &b, &elementType, &dimensions,
                           &aCleanup, &bCleanup, &error);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, error, -1);
    sqlite3_free(error);
    return;
  }

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    sqlite3_result_error(context, "Cannot subtract two bitvectors together.",
                         -1);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    size_t outSize = dimensions * sizeof(f32);
    f32 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto finish;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < dimensions; i++) {
      out[i] = ((f32 *)a)[i] - ((f32 *)b)[i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    goto finish;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    size_t outSize = dimensions * sizeof(i8);
    i8 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto finish;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < dimensions; i++) {
      out[i] = ((i8 *)a)[i] - ((i8 *)b)[i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);
    goto finish;
  }
  }
finish:
  aCleanup(a);
  bCleanup(b);
  return;
}
static void vec_slice(sqlite3_context *context, int argc,
                      sqlite3_value **argv) {
  assert(argc == 3);

  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *err;
  enum VectorElementType elementType;

  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &cleanup, &err);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }

  int start = sqlite3_value_int(argv[1]);
  int end = sqlite3_value_int(argv[2]);

  if (start < 0) {
    sqlite3_result_error(context,
                         "slice 'start' index must be a postive number.", -1);
    goto done;
  }
  if (end < 0) {
    sqlite3_result_error(context, "slice 'end' index must be a postive number.",
                         -1);
    goto done;
  }
  if (((size_t)start) > dimensions) {
    sqlite3_result_error(
        context, "slice 'start' index is greater than the number of dimensions",
        -1);
    goto done;
  }
  if (((size_t)end) > dimensions) {
    sqlite3_result_error(
        context, "slice 'end' index is greater than the number of dimensions",
        -1);
    goto done;
  }
  if (start > end) {
    sqlite3_result_error(context,
                         "slice 'start' index is greater than 'end' index", -1);
    goto done;
  }
  if (start == end) {
    sqlite3_result_error(context,
                         "slice 'start' index is equal to the 'end' index, "
                         "vectors must have non-zero length",
                         -1);
    goto done;
  }
  size_t n = end - start;

  switch (elementType) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
    int outSize = n * sizeof(f32);
    f32 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      goto done;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < n; i++) {
      out[i] = ((f32 *)vector)[start + i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    goto done;
  }
  case SQLITE_VEC_ELEMENT_TYPE_INT8: {
    int outSize = n * sizeof(i8);
    i8 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      return;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < n; i++) {
      out[i] = ((i8 *)vector)[start + i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_INT8);
    goto done;
  }
  case SQLITE_VEC_ELEMENT_TYPE_BIT: {
    if ((start % CHAR_BIT) != 0) {
      sqlite3_result_error(context, "start index must be divisible by 8.", -1);
      goto done;
    }
    if ((end % CHAR_BIT) != 0) {
      sqlite3_result_error(context, "end index must be divisible by 8.", -1);
      goto done;
    }
    int outSize = n / CHAR_BIT;
    u8 *out = sqlite3_malloc(outSize);
    if (!out) {
      sqlite3_result_error_nomem(context);
      return;
    }
    memset(out, 0, outSize);
    for (size_t i = 0; i < n / CHAR_BIT; i++) {
      out[i] = ((u8 *)vector)[(start / CHAR_BIT) + i];
    }
    sqlite3_result_blob(context, out, outSize, sqlite3_free);
    sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_BIT);
    goto done;
  }
  }
done:
  cleanup(vector);
}

static void vec_to_json(sqlite3_context *context, int argc,
                        sqlite3_value **argv) {
  assert(argc == 1);
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *err;
  enum VectorElementType elementType;

  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &cleanup, &err);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }

  sqlite3_str *str = sqlite3_str_new(sqlite3_context_db_handle(context));
  sqlite3_str_appendall(str, "[");
  for (size_t i = 0; i < dimensions; i++) {
    if (i != 0) {
      sqlite3_str_appendall(str, ",");
    }
    if (elementType == SQLITE_VEC_ELEMENT_TYPE_FLOAT32) {
      f32 value = ((f32 *)vector)[i];
      if (isnan(value)) {
        sqlite3_str_appendall(str, "null");
      } else {
        sqlite3_str_appendf(str, "%f", value);
      }

    } else if (elementType == SQLITE_VEC_ELEMENT_TYPE_INT8) {
      sqlite3_str_appendf(str, "%d", ((i8 *)vector)[i]);
    } else if (elementType == SQLITE_VEC_ELEMENT_TYPE_BIT) {
      u8 b = (((u8 *)vector)[i / 8] >> (i % CHAR_BIT)) & 1;
      sqlite3_str_appendf(str, "%d", b);
    }
  }
  sqlite3_str_appendall(str, "]");
  int len = sqlite3_str_length(str);
  char *s = sqlite3_str_finish(str);
  if (s) {
    sqlite3_result_text(context, s, len, sqlite3_free);
    sqlite3_result_subtype(context, JSON_SUBTYPE);
  } else {
    sqlite3_result_error_nomem(context);
  }
  cleanup(vector);
}

static void vec_normalize(sqlite3_context *context, int argc,
                          sqlite3_value **argv) {
  assert(argc == 1);
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
  char *err;
  enum VectorElementType elementType;

  int rc = vector_from_value(argv[0], &vector, &dimensions, &elementType,
                             &cleanup, &err);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }

  if (elementType != SQLITE_VEC_ELEMENT_TYPE_FLOAT32) {
    sqlite3_result_error(
        context, "only float32 vectors are supported when normalizing", -1);
    cleanup(vector);
    return;
  }

  int outSize = dimensions * sizeof(f32);
  f32 *out = sqlite3_malloc(outSize);
  if (!out) {
    cleanup(vector);
    sqlite3_result_error_code(context, SQLITE_NOMEM);
    return;
  }
  memset(out, 0, outSize);

  f32 *v = (f32 *)vector;

  f32 norm = 0;
  for (size_t i = 0; i < dimensions; i++) {
    norm += v[i] * v[i];
  }
  norm = sqrt(norm);
  for (size_t i = 0; i < dimensions; i++) {
    out[i] = v[i] / norm;
  }

  sqlite3_result_blob(context, out, dimensions * sizeof(f32), sqlite3_free);
  sqlite3_result_subtype(context, SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
  cleanup(vector);
}

static void _static_text_func(sqlite3_context *context, int argc,
                              sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  sqlite3_result_text(context, sqlite3_user_data(context), -1, SQLITE_STATIC);
}

#pragma endregion

enum Vec0TokenType {
  TOKEN_TYPE_IDENTIFIER,
  TOKEN_TYPE_DIGIT,
  TOKEN_TYPE_LBRACKET,
  TOKEN_TYPE_RBRACKET,
  TOKEN_TYPE_PLUS,
  TOKEN_TYPE_EQ,
};
struct Vec0Token {
  enum Vec0TokenType token_type;
  char *start;
  char *end;
};

int is_alpha(char x) {
  return (x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z');
}
int is_digit(char x) { return (x >= '0' && x <= '9'); }
int is_whitespace(char x) {
  return x == ' ' || x == '\t' || x == '\n' || x == '\r';
}

#define VEC0_TOKEN_RESULT_EOF 1
#define VEC0_TOKEN_RESULT_SOME 2
#define VEC0_TOKEN_RESULT_ERROR 3

int vec0_token_next(char *start, char *end, struct Vec0Token *out) {
  char *ptr = start;
  while (ptr < end) {
    char curr = *ptr;
    if (is_whitespace(curr)) {
      ptr++;
      continue;
    } else if (curr == '+') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_PLUS;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == '[') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_LBRACKET;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == ']') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_RBRACKET;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == '=') {
      ptr++;
      out->start = ptr;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_EQ;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (is_alpha(curr)) {
      char *start = ptr;
      while (ptr < end && (is_alpha(*ptr) || is_digit(*ptr) || *ptr == '_')) {
        ptr++;
      }
      out->start = start;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_IDENTIFIER;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (is_digit(curr)) {
      char *start = ptr;
      while (ptr < end && (is_digit(*ptr))) {
        ptr++;
      }
      out->start = start;
      out->end = ptr;
      out->token_type = TOKEN_TYPE_DIGIT;
      return VEC0_TOKEN_RESULT_SOME;
    } else {
      return VEC0_TOKEN_RESULT_ERROR;
    }
  }
  return VEC0_TOKEN_RESULT_EOF;
}

struct Vec0Scanner {
  char *start;
  char *end;
  char *ptr;
};

void vec0_scanner_init(struct Vec0Scanner *scanner, const char *source,
                       int source_length) {
  scanner->start = (char *)source;
  scanner->end = (char *)source + source_length;
  scanner->ptr = (char *)source;
}
int vec0_scanner_next(struct Vec0Scanner *scanner, struct Vec0Token *out) {
  int rc = vec0_token_next(scanner->start, scanner->end, out);
  if (rc == VEC0_TOKEN_RESULT_SOME) {
    scanner->start = out->end;
  }
  return rc;
}

int vec0_parse_table_option(const char *source, int source_length,
                            char **out_key, int *out_key_length,
                            char **out_value, int *out_value_length) {
  int rc;
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *key;
  char *value;
  int keyLength, valueLength;

  vec0_scanner_init(&scanner, source, source_length);

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  key = token.start;
  keyLength = token.end - token.start;

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME && token.token_type != TOKEN_TYPE_EQ) {
    return SQLITE_EMPTY;
  }

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      !((token.token_type == TOKEN_TYPE_IDENTIFIER) ||
        (token.token_type == TOKEN_TYPE_DIGIT))) {
    return SQLITE_ERROR;
  }
  value = token.start;
  valueLength = token.end - token.start;

  rc = vec0_scanner_next(&scanner, &token);
  if (rc == VEC0_TOKEN_RESULT_EOF) {
    *out_key = key;
    *out_key_length = keyLength;
    *out_value = value;
    *out_value_length = valueLength;
    return SQLITE_OK;
  }
  return SQLITE_ERROR;
}
/**
 * @brief Parse an argv[i] entry of a vec0 virtual table definition, and see if
 * it's a PARTITION KEY definition.
 *
 * @param source: argv[i] source string
 * @param source_length: length of the source string
 * @param out_column_name: If it is a partition key, the output column name. Same lifetime
 * as source, points to specific char *
 * @param out_column_name_length: Length of out_column_name in bytes
 * @param out_column_type: SQLITE_TEXT or SQLITE_INTEGER.
 * @return int: SQLITE_EMPTY if not a PK, SQLITE_OK if it is.
 */
int vec0_parse_partition_key_definition(const char *source, int source_length,
                                 char **out_column_name,
                                 int *out_column_name_length,
                                 int *out_column_type) {
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *column_name;
  int column_name_length;
  int column_type;
  vec0_scanner_init(&scanner, source, source_length);

  // Check first token is identifier, will be the column name
  int rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  column_name = token.start;
  column_name_length = token.end - token.start;

  // Check the next token matches "text" or "integer", as column type
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "text", token.end - token.start) == 0) {
    column_type = SQLITE_TEXT;
  } else if (sqlite3_strnicmp(token.start, "int", token.end - token.start) ==
                 0 ||
             sqlite3_strnicmp(token.start, "integer",
                              token.end - token.start) == 0) {
    column_type = SQLITE_INTEGER;
  } else {
    return SQLITE_EMPTY;
  }

  // Check the next token is identifier and matches "partition"
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "partition", token.end - token.start) != 0) {
    return SQLITE_EMPTY;
  }

  // Check the next token is identifier and matches "key"
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "key", token.end - token.start) != 0) {
    return SQLITE_EMPTY;
  }

  *out_column_name = column_name;
  *out_column_name_length = column_name_length;
  *out_column_type = column_type;

  return SQLITE_OK;
}

/**
 * @brief Parse an argv[i] entry of a vec0 virtual table definition, and see if
 * it's an auxiliar column definition, ie `+[name] [type]` like `+contents text`
 *
 * @param source: argv[i] source string
 * @param source_length: length of the source string
 * @param out_column_name: If it is a partition key, the output column name. Same lifetime
 * as source, points to specific char *
 * @param out_column_name_length: Length of out_column_name in bytes
 * @param out_column_type: SQLITE_TEXT, SQLITE_INTEGER, SQLITE_FLOAT, or SQLITE_BLOB.
 * @return int: SQLITE_EMPTY if not an aux column, SQLITE_OK if it is.
 */
int vec0_parse_auxiliary_column_definition(const char *source, int source_length,
                                 char **out_column_name,
                                 int *out_column_name_length,
                                 int *out_column_type) {
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *column_name;
  int column_name_length;
  int column_type;
  vec0_scanner_init(&scanner, source, source_length);

  // Check first token is '+', which denotes aux columns
  int rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_PLUS) {
    return SQLITE_EMPTY;
  }

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  column_name = token.start;
  column_name_length = token.end - token.start;

  // Check the next token matches "text" or "integer", as column type
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "text", token.end - token.start) == 0) {
    column_type = SQLITE_TEXT;
  } else if (sqlite3_strnicmp(token.start, "int", token.end - token.start) ==
                 0 ||
             sqlite3_strnicmp(token.start, "integer",
                              token.end - token.start) == 0) {
    column_type = SQLITE_INTEGER;
  } else if (sqlite3_strnicmp(token.start, "float", token.end - token.start) ==
                 0 ||
             sqlite3_strnicmp(token.start, "double",
                              token.end - token.start) == 0) {
    column_type = SQLITE_FLOAT;
  } else if (sqlite3_strnicmp(token.start, "blob", token.end - token.start) ==0) {
    column_type = SQLITE_BLOB;
  } else {
    return SQLITE_EMPTY;
  }

  *out_column_name = column_name;
  *out_column_name_length = column_name_length;
  *out_column_type = column_type;

  return SQLITE_OK;
}

typedef enum {
  VEC0_METADATA_COLUMN_KIND_BOOLEAN,
  VEC0_METADATA_COLUMN_KIND_INTEGER,
  VEC0_METADATA_COLUMN_KIND_FLOAT,
  VEC0_METADATA_COLUMN_KIND_TEXT,
  // future: blob, date, datetime
} vec0_metadata_column_kind;

/**
 * @brief Parse an argv[i] entry of a vec0 virtual table definition, and see if
 * it's an metadata column definition, ie `[name] [type]` like `is_released boolean`
 *
 * @param source: argv[i] source string
 * @param source_length: length of the source string
 * @param out_column_name: If it is a metadata column, the output column name. Same lifetime
 * as source, points to specific char *
 * @param out_column_name_length: Length of out_column_name in bytes
 * @param out_column_type: one of vec0_metadata_column_kind
 * @return int: SQLITE_EMPTY if not an metadata column, SQLITE_OK if it is.
 */
int vec0_parse_metadata_column_definition(const char *source, int source_length,
                                 char **out_column_name,
                                 int *out_column_name_length,
                                 vec0_metadata_column_kind *out_column_type) {
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *column_name;
  int column_name_length;
  vec0_metadata_column_kind column_type;
  int rc;
  vec0_scanner_init(&scanner, source, source_length);

  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  column_name = token.start;
  column_name_length = token.end - token.start;

  // Check the next token matches a valid metadata type
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  char * t = token.start;
  int n = token.end - token.start;
  if (sqlite3_strnicmp(t, "boolean", n) == 0 || sqlite3_strnicmp(t, "bool", n) == 0) {
    column_type = VEC0_METADATA_COLUMN_KIND_BOOLEAN;
  }else if (sqlite3_strnicmp(t, "int64", n) == 0 || sqlite3_strnicmp(t, "integer64", n) == 0 || sqlite3_strnicmp(t, "integer", n) == 0 || sqlite3_strnicmp(t, "int", n) == 0) {
    column_type = VEC0_METADATA_COLUMN_KIND_INTEGER;
  }else if (sqlite3_strnicmp(t, "float", n) == 0 || sqlite3_strnicmp(t, "double", n) == 0 || sqlite3_strnicmp(t, "float64", n) == 0 || sqlite3_strnicmp(t, "f64", n) == 0) {
    column_type = VEC0_METADATA_COLUMN_KIND_FLOAT;
  } else if (sqlite3_strnicmp(t, "text", n) == 0) {
    column_type = VEC0_METADATA_COLUMN_KIND_TEXT;
  } else {
    return SQLITE_EMPTY;
  }

  *out_column_name = column_name;
  *out_column_name_length = column_name_length;
  *out_column_type = column_type;

  return SQLITE_OK;
}

/**
 * @brief Parse an argv[i] entry of a vec0 virtual table definition, and see if
 * it's a PRIMARY KEY definition.
 *
 * @param source: argv[i] source string
 * @param source_length: length of the source string
 * @param out_column_name: If it is a PK, the output column name. Same lifetime
 * as source, points to specific char *
 * @param out_column_name_length: Length of out_column_name in bytes
 * @param out_column_type: SQLITE_TEXT or SQLITE_INTEGER.
 * @return int: SQLITE_EMPTY if not a PK, SQLITE_OK if it is.
 */
int vec0_parse_primary_key_definition(const char *source, int source_length,
                                 char **out_column_name,
                                 int *out_column_name_length,
                                 int *out_column_type) {
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  char *column_name;
  int column_name_length;
  int column_type;
  vec0_scanner_init(&scanner, source, source_length);

  // Check first token is identifier, will be the column name
  int rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  column_name = token.start;
  column_name_length = token.end - token.start;

  // Check the next token matches "text" or "integer", as column type
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "text", token.end - token.start) == 0) {
    column_type = SQLITE_TEXT;
  } else if (sqlite3_strnicmp(token.start, "int", token.end - token.start) ==
                 0 ||
             sqlite3_strnicmp(token.start, "integer",
                              token.end - token.start) == 0) {
    column_type = SQLITE_INTEGER;
  } else {
    return SQLITE_EMPTY;
  }

  // Check the next token is identifier and matches "primary"
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "primary", token.end - token.start) != 0) {
    return SQLITE_EMPTY;
  }

  // Check the next token is identifier and matches "key"
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "key", token.end - token.start) != 0) {
    return SQLITE_EMPTY;
  }

  *out_column_name = column_name;
  *out_column_name_length = column_name_length;
  *out_column_type = column_type;

  return SQLITE_OK;
}

enum Vec0DistanceMetrics {
  VEC0_DISTANCE_METRIC_L2 = 1,
  VEC0_DISTANCE_METRIC_COSINE = 2,
  VEC0_DISTANCE_METRIC_L1 = 3,
};

struct VectorColumnDefinition {
  char *name;
  int name_length;
  size_t dimensions;
  enum VectorElementType element_type;
  enum Vec0DistanceMetrics distance_metric;
};

struct Vec0PartitionColumnDefinition {
  int type;
  char * name;
  int name_length;
};

struct Vec0AuxiliaryColumnDefinition {
  int type;
  char * name;
  int name_length;
};
struct Vec0MetadataColumnDefinition {
  vec0_metadata_column_kind kind;
  char * name;
  int name_length;
};

size_t vector_byte_size(enum VectorElementType element_type,
                        size_t dimensions) {
  switch (element_type) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
    return dimensions * sizeof(f32);
  case SQLITE_VEC_ELEMENT_TYPE_INT8:
    return dimensions * sizeof(i8);
  case SQLITE_VEC_ELEMENT_TYPE_BIT:
    return dimensions / CHAR_BIT;
  }
  return 0;
}

size_t vector_column_byte_size(struct VectorColumnDefinition column) {
  return vector_byte_size(column.element_type, column.dimensions);
}

/**
 * @brief Parse an vec0 vtab argv[i] column definition and see if
 * it's a vector column defintion, ex `contents_embedding float[768]`.
 *
 * @param source vec0 argv[i] item
 * @param source_length length of source in bytes
 * @param outColumn Output the parse vector column to this struct, if success
 * @return int SQLITE_OK on success, SQLITE_EMPTY is it's not a vector column
 * definition, SQLITE_ERROR on error.
 */
int vec0_parse_vector_column(const char *source, int source_length,
                        struct VectorColumnDefinition *outColumn) {
  // parses a vector column definition like so:
  // "abc float[123]", "abc_123 bit[1234]", eetc.
  // https://github.com/asg017/sqlite-vec/issues/46
  int rc;
  struct Vec0Scanner scanner;
  struct Vec0Token token;

  char *name;
  int nameLength;
  enum VectorElementType elementType;
  enum Vec0DistanceMetrics distanceMetric = VEC0_DISTANCE_METRIC_L2;
  int dimensions;

  vec0_scanner_init(&scanner, source, source_length);

  // starts with an identifier
  rc = vec0_scanner_next(&scanner, &token);

  if (rc != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }

  name = token.start;
  nameLength = token.end - token.start;

  // vector column type comes next: float, int, or bit
  rc = vec0_scanner_next(&scanner, &token);

  if (rc != VEC0_TOKEN_RESULT_SOME ||
      token.token_type != TOKEN_TYPE_IDENTIFIER) {
    return SQLITE_EMPTY;
  }
  if (sqlite3_strnicmp(token.start, "float", 5) == 0 ||
      sqlite3_strnicmp(token.start, "f32", 3) == 0) {
    elementType = SQLITE_VEC_ELEMENT_TYPE_FLOAT32;
  } else if (sqlite3_strnicmp(token.start, "int8", 4) == 0 ||
             sqlite3_strnicmp(token.start, "i8", 2) == 0) {
    elementType = SQLITE_VEC_ELEMENT_TYPE_INT8;
  } else if (sqlite3_strnicmp(token.start, "bit", 3) == 0) {
    elementType = SQLITE_VEC_ELEMENT_TYPE_BIT;
  } else {
    return SQLITE_EMPTY;
  }

  // left '[' bracket
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME && token.token_type != TOKEN_TYPE_LBRACKET) {
    return SQLITE_EMPTY;
  }

  // digit, for vector dimension length
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME && token.token_type != TOKEN_TYPE_DIGIT) {
    return SQLITE_ERROR;
  }
  dimensions = atoi(token.start);
  if (dimensions <= 0) {
    return SQLITE_ERROR;
  }

  // // right ']' bracket
  rc = vec0_scanner_next(&scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME && token.token_type != TOKEN_TYPE_RBRACKET) {
    return SQLITE_ERROR;
  }

  // any other tokens left should be column-level options , ex `key=value`
  // ex `distance_metric=L2 distance_metric=cosine` should error
  while (1) {
    // should be EOF or identifier (option key)
    rc = vec0_scanner_next(&scanner, &token);
    if (rc == VEC0_TOKEN_RESULT_EOF) {
      break;
    }

    if (rc != VEC0_TOKEN_RESULT_SOME &&
        token.token_type != TOKEN_TYPE_IDENTIFIER) {
      return SQLITE_ERROR;
    }

    char *key = token.start;
    int keyLength = token.end - token.start;

    if (sqlite3_strnicmp(key, "distance_metric", keyLength) == 0) {

      if (elementType == SQLITE_VEC_ELEMENT_TYPE_BIT) {
        return SQLITE_ERROR;
      }
      // ensure equal sign after distance_metric
      rc = vec0_scanner_next(&scanner, &token);
      if (rc != VEC0_TOKEN_RESULT_SOME && token.token_type != TOKEN_TYPE_EQ) {
        return SQLITE_ERROR;
      }

      // distance_metric value, an identifier (L2, cosine, etc)
      rc = vec0_scanner_next(&scanner, &token);
      if (rc != VEC0_TOKEN_RESULT_SOME &&
          token.token_type != TOKEN_TYPE_IDENTIFIER) {
        return SQLITE_ERROR;
      }

      char *value = token.start;
      int valueLength = token.end - token.start;
      if (sqlite3_strnicmp(value, "l2", valueLength) == 0) {
        distanceMetric = VEC0_DISTANCE_METRIC_L2;
      } else if (sqlite3_strnicmp(value, "l1", valueLength) == 0) {
        distanceMetric = VEC0_DISTANCE_METRIC_L1;
      } else if (sqlite3_strnicmp(value, "cosine", valueLength) == 0) {
        distanceMetric = VEC0_DISTANCE_METRIC_COSINE;
      } else {
        return SQLITE_ERROR;
      }
    }
    // unknown key
    else {
      return SQLITE_ERROR;
    }
  }

  outColumn->name = sqlite3_mprintf("%.*s", nameLength, name);
  if (!outColumn->name) {
    return SQLITE_ERROR;
  }
  outColumn->name_length = nameLength;
  outColumn->distance_metric = distanceMetric;
  outColumn->element_type = elementType;
  outColumn->dimensions = dimensions;
  return SQLITE_OK;
}

#pragma region vec_each table function

typedef struct vec_each_vtab vec_each_vtab;
struct vec_each_vtab {
  sqlite3_vtab base;
};

typedef struct vec_each_cursor vec_each_cursor;
struct vec_each_cursor {
  sqlite3_vtab_cursor base;
  i64 iRowid;
  enum VectorElementType vector_type;
  void *vector;
  size_t dimensions;
  vector_cleanup cleanup;
};

static int vec_eachConnect(sqlite3 *db, void *pAux, int argc,
                           const char *const *argv, sqlite3_vtab **ppVtab,
                           char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  vec_each_vtab *pNew;
  int rc;

  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(value, vector hidden)");
#define VEC_EACH_COLUMN_VALUE 0
#define VEC_EACH_COLUMN_VECTOR 1
  if (rc == SQLITE_OK) {
    pNew = sqlite3_malloc(sizeof(*pNew));
    *ppVtab = (sqlite3_vtab *)pNew;
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
  }
  return rc;
}

static int vec_eachDisconnect(sqlite3_vtab *pVtab) {
  vec_each_vtab *p = (vec_each_vtab *)pVtab;
  sqlite3_free(p);
  return SQLITE_OK;
}

static int vec_eachOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec_each_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec_eachClose(sqlite3_vtab_cursor *cur) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  pCur->cleanup(pCur->vector);
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int vec_eachBestIndex(sqlite3_vtab *pVTab,
                             sqlite3_index_info *pIdxInfo) {
  UNUSED_PARAMETER(pVTab);
  int hasVector = 0;
  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *pCons = &pIdxInfo->aConstraint[i];
    // printf("i=%d iColumn=%d, op=%d, usable=%d\n", i, pCons->iColumn,
    // pCons->op, pCons->usable);
    switch (pCons->iColumn) {
    case VEC_EACH_COLUMN_VECTOR: {
      if (pCons->op == SQLITE_INDEX_CONSTRAINT_EQ && pCons->usable) {
        hasVector = 1;
        pIdxInfo->aConstraintUsage[i].argvIndex = 1;
        pIdxInfo->aConstraintUsage[i].omit = 1;
      }
      break;
    }
    }
  }
  if (!hasVector) {
    return SQLITE_CONSTRAINT;
  }

  pIdxInfo->estimatedCost = (double)100000;
  pIdxInfo->estimatedRows = 100000;

  return SQLITE_OK;
}

static int vec_eachFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                          const char *idxStr, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  assert(argc == 1);
  vec_each_cursor *pCur = (vec_each_cursor *)pVtabCursor;

  if (pCur->vector) {
    pCur->cleanup(pCur->vector);
    pCur->vector = NULL;
  }

  char *pzErrMsg;
  int rc = vector_from_value(argv[0], &pCur->vector, &pCur->dimensions,
                             &pCur->vector_type, &pCur->cleanup, &pzErrMsg);
  if (rc != SQLITE_OK) {
    return SQLITE_ERROR;
  }
  pCur->iRowid = 0;
  return SQLITE_OK;
}

static int vec_eachRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}

static int vec_eachEof(sqlite3_vtab_cursor *cur) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  return pCur->iRowid >= (i64)pCur->dimensions;
}

static int vec_eachNext(sqlite3_vtab_cursor *cur) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  pCur->iRowid++;
  return SQLITE_OK;
}

static int vec_eachColumn(sqlite3_vtab_cursor *cur, sqlite3_context *context,
                          int i) {
  vec_each_cursor *pCur = (vec_each_cursor *)cur;
  switch (i) {
  case VEC_EACH_COLUMN_VALUE:
    switch (pCur->vector_type) {
    case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
      sqlite3_result_double(context, ((f32 *)pCur->vector)[pCur->iRowid]);
      break;
    }
    case SQLITE_VEC_ELEMENT_TYPE_BIT: {
      u8 x = ((u8 *)pCur->vector)[pCur->iRowid / CHAR_BIT];
      sqlite3_result_int(context,
                         (x & (0b10000000 >> ((pCur->iRowid % CHAR_BIT)))) > 0);
      break;
    }
    case SQLITE_VEC_ELEMENT_TYPE_INT8: {
      sqlite3_result_int(context, ((i8 *)pCur->vector)[pCur->iRowid]);
      break;
    }
    }

    break;
  }
  return SQLITE_OK;
}

static sqlite3_module vec_eachModule = {
    /* iVersion    */ 0,
    /* xCreate     */ 0,
    /* xConnect    */ vec_eachConnect,
    /* xBestIndex  */ vec_eachBestIndex,
    /* xDisconnect */ vec_eachDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ vec_eachOpen,
    /* xClose      */ vec_eachClose,
    /* xFilter     */ vec_eachFilter,
    /* xNext       */ vec_eachNext,
    /* xEof        */ vec_eachEof,
    /* xColumn     */ vec_eachColumn,
    /* xRowid      */ vec_eachRowid,
    /* xUpdate     */ 0,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity  */ 0
#endif
};

#pragma endregion

#pragma region vec_npy_each table function

enum NpyTokenType {
  NPY_TOKEN_TYPE_IDENTIFIER,
  NPY_TOKEN_TYPE_NUMBER,
  NPY_TOKEN_TYPE_LPAREN,
  NPY_TOKEN_TYPE_RPAREN,
  NPY_TOKEN_TYPE_LBRACE,
  NPY_TOKEN_TYPE_RBRACE,
  NPY_TOKEN_TYPE_COLON,
  NPY_TOKEN_TYPE_COMMA,
  NPY_TOKEN_TYPE_STRING,
  NPY_TOKEN_TYPE_FALSE,
};

struct NpyToken {
  enum NpyTokenType token_type;
  unsigned char *start;
  unsigned char *end;
};

int npy_token_next(unsigned char *start, unsigned char *end,
                   struct NpyToken *out) {
  unsigned char *ptr = start;
  while (ptr < end) {
    unsigned char curr = *ptr;
    if (is_whitespace(curr)) {
      ptr++;
      continue;
    } else if (curr == '(') {
      out->start = ptr++;
      out->end = ptr;
      out->token_type = NPY_TOKEN_TYPE_LPAREN;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == ')') {
      out->start = ptr++;
      out->end = ptr;
      out->token_type = NPY_TOKEN_TYPE_RPAREN;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == '{') {
      out->start = ptr++;
      out->end = ptr;
      out->token_type = NPY_TOKEN_TYPE_LBRACE;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == '}') {
      out->start = ptr++;
      out->end = ptr;
      out->token_type = NPY_TOKEN_TYPE_RBRACE;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == ':') {
      out->start = ptr++;
      out->end = ptr;
      out->token_type = NPY_TOKEN_TYPE_COLON;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == ',') {
      out->start = ptr++;
      out->end = ptr;
      out->token_type = NPY_TOKEN_TYPE_COMMA;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == '\'') {
      unsigned char *start = ptr;
      ptr++;
      while (ptr < end) {
        if ((*ptr) == '\'') {
          break;
        }
        ptr++;
      }
      if ((*ptr) != '\'') {
        return VEC0_TOKEN_RESULT_ERROR;
      }
      out->start = start;
      out->end = ++ptr;
      out->token_type = NPY_TOKEN_TYPE_STRING;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (curr == 'F' &&
               strncmp((char *)ptr, "False", strlen("False")) == 0) {
      out->start = ptr;
      out->end = (ptr + (int)strlen("False"));
      ptr = out->end;
      out->token_type = NPY_TOKEN_TYPE_FALSE;
      return VEC0_TOKEN_RESULT_SOME;
    } else if (is_digit(curr)) {
      unsigned char *start = ptr;
      while (ptr < end && (is_digit(*ptr))) {
        ptr++;
      }
      out->start = start;
      out->end = ptr;
      out->token_type = NPY_TOKEN_TYPE_NUMBER;
      return VEC0_TOKEN_RESULT_SOME;
    } else {
      return VEC0_TOKEN_RESULT_ERROR;
    }
  }
  return VEC0_TOKEN_RESULT_ERROR;
}

struct NpyScanner {
  unsigned char *start;
  unsigned char *end;
  unsigned char *ptr;
};

void npy_scanner_init(struct NpyScanner *scanner, const unsigned char *source,
                      int source_length) {
  scanner->start = (unsigned char *)source;
  scanner->end = (unsigned char *)source + source_length;
  scanner->ptr = (unsigned char *)source;
}

int npy_scanner_next(struct NpyScanner *scanner, struct NpyToken *out) {
  int rc = npy_token_next(scanner->start, scanner->end, out);
  if (rc == VEC0_TOKEN_RESULT_SOME) {
    scanner->start = out->end;
  }
  return rc;
}

#define NPY_PARSE_ERROR "Error parsing numpy array: "
int parse_npy_header(sqlite3_vtab *pVTab, const unsigned char *header,
                     size_t headerLength,
                     enum VectorElementType *out_element_type,
                     int *fortran_order, size_t *numElements,
                     size_t *numDimensions) {

  struct NpyScanner scanner;
  struct NpyToken token;
  int rc;
  npy_scanner_init(&scanner, header, headerLength);

  if (npy_scanner_next(&scanner, &token) != VEC0_TOKEN_RESULT_SOME &&
      token.token_type != NPY_TOKEN_TYPE_LBRACE) {
    vtab_set_error(pVTab,
                   NPY_PARSE_ERROR "numpy header did not start with '{'");
    return SQLITE_ERROR;
  }
  while (1) {
    rc = npy_scanner_next(&scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME) {
      vtab_set_error(pVTab, NPY_PARSE_ERROR "expected key in numpy header");
      return SQLITE_ERROR;
    }

    if (token.token_type == NPY_TOKEN_TYPE_RBRACE) {
      break;
    }
    if (token.token_type != NPY_TOKEN_TYPE_STRING) {
      vtab_set_error(pVTab, NPY_PARSE_ERROR
                     "expected a string as key in numpy header");
      return SQLITE_ERROR;
    }
    unsigned char *key = token.start;

    rc = npy_scanner_next(&scanner, &token);
    if ((rc != VEC0_TOKEN_RESULT_SOME) ||
        (token.token_type != NPY_TOKEN_TYPE_COLON)) {
      vtab_set_error(pVTab, NPY_PARSE_ERROR
                     "expected a ':' after key in numpy header");
      return SQLITE_ERROR;
    }

    if (strncmp((char *)key, "'descr'", strlen("'descr'")) == 0) {
      rc = npy_scanner_next(&scanner, &token);
      if ((rc != VEC0_TOKEN_RESULT_SOME) ||
          (token.token_type != NPY_TOKEN_TYPE_STRING)) {
        vtab_set_error(pVTab, NPY_PARSE_ERROR
                       "expected a string value after 'descr' key");
        return SQLITE_ERROR;
      }
      if (strncmp((char *)token.start, "'<f4'", strlen("'<f4'")) != 0) {
        vtab_set_error(
            pVTab, NPY_PARSE_ERROR
            "Only '<f4' values are supported in sqlite-vec numpy functions");
        return SQLITE_ERROR;
      }
      *out_element_type = SQLITE_VEC_ELEMENT_TYPE_FLOAT32;
    } else if (strncmp((char *)key, "'fortran_order'",
                       strlen("'fortran_order'")) == 0) {
      rc = npy_scanner_next(&scanner, &token);
      if (rc != VEC0_TOKEN_RESULT_SOME ||
          token.token_type != NPY_TOKEN_TYPE_FALSE) {
        vtab_set_error(pVTab, NPY_PARSE_ERROR
                       "Only fortran_order = False is supported in sqlite-vec "
                       "numpy functions");
        return SQLITE_ERROR;
      }
      *fortran_order = 0;
    } else if (strncmp((char *)key, "'shape'", strlen("'shape'")) == 0) {
      // "(xxx, xxx)" OR (xxx,)
      size_t first;
      rc = npy_scanner_next(&scanner, &token);
      if ((rc != VEC0_TOKEN_RESULT_SOME) ||
          (token.token_type != NPY_TOKEN_TYPE_LPAREN)) {
        vtab_set_error(pVTab, NPY_PARSE_ERROR
                       "Expected left parenthesis '(' after shape key");
        return SQLITE_ERROR;
      }

      rc = npy_scanner_next(&scanner, &token);
      if ((rc != VEC0_TOKEN_RESULT_SOME) ||
          (token.token_type != NPY_TOKEN_TYPE_NUMBER)) {
        vtab_set_error(pVTab, NPY_PARSE_ERROR
                       "Expected an initial number in shape value");
        return SQLITE_ERROR;
      }
      first = strtol((char *)token.start, NULL, 10);

      rc = npy_scanner_next(&scanner, &token);
      if ((rc != VEC0_TOKEN_RESULT_SOME) ||
          (token.token_type != NPY_TOKEN_TYPE_COMMA)) {
        vtab_set_error(pVTab, NPY_PARSE_ERROR
                       "Expected comma after first shape value");
        return SQLITE_ERROR;
      }

      rc = npy_scanner_next(&scanner, &token);
      if (rc != VEC0_TOKEN_RESULT_SOME) {
        vtab_set_error(pVTab, NPY_PARSE_ERROR
                       "unexpected header EOF while parsing shape");
        return SQLITE_ERROR;
      }
      if (token.token_type == NPY_TOKEN_TYPE_NUMBER) {
        *numElements = first;
        *numDimensions = strtol((char *)token.start, NULL, 10);
        rc = npy_scanner_next(&scanner, &token);
        if ((rc != VEC0_TOKEN_RESULT_SOME) ||
            (token.token_type != NPY_TOKEN_TYPE_RPAREN)) {
          vtab_set_error(pVTab, NPY_PARSE_ERROR
                         "expected right parenthesis after shape value");
          return SQLITE_ERROR;
        }
      } else if (token.token_type == NPY_TOKEN_TYPE_RPAREN) {
        // '(0,)' means an empty array!
        *numElements = first ? 1 : 0;
        *numDimensions = first;
      } else {
        vtab_set_error(pVTab, NPY_PARSE_ERROR "unknown type in shape value");
        return SQLITE_ERROR;
      }
    } else {
      vtab_set_error(pVTab, NPY_PARSE_ERROR "unknown key in numpy header");
      return SQLITE_ERROR;
    }

    rc = npy_scanner_next(&scanner, &token);
    if ((rc != VEC0_TOKEN_RESULT_SOME) ||
        (token.token_type != NPY_TOKEN_TYPE_COMMA)) {
      vtab_set_error(pVTab, NPY_PARSE_ERROR "unknown extra token after value");
      return SQLITE_ERROR;
    }
  }

  return SQLITE_OK;
}

typedef struct vec_npy_each_vtab vec_npy_each_vtab;
struct vec_npy_each_vtab {
  sqlite3_vtab base;
};

typedef enum {
  VEC_NPY_EACH_INPUT_BUFFER,
  VEC_NPY_EACH_INPUT_FILE,
} vec_npy_each_input_type;

typedef struct vec_npy_each_cursor vec_npy_each_cursor;
struct vec_npy_each_cursor {
  sqlite3_vtab_cursor base;
  i64 iRowid;
  // sqlite-vec compatible type of vector
  enum VectorElementType elementType;
  // number of vectors in the npy array
  size_t nElements;
  // number of dimensions each vector has
  size_t nDimensions;

  vec_npy_each_input_type input_type;

  // when input_type == VEC_NPY_EACH_INPUT_BUFFER

  // Buffer containing the vector data, when reading from an in-memory buffer.
  // Size: nElements * nDimensions * element_size
  // Clean up with sqlite3_free() once complete
  void *vector;

  // when input_type == VEC_NPY_EACH_INPUT_FILE

  // Opened npy file, when reading from a file.
  // fclose() when complete.
#ifndef SQLITE_VEC_OMIT_FS
  FILE *file;
#endif

  // an in-memory buffer containing a portion of the npy array.
  // Used for faster reading, instead of calling fread a lot.
  // Will have a byte-size of fileBufferSize
  void *chunksBuffer;
  // size of allocated fileBuffer in bytes
  size_t chunksBufferSize;
  //// Maximum length of the buffer, in terms of number of vectors.
  size_t maxChunks;

  // Counter index of the current vector into of fileBuffer to yield.
  // Starts at 0 once fileBuffer is read, and iterates to bufferLength.
  // Resets to 0 once that "buffer" is yielded and a new one is read.
  size_t currentChunkIndex;
  size_t currentChunkSize;

  // 0 when there are still more elements to read/yield, 1 when complete.
  int eof;
};

static unsigned char NPY_MAGIC[6] = "\x93NUMPY";

#ifndef SQLITE_VEC_OMIT_FS
int parse_npy_file(sqlite3_vtab *pVTab, FILE *file, vec_npy_each_cursor *pCur) {
  int n;
  fseek(file, 0, SEEK_END);
  long fileSize = ftell(file);

  fseek(file, 0L, SEEK_SET);

  unsigned char header[10];
  n = fread(&header, sizeof(unsigned char), 10, file);
  if (n != 10) {
    vtab_set_error(pVTab, "numpy array file too short");
    return SQLITE_ERROR;
  }

  if (memcmp(NPY_MAGIC, header, sizeof(NPY_MAGIC)) != 0) {
    vtab_set_error(pVTab,
                   "numpy array file does not contain the 'magic' header");
    return SQLITE_ERROR;
  }

  u8 major = header[6];
  u8 minor = header[7];
  uint16_t headerLength = 0;
  memcpy(&headerLength, &header[8], sizeof(uint16_t));

  size_t totalHeaderLength = sizeof(NPY_MAGIC) + sizeof(major) + sizeof(minor) +
                             sizeof(headerLength) + headerLength;
  i32 dataSize = fileSize - totalHeaderLength;
  if (dataSize < 0) {
    vtab_set_error(pVTab, "numpy array file header length is invalid");
    return SQLITE_ERROR;
  }

  unsigned char *headerX = sqlite3_malloc(headerLength);
  if (headerLength && !headerX) {
    return SQLITE_NOMEM;
  }

  n = fread(headerX, sizeof(char), headerLength, file);
  if (n != headerLength) {
    sqlite3_free(headerX);
    vtab_set_error(pVTab, "numpy array file header length is invalid");
    return SQLITE_ERROR;
  }

  int fortran_order;
  enum VectorElementType element_type;
  size_t numElements;
  size_t numDimensions;
  int rc = parse_npy_header(pVTab, headerX, headerLength, &element_type,
                            &fortran_order, &numElements, &numDimensions);
  sqlite3_free(headerX);
  if (rc != SQLITE_OK) {
    // parse_npy_header already attackes an error emssage
    return rc;
  }

  i32 expectedDataSize =
      numElements * vector_byte_size(element_type, numDimensions);
  if (expectedDataSize != dataSize) {
    vtab_set_error(
        pVTab, "numpy array file error: Expected a data size of %d, found %d",
        expectedDataSize, dataSize);
    return SQLITE_ERROR;
  }

  pCur->maxChunks = 1024;
  pCur->chunksBufferSize =
      (vector_byte_size(element_type, numDimensions)) * pCur->maxChunks;
  pCur->chunksBuffer = sqlite3_malloc(pCur->chunksBufferSize);
  if (pCur->chunksBufferSize && !pCur->chunksBuffer) {
    return SQLITE_NOMEM;
  }

  pCur->currentChunkSize =
      fread(pCur->chunksBuffer, vector_byte_size(element_type, numDimensions),
            pCur->maxChunks, file);

  pCur->currentChunkIndex = 0;
  pCur->elementType = element_type;
  pCur->nElements = numElements;
  pCur->nDimensions = numDimensions;
  pCur->input_type = VEC_NPY_EACH_INPUT_FILE;

  pCur->eof = pCur->currentChunkSize == 0;
  pCur->file = file;
  return SQLITE_OK;
}
#endif

int parse_npy_buffer(sqlite3_vtab *pVTab, const unsigned char *buffer,
                     int bufferLength, void **data, size_t *numElements,
                     size_t *numDimensions,
                     enum VectorElementType *element_type) {

  if (bufferLength < 10) {
    // IMP: V03312_20150
    vtab_set_error(pVTab, "numpy array too short");
    return SQLITE_ERROR;
  }
  if (memcmp(NPY_MAGIC, buffer, sizeof(NPY_MAGIC)) != 0) {
    // V11954_28792
    vtab_set_error(pVTab, "numpy array does not contain the 'magic' header");
    return SQLITE_ERROR;
  }

  u8 major = buffer[6];
  u8 minor = buffer[7];
  uint16_t headerLength = 0;
  memcpy(&headerLength, &buffer[8], sizeof(uint16_t));

  i32 totalHeaderLength = sizeof(NPY_MAGIC) + sizeof(major) + sizeof(minor) +
                          sizeof(headerLength) + headerLength;
  i32 dataSize = bufferLength - totalHeaderLength;

  if (dataSize < 0) {
    vtab_set_error(pVTab, "numpy array header length is invalid");
    return SQLITE_ERROR;
  }

  const unsigned char *header = &buffer[10];
  int fortran_order;

  int rc = parse_npy_header(pVTab, header, headerLength, element_type,
                            &fortran_order, numElements, numDimensions);
  if (rc != SQLITE_OK) {
    return rc;
  }

  i32 expectedDataSize =
      (*numElements * vector_byte_size(*element_type, *numDimensions));
  if (expectedDataSize != dataSize) {
    vtab_set_error(pVTab,
                   "numpy array error: Expected a data size of %d, found %d",
                   expectedDataSize, dataSize);
    return SQLITE_ERROR;
  }

  *data = (void *)&buffer[totalHeaderLength];
  return SQLITE_OK;
}

static int vec_npy_eachConnect(sqlite3 *db, void *pAux, int argc,
                               const char *const *argv, sqlite3_vtab **ppVtab,
                               char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  vec_npy_each_vtab *pNew;
  int rc;

  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(vector, input hidden)");
#define VEC_NPY_EACH_COLUMN_VECTOR 0
#define VEC_NPY_EACH_COLUMN_INPUT 1
  if (rc == SQLITE_OK) {
    pNew = sqlite3_malloc(sizeof(*pNew));
    *ppVtab = (sqlite3_vtab *)pNew;
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
  }
  return rc;
}

static int vec_npy_eachDisconnect(sqlite3_vtab *pVtab) {
  vec_npy_each_vtab *p = (vec_npy_each_vtab *)pVtab;
  sqlite3_free(p);
  return SQLITE_OK;
}

static int vec_npy_eachOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec_npy_each_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec_npy_eachClose(sqlite3_vtab_cursor *cur) {
  vec_npy_each_cursor *pCur = (vec_npy_each_cursor *)cur;
#ifndef SQLITE_VEC_OMIT_FS
  if (pCur->file) {
    fclose(pCur->file);
    pCur->file = NULL;
  }
#endif
  if (pCur->chunksBuffer) {
    sqlite3_free(pCur->chunksBuffer);
    pCur->chunksBuffer = NULL;
  }
  if (pCur->vector) {
    pCur->vector = NULL;
  }
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int vec_npy_eachBestIndex(sqlite3_vtab *pVTab,
                                 sqlite3_index_info *pIdxInfo) {
  int hasInput;
  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *pCons = &pIdxInfo->aConstraint[i];
    // printf("i=%d iColumn=%d, op=%d, usable=%d\n", i, pCons->iColumn,
    // pCons->op, pCons->usable);
    switch (pCons->iColumn) {
    case VEC_NPY_EACH_COLUMN_INPUT: {
      if (pCons->op == SQLITE_INDEX_CONSTRAINT_EQ && pCons->usable) {
        hasInput = 1;
        pIdxInfo->aConstraintUsage[i].argvIndex = 1;
        pIdxInfo->aConstraintUsage[i].omit = 1;
      }
      break;
    }
    }
  }
  if (!hasInput) {
    pVTab->zErrMsg = sqlite3_mprintf("input argument is required");
    return SQLITE_ERROR;
  }

  pIdxInfo->estimatedCost = (double)100000;
  pIdxInfo->estimatedRows = 100000;

  return SQLITE_OK;
}

static int vec_npy_eachFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                              const char *idxStr, int argc,
                              sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  assert(argc == 1);
  int rc;

  vec_npy_each_cursor *pCur = (vec_npy_each_cursor *)pVtabCursor;

#ifndef SQLITE_VEC_OMIT_FS
  if (pCur->file) {
    fclose(pCur->file);
    pCur->file = NULL;
  }
#endif
  if (pCur->chunksBuffer) {
    sqlite3_free(pCur->chunksBuffer);
    pCur->chunksBuffer = NULL;
  }
  if (pCur->vector) {
    pCur->vector = NULL;
  }

#ifndef SQLITE_VEC_OMIT_FS
  struct VecNpyFile *f = NULL;
  if ((f = sqlite3_value_pointer(argv[0], SQLITE_VEC_NPY_FILE_NAME))) {
    FILE *file = fopen(f->path, "r");
    if (!file) {
      vtab_set_error(pVtabCursor->pVtab, "Could not open numpy file");
      return SQLITE_ERROR;
    }

    rc = parse_npy_file(pVtabCursor->pVtab, file, pCur);
    if (rc != SQLITE_OK) {
#ifndef SQLITE_VEC_OMIT_FS
      fclose(file);
#endif
      return rc;
    }

  } else
#endif
  {

    const unsigned char *input = sqlite3_value_blob(argv[0]);
    int inputLength = sqlite3_value_bytes(argv[0]);
    void *data;
    size_t numElements;
    size_t numDimensions;
    enum VectorElementType element_type;

    rc = parse_npy_buffer(pVtabCursor->pVtab, input, inputLength, &data,
                          &numElements, &numDimensions, &element_type);
    if (rc != SQLITE_OK) {
      return rc;
    }

    pCur->vector = data;
    pCur->elementType = element_type;
    pCur->nElements = numElements;
    pCur->nDimensions = numDimensions;
    pCur->input_type = VEC_NPY_EACH_INPUT_BUFFER;
  }

  pCur->iRowid = 0;
  return SQLITE_OK;
}

static int vec_npy_eachRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  vec_npy_each_cursor *pCur = (vec_npy_each_cursor *)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}

static int vec_npy_eachEof(sqlite3_vtab_cursor *cur) {
  vec_npy_each_cursor *pCur = (vec_npy_each_cursor *)cur;
  if (pCur->input_type == VEC_NPY_EACH_INPUT_BUFFER) {
    return (!pCur->nElements) || (size_t)pCur->iRowid >= pCur->nElements;
  }
  return pCur->eof;
}

static int vec_npy_eachNext(sqlite3_vtab_cursor *cur) {
  vec_npy_each_cursor *pCur = (vec_npy_each_cursor *)cur;
  pCur->iRowid++;
  if (pCur->input_type == VEC_NPY_EACH_INPUT_BUFFER) {
    return SQLITE_OK;
  }

#ifndef SQLITE_VEC_OMIT_FS
  // else: input is a file
  pCur->currentChunkIndex++;
  if (pCur->currentChunkIndex >= pCur->currentChunkSize) {
    pCur->currentChunkSize =
        fread(pCur->chunksBuffer,
              vector_byte_size(pCur->elementType, pCur->nDimensions),
              pCur->maxChunks, pCur->file);
    if (!pCur->currentChunkSize) {
      pCur->eof = 1;
    }
    pCur->currentChunkIndex = 0;
  }
#endif
  return SQLITE_OK;
}

static int vec_npy_eachColumnBuffer(vec_npy_each_cursor *pCur,
                                    sqlite3_context *context, int i) {
  switch (i) {
  case VEC_NPY_EACH_COLUMN_VECTOR: {
    sqlite3_result_subtype(context, pCur->elementType);
    switch (pCur->elementType) {
    case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
      sqlite3_result_blob(
          context,
          &((unsigned char *)
                pCur->vector)[pCur->iRowid * pCur->nDimensions * sizeof(f32)],
          pCur->nDimensions * sizeof(f32), SQLITE_TRANSIENT);

      break;
    }
    case SQLITE_VEC_ELEMENT_TYPE_INT8:
    case SQLITE_VEC_ELEMENT_TYPE_BIT: {
      // https://github.com/asg017/sqlite-vec/issues/42
      sqlite3_result_error(context,
                           "vec_npy_each only supports float32 vectors", -1);
      break;
    }
    }

    break;
  }
  }
  return SQLITE_OK;
}
static int vec_npy_eachColumnFile(vec_npy_each_cursor *pCur,
                                  sqlite3_context *context, int i) {
  switch (i) {
  case VEC_NPY_EACH_COLUMN_VECTOR: {
    switch (pCur->elementType) {
    case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
      sqlite3_result_blob(
          context,
          &((unsigned char *)
                pCur->chunksBuffer)[pCur->currentChunkIndex *
                                    pCur->nDimensions * sizeof(f32)],
          pCur->nDimensions * sizeof(f32), SQLITE_TRANSIENT);
      break;
    }
    case SQLITE_VEC_ELEMENT_TYPE_INT8:
    case SQLITE_VEC_ELEMENT_TYPE_BIT: {
      // https://github.com/asg017/sqlite-vec/issues/42
      sqlite3_result_error(context,
                           "vec_npy_each only supports float32 vectors", -1);
      break;
    }
    }
    break;
  }
  }
  return SQLITE_OK;
}
static int vec_npy_eachColumn(sqlite3_vtab_cursor *cur,
                              sqlite3_context *context, int i) {
  vec_npy_each_cursor *pCur = (vec_npy_each_cursor *)cur;
  switch (pCur->input_type) {
  case VEC_NPY_EACH_INPUT_BUFFER:
    return vec_npy_eachColumnBuffer(pCur, context, i);
  case VEC_NPY_EACH_INPUT_FILE:
    return vec_npy_eachColumnFile(pCur, context, i);
  }
  return SQLITE_ERROR;
}

static sqlite3_module vec_npy_eachModule = {
    /* iVersion    */ 0,
    /* xCreate     */ 0,
    /* xConnect    */ vec_npy_eachConnect,
    /* xBestIndex  */ vec_npy_eachBestIndex,
    /* xDisconnect */ vec_npy_eachDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ vec_npy_eachOpen,
    /* xClose      */ vec_npy_eachClose,
    /* xFilter     */ vec_npy_eachFilter,
    /* xNext       */ vec_npy_eachNext,
    /* xEof        */ vec_npy_eachEof,
    /* xColumn     */ vec_npy_eachColumn,
    /* xRowid      */ vec_npy_eachRowid,
    /* xUpdate     */ 0,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity  */ 0,
#endif
};

#pragma endregion

#pragma region vec0 virtual table

#define VEC0_COLUMN_ID 0
#define VEC0_COLUMN_USERN_START 1
#define VEC0_COLUMN_OFFSET_DISTANCE 1
#define VEC0_COLUMN_OFFSET_K 2

#define VEC0_SHADOW_INFO_NAME "\"%w\".\"%w_info\""

#define VEC0_SHADOW_CHUNKS_NAME "\"%w\".\"%w_chunks\""
/// 1) schema, 2) original vtab table name
#define VEC0_SHADOW_CHUNKS_CREATE                                              \
  "CREATE TABLE " VEC0_SHADOW_CHUNKS_NAME "("                                  \
  "chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,"                                \
  "size INTEGER NOT NULL,"                                                     \
  "validity BLOB NOT NULL,"                                                    \
  "rowids BLOB NOT NULL"                                                       \
  ");"

#define VEC0_SHADOW_ROWIDS_NAME "\"%w\".\"%w_rowids\""
/// 1) schema, 2) original vtab table name
#define VEC0_SHADOW_ROWIDS_CREATE_BASIC                                        \
  "CREATE TABLE " VEC0_SHADOW_ROWIDS_NAME "("                                  \
  "rowid INTEGER PRIMARY KEY AUTOINCREMENT,"                                   \
  "id,"                                                                        \
  "chunk_id INTEGER,"                                                          \
  "chunk_offset INTEGER"                                                       \
  ");"

// vec0 tables with a text primary keys are still backed by int64 primary keys,
// since a fixed-length rowid is required for vec0 chunks. But we add a new 'id
// text unique' column to emulate a text primary key interface.
#define VEC0_SHADOW_ROWIDS_CREATE_PK_TEXT                                      \
  "CREATE TABLE " VEC0_SHADOW_ROWIDS_NAME "("                                  \
  "rowid INTEGER PRIMARY KEY AUTOINCREMENT,"                                   \
  "id TEXT UNIQUE NOT NULL,"                                                   \
  "chunk_id INTEGER,"                                                          \
  "chunk_offset INTEGER"                                                       \
  ");"

/// 1) schema, 2) original vtab table name
#define VEC0_SHADOW_VECTOR_N_NAME "\"%w\".\"%w_vector_chunks%02d\""

/// 1) schema, 2) original vtab table name
#define VEC0_SHADOW_VECTOR_N_CREATE                                            \
  "CREATE TABLE " VEC0_SHADOW_VECTOR_N_NAME "("                                \
  "rowid PRIMARY KEY,"                                                         \
  "vectors BLOB NOT NULL"                                                      \
  ");"

#define VEC0_SHADOW_AUXILIARY_NAME "\"%w\".\"%w_auxiliary\""

#define VEC0_SHADOW_METADATA_N_NAME "\"%w\".\"%w_metadatachunks%02d\""
#define VEC0_SHADOW_METADATA_TEXT_DATA_NAME "\"%w\".\"%w_metadatatext%02d\""

#define VEC_INTERAL_ERROR "Internal sqlite-vec error: "
#define REPORT_URL "https://github.com/asg017/sqlite-vec/issues/new"

typedef struct vec0_vtab vec0_vtab;

#define VEC0_MAX_VECTOR_COLUMNS   16
#define VEC0_MAX_PARTITION_COLUMNS 4
#define VEC0_MAX_AUXILIARY_COLUMNS 16
#define VEC0_MAX_METADATA_COLUMNS 16

#define SQLITE_VEC_VEC0_MAX_DIMENSIONS 8192
#define VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH 16
#define VEC0_METADATA_TEXT_VIEW_DATA_LENGTH 12

typedef enum {
  // vector column, ie "contents_embedding float[1024]"
  SQLITE_VEC0_USER_COLUMN_KIND_VECTOR = 1,

  // partition key column, ie "user_id integer partition key"
  SQLITE_VEC0_USER_COLUMN_KIND_PARTITION = 2,

  //
  SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY = 3,

  // metadata column that can be filtered, ie "genre text"
  SQLITE_VEC0_USER_COLUMN_KIND_METADATA = 4,
} vec0_user_column_kind;

struct vec0_vtab {
  sqlite3_vtab base;

  // the SQLite connection of the host database
  sqlite3 *db;

  // True if the primary key of the vec0 table has a column type TEXT.
  // Will change the schema of the _rowids table, and insert/query logic.
  int pkIsText;

  // number of defined vector columns.
  int numVectorColumns;

  // number of defined PARTITION KEY columns.
  int numPartitionColumns;

  // number of defined auxiliary columns
  int numAuxiliaryColumns;

  // number of defined metadata columns
  int numMetadataColumns;


  // Name of the schema the table exists on.
  // Must be freed with sqlite3_free()
  char *schemaName;

  // Name of the table the table exists on.
  // Must be freed with sqlite3_free()
  char *tableName;

  // Name of the _rowids shadow table.
  // Must be freed with sqlite3_free()
  char *shadowRowidsName;

  // Name of the _chunks shadow table.
  // Must be freed with sqlite3_free()
  char *shadowChunksName;

  // contains enum vec0_user_column_kind values for up to
  // numVectorColumns + numPartitionColumns entries
  vec0_user_column_kind user_column_kinds[VEC0_MAX_VECTOR_COLUMNS + VEC0_MAX_PARTITION_COLUMNS + VEC0_MAX_AUXILIARY_COLUMNS + VEC0_MAX_METADATA_COLUMNS];

  uint8_t user_column_idxs[VEC0_MAX_VECTOR_COLUMNS + VEC0_MAX_PARTITION_COLUMNS + VEC0_MAX_AUXILIARY_COLUMNS + VEC0_MAX_METADATA_COLUMNS];


  // Name of all the vector chunk shadow tables.
  // Ex '_vector_chunks00'
  // Only the first numVectorColumns entries will be available.
  // The first numVectorColumns entries must be freed with sqlite3_free()
  char *shadowVectorChunksNames[VEC0_MAX_VECTOR_COLUMNS];

  // Name of all metadata chunk shadow tables, ie `_metadatachunks00`
  // Only the first numMetadataColumns entries will be available.
  // The first numMetadataColumns entries must be freed with sqlite3_free()
  char *shadowMetadataChunksNames[VEC0_MAX_METADATA_COLUMNS];

  struct VectorColumnDefinition vector_columns[VEC0_MAX_VECTOR_COLUMNS];
  struct Vec0PartitionColumnDefinition paritition_columns[VEC0_MAX_PARTITION_COLUMNS];
  struct Vec0AuxiliaryColumnDefinition auxiliary_columns[VEC0_MAX_AUXILIARY_COLUMNS];
  struct Vec0MetadataColumnDefinition metadata_columns[VEC0_MAX_METADATA_COLUMNS];

  int chunk_size;

  // select latest chunk from _chunks, getting chunk_id
  sqlite3_stmt *stmtLatestChunk;

  /**
   * Statement to insert a row into the _rowids table, with a rowid.
   * Parameters:
   *    1: int64, rowid to insert
   * Result columns: none
   * SQL: "INSERT INTO _rowids(rowid) VALUES (?)"
   *
   * Must be cleaned up with sqlite3_finalize().
   */
  sqlite3_stmt *stmtRowidsInsertRowid;

  /**
   * Statement to insert a row into the _rowids table, with an id.
   * The id column isn't a tradition primary key, but instead a unique
   * column to handle "text primary key" vec0 tables. The true int64 rowid
   * can be retrieved after inserting with sqlite3_last_rowid().
   *
   * Parameters:
   *    1: text or null, id to insert
   * Result columns: none
   *
   * Must be cleaned up with sqlite3_finalize().
   */
  sqlite3_stmt *stmtRowidsInsertId;

  /**
   * Statement to update the "position" columns chunk_id and chunk_offset for
   * a given _rowids row. Used when the "next available" chunk position is found
   * for a vector.
   *
   * Parameters:
   *    1: int64, chunk_id value
   *    2: int64, chunk_offset value
   *    3: int64, rowid value
   * Result columns: none
   *
   * Must be cleaned up with sqlite3_finalize().
   */
  sqlite3_stmt *stmtRowidsUpdatePosition;

  /**
   * Statement to quickly find the chunk_id + chunk_offset of a given row.
   * Parameters:
   *  1: rowid of the row/vector to lookup
   * Result columns:
   *  0: chunk_id (i64)
   *  1: chunk_offset (i64)
   * SQL: "SELECT id, chunk_id, chunk_offset FROM _rowids WHERE rowid = ?""
   *
   * Must be cleaned up with sqlite3_finalize().
   */
  sqlite3_stmt *stmtRowidsGetChunkPosition;
};

/**
 * @brief Finalize all the sqlite3_stmt members in a vec0_vtab.
 *
 * @param p vec0_vtab pointer
 */
void vec0_free_resources(vec0_vtab *p) {
  sqlite3_finalize(p->stmtLatestChunk);
  p->stmtLatestChunk = NULL;
  sqlite3_finalize(p->stmtRowidsInsertRowid);
  p->stmtRowidsInsertRowid = NULL;
  sqlite3_finalize(p->stmtRowidsInsertId);
  p->stmtRowidsInsertId = NULL;
  sqlite3_finalize(p->stmtRowidsUpdatePosition);
  p->stmtRowidsUpdatePosition = NULL;
  sqlite3_finalize(p->stmtRowidsGetChunkPosition);
  p->stmtRowidsGetChunkPosition = NULL;
}

/**
 * @brief Free all memory and sqlite3_stmt members of a vec0_vtab
 *
 * @param p vec0_vtab pointer
 */
void vec0_free(vec0_vtab *p) {
  vec0_free_resources(p);

  sqlite3_free(p->schemaName);
  p->schemaName = NULL;
  sqlite3_free(p->tableName);
  p->tableName = NULL;
  sqlite3_free(p->shadowChunksName);
  p->shadowChunksName = NULL;
  sqlite3_free(p->shadowRowidsName);
  p->shadowRowidsName = NULL;

  for (int i = 0; i < p->numVectorColumns; i++) {
    sqlite3_free(p->shadowVectorChunksNames[i]);
    p->shadowVectorChunksNames[i] = NULL;

    sqlite3_free(p->vector_columns[i].name);
    p->vector_columns[i].name = NULL;
  }
}

int vec0_num_defined_user_columns(vec0_vtab *p) {
  return p->numVectorColumns + p->numPartitionColumns + p->numAuxiliaryColumns + p->numMetadataColumns;
}

/**
 * @brief Returns the index of the distance hidden column for the given vec0
 * table.
 *
 * @param p vec0 table
 * @return int
 */
int vec0_column_distance_idx(vec0_vtab *p) {
  return VEC0_COLUMN_USERN_START + (vec0_num_defined_user_columns(p) - 1) +
         VEC0_COLUMN_OFFSET_DISTANCE;
}

/**
 * @brief Returns the index of the k hidden column for the given vec0 table.
 *
 * @param p vec0 table
 * @return int k column index
 */
int vec0_column_k_idx(vec0_vtab *p) {
  return VEC0_COLUMN_USERN_START + (vec0_num_defined_user_columns(p) - 1) +
         VEC0_COLUMN_OFFSET_K;
}

/**
 * Returns 1 if the given column-based index is a valid vector column,
 * 0 otherwise.
 */
int vec0_column_idx_is_vector(vec0_vtab *pVtab, int column_idx) {
  return column_idx >= VEC0_COLUMN_USERN_START &&
         column_idx <= (VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(pVtab) - 1) &&
         pVtab->user_column_kinds[column_idx - VEC0_COLUMN_USERN_START] == SQLITE_VEC0_USER_COLUMN_KIND_VECTOR;
}

/**
 * Returns the vector index of the given user column index.
 * ONLY call if validated with vec0_column_idx_is_vector before
 */
int vec0_column_idx_to_vector_idx(vec0_vtab *pVtab, int column_idx) {
  UNUSED_PARAMETER(pVtab);
  return pVtab->user_column_idxs[column_idx - VEC0_COLUMN_USERN_START];
}
/**
 * Returns 1 if the given column-based index is a "partition key" column,
 * 0 otherwise.
 */
int vec0_column_idx_is_partition(vec0_vtab *pVtab, int column_idx) {
  return column_idx >= VEC0_COLUMN_USERN_START &&
         column_idx <= (VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(pVtab) - 1) &&
         pVtab->user_column_kinds[column_idx - VEC0_COLUMN_USERN_START] == SQLITE_VEC0_USER_COLUMN_KIND_PARTITION;
}

/**
 * Returns the partition column index of the given user column index.
 * ONLY call if validated with vec0_column_idx_is_vector before
 */
int vec0_column_idx_to_partition_idx(vec0_vtab *pVtab, int column_idx) {
  UNUSED_PARAMETER(pVtab);
  return pVtab->user_column_idxs[column_idx - VEC0_COLUMN_USERN_START];
}

/**
 * Returns 1 if the given column-based index is a auxiliary column,
 * 0 otherwise.
 */
int vec0_column_idx_is_auxiliary(vec0_vtab *pVtab, int column_idx) {
  return column_idx >= VEC0_COLUMN_USERN_START &&
         column_idx <= (VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(pVtab) - 1) &&
         pVtab->user_column_kinds[column_idx - VEC0_COLUMN_USERN_START] == SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY;
}

/**
 * Returns the auxiliary column index of the given user column index.
 * ONLY call if validated with vec0_column_idx_to_partition_idx before
 */
int vec0_column_idx_to_auxiliary_idx(vec0_vtab *pVtab, int column_idx) {
  UNUSED_PARAMETER(pVtab);
  return pVtab->user_column_idxs[column_idx - VEC0_COLUMN_USERN_START];
}

/**
 * Returns 1 if the given column-based index is a metadata column,
 * 0 otherwise.
 */
int vec0_column_idx_is_metadata(vec0_vtab *pVtab, int column_idx) {
  return column_idx >= VEC0_COLUMN_USERN_START &&
         column_idx <= (VEC0_COLUMN_USERN_START + vec0_num_defined_user_columns(pVtab) - 1) &&
         pVtab->user_column_kinds[column_idx - VEC0_COLUMN_USERN_START] == SQLITE_VEC0_USER_COLUMN_KIND_METADATA;
}

/**
 * Returns the metadata column index of the given user column index.
 * ONLY call if validated with vec0_column_idx_is_metadata before
 */
int vec0_column_idx_to_metadata_idx(vec0_vtab *pVtab, int column_idx) {
  UNUSED_PARAMETER(pVtab);
  return pVtab->user_column_idxs[column_idx - VEC0_COLUMN_USERN_START];
}

/**
 * @brief Retrieve the chunk_id, chunk_offset, and possible "id" value
 * of a vec0_vtab row with the provided rowid
 *
 * @param p vec0_vtab
 * @param rowid the rowid of the row to query
 * @param id output, optional sqlite3_value to provide the id.
 *            Useful for text PK rows. Must be freed with sqlite3_value_free()
 * @param chunk_id output, the chunk_id the row belongs to
 * @param chunk_offset  output, the offset within the chunk the row belongs to
 * @return SQLITE_ROW on success, error code otherwise. SQLITE_EMPTY if row DNE
 */
int vec0_get_chunk_position(vec0_vtab *p, i64 rowid, sqlite3_value **id,
                            i64 *chunk_id, i64 *chunk_offset) {
  int rc;

  if (!p->stmtRowidsGetChunkPosition) {
    const char *zSql =
        sqlite3_mprintf("SELECT id, chunk_id, chunk_offset "
                        "FROM " VEC0_SHADOW_ROWIDS_NAME " WHERE rowid = ?",
                        p->schemaName, p->tableName);
    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtRowidsGetChunkPosition, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      vtab_set_error(
          &p->base, VEC_INTERAL_ERROR
          "could not initialize 'rowids get chunk position' statement");
      goto cleanup;
    }
  }

  sqlite3_bind_int64(p->stmtRowidsGetChunkPosition, 1, rowid);
  rc = sqlite3_step(p->stmtRowidsGetChunkPosition);
  // special case: when no results, return SQLITE_EMPTY to convey "that chunk
  // position doesnt exist"
  if (rc == SQLITE_DONE) {
    rc = SQLITE_EMPTY;
    goto cleanup;
  }
  if (rc != SQLITE_ROW) {
    goto cleanup;
  }

  if (id) {
    sqlite3_value *value =
        sqlite3_column_value(p->stmtRowidsGetChunkPosition, 0);
    *id = sqlite3_value_dup(value);
    if (!*id) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
  }

  if (chunk_id) {
    *chunk_id = sqlite3_column_int64(p->stmtRowidsGetChunkPosition, 1);
  }
  if (chunk_offset) {
    *chunk_offset = sqlite3_column_int64(p->stmtRowidsGetChunkPosition, 2);
  }

  rc = SQLITE_OK;

cleanup:
  sqlite3_reset(p->stmtRowidsGetChunkPosition);
  sqlite3_clear_bindings(p->stmtRowidsGetChunkPosition);
  return rc;
}

/**
 * @brief Return the id value from the _rowids table where _rowids.rowid =
 * rowid.
 *
 * @param pVtab: vec0 table to query
 * @param rowid: rowid of the row to query.
 * @param out: A dup'ed sqlite3_value of the id column. Might be null.
 *                         Must be cleaned up with sqlite3_value_free().
 * @returns SQLITE_OK on success, error code on failure
 */
int vec0_get_id_value_from_rowid(vec0_vtab *pVtab, i64 rowid,
                                 sqlite3_value **out) {
  // PERF: different strategy than get_chunk_position?
  return vec0_get_chunk_position((vec0_vtab *)pVtab, rowid, out, NULL, NULL);
}

int vec0_rowid_from_id(vec0_vtab *p, sqlite3_value *valueId, i64 *rowid) {
  sqlite3_stmt *stmt = NULL;
  int rc;
  char *zSql;
  zSql = sqlite3_mprintf("SELECT rowid"
                         " FROM " VEC0_SHADOW_ROWIDS_NAME " WHERE id = ?",
                         p->schemaName, p->tableName);
  if (!zSql) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }
  sqlite3_bind_value(stmt, 1, valueId);
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_EMPTY;
    goto cleanup;
  }
  if (rc != SQLITE_ROW) {
    goto cleanup;
  }
  *rowid = sqlite3_column_int64(stmt, 0);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_DONE) {
    goto cleanup;
  }

  rc = SQLITE_OK;

cleanup:
  sqlite3_finalize(stmt);
  return rc;
}

int vec0_result_id(vec0_vtab *p, sqlite3_context *context, i64 rowid) {
  if (!p->pkIsText) {
    sqlite3_result_int64(context, rowid);
    return SQLITE_OK;
  }
  sqlite3_value *valueId;
  int rc = vec0_get_id_value_from_rowid(p, rowid, &valueId);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (!valueId) {
    sqlite3_result_error_nomem(context);
  } else {
    sqlite3_result_value(context, valueId);
    sqlite3_value_free(valueId);
  }
  return SQLITE_OK;
}

/**
 * @brief
 *
 * @param pVtab: virtual table to query
 * @param rowid: row to lookup
 * @param vector_column_idx: which vector column to query
 * @param outVector: Output pointer to the vector buffer.
 *                    Must be sqlite3_free()'ed.
 * @param outVectorSize: Pointer to a int where the size of outVector
 *                       will be stored.
 * @return int SQLITE_OK on success.
 */
int vec0_get_vector_data(vec0_vtab *pVtab, i64 rowid, int vector_column_idx,
                         void **outVector, int *outVectorSize) {
  vec0_vtab *p = pVtab;
  int rc, brc;
  i64 chunk_id;
  i64 chunk_offset;
  size_t size;
  void *buf = NULL;
  int blobOffset;
  sqlite3_blob *vectorBlob = NULL;
  assert((vector_column_idx >= 0) &&
         (vector_column_idx < pVtab->numVectorColumns));

  rc = vec0_get_chunk_position(pVtab, rowid, NULL, &chunk_id, &chunk_offset);
  if (rc == SQLITE_EMPTY) {
    vtab_set_error(&pVtab->base, "Could not find a row with rowid %lld", rowid);
    goto cleanup;
  }
  if (rc != SQLITE_OK) {
    goto cleanup;
  }

  rc = sqlite3_blob_open(p->db, p->schemaName,
                         p->shadowVectorChunksNames[vector_column_idx],
                         "vectors", chunk_id, 0, &vectorBlob);

  if (rc != SQLITE_OK) {
    vtab_set_error(&pVtab->base,
                   "Could not fetch vector data for %lld, opening blob failed",
                   rowid);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  size = vector_column_byte_size(pVtab->vector_columns[vector_column_idx]);
  blobOffset = chunk_offset * size;

  buf = sqlite3_malloc(size);
  if (!buf) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  rc = sqlite3_blob_read(vectorBlob, buf, size, blobOffset);
  if (rc != SQLITE_OK) {
    sqlite3_free(buf);
    buf = NULL;
    vtab_set_error(
        &pVtab->base,
        "Could not fetch vector data for %lld, reading from blob failed",
        rowid);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  *outVector = buf;
  if (outVectorSize) {
    *outVectorSize = size;
  }
  rc = SQLITE_OK;

cleanup:
  brc = sqlite3_blob_close(vectorBlob);
  if ((rc == SQLITE_OK) && (brc != SQLITE_OK)) {
    vtab_set_error(
        &p->base, VEC_INTERAL_ERROR
        "unknown error, could not close vector blob, please file an issue");
    return brc;
  }

  return rc;
}

/**
 * @brief Retrieve the sqlite3_value of the i'th partition value for the given row.
 *
 * @param pVtab - the vec0_vtab in questions
 * @param rowid - rowid of target row
 * @param partition_idx - which partition column to retrieve
 * @param outValue - output sqlite3_value
 * @return int - SQLITE_OK on success, otherwise error code
 */
int vec0_get_partition_value_for_rowid(vec0_vtab *pVtab, i64 rowid, int partition_idx, sqlite3_value ** outValue) {
  int rc;
  i64 chunk_id;
  i64 chunk_offset;
  rc = vec0_get_chunk_position(pVtab, rowid, NULL, &chunk_id, &chunk_offset);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_stmt * stmt = NULL;
  char * zSql = sqlite3_mprintf("SELECT partition%02d FROM " VEC0_SHADOW_CHUNKS_NAME " WHERE chunk_id = ?", partition_idx, pVtab->schemaName, pVtab->tableName);
  if(!zSql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(pVtab->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_bind_int64(stmt, 1, chunk_id);
  rc = sqlite3_step(stmt);
  if(rc != SQLITE_ROW) {
    rc = SQLITE_ERROR;
    goto done;
  }
  *outValue = sqlite3_value_dup(sqlite3_column_value(stmt, 0));
  if(!*outValue) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = SQLITE_OK;

  done:
    sqlite3_finalize(stmt);
    return rc;

}

/**
 * @brief Get the value of an auxiliary column for the given rowid
 *
 * @param pVtab vec0_vtab
 * @param rowid the rowid of the row to lookup
 * @param auxiliary_idx aux index of the column we care about
 * @param outValue Output sqlite3_value to store
 * @return int SQLITE_OK on success, error code otherwise
 */
int vec0_get_auxiliary_value_for_rowid(vec0_vtab *pVtab, i64 rowid, int auxiliary_idx, sqlite3_value ** outValue) {
  int rc;
  sqlite3_stmt * stmt = NULL;
  char * zSql = sqlite3_mprintf("SELECT value%02d FROM " VEC0_SHADOW_AUXILIARY_NAME " WHERE rowid = ?", auxiliary_idx, pVtab->schemaName, pVtab->tableName);
  if(!zSql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(pVtab->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  if(rc != SQLITE_ROW) {
    rc = SQLITE_ERROR;
    goto done;
  }
  *outValue = sqlite3_value_dup(sqlite3_column_value(stmt, 0));
  if(!*outValue) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = SQLITE_OK;

  done:
    sqlite3_finalize(stmt);
    return rc;
}

/**
 * @brief Result the given metadata value for the given row and metadata column index.
 * Will traverse the metadatachunksNN table with BLOB I/0 for the given rowid.
 *
 * @param p
 * @param rowid
 * @param metadata_idx
 * @param context
 * @return int
 */
int vec0_result_metadata_value_for_rowid(vec0_vtab *p, i64 rowid, int metadata_idx, sqlite3_context * context) {
  int rc;
  i64 chunk_id;
  i64 chunk_offset;
  rc = vec0_get_chunk_position(p, rowid, NULL, &chunk_id, &chunk_offset);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_blob * blobValue;
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowMetadataChunksNames[metadata_idx], "data", chunk_id, 0, &blobValue);
  if(rc != SQLITE_OK) {
    return rc;
  }

  switch(p->metadata_columns[metadata_idx].kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      u8 block;
      rc = sqlite3_blob_read(blobValue, &block, sizeof(block), chunk_offset / CHAR_BIT);
      if(rc != SQLITE_OK) {
        goto done;
      }
      int value = block >> ((chunk_offset % CHAR_BIT)) & 1;
      sqlite3_result_int(context, value);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      i64 value;
      rc = sqlite3_blob_read(blobValue, &value, sizeof(value), chunk_offset * sizeof(i64));
      if(rc != SQLITE_OK) {
        goto done;
      }
      sqlite3_result_int64(context, value);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      double value;
      rc = sqlite3_blob_read(blobValue, &value, sizeof(value), chunk_offset * sizeof(double));
      if(rc != SQLITE_OK) {
        goto done;
      }
      sqlite3_result_double(context, value);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      u8 view[VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
      rc = sqlite3_blob_read(blobValue, &view, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH, chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(rc != SQLITE_OK) {
        goto done;
      }
      int length = ((int *)view)[0];
      if(length <= VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
        sqlite3_result_text(context, (const char*) (view + 4), length, SQLITE_TRANSIENT);
      }
      else {
        sqlite3_stmt * stmt;
        const char * zSql = sqlite3_mprintf("SELECT data FROM " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " WHERE rowid = ?", p->schemaName, p->tableName, metadata_idx);
        if(!zSql) {
          rc = SQLITE_ERROR;
          goto done;
        }
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
        sqlite3_free((void *) zSql);
        if(rc != SQLITE_OK) {
          goto done;
        }
        sqlite3_bind_int64(stmt, 1, rowid);
        rc = sqlite3_step(stmt);
        if(rc != SQLITE_ROW) {
          sqlite3_finalize(stmt);
          rc = SQLITE_ERROR;
          goto done;
        }
        sqlite3_result_value(context, sqlite3_column_value(stmt, 0));
        sqlite3_finalize(stmt);
        rc = SQLITE_OK;
      }
      break;
    }
  }
  done:
    // blobValue is read-only, will not fail on close
    sqlite3_blob_close(blobValue);
    return rc;

}

int vec0_get_latest_chunk_rowid(vec0_vtab *p, i64 *chunk_rowid, sqlite3_value ** partitionKeyValues) {
  int rc;
  const char *zSql;
  // lazy initialize stmtLatestChunk when needed. May be cleared during xSync()
  if (!p->stmtLatestChunk) {
    if(p->numPartitionColumns > 0) {
      sqlite3_str * s = sqlite3_str_new(NULL);
      sqlite3_str_appendf(s, "SELECT max(rowid) FROM " VEC0_SHADOW_CHUNKS_NAME " WHERE ",
                           p->schemaName, p->tableName);

      for(int i = 0; i < p->numPartitionColumns; i++) {
        if(i != 0) {
          sqlite3_str_appendall(s, " AND ");
        }
        sqlite3_str_appendf(s, " partition%02d = ? ", i);
      }
      zSql = sqlite3_str_finish(s);
    }else {
      zSql = sqlite3_mprintf("SELECT max(rowid) FROM " VEC0_SHADOW_CHUNKS_NAME,
                           p->schemaName, p->tableName);
    }

    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtLatestChunk, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      // IMP: V21406_05476
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "could not initialize 'latest chunk' statement");
      goto cleanup;
    }
  }

  for(int i = 0; i < p->numPartitionColumns; i++) {
    sqlite3_bind_value(p->stmtLatestChunk, i+1, (partitionKeyValues[i]));
  }

  rc = sqlite3_step(p->stmtLatestChunk);
  if (rc != SQLITE_ROW) {
    // IMP: V31559_15629
    vtab_set_error(&p->base, VEC_INTERAL_ERROR "Could not find latest chunk");
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if(sqlite3_column_type(p->stmtLatestChunk, 0) == SQLITE_NULL){
    rc = SQLITE_EMPTY;
    goto cleanup;
  }
  *chunk_rowid = sqlite3_column_int64(p->stmtLatestChunk, 0);
  rc = sqlite3_step(p->stmtLatestChunk);
  if (rc != SQLITE_DONE) {
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "unknown result code when closing out stmtLatestChunk. "
                   "Please file an issue: " REPORT_URL,
                   p->schemaName, p->shadowChunksName);
    goto cleanup;
  }
  rc = SQLITE_OK;

cleanup:
  if (p->stmtLatestChunk) {
    sqlite3_reset(p->stmtLatestChunk);
    sqlite3_clear_bindings(p->stmtLatestChunk);
  }
  return rc;
}

int vec0_rowids_insert_rowid(vec0_vtab *p, i64 rowid) {
  int rc = SQLITE_OK;
  int entered = 0;
  UNUSED_PARAMETER(entered); // temporary
  if (!p->stmtRowidsInsertRowid) {
    const char *zSql =
        sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_ROWIDS_NAME "(rowid)"
                        "VALUES (?);",
                        p->schemaName, p->tableName);
    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtRowidsInsertRowid, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "could not initialize 'insert rowids' statement");
      goto cleanup;
    }
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_enter) {
    sqlite3_mutex_enter(sqlite3_db_mutex(p->db));
    entered = 1;
  }
#endif
  sqlite3_bind_int64(p->stmtRowidsInsertRowid, 1, rowid);
  rc = sqlite3_step(p->stmtRowidsInsertRowid);

  if (rc != SQLITE_DONE) {
    if (sqlite3_extended_errcode(p->db) == SQLITE_CONSTRAINT_PRIMARYKEY) {
      // IMP: V17090_01160
      vtab_set_error(&p->base, "UNIQUE constraint failed on %s primary key",
                     p->tableName);
    } else {
      // IMP: V04679_21517
      vtab_set_error(&p->base,
                     "Error inserting rowid into rowids shadow table: %s",
                     sqlite3_errmsg(sqlite3_db_handle(p->stmtRowidsInsertId)));
    }
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  rc = SQLITE_OK;

cleanup:
  if (p->stmtRowidsInsertRowid) {
    sqlite3_reset(p->stmtRowidsInsertRowid);
    sqlite3_clear_bindings(p->stmtRowidsInsertRowid);
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_leave && entered) {
    sqlite3_mutex_leave(sqlite3_db_mutex(p->db));
  }
#endif
  return rc;
}

int vec0_rowids_insert_id(vec0_vtab *p, sqlite3_value *idValue, i64 *rowid) {
  int rc = SQLITE_OK;
  int entered = 0;
  UNUSED_PARAMETER(entered); // temporary
  if (!p->stmtRowidsInsertId) {
    const char *zSql =
        sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_ROWIDS_NAME "(id)"
                        "VALUES (?);",
                        p->schemaName, p->tableName);
    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto complete;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtRowidsInsertId, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "could not initialize 'insert rowids id' statement");
      goto complete;
    }
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_enter) {
    sqlite3_mutex_enter(sqlite3_db_mutex(p->db));
    entered = 1;
  }
#endif

  if (idValue) {
    sqlite3_bind_value(p->stmtRowidsInsertId, 1, idValue);
  }
  rc = sqlite3_step(p->stmtRowidsInsertId);

  if (rc != SQLITE_DONE) {
    if (sqlite3_extended_errcode(p->db) == SQLITE_CONSTRAINT_UNIQUE) {
      // IMP: V20497_04568
      vtab_set_error(&p->base, "UNIQUE constraint failed on %s primary key",
                     p->tableName);
    } else {
      // IMP: V24016_08086
      // IMP: V15177_32015
      vtab_set_error(&p->base,
                     "Error inserting id into rowids shadow table: %s",
                     sqlite3_errmsg(sqlite3_db_handle(p->stmtRowidsInsertId)));
    }
    rc = SQLITE_ERROR;
    goto complete;
  }

  *rowid = sqlite3_last_insert_rowid(p->db);
  rc = SQLITE_OK;

complete:
  if (p->stmtRowidsInsertId) {
    sqlite3_reset(p->stmtRowidsInsertId);
    sqlite3_clear_bindings(p->stmtRowidsInsertId);
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_leave && entered) {
    sqlite3_mutex_leave(sqlite3_db_mutex(p->db));
  }
#endif
  return rc;
}

int vec0_metadata_chunk_size(vec0_metadata_column_kind kind, int chunk_size) {
  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN:
      return chunk_size / 8;
    case VEC0_METADATA_COLUMN_KIND_INTEGER:
      return chunk_size * sizeof(i64);
    case VEC0_METADATA_COLUMN_KIND_FLOAT:
      return chunk_size * sizeof(double);
    case VEC0_METADATA_COLUMN_KIND_TEXT:
      return chunk_size * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH;
  }
  return 0;
}

int vec0_rowids_update_position(vec0_vtab *p, i64 rowid, i64 chunk_rowid,
                                i64 chunk_offset) {
  int rc = SQLITE_OK;

  if (!p->stmtRowidsUpdatePosition) {
    const char *zSql = sqlite3_mprintf(" UPDATE " VEC0_SHADOW_ROWIDS_NAME
                                       " SET chunk_id = ?, chunk_offset = ?"
                                       " WHERE rowid = ?",
                                       p->schemaName, p->tableName);
    if (!zSql) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &p->stmtRowidsUpdatePosition, 0);
    sqlite3_free((void *)zSql);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "could not initialize 'update rowids position' statement");
      goto cleanup;
    }
  }

  sqlite3_bind_int64(p->stmtRowidsUpdatePosition, 1, chunk_rowid);
  sqlite3_bind_int64(p->stmtRowidsUpdatePosition, 2, chunk_offset);
  sqlite3_bind_int64(p->stmtRowidsUpdatePosition, 3, rowid);

  rc = sqlite3_step(p->stmtRowidsUpdatePosition);
  if (rc != SQLITE_DONE) {
    // IMP: V21925_05995
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "could not update rowids position for rowid=%lld, "
                   "chunk_rowid=%lld, chunk_offset=%lld",
                   rowid, chunk_rowid, chunk_offset);
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  rc = SQLITE_OK;

cleanup:
  if (p->stmtRowidsUpdatePosition) {
    sqlite3_reset(p->stmtRowidsUpdatePosition);
    sqlite3_clear_bindings(p->stmtRowidsUpdatePosition);
  }

  return rc;
}

/**
 * @brief Adds a new chunk for the vec0 table, and the corresponding vector
 * chunks.
 *
 * Inserts a new row into the _chunks table, with blank data, and uses that new
 * rowid to insert new blank rows into _vector_chunksXX tables.
 *
 * @param p: vec0 table to add new chunk
 * @param paritionKeyValues: Array of partition key valeus for the new chunk, if available
 * @param chunk_rowid: Output pointer, if not NULL, then will be filled with the
 * new chunk rowid.
 * @return int SQLITE_OK on success, error code otherwise.
 */
int vec0_new_chunk(vec0_vtab *p, sqlite3_value ** partitionKeyValues, i64 *chunk_rowid) {
  int rc;
  char *zSql;
  sqlite3_stmt *stmt;
  i64 rowid;

  // Step 1: Insert a new row in _chunks, capture that new rowid
  if(p->numPartitionColumns > 0) {
    sqlite3_str * s = sqlite3_str_new(NULL);
    sqlite3_str_appendf(s, "INSERT INTO " VEC0_SHADOW_CHUNKS_NAME, p->schemaName, p->tableName);
    sqlite3_str_appendall(s, "(size, validity, rowids");
    for(int i = 0; i < p->numPartitionColumns; i++) {
      sqlite3_str_appendf(s, ", partition%02d", i);
    }
    sqlite3_str_appendall(s, ") VALUES (?, ?, ?");
    for(int i = 0; i < p->numPartitionColumns; i++) {
      sqlite3_str_appendall(s, ", ?");
    }
    sqlite3_str_appendall(s, ")");

    zSql = sqlite3_str_finish(s);
  }else {
    zSql = sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_CHUNKS_NAME
                         "(size, validity, rowids) "
                         "VALUES (?, ?, ?);",
                         p->schemaName, p->tableName);
  }

  if (!zSql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    sqlite3_finalize(stmt);
    return rc;
  }

#if SQLITE_THREADSAFE
  if (sqlite3_mutex_enter) {
    sqlite3_mutex_enter(sqlite3_db_mutex(p->db));
  }
#endif

  sqlite3_bind_int64(stmt, 1, p->chunk_size);               // size
  sqlite3_bind_zeroblob(stmt, 2, p->chunk_size / CHAR_BIT); // validity bitmap
  sqlite3_bind_zeroblob(stmt, 3, p->chunk_size * sizeof(i64)); // rowids

  for(int i = 0; i < p->numPartitionColumns; i++) {
    sqlite3_bind_value(stmt, 4 + i, partitionKeyValues[i]);
  }

  rc = sqlite3_step(stmt);
  int failed = rc != SQLITE_DONE;
  rowid = sqlite3_last_insert_rowid(p->db);
#if SQLITE_THREADSAFE
  if (sqlite3_mutex_leave) {
    sqlite3_mutex_leave(sqlite3_db_mutex(p->db));
  }
#endif
  sqlite3_finalize(stmt);
  if (failed) {
    return SQLITE_ERROR;
  }

  // Step 2: Create new vector chunks for each vector column, with
  //          that new chunk_rowid.

  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_VECTOR) {
      continue;
    }
    int vector_column_idx = p->user_column_idxs[i];
    i64 vectorsSize =
        p->chunk_size * vector_column_byte_size(p->vector_columns[vector_column_idx]);

    zSql = sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_VECTOR_N_NAME
                           "(rowid, vectors)"
                           "VALUES (?, ?)",
                           p->schemaName, p->tableName, vector_column_idx);
    if (!zSql) {
      return SQLITE_NOMEM;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);

    if (rc != SQLITE_OK) {
      sqlite3_finalize(stmt);
      return rc;
    }

    sqlite3_bind_int64(stmt, 1, rowid);
    sqlite3_bind_zeroblob64(stmt, 2, vectorsSize);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) {
      return rc;
    }
  }

  // Step 3: Create new metadata chunks for each metadata column
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_METADATA) {
      continue;
    }
    int metadata_column_idx = p->user_column_idxs[i];
    zSql = sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_METADATA_N_NAME
                           "(rowid, data)"
                           "VALUES (?, ?)",
                           p->schemaName, p->tableName, metadata_column_idx);
    if (!zSql) {
      return SQLITE_NOMEM;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    sqlite3_free(zSql);

    if (rc != SQLITE_OK) {
      sqlite3_finalize(stmt);
      return rc;
    }

    sqlite3_bind_int64(stmt, 1, rowid);
    sqlite3_bind_zeroblob64(stmt, 2, vec0_metadata_chunk_size(p->metadata_columns[metadata_column_idx].kind, p->chunk_size));

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) {
      return rc;
    }
  }


  if (chunk_rowid) {
    *chunk_rowid = rowid;
  }

  return SQLITE_OK;
}

struct vec0_query_fullscan_data {
  sqlite3_stmt *rowids_stmt;
  i8 done;
};
void vec0_query_fullscan_data_clear(
    struct vec0_query_fullscan_data *fullscan_data) {
  if (!fullscan_data)
    return;

  if (fullscan_data->rowids_stmt) {
    sqlite3_finalize(fullscan_data->rowids_stmt);
    fullscan_data->rowids_stmt = NULL;
  }
}

struct vec0_query_knn_data {
  i64 k;
  i64 k_used;
  // Array of rowids of size k. Must be freed with sqlite3_free().
  i64 *rowids;
  // Array of distances of size k. Must be freed with sqlite3_free().
  f32 *distances;
  i64 current_idx;
};
void vec0_query_knn_data_clear(struct vec0_query_knn_data *knn_data) {
  if (!knn_data)
    return;

  if (knn_data->rowids) {
    sqlite3_free(knn_data->rowids);
    knn_data->rowids = NULL;
  }
  if (knn_data->distances) {
    sqlite3_free(knn_data->distances);
    knn_data->distances = NULL;
  }
}

struct vec0_query_point_data {
  i64 rowid;
  void *vectors[VEC0_MAX_VECTOR_COLUMNS];
  int done;
};
void vec0_query_point_data_clear(struct vec0_query_point_data *point_data) {
  if (!point_data)
    return;
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    sqlite3_free(point_data->vectors[i]);
    point_data->vectors[i] = NULL;
  }
}

typedef enum {
  // If any values are updated, please update the ARCHITECTURE.md docs accordingly!

 VEC0_QUERY_PLAN_FULLSCAN = '1',
 VEC0_QUERY_PLAN_POINT = '2',
 VEC0_QUERY_PLAN_KNN = '3',
} vec0_query_plan;

typedef struct vec0_cursor vec0_cursor;
struct vec0_cursor {
  sqlite3_vtab_cursor base;

  vec0_query_plan query_plan;
  struct vec0_query_fullscan_data *fullscan_data;
  struct vec0_query_knn_data *knn_data;
  struct vec0_query_point_data *point_data;
};

void vec0_cursor_clear(vec0_cursor *pCur) {
  if (pCur->fullscan_data) {
    vec0_query_fullscan_data_clear(pCur->fullscan_data);
    sqlite3_free(pCur->fullscan_data);
    pCur->fullscan_data = NULL;
  }
  if (pCur->knn_data) {
    vec0_query_knn_data_clear(pCur->knn_data);
    sqlite3_free(pCur->knn_data);
    pCur->knn_data = NULL;
  }
  if (pCur->point_data) {
    vec0_query_point_data_clear(pCur->point_data);
    sqlite3_free(pCur->point_data);
    pCur->point_data = NULL;
  }
}

#define VEC_CONSTRUCTOR_ERROR "vec0 constructor error: "
static int vec0_init(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                     sqlite3_vtab **ppVtab, char **pzErr, bool isCreate) {
  UNUSED_PARAMETER(pAux);
  vec0_vtab *pNew;
  int rc;
  const char *zSql;

  pNew = sqlite3_malloc(sizeof(*pNew));
  if (pNew == 0)
    return SQLITE_NOMEM;
  memset(pNew, 0, sizeof(*pNew));

  // Declared chunk_size=N for entire table.
  // -1 to use the defualt, otherwise will get re-assigned on `chunk_size=N`
  // option
  int chunk_size = -1;
  int numVectorColumns = 0;
  int numPartitionColumns = 0;
  int numAuxiliaryColumns = 0;
  int numMetadataColumns = 0;
  int user_column_idx = 0;

  // track if a "primary key" column is defined
  char *pkColumnName = NULL;
  int pkColumnNameLength;
  int pkColumnType = SQLITE_INTEGER;

  for (int i = 3; i < argc; i++) {
    struct VectorColumnDefinition vecColumn;
    struct Vec0PartitionColumnDefinition partitionColumn;
    struct Vec0AuxiliaryColumnDefinition auxColumn;
    struct Vec0MetadataColumnDefinition metadataColumn;
    char *cName = NULL;
    int cNameLength;
    int cType;

    // Scenario #1: Constructor argument is a vector column definition, ie `foo float[1024]`
    rc = vec0_parse_vector_column(argv[i], strlen(argv[i]), &vecColumn);
    if (rc == SQLITE_ERROR) {
      *pzErr = sqlite3_mprintf(
          VEC_CONSTRUCTOR_ERROR "could not parse vector column '%s'", argv[i]);
      goto error;
    }
    if (rc == SQLITE_OK) {
      if (numVectorColumns >= VEC0_MAX_VECTOR_COLUMNS) {
        sqlite3_free(vecColumn.name);
        *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                                 "Too many provided vector columns, maximum %d",
                                 VEC0_MAX_VECTOR_COLUMNS);
        goto error;
      }

      if (vecColumn.dimensions > SQLITE_VEC_VEC0_MAX_DIMENSIONS) {
        sqlite3_free(vecColumn.name);
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "Dimension on vector column too large, provided %lld, maximum %lld",
            (i64)vecColumn.dimensions, SQLITE_VEC_VEC0_MAX_DIMENSIONS);
        goto error;
      }
      pNew->user_column_kinds[user_column_idx] = SQLITE_VEC0_USER_COLUMN_KIND_VECTOR;
      pNew->user_column_idxs[user_column_idx] = numVectorColumns;
      memcpy(&pNew->vector_columns[numVectorColumns], &vecColumn, sizeof(vecColumn));
      numVectorColumns++;
      user_column_idx++;

      continue;
    }

    // Scenario #2: Constructor argument is a partition key column definition, ie `user_id text partition key`
    rc = vec0_parse_partition_key_definition(argv[i], strlen(argv[i]), &cName,
                                      &cNameLength, &cType);
    if (rc == SQLITE_OK) {
      if (numPartitionColumns >= VEC0_MAX_PARTITION_COLUMNS) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "More than %d partition key columns were provided",
            VEC0_MAX_PARTITION_COLUMNS);
        goto error;
      }
      partitionColumn.type = cType;
      partitionColumn.name_length = cNameLength;
      partitionColumn.name = sqlite3_mprintf("%.*s", cNameLength, cName);
      if(!partitionColumn.name) {
        rc = SQLITE_NOMEM;
        goto error;
      }

      pNew->user_column_kinds[user_column_idx] = SQLITE_VEC0_USER_COLUMN_KIND_PARTITION;
      pNew->user_column_idxs[user_column_idx] = numPartitionColumns;
      memcpy(&pNew->paritition_columns[numPartitionColumns], &partitionColumn, sizeof(partitionColumn));
      numPartitionColumns++;
      user_column_idx++;
      continue;
    }

    // Scenario #3: Constructor argument is a primary key column definition, ie `article_id text primary key`
    rc = vec0_parse_primary_key_definition(argv[i], strlen(argv[i]), &cName,
                                      &cNameLength, &cType);
    if (rc == SQLITE_OK) {
      if (pkColumnName) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "More than one primary key definition was provided, vec0 only "
            "suports a single primary key column",
            argv[i]);
        goto error;
      }
      pkColumnName = cName;
      pkColumnNameLength = cNameLength;
      pkColumnType = cType;
      continue;
    }

    // Scenario #4: Constructor argument is a auxiliary column definition, ie `+contents text`
    rc = vec0_parse_auxiliary_column_definition(argv[i], strlen(argv[i]), &cName,
                                      &cNameLength, &cType);
    if(rc == SQLITE_OK) {
      if (numAuxiliaryColumns >= VEC0_MAX_AUXILIARY_COLUMNS) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "More than %d auxiliary columns were provided",
            VEC0_MAX_AUXILIARY_COLUMNS);
        goto error;
      }
      auxColumn.type = cType;
      auxColumn.name_length = cNameLength;
      auxColumn.name = sqlite3_mprintf("%.*s", cNameLength, cName);
      if(!auxColumn.name) {
        rc = SQLITE_NOMEM;
        goto error;
      }

      pNew->user_column_kinds[user_column_idx] = SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY;
      pNew->user_column_idxs[user_column_idx] = numAuxiliaryColumns;
      memcpy(&pNew->auxiliary_columns[numAuxiliaryColumns], &auxColumn, sizeof(auxColumn));
      numAuxiliaryColumns++;
      user_column_idx++;
      continue;
    }

    vec0_metadata_column_kind kind;
    rc = vec0_parse_metadata_column_definition(argv[i], strlen(argv[i]), &cName,
                                      &cNameLength, &kind);
    if(rc == SQLITE_OK) {
      if (numMetadataColumns >= VEC0_MAX_METADATA_COLUMNS) {
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR
            "More than %d metadata columns were provided",
            VEC0_MAX_METADATA_COLUMNS);
        goto error;
      }
      metadataColumn.kind = kind;
      metadataColumn.name_length = cNameLength;
      metadataColumn.name = sqlite3_mprintf("%.*s", cNameLength, cName);
      if(!metadataColumn.name) {
        rc = SQLITE_NOMEM;
        goto error;
      }

      pNew->user_column_kinds[user_column_idx] = SQLITE_VEC0_USER_COLUMN_KIND_METADATA;
      pNew->user_column_idxs[user_column_idx] = numMetadataColumns;
      memcpy(&pNew->metadata_columns[numMetadataColumns], &metadataColumn, sizeof(metadataColumn));
      numMetadataColumns++;
      user_column_idx++;
      continue;
    }

    // Scenario #4: Constructor argument is a table-level option, ie `chunk_size`

    char *key;
    char *value;
    int keyLength, valueLength;
    rc = vec0_parse_table_option(argv[i], strlen(argv[i]), &key, &keyLength,
                                 &value, &valueLength);
    if (rc == SQLITE_ERROR) {
      *pzErr = sqlite3_mprintf(
          VEC_CONSTRUCTOR_ERROR "could not parse table option '%s'", argv[i]);
      goto error;
    }
    if (rc == SQLITE_OK) {
      if (sqlite3_strnicmp(key, "chunk_size", keyLength) == 0) {
        chunk_size = atoi(value);
        if (chunk_size <= 0) {
          // IMP: V01931_18769
          *pzErr =
              sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                              "chunk_size must be a non-zero positive integer");
          goto error;
        }
        if ((chunk_size % 8) != 0) {
          // IMP: V14110_30948
          *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                                   "chunk_size must be divisible by 8");
          goto error;
        }
#define SQLITE_VEC_CHUNK_SIZE_MAX 4096
        if (chunk_size > SQLITE_VEC_CHUNK_SIZE_MAX) {
          *pzErr =
              sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR "chunk_size too large");
          goto error;
        }
      } else {
        // IMP: V27642_11712
        *pzErr = sqlite3_mprintf(
            VEC_CONSTRUCTOR_ERROR "Unknown table option: %.*s", keyLength, key);
        goto error;
      }
      continue;
    }

    // Scenario #5: Unknown constructor argument
    *pzErr =
        sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR "Could not parse '%s'", argv[i]);
    goto error;
  }

  if (chunk_size < 0) {
    chunk_size = 1024;
  }

  if (numVectorColumns <= 0) {
    *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                             "At least one vector column is required");
    goto error;
  }

  sqlite3_str *createStr = sqlite3_str_new(NULL);
  sqlite3_str_appendall(createStr, "CREATE TABLE x(");
  if (pkColumnName) {
    sqlite3_str_appendf(createStr, "\"%.*w\" primary key, ", pkColumnNameLength,
                        pkColumnName);
  } else {
    sqlite3_str_appendall(createStr, "rowid, ");
  }
  for (int i = 0; i < numVectorColumns + numPartitionColumns + numAuxiliaryColumns + numMetadataColumns; i++) {
    switch(pNew->user_column_kinds[i]) {
      case SQLITE_VEC0_USER_COLUMN_KIND_VECTOR: {
        int vector_idx = pNew->user_column_idxs[i];
        sqlite3_str_appendf(createStr, "\"%.*w\", ",
                        pNew->vector_columns[vector_idx].name_length,
                        pNew->vector_columns[vector_idx].name);
        break;
      }
      case SQLITE_VEC0_USER_COLUMN_KIND_PARTITION: {
        int partition_idx = pNew->user_column_idxs[i];
        sqlite3_str_appendf(createStr, "\"%.*w\", ",
                        pNew->paritition_columns[partition_idx].name_length,
                        pNew->paritition_columns[partition_idx].name);
        break;
      }
      case SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY: {
        int auxiliary_idx = pNew->user_column_idxs[i];
        sqlite3_str_appendf(createStr, "\"%.*w\", ",
                        pNew->auxiliary_columns[auxiliary_idx].name_length,
                        pNew->auxiliary_columns[auxiliary_idx].name);
        break;
      }
      case SQLITE_VEC0_USER_COLUMN_KIND_METADATA: {
        int metadata_idx = pNew->user_column_idxs[i];
        sqlite3_str_appendf(createStr, "\"%.*w\", ",
                        pNew->metadata_columns[metadata_idx].name_length,
                        pNew->metadata_columns[metadata_idx].name);
        break;
      }
    }

  }
  sqlite3_str_appendall(createStr, " distance hidden, k hidden) ");
  if (pkColumnName) {
    sqlite3_str_appendall(createStr, "without rowid ");
  }
  zSql = sqlite3_str_finish(createStr);
  if (!zSql) {
    goto error;
  }
  rc = sqlite3_declare_vtab(db, zSql);
  sqlite3_free((void *)zSql);
  if (rc != SQLITE_OK) {
    *pzErr = sqlite3_mprintf(VEC_CONSTRUCTOR_ERROR
                             "could not declare virtual table, '%s'",
                             sqlite3_errmsg(db));
    goto error;
  }

  const char *schemaName = argv[1];
  const char *tableName = argv[2];

  pNew->db = db;
  pNew->pkIsText = pkColumnType == SQLITE_TEXT;
  pNew->schemaName = sqlite3_mprintf("%s", schemaName);
  if (!pNew->schemaName) {
    goto error;
  }
  pNew->tableName = sqlite3_mprintf("%s", tableName);
  if (!pNew->tableName) {
    goto error;
  }
  pNew->shadowRowidsName = sqlite3_mprintf("%s_rowids", tableName);
  if (!pNew->shadowRowidsName) {
    goto error;
  }
  pNew->shadowChunksName = sqlite3_mprintf("%s_chunks", tableName);
  if (!pNew->shadowChunksName) {
    goto error;
  }
  pNew->numVectorColumns = numVectorColumns;
  pNew->numPartitionColumns = numPartitionColumns;
  pNew->numAuxiliaryColumns = numAuxiliaryColumns;
  pNew->numMetadataColumns = numMetadataColumns;

  for (int i = 0; i < pNew->numVectorColumns; i++) {
    pNew->shadowVectorChunksNames[i] =
        sqlite3_mprintf("%s_vector_chunks%02d", tableName, i);
    if (!pNew->shadowVectorChunksNames[i]) {
      goto error;
    }
  }
  for (int i = 0; i < pNew->numMetadataColumns; i++) {
    pNew->shadowMetadataChunksNames[i] =
        sqlite3_mprintf("%s_metadatachunks%02d", tableName, i);
    if (!pNew->shadowMetadataChunksNames[i]) {
      goto error;
    }
  }
  pNew->chunk_size = chunk_size;

  // if xCreate, then create the necessary shadow tables
  if (isCreate) {
    sqlite3_stmt *stmt;
    int rc;

    char * zCreateInfo = sqlite3_mprintf("CREATE TABLE "VEC0_SHADOW_INFO_NAME " (key text primary key, value any)", pNew->schemaName, pNew->tableName);
    if(!zCreateInfo) {
      goto error;
    }
    rc = sqlite3_prepare_v2(db, zCreateInfo, -1, &stmt, NULL);

    sqlite3_free((void *) zCreateInfo);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      // TODO(IMP)
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not create '_info' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_finalize(stmt);

    char * zSeedInfo = sqlite3_mprintf(
      "INSERT INTO "VEC0_SHADOW_INFO_NAME "(key, value) VALUES "
      "(?1, ?2), (?3, ?4), (?5, ?6), (?7, ?8) ",
      pNew->schemaName, pNew->tableName
    );
    if(!zSeedInfo) {
      goto error;
    }
    rc = sqlite3_prepare_v2(db, zSeedInfo, -1, &stmt, NULL);
    sqlite3_free((void *) zSeedInfo);
    if (rc != SQLITE_OK) {
      // TODO(IMP)
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not seed '_info' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_bind_text(stmt, 1, "CREATE_VERSION", -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, SQLITE_VEC_VERSION, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, "CREATE_VERSION_MAJOR", -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 4, SQLITE_VEC_VERSION_MAJOR);
    sqlite3_bind_text(stmt, 5, "CREATE_VERSION_MINOR", -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 6, SQLITE_VEC_VERSION_MINOR);
    sqlite3_bind_text(stmt, 7, "CREATE_VERSION_PATCH", -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 8, SQLITE_VEC_VERSION_PATCH);

    if(sqlite3_step(stmt) != SQLITE_DONE) {
      // TODO(IMP)
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not seed '_info' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_finalize(stmt);



    // create the _chunks shadow table
    char *zCreateShadowChunks = NULL;
    if(pNew->numPartitionColumns) {
      sqlite3_str * s = sqlite3_str_new(NULL);
      sqlite3_str_appendf(s, "CREATE TABLE " VEC0_SHADOW_CHUNKS_NAME "(", pNew->schemaName, pNew->tableName);
      sqlite3_str_appendall(s, "chunk_id INTEGER PRIMARY KEY AUTOINCREMENT," "size INTEGER NOT NULL,");
      sqlite3_str_appendall(s, "sequence_id integer,");
      for(int i = 0; i < pNew->numPartitionColumns;i++) {
        sqlite3_str_appendf(s, "partition%02d,", i);
      }
      sqlite3_str_appendall(s, "validity BLOB NOT NULL, rowids BLOB NOT NULL);");
      zCreateShadowChunks = sqlite3_str_finish(s);
    }else {
      zCreateShadowChunks = sqlite3_mprintf(VEC0_SHADOW_CHUNKS_CREATE,
                                          pNew->schemaName, pNew->tableName);
    }
    if (!zCreateShadowChunks) {
        goto error;
      }
    rc = sqlite3_prepare_v2(db, zCreateShadowChunks, -1, &stmt, 0);
    sqlite3_free((void *)zCreateShadowChunks);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      // IMP: V17740_01811
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not create '_chunks' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_finalize(stmt);

    // create the _rowids shadow table
    char *zCreateShadowRowids;
    if (pNew->pkIsText) {
      // adds a "text unique not null" constraint to the id column
      zCreateShadowRowids = sqlite3_mprintf(VEC0_SHADOW_ROWIDS_CREATE_PK_TEXT,
                                            pNew->schemaName, pNew->tableName);
    } else {
      zCreateShadowRowids = sqlite3_mprintf(VEC0_SHADOW_ROWIDS_CREATE_BASIC,
                                            pNew->schemaName, pNew->tableName);
    }
    if (!zCreateShadowRowids) {
      goto error;
    }
    rc = sqlite3_prepare_v2(db, zCreateShadowRowids, -1, &stmt, 0);
    sqlite3_free((void *)zCreateShadowRowids);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      // IMP: V11631_28470
      sqlite3_finalize(stmt);
      *pzErr = sqlite3_mprintf("Could not create '_rowids' shadow table: %s",
                               sqlite3_errmsg(db));
      goto error;
    }
    sqlite3_finalize(stmt);

    for (int i = 0; i < pNew->numVectorColumns; i++) {
      char *zSql = sqlite3_mprintf(VEC0_SHADOW_VECTOR_N_CREATE,
                                   pNew->schemaName, pNew->tableName, i);
      if (!zSql) {
        goto error;
      }
      rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
      sqlite3_free((void *)zSql);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        // IMP: V25919_09989
        sqlite3_finalize(stmt);
        *pzErr = sqlite3_mprintf(
            "Could not create '_vector_chunks%02d' shadow table: %s", i,
            sqlite3_errmsg(db));
        goto error;
      }
      sqlite3_finalize(stmt);
    }

    for (int i = 0; i < pNew->numMetadataColumns; i++) {
      char *zSql = sqlite3_mprintf("CREATE TABLE " VEC0_SHADOW_METADATA_N_NAME "(rowid PRIMARY KEY, data BLOB NOT NULL);",
                                   pNew->schemaName, pNew->tableName, i);
      if (!zSql) {
        goto error;
      }
      rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
      sqlite3_free((void *)zSql);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        sqlite3_finalize(stmt);
        *pzErr = sqlite3_mprintf(
            "Could not create '_metata_chunks%02d' shadow table: %s", i,
            sqlite3_errmsg(db));
        goto error;
      }
      sqlite3_finalize(stmt);

      if(pNew->metadata_columns[i].kind == VEC0_METADATA_COLUMN_KIND_TEXT) {
        char *zSql = sqlite3_mprintf("CREATE TABLE " VEC0_SHADOW_METADATA_TEXT_DATA_NAME "(rowid PRIMARY KEY, data TEXT);",
                                   pNew->schemaName, pNew->tableName, i);
        if (!zSql) {
          goto error;
        }
        rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, 0);
        sqlite3_free((void *)zSql);
        if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
          sqlite3_finalize(stmt);
          *pzErr = sqlite3_mprintf(
              "Could not create '_metadatatext%02d' shadow table: %s", i,
              sqlite3_errmsg(db));
          goto error;
        }
        sqlite3_finalize(stmt);

      }
    }

    if(pNew->numAuxiliaryColumns > 0) {
      sqlite3_stmt * stmt;
      sqlite3_str * s = sqlite3_str_new(NULL);
      sqlite3_str_appendf(s, "CREATE TABLE " VEC0_SHADOW_AUXILIARY_NAME "( rowid integer PRIMARY KEY ", pNew->schemaName, pNew->tableName);
      for(int i = 0; i < pNew->numAuxiliaryColumns; i++) {
        sqlite3_str_appendf(s, ", value%02d", i);
      }
      sqlite3_str_appendall(s, ")");
      char *zSql = sqlite3_str_finish(s);
      if(!zSql) {
        goto error;
      }
      rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        sqlite3_finalize(stmt);
        *pzErr = sqlite3_mprintf(
            "Could not create auxiliary shadow table: %s",
            sqlite3_errmsg(db));

        goto error;
      }
      sqlite3_finalize(stmt);
    }
  }

  *ppVtab = (sqlite3_vtab *)pNew;
  return SQLITE_OK;

error:
  vec0_free(pNew);
  return SQLITE_ERROR;
}

static int vec0Create(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  return vec0_init(db, pAux, argc, argv, ppVtab, pzErr, true);
}
static int vec0Connect(sqlite3 *db, void *pAux, int argc,
                       const char *const *argv, sqlite3_vtab **ppVtab,
                       char **pzErr) {
  return vec0_init(db, pAux, argc, argv, ppVtab, pzErr, false);
}

static int vec0Disconnect(sqlite3_vtab *pVtab) {
  vec0_vtab *p = (vec0_vtab *)pVtab;
  vec0_free(p);
  sqlite3_free(p);
  return SQLITE_OK;
}
static int vec0Destroy(sqlite3_vtab *pVtab) {
  vec0_vtab *p = (vec0_vtab *)pVtab;
  sqlite3_stmt *stmt;
  int rc;
  const char *zSql;

  // Free up any sqlite3_stmt, otherwise DROPs on those tables will fail
  vec0_free_resources(p);

  // TODO(test) later: can't evidence-of here, bc always gives "SQL logic error" instead of
  // provided error
  zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_CHUNKS_NAME, p->schemaName,
                         p->tableName);
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
  sqlite3_free((void *)zSql);
  if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
    rc = SQLITE_ERROR;
    vtab_set_error(pVtab, "could not drop chunks shadow table");
    goto done;
  }
  sqlite3_finalize(stmt);

  zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_INFO_NAME, p->schemaName,
                         p->tableName);
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
  sqlite3_free((void *)zSql);
  if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
    rc = SQLITE_ERROR;
    vtab_set_error(pVtab, "could not drop info shadow table");
    goto done;
  }
  sqlite3_finalize(stmt);

  zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_ROWIDS_NAME, p->schemaName,
                         p->tableName);
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
  sqlite3_free((void *)zSql);
  if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
    rc = SQLITE_ERROR;
    goto done;
  }
  sqlite3_finalize(stmt);

  for (int i = 0; i < p->numVectorColumns; i++) {
    zSql = sqlite3_mprintf("DROP TABLE \"%w\".\"%w\"", p->schemaName,
                           p->shadowVectorChunksNames[i]);
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
    sqlite3_free((void *)zSql);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      rc = SQLITE_ERROR;
      goto done;
    }
    sqlite3_finalize(stmt);
  }

  if(p->numAuxiliaryColumns > 0) {
    zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_AUXILIARY_NAME, p->schemaName, p->tableName);
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
    sqlite3_free((void *)zSql);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      rc = SQLITE_ERROR;
      goto done;
    }
    sqlite3_finalize(stmt);
  }


  for (int i = 0; i < p->numMetadataColumns; i++) {
    zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_METADATA_N_NAME, p->schemaName,p->tableName, i);
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
    sqlite3_free((void *)zSql);
    if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
      rc = SQLITE_ERROR;
      goto done;
    }
    sqlite3_finalize(stmt);

    if(p->metadata_columns[i].kind == VEC0_METADATA_COLUMN_KIND_TEXT) {
      zSql = sqlite3_mprintf("DROP TABLE " VEC0_SHADOW_METADATA_TEXT_DATA_NAME, p->schemaName,p->tableName, i);
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, 0);
      sqlite3_free((void *)zSql);
      if ((rc != SQLITE_OK) || (sqlite3_step(stmt) != SQLITE_DONE)) {
        rc = SQLITE_ERROR;
        goto done;
      }
      sqlite3_finalize(stmt);
    }
  }

  stmt = NULL;
  rc = SQLITE_OK;

done:
  sqlite3_finalize(stmt);
  vec0_free(p);
  // If there was an error
  if (rc == SQLITE_OK) {
    sqlite3_free(p);
  }
  return rc;
}

static int vec0Open(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec0_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec0Close(sqlite3_vtab_cursor *cur) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  vec0_cursor_clear(pCur);
  sqlite3_free(pCur);
  return SQLITE_OK;
}

// All the different type of "values" provided to argv/argc in vec0Filter.
// These enums denote the use and purpose of all of them.
typedef enum  {
  // If any values are updated, please update the ARCHITECTURE.md docs accordingly!

  VEC0_IDXSTR_KIND_KNN_MATCH = '{',
  VEC0_IDXSTR_KIND_KNN_K = '}',
  VEC0_IDXSTR_KIND_KNN_ROWID_IN = '[',
  VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT = ']',
  VEC0_IDXSTR_KIND_POINT_ID = '!',
  VEC0_IDXSTR_KIND_METADATA_CONSTRAINT = '&',
} vec0_idxstr_kind;

// The different SQLITE_INDEX_CONSTRAINT values that vec0 partition key columns
// support, but as characters that fit nicely in idxstr.
typedef enum  {
  // If any values are updated, please update the ARCHITECTURE.md docs accordingly!

  VEC0_PARTITION_OPERATOR_EQ = 'a',
  VEC0_PARTITION_OPERATOR_GT = 'b',
  VEC0_PARTITION_OPERATOR_LE = 'c',
  VEC0_PARTITION_OPERATOR_LT = 'd',
  VEC0_PARTITION_OPERATOR_GE = 'e',
  VEC0_PARTITION_OPERATOR_NE = 'f',
} vec0_partition_operator;
typedef enum  {
  VEC0_METADATA_OPERATOR_EQ = 'a',
  VEC0_METADATA_OPERATOR_GT = 'b',
  VEC0_METADATA_OPERATOR_LE = 'c',
  VEC0_METADATA_OPERATOR_LT = 'd',
  VEC0_METADATA_OPERATOR_GE = 'e',
  VEC0_METADATA_OPERATOR_NE = 'f',
  VEC0_METADATA_OPERATOR_IN = 'g',
} vec0_metadata_operator;

static int vec0BestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
  /**
   * Possible query plans are:
   * 1. KNN when:
   *    a) An `MATCH` op on vector column
   *    b) ORDER BY on distance column
   *    c) LIMIT
   *    d) rowid in (...) OPTIONAL
   * 2. Point when:
   *    a) An `EQ` op on rowid column
   * 3. else: fullscan
   *
   */
  int iMatchTerm = -1;
  int iMatchVectorTerm = -1;
  int iLimitTerm = -1;
  int iRowidTerm = -1;
  int iKTerm = -1;
  int iRowidInTerm = -1;
  int hasAuxConstraint = 0;

#ifdef SQLITE_VEC_DEBUG
  printf("pIdxInfo->nOrderBy=%d, pIdxInfo->nConstraint=%d\n", pIdxInfo->nOrderBy, pIdxInfo->nConstraint);
#endif

  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    u8 vtabIn = 0;

#if COMPILER_SUPPORTS_VTAB_IN
    if (sqlite3_libversion_number() >= 3038000) {
      vtabIn = sqlite3_vtab_in(pIdxInfo, i, -1);
    }
#endif

#ifdef SQLITE_VEC_DEBUG
    printf("xBestIndex [%d] usable=%d iColumn=%d op=%d vtabin=%d\n", i,
           pIdxInfo->aConstraint[i].usable, pIdxInfo->aConstraint[i].iColumn,
           pIdxInfo->aConstraint[i].op, vtabIn);
#endif
    if (!pIdxInfo->aConstraint[i].usable)
      continue;

    int iColumn = pIdxInfo->aConstraint[i].iColumn;
    int op = pIdxInfo->aConstraint[i].op;

    if (op == SQLITE_INDEX_CONSTRAINT_LIMIT) {
      iLimitTerm = i;
    }
    if (op == SQLITE_INDEX_CONSTRAINT_MATCH &&
        vec0_column_idx_is_vector(p, iColumn)) {
      if (iMatchTerm > -1) {
        vtab_set_error(
            pVTab, "only 1 MATCH operator is allowed in a single vec0 query");
        return SQLITE_ERROR;
      }
      iMatchTerm = i;
      iMatchVectorTerm = vec0_column_idx_to_vector_idx(p, iColumn);
    }
    if (op == SQLITE_INDEX_CONSTRAINT_EQ && iColumn == VEC0_COLUMN_ID) {
      if (vtabIn) {
        if (iRowidInTerm != -1) {
          vtab_set_error(pVTab, "only 1 'rowid in (..)' operator is allowed in "
                                "a single vec0 query");
          return SQLITE_ERROR;
        }
        iRowidInTerm = i;

      } else {
        iRowidTerm = i;
      }
    }
    if (op == SQLITE_INDEX_CONSTRAINT_EQ && iColumn == vec0_column_k_idx(p)) {
      iKTerm = i;
    }
    if(
      (op != SQLITE_INDEX_CONSTRAINT_LIMIT && op != SQLITE_INDEX_CONSTRAINT_OFFSET)
      && vec0_column_idx_is_auxiliary(p, iColumn)) {
        hasAuxConstraint = 1;
      }
  }

  sqlite3_str *idxStr = sqlite3_str_new(NULL);
  int rc;

  if (iMatchTerm >= 0) {
    if (iLimitTerm < 0 && iKTerm < 0) {
      vtab_set_error(
          pVTab,
          "A LIMIT or 'k = ?' constraint is required on vec0 knn queries.");
      rc = SQLITE_ERROR;
      goto done;
    }
    if (iLimitTerm >= 0 && iKTerm >= 0) {
      vtab_set_error(pVTab, "Only LIMIT or 'k =?' can be provided, not both");
      rc = SQLITE_ERROR;
      goto done;
    }

    if (pIdxInfo->nOrderBy) {
      if (pIdxInfo->nOrderBy > 1) {
        vtab_set_error(pVTab, "Only a single 'ORDER BY distance' clause is "
                              "allowed on vec0 KNN queries");
        rc = SQLITE_ERROR;
      goto done;
      }
      if (pIdxInfo->aOrderBy[0].iColumn != vec0_column_distance_idx(p)) {
        vtab_set_error(pVTab,
                       "Only a single 'ORDER BY distance' clause is allowed on "
                       "vec0 KNN queries, not on other columns");
        rc = SQLITE_ERROR;
      goto done;
      }
      if (pIdxInfo->aOrderBy[0].desc) {
        vtab_set_error(
            pVTab, "Only ascending in ORDER BY distance clause is supported, "
                   "DESC is not supported yet.");
        rc = SQLITE_ERROR;
      goto done;
      }
    }

    if(hasAuxConstraint) {
      // IMP: V25623_09693
      vtab_set_error(pVTab, "An illegal WHERE constraint was provided on a vec0 auxiliary column in a KNN query.");
      rc = SQLITE_ERROR;
      goto done;
    }

    sqlite3_str_appendchar(idxStr, 1, VEC0_QUERY_PLAN_KNN);

    int argvIndex = 1;
    pIdxInfo->aConstraintUsage[iMatchTerm].argvIndex = argvIndex++;
    pIdxInfo->aConstraintUsage[iMatchTerm].omit = 1;
    sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_MATCH);
    sqlite3_str_appendchar(idxStr, 3, '_');

    if (iLimitTerm >= 0) {
      pIdxInfo->aConstraintUsage[iLimitTerm].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[iLimitTerm].omit = 1;
    } else {
      pIdxInfo->aConstraintUsage[iKTerm].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[iKTerm].omit = 1;
    }
    sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_K);
    sqlite3_str_appendchar(idxStr, 3, '_');

#if COMPILER_SUPPORTS_VTAB_IN
    if (iRowidInTerm >= 0) {
      // already validated as  >= SQLite 3.38 bc iRowidInTerm is only >= 0 when
      // vtabIn == 1
      sqlite3_vtab_in(pIdxInfo, iRowidInTerm, 1);
      pIdxInfo->aConstraintUsage[iRowidInTerm].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[iRowidInTerm].omit = 1;
      sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_ROWID_IN);
      sqlite3_str_appendchar(idxStr, 3, '_');
    }
#endif

    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
      if (!pIdxInfo->aConstraint[i].usable)
        continue;

      int iColumn = pIdxInfo->aConstraint[i].iColumn;
      int op = pIdxInfo->aConstraint[i].op;
      if(op == SQLITE_INDEX_CONSTRAINT_LIMIT || op == SQLITE_INDEX_CONSTRAINT_OFFSET) {
        continue;
      }
      if(!vec0_column_idx_is_partition(p, iColumn)) {
        continue;
      }

      int partition_idx = vec0_column_idx_to_partition_idx(p, iColumn);
      char value = 0;

      switch(op) {
        case SQLITE_INDEX_CONSTRAINT_EQ: {
          value = VEC0_PARTITION_OPERATOR_EQ;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_GT: {
          value = VEC0_PARTITION_OPERATOR_GT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LE: {
          value = VEC0_PARTITION_OPERATOR_LE;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LT: {
          value = VEC0_PARTITION_OPERATOR_LT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_GE: {
          value = VEC0_PARTITION_OPERATOR_GE;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_NE: {
          value = VEC0_PARTITION_OPERATOR_NE;
          break;
        }
      }

      if(value) {
        pIdxInfo->aConstraintUsage[i].argvIndex = argvIndex++;
        pIdxInfo->aConstraintUsage[i].omit = 1;
        sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT);
        sqlite3_str_appendchar(idxStr, 1, 'A' + partition_idx);
        sqlite3_str_appendchar(idxStr, 1, value);
        sqlite3_str_appendchar(idxStr, 1, '_');
      }

    }

    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
      if (!pIdxInfo->aConstraint[i].usable)
        continue;

      int iColumn = pIdxInfo->aConstraint[i].iColumn;
      int op = pIdxInfo->aConstraint[i].op;
      if(op == SQLITE_INDEX_CONSTRAINT_LIMIT || op == SQLITE_INDEX_CONSTRAINT_OFFSET) {
        continue;
      }
      if(!vec0_column_idx_is_metadata(p, iColumn)) {
        continue;
      }

      int metadata_idx = vec0_column_idx_to_metadata_idx(p, iColumn);
      char value = 0;

      switch(op) {
        case SQLITE_INDEX_CONSTRAINT_EQ: {
          int vtabIn = 0;
          #if COMPILER_SUPPORTS_VTAB_IN
          if (sqlite3_libversion_number() >= 3038000) {
            vtabIn = sqlite3_vtab_in(pIdxInfo, i, -1);
          }
          if(vtabIn) {
            switch(p->metadata_columns[metadata_idx].kind) {
              case VEC0_METADATA_COLUMN_KIND_FLOAT:
              case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
                // IMP: V15248_32086
                rc = SQLITE_ERROR;
                vtab_set_error(pVTab, "'xxx in (...)' is only available on INTEGER or TEXT metadata columns.");
                goto done;
                break;
              }
              case VEC0_METADATA_COLUMN_KIND_INTEGER:
              case VEC0_METADATA_COLUMN_KIND_TEXT: {
                break;
              }
            }
            value = VEC0_METADATA_OPERATOR_IN;
            sqlite3_vtab_in(pIdxInfo, i, 1);
          }else
          #endif
           {
            value = VEC0_PARTITION_OPERATOR_EQ;
          }
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_GT: {
          value = VEC0_METADATA_OPERATOR_GT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LE: {
          value = VEC0_METADATA_OPERATOR_LE;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_LT: {
          value = VEC0_METADATA_OPERATOR_LT;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_GE: {
          value = VEC0_METADATA_OPERATOR_GE;
          break;
        }
        case SQLITE_INDEX_CONSTRAINT_NE: {
          value = VEC0_METADATA_OPERATOR_NE;
          break;
        }
        default: {
          // IMP: V16511_00582
          rc = SQLITE_ERROR;
          vtab_set_error(pVTab,
          "An illegal WHERE constraint was provided on a vec0 metadata column in a KNN query. "
          "Only one of EQUALS, GREATER_THAN, LESS_THAN_OR_EQUAL, LESS_THAN, GREATER_THAN_OR_EQUAL, NOT_EQUALS is allowed."
          );
          goto done;
        }
      }

      if(p->metadata_columns[metadata_idx].kind == VEC0_METADATA_COLUMN_KIND_BOOLEAN) {
        if(!(value == VEC0_METADATA_OPERATOR_EQ || value == VEC0_METADATA_OPERATOR_NE)) {
          // IMP: V10145_26984
          rc = SQLITE_ERROR;
          vtab_set_error(pVTab, "ONLY EQUALS (=) or NOT_EQUALS (!=) operators are allowed on boolean metadata columns.");
          goto done;
        }
      }

      pIdxInfo->aConstraintUsage[i].argvIndex = argvIndex++;
      pIdxInfo->aConstraintUsage[i].omit = 1;
      sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_METADATA_CONSTRAINT);
      sqlite3_str_appendchar(idxStr, 1, 'A' + metadata_idx);
      sqlite3_str_appendchar(idxStr, 1, value);
      sqlite3_str_appendchar(idxStr, 1, '_');

    }



    pIdxInfo->idxNum = iMatchVectorTerm;
    pIdxInfo->estimatedCost = 30.0;
    pIdxInfo->estimatedRows = 10;

  } else if (iRowidTerm >= 0) {
    sqlite3_str_appendchar(idxStr, 1, VEC0_QUERY_PLAN_POINT);
    pIdxInfo->aConstraintUsage[iRowidTerm].argvIndex = 1;
    pIdxInfo->aConstraintUsage[iRowidTerm].omit = 1;
    sqlite3_str_appendchar(idxStr, 1, VEC0_IDXSTR_KIND_POINT_ID);
    sqlite3_str_appendchar(idxStr, 3, '_');
    pIdxInfo->idxNum = pIdxInfo->colUsed;
    pIdxInfo->estimatedCost = 10.0;
    pIdxInfo->estimatedRows = 1;
  } else {
    sqlite3_str_appendchar(idxStr, 1, VEC0_QUERY_PLAN_FULLSCAN);
    pIdxInfo->estimatedCost = 3000000.0;
    pIdxInfo->estimatedRows = 100000;
  }
  pIdxInfo->idxStr = sqlite3_str_finish(idxStr);
  idxStr = NULL;
  if (!pIdxInfo->idxStr) {
    rc = SQLITE_OK;
    goto done;
  }
  pIdxInfo->needToFreeIdxStr = 1;


  rc = SQLITE_OK;

  done:
    if(idxStr) {
      sqlite3_str_finish(idxStr);
    }
    return rc;
}

// forward delcaration bc vec0Filter uses it
static int vec0Next(sqlite3_vtab_cursor *cur);

void merge_sorted_lists(f32 *a, i64 *a_rowids, i64 a_length, f32 *b,
                        i64 *b_rowids, i32 *b_top_idxs, i64 b_length, f32 *out,
                        i64 *out_rowids, i64 out_length, i64 *out_used) {
  // assert((a_length >= out_length) || (b_length >= out_length));
  i64 ptrA = 0;
  i64 ptrB = 0;
  for (int i = 0; i < out_length; i++) {
    if ((ptrA >= a_length) && (ptrB >= b_length)) {
      *out_used = i;
      return;
    }
    if (ptrA >= a_length) {
      out[i] = b[b_top_idxs[ptrB]];
      out_rowids[i] = b_rowids[b_top_idxs[ptrB]];
      ptrB++;
    } else if (ptrB >= b_length) {
      out[i] = a[ptrA];
      out_rowids[i] = a_rowids[ptrA];
      ptrA++;
    } else {
      if (a[ptrA] <= b[b_top_idxs[ptrB]]) {
        out[i] = a[ptrA];
        out_rowids[i] = a_rowids[ptrA];
        ptrA++;
      } else {
        out[i] = b[b_top_idxs[ptrB]];
        out_rowids[i] = b_rowids[b_top_idxs[ptrB]];
        ptrB++;
      }
    }
  }

  *out_used = out_length;
}

u8 *bitmap_new(i32 n) {
  assert(n % 8 == 0);
  u8 *p = sqlite3_malloc(n * sizeof(u8) / CHAR_BIT);
  if (p) {
    memset(p, 0, n * sizeof(u8) / CHAR_BIT);
  }
  return p;
}
u8 *bitmap_new_from(i32 n, u8 *from) {
  assert(n % 8 == 0);
  u8 *p = sqlite3_malloc(n * sizeof(u8) / CHAR_BIT);
  if (p) {
    memcpy(p, from, n / CHAR_BIT);
  }
  return p;
}

void bitmap_copy(u8 *base, u8 *from, i32 n) {
  assert(n % 8 == 0);
  memcpy(base, from, n / CHAR_BIT);
}

void bitmap_and_inplace(u8 *base, u8 *other, i32 n) {
  assert((n % 8) == 0);
  for (int i = 0; i < n / CHAR_BIT; i++) {
    base[i] = base[i] & other[i];
  }
}

void bitmap_set(u8 *bitmap, i32 position, int value) {
  if (value) {
    bitmap[position / CHAR_BIT] |= 1 << (position % CHAR_BIT);
  } else {
    bitmap[position / CHAR_BIT] &= ~(1 << (position % CHAR_BIT));
  }
}

int bitmap_get(u8 *bitmap, i32 position) {
  return (((bitmap[position / CHAR_BIT]) >> (position % CHAR_BIT)) & 1);
}

void bitmap_clear(u8 *bitmap, i32 n) {
  assert((n % 8) == 0);
  memset(bitmap, 0, n / CHAR_BIT);
}

void bitmap_fill(u8 *bitmap, i32 n) {
  assert((n % 8) == 0);
  memset(bitmap, 0xFF, n / CHAR_BIT);
}

/**
 * @brief Finds the minimum k items in distances, and writes the indicies to
 * out.
 *
 * @param distances input f32 array of size n, the items to consider.
 * @param n: size of distances array.
 * @param out: Output array of size k, will contain at most k element indicies
 * @param k: Size of output array
 * @return int
 */
int min_idx(const f32 *distances, i32 n, u8 *candidates, i32 *out, i32 k,
            u8 *bTaken, i32 *k_used) {
  assert(k > 0);
  assert(k <= n);

  bitmap_clear(bTaken, n);

  for (int ik = 0; ik < k; ik++) {
    int min_idx = 0;
    while (min_idx < n &&
           (bitmap_get(bTaken, min_idx) || !bitmap_get(candidates, min_idx))) {
      min_idx++;
    }
    if (min_idx >= n) {
      *k_used = ik;
      return SQLITE_OK;
    }

    for (int i = 0; i < n; i++) {
      if (distances[i] <= distances[min_idx] && !bitmap_get(bTaken, i) &&
          (bitmap_get(candidates, i))) {
        min_idx = i;
      }
    }

    out[ik] = min_idx;
    bitmap_set(bTaken, min_idx, 1);
  }
  *k_used = k;
  return SQLITE_OK;
}

int vec0_get_metadata_text_long_value(
  vec0_vtab * p,
  sqlite3_stmt ** stmt,
  int metadata_idx,
  i64 rowid,
  int *n,
  char ** s) {
  int rc;
  if(!(*stmt)) {
    const char * zSql = sqlite3_mprintf("select data from " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " where rowid = ?", p->schemaName, p->tableName, metadata_idx);
    if(!zSql) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, stmt, NULL);
    sqlite3_free( (void *) zSql);
    if(rc != SQLITE_OK) {
      goto done;
    }
  }

  sqlite3_reset(*stmt);
  sqlite3_bind_int64(*stmt, 1, rowid);
  rc = sqlite3_step(*stmt);
  if(rc != SQLITE_ROW) {
    rc = SQLITE_ERROR;
    goto done;
  }
  *s = (char *) sqlite3_column_text(*stmt, 0);
  *n = sqlite3_column_bytes(*stmt, 0);
  rc = SQLITE_OK;
  done:
    return rc;
}

/**
 * @brief Crete at "iterator" (sqlite3_stmt) of chunks with the given constraints
 *
 * Any VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT values in idxStr/argv will be applied
 * as WHERE constraints in the underlying stmt SQL, and any consumer of the stmt
 * can freely step through the stmt with all constraints satisfied.
 *
 * @param p - vec0_vtab
 * @param idxStr - the xBestIndex/xFilter idxstr containing VEC0_IDXSTR values
 * @param argc - number of argv values from xFilter
 * @param argv - array of sqlite3_value from xFilter
 * @param outStmt - output sqlite3_stmt of chunks with all filters applied
 * @return int SQLITE_OK on success, error code otherwise
 */
int vec0_chunks_iter(vec0_vtab * p, const char * idxStr, int argc, sqlite3_value ** argv, sqlite3_stmt** outStmt) {
  // always null terminated, enforced by SQLite
  int idxStrLength = strlen(idxStr);
  // "1" refers to the initial vec0_query_plan char, 4 is the number of chars per "element"
  int numValueEntries = (idxStrLength-1) / 4;
  assert(argc == numValueEntries);

  int rc;
  sqlite3_str * s = sqlite3_str_new(NULL);
  sqlite3_str_appendf(s, "select chunk_id, validity, rowids "
                         " from " VEC0_SHADOW_CHUNKS_NAME,
                         p->schemaName, p->tableName);

  int appendedWhere = 0;
  for(int i = 0; i < numValueEntries; i++) {
    int idx = 1 + (i * 4);
    char kind = idxStr[idx + 0];
    if(kind != VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT) {
      continue;
    }

    int partition_idx = idxStr[idx + 1] - 'A';
    int operator = idxStr[idx + 2];
    // idxStr[idx + 3] is just null, a '_' placeholder

    if(!appendedWhere) {
      sqlite3_str_appendall(s, " WHERE ");
      appendedWhere = 1;
    }else {
      sqlite3_str_appendall(s, " AND ");
    }
    switch(operator) {
     case VEC0_PARTITION_OPERATOR_EQ:
      sqlite3_str_appendf(s, " partition%02d = ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_GT:
      sqlite3_str_appendf(s, " partition%02d > ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_LE:
      sqlite3_str_appendf(s, " partition%02d <= ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_LT:
      sqlite3_str_appendf(s, " partition%02d < ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_GE:
      sqlite3_str_appendf(s, " partition%02d >= ? ", partition_idx);
      break;
     case VEC0_PARTITION_OPERATOR_NE:
      sqlite3_str_appendf(s, " partition%02d != ? ", partition_idx);
      break;
     default: {
      char * zSql = sqlite3_str_finish(s);
      sqlite3_free(zSql);
      return SQLITE_ERROR;
     }

    }

  }

  char *zSql = sqlite3_str_finish(s);
  if (!zSql) {
    return SQLITE_NOMEM;
  }

  rc = sqlite3_prepare_v2(p->db, zSql, -1, outStmt, NULL);
  sqlite3_free(zSql);
  if(rc != SQLITE_OK) {
    return rc;
  }

  int n = 1;
  for(int i = 0; i < numValueEntries; i++) {
    int idx = 1 + (i * 4);
    char kind = idxStr[idx + 0];
    if(kind != VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT) {
      continue;
    }
    sqlite3_bind_value(*outStmt, n++, argv[i]);
  }

  return rc;
}

// a single `xxx in (...)` constraint on a metadata column. TEXT or INTEGER only for now.
struct Vec0MetadataIn{
  // index of argv[i]` the constraint is on
  int argv_idx;
  // metadata column index of the constraint, derived from idxStr + argv_idx
  int metadata_idx;
  // array of the copied `(...)` values from sqlite3_vtab_in_first()/sqlite3_vtab_in_next()
  struct Array array;
};

// Array elements for `xxx in (...)` values for a text column. basically just a string
struct Vec0MetadataInTextEntry {
  int n;
  char * zString;
};


int vec0_metadata_filter_text(vec0_vtab * p, sqlite3_value * value, const void * buffer, int size, vec0_metadata_operator op, u8* b, int metadata_idx, int chunk_rowid, struct Array * aMetadataIn, int argv_idx) {
  int rc;
  sqlite3_stmt * stmt = NULL;
  i64 * rowids = NULL;
  sqlite3_blob * rowidsBlob;
  const char * sTarget = (const char *) sqlite3_value_text(value);
  int nTarget = sqlite3_value_bytes(value);


  // TODO(perf): only text metadata news the rowids BLOB. Make it so that
  // rowids BLOB is re-used when multiple fitlers on text columns,
  // ex "name BETWEEN 'a' and 'b'""
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName, "rowids", chunk_rowid, 0, &rowidsBlob);
  if(rc != SQLITE_OK) {
    return rc;
  }
  assert(sqlite3_blob_bytes(rowidsBlob) % sizeof(i64) == 0);
  assert((sqlite3_blob_bytes(rowidsBlob) / sizeof(i64)) == size);

  rowids = sqlite3_malloc(sqlite3_blob_bytes(rowidsBlob));
  if(!rowids) {
    sqlite3_blob_close(rowidsBlob);
    return SQLITE_NOMEM;
  }

  rc = sqlite3_blob_read(rowidsBlob, rowids, sqlite3_blob_bytes(rowidsBlob), 0);
  if(rc != SQLITE_OK) {
    sqlite3_blob_close(rowidsBlob);
    return rc;
  }
  sqlite3_blob_close(rowidsBlob);

  switch(op) {
    int nPrefix;
    char * sPrefix;
    char *sFull;
    int nFull;
    u8 * view;
    case VEC0_METADATA_OPERATOR_EQ: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];

        // for EQ the text lengths must match
        if(nPrefix != nTarget) {
          bitmap_set(b, i, 0);
          continue;
        }
        int cmpPrefix = strncmp(sPrefix, sTarget, min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH));

        // for short strings, use the prefix comparison direclty
        if(nPrefix <= VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          bitmap_set(b, i, cmpPrefix == 0);
          continue;
        }
        // for EQ on longs strings, the prefix must match
        if(cmpPrefix) {
          bitmap_set(b, i, 0);
          continue;
        }
        // consult the full string
        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) == 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_NE: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];

        // for NE if text lengths dont match, it never will
        if(nPrefix != nTarget) {
          bitmap_set(b, i, 1);
          continue;
        }

        int cmpPrefix = strncmp(sPrefix, sTarget, min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH));

        // for short strings, use the prefix comparison direclty
        if(nPrefix <= VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          bitmap_set(b, i, cmpPrefix != 0);
          continue;
        }
        // for NE on longs strings, if prefixes dont match, then long string wont
        if(cmpPrefix) {
          bitmap_set(b, i, 1);
          continue;
        }
        // consult the full string
        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) != 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_GT: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        int cmpPrefix = strncmp(sPrefix, sTarget, min(min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH), nTarget));

        if(nPrefix < VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          // if prefix match, check which is longer
          if(cmpPrefix == 0) {
            bitmap_set(b, i, nPrefix > nTarget);
          }
          else {
            bitmap_set(b, i, cmpPrefix > 0);
          }
          continue;
        }
        // TODO(perf): may not need to compare full text in some cases

        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) > 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_GE: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        int cmpPrefix = strncmp(sPrefix, sTarget, min(min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH), nTarget));

        if(nPrefix < VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          // if prefix match, check which is longer
          if(cmpPrefix == 0) {
            bitmap_set(b, i, nPrefix >= nTarget);
          }
          else {
            bitmap_set(b, i, cmpPrefix >= 0);
          }
          continue;
        }
        // TODO(perf): may not need to compare full text in some cases

        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) >= 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_LE: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        int cmpPrefix = strncmp(sPrefix, sTarget, min(min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH), nTarget));

        if(nPrefix < VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          // if prefix match, check which is longer
          if(cmpPrefix == 0) {
            bitmap_set(b, i, nPrefix <= nTarget);
          }
          else {
            bitmap_set(b, i, cmpPrefix <= 0);
          }
          continue;
        }
        // TODO(perf): may not need to compare full text in some cases

        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) <= 0);
      }
      break;
    }
    case VEC0_METADATA_OPERATOR_LT: {
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        int cmpPrefix = strncmp(sPrefix, sTarget, min(min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH), nTarget));

        if(nPrefix < VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
          // if prefix match, check which is longer
          if(cmpPrefix == 0) {
            bitmap_set(b, i, nPrefix < nTarget);
          }
          else {
            bitmap_set(b, i, cmpPrefix < 0);
          }
          continue;
        }
        // TODO(perf): may not need to compare full text in some cases

        rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
        if(rc != SQLITE_OK) {
          goto done;
        }
        if(nPrefix != nFull) {
          rc = SQLITE_ERROR;
          goto done;
        }
        bitmap_set(b, i, strncmp(sFull, sTarget, nFull) < 0);
      }
      break;
    }

    case VEC0_METADATA_OPERATOR_IN: {
      size_t metadataInIdx = -1;
      for(size_t i = 0; i < aMetadataIn->length; i++) {
        struct Vec0MetadataIn * metadataIn = &(((struct Vec0MetadataIn *) aMetadataIn->z)[i]);
        if(metadataIn->argv_idx == argv_idx) {
          metadataInIdx = i;
          break;
        }
      }
      if(metadataInIdx < 0) {
        rc = SQLITE_ERROR;
        goto done;
      }

      struct Vec0MetadataIn * metadataIn = &((struct Vec0MetadataIn *) aMetadataIn->z)[metadataInIdx];
      struct Array * aTarget = &(metadataIn->array);


      int nPrefix;
      char * sPrefix;
      char *sFull;
      int nFull;
      u8 * view;
      for(int i = 0; i < size; i++) {
        view = &((u8*) buffer)[i * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
        nPrefix = ((int*) view)[0];
        sPrefix = (char *) &view[4];
        for(size_t target_idx = 0; target_idx < aTarget->length; target_idx++) {
          struct Vec0MetadataInTextEntry * entry = &(((struct Vec0MetadataInTextEntry*)aTarget->z)[target_idx]);
          if(entry->n != nPrefix) {
            continue;
          }
          int cmpPrefix = strncmp(sPrefix, entry->zString, min(nPrefix, VEC0_METADATA_TEXT_VIEW_DATA_LENGTH));
          if(nPrefix <= VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
            if(cmpPrefix == 0) {
              bitmap_set(b, i, 1);
              break;
            }
            continue;
          }
          if(cmpPrefix) {
            continue;
          }

          rc = vec0_get_metadata_text_long_value(p, &stmt, metadata_idx, rowids[i], &nFull, &sFull);
          if(rc != SQLITE_OK) {
            goto done;
          }
          if(nPrefix != nFull) {
            rc = SQLITE_ERROR;
            goto done;
          }
          if(strncmp(sFull, entry->zString, nFull) == 0) {
            bitmap_set(b, i, 1);
            break;
          }
        }
      }
      break;
    }

  }
  rc = SQLITE_OK;

  done:
    sqlite3_finalize(stmt);
    sqlite3_free(rowids);
    return rc;

}

/**
 * @brief Fill in bitmap of chunk values, whether or not the values match a metadata constraint
 *
 * @param p vec0_vtab
 * @param metadata_idx index of the metatadata column to perfrom constraints on
 * @param value sqlite3_value of the constraints value
 * @param blob sqlite3_blob that is already opened on the metdata column's shadow chunk table
 * @param chunk_rowid rowid of the chunk to calculate on
 * @param b pre-allocated and zero'd out bitmap to write results to
 * @param size size of the chunk
 * @return int SQLITE_OK on success, error code otherwise
 */
int vec0_set_metadata_filter_bitmap(
  vec0_vtab *p,
  int metadata_idx,
  vec0_metadata_operator op,
  sqlite3_value * value,
  sqlite3_blob * blob,
  i64 chunk_rowid,
  u8* b,
  int size,
  struct Array * aMetadataIn, int argv_idx) {
  // TODO: shouldn't this skip in-valid entries from the chunk's  validity bitmap?

  int rc;
  rc = sqlite3_blob_reopen(blob, chunk_rowid);
  if(rc != SQLITE_OK) {
    return rc;
  }

  vec0_metadata_column_kind kind = p->metadata_columns[metadata_idx].kind;
  int szMatch = 0;
  int blobSize = sqlite3_blob_bytes(blob);
  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      szMatch = blobSize == size / CHAR_BIT;
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      szMatch = blobSize == size * sizeof(i64);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      szMatch = blobSize == size * sizeof(double);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      szMatch = blobSize == size * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH;
      break;
    }
  }
  if(!szMatch) {
    return SQLITE_ERROR;
  }
  void * buffer = sqlite3_malloc(blobSize);
  if(!buffer) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_blob_read(blob, buffer, blobSize, 0);
  if(rc != SQLITE_OK) {
    goto done;
  }
  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      int target = sqlite3_value_int(value);
      if( (target && op == VEC0_METADATA_OPERATOR_EQ) || (!target && op == VEC0_METADATA_OPERATOR_NE)) {
        for(int i = 0; i < size; i++) { bitmap_set(b, i, bitmap_get((u8*) buffer, i)); }
      }
      else {
        for(int i = 0; i < size; i++) { bitmap_set(b, i, !bitmap_get((u8*) buffer, i)); }
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      i64 * array = (i64*) buffer;
      i64 target = sqlite3_value_int64(value);
      switch(op) {
        case VEC0_METADATA_OPERATOR_EQ: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] == target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_GT: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] > target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_LE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] <= target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_LT: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] < target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_GE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] >= target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_NE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] != target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_IN: {
          int metadataInIdx = -1;
          for(size_t i = 0; i < aMetadataIn->length; i++) {
            struct Vec0MetadataIn * metadataIn = &((struct Vec0MetadataIn *) aMetadataIn->z)[i];
            if(metadataIn->argv_idx == argv_idx) {
              metadataInIdx = i;
              break;
            }
          }
          if(metadataInIdx < 0) {
            rc = SQLITE_ERROR;
            goto done;
          }
          struct Vec0MetadataIn * metadataIn = &((struct Vec0MetadataIn *) aMetadataIn->z)[metadataInIdx];
          struct Array * aTarget = &(metadataIn->array);

          for(int i = 0; i < size; i++) {
            for(size_t target_idx = 0; target_idx < aTarget->length; target_idx++) {
              if( ((i64*)aTarget->z)[target_idx] == array[i]) {
                bitmap_set(b, i, 1);
                break;
              }
            }
          }
          break;
        }
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      double * array = (double*) buffer;
      double target = sqlite3_value_double(value);
      switch(op) {
        case VEC0_METADATA_OPERATOR_EQ: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] == target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_GT: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] > target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_LE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] <= target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_LT: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] < target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_GE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] >= target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_NE: {
          for(int i = 0; i < size; i++) { bitmap_set(b, i, array[i] != target); }
          break;
        }
        case VEC0_METADATA_OPERATOR_IN: {
          // should never be reached
          break;
        }
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      rc = vec0_metadata_filter_text(p, value, buffer, size, op, b, metadata_idx, chunk_rowid, aMetadataIn, argv_idx);
      if(rc != SQLITE_OK) {
        goto done;
      }
      break;
    }
  }
  done:
    sqlite3_free(buffer);
    return rc;
}

int vec0Filter_knn_chunks_iter(vec0_vtab *p, sqlite3_stmt *stmtChunks,
                               struct VectorColumnDefinition *vector_column,
                               int vectorColumnIdx, struct Array *arrayRowidsIn,
                               struct Array * aMetadataIn,
                               const char * idxStr, int argc, sqlite3_value ** argv,
                               void *queryVector, i64 k, i64 **out_topk_rowids,
                               f32 **out_topk_distances, i64 *out_used) {
  // for each chunk, get top min(k, chunk_size) rowid + distances to query vec.
  // then reconcile all topk_chunks for a true top k.
  // output only rowids + distances for now

  int rc = SQLITE_OK;
  sqlite3_blob *blobVectors = NULL;

  void *baseVectors = NULL; // memory: chunk_size * dimensions * element_size

  // OWNED BY CALLER ON SUCCESS
  i64 *topk_rowids = NULL; // memory: k * 4
  // OWNED BY CALLER ON SUCCESS
  f32 *topk_distances = NULL; // memory: k * 4

  i64 *tmp_topk_rowids = NULL;    // memory: k * 4
  f32 *tmp_topk_distances = NULL; // memory: k * 4
  f32 *chunk_distances = NULL;    // memory: chunk_size * 4
  u8 *b = NULL;                   // memory: chunk_size / 8
  u8 *bTaken = NULL;              // memory: chunk_size / 8
  i32 *chunk_topk_idxs = NULL;    // memory: k * 4
  u8 *bmRowids = NULL;            // memory: chunk_size / 8
  u8 *bmMetadata = NULL;            // memory: chunk_size / 8
  //                        // total: a lot???

  // 6 * (k * 4) + (k * 2) + (chunk_size / 8) + (chunk_size * dimensions * 4)

  topk_rowids = sqlite3_malloc(k * sizeof(i64));
  if (!topk_rowids) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  memset(topk_rowids, 0, k * sizeof(i64));

  topk_distances = sqlite3_malloc(k * sizeof(f32));
  if (!topk_distances) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  memset(topk_distances, 0, k * sizeof(f32));

  tmp_topk_rowids = sqlite3_malloc(k * sizeof(i64));
  if (!tmp_topk_rowids) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  memset(tmp_topk_rowids, 0, k * sizeof(i64));

  tmp_topk_distances = sqlite3_malloc(k * sizeof(f32));
  if (!tmp_topk_distances) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }
  memset(tmp_topk_distances, 0, k * sizeof(f32));

  i64 k_used = 0;
  i64 baseVectorsSize = p->chunk_size * vector_column_byte_size(*vector_column);
  baseVectors = sqlite3_malloc(baseVectorsSize);
  if (!baseVectors) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  chunk_distances = sqlite3_malloc(p->chunk_size * sizeof(f32));
  if (!chunk_distances) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  b = bitmap_new(p->chunk_size);
  if (!b) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  bTaken = bitmap_new(p->chunk_size);
  if (!bTaken) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  chunk_topk_idxs = sqlite3_malloc(k * sizeof(i32));
  if (!chunk_topk_idxs) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  bmRowids = arrayRowidsIn ? bitmap_new(p->chunk_size) : NULL;
  if (arrayRowidsIn && !bmRowids) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  sqlite3_blob * metadataBlobs[VEC0_MAX_METADATA_COLUMNS];
  memset(metadataBlobs, 0, sizeof(sqlite3_blob*) * VEC0_MAX_METADATA_COLUMNS);

  bmMetadata = bitmap_new(p->chunk_size);
  if(!bmMetadata) {
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  int idxStrLength = strlen(idxStr);
  int numValueEntries = (idxStrLength-1) / 4;
  assert(numValueEntries == argc);
  int hasMetadataFilters = 0;
  for(int i = 0; i < argc; i++) {
    int idx = 1 + (i * 4);
    char kind = idxStr[idx + 0];
    if(kind == VEC0_IDXSTR_KIND_METADATA_CONSTRAINT) {
      hasMetadataFilters = 1;
      break;
    }
  }

  while (true) {
    rc = sqlite3_step(stmtChunks);
    if (rc == SQLITE_DONE) {
      break;
    }
    if (rc != SQLITE_ROW) {
      vtab_set_error(&p->base, "chunks iter error");
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    memset(chunk_distances, 0, p->chunk_size * sizeof(f32));
    memset(chunk_topk_idxs, 0, k * sizeof(i32));
    bitmap_clear(b, p->chunk_size);

    i64 chunk_id = sqlite3_column_int64(stmtChunks, 0);
    unsigned char *chunkValidity =
        (unsigned char *)sqlite3_column_blob(stmtChunks, 1);
    i64 validitySize = sqlite3_column_bytes(stmtChunks, 1);
    if (validitySize != p->chunk_size / CHAR_BIT) {
      // IMP: V05271_22109
      vtab_set_error(
          &p->base,
          "chunk validity size doesn't match - expected %lld, found %lld",
          p->chunk_size / CHAR_BIT, validitySize);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    i64 *chunkRowids = (i64 *)sqlite3_column_blob(stmtChunks, 2);
    i64 rowidsSize = sqlite3_column_bytes(stmtChunks, 2);
    if (rowidsSize != p->chunk_size * sizeof(i64)) {
      // IMP: V02796_19635
      vtab_set_error(&p->base, "rowids size doesn't match");
      vtab_set_error(
          &p->base,
          "chunk rowids size doesn't match - expected %lld, found %lld",
          p->chunk_size * sizeof(i64), rowidsSize);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    // open the vector chunk blob for the current chunk
    rc = sqlite3_blob_open(p->db, p->schemaName,
                           p->shadowVectorChunksNames[vectorColumnIdx],
                           "vectors", chunk_id, 0, &blobVectors);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, "could not open vectors blob for chunk %lld",
                     chunk_id);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    i64 currentBaseVectorsSize = sqlite3_blob_bytes(blobVectors);
    i64 expectedBaseVectorsSize =
        p->chunk_size * vector_column_byte_size(*vector_column);
    if (currentBaseVectorsSize != expectedBaseVectorsSize) {
      // IMP: V16465_00535
      vtab_set_error(
          &p->base,
          "vectors blob size doesn't match - expected %lld, found %lld",
          expectedBaseVectorsSize, currentBaseVectorsSize);
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    rc = sqlite3_blob_read(blobVectors, baseVectors, currentBaseVectorsSize, 0);

    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, "vectors blob read error for %lld", chunk_id);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    bitmap_copy(b, chunkValidity, p->chunk_size);
    if (arrayRowidsIn) {
      bitmap_clear(bmRowids, p->chunk_size);

      for (int i = 0; i < p->chunk_size; i++) {
        if (!bitmap_get(chunkValidity, i)) {
          continue;
        }
        i64 rowid = chunkRowids[i];
        void *in = bsearch(&rowid, arrayRowidsIn->z, arrayRowidsIn->length,
                           sizeof(i64), _cmp);
        bitmap_set(bmRowids, i, in ? 1 : 0);
      }
      bitmap_and_inplace(b, bmRowids, p->chunk_size);
    }

    if(hasMetadataFilters) {
      for(int i = 0; i < argc; i++) {
        int idx = 1 + (i * 4);
        char kind = idxStr[idx + 0];
        if(kind != VEC0_IDXSTR_KIND_METADATA_CONSTRAINT) {
          continue;
        }
        int metadata_idx = idxStr[idx + 1] - 'A';
        int operator = idxStr[idx + 2];

        if(!metadataBlobs[metadata_idx]) {
          rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowMetadataChunksNames[metadata_idx], "data", chunk_id, 0, &metadataBlobs[metadata_idx]);
          vtab_set_error(&p->base, "Could not open metadata blob");
          if(rc != SQLITE_OK) {
            goto cleanup;
          }
        }

        bitmap_clear(bmMetadata, p->chunk_size);
        rc = vec0_set_metadata_filter_bitmap(p, metadata_idx, operator, argv[i], metadataBlobs[metadata_idx], chunk_id, bmMetadata, p->chunk_size, aMetadataIn, i);
        if(rc != SQLITE_OK) {
          vtab_set_error(&p->base, "Could not filter metadata fields");
          if(rc != SQLITE_OK) {
            goto cleanup;
          }
        }
        bitmap_and_inplace(b, bmMetadata, p->chunk_size);
      }
    }


    for (int i = 0; i < p->chunk_size; i++) {
      if (!bitmap_get(b, i)) {
        continue;
      };

      f32 result;
      switch (vector_column->element_type) {
      case SQLITE_VEC_ELEMENT_TYPE_FLOAT32: {
        const f32 *base_i =
            ((f32 *)baseVectors) + (i * vector_column->dimensions);
        switch (vector_column->distance_metric) {
        case VEC0_DISTANCE_METRIC_L2: {
          result = distance_l2_sqr_float(base_i, (f32 *)queryVector,
                                         &vector_column->dimensions);
          break;
        }
        case VEC0_DISTANCE_METRIC_L1: {
          result = distance_l1_f32(base_i, (f32 *)queryVector,
                                   &vector_column->dimensions);
          break;
        }
        case VEC0_DISTANCE_METRIC_COSINE: {
          result = distance_cosine_float(base_i, (f32 *)queryVector,
                                         &vector_column->dimensions);
          break;
        }
        }
        break;
      }
      case SQLITE_VEC_ELEMENT_TYPE_INT8: {
        const i8 *base_i =
            ((i8 *)baseVectors) + (i * vector_column->dimensions);
        switch (vector_column->distance_metric) {
        case VEC0_DISTANCE_METRIC_L2: {
          result = distance_l2_sqr_int8(base_i, (i8 *)queryVector,
                                        &vector_column->dimensions);
          break;
        }
        case VEC0_DISTANCE_METRIC_L1: {
          result = distance_l1_int8(base_i, (i8 *)queryVector,
                                    &vector_column->dimensions);
          break;
        }
        case VEC0_DISTANCE_METRIC_COSINE: {
          result = distance_cosine_int8(base_i, (i8 *)queryVector,
                                        &vector_column->dimensions);
          break;
        }
        }

        break;
      }
      case SQLITE_VEC_ELEMENT_TYPE_BIT: {
        const u8 *base_i =
            ((u8 *)baseVectors) + (i * (vector_column->dimensions / CHAR_BIT));
        result = distance_hamming(base_i, (u8 *)queryVector,
                                  &vector_column->dimensions);
        break;
      }
      }

      chunk_distances[i] = result;
    }

    int used1;
    min_idx(chunk_distances, p->chunk_size, b, chunk_topk_idxs,
            min(k, p->chunk_size), bTaken, &used1);

    i64 used;
    merge_sorted_lists(topk_distances, topk_rowids, k_used, chunk_distances,
                       chunkRowids, chunk_topk_idxs,
                       min(min(k, p->chunk_size), used1), tmp_topk_distances,
                       tmp_topk_rowids, k, &used);

    for (int i = 0; i < used; i++) {
      topk_rowids[i] = tmp_topk_rowids[i];
      topk_distances[i] = tmp_topk_distances[i];
    }
    k_used = used;
    // blobVectors is always opened with read-only permissions, so this never
    // fails.
    sqlite3_blob_close(blobVectors);
    blobVectors = NULL;
  }

  *out_topk_rowids = topk_rowids;
  *out_topk_distances = topk_distances;
  *out_used = k_used;
  rc = SQLITE_OK;

cleanup:
  if (rc != SQLITE_OK) {
    sqlite3_free(topk_rowids);
    sqlite3_free(topk_distances);
  }
  sqlite3_free(chunk_topk_idxs);
  sqlite3_free(tmp_topk_rowids);
  sqlite3_free(tmp_topk_distances);
  sqlite3_free(b);
  sqlite3_free(bTaken);
  sqlite3_free(bmRowids);
  sqlite3_free(baseVectors);
  sqlite3_free(chunk_distances);
  sqlite3_free(bmMetadata);
  for(int i = 0; i < VEC0_MAX_METADATA_COLUMNS; i++) {
    sqlite3_blob_close(metadataBlobs[i]);
  }
  // blobVectors is always opened with read-only permissions, so this never
  // fails.
  sqlite3_blob_close(blobVectors);
  return rc;
}

int vec0Filter_knn(vec0_cursor *pCur, vec0_vtab *p, int idxNum,
                   const char *idxStr, int argc, sqlite3_value **argv) {
  assert(argc == (strlen(idxStr)-1) / 4);
  int rc;
  struct vec0_query_knn_data *knn_data;

  int vectorColumnIdx = idxNum;
  struct VectorColumnDefinition *vector_column =
      &p->vector_columns[vectorColumnIdx];

  struct Array *arrayRowidsIn = NULL;
  sqlite3_stmt *stmtChunks = NULL;
  void *queryVector;
  size_t dimensions;
  enum VectorElementType elementType;
  vector_cleanup queryVectorCleanup = vector_cleanup_noop;
  char *pzError;
  knn_data = sqlite3_malloc(sizeof(*knn_data));
  if (!knn_data) {
    return SQLITE_NOMEM;
  }
  memset(knn_data, 0, sizeof(*knn_data));
  // array of `struct Vec0MetadataIn`, IF there are any `xxx in (...)` metadata constraints
  struct Array * aMetadataIn = NULL;

  int query_idx =-1;
  int k_idx = -1;
  int rowid_in_idx = -1;
  for(int i = 0; i < argc; i++) {
    if(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_KNN_MATCH) {
      query_idx = i;
    }
    if(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_KNN_K) {
      k_idx = i;
    }
    if(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_KNN_ROWID_IN) {
      rowid_in_idx = i;
    }
  }
  assert(query_idx >= 0);
  assert(k_idx >= 0);

  // make sure the query vector matches the vector column (type dimensions etc.)
  rc = vector_from_value(argv[query_idx], &queryVector, &dimensions, &elementType,
                         &queryVectorCleanup, &pzError);

  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base,
                   "Query vector on the \"%.*s\" column is invalid: %z",
                   vector_column->name_length, vector_column->name, pzError);
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if (elementType != vector_column->element_type) {
    vtab_set_error(
        &p->base,
        "Query vector for the \"%.*s\" column is expected to be of type "
        "%s, but a %s vector was provided.",
        vector_column->name_length, vector_column->name,
        vector_subtype_name(vector_column->element_type),
        vector_subtype_name(elementType));
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if (dimensions != vector_column->dimensions) {
    vtab_set_error(
        &p->base,
        "Dimension mismatch for query vector for the \"%.*s\" column. "
        "Expected %d dimensions but received %d.",
        vector_column->name_length, vector_column->name,
        vector_column->dimensions, dimensions);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  i64 k = sqlite3_value_int64(argv[k_idx]);
  if (k < 0) {
    vtab_set_error(
        &p->base, "k value in knn queries must be greater than or equal to 0.");
    rc = SQLITE_ERROR;
    goto cleanup;
  }
#define SQLITE_VEC_VEC0_K_MAX 4096
  if (k > SQLITE_VEC_VEC0_K_MAX) {
    vtab_set_error(
        &p->base,
        "k value in knn query too large, provided %lld and the limit is %lld",
        k, SQLITE_VEC_VEC0_K_MAX);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  if (k == 0) {
    knn_data->k = 0;
    pCur->knn_data = knn_data;
    pCur->query_plan = VEC0_QUERY_PLAN_KNN;
    rc = SQLITE_OK;
    goto cleanup;
  }

// handle when a `rowid in (...)` operation was provided
// Array of all the rowids that appear in any `rowid in (...)` constraint.
// NULL if none were provided, which means a "full" scan.
#if COMPILER_SUPPORTS_VTAB_IN
  if (rowid_in_idx >= 0) {
    sqlite3_value *item;
    int rc;
    arrayRowidsIn = sqlite3_malloc(sizeof(*arrayRowidsIn));
    if (!arrayRowidsIn) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    memset(arrayRowidsIn, 0, sizeof(*arrayRowidsIn));

    rc = array_init(arrayRowidsIn, sizeof(i64), 32);
    if (rc != SQLITE_OK) {
      goto cleanup;
    }
    for (rc = sqlite3_vtab_in_first(argv[rowid_in_idx], &item); rc == SQLITE_OK && item;
         rc = sqlite3_vtab_in_next(argv[rowid_in_idx], &item)) {
      i64 rowid;
      if (p->pkIsText) {
        rc = vec0_rowid_from_id(p, item, &rowid);
        if (rc != SQLITE_OK) {
          goto cleanup;
        }
      } else {
        rowid = sqlite3_value_int64(item);
      }
      rc = array_append(arrayRowidsIn, &rowid);
      if (rc != SQLITE_OK) {
        goto cleanup;
      }
    }
    if (rc != SQLITE_DONE) {
      vtab_set_error(&p->base, "error processing rowid in (...) array");
      goto cleanup;
    }
    qsort(arrayRowidsIn->z, arrayRowidsIn->length, arrayRowidsIn->element_size,
          _cmp);
  }
#endif

  #if COMPILER_SUPPORTS_VTAB_IN
  for(int i = 0; i < argc; i++) {
    if(!(idxStr[1 + (i*4)] == VEC0_IDXSTR_KIND_METADATA_CONSTRAINT && idxStr[1 + (i*4) + 2] == VEC0_METADATA_OPERATOR_IN)) {
      continue;
    }
    int metadata_idx = idxStr[1 + (i*4) + 1]  - 'A';
    if(!aMetadataIn) {
      aMetadataIn = sqlite3_malloc(sizeof(*aMetadataIn));
      if(!aMetadataIn) {
        rc = SQLITE_NOMEM;
        goto cleanup;
      }
      memset(aMetadataIn, 0, sizeof(*aMetadataIn));
      rc = array_init(aMetadataIn, sizeof(struct Vec0MetadataIn), 8);
      if(rc != SQLITE_OK) {
        goto cleanup;
      }
    }

    struct Vec0MetadataIn item;
    memset(&item, 0, sizeof(item));
    item.metadata_idx=metadata_idx;
    item.argv_idx = i;

    switch(p->metadata_columns[metadata_idx].kind) {
      case VEC0_METADATA_COLUMN_KIND_INTEGER: {
        rc = array_init(&item.array, sizeof(i64), 16);
        if(rc != SQLITE_OK) {
          goto cleanup;
        }
        sqlite3_value *entry;
        for (rc = sqlite3_vtab_in_first(argv[i], &entry); rc == SQLITE_OK && entry; rc = sqlite3_vtab_in_next(argv[i], &entry)) {
          i64 v = sqlite3_value_int64(entry);
          rc = array_append(&item.array, &v);
          if (rc != SQLITE_OK) {
            goto cleanup;
          }
        }

        if (rc != SQLITE_DONE) {
          vtab_set_error(&p->base, "Error fetching next value in `x in (...)` integer expression");
          goto cleanup;
        }

        break;
      }
      case VEC0_METADATA_COLUMN_KIND_TEXT: {
        rc = array_init(&item.array, sizeof(struct Vec0MetadataInTextEntry), 16);
        if(rc != SQLITE_OK) {
          goto cleanup;
        }
        sqlite3_value *entry;
        for (rc = sqlite3_vtab_in_first(argv[i], &entry); rc == SQLITE_OK && entry; rc = sqlite3_vtab_in_next(argv[i], &entry)) {
          const char * s = (const char *) sqlite3_value_text(entry);
          int n = sqlite3_value_bytes(entry);

          struct Vec0MetadataInTextEntry entry;
          entry.zString = sqlite3_mprintf("%.*s", n, s);
          if(!entry.zString) {
            rc = SQLITE_NOMEM;
            goto cleanup;
          }
          entry.n = n;
          rc = array_append(&item.array, &entry);
          if (rc != SQLITE_OK) {
            goto cleanup;
          }
        }

        if (rc != SQLITE_DONE) {
          vtab_set_error(&p->base, "Error fetching next value in `x in (...)` text expression");
          goto cleanup;
        }

        break;
      }
      default: {
        vtab_set_error(&p->base, "Internal sqlite-vec error");
        goto cleanup;
      }
    }

    rc = array_append(aMetadataIn, &item);
    if(rc != SQLITE_OK) {
      goto cleanup;
    }
  }
  #endif

  rc = vec0_chunks_iter(p, idxStr, argc, argv, &stmtChunks);
  if (rc != SQLITE_OK) {
    // IMP: V06942_23781
    vtab_set_error(&p->base, "Error preparing stmtChunk: %s",
                   sqlite3_errmsg(p->db));
    goto cleanup;
  }

  i64 *topk_rowids = NULL;
  f32 *topk_distances = NULL;
  i64 k_used = 0;
  rc = vec0Filter_knn_chunks_iter(p, stmtChunks, vector_column, vectorColumnIdx,
                                  arrayRowidsIn, aMetadataIn, idxStr, argc, argv, queryVector, k, &topk_rowids,
                                  &topk_distances, &k_used);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }

  knn_data->current_idx = 0;
  knn_data->k = k;
  knn_data->rowids = topk_rowids;
  knn_data->distances = topk_distances;
  knn_data->k_used = k_used;

  pCur->knn_data = knn_data;
  pCur->query_plan = VEC0_QUERY_PLAN_KNN;
  rc = SQLITE_OK;

cleanup:
  sqlite3_finalize(stmtChunks);
  array_cleanup(arrayRowidsIn);
  sqlite3_free(arrayRowidsIn);
  queryVectorCleanup(queryVector);
  if(aMetadataIn) {
    for(size_t i = 0; i < aMetadataIn->length; i++) {
      struct Vec0MetadataIn* item = &((struct Vec0MetadataIn *) aMetadataIn->z)[i];
      for(size_t j = 0; j < item->array.length; j++) {
        if(p->metadata_columns[item->metadata_idx].kind == VEC0_METADATA_COLUMN_KIND_TEXT) {
          struct Vec0MetadataInTextEntry entry = ((struct Vec0MetadataInTextEntry*)item->array.z)[j];
          sqlite3_free(entry.zString);
        }
      }
      array_cleanup(&item->array);
    }
    array_cleanup(aMetadataIn);
  }

  sqlite3_free(aMetadataIn);

  return rc;
}

int vec0Filter_fullscan(vec0_vtab *p, vec0_cursor *pCur) {
  int rc;
  char *zSql;
  struct vec0_query_fullscan_data *fullscan_data;

  fullscan_data = sqlite3_malloc(sizeof(*fullscan_data));
  if (!fullscan_data) {
    return SQLITE_NOMEM;
  }
  memset(fullscan_data, 0, sizeof(*fullscan_data));

  zSql = sqlite3_mprintf(" SELECT rowid "
                         " FROM " VEC0_SHADOW_ROWIDS_NAME
                         " ORDER by chunk_id, chunk_offset ",
                         p->schemaName, p->tableName);
  if (!zSql) {
    rc = SQLITE_NOMEM;
    goto error;
  }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &fullscan_data->rowids_stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    // IMP: V09901_26739
    vtab_set_error(&p->base, "Error preparing rowid scan: %s",
                   sqlite3_errmsg(p->db));
    goto error;
  }

  rc = sqlite3_step(fullscan_data->rowids_stmt);

  // DONE when there's no rowids, ROW when there are, both "success"
  if (!(rc == SQLITE_ROW || rc == SQLITE_DONE)) {
    goto error;
  }

  fullscan_data->done = rc == SQLITE_DONE;
  pCur->query_plan = VEC0_QUERY_PLAN_FULLSCAN;
  pCur->fullscan_data = fullscan_data;
  return SQLITE_OK;

error:
  vec0_query_fullscan_data_clear(fullscan_data);
  sqlite3_free(fullscan_data);
  return rc;
}

int vec0Filter_point(vec0_cursor *pCur, vec0_vtab *p, int argc,
                     sqlite3_value **argv) {
  int rc;
  assert(argc == 1);
  i64 rowid;
  struct vec0_query_point_data *point_data = NULL;

  point_data = sqlite3_malloc(sizeof(*point_data));
  if (!point_data) {
    rc = SQLITE_NOMEM;
    goto error;
  }
  memset(point_data, 0, sizeof(*point_data));

  if (p->pkIsText) {
    rc = vec0_rowid_from_id(p, argv[0], &rowid);
    if (rc == SQLITE_EMPTY) {
      goto eof;
    }
    if (rc != SQLITE_OK) {
      goto error;
    }
  } else {
    rowid = sqlite3_value_int64(argv[0]);
  }

  for (int i = 0; i < p->numVectorColumns; i++) {
    rc = vec0_get_vector_data(p, rowid, i, &point_data->vectors[i], NULL);
    if (rc == SQLITE_EMPTY) {
      goto eof;
    }
    if (rc != SQLITE_OK) {
      goto error;
    }
  }

  point_data->rowid = rowid;
  point_data->done = 0;
  pCur->point_data = point_data;
  pCur->query_plan = VEC0_QUERY_PLAN_POINT;
  return SQLITE_OK;

eof:
  point_data->rowid = rowid;
  point_data->done = 1;
  pCur->point_data = point_data;
  pCur->query_plan = VEC0_QUERY_PLAN_POINT;
  return SQLITE_OK;

error:
  vec0_query_point_data_clear(point_data);
  sqlite3_free(point_data);
  return rc;
}

static int vec0Filter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
  vec0_vtab *p = (vec0_vtab *)pVtabCursor->pVtab;
  vec0_cursor *pCur = (vec0_cursor *)pVtabCursor;
  vec0_cursor_clear(pCur);

  int idxStrLength = strlen(idxStr);
  if(idxStrLength <= 0) {
    return SQLITE_ERROR;
  }
  if((idxStrLength-1) % 4 != 0) {
    return SQLITE_ERROR;
  }
  int numValueEntries = (idxStrLength-1) / 4;
  if(numValueEntries != argc) {
    return SQLITE_ERROR;
  }

  char query_plan = idxStr[0];
  switch(query_plan) {
    case VEC0_QUERY_PLAN_FULLSCAN:
      return vec0Filter_fullscan(p, pCur);
    case VEC0_QUERY_PLAN_KNN:
      return vec0Filter_knn(pCur, p, idxNum, idxStr, argc, argv);
    case VEC0_QUERY_PLAN_POINT:
      return vec0Filter_point(pCur, p, argc, argv);
    default:
      vtab_set_error(pVtabCursor->pVtab, "unknown idxStr '%s'", idxStr);
      return SQLITE_ERROR;
  }
}

static int vec0Rowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  switch (pCur->query_plan) {
  case VEC0_QUERY_PLAN_FULLSCAN: {
    *pRowid = sqlite3_column_int64(pCur->fullscan_data->rowids_stmt, 0);
    return SQLITE_OK;
  }
  case VEC0_QUERY_PLAN_POINT: {
    *pRowid = pCur->point_data->rowid;
    return SQLITE_OK;
  }
  case VEC0_QUERY_PLAN_KNN: {
    vtab_set_error(cur->pVtab,
                   "Internal sqlite-vec error: expected point query plan in "
                   "vec0Rowid, found %d",
                   pCur->query_plan);
    return SQLITE_ERROR;
  }
  }
  return SQLITE_ERROR;
}

static int vec0Next(sqlite3_vtab_cursor *cur) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  switch (pCur->query_plan) {
  case VEC0_QUERY_PLAN_FULLSCAN: {
    if (!pCur->fullscan_data) {
      return SQLITE_ERROR;
    }
    int rc = sqlite3_step(pCur->fullscan_data->rowids_stmt);
    if (rc == SQLITE_DONE) {
      pCur->fullscan_data->done = 1;
      return SQLITE_OK;
    }
    if (rc == SQLITE_ROW) {
      return SQLITE_OK;
    }
    return SQLITE_ERROR;
  }
  case VEC0_QUERY_PLAN_KNN: {
    if (!pCur->knn_data) {
      return SQLITE_ERROR;
    }

    pCur->knn_data->current_idx++;
    return SQLITE_OK;
  }
  case VEC0_QUERY_PLAN_POINT: {
    if (!pCur->point_data) {
      return SQLITE_ERROR;
    }
    pCur->point_data->done = 1;
    return SQLITE_OK;
  }
  }
  return SQLITE_ERROR;
}

static int vec0Eof(sqlite3_vtab_cursor *cur) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  switch (pCur->query_plan) {
  case VEC0_QUERY_PLAN_FULLSCAN: {
    if (!pCur->fullscan_data) {
      return 1;
    }
    return pCur->fullscan_data->done;
  }
  case VEC0_QUERY_PLAN_KNN: {
    if (!pCur->knn_data) {
      return 1;
    }
    // return (pCur->knn_data->current_idx >= pCur->knn_data->k) ||
    // (pCur->knn_data->distances[pCur->knn_data->current_idx] == FLT_MAX);
    return (pCur->knn_data->current_idx >= pCur->knn_data->k_used);
  }
  case VEC0_QUERY_PLAN_POINT: {
    if (!pCur->point_data) {
      return 1;
    }
    return pCur->point_data->done;
  }
  }
  return 1;
}

static int vec0Column_fullscan(vec0_vtab *pVtab, vec0_cursor *pCur,
                               sqlite3_context *context, int i) {
  if (!pCur->fullscan_data) {
    sqlite3_result_error(
        context, "Internal sqlite-vec error: fullscan_data is NULL.", -1);
    return SQLITE_ERROR;
  }
  i64 rowid = sqlite3_column_int64(pCur->fullscan_data->rowids_stmt, 0);
  if (i == VEC0_COLUMN_ID) {
    return vec0_result_id(pVtab, context, rowid);
  }
  else if (vec0_column_idx_is_vector(pVtab, i)) {
    void *v;
    int sz;
    int vector_idx = vec0_column_idx_to_vector_idx(pVtab, i);
    int rc = vec0_get_vector_data(pVtab, rowid, vector_idx, &v, &sz);
    if (rc != SQLITE_OK) {
      return rc;
    }
    sqlite3_result_blob(context, v, sz, sqlite3_free);
    sqlite3_result_subtype(context,
                           pVtab->vector_columns[vector_idx].element_type);

  }
  else if (i == vec0_column_distance_idx(pVtab)) {
    sqlite3_result_null(context);
  }
  else if(vec0_column_idx_is_partition(pVtab, i)) {
    int partition_idx = vec0_column_idx_to_partition_idx(pVtab, i);
    sqlite3_value * v;
    int rc = vec0_get_partition_value_for_rowid(pVtab, rowid, partition_idx, &v);
    if(rc == SQLITE_OK) {
      sqlite3_result_value(context, v);
      sqlite3_value_free(v);
    }else {
      sqlite3_result_error_code(context, rc);
    }
  }
  else if(vec0_column_idx_is_auxiliary(pVtab, i)) {
    int auxiliary_idx = vec0_column_idx_to_auxiliary_idx(pVtab, i);
    sqlite3_value * v;
    int rc = vec0_get_auxiliary_value_for_rowid(pVtab, rowid, auxiliary_idx, &v);
    if(rc == SQLITE_OK) {
      sqlite3_result_value(context, v);
      sqlite3_value_free(v);
    }else {
      sqlite3_result_error_code(context, rc);
    }
  }

  else if(vec0_column_idx_is_metadata(pVtab, i)) {
    if(sqlite3_vtab_nochange(context)) {
      return SQLITE_OK;
    }
    int metadata_idx = vec0_column_idx_to_metadata_idx(pVtab, i);
    int rc = vec0_result_metadata_value_for_rowid(pVtab, rowid, metadata_idx, context);
    if(rc != SQLITE_OK) {
      // IMP: V15466_32305
      const char * zErr = sqlite3_mprintf(
        "Could not extract metadata value for column %.*s at rowid %lld",
        pVtab->metadata_columns[metadata_idx].name_length,
        pVtab->metadata_columns[metadata_idx].name, rowid
      );
      if(zErr) {
        sqlite3_result_error(context, zErr, -1);
        sqlite3_free((void *) zErr);
      }else {
        sqlite3_result_error_nomem(context);
      }
    }
  }

  return SQLITE_OK;
}

static int vec0Column_point(vec0_vtab *pVtab, vec0_cursor *pCur,
                            sqlite3_context *context, int i) {
  if (!pCur->point_data) {
    sqlite3_result_error(context,
                         "Internal sqlite-vec error: point_data is NULL.", -1);
    return SQLITE_ERROR;
  }
  if (i == VEC0_COLUMN_ID) {
    return vec0_result_id(pVtab, context, pCur->point_data->rowid);
  }
  else if (i == vec0_column_distance_idx(pVtab)) {
    sqlite3_result_null(context);
    return SQLITE_OK;
  }
  else if (vec0_column_idx_is_vector(pVtab, i)) {
    if (sqlite3_vtab_nochange(context)) {
      sqlite3_result_null(context);
      return SQLITE_OK;
    }
    int vector_idx = vec0_column_idx_to_vector_idx(pVtab, i);
    sqlite3_result_blob(
        context, pCur->point_data->vectors[vector_idx],
        vector_column_byte_size(pVtab->vector_columns[vector_idx]),
        SQLITE_TRANSIENT);
    sqlite3_result_subtype(context,
                           pVtab->vector_columns[vector_idx].element_type);
    return SQLITE_OK;
  }
  else if(vec0_column_idx_is_partition(pVtab, i)) {
    if(sqlite3_vtab_nochange(context)) {
      return SQLITE_OK;
    }
    int partition_idx = vec0_column_idx_to_partition_idx(pVtab, i);
    i64 rowid = pCur->point_data->rowid;
    sqlite3_value * v;
    int rc = vec0_get_partition_value_for_rowid(pVtab, rowid, partition_idx, &v);
    if(rc == SQLITE_OK) {
      sqlite3_result_value(context, v);
      sqlite3_value_free(v);
    }else {
      sqlite3_result_error_code(context, rc);
    }
  }
  else if(vec0_column_idx_is_auxiliary(pVtab, i)) {
    if(sqlite3_vtab_nochange(context)) {
      return SQLITE_OK;
    }
    i64 rowid = pCur->point_data->rowid;
    int auxiliary_idx = vec0_column_idx_to_auxiliary_idx(pVtab, i);
    sqlite3_value * v;
    int rc = vec0_get_auxiliary_value_for_rowid(pVtab, rowid, auxiliary_idx, &v);
    if(rc == SQLITE_OK) {
      sqlite3_result_value(context, v);
      sqlite3_value_free(v);
    }else {
      sqlite3_result_error_code(context, rc);
    }
  }

  else if(vec0_column_idx_is_metadata(pVtab, i)) {
    if(sqlite3_vtab_nochange(context)) {
      return SQLITE_OK;
    }
    i64 rowid = pCur->point_data->rowid;
    int metadata_idx = vec0_column_idx_to_metadata_idx(pVtab, i);
    int rc = vec0_result_metadata_value_for_rowid(pVtab, rowid, metadata_idx, context);
    if(rc != SQLITE_OK) {
      const char * zErr = sqlite3_mprintf(
        "Could not extract metadata value for column %.*s at rowid %lld",
        pVtab->metadata_columns[metadata_idx].name_length,
        pVtab->metadata_columns[metadata_idx].name, rowid
      );
      if(zErr) {
        sqlite3_result_error(context, zErr, -1);
        sqlite3_free((void *) zErr);
      }else {
        sqlite3_result_error_nomem(context);
      }
    }
  }

  return SQLITE_OK;
}

static int vec0Column_knn(vec0_vtab *pVtab, vec0_cursor *pCur,
                          sqlite3_context *context, int i) {
  if (!pCur->knn_data) {
    sqlite3_result_error(context,
                         "Internal sqlite-vec error: knn_data is NULL.", -1);
    return SQLITE_ERROR;
  }
  if (i == VEC0_COLUMN_ID) {
    i64 rowid = pCur->knn_data->rowids[pCur->knn_data->current_idx];
    return vec0_result_id(pVtab, context, rowid);
  }
  else if (i == vec0_column_distance_idx(pVtab)) {
    sqlite3_result_double(
        context, pCur->knn_data->distances[pCur->knn_data->current_idx]);
    return SQLITE_OK;
  }
  else if (vec0_column_idx_is_vector(pVtab, i)) {
    void *out;
    int sz;
    int vector_idx = vec0_column_idx_to_vector_idx(pVtab, i);
    int rc = vec0_get_vector_data(
        pVtab, pCur->knn_data->rowids[pCur->knn_data->current_idx], vector_idx,
        &out, &sz);
    if (rc != SQLITE_OK) {
      return rc;
    }
    sqlite3_result_blob(context, out, sz, sqlite3_free);
    sqlite3_result_subtype(context,
                           pVtab->vector_columns[vector_idx].element_type);
    return SQLITE_OK;
  }
  else if(vec0_column_idx_is_partition(pVtab, i)) {
    int partition_idx = vec0_column_idx_to_partition_idx(pVtab, i);
    i64 rowid = pCur->knn_data->rowids[pCur->knn_data->current_idx];
    sqlite3_value * v;
    int rc = vec0_get_partition_value_for_rowid(pVtab, rowid, partition_idx, &v);
    if(rc == SQLITE_OK) {
      sqlite3_result_value(context, v);
      sqlite3_value_free(v);
    }else {
      sqlite3_result_error_code(context, rc);
    }
  }
  else if(vec0_column_idx_is_auxiliary(pVtab, i)) {
    int auxiliary_idx = vec0_column_idx_to_auxiliary_idx(pVtab, i);
    i64 rowid = pCur->knn_data->rowids[pCur->knn_data->current_idx];
    sqlite3_value * v;
    int rc = vec0_get_auxiliary_value_for_rowid(pVtab, rowid, auxiliary_idx, &v);
    if(rc == SQLITE_OK) {
      sqlite3_result_value(context, v);
      sqlite3_value_free(v);
    }else {
      sqlite3_result_error_code(context, rc);
    }
  }

  else if(vec0_column_idx_is_metadata(pVtab, i)) {
    int metadata_idx = vec0_column_idx_to_metadata_idx(pVtab, i);
    i64 rowid = pCur->knn_data->rowids[pCur->knn_data->current_idx];
    int rc = vec0_result_metadata_value_for_rowid(pVtab, rowid, metadata_idx, context);
    if(rc != SQLITE_OK) {
      const char * zErr = sqlite3_mprintf(
        "Could not extract metadata value for column %.*s at rowid %lld",
        pVtab->metadata_columns[metadata_idx].name_length,
        pVtab->metadata_columns[metadata_idx].name, rowid
      );
      if(zErr) {
        sqlite3_result_error(context, zErr, -1);
        sqlite3_free((void *) zErr);
      }else {
        sqlite3_result_error_nomem(context);
      }
    }
  }

  return SQLITE_OK;
}

static int vec0Column(sqlite3_vtab_cursor *cur, sqlite3_context *context,
                      int i) {
  vec0_cursor *pCur = (vec0_cursor *)cur;
  vec0_vtab *pVtab = (vec0_vtab *)cur->pVtab;
  switch (pCur->query_plan) {
  case VEC0_QUERY_PLAN_FULLSCAN: {
    return vec0Column_fullscan(pVtab, pCur, context, i);
  }
  case VEC0_QUERY_PLAN_KNN: {
    return vec0Column_knn(pVtab, pCur, context, i);
  }
  case VEC0_QUERY_PLAN_POINT: {
    return vec0Column_point(pVtab, pCur, context, i);
  }
  }
  return SQLITE_OK;
}

/**
 * @brief Handles the "insert rowid" step of a row insert operation of a vec0
 * table.
 *
 * This function will insert a new row into the _rowids vec0 shadow table.
 *
 * @param p: virtual table
 * @param idValue: Value containing the inserted rowid/id value.
 * @param rowid: Output rowid, will point to the "real" i64 rowid
 * value that was inserted
 * @return int SQLITE_OK on success, error code on failure
 */
int vec0Update_InsertRowidStep(vec0_vtab *p, sqlite3_value *idValue,
                               i64 *rowid) {

  /**
   * An insert into a vec0 table can happen a few different ways:
   *  1) With default INTEGER primary key: With a supplied i64 rowid
   *  2) With default INTEGER primary key: WITHOUT a supplied rowid
   *  3) With TEXT primary key: supplied text rowid
   */

  int rc;

  // Option 3: vtab has a user-defined TEXT primary key, so ensure a text value
  // is provided.
  if (p->pkIsText) {
    if (sqlite3_value_type(idValue) != SQLITE_TEXT) {
      // IMP: V04200_21039
      vtab_set_error(&p->base,
                     "The %s virtual table was declared with a TEXT primary "
                     "key, but a non-TEXT value was provided in an INSERT.",
                     p->tableName);
      return SQLITE_ERROR;
    }

    return vec0_rowids_insert_id(p, idValue, rowid);
  }

  // Option 1: User supplied a i64 rowid
  if (sqlite3_value_type(idValue) == SQLITE_INTEGER) {
    i64 suppliedRowid = sqlite3_value_int64(idValue);
    rc = vec0_rowids_insert_rowid(p, suppliedRowid);
    if (rc == SQLITE_OK) {
      *rowid = suppliedRowid;
    }
    return rc;
  }

  // Option 2: User did not suppled a rowid

  if (sqlite3_value_type(idValue) != SQLITE_NULL) {
    // IMP: V30855_14925
    vtab_set_error(&p->base,
                   "Only integers are allows for primary key values on %s",
                   p->tableName);
    return SQLITE_ERROR;
  }
  // NULL to get next auto-incremented value
  return vec0_rowids_insert_id(p, NULL, rowid);
}

/**
 * @brief Determines the "next available" chunk position for a newly inserted
 * vec0 row.
 *
 * This operation may insert a new "blank" chunk the _chunks table, if there is
 * no more space in previous chunks.
 *
 * @param p: virtual table
 * @param partitionKeyValues: array of partition key column values, to constrain
 * against any partition key columns.
 * @param chunk_rowid: Output rowid of the chunk in the _chunks virtual table
 * that has the avialabiity.
 * @param chunk_offset: Output the index of the available space insert the
 * chunk, based on the index of the first available validity bit.
 * @param pBlobValidity: Output blob of the validity column of the available
 * chunk. Will be opened with read/write permissions.
 * @param pValidity: Output buffer of the original chunk's validity column.
 *    Needs to be cleaned up with sqlite3_free().
 * @return int SQLITE_OK on success, error code on failure
 */
int vec0Update_InsertNextAvailableStep(
    vec0_vtab *p,
    sqlite3_value ** partitionKeyValues,
    i64 *chunk_rowid, i64 *chunk_offset,
    sqlite3_blob **blobChunksValidity,
    const unsigned char **bufferChunksValidity) {

  int rc;
  i64 validitySize;
  *chunk_offset = -1;

  rc = vec0_get_latest_chunk_rowid(p, chunk_rowid, partitionKeyValues);
  if(rc == SQLITE_EMPTY) {
    goto done;
  }
  if (rc != SQLITE_OK) {
    goto cleanup;
  }

  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName, "validity",
                         *chunk_rowid, 1, blobChunksValidity);
  if (rc != SQLITE_OK) {
    // IMP: V22053_06123
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "could not open validity blob on %s.%s.%lld",
                   p->schemaName, p->shadowChunksName, *chunk_rowid);
    goto cleanup;
  }

  validitySize = sqlite3_blob_bytes(*blobChunksValidity);
  if (validitySize != p->chunk_size / CHAR_BIT) {
    // IMP: V29362_13432
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "validity blob size mismatch on "
                   "%s.%s.%lld, expected %lld but received %lld.",
                   p->schemaName, p->shadowChunksName, *chunk_rowid,
                   (i64)(p->chunk_size / CHAR_BIT), validitySize);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  *bufferChunksValidity = sqlite3_malloc(validitySize);
  if (!(*bufferChunksValidity)) {
    vtab_set_error(&p->base, VEC_INTERAL_ERROR
                   "Could not allocate memory for validity bitmap");
    rc = SQLITE_NOMEM;
    goto cleanup;
  }

  rc = sqlite3_blob_read(*blobChunksValidity, (void *)*bufferChunksValidity,
                         validitySize, 0);

  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR
                   "Could not read validity bitmap for %s.%s.%lld",
                   p->schemaName, p->shadowChunksName, *chunk_rowid);
    goto cleanup;
  }

  // find the next available offset, ie first `0` in the bitmap.
  for (int i = 0; i < validitySize; i++) {
    if ((*bufferChunksValidity)[i] == 0b11111111)
      continue;
    for (int j = 0; j < CHAR_BIT; j++) {
      if (((((*bufferChunksValidity)[i] >> j) & 1) == 0)) {
        *chunk_offset = (i * CHAR_BIT) + j;
        goto done;
      }
    }
  }

done:
  // latest chunk was full, so need to create a new one
  if (*chunk_offset == -1) {
    rc = vec0_new_chunk(p, partitionKeyValues, chunk_rowid);
    if (rc != SQLITE_OK) {
      // IMP: V08441_25279
      vtab_set_error(&p->base,
                     VEC_INTERAL_ERROR "Could not insert a new vector chunk");
      rc = SQLITE_ERROR; // otherwise raises a DatabaseError and not operational
                         // error?
      goto cleanup;
    }
    *chunk_offset = 0;

    // blobChunksValidity and pValidity are stale, pointing to the previous
    // (full) chunk. to re-assign them
    rc = sqlite3_blob_close(*blobChunksValidity);
    sqlite3_free((void *)*bufferChunksValidity);
    *blobChunksValidity = NULL;
    *bufferChunksValidity = NULL;
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, VEC_INTERAL_ERROR
                     "unknown error, blobChunksValidity could not be closed, "
                     "please file an issue.");
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName,
                           "validity", *chunk_rowid, 1, blobChunksValidity);
    if (rc != SQLITE_OK) {
      vtab_set_error(
          &p->base,
          VEC_INTERAL_ERROR
          "Could not open validity blob for newly created chunk %s.%s.%lld",
          p->schemaName, p->shadowChunksName, *chunk_rowid);
      goto cleanup;
    }
    validitySize = sqlite3_blob_bytes(*blobChunksValidity);
    if (validitySize != p->chunk_size / CHAR_BIT) {
      vtab_set_error(&p->base,
                     VEC_INTERAL_ERROR
                     "validity blob size mismatch for newly created chunk "
                     "%s.%s.%lld. Exepcted %lld, got %lld",
                     p->schemaName, p->shadowChunksName, *chunk_rowid,
                     p->chunk_size / CHAR_BIT, validitySize);
      goto cleanup;
    }
    *bufferChunksValidity = sqlite3_malloc(validitySize);
    rc = sqlite3_blob_read(*blobChunksValidity, (void *)*bufferChunksValidity,
                           validitySize, 0);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base,
                     VEC_INTERAL_ERROR
                     "could not read validity blob newly created chunk "
                     "%s.%s.%lld",
                     p->schemaName, p->shadowChunksName, *chunk_rowid);
      goto cleanup;
    }
  }

  rc = SQLITE_OK;

cleanup:
  return rc;
}

/**
 * @brief Write the vector data into the provided vector blob at the given
 * offset
 *
 * @param blobVectors SQLite BLOB to write to
 * @param chunk_offset the "offset" (ie validity bitmap position) to write the
 * vector to
 * @param bVector pointer to the vector containing data
 * @param dimensions how many dimensions the vector has
 * @param element_type the vector type
 * @return result of sqlite3_blob_write, SQLITE_OK on success, otherwise failure
 */
static int
vec0_write_vector_to_vector_blob(sqlite3_blob *blobVectors, i64 chunk_offset,
                                 const void *bVector, size_t dimensions,
                                 enum VectorElementType element_type) {
  int n;
  int offset;

  switch (element_type) {
  case SQLITE_VEC_ELEMENT_TYPE_FLOAT32:
    n = dimensions * sizeof(f32);
    offset = chunk_offset * dimensions * sizeof(f32);
    break;
  case SQLITE_VEC_ELEMENT_TYPE_INT8:
    n = dimensions * sizeof(i8);
    offset = chunk_offset * dimensions * sizeof(i8);
    break;
  case SQLITE_VEC_ELEMENT_TYPE_BIT:
    n = dimensions / CHAR_BIT;
    offset = chunk_offset * dimensions / CHAR_BIT;
    break;
  }

  return sqlite3_blob_write(blobVectors, bVector, n, offset);
}

/**
 * @brief
 *
 * @param p vec0 virtual table
 * @param chunk_rowid: which chunk to write to
 * @param chunk_offset: the offset inside the chunk to write the vector to.
 * @param rowid: the rowid of the inserting row
 * @param vectorDatas: array of the vector data to insert
 * @param blobValidity: writeable validity blob of the row's assigned chunk.
 * @param validity: snapshot buffer of the valdity column from the row's
 * assigned chunk.
 * @return int SQLITE_OK on success, error code on failure
 */
int vec0Update_InsertWriteFinalStep(vec0_vtab *p, i64 chunk_rowid,
                                    i64 chunk_offset, i64 rowid,
                                    void *vectorDatas[],
                                    sqlite3_blob *blobChunksValidity,
                                    const unsigned char *bufferChunksValidity) {
  int rc, brc;
  sqlite3_blob *blobChunksRowids = NULL;

  // mark the validity bit for this row in the chunk's validity bitmap
  // Get the byte offset of the bitmap
  char unsigned bx = bufferChunksValidity[chunk_offset / CHAR_BIT];
  // set the bit at the chunk_offset position inside that byte
  bx = bx | (1 << (chunk_offset % CHAR_BIT));
  // write that 1 byte
  rc = sqlite3_blob_write(blobChunksValidity, &bx, 1, chunk_offset / CHAR_BIT);
  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base, VEC_INTERAL_ERROR "could not mark validity bit ");
    return rc;
  }

  // Go insert the vector data into the vector chunk shadow tables
  for (int i = 0; i < p->numVectorColumns; i++) {
    sqlite3_blob *blobVectors;
    rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowVectorChunksNames[i],
                           "vectors", chunk_rowid, 1, &blobVectors);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base, "Error opening vector blob at %s.%s.%lld",
                     p->schemaName, p->shadowVectorChunksNames[i], chunk_rowid);
      goto cleanup;
    }

    i64 expected =
        p->chunk_size * vector_column_byte_size(p->vector_columns[i]);
    i64 actual = sqlite3_blob_bytes(blobVectors);

    if (actual != expected) {
      // IMP: V16386_00456
      vtab_set_error(
          &p->base,
          VEC_INTERAL_ERROR
          "vector blob size mismatch on %s.%s.%lld. Expected %lld, actual %lld",
          p->schemaName, p->shadowVectorChunksNames[i], chunk_rowid, expected,
          actual);
      rc = SQLITE_ERROR;
      // already error, can ignore result code
      sqlite3_blob_close(blobVectors);
      goto cleanup;
    };

    rc = vec0_write_vector_to_vector_blob(
        blobVectors, chunk_offset, vectorDatas[i],
        p->vector_columns[i].dimensions, p->vector_columns[i].element_type);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base,
                     VEC_INTERAL_ERROR
                     "could not write vector blob on %s.%s.%lld",
                     p->schemaName, p->shadowVectorChunksNames[i], chunk_rowid);
      rc = SQLITE_ERROR;
      // already error, can ignore result code
      sqlite3_blob_close(blobVectors);
      goto cleanup;
    }
    rc = sqlite3_blob_close(blobVectors);
    if (rc != SQLITE_OK) {
      vtab_set_error(&p->base,
                     VEC_INTERAL_ERROR
                     "could not close vector blob on %s.%s.%lld",
                     p->schemaName, p->shadowVectorChunksNames[i], chunk_rowid);
      rc = SQLITE_ERROR;
      goto cleanup;
    }
  }

  // write the new rowid to the rowids column of the _chunks table
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName, "rowids",
                         chunk_rowid, 1, &blobChunksRowids);
  if (rc != SQLITE_OK) {
    // IMP: V09221_26060
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR "could not open rowids blob on %s.%s.%lld",
                   p->schemaName, p->shadowChunksName, chunk_rowid);
    goto cleanup;
  }
  i64 expected = p->chunk_size * sizeof(i64);
  i64 actual = sqlite3_blob_bytes(blobChunksRowids);
  if (expected != actual) {
    // IMP: V12779_29618
    vtab_set_error(
        &p->base,
        VEC_INTERAL_ERROR
        "rowids blob size mismatch on %s.%s.%lld. Expected %lld, actual %lld",
        p->schemaName, p->shadowChunksName, chunk_rowid, expected, actual);
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  rc = sqlite3_blob_write(blobChunksRowids, &rowid, sizeof(i64),
                          chunk_offset * sizeof(i64));
  if (rc != SQLITE_OK) {
    vtab_set_error(
        &p->base, VEC_INTERAL_ERROR "could not write rowids blob on %s.%s.%lld",
        p->schemaName, p->shadowChunksName, chunk_rowid);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  // Now with all the vectors inserted, go back and update the _rowids table
  // with the new chunk_rowid/chunk_offset values
  rc = vec0_rowids_update_position(p, rowid, chunk_rowid, chunk_offset);

cleanup:
  brc = sqlite3_blob_close(blobChunksRowids);
  if ((rc == SQLITE_OK) && (brc != SQLITE_OK)) {
    vtab_set_error(
        &p->base, VEC_INTERAL_ERROR "could not close rowids blob on %s.%s.%lld",
        p->schemaName, p->shadowChunksName, chunk_rowid);
    return brc;
  }
  return rc;
}

int vec0_write_metadata_value(vec0_vtab *p, int metadata_column_idx, i64 rowid, i64 chunk_id, i64 chunk_offset, sqlite3_value * v, int isupdate) {
  int rc;
  struct Vec0MetadataColumnDefinition * metadata_column = &p->metadata_columns[metadata_column_idx];
  vec0_metadata_column_kind kind = metadata_column->kind;

  // verify input value matches column type
  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      if(sqlite3_value_type(v) != SQLITE_INTEGER || ((sqlite3_value_int(v) != 0) && (sqlite3_value_int(v) != 1))) {
        rc = SQLITE_ERROR;
        vtab_set_error(&p->base, "Expected 0 or 1 for BOOLEAN metadata column %.*s", metadata_column->name_length, metadata_column->name);
        goto done;
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      if(sqlite3_value_type(v) != SQLITE_INTEGER) {
        rc = SQLITE_ERROR;
        vtab_set_error(&p->base, "Expected integer for INTEGER metadata column %.*s, received %s", metadata_column->name_length, metadata_column->name, type_name(sqlite3_value_type(v)));
        goto done;
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      if(sqlite3_value_type(v) != SQLITE_FLOAT) {
        rc = SQLITE_ERROR;
        vtab_set_error(&p->base, "Expected float for FLOAT metadata column %.*s, received %s", metadata_column->name_length, metadata_column->name, type_name(sqlite3_value_type(v)));
        goto done;
      }
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      if(sqlite3_value_type(v) != SQLITE_TEXT) {
        rc = SQLITE_ERROR;
        vtab_set_error(&p->base, "Expected text for TEXT metadata column %.*s, received %s", metadata_column->name_length, metadata_column->name, type_name(sqlite3_value_type(v)));
        goto done;
      }
      break;
    }
  }

  sqlite3_blob * blobValue = NULL;
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowMetadataChunksNames[metadata_column_idx], "data", chunk_id, 1, &blobValue);
  if(rc != SQLITE_OK) {
    goto done;
  }

  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      u8 block;
      int value = sqlite3_value_int(v);
      rc = sqlite3_blob_read(blobValue, &block, sizeof(u8), (int) (chunk_offset / CHAR_BIT));
      if(rc != SQLITE_OK) {
        goto done;
      }

      if (value) {
        block |= 1 << (chunk_offset % CHAR_BIT);
      } else {
        block &= ~(1 << (chunk_offset % CHAR_BIT));
      }

      rc = sqlite3_blob_write(blobValue, &block, sizeof(u8), chunk_offset / CHAR_BIT);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      i64 value = sqlite3_value_int64(v);
      rc = sqlite3_blob_write(blobValue, &value, sizeof(value), chunk_offset * sizeof(i64));
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      double value = sqlite3_value_double(v);
      rc = sqlite3_blob_write(blobValue, &value, sizeof(value), chunk_offset * sizeof(double));
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      int prev_n;
      rc = sqlite3_blob_read(blobValue, &prev_n, sizeof(int), chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(rc != SQLITE_OK) {
        goto done;
      }

      const char * s = (const char *) sqlite3_value_text(v);
      int n = sqlite3_value_bytes(v);
      u8 view[VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
      memset(view, 0, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      memcpy(view, &n, sizeof(int));
      memcpy(view+4, s, min(n, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH-4));

      rc = sqlite3_blob_write(blobValue, &view, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH, chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(n > VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
        const char * zSql;

        if(isupdate && (prev_n > VEC0_METADATA_TEXT_VIEW_DATA_LENGTH)) {
          zSql = sqlite3_mprintf("UPDATE " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " SET data = ?2 WHERE rowid = ?1", p->schemaName, p->tableName, metadata_column_idx);
        }else {
          zSql = sqlite3_mprintf("INSERT INTO " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " (rowid, data) VALUES (?1, ?2)", p->schemaName, p->tableName, metadata_column_idx);
        }
        if(!zSql) {
          rc = SQLITE_NOMEM;
          goto done;
        }
        sqlite3_stmt * stmt;
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
        if(rc != SQLITE_OK) {
          goto done;
        }
        sqlite3_bind_int64(stmt, 1, rowid);
        sqlite3_bind_text(stmt, 2, s, n, SQLITE_STATIC);
        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        if(rc != SQLITE_DONE) {
          rc = SQLITE_ERROR;
          goto done;
        }
      }
      else if(prev_n > VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
        const char * zSql = sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " WHERE rowid = ?", p->schemaName, p->tableName, metadata_column_idx);
        if(!zSql) {
          rc = SQLITE_NOMEM;
          goto done;
        }
        sqlite3_stmt * stmt;
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
        if(rc != SQLITE_OK) {
          goto done;
        }
        sqlite3_bind_int64(stmt, 1, rowid);
        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        if(rc != SQLITE_DONE) {
          rc = SQLITE_ERROR;
          goto done;
        }
      }
      break;
    }
  }

  if(rc != SQLITE_OK) {

  }
  rc = sqlite3_blob_close(blobValue);
  if(rc != SQLITE_OK) {
    goto done;
  }

  done:
    return rc;
}


/**
 * @brief Handles INSERT INTO operations on a vec0 table.
 *
 * @return int SQLITE_OK on success, otherwise error code on failure
 */
int vec0Update_Insert(sqlite3_vtab *pVTab, int argc, sqlite3_value **argv,
                      sqlite_int64 *pRowid) {
  UNUSED_PARAMETER(argc);
  vec0_vtab *p = (vec0_vtab *)pVTab;
  int rc;
  // Rowid for the inserted row, deterimined by the inserted ID + _rowids shadow
  // table
  i64 rowid;

  // Array to hold the vector data of the inserted row. Individual elements will
  // have a lifetime bound to the argv[..] values.
  void *vectorDatas[VEC0_MAX_VECTOR_COLUMNS];
  // Array to hold cleanup functions for vectorDatas[]
  vector_cleanup cleanups[VEC0_MAX_VECTOR_COLUMNS];

  sqlite3_value * partitionKeyValues[VEC0_MAX_PARTITION_COLUMNS];

  // Rowid of the chunk in the _chunks shadow table that the row will be a part
  // of.
  i64 chunk_rowid;
  // offset within the chunk where the rowid belongs
  i64 chunk_offset;

  // a write-able blob of the validity column for the given chunk. Used to mark
  // validity bit
  sqlite3_blob *blobChunksValidity = NULL;
  // buffer for the valididty column for the given chunk. Maybe not needed here?
  const unsigned char *bufferChunksValidity = NULL;
  int numReadVectors = 0;

  // Read all provided partition key values into partitionKeyValues
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_PARTITION) {
      continue;
    }
    int partition_key_idx = p->user_column_idxs[i];
    partitionKeyValues[partition_key_idx] = argv[2+VEC0_COLUMN_USERN_START + i];

    int new_value_type = sqlite3_value_type(partitionKeyValues[partition_key_idx]);
    if((new_value_type != SQLITE_NULL) && (new_value_type != p->paritition_columns[partition_key_idx].type)) {
      // IMP: V11454_28292
      vtab_set_error(
        pVTab,
        "Parition key type mismatch: The partition key column %.*s has type %s, but %s was provided.",
        p->paritition_columns[partition_key_idx].name_length,
        p->paritition_columns[partition_key_idx].name,
        type_name(p->paritition_columns[partition_key_idx].type),
        type_name(new_value_type)
      );
      rc = SQLITE_ERROR;
      goto cleanup;
    }
  }

  // read all the inserted vectors  into vectorDatas, validate their lengths.
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_VECTOR) {
      continue;
    }
    int vector_column_idx = p->user_column_idxs[i];
    sqlite3_value *valueVector = argv[2 + VEC0_COLUMN_USERN_START + i];
    size_t dimensions;

    char *pzError;
    enum VectorElementType elementType;
    rc = vector_from_value(valueVector, &vectorDatas[vector_column_idx], &dimensions,
                           &elementType, &cleanups[vector_column_idx], &pzError);
    if (rc != SQLITE_OK) {
      // IMP: V06519_23358
      vtab_set_error(
          pVTab, "Inserted vector for the \"%.*s\" column is invalid: %z",
          p->vector_columns[vector_column_idx].name_length, p->vector_columns[vector_column_idx].name, pzError);
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    numReadVectors++;
    if (elementType != p->vector_columns[vector_column_idx].element_type) {
      // IMP: V08221_25059
      vtab_set_error(
          pVTab,
          "Inserted vector for the \"%.*s\" column is expected to be of type "
          "%s, but a %s vector was provided.",
          p->vector_columns[i].name_length, p->vector_columns[i].name,
          vector_subtype_name(p->vector_columns[i].element_type),
          vector_subtype_name(elementType));
      rc = SQLITE_ERROR;
      goto cleanup;
    }

    if (dimensions != p->vector_columns[vector_column_idx].dimensions) {
      // IMP: V01145_17984
      vtab_set_error(
          pVTab,
          "Dimension mismatch for inserted vector for the \"%.*s\" column. "
          "Expected %d dimensions but received %d.",
          p->vector_columns[vector_column_idx].name_length, p->vector_columns[vector_column_idx].name,
          p->vector_columns[vector_column_idx].dimensions, dimensions);
      rc = SQLITE_ERROR;
      goto cleanup;
    }
  }

  // Cannot insert a value in the hidden "distance" column
  if (sqlite3_value_type(argv[2 + vec0_column_distance_idx(p)]) !=
      SQLITE_NULL) {
    // IMP: V24228_08298
    vtab_set_error(pVTab,
                   "A value was provided for the hidden \"distance\" column.");
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  // Cannot insert a value in the hidden "k" column
  if (sqlite3_value_type(argv[2 + vec0_column_k_idx(p)]) != SQLITE_NULL) {
    // IMP: V11875_28713
    vtab_set_error(pVTab, "A value was provided for the hidden \"k\" column.");
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  // Step #1: Insert/get a rowid for this row, from the _rowids table.
  rc = vec0Update_InsertRowidStep(p, argv[2 + VEC0_COLUMN_ID], &rowid);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }

  // Step #2: Find the next "available" position in the _chunks table for this
  // row.
  rc = vec0Update_InsertNextAvailableStep(p, partitionKeyValues,
  &chunk_rowid, &chunk_offset,
                                          &blobChunksValidity,
                                          &bufferChunksValidity);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }

  // Step #3: With the next available chunk position, write out all the vectors
  //          to their specified location.
  rc = vec0Update_InsertWriteFinalStep(p, chunk_rowid, chunk_offset, rowid,
                                       vectorDatas, blobChunksValidity,
                                       bufferChunksValidity);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }

  if(p->numAuxiliaryColumns > 0) {
    sqlite3_stmt *stmt;
    sqlite3_str * s = sqlite3_str_new(NULL);
    sqlite3_str_appendf(s, "INSERT INTO " VEC0_SHADOW_AUXILIARY_NAME "(rowid ", p->schemaName, p->tableName);
    for(int i = 0; i < p->numAuxiliaryColumns; i++) {
      sqlite3_str_appendf(s, ", value%02d", i);
    }
    sqlite3_str_appendall(s, ") VALUES (? ");
    for(int i = 0; i < p->numAuxiliaryColumns; i++) {
      sqlite3_str_appendall(s, ", ?");
    }
    sqlite3_str_appendall(s, ")");
    char * zSql = sqlite3_str_finish(s);
    // TODO double check error handling ehre
    if(!zSql) {
      rc = SQLITE_NOMEM;
      goto cleanup;
    }
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
    if(rc != SQLITE_OK) {
      goto cleanup;
    }
    sqlite3_bind_int64(stmt, 1, rowid);

    for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
      if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY) {
        continue;
      }
      int auxiliary_key_idx = p->user_column_idxs[i];
      sqlite3_value * v = argv[2+VEC0_COLUMN_USERN_START + i];
      int v_type = sqlite3_value_type(v);
      if(v_type != SQLITE_NULL && (v_type != p->auxiliary_columns[auxiliary_key_idx].type)) {
        sqlite3_finalize(stmt);
        rc = SQLITE_CONSTRAINT;
        vtab_set_error(
          pVTab,
          "Auxiliary column type mismatch: The auxiliary column %.*s has type %s, but %s was provided.",
          p->auxiliary_columns[auxiliary_key_idx].name_length,
          p->auxiliary_columns[auxiliary_key_idx].name,
          type_name(p->auxiliary_columns[auxiliary_key_idx].type),
          type_name(v_type)
        );
        goto cleanup;
      }
      // first 1 is for 1-based indexing on sqlite3_bind_*, second 1 is to account for initial rowid parameter
      sqlite3_bind_value(stmt, 1 + 1 + auxiliary_key_idx, v);
    }

    rc = sqlite3_step(stmt);
    if(rc != SQLITE_DONE) {
      sqlite3_finalize(stmt);
      rc = SQLITE_ERROR;
      goto cleanup;
    }
    sqlite3_finalize(stmt);
  }


  for(int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_METADATA) {
      continue;
    }
    int metadata_idx = p->user_column_idxs[i];
    sqlite3_value *v = argv[2 + VEC0_COLUMN_USERN_START + i];
    rc = vec0_write_metadata_value(p, metadata_idx, rowid, chunk_rowid, chunk_offset, v, 0);
    if(rc != SQLITE_OK) {
      goto cleanup;
    }
  }

  *pRowid = rowid;
  rc = SQLITE_OK;

cleanup:
  for (int i = 0; i < numReadVectors; i++) {
    cleanups[i](vectorDatas[i]);
  }
  sqlite3_free((void *)bufferChunksValidity);
  int brc = sqlite3_blob_close(blobChunksValidity);
  if ((rc == SQLITE_OK) && (brc != SQLITE_OK)) {
    vtab_set_error(&p->base,
                   VEC_INTERAL_ERROR "unknown error, blobChunksValidity could "
                                     "not be closed, please file an issue");
    return brc;
  }
  return rc;
}

int vec0Update_Delete_ClearValidity(vec0_vtab *p, i64 chunk_id,
                                    u64 chunk_offset) {
  int rc, brc;
  sqlite3_blob *blobChunksValidity = NULL;
  char unsigned bx;
  int validityOffset = chunk_offset / CHAR_BIT;

  // 2. ensure chunks.validity bit is 1, then set to 0
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowChunksName, "validity",
                         chunk_id, 1, &blobChunksValidity);
  if (rc != SQLITE_OK) {
    // IMP: V26002_10073
    vtab_set_error(&p->base, "could not open validity blob for %s.%s.%lld",
                   p->schemaName, p->shadowChunksName, chunk_id);
    return SQLITE_ERROR;
  }
  // will skip the sqlite3_blob_bytes(blobChunksValidity) check for now,
  // the read below would catch it

  rc = sqlite3_blob_read(blobChunksValidity, &bx, sizeof(bx), validityOffset);
  if (rc != SQLITE_OK) {
    // IMP: V21193_05263
    vtab_set_error(
        &p->base, "could not read validity blob for %s.%s.%lld at %d",
        p->schemaName, p->shadowChunksName, chunk_id, validityOffset);
    goto cleanup;
  }
  if (!(bx >> (chunk_offset % CHAR_BIT))) {
    // IMP: V21193_05263
    rc = SQLITE_ERROR;
    vtab_set_error(
        &p->base,
        "vec0 deletion error: validity bit is not set for %s.%s.%lld at %d",
        p->schemaName, p->shadowChunksName, chunk_id, validityOffset);
    goto cleanup;
  }
  char unsigned mask = ~(1 << (chunk_offset % CHAR_BIT));
  char result = bx & mask;
  rc = sqlite3_blob_write(blobChunksValidity, &result, sizeof(bx),
                          validityOffset);
  if (rc != SQLITE_OK) {
    vtab_set_error(
        &p->base, "could not write to validity blob for %s.%s.%lld at %d",
        p->schemaName, p->shadowChunksName, chunk_id, validityOffset);
    goto cleanup;
  }

cleanup:

  brc = sqlite3_blob_close(blobChunksValidity);
  if (rc != SQLITE_OK)
    return rc;
  if (brc != SQLITE_OK) {
    vtab_set_error(&p->base,
                   "vec0 deletion error: Error commiting validity blob "
                   "transaction on %s.%s.%lld at %d",
                   p->schemaName, p->shadowChunksName, chunk_id,
                   validityOffset);
    return brc;
  }
  return SQLITE_OK;
}

int vec0Update_Delete_DeleteRowids(vec0_vtab *p, i64 rowid) {
  int rc;
  sqlite3_stmt *stmt = NULL;

  char *zSql =
      sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_ROWIDS_NAME " WHERE rowid = ?",
                      p->schemaName, p->tableName);
  if (!zSql) {
    return SQLITE_NOMEM;
  }

  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_DONE) {
    goto cleanup;
  }
  rc = SQLITE_OK;

cleanup:
  sqlite3_finalize(stmt);
  return rc;
}

int vec0Update_Delete_DeleteAux(vec0_vtab *p, i64 rowid) {
  int rc;
  sqlite3_stmt *stmt = NULL;

  char *zSql =
      sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_AUXILIARY_NAME " WHERE rowid = ?",
                      p->schemaName, p->tableName);
  if (!zSql) {
    return SQLITE_NOMEM;
  }

  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    goto cleanup;
  }
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_DONE) {
    goto cleanup;
  }
  rc = SQLITE_OK;

cleanup:
  sqlite3_finalize(stmt);
  return rc;
}

int vec0Update_Delete_ClearMetadata(vec0_vtab *p, int metadata_idx, i64 rowid, i64 chunk_id,
                                    u64 chunk_offset) {
  int rc;
  sqlite3_blob * blobValue;
  vec0_metadata_column_kind kind = p->metadata_columns[metadata_idx].kind;
  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowMetadataChunksNames[metadata_idx], "data", chunk_id, 1, &blobValue);
  if(rc != SQLITE_OK) {
    return rc;
  }

  switch(kind) {
    case VEC0_METADATA_COLUMN_KIND_BOOLEAN: {
      u8 block;
      rc = sqlite3_blob_read(blobValue, &block, sizeof(u8), (int) (chunk_offset / CHAR_BIT));
      if(rc != SQLITE_OK) {
        goto done;
      }

      block &= ~(1 << (chunk_offset % CHAR_BIT));
      rc = sqlite3_blob_write(blobValue, &block, sizeof(u8), chunk_offset / CHAR_BIT);
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_INTEGER: {
      i64 v = 0;
      rc = sqlite3_blob_write(blobValue, &v, sizeof(v), chunk_offset * sizeof(i64));
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_FLOAT: {
      double v = 0;
      rc = sqlite3_blob_write(blobValue, &v, sizeof(v), chunk_offset * sizeof(double));
      break;
    }
    case VEC0_METADATA_COLUMN_KIND_TEXT: {
      int n;
      rc = sqlite3_blob_read(blobValue, &n, sizeof(int), chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(rc != SQLITE_OK) {
        goto done;
      }

      u8 view[VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH];
      memset(view, 0, VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      rc = sqlite3_blob_write(blobValue, &view, sizeof(view), chunk_offset * VEC0_METADATA_TEXT_VIEW_BUFFER_LENGTH);
      if(rc != SQLITE_OK) {
        goto done;
      }

      if(n > VEC0_METADATA_TEXT_VIEW_DATA_LENGTH) {
        const char * zSql = sqlite3_mprintf("DELETE FROM " VEC0_SHADOW_METADATA_TEXT_DATA_NAME " WHERE rowid = ?", p->schemaName, p->tableName, metadata_idx);
        if(!zSql) {
          rc = SQLITE_NOMEM;
          goto done;
        }
        sqlite3_stmt * stmt;
        rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
        if(rc != SQLITE_OK) {
          goto done;
        }
        sqlite3_bind_int64(stmt, 1, rowid);
        rc = sqlite3_step(stmt);
        if(rc != SQLITE_DONE) {
          rc = SQLITE_ERROR;
          goto done;
        }
        sqlite3_finalize(stmt);
      }
      break;
    }
  }
  int rc2;
  done:
  rc2 = sqlite3_blob_close(blobValue);
  if(rc == SQLITE_OK) {
    return rc2;
  }
  return rc;
}

int vec0Update_Delete(sqlite3_vtab *pVTab, sqlite3_value *idValue) {
  vec0_vtab *p = (vec0_vtab *)pVTab;
  int rc;
  i64 rowid;
  i64 chunk_id;
  i64 chunk_offset;

  if (p->pkIsText) {
    rc = vec0_rowid_from_id(p, idValue, &rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
  } else {
    rowid = sqlite3_value_int64(idValue);
  }

  // 1. Find chunk position for given rowid
  // 2. Ensure that validity bit for position is 1, then set to 0
  // 3. Zero out rowid in chunks.rowid
  // 4. Zero out vector data in all vector column chunks
  // 5. Delete value in _rowids table

  // 1. get chunk_id and chunk_offset from _rowids
  rc = vec0_get_chunk_position(p, rowid, NULL, &chunk_id, &chunk_offset);
  if (rc != SQLITE_OK) {
    return rc;
  }

  rc = vec0Update_Delete_ClearValidity(p, chunk_id, chunk_offset);
  if (rc != SQLITE_OK) {
    return rc;
  }

  // 3. zero out rowid in chunks.rowids
  // https://github.com/asg017/sqlite-vec/issues/54

  // 4. zero out any data in vector chunks tables
  // https://github.com/asg017/sqlite-vec/issues/54

  // 5. delete from _rowids table
  rc = vec0Update_Delete_DeleteRowids(p, rowid);
  if (rc != SQLITE_OK) {
    return rc;
  }

  // 6. delete any auxiliary rows
  if(p->numAuxiliaryColumns > 0) {
    rc = vec0Update_Delete_DeleteAux(p, rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
  }

  // 6. delete metadata
  for(int i = 0; i < p->numMetadataColumns; i++) {
    rc = vec0Update_Delete_ClearMetadata(p, i, rowid, chunk_id, chunk_offset);
  }

  return SQLITE_OK;
}

int vec0Update_UpdateAuxColumn(vec0_vtab *p, int auxiliary_column_idx, sqlite3_value * value, i64 rowid) {
  int rc;
  sqlite3_stmt *stmt;
  const char * zSql = sqlite3_mprintf("UPDATE " VEC0_SHADOW_AUXILIARY_NAME " SET value%02d = ? WHERE rowid = ?", p->schemaName, p->tableName, auxiliary_column_idx);
  if(!zSql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  if(rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_bind_value(stmt, 1, value);
  sqlite3_bind_int64(stmt, 2, rowid);
  rc = sqlite3_step(stmt);
  if(rc != SQLITE_DONE) {
    sqlite3_finalize(stmt);
    return SQLITE_ERROR;
  }
  sqlite3_finalize(stmt);
  return SQLITE_OK;
}

int vec0Update_UpdateVectorColumn(vec0_vtab *p, i64 chunk_id, i64 chunk_offset,
                                  int i, sqlite3_value *valueVector) {
  int rc;

  sqlite3_blob *blobVectors = NULL;

  char *pzError;
  size_t dimensions;
  enum VectorElementType elementType;
  void *vector;
  vector_cleanup cleanup = vector_cleanup_noop;
  // https://github.com/asg017/sqlite-vec/issues/53
  rc = vector_from_value(valueVector, &vector, &dimensions, &elementType,
                         &cleanup, &pzError);
  if (rc != SQLITE_OK) {
    // IMP: V15203_32042
    vtab_set_error(
        &p->base, "Updated vector for the \"%.*s\" column is invalid: %z",
        p->vector_columns[i].name_length, p->vector_columns[i].name, pzError);
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if (elementType != p->vector_columns[i].element_type) {
    // IMP: V03643_20481
    vtab_set_error(
        &p->base,
        "Updated vector for the \"%.*s\" column is expected to be of type "
        "%s, but a %s vector was provided.",
        p->vector_columns[i].name_length, p->vector_columns[i].name,
        vector_subtype_name(p->vector_columns[i].element_type),
        vector_subtype_name(elementType));
    rc = SQLITE_ERROR;
    goto cleanup;
  }
  if (dimensions != p->vector_columns[i].dimensions) {
    // IMP: V25739_09810
    vtab_set_error(
        &p->base,
        "Dimension mismatch for new updated vector for the \"%.*s\" column. "
        "Expected %d dimensions but received %d.",
        p->vector_columns[i].name_length, p->vector_columns[i].name,
        p->vector_columns[i].dimensions, dimensions);
    rc = SQLITE_ERROR;
    goto cleanup;
  }

  rc = sqlite3_blob_open(p->db, p->schemaName, p->shadowVectorChunksNames[i],
                         "vectors", chunk_id, 1, &blobVectors);
  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base, "Could not open vectors blob for %s.%s.%lld",
                   p->schemaName, p->shadowVectorChunksNames[i], chunk_id);
    goto cleanup;
  }
  rc = vec0_write_vector_to_vector_blob(blobVectors, chunk_offset, vector,
                                        p->vector_columns[i].dimensions,
                                        p->vector_columns[i].element_type);
  if (rc != SQLITE_OK) {
    vtab_set_error(&p->base, "Could not write to vectors blob for %s.%s.%lld",
                   p->schemaName, p->shadowVectorChunksNames[i], chunk_id);
    goto cleanup;
  }

cleanup:
  cleanup(vector);
  int brc = sqlite3_blob_close(blobVectors);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (brc != SQLITE_OK) {
    vtab_set_error(
        &p->base,
        "Could not commit blob transaction for vectors blob for %s.%s.%lld",
        p->schemaName, p->shadowVectorChunksNames[i], chunk_id);
    return brc;
  }
  return SQLITE_OK;
}

int vec0Update_Update(sqlite3_vtab *pVTab, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  vec0_vtab *p = (vec0_vtab *)pVTab;
  int rc;
  i64 chunk_id;
  i64 chunk_offset;

  i64 rowid;
  if (p->pkIsText) {
    const char *a = (const char *)sqlite3_value_text(argv[0]);
    const char *b = (const char *)sqlite3_value_text(argv[1]);
    // IMP: V08886_25725
    if ((sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) ||
        strncmp(a, b, sqlite3_value_bytes(argv[0])) != 0) {
      vtab_set_error(pVTab,
                     "UPDATEs on vec0 primary key values are not allowed.");
      return SQLITE_ERROR;
    }
    rc = vec0_rowid_from_id(p, argv[0], &rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
  } else {
    rowid = sqlite3_value_int64(argv[0]);
  }

  // 1) get chunk_id and chunk_offset from _rowids
  rc = vec0_get_chunk_position(p, rowid, NULL, &chunk_id, &chunk_offset);
  if (rc != SQLITE_OK) {
    return rc;
  }

  // 2) update any partition key values
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_PARTITION) {
      continue;
    }
    sqlite3_value * value = argv[2+VEC0_COLUMN_USERN_START + i];
    if(sqlite3_value_nochange(value)) {
      continue;
    }
    vtab_set_error(pVTab, "UPDATE on partition key columns are not supported yet. ");
    return SQLITE_ERROR;
  }

  // 3) handle auxiliary column updates
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_AUXILIARY) {
      continue;
    }
    int auxiliary_column_idx = p->user_column_idxs[i];
    sqlite3_value * value = argv[2+VEC0_COLUMN_USERN_START + i];
    if(sqlite3_value_nochange(value)) {
      continue;
    }
    rc = vec0Update_UpdateAuxColumn(p, auxiliary_column_idx, value, rowid);
    if(rc != SQLITE_OK) {
      return SQLITE_ERROR;
    }
  }

  // 4) handle metadata column updates
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_METADATA) {
      continue;
    }
    int metadata_column_idx = p->user_column_idxs[i];
    sqlite3_value * value = argv[2+VEC0_COLUMN_USERN_START + i];
    if(sqlite3_value_nochange(value)) {
      continue;
    }
    rc = vec0_write_metadata_value(p, metadata_column_idx, rowid, chunk_id, chunk_offset, value, 1);
    if(rc != SQLITE_OK) {
      return rc;
    }
  }

  // 5) iterate over all new vectors, update the vectors
  for (int i = 0; i < vec0_num_defined_user_columns(p); i++) {
    if(p->user_column_kinds[i] != SQLITE_VEC0_USER_COLUMN_KIND_VECTOR) {
      continue;
    }
    int vector_idx = p->user_column_idxs[i];
    sqlite3_value *valueVector = argv[2 + VEC0_COLUMN_USERN_START + i];
    // in vec0Column, we check sqlite3_vtab_nochange() on vector columns.
    // If the vector column isn't being changed, we return NULL;
    // That's not great, that means vector columns can never be NULLABLE
    // (bc we cant distinguish if an updated vector is truly NULL or nochange).
    // Also it means that if someone tries to run `UPDATE v SET X = NULL`,
    // we can't effectively detect and raise an error.
    // A better solution would be to use a custom result_type for "empty",
    // but subtypes don't appear to survive xColumn -> xUpdate, it's always 0.
    // So for now, we'll just use NULL and warn people to not SET X = NULL
    // in the docs.
    if (sqlite3_value_type(valueVector) == SQLITE_NULL) {
      continue;
    }

    rc = vec0Update_UpdateVectorColumn(p, chunk_id, chunk_offset, vector_idx,
                                       valueVector);
    if (rc != SQLITE_OK) {
      return SQLITE_ERROR;
    }
  }

  return SQLITE_OK;
}

static int vec0Update(sqlite3_vtab *pVTab, int argc, sqlite3_value **argv,
                      sqlite_int64 *pRowid) {
  // DELETE operation
  if (argc == 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    return vec0Update_Delete(pVTab, argv[0]);
  }
  // INSERT operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    return vec0Update_Insert(pVTab, argc, argv, pRowid);
  }
  // UPDATE operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    return vec0Update_Update(pVTab, argc, argv);
  } else {
    vtab_set_error(pVTab, "Unrecognized xUpdate operation provided for vec0.");
    return SQLITE_ERROR;
  }
}

static int vec0ShadowName(const char *zName) {
  static const char *azName[] = {
    "rowids", "chunks", "auxiliary", "info",

  // Up to VEC0_MAX_METADATA_COLUMNS
  // TODO be smarter about this man
  "metadatachunks00",
  "metadatachunks01",
  "metadatachunks02",
  "metadatachunks03",
  "metadatachunks04",
  "metadatachunks05",
  "metadatachunks06",
  "metadatachunks07",
  "metadatachunks08",
  "metadatachunks09",
  "metadatachunks10",
  "metadatachunks11",
  "metadatachunks12",
  "metadatachunks13",
  "metadatachunks14",
  "metadatachunks15",

  // Up to
  "metadatatext00",
  "metadatatext01",
  "metadatatext02",
  "metadatatext03",
  "metadatatext04",
  "metadatatext05",
  "metadatatext06",
  "metadatatext07",
  "metadatatext08",
  "metadatatext09",
  "metadatatext10",
  "metadatatext11",
  "metadatatext12",
  "metadatatext13",
  "metadatatext14",
  "metadatatext15",
  };

  for (size_t i = 0; i < sizeof(azName) / sizeof(azName[0]); i++) {
    if (sqlite3_stricmp(zName, azName[i]) == 0)
      return 1;
  }
  //for(size_t i = 0; i < )"vector_chunks", "metadatachunks"
  return 0;
}

static int vec0Begin(sqlite3_vtab *pVTab) {
  UNUSED_PARAMETER(pVTab);
  return SQLITE_OK;
}
static int vec0Sync(sqlite3_vtab *pVTab) {
  UNUSED_PARAMETER(pVTab);
  vec0_vtab *p = (vec0_vtab *)pVTab;
  if (p->stmtLatestChunk) {
    sqlite3_finalize(p->stmtLatestChunk);
    p->stmtLatestChunk = NULL;
  }
  if (p->stmtRowidsInsertRowid) {
    sqlite3_finalize(p->stmtRowidsInsertRowid);
    p->stmtRowidsInsertRowid = NULL;
  }
  if (p->stmtRowidsInsertId) {
    sqlite3_finalize(p->stmtRowidsInsertId);
    p->stmtRowidsInsertId = NULL;
  }
  if (p->stmtRowidsUpdatePosition) {
    sqlite3_finalize(p->stmtRowidsUpdatePosition);
    p->stmtRowidsUpdatePosition = NULL;
  }
  if (p->stmtRowidsGetChunkPosition) {
    sqlite3_finalize(p->stmtRowidsGetChunkPosition);
    p->stmtRowidsGetChunkPosition = NULL;
  }
  return SQLITE_OK;
}
static int vec0Commit(sqlite3_vtab *pVTab) {
  UNUSED_PARAMETER(pVTab);
  return SQLITE_OK;
}
static int vec0Rollback(sqlite3_vtab *pVTab) {
  UNUSED_PARAMETER(pVTab);
  return SQLITE_OK;
}

static sqlite3_module vec0Module = {
    /* iVersion      */ 3,
    /* xCreate       */ vec0Create,
    /* xConnect      */ vec0Connect,
    /* xBestIndex    */ vec0BestIndex,
    /* xDisconnect   */ vec0Disconnect,
    /* xDestroy      */ vec0Destroy,
    /* xOpen         */ vec0Open,
    /* xClose        */ vec0Close,
    /* xFilter       */ vec0Filter,
    /* xNext         */ vec0Next,
    /* xEof          */ vec0Eof,
    /* xColumn       */ vec0Column,
    /* xRowid        */ vec0Rowid,
    /* xUpdate       */ vec0Update,
    /* xBegin        */ vec0Begin,
    /* xSync         */ vec0Sync,
    /* xCommit       */ vec0Commit,
    /* xRollback     */ vec0Rollback,
    /* xFindFunction */ 0,
    /* xRename       */ 0, // https://github.com/asg017/sqlite-vec/issues/43
    /* xSavepoint    */ 0,
    /* xRelease      */ 0,
    /* xRollbackTo   */ 0,
    /* xShadowName   */ vec0ShadowName,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity    */ 0, // https://github.com/asg017/sqlite-vec/issues/44
#endif
};
#pragma endregion

static char *POINTER_NAME_STATIC_BLOB_DEF = "vec0-static_blob_def";
struct static_blob_definition {
  void *p;
  size_t dimensions;
  size_t nvectors;
  enum VectorElementType element_type;
};
static void vec_static_blob_from_raw(sqlite3_context *context, int argc,
                                     sqlite3_value **argv) {

  assert(argc == 4);
  struct static_blob_definition *p;
  p = sqlite3_malloc(sizeof(*p));
  if (!p) {
    sqlite3_result_error_nomem(context);
    return;
  }
  memset(p, 0, sizeof(*p));
  p->p = (void *)sqlite3_value_int64(argv[0]);
  p->element_type = SQLITE_VEC_ELEMENT_TYPE_FLOAT32;
  p->dimensions = sqlite3_value_int64(argv[2]);
  p->nvectors = sqlite3_value_int64(argv[3]);
  sqlite3_result_pointer(context, p, POINTER_NAME_STATIC_BLOB_DEF,
                         sqlite3_free);
}
#pragma region vec_static_blobs() table function

#define MAX_STATIC_BLOBS 16

typedef struct static_blob static_blob;
struct static_blob {
  char *name;
  void *p;
  size_t dimensions;
  size_t nvectors;
  enum VectorElementType element_type;
};

typedef struct vec_static_blob_data vec_static_blob_data;
struct vec_static_blob_data {
  static_blob static_blobs[MAX_STATIC_BLOBS];
};

typedef struct vec_static_blobs_vtab vec_static_blobs_vtab;
struct vec_static_blobs_vtab {
  sqlite3_vtab base;
  vec_static_blob_data *data;
};

typedef struct vec_static_blobs_cursor vec_static_blobs_cursor;
struct vec_static_blobs_cursor {
  sqlite3_vtab_cursor base;
  sqlite3_int64 iRowid;
};

static int vec_static_blobsConnect(sqlite3 *db, void *pAux, int argc,
                                   const char *const *argv,
                                   sqlite3_vtab **ppVtab, char **pzErr) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);

  vec_static_blobs_vtab *pNew;
#define VEC_STATIC_BLOBS_NAME 0
#define VEC_STATIC_BLOBS_DATA 1
#define VEC_STATIC_BLOBS_DIMENSIONS 2
#define VEC_STATIC_BLOBS_COUNT 3
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(name, data, dimensions hidden, count hidden)");
  if (rc == SQLITE_OK) {
    pNew = sqlite3_malloc(sizeof(*pNew));
    *ppVtab = (sqlite3_vtab *)pNew;
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
    pNew->data = pAux;
  }
  return rc;
}

static int vec_static_blobsDisconnect(sqlite3_vtab *pVtab) {
  vec_static_blobs_vtab *p = (vec_static_blobs_vtab *)pVtab;
  sqlite3_free(p);
  return SQLITE_OK;
}

static int vec_static_blobsUpdate(sqlite3_vtab *pVTab, int argc,
                                  sqlite3_value **argv, sqlite_int64 *pRowid) {
  UNUSED_PARAMETER(pRowid);
  vec_static_blobs_vtab *p = (vec_static_blobs_vtab *)pVTab;
  // DELETE operation
  if (argc == 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    return SQLITE_ERROR;
  }
  // INSERT operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    const char *key =
        (const char *)sqlite3_value_text(argv[2 + VEC_STATIC_BLOBS_NAME]);
    int idx = -1;
    for (int i = 0; i < MAX_STATIC_BLOBS; i++) {
      if (!p->data->static_blobs[i].name) {
        p->data->static_blobs[i].name = sqlite3_mprintf("%s", key);
        idx = i;
        break;
      }
    }
    if (idx < 0)
      abort();
    struct static_blob_definition *def = sqlite3_value_pointer(
        argv[2 + VEC_STATIC_BLOBS_DATA], POINTER_NAME_STATIC_BLOB_DEF);
    p->data->static_blobs[idx].p = def->p;
    p->data->static_blobs[idx].dimensions = def->dimensions;
    p->data->static_blobs[idx].nvectors = def->nvectors;
    p->data->static_blobs[idx].element_type = def->element_type;

    return SQLITE_OK;
  }
  // UPDATE operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    return SQLITE_ERROR;
  }
  return SQLITE_ERROR;
}

static int vec_static_blobsOpen(sqlite3_vtab *p,
                                sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec_static_blobs_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec_static_blobsClose(sqlite3_vtab_cursor *cur) {
  vec_static_blobs_cursor *pCur = (vec_static_blobs_cursor *)cur;
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int vec_static_blobsBestIndex(sqlite3_vtab *pVTab,
                                     sqlite3_index_info *pIdxInfo) {
  UNUSED_PARAMETER(pVTab);
  pIdxInfo->idxNum = 1;
  pIdxInfo->estimatedCost = (double)10;
  pIdxInfo->estimatedRows = 10;
  return SQLITE_OK;
}

static int vec_static_blobsNext(sqlite3_vtab_cursor *cur);
static int vec_static_blobsFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                                  const char *idxStr, int argc,
                                  sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  vec_static_blobs_cursor *pCur = (vec_static_blobs_cursor *)pVtabCursor;
  pCur->iRowid = -1;
  vec_static_blobsNext(pVtabCursor);
  return SQLITE_OK;
}

static int vec_static_blobsRowid(sqlite3_vtab_cursor *cur,
                                 sqlite_int64 *pRowid) {
  vec_static_blobs_cursor *pCur = (vec_static_blobs_cursor *)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}

static int vec_static_blobsNext(sqlite3_vtab_cursor *cur) {
  vec_static_blobs_cursor *pCur = (vec_static_blobs_cursor *)cur;
  vec_static_blobs_vtab *p = (vec_static_blobs_vtab *)pCur->base.pVtab;
  pCur->iRowid++;
  while (pCur->iRowid < MAX_STATIC_BLOBS) {
    if (p->data->static_blobs[pCur->iRowid].name) {
      return SQLITE_OK;
    }
    pCur->iRowid++;
  }
  return SQLITE_OK;
}

static int vec_static_blobsEof(sqlite3_vtab_cursor *cur) {
  vec_static_blobs_cursor *pCur = (vec_static_blobs_cursor *)cur;
  return pCur->iRowid >= MAX_STATIC_BLOBS;
}

static int vec_static_blobsColumn(sqlite3_vtab_cursor *cur,
                                  sqlite3_context *context, int i) {
  vec_static_blobs_cursor *pCur = (vec_static_blobs_cursor *)cur;
  vec_static_blobs_vtab *p = (vec_static_blobs_vtab *)cur->pVtab;
  switch (i) {
  case VEC_STATIC_BLOBS_NAME:
    sqlite3_result_text(context, p->data->static_blobs[pCur->iRowid].name, -1,
                        SQLITE_TRANSIENT);
    break;
  case VEC_STATIC_BLOBS_DATA:
    sqlite3_result_null(context);
    break;
  case VEC_STATIC_BLOBS_DIMENSIONS:
    sqlite3_result_int64(context,
                         p->data->static_blobs[pCur->iRowid].dimensions);
    break;
  case VEC_STATIC_BLOBS_COUNT:
    sqlite3_result_int64(context, p->data->static_blobs[pCur->iRowid].nvectors);
    break;
  }
  return SQLITE_OK;
}

static sqlite3_module vec_static_blobsModule = {
    /* iVersion    */ 3,
    /* xCreate     */ 0,
    /* xConnect    */ vec_static_blobsConnect,
    /* xBestIndex  */ vec_static_blobsBestIndex,
    /* xDisconnect */ vec_static_blobsDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ vec_static_blobsOpen,
    /* xClose      */ vec_static_blobsClose,
    /* xFilter     */ vec_static_blobsFilter,
    /* xNext       */ vec_static_blobsNext,
    /* xEof        */ vec_static_blobsEof,
    /* xColumn     */ vec_static_blobsColumn,
    /* xRowid      */ vec_static_blobsRowid,
    /* xUpdate     */ vec_static_blobsUpdate,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity  */ 0
#endif
};
#pragma endregion

#pragma region vec_static_blob_entries() table function

typedef struct vec_static_blob_entries_vtab vec_static_blob_entries_vtab;
struct vec_static_blob_entries_vtab {
  sqlite3_vtab base;
  static_blob *blob;
};
typedef enum {
  VEC_SBE__QUERYPLAN_FULLSCAN = 1,
  VEC_SBE__QUERYPLAN_KNN = 2
} vec_sbe_query_plan;

struct sbe_query_knn_data {
  i64 k;
  i64 k_used;
  // Array of rowids of size k. Must be freed with sqlite3_free().
  i32 *rowids;
  // Array of distances of size k. Must be freed with sqlite3_free().
  f32 *distances;
  i64 current_idx;
};
void sbe_query_knn_data_clear(struct sbe_query_knn_data *knn_data) {
  if (!knn_data)
    return;

  if (knn_data->rowids) {
    sqlite3_free(knn_data->rowids);
    knn_data->rowids = NULL;
  }
  if (knn_data->distances) {
    sqlite3_free(knn_data->distances);
    knn_data->distances = NULL;
  }
}

typedef struct vec_static_blob_entries_cursor vec_static_blob_entries_cursor;
struct vec_static_blob_entries_cursor {
  sqlite3_vtab_cursor base;
  sqlite3_int64 iRowid;
  vec_sbe_query_plan query_plan;
  struct sbe_query_knn_data *knn_data;
};

static int vec_static_blob_entriesConnect(sqlite3 *db, void *pAux, int argc,
                                          const char *const *argv,
                                          sqlite3_vtab **ppVtab, char **pzErr) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  vec_static_blob_data *blob_data = pAux;
  int idx = -1;
  for (int i = 0; i < MAX_STATIC_BLOBS; i++) {
    if (!blob_data->static_blobs[i].name)
      continue;
    if (strncmp(blob_data->static_blobs[i].name, argv[3],
                strlen(blob_data->static_blobs[i].name)) == 0) {
      idx = i;
      break;
    }
  }
  if (idx < 0)
    abort();
  vec_static_blob_entries_vtab *pNew;
#define VEC_STATIC_BLOB_ENTRIES_VECTOR 0
#define VEC_STATIC_BLOB_ENTRIES_DISTANCE 1
#define VEC_STATIC_BLOB_ENTRIES_K 2
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(vector, distance hidden, k hidden)");
  if (rc == SQLITE_OK) {
    pNew = sqlite3_malloc(sizeof(*pNew));
    *ppVtab = (sqlite3_vtab *)pNew;
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
    pNew->blob = &blob_data->static_blobs[idx];
  }
  return rc;
}

static int vec_static_blob_entriesCreate(sqlite3 *db, void *pAux, int argc,
                                         const char *const *argv,
                                         sqlite3_vtab **ppVtab, char **pzErr) {
  return vec_static_blob_entriesConnect(db, pAux, argc, argv, ppVtab, pzErr);
}

static int vec_static_blob_entriesDisconnect(sqlite3_vtab *pVtab) {
  vec_static_blob_entries_vtab *p = (vec_static_blob_entries_vtab *)pVtab;
  sqlite3_free(p);
  return SQLITE_OK;
}

static int vec_static_blob_entriesOpen(sqlite3_vtab *p,
                                       sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec_static_blob_entries_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec_static_blob_entriesClose(sqlite3_vtab_cursor *cur) {
  vec_static_blob_entries_cursor *pCur = (vec_static_blob_entries_cursor *)cur;
  sqlite3_free(pCur->knn_data);
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int vec_static_blob_entriesBestIndex(sqlite3_vtab *pVTab,
                                            sqlite3_index_info *pIdxInfo) {
  vec_static_blob_entries_vtab *p = (vec_static_blob_entries_vtab *)pVTab;
  int iMatchTerm = -1;
  int iLimitTerm = -1;
  // int iRowidTerm = -1; // https://github.com/asg017/sqlite-vec/issues/47
  int iKTerm = -1;

  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    if (!pIdxInfo->aConstraint[i].usable)
      continue;

    int iColumn = pIdxInfo->aConstraint[i].iColumn;
    int op = pIdxInfo->aConstraint[i].op;
    if (op == SQLITE_INDEX_CONSTRAINT_MATCH &&
        iColumn == VEC_STATIC_BLOB_ENTRIES_VECTOR) {
      if (iMatchTerm > -1) {
        // https://github.com/asg017/sqlite-vec/issues/51
        return SQLITE_ERROR;
      }
      iMatchTerm = i;
    }
    if (op == SQLITE_INDEX_CONSTRAINT_LIMIT) {
      iLimitTerm = i;
    }
    if (op == SQLITE_INDEX_CONSTRAINT_EQ &&
        iColumn == VEC_STATIC_BLOB_ENTRIES_K) {
      iKTerm = i;
    }
  }
  if (iMatchTerm >= 0) {
    if (iLimitTerm < 0 && iKTerm < 0) {
      // https://github.com/asg017/sqlite-vec/issues/51
      return SQLITE_ERROR;
    }
    if (iLimitTerm >= 0 && iKTerm >= 0) {
      return SQLITE_ERROR; // limit or k, not both
    }
    if (pIdxInfo->nOrderBy < 1) {
      vtab_set_error(pVTab, "ORDER BY distance required");
      return SQLITE_CONSTRAINT;
    }
    if (pIdxInfo->nOrderBy > 1) {
      // https://github.com/asg017/sqlite-vec/issues/51
      vtab_set_error(pVTab, "more than 1 ORDER BY clause provided");
      return SQLITE_CONSTRAINT;
    }
    if (pIdxInfo->aOrderBy[0].iColumn != VEC_STATIC_BLOB_ENTRIES_DISTANCE) {
      vtab_set_error(pVTab, "ORDER BY must be on the distance column");
      return SQLITE_CONSTRAINT;
    }
    if (pIdxInfo->aOrderBy[0].desc) {
      vtab_set_error(pVTab,
                     "Only ascending in ORDER BY distance clause is supported, "
                     "DESC is not supported yet.");
      return SQLITE_CONSTRAINT;
    }

    pIdxInfo->idxNum = VEC_SBE__QUERYPLAN_KNN;
    pIdxInfo->estimatedCost = (double)10;
    pIdxInfo->estimatedRows = 10;

    pIdxInfo->orderByConsumed = 1;
    pIdxInfo->aConstraintUsage[iMatchTerm].argvIndex = 1;
    pIdxInfo->aConstraintUsage[iMatchTerm].omit = 1;
    if (iLimitTerm >= 0) {
      pIdxInfo->aConstraintUsage[iLimitTerm].argvIndex = 2;
      pIdxInfo->aConstraintUsage[iLimitTerm].omit = 1;
    } else {
      pIdxInfo->aConstraintUsage[iKTerm].argvIndex = 2;
      pIdxInfo->aConstraintUsage[iKTerm].omit = 1;
    }

  } else {
    pIdxInfo->idxNum = VEC_SBE__QUERYPLAN_FULLSCAN;
    pIdxInfo->estimatedCost = (double)p->blob->nvectors;
    pIdxInfo->estimatedRows = p->blob->nvectors;
  }
  return SQLITE_OK;
}

static int vec_static_blob_entriesFilter(sqlite3_vtab_cursor *pVtabCursor,
                                         int idxNum, const char *idxStr,
                                         int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxStr);
  assert(argc >= 0 && argc <= 3);
  vec_static_blob_entries_cursor *pCur =
      (vec_static_blob_entries_cursor *)pVtabCursor;
  vec_static_blob_entries_vtab *p =
      (vec_static_blob_entries_vtab *)pCur->base.pVtab;

  if (idxNum == VEC_SBE__QUERYPLAN_KNN) {
    assert(argc == 2);
    pCur->query_plan = VEC_SBE__QUERYPLAN_KNN;
    struct sbe_query_knn_data *knn_data;
    knn_data = sqlite3_malloc(sizeof(*knn_data));
    if (!knn_data) {
      return SQLITE_NOMEM;
    }
    memset(knn_data, 0, sizeof(*knn_data));

    void *queryVector;
    size_t dimensions;
    enum VectorElementType elementType;
    vector_cleanup cleanup;
    char *err;
    int rc = vector_from_value(argv[0], &queryVector, &dimensions, &elementType,
                               &cleanup, &err);
    if (rc != SQLITE_OK) {
      return SQLITE_ERROR;
    }
    if (elementType != p->blob->element_type) {
      return SQLITE_ERROR;
    }
    if (dimensions != p->blob->dimensions) {
      return SQLITE_ERROR;
    }

    i64 k = min(sqlite3_value_int64(argv[1]), (i64)p->blob->nvectors);
    if (k < 0) {
      // HANDLE https://github.com/asg017/sqlite-vec/issues/55
      return SQLITE_ERROR;
    }
    if (k == 0) {
      knn_data->k = 0;
      pCur->knn_data = knn_data;
      return SQLITE_OK;
    }

    size_t bsize = (p->blob->nvectors + 7) & ~7;

    i32 *topk_rowids = sqlite3_malloc(k * sizeof(i32));
    if (!topk_rowids) {
      // HANDLE https://github.com/asg017/sqlite-vec/issues/55
      return SQLITE_ERROR;
    }
    f32 *distances = sqlite3_malloc(bsize * sizeof(f32));
    if (!distances) {
      // HANDLE https://github.com/asg017/sqlite-vec/issues/55
      return SQLITE_ERROR;
    }

    for (size_t i = 0; i < p->blob->nvectors; i++) {
      // https://github.com/asg017/sqlite-vec/issues/52
      float *v = ((float *)p->blob->p) + (i * p->blob->dimensions);
      distances[i] =
          distance_l2_sqr_float(v, (float *)queryVector, &p->blob->dimensions);
    }
    u8 *candidates = bitmap_new(bsize);
    assert(candidates);

    u8 *taken = bitmap_new(bsize);
    assert(taken);

    bitmap_fill(candidates, bsize);
    for (size_t i = bsize; i >= p->blob->nvectors; i--) {
      bitmap_set(candidates, i, 0);
    }
    i32 k_used = 0;
    min_idx(distances, bsize, candidates, topk_rowids, k, taken, &k_used);
    knn_data->current_idx = 0;
    knn_data->distances = distances;
    knn_data->k = k;
    knn_data->rowids = topk_rowids;

    pCur->knn_data = knn_data;
  } else {
    pCur->query_plan = VEC_SBE__QUERYPLAN_FULLSCAN;
    pCur->iRowid = 0;
  }

  return SQLITE_OK;
}

static int vec_static_blob_entriesRowid(sqlite3_vtab_cursor *cur,
                                        sqlite_int64 *pRowid) {
  vec_static_blob_entries_cursor *pCur = (vec_static_blob_entries_cursor *)cur;
  switch (pCur->query_plan) {
  case VEC_SBE__QUERYPLAN_FULLSCAN: {
    *pRowid = pCur->iRowid;
    return SQLITE_OK;
  }
  case VEC_SBE__QUERYPLAN_KNN: {
    i32 rowid = ((i32 *)pCur->knn_data->rowids)[pCur->knn_data->current_idx];
    *pRowid = (sqlite3_int64)rowid;
    return SQLITE_OK;
  }
  }
  return SQLITE_ERROR;
}

static int vec_static_blob_entriesNext(sqlite3_vtab_cursor *cur) {
  vec_static_blob_entries_cursor *pCur = (vec_static_blob_entries_cursor *)cur;
  switch (pCur->query_plan) {
  case VEC_SBE__QUERYPLAN_FULLSCAN: {
    pCur->iRowid++;
    return SQLITE_OK;
  }
  case VEC_SBE__QUERYPLAN_KNN: {
    pCur->knn_data->current_idx++;
    return SQLITE_OK;
  }
  }
  return SQLITE_ERROR;
}

static int vec_static_blob_entriesEof(sqlite3_vtab_cursor *cur) {
  vec_static_blob_entries_cursor *pCur = (vec_static_blob_entries_cursor *)cur;
  vec_static_blob_entries_vtab *p =
      (vec_static_blob_entries_vtab *)pCur->base.pVtab;
  switch (pCur->query_plan) {
  case VEC_SBE__QUERYPLAN_FULLSCAN: {
    return (size_t)pCur->iRowid >= p->blob->nvectors;
  }
  case VEC_SBE__QUERYPLAN_KNN: {
    return pCur->knn_data->current_idx >= pCur->knn_data->k;
  }
  }
  return SQLITE_ERROR;
}

static int vec_static_blob_entriesColumn(sqlite3_vtab_cursor *cur,
                                         sqlite3_context *context, int i) {
  vec_static_blob_entries_cursor *pCur = (vec_static_blob_entries_cursor *)cur;
  vec_static_blob_entries_vtab *p = (vec_static_blob_entries_vtab *)cur->pVtab;

  switch (pCur->query_plan) {
  case VEC_SBE__QUERYPLAN_FULLSCAN: {
    switch (i) {
    case VEC_STATIC_BLOB_ENTRIES_VECTOR:

      sqlite3_result_blob(
          context,
          ((unsigned char *)p->blob->p) +
              (pCur->iRowid * p->blob->dimensions * sizeof(float)),
          p->blob->dimensions * sizeof(float), SQLITE_TRANSIENT);
      sqlite3_result_subtype(context, p->blob->element_type);
      break;
    }
    return SQLITE_OK;
  }
  case VEC_SBE__QUERYPLAN_KNN: {
    switch (i) {
    case VEC_STATIC_BLOB_ENTRIES_VECTOR: {
      i32 rowid = ((i32 *)pCur->knn_data->rowids)[pCur->knn_data->current_idx];
      sqlite3_result_blob(context,
                          ((unsigned char *)p->blob->p) +
                              (rowid * p->blob->dimensions * sizeof(float)),
                          p->blob->dimensions * sizeof(float),
                          SQLITE_TRANSIENT);
      sqlite3_result_subtype(context, p->blob->element_type);
      break;
    }
    }
    return SQLITE_OK;
  }
  }
  return SQLITE_ERROR;
}

static sqlite3_module vec_static_blob_entriesModule = {
    /* iVersion    */ 3,
    /* xCreate     */
    vec_static_blob_entriesCreate, // handle rm?
                                   // https://github.com/asg017/sqlite-vec/issues/55
    /* xConnect    */ vec_static_blob_entriesConnect,
    /* xBestIndex  */ vec_static_blob_entriesBestIndex,
    /* xDisconnect */ vec_static_blob_entriesDisconnect,
    /* xDestroy    */ vec_static_blob_entriesDisconnect,
    /* xOpen       */ vec_static_blob_entriesOpen,
    /* xClose      */ vec_static_blob_entriesClose,
    /* xFilter     */ vec_static_blob_entriesFilter,
    /* xNext       */ vec_static_blob_entriesNext,
    /* xEof        */ vec_static_blob_entriesEof,
    /* xColumn     */ vec_static_blob_entriesColumn,
    /* xRowid      */ vec_static_blob_entriesRowid,
    /* xUpdate     */ 0,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity  */ 0
#endif
};
#pragma endregion

#ifdef SQLITE_VEC_ENABLE_AVX
#define SQLITE_VEC_DEBUG_BUILD_AVX "avx"
#else
#define SQLITE_VEC_DEBUG_BUILD_AVX ""
#endif
#ifdef SQLITE_VEC_ENABLE_NEON
#define SQLITE_VEC_DEBUG_BUILD_NEON "neon"
#else
#define SQLITE_VEC_DEBUG_BUILD_NEON ""
#endif

#define SQLITE_VEC_DEBUG_BUILD                                                 \
  SQLITE_VEC_DEBUG_BUILD_AVX " " SQLITE_VEC_DEBUG_BUILD_NEON

#define SQLITE_VEC_DEBUG_STRING                                                \
  "Version: " SQLITE_VEC_VERSION "\n"                                          \
  "Date: " SQLITE_VEC_DATE "\n"                                                \
  "Commit: " SQLITE_VEC_SOURCE "\n"                                            \
  "Build flags: " SQLITE_VEC_DEBUG_BUILD

SQLITE_VEC_API int sqlite3_vec_init(sqlite3 *db, char **pzErrMsg,
                                    const sqlite3_api_routines *pApi) {
#ifndef SQLITE_CORE
  SQLITE_EXTENSION_INIT2(pApi);
#endif
  int rc = SQLITE_OK;

#define DEFAULT_FLAGS (SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC)

  rc = sqlite3_create_function_v2(db, "vec_version", 0, DEFAULT_FLAGS,
                                  SQLITE_VEC_VERSION, _static_text_func, NULL,
                                  NULL, NULL);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_create_function_v2(db, "vec_debug", 0, DEFAULT_FLAGS,
                                  SQLITE_VEC_DEBUG_STRING, _static_text_func,
                                  NULL, NULL, NULL);
  if (rc != SQLITE_OK) {
    return rc;
  }
  static struct {
    const char *zFName;
    void (*xFunc)(sqlite3_context *, int, sqlite3_value **);
    int nArg;
    int flags;
  } aFunc[] = {
      // clang-format off
    //{"vec_version",         _static_text_func,    0, DEFAULT_FLAGS,                                          (void *) SQLITE_VEC_VERSION },
    //{"vec_debug",           _static_text_func,    0, DEFAULT_FLAGS,                                          (void *) SQLITE_VEC_DEBUG_STRING },
    {"vec_distance_l2",     vec_distance_l2,      2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_distance_l1",     vec_distance_l1,      2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_distance_hamming",vec_distance_hamming, 2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_distance_cosine", vec_distance_cosine,  2, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_length",          vec_length,           1, DEFAULT_FLAGS | SQLITE_SUBTYPE,                         },
    {"vec_type",           vec_type,           1, DEFAULT_FLAGS,                         },
    {"vec_to_json",         vec_to_json,          1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_add",             vec_add,              2, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_sub",             vec_sub,              2, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_slice",           vec_slice,            3, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_normalize",       vec_normalize,        1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_f32",             vec_f32,              1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_bit",             vec_bit,              1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_int8",            vec_int8,             1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_quantize_int8",     vec_quantize_int8,      2, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
    {"vec_quantize_binary", vec_quantize_binary,  1, DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, },
      // clang-format on
  };

  static struct {
    char *name;
    const sqlite3_module *module;
    void *p;
    void (*xDestroy)(void *);
  } aMod[] = {
      // clang-format off
    {"vec0",          &vec0Module,          NULL, NULL},
    {"vec_each",      &vec_eachModule,      NULL, NULL},
      // clang-format on
  };

  for (unsigned long i = 0; i < countof(aFunc) && rc == SQLITE_OK; i++) {
    rc = sqlite3_create_function_v2(db, aFunc[i].zFName, aFunc[i].nArg,
                                    aFunc[i].flags, NULL, aFunc[i].xFunc, NULL,
                                    NULL, NULL);
    if (rc != SQLITE_OK) {
      *pzErrMsg = sqlite3_mprintf("Error creating function %s: %s",
                                  aFunc[i].zFName, sqlite3_errmsg(db));
      return rc;
    }
  }

  for (unsigned long i = 0; i < countof(aMod) && rc == SQLITE_OK; i++) {
    rc = sqlite3_create_module_v2(db, aMod[i].name, aMod[i].module, NULL, NULL);
    if (rc != SQLITE_OK) {
      *pzErrMsg = sqlite3_mprintf("Error creating module %s: %s", aMod[i].name,
                                  sqlite3_errmsg(db));
      return rc;
    }
  }

  return SQLITE_OK;
}

#ifndef SQLITE_VEC_OMIT_FS
SQLITE_VEC_API int sqlite3_vec_numpy_init(sqlite3 *db, char **pzErrMsg,
                                            const sqlite3_api_routines *pApi) {
  UNUSED_PARAMETER(pzErrMsg);
#ifndef SQLITE_CORE
  SQLITE_EXTENSION_INIT2(pApi);
#endif
  int rc = SQLITE_OK;
  rc = sqlite3_create_function_v2(db, "vec_npy_file", 1, SQLITE_RESULT_SUBTYPE,
                                  NULL, vec_npy_file, NULL, NULL, NULL);
  if(rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_create_module_v2(db, "vec_npy_each", &vec_npy_eachModule, NULL, NULL);
  return rc;
}
#endif

SQLITE_VEC_API int
sqlite3_vec_static_blobs_init(sqlite3 *db, char **pzErrMsg,
                              const sqlite3_api_routines *pApi) {
  UNUSED_PARAMETER(pzErrMsg);
#ifndef SQLITE_CORE
  SQLITE_EXTENSION_INIT2(pApi);
#endif

  int rc = SQLITE_OK;
  vec_static_blob_data *static_blob_data;
  static_blob_data = sqlite3_malloc(sizeof(*static_blob_data));
  if (!static_blob_data) {
    return SQLITE_NOMEM;
  }
  memset(static_blob_data, 0, sizeof(*static_blob_data));

  rc = sqlite3_create_function_v2(
      db, "vec_static_blob_from_raw", 4,
      DEFAULT_FLAGS | SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, NULL,
      vec_static_blob_from_raw, NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_create_module_v2(db, "vec_static_blobs", &vec_static_blobsModule,
                                static_blob_data, sqlite3_free);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_module_v2(db, "vec_static_blob_entries",
                                &vec_static_blob_entriesModule,
                                static_blob_data, NULL);
  if (rc != SQLITE_OK)
    return rc;
  return rc;
}
