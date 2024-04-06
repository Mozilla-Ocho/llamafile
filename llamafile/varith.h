// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
// clang-format off
#pragma once

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __x86_64__
#include <immintrin.h>
#endif

inline float add(float x, float y) { return x + y; }
inline float sub(float x, float y) { return x - y; }
inline float mul(float x, float y) { return x * y; }

#ifdef __SSE__
inline __m128 add(__m128 x, __m128 y) { return _mm_add_ps(x, y); }
inline __m128 sub(__m128 x, __m128 y) { return _mm_sub_ps(x, y); }
inline __m128 mul(__m128 x, __m128 y) { return _mm_mul_ps(x, y); }
#endif  // __SSE__

#ifdef __AVX__
inline __m256 add(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
#endif // __AVX__

#ifdef __AVX512F__
inline __m512 add(__m512 x, __m512 y) { return _mm512_add_ps(x, y); }
inline __m512 sub(__m512 x, __m512 y) { return _mm512_sub_ps(x, y); }
inline __m512 mul(__m512 x, __m512 y) { return _mm512_mul_ps(x, y); }
#endif // __AVX512F__

#ifdef __ARM_NEON
inline float32x4_t add(float32x4_t x, float32x4_t y) { return vaddq_f32(x, y); }
inline float32x4_t sub(float32x4_t x, float32x4_t y) { return vsubq_f32(x, y); }
inline float32x4_t mul(float32x4_t x, float32x4_t y) { return vmulq_f32(x, y); }
#endif // __ARM_NEON

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float16x8_t add(float16x8_t x, float16x8_t y) { return vaddq_f16(x, y); }
inline float16x8_t sub(float16x8_t x, float16x8_t y) { return vsubq_f16(x, y); }
inline float16x8_t mul(float16x8_t x, float16x8_t y) { return vmulq_f16(x, y); }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

/**
 * Computes a * b + c.
 *
 * This operation will become fused into a single arithmetic instruction
 * if the hardware has support for this feature, e.g. Intel Haswell+ (c.
 * 2013), AMD Bulldozer+ (c. 2011), etc.
 */
template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

/**
 * Computes a * b + c with error correction.
 *
 * @see W. Kahan, "Further remarks on reducing truncation errors,"
 *    Communications of the ACM, vol. 8, no. 1, p. 40, Jan. 1965,
 *    doi: 10.1145/363707.363723.
 */
template <typename T, typename U>
inline U madder(T a, T b, U c, U *e) {
    U y = sub(mul(a, b), *e);
    U t = add(c, y);
    *e = sub(sub(t, c), y);
    return t;
}
