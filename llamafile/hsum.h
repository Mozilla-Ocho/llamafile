// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __x86_64__
#include <immintrin.h>
#endif

inline float hsum(float x) {
    return x;
}

#ifdef __ARM_NEON
inline float hsum(float32x4_t x) {
    return vaddvq_f32(x);
}
#endif

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float hsum(float16x8_t x) {
    float32x4_t t = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t u = vcvt_f32_f16(vget_high_f16(x));
    return vaddvq_f32(vaddq_f32(t, u));
}
#endif

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m128 x) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
#else
    __m128 shuf = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(x, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
#endif
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}
#endif

#if defined(__AVX512F__)
inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#endif
