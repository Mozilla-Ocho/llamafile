// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)
#include <arm_neon.h>

inline float hsum(float32x4_t x) {
    return vaddvq_f32(x);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float hsum(float16x8_t x) {
    float32x4_t t = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t u = vcvt_f32_f16(vget_high_f16(x));
    return vaddvq_f32(vaddq_f32(t, u));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#elif defined(__SSE__)
#include <immintrin.h>

inline float hsum(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}

#ifdef __AVX__

inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}

#ifdef __AVX512F__

inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}

#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE__

template <typename T> float hsums(const T *x, int n) {
    float sum = 0;
    for (int i = 0; i < n; ++i)
        sum += hsum(x[i]);
    return sum;
}
