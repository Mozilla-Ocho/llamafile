// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once
#ifdef __SSE__
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

#include <immintrin.h>

inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}

#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE__
