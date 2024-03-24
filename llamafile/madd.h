// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once
#ifdef __SSE__
#include <immintrin.h>

inline __m128 madd(__m128 x, __m128 y, __m128 z) {
    return _mm_add_ps(_mm_mul_ps(x, y), z);
}

#ifdef __AVX__

inline __m256 madd(__m256 x, __m256 y, __m256 z) {
    return _mm256_add_ps(_mm256_mul_ps(x, y), z);
}

#ifdef __AVX512F__

inline __m512 madd(__m512 x, __m512 y, __m512 z) {
    return _mm512_add_ps(_mm512_mul_ps(x, y), z);
}

#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE__
