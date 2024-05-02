// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2023 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef __AVX512F__
#include <immintrin.h>

// computes expf() for each element in vector.
//
// the maximum error is 1.45358 +0.5 ulp. the only difference between
// this function and expf(), is that there's currently no support for
// subnormals. input values are clamped to range: [-87.6831, 88.3763]
// whereas expf() allows inputs as low as -103.972. therefore numbers
// will be flushed to zero sooner than they otherwise would with this
// function. nearest rounding mode is always used. exception trapping
// isn't supported although this function does a good job avoiding it
//
static inline __m512 llamafile_expf_avx512(__m512 x) {
    __m512 a, b, c, d, e, f, g;
    __m512 will_turn_into_inf = _mm512_set1_ps(0x1.62e44p+6f);
    __m512 max_before_overflow = _mm512_set1_ps(0x1.61814cp+6f);
    __m512 min_before_underflow = _mm512_set1_ps(-0x1.5ebb86p+6f);
    x = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, max_before_overflow, _CMP_GE_OQ), x,
                             will_turn_into_inf);
    x = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, min_before_underflow, _CMP_LE_OQ), x,
                             min_before_underflow);
    a = _mm512_fmadd_round_ps(_mm512_set1_ps(0x1.715476p+0f), x, _mm512_set1_ps(0x1.8p23f),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    b = _mm512_sub_ps(a, _mm512_set1_ps(0x1.8p23f));
    c = _mm512_fnmadd_round_ps(_mm512_set1_ps(0x1.62e4p-1f), b, x,
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    c = _mm512_fnmadd_round_ps(_mm512_set1_ps(0x1.7f7d1cp-20f), b, c,
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    d = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_slli_epi32(_mm512_castps_si512(a), 23),
                                             _mm512_set1_epi32(0x3f800000u)));
    e = _mm512_mul_round_ps(c, c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    f = _mm512_fmadd_round_ps(_mm512_set1_ps(0x1.0e4020p-7f), c, _mm512_set1_ps(0x1.573e2ep-5f),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    g = _mm512_fmadd_round_ps(_mm512_set1_ps(0x1.555e66p-3f), c, _mm512_set1_ps(0x1.fffdb6p-2f),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    g = _mm512_fmadd_round_ps(f, e, g, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    f = _mm512_mul_round_ps(_mm512_set1_ps(0x1.ffffecp-1f), c,
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm512_fmadd_round_ps(
        _mm512_fmadd_round_ps(g, e, f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), d, d,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

// computes silu x/(1+exp(-x)) in single precision
static inline __m512 llamafile_silu_avx512(__m512 x) {
    __m512 one = _mm512_set1_ps(1);
    __m512 zero = _mm512_setzero_ps();
    __m512 neg_x = _mm512_sub_ps(zero, x);
    __m512 exp_neg_x = llamafile_expf_avx512(neg_x);
    __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, one_plus_exp_neg_x);
}

#endif // __AVX512F__

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

// computes expf() for each element in vector.
//
// the maximum error is 1.45358 +0.5 ulp. the only difference between
// this function and expf(), is that there's currently no support for
// subnormals. input values are clamped to range: [-87.6831, 88.3763]
// whereas expf() allows inputs as low as -103.972. therefore numbers
// will be flushed to zero sooner than they otherwise would with this
// function. exception trapping isnt supported although this function
// does a good job avoiding it.
//
static inline __m256 llamafile_expf_avx2fma(__m256 x) {
    __m256 a, b, c, d, e, f, g;
    __m256 will_turn_into_inf = _mm256_set1_ps(0x1.62e44p+6f);
    __m256 max_before_overflow = _mm256_set1_ps(0x1.61814cp+6f);
    __m256 min_before_underflow = _mm256_set1_ps(-0x1.5ebb86p+6f);
    __m256 min_mask = _mm256_cmp_ps(x, min_before_underflow, _CMP_LE_OQ);
    __m256 max_mask = _mm256_cmp_ps(x, max_before_overflow, _CMP_GE_OQ);
    x = _mm256_or_ps(_mm256_and_ps(min_mask, min_before_underflow), _mm256_andnot_ps(min_mask, x));
    x = _mm256_or_ps(_mm256_and_ps(max_mask, will_turn_into_inf), _mm256_andnot_ps(max_mask, x));
    a = _mm256_fmadd_ps(_mm256_set1_ps(0x1.715476p+0f), x, _mm256_set1_ps(0x1.8p23f));
    b = _mm256_sub_ps(a, _mm256_set1_ps(0x1.8p23f));
    c = _mm256_fnmadd_ps(_mm256_set1_ps(0x1.62e4p-1f), b, x);
    c = _mm256_fnmadd_ps(_mm256_set1_ps(0x1.7f7d1cp-20f), b, c);
    d = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(_mm256_castps_si256(a), 23),
                                             _mm256_set1_epi32(0x3f800000u)));
    e = _mm256_mul_ps(c, c);
    f = _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), c, _mm256_set1_ps(0x1.573e2ep-5f));
    g = _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), c, _mm256_set1_ps(0x1.fffdb6p-2f));
    g = _mm256_fmadd_ps(f, e, g);
    f = _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), c);
    return _mm256_fmadd_ps(_mm256_fmadd_ps(g, e, f), d, d);
}

// computes silu x/(1+exp(-x)) in single precision
static inline __m256 llamafile_silu_avx2fma(__m256 x) {
    __m256 one = _mm256_set1_ps(1);
    __m256 zero = _mm256_setzero_ps();
    __m256 neg_x = _mm256_sub_ps(zero, x);
    __m256 exp_neg_x = llamafile_expf_avx2fma(neg_x);
    __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(x, one_plus_exp_neg_x);
}

#endif // __AVX2__

#ifdef __ARM_NEON
#include <arm_neon.h>

float32x4_t v_expf(float32x4_t);

static inline float32x4_t llamafile_expf_neon(float32x4_t x) {
    return v_expf(x);
}

static inline float32x4_t llamafile_silu_neon(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t neg_x = vsubq_f32(zero, x);
    float32x4_t exp_neg_x = llamafile_expf_neon(neg_x);
    float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, one_plus_exp_neg_x);
}

#endif

#ifdef __cplusplus
}
#endif
