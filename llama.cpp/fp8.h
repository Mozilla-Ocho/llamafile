// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "ggml.h"
#include <math.h>

// FP8 (E4M3)
//
//   Exponent bits : 4
//   Mantissa bits : 3
//   Exponent bias : 7
//   Infinities    : N/A
//   NaN           : S.1111.111
//   Zeros         : S.0000.000
//   Max normal    : S.1111.110 = 1.75 * 2**8 = 448
//   Min normal    : S.0001.000 = 2**(−6)
//   Max subnorm   : S.0000.111 = 0.875 ∗ 2**(−6)
//   Min subnorm   : S.0000.001 = 2**(−9)
//
// See "FP8 Formats For Deep Learning"
//   §3 FP8 Binary Interchange Format
//        NVIDIA / ARM / Intel

static ggml_fp8_t llamafile_fp32_to_fp8_e4m3(float f) {
    union {
        unsigned char i;
        ggml_fp8_t f;
    } out;
    uint8_t sign = signbit(f) ? 128 : 0;
    if (isnan(f)) {
        out.i = sign | 127;
    } else if (!f) {
        out.i = sign;
    } else {
        f = fabsf(f);
        int exp = floorf(log2f(f));
        float mantissa = f / exp2f(exp) - 1;
        if (exp < -6) {
            mantissa = f / exp2f(-6); // subnormal
            exp = -7;
        }
        if (exp > 8) {
            out.i = sign | 0x7E; // overflow
        } else {
            uint8_t exp_bits = (exp + 7) & 15;
            uint8_t mantissa_bits = (uint8_t)(mantissa * 8) & 7;
            // [jpp] avoid generate NAN ?
            if (exp_bits == 15 && mantissa_bits == 0x07) mantissa_bits = 6;
            out.i = sign | (exp_bits << 3) | mantissa_bits;
        }
    }
    return out.f;
}

static inline unsigned llamafile_fp8_select(bool b, unsigned x, unsigned y) {
    unsigned p = b - 1;
    return (x & ~p) | (y & p);
}

static float llamafile_fp8_e4m3_to_fp32(ggml_fp8_t fp8) {
    union {
        ggml_fp8_t f;
        unsigned char i;
    } in = {fp8};
    union {
        float f;
        unsigned i;
    } u;
    unsigned x = in.i;
    u.i = (x & 128) << 24; // sign
    if (x & 127) {
        if ((x & 127) == 127) {
            u.i |= 0x7fc00001; // nan
        } else if ((x & 127) >= 8) {
            u.i |= (x & 7) << 20; // mantissa: bit 2-0 -> 22-20
            u.i |= (((x >> 3) & 15) + 120) << 23; // exponent
        } else {
            int lg2mant = llamafile_fp8_select(x & 2, 1, 0);
            lg2mant = llamafile_fp8_select(x & 4, 2, lg2mant);
            u.i |= ((x & 3) << (23 - lg2mant)) & 0x007fffff; // mantissa
            u.i |= (lg2mant + 118) << 23; // exponent
        }
    }
    return u.f;
}

#if defined(__AVX512F__) && defined(__AVX512VL__)
#include <immintrin.h>
static __m512 llamafile_from_fp8_e4m3_avx512(__m128i fp8_vec) {
    __m512i x = _mm512_cvtepu8_epi32(fp8_vec);
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(x, _mm512_set1_epi32(128)), 24);
    __m512i mantissa = _mm512_and_si512(x, _mm512_set1_epi32(7));
    __m512i exponent = _mm512_and_si512(_mm512_srli_epi32(x, 3), _mm512_set1_epi32(15));
    __mmask16 is_zero = _mm512_cmpeq_epi32_mask(_mm512_and_si512(x, _mm512_set1_epi32(127)),
                                                _mm512_setzero_si512());
    __mmask16 is_nan_inf = _mm512_cmpeq_epi32_mask(_mm512_and_si512(x, _mm512_set1_epi32(127)),
                                                   _mm512_set1_epi32(127));
    __mmask16 is_normal = _mm512_cmpge_epi32_mask(exponent, _mm512_set1_epi32(1));
    __m512i normal_mantissa = _mm512_slli_epi32(mantissa, 20);
    __m512i normal_exponent =
        _mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(120)), 23);
    __m512i subnormal_lg2mant =
        _mm512_mask_blend_epi32(_mm512_cmpgt_epi32_mask(mantissa, _mm512_set1_epi32(1)),
                                _mm512_set1_epi32(0), _mm512_set1_epi32(1));
    subnormal_lg2mant =
        _mm512_mask_blend_epi32(_mm512_cmpgt_epi32_mask(mantissa, _mm512_set1_epi32(3)),
                                subnormal_lg2mant, _mm512_set1_epi32(2));
    __m512i subnormal_mantissa = _mm512_and_si512(
        _mm512_sllv_epi32(mantissa, _mm512_sub_epi32(_mm512_set1_epi32(23), subnormal_lg2mant)),
        _mm512_set1_epi32(0x007fffff));
    __m512i subnormal_exponent =
        _mm512_slli_epi32(_mm512_add_epi32(subnormal_lg2mant, _mm512_set1_epi32(118)), 23);
    __m512i result =
        _mm512_mask_blend_epi32(is_normal, _mm512_or_si512(subnormal_mantissa, subnormal_exponent),
                                _mm512_or_si512(normal_mantissa, normal_exponent));
    result = _mm512_mask_blend_epi32(is_nan_inf, result, _mm512_set1_epi32(0x7fc00001));
    result = _mm512_or_si512(result, sign);
    result = _mm512_mask_mov_epi32(result, is_zero, sign);
    return _mm512_castsi512_ps(result);
}
#endif // __AVX512F__ + __AVX512VL__
