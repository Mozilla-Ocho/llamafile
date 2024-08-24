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
#include <stdint.h>
#include <stdbool.h>

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

static inline float
llamafile_fp8_e4m3_to_fp32(ggml_fp8_t f)
{
    union
    {
        ggml_fp8_t f;
        unsigned char i;
    } in = { f };
    union
    {
        float f;
        unsigned i;
    } u;
    unsigned x = in.i;
    // // quantization checks for nans
    // if ((x & 127) == 127)
    //   return x==255 ? -NAN : +NAN;
    auto exp_8 = x & 0x78;
    auto exp_32 = (exp_8 + (120 << 3)) << 20;
    u.i = (x & 7) << 20;                       // mantissa: bit 2-0 -> 22-20
    u.i |= exp_8 ? exp_32 : (-6 + 127) << 23;  // exponent
    if (!exp_8)
        u.f -= 1.0 / 64;      // 2⁻⁶
    u.i |= (x & 0x80) << 24;  // sign
    return u.f;
}

static inline ggml_fp8_t
llamafile_fp32_to_fp8_e4m3(float f)
{
    union
    {
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
            if (exp_bits == 15 && mantissa_bits == 0x07)
                mantissa_bits = 6;
            out.i = sign | (exp_bits << 3) | mantissa_bits;
        }
    }
    return out.f;
}

static inline unsigned
llamafile_fp8_select(bool b, unsigned x, unsigned y)
{
    unsigned p = b - 1;
    return (x & ~p) | (y & p);
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
static __m512
llamafile_from_fp8_e4m3_avx512(__m128i fp8_vec)
{
    // extract componants
    __m128i expo_8 = _mm_and_si128(fp8_vec, _mm_set1_epi8(0x78));
    __m128i mant_8 = _mm_and_si128(fp8_vec, _mm_set1_epi8(0x07));
    __m128i sign_8 = _mm_and_si128(fp8_vec, _mm_set1_epi8(0x80));
    // denorm mask
    __mmask16 is_denorm = _mm_cmpeq_epi8_mask(expo_8, _mm_set1_epi8(0));
    // convert to 32 bits
    __m512i expo_32 = _mm512_cvtepu8_epi32(expo_8);
    __m512i mant_32 = _mm512_cvtepu8_epi32(mant_8);
    __m512i sign_32 = _mm512_cvtepu8_epi32(sign_8);
    // shift
    expo_32 = _mm512_slli_epi32(_mm512_add_epi32(expo_32,_mm512_set1_epi32(120<<3)), 20);
    mant_32 = _mm512_slli_epi32(mant_32, 20);
    sign_32 = _mm512_slli_epi32(sign_32, 24);
    // correction denorm expo
    expo_32 = _mm512_mask_blend_epi32(is_denorm, expo_32, _mm512_set1_epi32((-6 + 127) << 23));
    // merge mantissa+exponent
    __m512 result = _mm512_castsi512_ps(_mm512_or_si512(expo_32,mant_32));
    // correction denorm mantissa
    result = _mm512_mask_add_ps(result, is_denorm, result, _mm512_set1_ps(-1.0/64));
    // add sign
    return _mm512_castsi512_ps(_mm512_or_si512(sign_32,_mm512_castps_si512(result)));
}
#elif defined(__AVX512F__)
#include <immintrin.h>
static inline __m512
llamafile_from_fp8_e4m3_avx512(__m128i xi)
{
    __m512i x = _mm512_cvtepu8_epi32(xi);
    return _mm512_castsi512_ps(
      _mm512_or_si512(
        _mm512_mask_blend_epi32(
          _mm512_cmpneq_epi32_mask(_mm512_and_si512(x, _mm512_set1_epi32(120)),
                                   _mm512_setzero_si512()),
          _mm512_castps_si512(_mm512_sub_ps(
            _mm512_castsi512_ps(_mm512_or_si512(
              _mm512_slli_epi32(_mm512_and_si512(x, _mm512_set1_epi32(7)), 14),
              _mm512_set1_epi32(127 << 23))),
            _mm512_set1_ps(1))),
          _mm512_or_si512(
            _mm512_slli_epi32(_mm512_and_si512(x, _mm512_set1_epi32(7)), 20),
            _mm512_slli_epi32(
              _mm512_add_epi32(_mm512_and_si512(_mm512_srli_epi32(x, 3),
                                                _mm512_set1_epi32(15)),
                               _mm512_set1_epi32(120)),
              23))),
        _mm512_slli_epi32(_mm512_and_si512(x, _mm512_set1_epi32(128)), 24)));
}
#endif // __AVX512F__

static inline ggml_bf16_t
llamafile_fp8_e4m3_to_bf16(ggml_fp8_t fp8)
{
    union
    {
        ggml_fp8_t f;
        unsigned char i;
    } in = { fp8 };
    union
    {
        ggml_bf16_t f;
        unsigned short i;
    } out;
    unsigned x = in.i;
    if (x & 127) {
        if (x & 120) {
            out.i = (x & 7) << 4;
            out.i |= (((x >> 3) & 15) + 120) << 7;
        } else {
            int lg2mant = llamafile_fp8_select(x & 2, 1, 0);
            lg2mant = llamafile_fp8_select(x & 4, 2, lg2mant);
            out.i = ((x & 3) << (7 - lg2mant)) & 0x007f;
            out.i |= (lg2mant + 118) << 7;
        }
    } else {
        out.i = 0;
    }
    out.i |= (x & 128) << 8;
    return out.f;
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
static inline __m512bh
llamafile_fp8_e4m3_to_bf16_avx512(__m256i fp8_vec)
{
    __m512i x = _mm512_cvtepu8_epi16(fp8_vec);
    __m512i lg2mant = _mm512_mask_mov_epi16(
      _mm512_mask_mov_epi16(_mm512_setzero_si512(),
                            _mm512_test_epi16_mask(x, _mm512_set1_epi16(2)),
                            _mm512_set1_epi16(1)),
      _mm512_test_epi16_mask(x, _mm512_set1_epi16(4)),
      _mm512_set1_epi16(2));
    return (__m512bh)_mm512_or_si512(
      _mm512_maskz_mov_epi16(
        _mm512_cmpneq_epi16_mask(_mm512_and_si512(x, _mm512_set1_epi16(127)),
                                 _mm512_setzero_si512()),
        _mm512_mask_blend_epi16(
          _mm512_test_epi16_mask(x, _mm512_set1_epi16(120)),
          _mm512_or_si512(
            _mm512_and_si512(_mm512_sllv_epi16(
                               _mm512_and_si512(x, _mm512_set1_epi16(3)),
                               _mm512_sub_epi16(_mm512_set1_epi16(7), lg2mant)),
                             _mm512_set1_epi16(0x007f)),
            _mm512_slli_epi16(_mm512_add_epi16(lg2mant, _mm512_set1_epi16(118)),
                              7)),
          _mm512_or_si512(
            _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(7)), 4),
            _mm512_slli_epi16(
              _mm512_add_epi16(
                _mm512_srli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(120)),
                                  3),
                _mm512_set1_epi16(120)),
              7)))),
      _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(128)), 8));
}
#endif // __AVX512F__ + __AVX512BW__
