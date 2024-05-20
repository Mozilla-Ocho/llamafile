// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Iwan Kawrakow
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

#ifdef __x86_64__

#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "sgemm.h"

// clang-format off

// This matrix - vector and matrix - matrix multiplication implementation
// for k-quants and IQ4_XS makes prompt processing 150-200% faster
// compared to mainline llama.cpp (and llamafile).
// It is AVX2 only for now.
//
// Main idea is that unpacking the quants and the block scales to
// be ready for dot products with the corresponding Q8_K quants
// takes time. Hence, if we are performing a QX x Q8_K matrix matrix
// multiplication (as needed for prompt processing), we can get
// a significant speedup by reusing the unpacked QX quants and scales
// for multiplication with several Q8_K columns.

#include <utility>

namespace {

typedef struct {
    int32_t i1;
    int32_t i2;
} mmid_row_mapping;

struct DataInfo {
    float       * s;
    const char  * cy;
    size_t        bs;
    size_t        by;
    int           cur_y = 0;
    int           ne11;
    const mmid_row_mapping * row_mapping = nullptr;
    size_t        bs2 = 0;

    inline const char * src1_row(int iy) const {
        if (!row_mapping) return cy + (cur_y + iy)*by;
        int i11 = row_mapping[cur_y + iy].i1 % ne11;
        int i12 = row_mapping[cur_y + iy].i2;
        return cy + (i11 + i12*ne11)*by;
    }

    inline void store(int ix, int iy, float result) const {
        *(dst_row(iy) + ix) = result;
        //dst_row(iy)[ix] = result;
    }
    inline float * dst_row(int iy) const {
        if (!row_mapping) return s + (cur_y + iy)*bs;
        int i12 = row_mapping[cur_y + iy].i2;
        int i1  = row_mapping[cur_y + iy].i1;
        int i2  = i12;
        return s + i1*bs + i2*bs2;
    }
};

inline void make_q4_scales(const uint8_t * scales8, uint32_t * aux32) {
    const uint16_t * scales = (const uint16_t *)scales8;
    const uint32_t a0 = scales[0] | (scales[1] << 16);
    const uint32_t a1 = scales[2] | (scales[3] << 16);
    const uint32_t a2 = scales[4] | (scales[5] << 16);
    aux32[3] = ((a2 >> 4) & 0x0f0f0f0f) | ((a1 >> 2) & 0x30303030);
    aux32[1] = ((a2 >> 0) & 0x0f0f0f0f) | ((a0 >> 2) & 0x30303030);
    aux32[2] = a1 & 0x3f3f3f3f;
    aux32[0] = a0 & 0x3f3f3f3f;
}

inline __m256i get_scale_shuffle_8(int i) {
    return _mm256_set1_epi16((2*i) | ((2*i+1) << 8));
}

static inline __m256i get_scale_shuffle_16(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

static inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
static inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

typedef void (*mul_mat_t)(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x);

inline void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y,
        mul_mat_t mm_nx1, mul_mat_t mm_nx2, mul_mat_t mm_nx4, mul_mat_t mm_nx8 = nullptr) {
    auto process = [n, &info, vx, bx, nrc_x, nrc_y] (mul_mat_t mul_mat, int step) {
        if (!mul_mat) return true;
        int n_step = (nrc_y - info.cur_y)/step;
        for (int iy = 0; iy < n_step; ++iy) {
            mul_mat(n, vx, bx, info, nrc_x);
            info.cur_y += step;
        }
        return info.cur_y < nrc_y;
    };
    process(mm_nx8, 8) && process(mm_nx4, 4) && process(mm_nx2, 2) && process(mm_nx1, 1);
}

template <int nrc_y, typename block_q8 = block_q8_K> struct Q8 {

    Q8(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8 *)info.src1_row(iy);
    }

    inline __m256i load_quants(int iy, int i, int j) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].qs + j); }
    inline __m256i load_bsums(int iy, int i) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].bsums); }
    inline float scale(int iy, int i) const { return y[iy][i].d; }

    const block_q8 * y[nrc_y];
};

}

//
// ================================== q2_K =============================================
//
namespace {
template <int nrc_y>
static void mul_mat_q2_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    constexpr int k_nrc = nrc_y <= 2 ? 2*nrc_y : nrc_y;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mc = _mm256_set1_epi8(12);
    const __m128i m4 = _mm_set1_epi8(0xF);

    Q8<nrc_y> q8(info);

    __m256i scales[2];
    __m256i sumi[k_nrc];
    __m256  accd[k_nrc];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < k_nrc; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {

            const block_q2_K * x = (const block_q2_K *)((const char *)vx + ix*bx);
            const uint8_t * q2 = x[i].qs;

            const float d2 = GGML_FP16_TO_FP32(x[i].d);
            const float c2 = -GGML_FP16_TO_FP32(x[i].dmin);

            {
                const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
                const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
                const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
                const __m256i mins = _mm256_cvtepi8_epi16(mins8);

                for (int iy = 0; iy < nrc_y; ++iy) {
                    const __m256i prod = _mm256_madd_epi16(mins, q8.load_bsums(iy, i));
                    if (nrc_y <= 2) {
                        accd[2*iy+0] = _mm256_fmadd_ps(_mm256_set1_ps(c2 * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accd[2*iy+0]);
                    } else {
                        accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(c2 * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accd[iy]);
                    }
                }

                const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
                const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
                const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
                scales[0] = MM256_SET_M128I(l_scales, l_scales);
                scales[1] = MM256_SET_M128I(h_scales, h_scales);
            }

            for (int iy = 0; iy < k_nrc; ++iy) sumi[iy] = _mm256_setzero_si256();

            for (int j = 0; j < QK_K/128; ++j) {

                __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

                for (int l = 0; l < 2; ++l) {

                    const __m256i scales_0 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(2*l+0));
                    const __m256i scales_1 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(2*l+1));

                    const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
                    const __m256i q2_1 = nrc_y <= 2 ? _mm256_and_si256(q2bits, mc)
                                                    : _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);

                    for (int iy = 0; iy < nrc_y; ++iy) {

                        const __m256i p0 = _mm256_maddubs_epi16(q2_0, q8.load_quants(iy, i,  4*j + 2*l + 0));
                        const __m256i p1 = _mm256_maddubs_epi16(q2_1, q8.load_quants(iy, i,  4*j + 2*l + 1));

                        if (nrc_y <= 2) {
                            sumi[2*iy+0] = _mm256_add_epi32(sumi[2*iy+0], _mm256_madd_epi16(scales_0, p0));
                            sumi[2*iy+1] = _mm256_add_epi32(sumi[2*iy+1], _mm256_madd_epi16(scales_1, p1));
                        } else {
                            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(_mm256_madd_epi16(scales_0, p0), _mm256_madd_epi16(scales_1, p1)));
                        }

                    }

                    q2bits = _mm256_srli_epi16(q2bits, 4);

                }

            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(d2 * q8.scale(iy, i));
                if (nrc_y <= 2) {
                    accd[2*iy+0] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[2*iy+0]), accd[2*iy+0]);
                    accd[2*iy+1] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[2*iy+1]), accd[2*iy+1]);
                } else {
                    accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
                }
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            if (nrc_y <= 2) {
                info.store(ix, iy, hsum_float_8(accd[2*iy+0]) + 0.25f*hsum_float_8(accd[2*iy+1]));
            } else {
                info.store(ix, iy, hsum_float_8(accd[iy]));
            }
        }

    }
}

//
// ================================== q3_K =============================================
//
template <int nrc_y>
static void mul_mat_q3_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Q8<nrc_y> q8(info);

    const __m256i m3l = _mm256_set1_epi8(0x03);
    const __m128i m32 = _mm_set1_epi8(32);
    const __m256i hml = _mm256_set1_epi8(0x04);

    __m256i scales[2];
    __m256i hbits[2];
    __m256  vd[nrc_y];

    uint32_t aux[3];

    for (int ix = 0; ix < nrc_x; ++ix) {

        const block_q3_K * x = (const block_q3_K *)((const char *)vx + ix*bx);

        __m256 accd[nrc_y], accm[nrc_y];
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = accm[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {

            const float d3 = GGML_FP16_TO_FP32(x[i].d);
            const uint8_t * q3 = x[i].qs;

            // Set up scales
            {
                const uint16_t * scales16 = (const uint16_t *)x[i].scales;
                aux[0] = scales16[0] | (scales16[1] << 16);
                aux[1] = scales16[2] | (scales16[3] << 16);
                aux[2] = scales16[4] | (scales16[5] << 16);
                __m128i scales128 = _mm_set_epi32(
                        ((aux[1] >> 4) & 0x0f0f0f0f) | ((aux[2] >> 2) & 0x30303030),
                        ((aux[0] >> 4) & 0x0f0f0f0f) | ((aux[2] >> 0) & 0x30303030),
                         (aux[1]       & 0x0f0f0f0f) | ((aux[2] << 2) & 0x30303030),
                         (aux[0]       & 0x0f0f0f0f) | ((aux[2] << 4) & 0x30303030));
                scales128 = _mm_sub_epi8(scales128, m32);
                const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    vd[iy] = _mm256_set1_ps(d3 * q8.scale(iy, i));
                    const __m256i prod  = _mm256_madd_epi16(all_scales, q8.load_bsums(iy, i));
                    accm[iy] = _mm256_fmadd_ps(vd[iy], _mm256_cvtepi32_ps(prod), accm[iy]);
                }
                const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
                const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
                scales[0] = MM256_SET_M128I(l_scales, l_scales);
                scales[1] = MM256_SET_M128I(h_scales, h_scales);
            }

            // high bit
            hbits[0] = _mm256_loadu_si256((const __m256i*)x[i].hmask);
            hbits[1] = _mm256_srli_epi16(hbits[0], 4);

            // integer accumulator
            __m256i sumi[nrc_y];
            for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = _mm256_setzero_si256();

            for (int j = 0; j < QK_K/128; ++j) {

                const __m256i scales_0 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(0));
                const __m256i scales_1 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(1));
                const __m256i scales_2 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(2));
                const __m256i scales_3 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(3));

                // load low 2 bits
                const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

                const __m256i q3h_0 = _mm256_and_si256(_mm256_slli_epi16(hbits[j], 2), hml);
                const __m256i q3h_1 = _mm256_and_si256(_mm256_slli_epi16(hbits[j], 1), hml);
                const __m256i q3h_2 = _mm256_and_si256(hbits[j], hml);
                const __m256i q3h_3 = _mm256_and_si256(_mm256_srli_epi16(hbits[j], 1), hml);

                // prepare low and high bits
                const __m256i q3_0 = _mm256_or_si256(_mm256_and_si256(q3bits, m3l), q3h_0);
                const __m256i q3_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3l), q3h_1);
                const __m256i q3_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3l), q3h_2);
                const __m256i q3_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3l), q3h_3);

                for (int iy = 0; iy < nrc_y; ++iy) {

                    __m256i p16_0 = _mm256_maddubs_epi16(q3_0, q8.load_quants(iy, i, 4*j+0));
                    __m256i p16_1 = _mm256_maddubs_epi16(q3_1, q8.load_quants(iy, i, 4*j+1));
                    __m256i p16_2 = _mm256_maddubs_epi16(q3_2, q8.load_quants(iy, i, 4*j+2));
                    __m256i p16_3 = _mm256_maddubs_epi16(q3_3, q8.load_quants(iy, i, 4*j+3));

                    // multiply with scales
                    p16_0 = _mm256_madd_epi16(scales_0, p16_0);
                    p16_1 = _mm256_madd_epi16(scales_1, p16_1);
                    p16_2 = _mm256_madd_epi16(scales_2, p16_2);
                    p16_3 = _mm256_madd_epi16(scales_3, p16_3);

                    sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p16_0, p16_1));
                    sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p16_2, p16_3));
                }

            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                // multiply with block scale and accumulate
                accd[iy] = _mm256_fmadd_ps(vd[iy], _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]) - 4.f*hsum_float_8(accm[iy]));
        }

    }

}

//
// ================================== q4_K =============================================
//

template <int nrc_y>
static void mul_mat_q4_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    constexpr int k_nrc = nrc_y <= 2 ? 2*nrc_y : nrc_y;

    Q8<nrc_y> q8(info);

    uint32_t utmp[4];

    const __m256i ml = _mm256_set1_epi8(0x0F);
    const __m256i mh = _mm256_set1_epi8(-16); // to avoid stupid warnings if using 0xF0

    __m128  accm[nrc_y];
    __m256i sumi[k_nrc];
    __m256  accd[k_nrc];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) {
            accm[iy] = _mm_setzero_ps();
            if (nrc_y <= 2) {
                accd[2*iy+0] = accd[2*iy+1] = _mm256_setzero_ps();
            } else {
                accd[iy] = _mm256_setzero_ps();
            }
        }

        const block_q4_K * x = (const block_q4_K *)((const char *)vx + bx*ix);

        for (int i = 0; i < nb; ++i) {

            const float d = GGML_FP16_TO_FP32(x[i].d), c = -GGML_FP16_TO_FP32(x[i].dmin);

            const uint8_t * q4 = x[i].qs;

            __m256i scales;
            {
                make_q4_scales(x[i].scales, utmp);
                const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
                const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
                const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                scales = MM256_SET_M128I(sc128, sc128);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const __m256i q8sums = q8.load_bsums(iy, i);
                    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                    const __m128i prod = _mm_madd_epi16(mins, q8s);
                    accm[iy] = _mm_fmadd_ps(_mm_set1_ps(c*q8.scale(iy, i)), _mm_cvtepi32_ps(prod), accm[iy]);
                }
            }

            for (int iy = 0; iy < k_nrc; ++iy) sumi[iy] = _mm256_setzero_si256();

            for (int j = 0; j < QK_K/64; ++j) {

                const __m256i scales_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_8(2*j+0));
                const __m256i scales_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_8(2*j+1));
                const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                const __m256i q4l = _mm256_and_si256(q4bits, ml);
                const __m256i q4h = nrc_y <= 2 ? _mm256_and_si256(q4bits, mh) : _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);

                for (int iy = 0; iy < nrc_y; ++iy) {
                    const __m256i q8l = q8.load_quants(iy, i, 2*j+0);
                    const __m256i q8h = q8.load_quants(iy, i, 2*j+1);
                    if (nrc_y <= 2) {
                        sumi[2*iy+0] = _mm256_add_epi32(sumi[2*iy+0], _mm256_madd_epi16(scales_l, _mm256_maddubs_epi16(q4l, q8l)));
                        sumi[2*iy+1] = _mm256_add_epi32(sumi[2*iy+1], _mm256_madd_epi16(scales_h, _mm256_maddubs_epi16(q4h, q8h)));
                    } else {
                        const __m256i pl = _mm256_madd_epi16(scales_l, _mm256_maddubs_epi16(q4l, q8l));
                        const __m256i ph = _mm256_madd_epi16(scales_h, _mm256_maddubs_epi16(q4h, q8h));
                        sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(pl, ph));
                    }
                }
            }

            if (nrc_y <= 2) {
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const __m256 vd = _mm256_set1_ps(d*q8.scale(iy, i));
                    accd[2*iy+0] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[2*iy+0]), accd[2*iy+0]);
                    accd[2*iy+1] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[2*iy+1]), accd[2*iy+1]);
                }
            } else {
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const __m256 vd = _mm256_set1_ps(d*q8.scale(iy, i));
                    accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
                }
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            if (nrc_y <= 2) {
                info.store(ix, iy, hsum_float_8(accd[2*iy+0]) + 0.0625f*hsum_float_8(accd[2*iy+1]) + hsum_float_4(accm[iy]));
            } else {
                const __m128 d = _mm_add_ps(_mm256_castps256_ps128(accd[iy]), _mm256_extractf128_ps(accd[iy], 1));
                info.store(ix, iy, hsum_float_4(_mm_add_ps(d, accm[iy])));
            }
        }

    }
}

//
// ========================================= q5_K ========================================================
//
template <int nrc_y>
static void mul_mat_q5_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(info);

    uint32_t utmp[4];

    const __m256i ml = _mm256_set1_epi8(0x0F);
    const __m256i mh = _mm256_set1_epi8(0x10);

    for (int ix = 0; ix < nrc_x; ++ix) {

        __m128 accm[nrc_y]; for (int iy = 0; iy < nrc_y; ++iy) accm[iy] = _mm_setzero_ps();
        __m256 accd[nrc_y]; for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        const block_q5_K * x = (const block_q5_K *)((const char *)vx + bx*ix);

        for (int i = 0; i < nb; ++i) {

            const float d = GGML_FP16_TO_FP32(x[i].d), c = -GGML_FP16_TO_FP32(x[i].dmin);

            const uint8_t * q5 = x[i].qs;

            __m256i scales;
            {
                make_q4_scales(x[i].scales, utmp);
                const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
                const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
                const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                scales = MM256_SET_M128I(sc128, sc128);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const __m256i q8sums = q8.load_bsums(iy, i);
                    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                    const __m128i prod = _mm_madd_epi16(mins, q8s);
                    accm[iy] = _mm_fmadd_ps(_mm_set1_ps(c*q8.scale(iy, i)), _mm_cvtepi32_ps(prod), accm[iy]);
                }
            }

            __m256i hbits[2];
            hbits[0] = _mm256_loadu_si256((const __m256i*)x[i].qh);
            hbits[1] = _mm256_srli_epi16(hbits[0], 4);

            __m256i sumi[nrc_y]; for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = _mm256_setzero_si256();

            for (int j = 0; j < QK_K/128; ++j) {

                const __m256i scales_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_8(4*j+0));
                const __m256i scales_2 = _mm256_shuffle_epi8(scales, get_scale_shuffle_8(4*j+1));
                const __m256i scales_3 = _mm256_shuffle_epi8(scales, get_scale_shuffle_8(4*j+2));
                const __m256i scales_4 = _mm256_shuffle_epi8(scales, get_scale_shuffle_8(4*j+3));

                const __m256i q5h_1 = _mm256_and_si256(_mm256_slli_epi16(hbits[j], 4), mh);
                const __m256i q5h_2 = _mm256_and_si256(_mm256_slli_epi16(hbits[j], 3), mh);
                const __m256i q5h_3 = _mm256_and_si256(_mm256_slli_epi16(hbits[j], 2), mh);
                const __m256i q5h_4 = _mm256_and_si256(_mm256_slli_epi16(hbits[j], 1), mh);

                __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;
                const __m256i q5_1  = _mm256_add_epi8(_mm256_and_si256(q5bits, ml), q5h_1);
                const __m256i q5_2  = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), ml), q5h_2);

                q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;
                const __m256i q5_3  = _mm256_add_epi8(_mm256_and_si256(q5bits, ml), q5h_3);
                const __m256i q5_4  = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), ml), q5h_4);

                for (int iy = 0; iy < nrc_y; ++iy) {
                    const __m256i p1  = _mm256_madd_epi16(scales_1, _mm256_maddubs_epi16(q5_1, q8.load_quants(iy, i, 4*j+0)));
                    const __m256i p2  = _mm256_madd_epi16(scales_2, _mm256_maddubs_epi16(q5_2, q8.load_quants(iy, i, 4*j+1)));
                    const __m256i p3  = _mm256_madd_epi16(scales_3, _mm256_maddubs_epi16(q5_3, q8.load_quants(iy, i, 4*j+2)));
                    const __m256i p4  = _mm256_madd_epi16(scales_4, _mm256_maddubs_epi16(q5_4, q8.load_quants(iy, i, 4*j+3)));
                    sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p1, p3));
                    sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p2, p4));
                }
            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(d*q8.scale(iy, i));
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            const __m128 d = _mm_add_ps(_mm256_castps256_ps128(accd[iy]), _mm256_extractf128_ps(accd[iy], 1));
            info.store(ix, iy, hsum_float_4(_mm_add_ps(d, accm[iy])));
        }

    }

}

//
// ========================================= q6_K ========================================================
//

template <int nrc_y>
static void mul_mat_q6_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i mh = _mm256_set1_epi8(0x30);

    Q8<nrc_y> q8(info);

    __m256i scales[2];
    __m256  vd[nrc_y];
    __m256  accm[nrc_y];
    __m256  accd[nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {

        const block_q6_K * x = (const block_q6_K *)((const char *)vx + ix*bx);

        for (int iy = 0; iy < nrc_y; ++iy) accm[iy] = accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {

            const float d6 = GGML_FP16_TO_FP32(x[i].d);

            const uint8_t * q4 = x[i].ql;
            const uint8_t * qh = x[i].qh;

            const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
            const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
            scales[0] = MM256_SET_M128I(_mm256_castsi256_si128(scales16), _mm256_castsi256_si128(scales16));
            scales[1] = MM256_SET_M128I(_mm256_extractf128_si256(scales16, 1), _mm256_extractf128_si256(scales16, 1));

            for (int iy = 0; iy < nrc_y; ++iy) {
                vd[iy] = _mm256_set1_ps(d6 * q8.scale(iy, i));
                const __m256i prod  = _mm256_madd_epi16(scales16, q8.load_bsums(iy, i));
                accm[iy] = _mm256_fmadd_ps(vd[iy], _mm256_cvtepi32_ps(prod), accm[iy]);
            }

            __m256i sumi[nrc_y];
            for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = _mm256_setzero_si256();

            for (int j = 0; j < QK_K/128; ++j) {

                const __m256i scale_0 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(0));
                const __m256i scale_1 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(1));
                const __m256i scale_2 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(2));
                const __m256i scale_3 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_16(3));

                const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
                const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

                const __m256i q4h_0 = _mm256_and_si256(_mm256_slli_epi16(q4bitsH, 4), mh);
                const __m256i q4h_1 = _mm256_and_si256(_mm256_slli_epi16(q4bitsH, 2), mh);
                const __m256i q4h_2 = _mm256_and_si256(q4bitsH, mh);
                const __m256i q4h_3 = _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), mh);

                const __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                const __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                const __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                const __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

                for (int iy = 0; iy < nrc_y; ++iy) {

                    __m256i p16_0 = _mm256_maddubs_epi16(q6_0, q8.load_quants(iy, i, 4*j+0));
                    __m256i p16_1 = _mm256_maddubs_epi16(q6_1, q8.load_quants(iy, i, 4*j+1));
                    __m256i p16_2 = _mm256_maddubs_epi16(q6_2, q8.load_quants(iy, i, 4*j+2));
                    __m256i p16_3 = _mm256_maddubs_epi16(q6_3, q8.load_quants(iy, i, 4*j+3));

                    p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                    p16_1 = _mm256_madd_epi16(scale_1, p16_1);
                    p16_2 = _mm256_madd_epi16(scale_2, p16_2);
                    p16_3 = _mm256_madd_epi16(scale_3, p16_3);

                    sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(_mm256_add_epi32(p16_0, p16_1), _mm256_add_epi32(p16_2, p16_3)));
                }

            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(vd[iy], _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]) - 32.f*hsum_float_8(accm[iy]));
        }

    }
}

//
// ========================================= IQ4_XS ========================================================
//

template <int nrc_y>
static void mul_mat_iq4_xs_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m256i values = MM256_SET_M128I(values128, values128);

    static const uint8_t k_shuffle[16] = {0, 4, 1, 5, 2, 6, 3, 7, 0, 4, 1, 5, 2, 6, 3, 7};
    const __m128i hshift = _mm_set_epi32(12, 8, 4, 0);
    const __m128i lshift = _mm_set_epi32(4, 0, 4, 0);
    const __m128i hmask  = _mm_set1_epi16(0x03);
    const __m128i lmask  = _mm_set1_epi8(0xf);
    const __m128i lshuffle = _mm_loadu_si128((const __m128i *)k_shuffle);
    const __m128i m32 = _mm_set1_epi16(-32);
    const __m256i m4 = _mm256_set1_epi8(0xf);

    auto dequant = [&m4] (const uint8_t * qs) {
        const __m128i aux128 = _mm_loadu_si128((const __m128i *)qs);
        const __m256i aux256 = MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128);
        return _mm256_and_si256(m4, aux256);
    };
    auto mul_signed = [] (__m256i x, __m256i y) {
        const __m256i ux = _mm256_sign_epi8(x, x);
        const __m256i sy = _mm256_sign_epi8(y, x);
        return _mm256_maddubs_epi16(ux, sy);
    };

    Q8<nrc_y> q8(info);

    for (int ix = 0; ix < nrc_x; ++ix) {

        const block_iq4_xs * x = (const block_iq4_xs *)((const char *)vx + ix*bx);

        __m256 accum[nrc_y];
        for(int iy = 0; iy < nrc_y; ++iy) accum[iy] = _mm256_setzero_ps();

        for (int ibl = 0; ibl < nb; ++ibl) {
            const uint8_t * qs = x[ibl].qs;
            uint32_t tmp32 = x[ibl].scales_h | (x[ibl].scales_h << 14);
            const __m128i sh = _mm_slli_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(tmp32), hshift), hmask), 4);
            const __m128i sl = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(*(const uint32_t *)x[ibl].scales_l), lshift), lmask);
            const __m128i scales128 = _mm_add_epi16(_mm_or_si128(sh, _mm_cvtepi8_epi16(_mm_shuffle_epi8(sl, lshuffle))), m32);
            const __m256i scales = MM256_SET_M128I(scales128, scales128);
            __m256i sumi[nrc_y];
            for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = _mm256_setzero_si256();
            for (int j = 0; j < QK_K/64; ++j) {
                const __m256i q4b_1 = _mm256_shuffle_epi8(values, dequant(qs)); qs += 16;
                const __m256i q4b_2 = _mm256_shuffle_epi8(values, dequant(qs)); qs += 16;
                const __m256i scales_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_8(2*j+0));
                const __m256i scales_2 = _mm256_shuffle_epi8(scales, get_scale_shuffle_8(2*j+1));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const __m256i p16_1 = mul_signed(q4b_1, q8.load_quants(iy, ibl, 2*j+0));
                    const __m256i p16_2 = mul_signed(q4b_2, q8.load_quants(iy, ibl, 2*j+1));
                    const __m256i p_1 = _mm256_madd_epi16(p16_1, scales_1);
                    const __m256i p_2 = _mm256_madd_epi16(p16_2, scales_2);
                    sumi[iy] = _mm256_add_epi32(_mm256_add_epi32(p_1, p_2), sumi[iy]);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(GGML_FP16_TO_FP32(x[ibl].d)*q8.scale(iy, ibl));
                accum[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accum[iy]);
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accum[iy]));
        }

    }

}

//
// ============================== Legacy quants
//

struct DotHelper {
    const __m256i m1 = _mm256_set1_epi16(1);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    inline __m256i dot(__m256i x, __m256i y) const {
        return _mm256_dpbusd_epi32(_mm256_setzero_si256(), x, y);
    }
#else
    inline __m256i dot(__m256i x, __m256i y) const {
        return _mm256_madd_epi16(m1, _mm256_maddubs_epi16(x, y));
    }
#endif
};

struct SignedDot {
    DotHelper helper;
    inline __m256i compute(__m256i x, __m256i y) const {
        return helper.dot(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
    }
};
struct UnsignedDot {
    DotHelper helper;
    inline __m256i compute(__m256i x, __m256i y) const {
        return helper.dot(x, y);
    }
};
template <typename Q8, typename Dot> struct Sum4 {
    Dot dot;
    inline __m256i compute(const __m256i * qx, const Q8 * y) const {
        const __m256i p0 = dot.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[0].qs));
        const __m256i p1 = dot.compute(qx[1], _mm256_loadu_si256((const __m256i *)y[1].qs));
        const __m256i p2 = dot.compute(qx[2], _mm256_loadu_si256((const __m256i *)y[2].qs));
        const __m256i p3 = dot.compute(qx[3], _mm256_loadu_si256((const __m256i *)y[3].qs));
        const __m256i p01 = _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p0, p1));    // 0,0, 1,1, 0,0, 1,1
        const __m256i p23 = _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p2, p3));    // 2,2, 3,3, 2,2, 3,3
        return _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p01, p23)); // 0,1,2,3, 0,1,2,3
    }
};

struct Sum4_Q8 {
    SignedDot dot;
    static inline __m256i add1(__m256i a, __m256i b) {
        return _mm256_add_epi32(_mm256_unpacklo_epi32(a, b), _mm256_unpackhi_epi32(a, b));
    }
    static inline __m256i add2(__m256i a, __m256i b) {
        return _mm256_add_epi32(_mm256_unpacklo_epi64(a, b), _mm256_unpackhi_epi64(a, b));
    }
    inline __m256i compute(const __m256i * qx, const block_q8_0 * y) const {
        const __m256i p0 = dot.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[0].qs));
        const __m256i p1 = dot.compute(qx[1], _mm256_loadu_si256((const __m256i *)y[1].qs));
        const __m256i p2 = dot.compute(qx[2], _mm256_loadu_si256((const __m256i *)y[2].qs));
        const __m256i p3 = dot.compute(qx[3], _mm256_loadu_si256((const __m256i *)y[3].qs));
        const __m256i p01 = add1(p0, p1);  // 0,1, 0,1, 0,1, 0,1
        const __m256i p23 = add1(p2, p3);  // 2,3, 2,3, 2,3, 2,3
        return add2(p01, p23); // returns 0,1,2,3, 0,1,2,3
    }
};

struct ScaleHelperQ_0 {
    ggml_half scales8[4];
    template <typename Q>
    inline __m128 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales8[j] = y[j].d;
        return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)scales8));
    }
    template <typename Q>
    inline __m128 prepare4(__m128 other_scales, const Q * y) {
        return _mm_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline float prepare1(const Q * y) const { return GGML_FP16_TO_FP32(y->d); }
    template <typename Q> inline float prepare1(float d, const Q * y) const { return d*prepare1(y); }
};

struct ScaleHelperQ_1 {
    uint32_t scales8[4];
    const __m128i shuffle = _mm_set_epi16(0x0f0e, 0x0b0a, 0x0706, 0x0302, 0x0d0c, 0x0908, 0x0504, 0x0100);

    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) {
            // it is slightly faster to directly dereference (const uint32 *)&y[j].d, but some compilers
            // complain that this breaks strict-aliasing rules.
            memcpy(scales8 + j, &y[j].d, sizeof(uint32_t));
        }
        return _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)scales8), shuffle));
    }

    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm256_mul_ps(other_scales, prepare4<Q>(y));
    }

    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        return std::make_pair(GGML_FP16_TO_FP32(y->d), GGML_FP16_TO_FP32(y->m));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const std::pair<float, float>& dm, const Q * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->m));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
};

struct MinusType0 {
    inline __m256 compute(__m128 d, int) const { return _mm256_set_m128(d, d); }
    inline float compute(float d, int) const { return d; }
    inline float result(__m256 acc, int) const { return hsum_float_8(acc); }
};

template <int nrc_y> struct MinusType1 {
    __m128 accm[nrc_y];
    MinusType1() { for (int iy = 0; iy < nrc_y; ++iy) accm[iy] = _mm_setzero_ps(); }
    inline __m256 compute(__m256 dm, int iy) {
        const __m128 d = _mm256_castps256_ps128(dm);
        const __m128 m = _mm256_extractf128_ps(dm, 1);
        accm[iy] = _mm_add_ps(accm[iy], m);
        return _mm256_set_m128(d, d);
    }
    inline float compute(const std::pair<float, float>& dm, int iy) {
        accm[iy] = _mm_add_ps(accm[iy], _mm_set1_ps(dm.second*0.25f));
        return dm.first;
    }
    inline float result(__m256 acc, int iy) const {
        const __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        return hsum_float_4(_mm_add_ps(sum, accm[iy]));
    }
};

template <typename Minus, int nrc_y, bool is_multiple_of_4> struct AccumT {
    __m256 acc[nrc_y];
    Minus accm;
    AccumT() {  for (int iy = 0; iy < nrc_y; ++iy) acc[iy] = _mm256_setzero_ps(); }
    template <typename Unpacker, typename Scales, typename Sum, typename Q8>
    inline void compute(int nb, Unpacker& unp, Scales& scales, Sum& sum, const Q8 ** y, const DataInfo& info, int ix) {
        auto qx = unp.quants();
        __m256 dall[nrc_y];
        for (int i = 0; i < nb/4; ++i) {
            auto other_scales = unp.set_block_4(i);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto s12 = scales.prepare4(other_scales, y[iy] + 4*i);
                dall[iy] = accm.compute(s12, iy);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto pall = sum.compute(qx, y[iy] + 4*i);
                acc[iy] = _mm256_fmadd_ps(dall[iy], _mm256_cvtepi32_ps(pall), acc[iy]);
            }
        }
        if (!is_multiple_of_4) {
            for (int i = 4*(nb/4); i < nb; ++i) {
                auto other_scales = unp.set_block(i);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto s12 = scales.prepare1(other_scales, y[iy] + i);
                    auto d = accm.compute(s12, iy);
                    const __m256i p0 = sum.dot.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[iy][i].qs));
                    acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, accm.result(acc[iy], iy));
            //s[iy*bs] = accm.result(acc[iy], iy);
        }
    }
};

template <int nrc_y, bool is_multiple_of_4>
using AccumType0 = AccumT<MinusType0, nrc_y, is_multiple_of_4>;

template <int nrc_y, bool is_multiple_of_4>
using AccumType1 = AccumT<MinusType1<nrc_y>, nrc_y, is_multiple_of_4>;

using Sum4Type0 = Sum4<block_q8_0, SignedDot>;
using Sum4Type1 = Sum4<block_q8_1, UnsignedDot>;

template <typename Unpacker, typename Sum4Type, typename AccumType, typename Scales, typename Q8, int nrc_y>
void mul_mat_qX_q8_Helper(int nb, const void * vx, size_t bx, const DataInfo& info, const Q8 ** y, int nrc_x) {
    Unpacker unp(vx, bx);
    Sum4Type sum4;
    Scales scales;
    for (int ix = 0; ix < nrc_x; ++ix) {
        unp.set_row(ix);
        AccumType accum;
        accum.compute(nb, unp, scales, sum4, y, info, ix);
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_0_q8_0_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_0> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, Sum4Type0, AccumType0<nrc_y, true>, ScaleHelperQ_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, Sum4Type0, AccumType0<nrc_y, false>, ScaleHelperQ_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_1_q8_1_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_1> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, Sum4Type1, AccumType1<nrc_y, true>, ScaleHelperQ_1, block_q8_1, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, Sum4Type1, AccumType1<nrc_y, false>, ScaleHelperQ_1, block_q8_1, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

struct Dequantizer4bit {
    const __m256i m4 = _mm256_set1_epi8(0xf);
    inline __m256i dequant(const uint8_t * qs) const {
        const __m128i aux128 = _mm_loadu_si128((const __m128i *)qs);
        return _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128), m4);
    }
};

struct Q8_0_Dequantizer {
    inline __m256i dequant(const block_q8_0 * x) const {
        return _mm256_loadu_si256((const __m256i *)x->qs);
    }
};

struct Q4_0_Dequantizer {
    Dequantizer4bit b4;
    const __m256i m8 = _mm256_set1_epi8(-8);
    inline __m256i dequant(const block_q4_0 * x) const {
        return _mm256_add_epi8(b4.dequant(x->qs), m8);
    }
};

struct Q4_1_Dequantizer {
    Dequantizer4bit b4;
    inline __m256i dequant(const block_q4_1 * x) const {
        return b4.dequant(x->qs);
    }
};

struct HBitDequantizer {
    const __m256i shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const __m256i mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    const __m256i minus1 = _mm256_set1_epi64x(-1);
    inline __m256i to_bytes(const uint8_t * bits) const {
        // Note: Data in all ggml quants is at least 2-byte aligned.
        // => we can cast to uint16_t and use or on two consecutive entries
        // which is faster than memcpy
        const uint16_t * aux16 = (const uint16_t *)bits;
        const uint32_t aux32 = aux16[0] | (aux16[1] << 16);
        //uint32_t aux32; memcpy(&aux32, bits, sizeof(uint32_t));
        __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(aux32), shuffle);
        bytes = _mm256_or_si256(bytes, mask);
        return _mm256_cmpeq_epi8(bytes, minus1);
    }
};

struct Q5_0_Dequantizer {
    Dequantizer4bit b4;
    HBitDequantizer hbit;
    const __m256i mh = _mm256_set1_epi8((char)0xF0);
    inline __m256i dequant(const block_q5_0 * x) const {
        const __m256i vqh = _mm256_andnot_si256(hbit.to_bytes(x->qh), mh);
        return _mm256_or_si256(b4.dequant(x->qs), vqh);
    }
};

struct Q5_1_Dequantizer {
    Dequantizer4bit b4;
    HBitDequantizer hbit;
    const __m256i mh = _mm256_set1_epi8(0x10);
    inline __m256i dequant(const block_q5_1 * x) const {
        const __m256i vqh = _mm256_and_si256(hbit.to_bytes(x->qh), mh);
        return _mm256_or_si256(b4.dequant(x->qs), vqh);
    }
};

template <typename Q, typename Scales, typename Dequantizer>
struct Q_Unpacker {
    Q_Unpacker(const void * vx, size_t bx) : cx_0((const char *)vx), x((const Q*)cx_0), bx(bx) {}

    const char * cx_0;
    const Q    * x;
    size_t       bx;

    Scales scales;
    Dequantizer deq;

    __m256i qx[4];

    inline const __m256i* quants() const { return qx; }

    inline void set_row(int ix) { x = (const Q*)(cx_0 + ix*bx); }

    inline auto set_block_4(int i) {
        for (int j = 0; j < 4; ++j) {
            qx[j] = deq.dequant(x + 4*i + j);
        }
        return scales.prepare4(x + 4*i);
    }
    inline auto set_block(int i) {
        qx[0] = deq.dequant(x + i);
        return scales.prepare1(x + i);
    }
};

struct Q8_0_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0, Q8_0_Dequantizer> {
    Q8_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK4_0; }
};
struct Q4_0_Unpacker final : public Q_Unpacker<block_q4_0, ScaleHelperQ_0, Q4_0_Dequantizer> {
    Q4_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK4_0; }
};
struct Q5_0_Unpacker final : public Q_Unpacker<block_q5_0, ScaleHelperQ_0, Q5_0_Dequantizer> {
    Q5_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK5_0; }
};
struct Q4_1_Unpacker final : public Q_Unpacker<block_q4_1, ScaleHelperQ_1, Q4_1_Dequantizer> {
    Q4_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK4_1; }
};
struct Q5_1_Unpacker final : public Q_Unpacker<block_q5_1, ScaleHelperQ_1, Q5_1_Dequantizer> {
    Q5_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK4_1; }
};

template <int nrc_y>
void mul_mat_q8_0_q8_0_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Q8_0_Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_0> q8(info);
    int nb = n/Q8_0_Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Q8_0_Unpacker, Sum4_Q8, AccumType0<nrc_y, true>, ScaleHelperQ_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Q8_0_Unpacker, Sum4_Q8, AccumType0<nrc_y, false>, ScaleHelperQ_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

bool set_mul_mat(int typeA, int ne00, mul_mat_t& mm_nx1, mul_mat_t& mm_nx2, mul_mat_t& mm_nx4, mul_mat_t& mm_nx8, int& row_size_q8) {

    row_size_q8 = ggml_row_size(GGML_TYPE_Q8_K, ne00);
    mm_nx1 = mm_nx2 = mm_nx4 = mm_nx8 = nullptr;

    switch (typeA) {
        case GGML_TYPE_Q2_K:
            assert (ne00 % QK_K == 0);
            mm_nx1 = mul_mat_q2_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q2_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q2_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q2_K_q8_K_T<8>;
            break;
        case GGML_TYPE_Q3_K:
            assert (ne00 % QK_K == 0);
            mm_nx1 = mul_mat_q3_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q3_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q3_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q3_K_q8_K_T<8>;
            break;
        case GGML_TYPE_Q4_K:
            assert (ne00 % QK_K == 0);
            mm_nx1 = mul_mat_q4_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q4_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q4_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q4_K_q8_K_T<8>;
            break;
        case GGML_TYPE_Q5_K:
            assert (ne00 % QK_K == 0);
            mm_nx1 = mul_mat_q5_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q5_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q5_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q5_K_q8_K_T<8>;
            break;
        case GGML_TYPE_Q6_K:
            assert (ne00 % QK_K == 0);
            mm_nx1 = mul_mat_q6_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q6_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q6_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q6_K_q8_K_T<8>;
            break;
        case GGML_TYPE_IQ4_XS:
            assert (ne00 % QK_K == 0);
            mm_nx1 = mul_mat_iq4_xs_q8_K_T<1>;
            mm_nx2 = mul_mat_iq4_xs_q8_K_T<2>;
            mm_nx4 = mul_mat_iq4_xs_q8_K_T<4>;
            mm_nx8 = mul_mat_iq4_xs_q8_K_T<8>;
            break;
        case GGML_TYPE_Q4_0:
            assert (ne00 % Q4K_0 == 0);
            mm_nx1 = mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 1>;
            mm_nx2 = mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 2>;
            mm_nx4 = mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 4>;
            mm_nx8 = mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 8>;
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, ne00);
            break;
        case GGML_TYPE_Q4_1:
            assert (ne00 % Q4K_1 == 0);
            mm_nx1 = mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 1>;
            mm_nx2 = mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 2>;
            mm_nx4 = mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 4>;
            mm_nx8 = mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 8>;
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_1, ne00);
            break;
        case GGML_TYPE_Q5_0:
            assert (ne00 % Q5K_0 == 0);
            mm_nx1 = mul_mat_qX_0_q8_0_T<Q5_0_Unpacker, 1>;
            mm_nx2 = mul_mat_qX_0_q8_0_T<Q5_0_Unpacker, 2>;
            mm_nx4 = mul_mat_qX_0_q8_0_T<Q5_0_Unpacker, 4>;
            mm_nx8 = mul_mat_qX_0_q8_0_T<Q5_0_Unpacker, 8>;
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, ne00);
            break;
        case GGML_TYPE_Q5_1:
            assert (ne00 % Q0K_1 == 0);
            mm_nx1 = mul_mat_qX_1_q8_1_T<Q5_1_Unpacker, 1>;
            mm_nx2 = mul_mat_qX_1_q8_1_T<Q5_1_Unpacker, 2>;
            mm_nx4 = mul_mat_qX_1_q8_1_T<Q5_1_Unpacker, 4>;
            mm_nx8 = mul_mat_qX_1_q8_1_T<Q5_1_Unpacker, 8>;
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_1, ne00);
            break;

        default:
            return false;
    }

    return true;
}

} // namespace

//
// ============================== Matrix multiplications
//

bool iqk_mul_mat(long Nx, long Ny, long ne00, int typeA, const void * A, const void * B,
        float * C, long stride_C, int ith, int nth) {

    mul_mat_t mm_nx1, mm_nx2, mm_nx4, mm_nx8;
    int row_size_q8;
    if (!set_mul_mat(typeA, ne00, mm_nx1, mm_nx2, mm_nx4, mm_nx8, row_size_q8)) {
        return false;
    }

    auto row_size_qx = ggml_row_size((ggml_type)typeA, ne00);

    auto nrc_x = (Nx + nth - 1)/nth;
    auto first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;

    DataInfo info{C + first_x, (const char *)B, (size_t)stride_C, (size_t)row_size_q8, 0, 1, nullptr, 0};

    mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x, row_size_qx, info,
                nrc_x, Ny, mm_nx1, mm_nx2, mm_nx4, mm_nx8);

    return true;
}

bool iqk_mul_mat_moe(int Nx, int Ny, int ne00, int ne11, int typeA, const void * A, const void * B,
        float * C, size_t nb1, size_t nb2, const void * vrow_mapping, int ith, int nth) {
    const mmid_row_mapping * row_mapping = (const mmid_row_mapping *)vrow_mapping;
    assert(row_mapping != nullptr);

    mul_mat_t mm_nx1, mm_nx2, mm_nx4, mm_nx8;
    int row_size_q8;
    if (!set_mul_mat(typeA, ne00, mm_nx1, mm_nx2, mm_nx4, mm_nx8, row_size_q8)) {
        return false;
    }
    int row_size_qx = ggml_row_size((ggml_type)typeA, ne00);
    int nrc_x = (Nx + nth - 1)/nth;
    int first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;
    DataInfo info{C + first_x, (const char *)B, nb1/sizeof(float), (size_t)row_size_q8, 0, ne11, row_mapping, nb2/sizeof(float)};
    mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x, row_size_qx, info,
                nrc_x, Ny, mm_nx1, mm_nx2, mm_nx4, mm_nx8);
    return true;
}

#endif // __x86_64__
