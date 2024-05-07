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
//
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
//

namespace {

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

typedef void (*mul_mat_t)(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc_x);

inline void mul_mat_NxM(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc_x, int nrc_y,
        mul_mat_t mm_nx1, mul_mat_t mm_nx2, mul_mat_t mm_nx4, mul_mat_t mm_nx8) {
    const char * char_y = (const char *)vy;
    auto process = [n, &s, bs, &char_y, by, vx, bx, nrc_x, &nrc_y] (mul_mat_t mul_mat, int step) {
        if (!mul_mat || nrc_y < step) return true;
        int n_step = nrc_y/step;
        for (int iy = 0; iy < n_step; ++iy) {
            mul_mat(n, s + step*iy*bs, bs, vx, bx, char_y + step*iy*by, by, nrc_x);
        }
        nrc_y -= step*n_step;
        if (nrc_y == 0) return false;
        char_y += step*n_step*by;
        s += step*n_step*bs;
        return true;
    };
    process(mm_nx8, 8) && process(mm_nx4, 4) && process(mm_nx2, 2) && process(mm_nx1, 1);
}

template <int nrc_y> struct Q8 {

    Q8(const void * vy, int by) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_K *)((const char *)vy + iy*by);
    }

    inline __m256i load_quants(int iy, int i, int j) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].qs + j); }
    inline __m256i load_bsums(int iy, int i) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].bsums); }
    inline float scale(int iy, int i) const { return y[iy][i].d; }

    const block_q8_K * y[nrc_y];
};

}

//
// ================================== q2_K =============================================
//
namespace {
template <int nrc_y>
static void mul_mat_q2_K_q8_K_T(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    constexpr int k_nrc = nrc_y <= 2 ? 2*nrc_y : nrc_y;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mc = _mm256_set1_epi8(12);
    const __m128i m4 = _mm_set1_epi8(0xF);

    Q8<nrc_y> q8(vy, by);

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
                s[ix+iy*bs] = hsum_float_8(accd[2*iy+0]) + 0.25f*hsum_float_8(accd[2*iy+1]);
            } else {
                s[ix+iy*bs] = hsum_float_8(accd[iy]);
            }
        }

    }
}

//
// ================================== q3_K =============================================
//
template <int nrc_y>
static void mul_mat_q3_K_q8_K_T(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Q8<nrc_y> q8(vy, by);

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
            s[ix + iy*bs] = hsum_float_8(accd[iy]) - 4.f*hsum_float_8(accm[iy]);
        }

    }

}

//
// ================================== q4_K =============================================
//

template <int nrc_y>
static void mul_mat_q4_K_q8_K_T(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    constexpr int k_nrc = nrc_y <= 2 ? 2*nrc_y : nrc_y;

    Q8<nrc_y> q8(vy, by);

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
                s[ix+iy*bs] = hsum_float_8(accd[2*iy+0]) + 0.0625f*hsum_float_8(accd[2*iy+1]) + hsum_float_4(accm[iy]);
            } else {
                const __m128 d = _mm_add_ps(_mm256_castps256_ps128(accd[iy]), _mm256_extractf128_ps(accd[iy], 1));
                s[ix+iy*bs] = hsum_float_4(_mm_add_ps(d, accm[iy]));
            }
        }

    }
}

//
// ========================================= q5_K ========================================================
//
template <int nrc_y>
static void mul_mat_q5_K_q8_K_T(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(vy, by);

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
            s[ix+iy*bs] = hsum_float_4(_mm_add_ps(d, accm[iy]));
        }

    }

}

//
// ========================================= q6_K ========================================================
//

template <int nrc_y>
static void mul_mat_q6_K_q8_K_T(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i mh = _mm256_set1_epi8(0x30);

    Q8<nrc_y> q8(vy, by);

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
            s[ix+iy*bs] = hsum_float_8(accd[iy]) - 32.f*hsum_float_8(accm[iy]);
        }

    }
}

//
// ========================================= IQ4_XS ========================================================
//

template <int nrc_y>
static void mul_mat_iq4_xs_q8_K_T(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc_x) {
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

    Q8<nrc_y> q8(vy, by);

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
            s[ix+iy*bs] = hsum_float_8(accum[iy]);
        }

    }

}

//
// ============================== Matrix multiplications
//

bool iqk_mul_mat(long Nx, long Ny, long ne00, int typeA, const void * A, const void * B,
        float * C, long stride_C, int ith, int nth) {

    assert (ne00 % QK_K == 0);

    mul_mat_t mm_nx1, mm_nx2, mm_nx4, mm_nx8;
    switch (typeA) {
        case GGML_TYPE_Q2_K:
            mm_nx1 = mul_mat_q2_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q2_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q2_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q2_K_q8_K_T<8>;
            break;
        case GGML_TYPE_Q3_K:
            mm_nx1 = mul_mat_q3_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q3_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q3_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q3_K_q8_K_T<8>;
            break;
        case GGML_TYPE_Q4_K:
            mm_nx1 = mul_mat_q4_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q4_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q4_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q4_K_q8_K_T<8>;
            break;
        case GGML_TYPE_Q5_K:
            mm_nx1 = mul_mat_q5_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q5_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q5_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q5_K_q8_K_T<8>;
            break;
        case GGML_TYPE_Q6_K:
            mm_nx1 = mul_mat_q6_K_q8_K_T<1>;
            mm_nx2 = mul_mat_q6_K_q8_K_T<2>;
            mm_nx4 = mul_mat_q6_K_q8_K_T<4>;
            mm_nx8 = mul_mat_q6_K_q8_K_T<8>;
            break;
        case GGML_TYPE_IQ4_XS:
            mm_nx1 = mul_mat_iq4_xs_q8_K_T<1>;
            mm_nx2 = mul_mat_iq4_xs_q8_K_T<2>;
            mm_nx4 = mul_mat_iq4_xs_q8_K_T<4>;
            mm_nx8 = mul_mat_iq4_xs_q8_K_T<8>;
            break;
        default:
            return false;
    }

    auto row_size_qx = ggml_row_size((ggml_type)typeA, ne00);
    auto row_size_q8 = ggml_row_size(GGML_TYPE_Q8_K, ne00);

    auto nrc_x = (Nx + nth - 1)/nth;
    auto first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;

    mul_mat_NxM(ne00, C + first_x, stride_C,
                (const char *)A + row_size_qx*first_x, row_size_qx,
                (const char *)B, row_size_q8,
                nrc_x, Ny,
                mm_nx1, mm_nx2, mm_nx4, mm_nx8);

    return true;
}

}
