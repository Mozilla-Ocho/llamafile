// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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

#include "llama.cpp/ggml-quants.h"
#include "llamafile.h"
#include <immintrin.h>
#include <stdio.h>

//
//                     _   _          ___ _      _   ___
//                    | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                    |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                     \__|_|_||_\_, |___/____/_/ \_\___/
//                               |__/
//
//                      BASIC LINEAR ALGEBRA SUBPROGRAMS
//

#define ASSERT(x)                                                              \
    if (!(x)) {                                                                \
        fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__,   \
                #x);                                                           \
        __builtin_trap();                                                      \
    }

#define BEGIN_KERNEL(RM, RN)                                                   \
    long ytiles = (m - m0) / RM;                                               \
    long xtiles = (n - n0) / RN;                                               \
    long tiles = ytiles * xtiles;                                              \
    double duty = (double)tiles / nth;                                         \
    if (duty < 1)                                                              \
        duty = 1;                                                              \
    double spot = duty * ith + .5;                                             \
    long end = spot + duty;                                                    \
    long start = spot;                                                         \
    if (end > tiles)                                                           \
        end = tiles;                                                           \
    for (long job = start; job < end; ++job) {                                 \
        long i = m0 + job / xtiles * RM;                                       \
        long j = n0 + job % xtiles * RN;

#define END_KERNEL() }

#ifdef __x86_64__
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")

dontinline static float hsum_float_8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256 loadf16x8(const ggml_fp16_t *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}

static inline __m256i vpdpbusd(__m256i x, __m256i y, __m256i s) {
    register __m256i ymm2 asm("ymm2") = x;
    register __m256i ymm1 asm("ymm1") = y;
    register __m256i ymm0 asm("ymm0") = s;
    asm(".byte\t0xc4,0xe2,0x6d,0x50,0xc1" : "+x"(ymm0) : "x"(ymm1), "x"(ymm2));
    return ymm0;
}

static inline __m256 mul_sum_i8_pairs_float(__m256i x, __m256i y) {
    return _mm256_cvtepi32_ps(
        vpdpbusd(_mm256_sign_epi8(x, x),   // make unsigned
                 _mm256_sign_epi8(y, x),   // make signed
                 _mm256_setzero_si256())); // no accumulator
}

static inline __m256 mul_sum_signed_i8_pairs_float(__m256i u, __m256i s) {
    return _mm256_cvtepi32_ps(vpdpbusd(u, s, _mm256_setzero_si256()));
}

class tinyBLAS_F32_AVX2 {
  public:
    tinyBLAS_F32_AVX2(long k, const float *A, long lda, const float *B,
                      long ldb, float *C, long ldc, long ith, long nth)
        : k(k), A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith),
          nth(nth) {
        ASSERT(A != nullptr);
        ASSERT(B != nullptr);
        ASSERT(C != nullptr);
        ASSERT(k >= 0 && k % 8 == 0);
        ASSERT(ith >= 0 && ith < nth);
    }

    void gemm(long m, long n) {
        ASSERT(m >= 0);
        ASSERT(n >= 0);
        ASSERT(lda >= k);
        ASSERT(ldb >= k);
        ASSERT(ldc >= m);
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(long m0, long m, long n0, long n) {
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        long mc, nc, mp, np;
        if (m - m0 >= 3 && n - n0 >= 4) {
            mc = 3;
            nc = 4;
            gemm3x4(m0, m, n0, n);
        } else if (m - m0 >= 4 && n - n0 >= 1) {
            mc = 4;
            nc = 1;
            gemm4x1(m0, m, n0, n);
        } else if (m - m0 >= 1 && n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm1x4(m0, m, n0, n);
        } else {
            mc = 1;
            nc = 1;
            gemm1x1(m0, m, n0, n);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, mp, np, n);
        mnpack(mp, m, np, n);
    }

    dontinline void gemm3x4(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(3, 4)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c11 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps();
        __m256 c13 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c21 = _mm256_setzero_ps();
        __m256 c22 = _mm256_setzero_ps();
        __m256 c23 = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            __m256 k0 = _mm256_loadu_ps(B + ldb * (j + 0) + l);
            __m256 k1 = _mm256_loadu_ps(B + ldb * (j + 1) + l);
            __m256 k2 = _mm256_loadu_ps(B + ldb * (j + 2) + l);
            __m256 k3 = _mm256_loadu_ps(B + ldb * (j + 3) + l);
            __m256 a0 = _mm256_loadu_ps(A + lda * (i + 0) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            c01 = _mm256_fmadd_ps(a0, k1, c01);
            c02 = _mm256_fmadd_ps(a0, k2, c02);
            c03 = _mm256_fmadd_ps(a0, k3, c03);
            __m256 a1 = _mm256_loadu_ps(A + lda * (i + 1) + l);
            c10 = _mm256_fmadd_ps(a1, k0, c10);
            c11 = _mm256_fmadd_ps(a1, k1, c11);
            c12 = _mm256_fmadd_ps(a1, k2, c12);
            c13 = _mm256_fmadd_ps(a1, k3, c13);
            __m256 a2 = _mm256_loadu_ps(A + lda * (i + 2) + l);
            c20 = _mm256_fmadd_ps(a2, k0, c20);
            c21 = _mm256_fmadd_ps(a2, k1, c21);
            c22 = _mm256_fmadd_ps(a2, k2, c22);
            c23 = _mm256_fmadd_ps(a2, k3, c23);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum_float_8(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum_float_8(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum_float_8(c20);
        C[ldc * (j + 1) + (i + 0)] = hsum_float_8(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum_float_8(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum_float_8(c21);
        C[ldc * (j + 2) + (i + 0)] = hsum_float_8(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum_float_8(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum_float_8(c22);
        C[ldc * (j + 3) + (i + 0)] = hsum_float_8(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum_float_8(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum_float_8(c23);
        END_KERNEL()
    }

    dontinline void gemm1x4(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 4)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            __m256 a0 = _mm256_loadu_ps(A + lda * (i + 0) + l);
            __m256 k0 = _mm256_loadu_ps(B + ldb * (j + 0) + l);
            __m256 k1 = _mm256_loadu_ps(B + ldb * (j + 1) + l);
            __m256 k2 = _mm256_loadu_ps(B + ldb * (j + 2) + l);
            __m256 k3 = _mm256_loadu_ps(B + ldb * (j + 3) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            c01 = _mm256_fmadd_ps(a0, k1, c01);
            c02 = _mm256_fmadd_ps(a0, k2, c02);
            c03 = _mm256_fmadd_ps(a0, k3, c03);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum_float_8(c00);
        C[ldc * (j + 1) + (i + 0)] = hsum_float_8(c01);
        C[ldc * (j + 2) + (i + 0)] = hsum_float_8(c02);
        C[ldc * (j + 3) + (i + 0)] = hsum_float_8(c03);
        END_KERNEL()
    }

    dontinline void gemm4x1(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(4, 1)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            __m256 k0 = _mm256_loadu_ps(B + ldb * (j + 0) + l);
            __m256 a0 = _mm256_loadu_ps(A + lda * (i + 0) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            __m256 a1 = _mm256_loadu_ps(A + lda * (i + 1) + l);
            c10 = _mm256_fmadd_ps(a1, k0, c10);
            __m256 a2 = _mm256_loadu_ps(A + lda * (i + 2) + l);
            c20 = _mm256_fmadd_ps(a2, k0, c20);
            __m256 a3 = _mm256_loadu_ps(A + lda * (i + 3) + l);
            c30 = _mm256_fmadd_ps(a3, k0, c30);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum_float_8(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum_float_8(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum_float_8(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum_float_8(c30);
        END_KERNEL()
    }

    dontinline void gemm1x1(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 1)
        __m256 c = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            c = _mm256_fmadd_ps(_mm256_loadu_ps(A + lda * i + l),
                                _mm256_loadu_ps(B + ldb * j + l), c);
        }
        C[ldc * j + i] = hsum_float_8(c);
        END_KERNEL()
    }

    const long k;
    const float *const A;
    const long lda;
    const float *const B;
    const long ldb;
    float *const C;
    const long ldc;
    const long ith;
    const long nth;
};

class tinyBLAS_F16_AVX2 {
  public:
    tinyBLAS_F16_AVX2(long k, const ggml_fp16_t *A, long lda,
                      const ggml_fp16_t *B, long ldb, float *C, long ldc,
                      long ith, long nth)
        : k(k), A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith),
          nth(nth) {
        ASSERT(A != nullptr);
        ASSERT(B != nullptr);
        ASSERT(C != nullptr);
        ASSERT(k >= 0 && k % 16 == 0);
        ASSERT(ith >= 0 && ith < nth);
    }

    void gemm(long m, long n) {
        ASSERT(m >= 0);
        ASSERT(n >= 0);
        ASSERT(ldc >= m);
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(long m0, long m, long n0, long n) {
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        long mc, nc, mp, np;
        if (m - m0 >= 3 && n - n0 >= 4) {
            mc = 3;
            nc = 4;
            gemm3x4(m0, m, n0, n);
        } else if (m - m0 >= 4 && n - n0 >= 1) {
            mc = 4;
            nc = 1;
            gemm4x1(m0, m, n0, n);
        } else if (m - m0 >= 1 && n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm1x4(m0, m, n0, n);
        } else {
            mc = 1;
            nc = 1;
            gemm1x1(m0, m, n0, n);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, mp, np, n);
        mnpack(mp, m, np, n);
    }

    dontinline void gemm3x4(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(3, 4)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c11 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps();
        __m256 c13 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c21 = _mm256_setzero_ps();
        __m256 c22 = _mm256_setzero_ps();
        __m256 c23 = _mm256_setzero_ps();
        const ggml_fp16_t *Ap0 = A + lda * (i + 0);
        const ggml_fp16_t *Ap1 = A + lda * (i + 1);
        const ggml_fp16_t *Ap2 = A + lda * (i + 2);
        const ggml_fp16_t *Bp0 = B + ldb * (j + 0);
        const ggml_fp16_t *Bp1 = B + ldb * (j + 1);
        const ggml_fp16_t *Bp2 = B + ldb * (j + 2);
        const ggml_fp16_t *Bp3 = B + ldb * (j + 3);
        for (long l = 0; l < k; l += 8) {
            __m256 k0 = loadf16x8(Bp0 + l);
            __m256 k1 = loadf16x8(Bp1 + l);
            __m256 k2 = loadf16x8(Bp2 + l);
            __m256 k3 = loadf16x8(Bp3 + l);
            __m256 a0 = loadf16x8(Ap0 + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            c01 = _mm256_fmadd_ps(a0, k1, c01);
            c02 = _mm256_fmadd_ps(a0, k2, c02);
            c03 = _mm256_fmadd_ps(a0, k3, c03);
            __m256 a1 = loadf16x8(Ap1 + l);
            c10 = _mm256_fmadd_ps(a1, k0, c10);
            c11 = _mm256_fmadd_ps(a1, k1, c11);
            c12 = _mm256_fmadd_ps(a1, k2, c12);
            c13 = _mm256_fmadd_ps(a1, k3, c13);
            __m256 a2 = loadf16x8(Ap2 + l);
            c20 = _mm256_fmadd_ps(a2, k0, c20);
            c21 = _mm256_fmadd_ps(a2, k1, c21);
            c22 = _mm256_fmadd_ps(a2, k2, c22);
            c23 = _mm256_fmadd_ps(a2, k3, c23);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum_float_8(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum_float_8(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum_float_8(c20);
        C[ldc * (j + 1) + (i + 0)] = hsum_float_8(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum_float_8(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum_float_8(c21);
        C[ldc * (j + 2) + (i + 0)] = hsum_float_8(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum_float_8(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum_float_8(c22);
        C[ldc * (j + 3) + (i + 0)] = hsum_float_8(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum_float_8(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum_float_8(c23);
        END_KERNEL()
    }

    dontinline void gemm1x4(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 4)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            __m256 a0 = loadf16x8(A + lda * (i + 0) + l);
            __m256 k0 = loadf16x8(B + ldb * (j + 0) + l);
            __m256 k1 = loadf16x8(B + ldb * (j + 1) + l);
            __m256 k2 = loadf16x8(B + ldb * (j + 2) + l);
            __m256 k3 = loadf16x8(B + ldb * (j + 3) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            c01 = _mm256_fmadd_ps(a0, k1, c01);
            c02 = _mm256_fmadd_ps(a0, k2, c02);
            c03 = _mm256_fmadd_ps(a0, k3, c03);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum_float_8(c00);
        C[ldc * (j + 1) + (i + 0)] = hsum_float_8(c01);
        C[ldc * (j + 2) + (i + 0)] = hsum_float_8(c02);
        C[ldc * (j + 3) + (i + 0)] = hsum_float_8(c03);
        END_KERNEL()
    }

    dontinline void gemm4x1(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(4, 1)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            __m256 k0 = loadf16x8(B + ldb * (j + 0) + l);
            __m256 a0 = loadf16x8(A + lda * (i + 0) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            __m256 a1 = loadf16x8(A + lda * (i + 1) + l);
            c10 = _mm256_fmadd_ps(a1, k0, c10);
            __m256 a2 = loadf16x8(A + lda * (i + 2) + l);
            c20 = _mm256_fmadd_ps(a2, k0, c20);
            __m256 a3 = loadf16x8(A + lda * (i + 3) + l);
            c30 = _mm256_fmadd_ps(a3, k0, c30);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum_float_8(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum_float_8(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum_float_8(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum_float_8(c30);
        END_KERNEL()
    }

    dontinline void gemm1x1(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 1)
        __m256 c = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            c = _mm256_fmadd_ps(loadf16x8(A + lda * i + l),
                                loadf16x8(B + ldb * j + l), c);
        }
        C[ldc * j + i] = hsum_float_8(c);
        END_KERNEL()
    }

    const long k;
    const ggml_fp16_t *const A;
    const long lda;
    const ggml_fp16_t *const B;
    const long ldb;
    float *const C;
    const long ldc;
    const long ith;
    const long nth;
};

class tinyBLAS_Q8_VNNI {
  public:
    tinyBLAS_Q8_VNNI(long k, const block_q8_0 *A, long lda, const block_q8_0 *B,
                     long ldb, float *C, long ldc, long ith, long nth)
        : k(k), A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith),
          nth(nth) {
        ASSERT(A != nullptr);
        ASSERT(B != nullptr);
        ASSERT(C != nullptr);
        ASSERT(k >= 0 && k % 32 == 0);
        ASSERT(ith >= 0 && ith < nth);
    }

    void gemm(long m, long n) {
        ASSERT(m >= 0);
        ASSERT(n >= 0);
        ASSERT(ldc >= m);
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(long m0, long m, long n0, long n) {
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        long mc, nc, mp, np;
        if (m - m0 >= 4 && n - n0 >= 1) {
            mc = 4;
            nc = 1;
            gemm4x1(m0, m, n0, n);
        } else if (m - m0 >= 1 && n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm1x4(m0, m, n0, n);
        } else {
            mc = 1;
            nc = 1;
            gemm1x1(m0, m, n0, n);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, mp, np, n);
        mnpack(mp, m, np, n);
    }

    dontinline void gemm4x1(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(4, 1)
        long k2 = k / 32;
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        const block_q8_0 *Ap0 = A + lda * (i + 0);
        const block_q8_0 *Ap1 = A + lda * (i + 1);
        const block_q8_0 *Ap2 = A + lda * (i + 2);
        const block_q8_0 *Ap3 = A + lda * (i + 3);
        const block_q8_0 *Bp = B + ldb * j;
        for (long l = 0; l < k2; ++l) {
            float db0 = GGML_FP16_TO_FP32(Bp[l].d);
            __m256i f = _mm256_loadu_si256((const __m256i *)Bp[l].qs);
            __m256i u = _mm256_sign_epi8(f, f);
            __m256 d0 = _mm256_set1_ps(GGML_FP16_TO_FP32(Ap0[l].d) * db0);
            __m256 d1 = _mm256_set1_ps(GGML_FP16_TO_FP32(Ap1[l].d) * db0);
            __m256 d2 = _mm256_set1_ps(GGML_FP16_TO_FP32(Ap2[l].d) * db0);
            __m256 d3 = _mm256_set1_ps(GGML_FP16_TO_FP32(Ap3[l].d) * db0);
            __m256i e0 = _mm256_loadu_si256((const __m256i *)Ap0[l].qs);
            __m256i e1 = _mm256_loadu_si256((const __m256i *)Ap1[l].qs);
            __m256i e2 = _mm256_loadu_si256((const __m256i *)Ap2[l].qs);
            __m256i e3 = _mm256_loadu_si256((const __m256i *)Ap3[l].qs);
            __m256i s0 = _mm256_sign_epi8(e0, f);
            __m256i s1 = _mm256_sign_epi8(e1, f);
            __m256i s2 = _mm256_sign_epi8(e2, f);
            __m256i s3 = _mm256_sign_epi8(e3, f);
            __m256 g0 = mul_sum_signed_i8_pairs_float(u, s0);
            __m256 g1 = mul_sum_signed_i8_pairs_float(u, s1);
            __m256 g2 = mul_sum_signed_i8_pairs_float(u, s2);
            __m256 g3 = mul_sum_signed_i8_pairs_float(u, s3);
            c0 = _mm256_fmadd_ps(d0, g0, c0);
            c1 = _mm256_fmadd_ps(d1, g1, c1);
            c2 = _mm256_fmadd_ps(d2, g2, c2);
            c3 = _mm256_fmadd_ps(d3, g3, c3);
        }
        C[ldc * j + (i + 0)] = hsum_float_8(c0);
        C[ldc * j + (i + 1)] = hsum_float_8(c1);
        C[ldc * j + (i + 2)] = hsum_float_8(c2);
        C[ldc * j + (i + 3)] = hsum_float_8(c3);
        END_KERNEL()
    }

    dontinline void gemm1x4(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 4)
        long k2 = k / 32;
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        const block_q8_0 *Bp0 = B + ldb * (j + 0);
        const block_q8_0 *Bp1 = B + ldb * (j + 1);
        const block_q8_0 *Bp2 = B + ldb * (j + 2);
        const block_q8_0 *Bp3 = B + ldb * (j + 3);
        const block_q8_0 *Ap = A + lda * i;
        for (long l = 0; l < k2; ++l) {
            float da0 = GGML_FP16_TO_FP32(Ap[l].d);
            __m256i f = _mm256_loadu_si256((const __m256i *)Ap[l].qs);
            __m256i u = _mm256_sign_epi8(f, f);
            __m256 d0 = _mm256_set1_ps(GGML_FP16_TO_FP32(Bp0[l].d) * da0);
            __m256 d1 = _mm256_set1_ps(GGML_FP16_TO_FP32(Bp1[l].d) * da0);
            __m256 d2 = _mm256_set1_ps(GGML_FP16_TO_FP32(Bp2[l].d) * da0);
            __m256 d3 = _mm256_set1_ps(GGML_FP16_TO_FP32(Bp3[l].d) * da0);
            __m256i e0 = _mm256_loadu_si256((const __m256i *)Bp0[l].qs);
            __m256i e1 = _mm256_loadu_si256((const __m256i *)Bp1[l].qs);
            __m256i e2 = _mm256_loadu_si256((const __m256i *)Bp2[l].qs);
            __m256i e3 = _mm256_loadu_si256((const __m256i *)Bp3[l].qs);
            __m256i s0 = _mm256_sign_epi8(e0, f);
            __m256i s1 = _mm256_sign_epi8(e1, f);
            __m256i s2 = _mm256_sign_epi8(e2, f);
            __m256i s3 = _mm256_sign_epi8(e3, f);
            __m256 g0 = mul_sum_signed_i8_pairs_float(u, s0);
            __m256 g1 = mul_sum_signed_i8_pairs_float(u, s1);
            __m256 g2 = mul_sum_signed_i8_pairs_float(u, s2);
            __m256 g3 = mul_sum_signed_i8_pairs_float(u, s3);
            c0 = _mm256_fmadd_ps(d0, g0, c0);
            c1 = _mm256_fmadd_ps(d1, g1, c1);
            c2 = _mm256_fmadd_ps(d2, g2, c2);
            c3 = _mm256_fmadd_ps(d3, g3, c3);
        }
        C[ldc * (j + 0) + i] = hsum_float_8(c0);
        C[ldc * (j + 1) + i] = hsum_float_8(c1);
        C[ldc * (j + 2) + i] = hsum_float_8(c2);
        C[ldc * (j + 3) + i] = hsum_float_8(c3);
        END_KERNEL()
    }

    dontinline void gemm1x1(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 1)
        long k2 = k / 32;
        __m256 c0 = _mm256_setzero_ps();
        const block_q8_0 *Ap = A + lda * i;
        const block_q8_0 *Bp = B + ldb * j;
        for (long l = 0; l < k2; ++l) {
            __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(Ap[l].d) *
                                      GGML_FP16_TO_FP32(Bp[l].d));
            __m256i e = _mm256_loadu_si256((const __m256i *)Ap[l].qs);
            __m256i f = _mm256_loadu_si256((const __m256i *)Bp[l].qs);
            __m256 g = mul_sum_i8_pairs_float(e, f);
            c0 = _mm256_fmadd_ps(d, g, c0);
        }
        C[ldc * j + i] = hsum_float_8(c0);
        END_KERNEL()
    }

    const long k;
    const block_q8_0 *const A;
    const long lda;
    const block_q8_0 *const B;
    const long ldb;
    float *const C;
    const long ldc;
    const long ith;
    const long nth;
};

#endif // __x86_64__

// computes m×k * n×k → n×m
void llamafile_sgemm(long m, long n, long k, int dtype, const void *A, long lda,
                     const void *B, long ldb, float *C, long ldc, long ith,
                     long nth) {
    switch (dtype) {
#ifdef __x86_64__
    case GGML_TYPE_F32: {
        tinyBLAS_F32_AVX2 tb{
            k, (const float *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
        tb.gemm(m, n);
        break;
    }
    case GGML_TYPE_F16: {
        tinyBLAS_F16_AVX2 tb{k,   (const ggml_fp16_t *)A,
                             lda, (const ggml_fp16_t *)B,
                             ldb, C,
                             ldc, ith,
                             nth};
        tb.gemm(m, n);
        break;
    }
    case GGML_TYPE_Q8_0: {
        tinyBLAS_Q8_VNNI tb{k,   (const block_q8_0 *)A,
                            lda, (const block_q8_0 *)B,
                            ldb, C,
                            ldc, ith,
                            nth};
        tb.gemm(m, n);
        break;
    }
#endif
    default:
        ASSERT(!"unsupported dtype");
    }
}
