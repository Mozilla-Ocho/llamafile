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
#include <assert.h>
#include <cosmo.h>
#include <immintrin.h>
#include <stdio.h>

//
//                 _   _          ___ _      _   ___
//                | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                 \__|_|_||_\_, |___/____/_/ \_\___/
//                           |__/
//
//                  BASIC LINEAR ALGEBRA SUBPROGRAMS
//

// matrix multiplication for x86 microprocessors
//
// this file implements optimized cpu kernels for llama.cpp that support
// column major ordering with the A matrix transposed. they work by
// computing multiple dot products at once, because it exploits
// instruction level parallelism, in addition to minimizing register
// loads since the adjacent dot products can share register loads.
//
// this technique usually slightly outperforms mkl and blis at single
// threaded performance, and usually goes 2x faster than mkl when openmp
// threading is used. however please note that this code is adapted to
// use the llama.cpp synchronization model, which isn't as advanced.
//
//     ┌───────────┬──────────────────┬───────────────────┐
//     │ software  │ skinny matrices  │ fat matrices      │
//     │           │ token generation │ prompt evaluation │
//     │ ───────── │ ──────────────── │ ───────────────── │
//     │ llama.cpp │ fast             │ slow              │
//     │ MKL       │ slow             │ fast              │
//     │ tinyBLAS  │ fast             │ fast              │
//     └───────────┴──────────────────┴───────────────────┘
//
// these kernels do not have any of the latency overhead typically
// associated with the blas, which means they can provide the same
// performance as llama.cpp when performing the most common type of
// matmul used by text generation, which is matrix-vector product.

#define END_KERNEL() }
#define BEGIN_KERNEL(RM, RN) \
    long ytiles = (m - m0) / RM; \
    long xtiles = (n - n0) / RN; \
    long tiles = ytiles * xtiles; \
    double duty = (double)tiles / nth; \
    if (duty < 1) \
        duty = 1; \
    double spot = duty * ith + .5; \
    long end = spot + duty; \
    long start = spot; \
    if (end > tiles) \
        end = tiles; \
    for (long job = start; job < end; ++job) { \
        long i = m0 + job / xtiles * RM; \
        long j = n0 + job % xtiles * RN;

#ifdef __x86_64__
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c,avxvnni")

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

static inline float unhalf(ggml_fp16_t d) {
    return GGML_FP16_TO_FP32(d);
}

static dontinline float hsum8f(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i denibble32(const uint8_t *p) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)p);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8(15);
    return _mm256_and_si256(lowMask, bytes);
}

template <typename T> static inline __m256 load8f(const T *);
template <> inline __m256 load8f(const float *p) {
    return _mm256_loadu_ps(p);
}
template <> inline __m256 load8f(const ggml_fp16_t *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}

template <typename T> static inline __m256i load32i(const T *);
template <> inline __m256i load32i(const block_q8_0 *b) {
    return _mm256_loadu_si256((const __m256i *)b->qs);
}
template <> inline __m256i load32i(const block_q4_0 *b) {
    return _mm256_sub_epi8(denibble32(b->qs), _mm256_set1_epi8(8));
}

template <typename T> class tinyBLAS {
  public:
    tinyBLAS(long k, const T *A, long lda, const T *B, long ldb, float *C, long ldc, long ith,
             long nth)
        : k(k), A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith), nth(nth) {
        assert(A != nullptr);
        assert(B != nullptr);
        assert(C != nullptr);
        assert(k >= 0 && k % 8 == 0);
        assert(ith >= 0 && ith < nth);
    }

    void gemm(long m, long n) {
        assert(m >= 0);
        assert(n >= 0);
        assert(ldc >= m);
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
            __m256 k0 = load8f(B + ldb * (j + 0) + l);
            __m256 k1 = load8f(B + ldb * (j + 1) + l);
            __m256 k2 = load8f(B + ldb * (j + 2) + l);
            __m256 k3 = load8f(B + ldb * (j + 3) + l);
            __m256 a0 = load8f(A + lda * (i + 0) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            c01 = _mm256_fmadd_ps(a0, k1, c01);
            c02 = _mm256_fmadd_ps(a0, k2, c02);
            c03 = _mm256_fmadd_ps(a0, k3, c03);
            __m256 a1 = load8f(A + lda * (i + 1) + l);
            c10 = _mm256_fmadd_ps(a1, k0, c10);
            c11 = _mm256_fmadd_ps(a1, k1, c11);
            c12 = _mm256_fmadd_ps(a1, k2, c12);
            c13 = _mm256_fmadd_ps(a1, k3, c13);
            __m256 a2 = load8f(A + lda * (i + 2) + l);
            c20 = _mm256_fmadd_ps(a2, k0, c20);
            c21 = _mm256_fmadd_ps(a2, k1, c21);
            c22 = _mm256_fmadd_ps(a2, k2, c22);
            c23 = _mm256_fmadd_ps(a2, k3, c23);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum8f(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum8f(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum8f(c20);
        C[ldc * (j + 1) + (i + 0)] = hsum8f(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum8f(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum8f(c21);
        C[ldc * (j + 2) + (i + 0)] = hsum8f(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum8f(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum8f(c22);
        C[ldc * (j + 3) + (i + 0)] = hsum8f(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum8f(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum8f(c23);
        END_KERNEL()
    }

    dontinline void gemm1x4(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 4)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            __m256 a0 = load8f(A + lda * (i + 0) + l);
            __m256 k0 = load8f(B + ldb * (j + 0) + l);
            __m256 k1 = load8f(B + ldb * (j + 1) + l);
            __m256 k2 = load8f(B + ldb * (j + 2) + l);
            __m256 k3 = load8f(B + ldb * (j + 3) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            c01 = _mm256_fmadd_ps(a0, k1, c01);
            c02 = _mm256_fmadd_ps(a0, k2, c02);
            c03 = _mm256_fmadd_ps(a0, k3, c03);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum8f(c00);
        C[ldc * (j + 1) + (i + 0)] = hsum8f(c01);
        C[ldc * (j + 2) + (i + 0)] = hsum8f(c02);
        C[ldc * (j + 3) + (i + 0)] = hsum8f(c03);
        END_KERNEL()
    }

    dontinline void gemm4x1(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(4, 1)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            __m256 k0 = load8f(B + ldb * (j + 0) + l);
            __m256 a0 = load8f(A + lda * (i + 0) + l);
            c00 = _mm256_fmadd_ps(a0, k0, c00);
            __m256 a1 = load8f(A + lda * (i + 1) + l);
            c10 = _mm256_fmadd_ps(a1, k0, c10);
            __m256 a2 = load8f(A + lda * (i + 2) + l);
            c20 = _mm256_fmadd_ps(a2, k0, c20);
            __m256 a3 = load8f(A + lda * (i + 3) + l);
            c30 = _mm256_fmadd_ps(a3, k0, c30);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum8f(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum8f(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum8f(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum8f(c30);
        END_KERNEL()
    }

    dontinline void gemm1x1(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 1)
        __m256 c = _mm256_setzero_ps();
        for (long l = 0; l < k; l += 8) {
            c = _mm256_fmadd_ps(load8f(A + lda * i + l), load8f(B + ldb * j + l), c);
        }
        C[ldc * j + i] = hsum8f(c);
        END_KERNEL()
    }

    const long k;
    const T *const A;
    const long lda;
    const T *const B;
    const long ldb;
    float *const C;
    const long ldc;
    const long ith;
    const long nth;
};

template <typename T, typename U, int avxvnni> class tinyBLASq {
  public:
    tinyBLASq(long k, const T *A, long lda, const U *B, long ldb, float *C, long ldc, long ith,
              long nth)
        : k(k), A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith), nth(nth) {
        assert(A != nullptr);
        assert(B != nullptr);
        assert(C != nullptr);
        assert(k >= 0 && k % 32 == 0);
        assert(ith >= 0 && ith < nth);
    }

    void gemm(long m, long n) {
        assert(m >= 0);
        assert(n >= 0);
        assert(ldc >= m);
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(long m0, long m, long n0, long n) {
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        long mc, nc, mp, np;
        if (m - m0 >= 1 && n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm1x4(m0, m, n0, n);
        } else if (m - m0 >= 4 && n - n0 >= 1) {
            mc = 4;
            nc = 1;
            gemm4x1(m0, m, n0, n);
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
        const T *Ap0 = A + lda * (i + 0);
        const T *Ap1 = A + lda * (i + 1);
        const T *Ap2 = A + lda * (i + 2);
        const T *Ap3 = A + lda * (i + 3);
        const U *Bp = B + ldb * j;
        for (long l = 0; l < k2; ++l) {
            float db0 = unhalf(Bp[l].d);
            __m256i f = load32i(Bp + l);
            __m256i u = _mm256_sign_epi8(f, f);
            __m256 d0 = _mm256_set1_ps(unhalf(Ap0[l].d) * db0);
            __m256 d1 = _mm256_set1_ps(unhalf(Ap1[l].d) * db0);
            __m256 d2 = _mm256_set1_ps(unhalf(Ap2[l].d) * db0);
            __m256 d3 = _mm256_set1_ps(unhalf(Ap3[l].d) * db0);
            __m256i e0 = load32i(Ap0 + l);
            __m256i e1 = load32i(Ap1 + l);
            __m256i e2 = load32i(Ap2 + l);
            __m256i e3 = load32i(Ap3 + l);
            __m256i s0 = _mm256_sign_epi8(e0, f);
            __m256i s1 = _mm256_sign_epi8(e1, f);
            __m256i s2 = _mm256_sign_epi8(e2, f);
            __m256i s3 = _mm256_sign_epi8(e3, f);
            __m256 g0 = dotsome(u, s0);
            __m256 g1 = dotsome(u, s1);
            __m256 g2 = dotsome(u, s2);
            __m256 g3 = dotsome(u, s3);
            c0 = _mm256_fmadd_ps(d0, g0, c0);
            c1 = _mm256_fmadd_ps(d1, g1, c1);
            c2 = _mm256_fmadd_ps(d2, g2, c2);
            c3 = _mm256_fmadd_ps(d3, g3, c3);
        }
        C[ldc * j + (i + 0)] = hsum8f(c0);
        C[ldc * j + (i + 1)] = hsum8f(c1);
        C[ldc * j + (i + 2)] = hsum8f(c2);
        C[ldc * j + (i + 3)] = hsum8f(c3);
        END_KERNEL()
    }

    dontinline void gemm1x4(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 4)
        long k2 = k / 32;
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        const U *Bp0 = B + ldb * (j + 0);
        const U *Bp1 = B + ldb * (j + 1);
        const U *Bp2 = B + ldb * (j + 2);
        const U *Bp3 = B + ldb * (j + 3);
        const T *Ap = A + lda * i;
        for (long l = 0; l < k2; ++l) {
            float da0 = unhalf(Ap[l].d);
            __m256i f = load32i(Ap + l);
            __m256i u = _mm256_sign_epi8(f, f);
            __m256 d0 = _mm256_set1_ps(unhalf(Bp0[l].d) * da0);
            __m256 d1 = _mm256_set1_ps(unhalf(Bp1[l].d) * da0);
            __m256 d2 = _mm256_set1_ps(unhalf(Bp2[l].d) * da0);
            __m256 d3 = _mm256_set1_ps(unhalf(Bp3[l].d) * da0);
            __m256 g0 = dotsome(u, _mm256_sign_epi8(load32i(Bp0 + l), f));
            __m256 g1 = dotsome(u, _mm256_sign_epi8(load32i(Bp1 + l), f));
            __m256 g2 = dotsome(u, _mm256_sign_epi8(load32i(Bp2 + l), f));
            __m256 g3 = dotsome(u, _mm256_sign_epi8(load32i(Bp3 + l), f));
            c0 = _mm256_fmadd_ps(d0, g0, c0);
            c1 = _mm256_fmadd_ps(d1, g1, c1);
            c2 = _mm256_fmadd_ps(d2, g2, c2);
            c3 = _mm256_fmadd_ps(d3, g3, c3);
        }
        C[ldc * (j + 0) + i] = hsum8f(c0);
        C[ldc * (j + 1) + i] = hsum8f(c1);
        C[ldc * (j + 2) + i] = hsum8f(c2);
        C[ldc * (j + 3) + i] = hsum8f(c3);
        END_KERNEL()
    }

    dontinline void gemm1x1(long m0, long m, long n0, long n) {
        BEGIN_KERNEL(1, 1)
        long k2 = k / 32;
        __m256 c = _mm256_setzero_ps();
        const T *Ap = A + lda * i;
        const U *Bp = B + ldb * j;
        for (long l = 0; l < k2; ++l) {
            __m256 d = _mm256_set1_ps(unhalf(Ap[l].d) * unhalf(Bp[l].d));
            __m256i e = load32i(Ap + l);
            __m256i f = load32i(Bp + l);
            __m256 g = dotsome(_mm256_sign_epi8(e, e), _mm256_sign_epi8(f, e));
            c = _mm256_fmadd_ps(d, g, c);
        }
        C[ldc * j + i] = hsum8f(c);
        END_KERNEL()
    }

    inline __m256 dotsome(__m256i u, __m256i s) {
        __m256i res;
        if (avxvnni)
            res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
        else
            res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
        return _mm256_cvtepi32_ps(res);
    }

    const long k;
    const T *const A;
    const long lda;
    const U *const B;
    const long ldb;
    float *const C;
    const long ldc;
    const long ith;
    const long nth;
};

#endif // __x86_64__

// computes m×k * n×k → n×m
void llamafile_sgemm(long m, long n, long k, int dtype, const void *A, long lda, const void *B,
                     long ldb, float *C, long ldc, long ith, long nth) {
    assert(X86_HAVE(FMA));
    assert(X86_HAVE(AVX2));
    switch (dtype) {
#ifdef __x86_64__

    case GGML_TYPE_F32: {
        tinyBLAS<float> tb{k, (const float *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
        tb.gemm(m, n);
        break;
    }

    case GGML_TYPE_F16: {
        assert(X86_HAVE(F16C));
        tinyBLAS<ggml_fp16_t> tb{
            k, (const ggml_fp16_t *)A, lda, (const ggml_fp16_t *)B, ldb, C, ldc, ith, nth,
        };
        tb.gemm(m, n);
        break;
    }

    case GGML_TYPE_Q8_0:
        if (X86_HAVE(AVXVNNI)) {
            tinyBLASq<block_q8_0, block_q8_0, true> tb{
                k, (const block_q8_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth,
            };
            tb.gemm(m, n);
        } else {
            tinyBLASq<block_q8_0, block_q8_0, false> tb{
                k, (const block_q8_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth,
            };
            tb.gemm(m, n);
        }
        break;

    case GGML_TYPE_Q4_0:
        if (X86_HAVE(AVXVNNI)) {
            tinyBLASq<block_q4_0, block_q8_0, true> tb{
                k, (const block_q4_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth,
            };
            tb.gemm(m, n);
        } else {
            tinyBLASq<block_q4_0, block_q8_0, false> tb{
                k, (const block_q4_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth,
            };
            tb.gemm(m, n);
        }
        break;

#endif
    default:
        assert(!"unsupported dtype");
    }
}
