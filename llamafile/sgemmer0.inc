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

#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml.h"

#include "hsum.h"
#include "kernel.h"
#include "varith.h"

namespace {

class SGEMMER0 {
  public:
    SGEMMER0(int k, const TA *A, int lda, const TB *B, int ldb, TC *C, int ldc, int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n) {
        mnpack(0, m, 0, n);
    }

  private:
    dontinline void mnpack(int m0, int m, int n0, int n) {
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        int mc, nc, mp, np;
        if (m - m0 >= 4 && n - n0 >= 3) {
            mc = 4;
            nc = 3;
            gemm4x3(m0, m, n0, n);
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

    dontinline void gemm4x3(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 3)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c11 = _mm256_setzero_ps();
        __m256 c21 = _mm256_setzero_ps();
        __m256 c31 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps();
        __m256 c22 = _mm256_setzero_ps();
        __m256 c32 = _mm256_setzero_ps();
        const TA *Ap0 = A + lda * (i + 0);
        const TA *Ap1 = A + lda * (i + 1);
        const TA *Ap2 = A + lda * (i + 2);
        const TA *Ap3 = A + lda * (i + 3);
        const TB *Bp0 = B + ldb * (j + 0);
        const TB *Bp1 = B + ldb * (j + 1);
        const TB *Bp2 = B + ldb * (j + 2);
        for (int l = 0; l < k; ++l) {
            float da0 = unhalf(Ap0[l].d);
            float da1 = unhalf(Ap1[l].d);
            float da2 = unhalf(Ap2[l].d);
            float da3 = unhalf(Ap3[l].d);
            __m256i e0 = load(Ap0 + l);
            __m256i e1 = load(Ap1 + l);
            __m256i e2 = load(Ap2 + l);
            __m256i e3 = load(Ap3 + l);
            float db0 = unhalf(Bp0[l].d);
            __m256 d00 = _mm256_set1_ps(da0 * db0);
            __m256 d10 = _mm256_set1_ps(da1 * db0);
            __m256 d20 = _mm256_set1_ps(da2 * db0);
            __m256 d30 = _mm256_set1_ps(da3 * db0);
            __m256i f0 = load(Bp0 + l);
            __m256i u0 = _mm256_sign_epi8(f0, f0);
            __m256i s00 = _mm256_sign_epi8(e0, f0);
            __m256i s10 = _mm256_sign_epi8(e1, f0);
            __m256i s20 = _mm256_sign_epi8(e2, f0);
            __m256i s30 = _mm256_sign_epi8(e3, f0);
            c00 = madd(d00, updot(u0, s00), c00);
            c10 = madd(d10, updot(u0, s10), c10);
            c20 = madd(d20, updot(u0, s20), c20);
            c30 = madd(d30, updot(u0, s30), c30);
            float db1 = unhalf(Bp1[l].d);
            __m256 d01 = _mm256_set1_ps(da0 * db1);
            __m256 d11 = _mm256_set1_ps(da1 * db1);
            __m256 d21 = _mm256_set1_ps(da2 * db1);
            __m256 d31 = _mm256_set1_ps(da3 * db1);
            __m256i f1 = load(Bp1 + l);
            __m256i u1 = _mm256_sign_epi8(f1, f1);
            __m256i s01 = _mm256_sign_epi8(e0, f1);
            __m256i s11 = _mm256_sign_epi8(e1, f1);
            __m256i s21 = _mm256_sign_epi8(e2, f1);
            __m256i s31 = _mm256_sign_epi8(e3, f1);
            c01 = madd(d01, updot(u1, s01), c01);
            c11 = madd(d11, updot(u1, s11), c11);
            c21 = madd(d21, updot(u1, s21), c21);
            c31 = madd(d31, updot(u1, s31), c31);
            float db2 = unhalf(Bp2[l].d);
            __m256 d02 = _mm256_set1_ps(da0 * db2);
            __m256 d12 = _mm256_set1_ps(da1 * db2);
            __m256 d22 = _mm256_set1_ps(da2 * db2);
            __m256 d32 = _mm256_set1_ps(da3 * db2);
            __m256i f2 = load(Bp2 + l);
            __m256i u2 = _mm256_sign_epi8(f2, f2);
            __m256i s02 = _mm256_sign_epi8(e0, f2);
            __m256i s12 = _mm256_sign_epi8(e1, f2);
            __m256i s22 = _mm256_sign_epi8(e2, f2);
            __m256i s32 = _mm256_sign_epi8(e3, f2);
            c02 = madd(d02, updot(u2, s02), c02);
            c12 = madd(d12, updot(u2, s12), c12);
            c22 = madd(d22, updot(u2, s22), c22);
            c32 = madd(d32, updot(u2, s32), c32);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 1) + (i + 3)] = hsum(c31);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        C[ldc * (j + 2) + (i + 3)] = hsum(c32);
        END_KERNEL()
    }

    dontinline void gemm4x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 1)
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        const TA *Ap0 = A + lda * (i + 0);
        const TA *Ap1 = A + lda * (i + 1);
        const TA *Ap2 = A + lda * (i + 2);
        const TA *Ap3 = A + lda * (i + 3);
        const TB *Bp = B + ldb * j;
        for (int l = 0; l < k; ++l) {
            float db0 = unhalf(Bp[l].d);
            __m256i f = load(Bp + l);
            __m256i u = _mm256_sign_epi8(f, f);
            __m256 d0 = _mm256_set1_ps(unhalf(Ap0[l].d) * db0);
            __m256 d1 = _mm256_set1_ps(unhalf(Ap1[l].d) * db0);
            __m256 d2 = _mm256_set1_ps(unhalf(Ap2[l].d) * db0);
            __m256 d3 = _mm256_set1_ps(unhalf(Ap3[l].d) * db0);
            __m256i e0 = load(Ap0 + l);
            __m256i e1 = load(Ap1 + l);
            __m256i e2 = load(Ap2 + l);
            __m256i e3 = load(Ap3 + l);
            __m256i s0 = _mm256_sign_epi8(e0, f);
            __m256i s1 = _mm256_sign_epi8(e1, f);
            __m256i s2 = _mm256_sign_epi8(e2, f);
            __m256i s3 = _mm256_sign_epi8(e3, f);
            __m256 g0 = updot(u, s0);
            __m256 g1 = updot(u, s1);
            __m256 g2 = updot(u, s2);
            __m256 g3 = updot(u, s3);
            c0 = madd(d0, g0, c0);
            c1 = madd(d1, g1, c1);
            c2 = madd(d2, g2, c2);
            c3 = madd(d3, g3, c3);
        }
        C[ldc * j + (i + 0)] = hsum(c0);
        C[ldc * j + (i + 1)] = hsum(c1);
        C[ldc * j + (i + 2)] = hsum(c2);
        C[ldc * j + (i + 3)] = hsum(c3);
        END_KERNEL()
    }

    dontinline void gemm1x4(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 4)
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        const TB *Bp0 = B + ldb * (j + 0);
        const TB *Bp1 = B + ldb * (j + 1);
        const TB *Bp2 = B + ldb * (j + 2);
        const TB *Bp3 = B + ldb * (j + 3);
        const TA *Ap = A + lda * i;
        for (int l = 0; l < k; ++l) {
            float da0 = unhalf(Ap[l].d);
            __m256i f = load(Ap + l);
            __m256i u = _mm256_sign_epi8(f, f);
            __m256 d0 = _mm256_set1_ps(unhalf(Bp0[l].d) * da0);
            __m256 d1 = _mm256_set1_ps(unhalf(Bp1[l].d) * da0);
            __m256 d2 = _mm256_set1_ps(unhalf(Bp2[l].d) * da0);
            __m256 d3 = _mm256_set1_ps(unhalf(Bp3[l].d) * da0);
            __m256 g0 = updot(u, _mm256_sign_epi8(load(Bp0 + l), f));
            __m256 g1 = updot(u, _mm256_sign_epi8(load(Bp1 + l), f));
            __m256 g2 = updot(u, _mm256_sign_epi8(load(Bp2 + l), f));
            __m256 g3 = updot(u, _mm256_sign_epi8(load(Bp3 + l), f));
            c0 = madd(d0, g0, c0);
            c1 = madd(d1, g1, c1);
            c2 = madd(d2, g2, c2);
            c3 = madd(d3, g3, c3);
        }
        C[ldc * (j + 0) + i] = hsum(c0);
        C[ldc * (j + 1) + i] = hsum(c1);
        C[ldc * (j + 2) + i] = hsum(c2);
        C[ldc * (j + 3) + i] = hsum(c3);
        END_KERNEL()
    }

    dontinline void gemm1x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 1)
        __m256 c = _mm256_setzero_ps();
        const TA *Ap = A + lda * i;
        const TB *Bp = B + ldb * j;
        for (int l = 0; l < k; ++l) {
            __m256 d = _mm256_set1_ps(unhalf(Ap[l].d) * unhalf(Bp[l].d));
            __m256i e = load(Ap + l);
            __m256i f = load(Bp + l);
            __m256 g = updot(_mm256_sign_epi8(e, e), _mm256_sign_epi8(f, e));
            c = madd(d, g, c);
        }
        C[ldc * j + i] = hsum(c);
        END_KERNEL()
    }

    inline __m256i load(const block_q8_0 *b) {
        return _mm256_loadu_si256((const __m256i *)b->qs);
    }

    inline __m256i load(const block_q4_0 *b) {
        return _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    inline __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#else
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
        return _mm256_cvtepi32_ps(res);
    }

    static inline __m256i denibble(const uint8_t *p) {
        const __m128i tmp = _mm_loadu_si128((const __m128i *)p);
        const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
        const __m256i lowMask = _mm256_set1_epi8(15);
        return _mm256_and_si256(lowMask, bytes);
    }

    static inline float unhalf(unsigned short d) {
        return GGML_FP16_TO_FP32(d);
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int k;
    const int lda;
    const int ldb;
    const int ldc;
    const int ith;
    const int nth;
};

} // namespace
