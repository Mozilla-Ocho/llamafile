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

class SGEMMER1 {
  public:
    SGEMMER1(int k, const TA *A, int lda, const TB *B, int ldb, TC *C, int ldc, int ith, int nth)
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
        if (VECTOR_REGISTERS >= 32 && m - m0 >= 4 && n - n0 >= 2) {
            mc = 4;
            nc = 2;
            gemm4x2(m0, m, n0, n);
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

    dontinline void gemm4x2(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 2)
        __m128 s0 = _mm_setzero_ps();
        __m128 s1 = _mm_setzero_ps();
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c11 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps();
        __m256 c13 = _mm256_setzero_ps();
        const TA *Ap0 = A + lda * (i + 0);
        const TA *Ap1 = A + lda * (i + 1);
        const TA *Ap2 = A + lda * (i + 2);
        const TA *Ap3 = A + lda * (i + 3);
        const TB *Bp0 = B + ldb * (j + 0);
        const TB *Bp1 = B + ldb * (j + 1);
        for (int l = 0; l < k; ++l) {
            __m128 sa = _mm_set_ps(unhalf(Ap3[l].dm >> 16), //
                                   unhalf(Ap2[l].dm >> 16), //
                                   unhalf(Ap1[l].dm >> 16), //
                                   unhalf(Ap0[l].dm >> 16));
            s0 = madd(_mm_set1_ps(unhalf(Bp0[l].ds >> 16)), sa, s0);
            s1 = madd(_mm_set1_ps(unhalf(Bp1[l].ds >> 16)), sa, s1);
            __m256i f0 = load(Bp0 + l);
            __m256 d0 = _mm256_set1_ps(unhalf(Bp0[l].ds));
            c00 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap0[l].dm)), d0),
                       updot(load(Ap0 + l), f0), c00);
            c01 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap1[l].dm)), d0),
                       updot(load(Ap1 + l), f0), c01);
            c02 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap2[l].dm)), d0),
                       updot(load(Ap2 + l), f0), c02);
            c03 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap3[l].dm)), d0),
                       updot(load(Ap3 + l), f0), c03);
            __m256i f1 = load(Bp1 + l);
            __m256 d1 = _mm256_set1_ps(unhalf(Bp1[l].ds));
            c10 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap0[l].dm)), d1),
                       updot(load(Ap0 + l), f1), c10);
            c11 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap1[l].dm)), d1),
                       updot(load(Ap1 + l), f1), c11);
            c12 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap2[l].dm)), d1),
                       updot(load(Ap2 + l), f1), c12);
            c13 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap3[l].dm)), d1),
                       updot(load(Ap3 + l), f1), c13);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00) + s0[0];
        C[ldc * (j + 0) + (i + 1)] = hsum(c01) + s0[1];
        C[ldc * (j + 0) + (i + 2)] = hsum(c02) + s0[2];
        C[ldc * (j + 0) + (i + 3)] = hsum(c03) + s0[3];
        C[ldc * (j + 1) + (i + 0)] = hsum(c10) + s1[0];
        C[ldc * (j + 1) + (i + 1)] = hsum(c11) + s1[1];
        C[ldc * (j + 1) + (i + 2)] = hsum(c12) + s1[2];
        C[ldc * (j + 1) + (i + 3)] = hsum(c13) + s1[3];
        END_KERNEL()
    }

    dontinline void gemm4x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 1)
        __m128 s = _mm_setzero_ps();
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
            s = madd(_mm_set1_ps(unhalf(Bp[l].ds >> 16)),
                     _mm_set_ps(unhalf(Ap3[l].dm >> 16), //
                                unhalf(Ap2[l].dm >> 16), //
                                unhalf(Ap1[l].dm >> 16), //
                                unhalf(Ap0[l].dm >> 16)),
                     s);
            __m256i f = load(Bp + l);
            __m256 d1 = _mm256_set1_ps(unhalf(Bp[l].ds));
            c0 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap0[l].dm)), d1), updot(load(Ap0 + l), f),
                      c0);
            c1 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap1[l].dm)), d1), updot(load(Ap1 + l), f),
                      c1);
            c2 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap2[l].dm)), d1), updot(load(Ap2 + l), f),
                      c2);
            c3 = madd(_mm256_mul_ps(_mm256_set1_ps(unhalf(Ap3[l].dm)), d1), updot(load(Ap3 + l), f),
                      c3);
        }
        C[ldc * j + (i + 0)] = hsum(c0) + s[0];
        C[ldc * j + (i + 1)] = hsum(c1) + s[1];
        C[ldc * j + (i + 2)] = hsum(c2) + s[2];
        C[ldc * j + (i + 3)] = hsum(c3) + s[3];
        END_KERNEL()
    }

    dontinline void gemm1x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 1)
        float s = 0;
        __m256 c = _mm256_setzero_ps();
        const TA *Ap = A + lda * i;
        const TB *Bp = B + ldb * j;
        for (int l = 0; l < k; ++l) {
            unsigned dm = Ap[l].dm;
            unsigned ds = Bp[l].ds;
            __m256 d0 = _mm256_set1_ps(unhalf(dm));
            __m256 d1 = _mm256_set1_ps(unhalf(ds));
            s += unhalf(dm >> 16) * unhalf(ds >> 16);
            __m256i e = load(Ap + l);
            __m256i f = load(Bp + l);
            c = madd(_mm256_mul_ps(d0, d1), updot(e, f), c);
        }
        C[ldc * j + i] = hsum(c) + s;
        END_KERNEL()
    }

    static inline nosideeffect __m256i load(const block_q8_1 *b) {
        return _mm256_loadu_si256((const __m256i *)b->qs);
    }

    static inline nosideeffect __m256i load(const block_q4_1 *b) {
        return denibble(b->qs);
    }

    static inline pureconst __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#else
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
        return _mm256_cvtepi32_ps(res);
    }

    static inline nosideeffect __m256i denibble(const uint8_t *p) {
        const __m128i tmp = _mm_loadu_si128((const __m128i *)p);
        const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
        const __m256i lowMask = _mm256_set1_epi8(15);
        return _mm256_and_si256(lowMask, bytes);
    }

    static inline pureconst float unhalf(unsigned short d) {
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
