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

#include <algorithm>
#include <cassert>

#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml.h"

#include "hsum.h"
#include "kernel.h"
#include "madd.h"

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

namespace {

class LLMMQ0 {
  public:
    LLMMQ0(int k, const TA *A, int lda, const TB *B, int ldb, TC *C, int ldc, int ith, int nth)
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

    dontinline void gemm4x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 1)
        int k2 = k / 32;
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        const TA *Ap0 = A + lda * (i + 0);
        const TA *Ap1 = A + lda * (i + 1);
        const TA *Ap2 = A + lda * (i + 2);
        const TA *Ap3 = A + lda * (i + 3);
        const TB *Bp = B + ldb * j;
        for (int l = 0; l < k2; ++l) {
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
        int k2 = k / 32;
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        const TB *Bp0 = B + ldb * (j + 0);
        const TB *Bp1 = B + ldb * (j + 1);
        const TB *Bp2 = B + ldb * (j + 2);
        const TB *Bp3 = B + ldb * (j + 3);
        const TA *Ap = A + lda * i;
        for (int l = 0; l < k2; ++l) {
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
        int k2 = k / 32;
        __m256 c = _mm256_setzero_ps();
        const TA *Ap = A + lda * i;
        const TB *Bp = B + ldb * j;
        for (int l = 0; l < k2; ++l) {
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
