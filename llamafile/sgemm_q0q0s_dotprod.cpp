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

#ifdef __aarch64__

#include <arm_neon.h>

#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml.h"

#include "hsum.h"
#include "kernel.h"
#include "sgemm.h"

namespace {

class GEMMERQ0ARM {
  public:
    GEMMERQ0ARM(int k, const block_q8_0 *A, int lda, const block_q8_0 *B, int ldb, float *C,
                int ldc, int ith, int nth)
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
        if (m - m0 >= 3 && n - n0 >= 3) {
            mc = 3;
            nc = 3;
            gemm3x3(m0, m, n0, n);
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

    dontinline void gemm3x3(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(3, 3)
        int32x4_t zero = vdupq_n_s32(0);
        float32x4_t c00 = vdupq_n_f32(0.f);
        float32x4_t c01 = vdupq_n_f32(0.f);
        float32x4_t c02 = vdupq_n_f32(0.f);
        float32x4_t c10 = vdupq_n_f32(0.f);
        float32x4_t c11 = vdupq_n_f32(0.f);
        float32x4_t c12 = vdupq_n_f32(0.f);
        float32x4_t c20 = vdupq_n_f32(0.f);
        float32x4_t c21 = vdupq_n_f32(0.f);
        float32x4_t c22 = vdupq_n_f32(0.f);
        const block_q8_0 *Ap0 = A + lda * (i + 0);
        const block_q8_0 *Ap1 = A + lda * (i + 1);
        const block_q8_0 *Ap2 = A + lda * (i + 2);
        const block_q8_0 *Bp0 = B + ldb * (j + 0);
        const block_q8_0 *Bp1 = B + ldb * (j + 1);
        const block_q8_0 *Bp2 = B + ldb * (j + 2);
        for (int l = 0; l < k; ++l) {
            c00 = vmlaq_n_f32(
                c00,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(Ap0[l].qs), vld1q_s8(Bp0[l].qs)),
                                        vld1q_s8(Ap0[l].qs + 16), vld1q_s8(Bp0[l].qs + 16))),
                unhalf(Ap0[l].d) * unhalf(Bp0[l].d));
            c01 = vmlaq_n_f32(
                c01,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(Ap0[l].qs), vld1q_s8(Bp1[l].qs)),
                                        vld1q_s8(Ap0[l].qs + 16), vld1q_s8(Bp1[l].qs + 16))),
                unhalf(Ap0[l].d) * unhalf(Bp1[l].d));
            c02 = vmlaq_n_f32(
                c02,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(Ap0[l].qs), vld1q_s8(Bp2[l].qs)),
                                        vld1q_s8(Ap0[l].qs + 16), vld1q_s8(Bp2[l].qs + 16))),
                unhalf(Ap0[l].d) * unhalf(Bp2[l].d));
            c10 = vmlaq_n_f32(
                c10,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(Ap1[l].qs), vld1q_s8(Bp0[l].qs)),
                                        vld1q_s8(Ap1[l].qs + 16), vld1q_s8(Bp0[l].qs + 16))),
                unhalf(Ap1[l].d) * unhalf(Bp0[l].d));
            c11 = vmlaq_n_f32(
                c11,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(Ap1[l].qs), vld1q_s8(Bp1[l].qs)),
                                        vld1q_s8(Ap1[l].qs + 16), vld1q_s8(Bp1[l].qs + 16))),
                unhalf(Ap1[l].d) * unhalf(Bp1[l].d));
            c12 = vmlaq_n_f32(
                c12,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(Ap1[l].qs), vld1q_s8(Bp2[l].qs)),
                                        vld1q_s8(Ap1[l].qs + 16), vld1q_s8(Bp2[l].qs + 16))),
                unhalf(Ap1[l].d) * unhalf(Bp2[l].d));
            c20 = vmlaq_n_f32(
                c20,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(Ap2[l].qs), vld1q_s8(Bp0[l].qs)),
                                        vld1q_s8(Ap2[l].qs + 16), vld1q_s8(Bp0[l].qs + 16))),
                unhalf(Ap2[l].d) * unhalf(Bp0[l].d));
            c21 = vmlaq_n_f32(
                c21,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(Ap2[l].qs), vld1q_s8(Bp1[l].qs)),
                                        vld1q_s8(Ap2[l].qs + 16), vld1q_s8(Bp1[l].qs + 16))),
                unhalf(Ap2[l].d) * unhalf(Bp1[l].d));
            c22 = vmlaq_n_f32(
                c22,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, vld1q_s8(Ap2[l].qs), vld1q_s8(Bp2[l].qs)),
                                        vld1q_s8(Ap2[l].qs + 16), vld1q_s8(Bp2[l].qs + 16))),
                unhalf(Ap2[l].d) * unhalf(Bp2[l].d));
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        END_KERNEL()
    }

    dontinline void gemm1x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 1)
        float32x4_t acc = vdupq_n_f32(0.f);
        const block_q8_0 *Ap = A + lda * i;
        const block_q8_0 *Bp = B + ldb * j;
        for (int l = 0; l < k; ++l) {
            acc = vmlaq_n_f32(acc,
                              vcvtq_f32_s32(vdotq_s32(
                                  vdotq_s32(vdupq_n_s32(0), vld1q_s8(Ap[l].qs), vld1q_s8(Bp[l].qs)),
                                  vld1q_s8(Ap[l].qs + 16), vld1q_s8(Bp[l].qs + 16))),
                              unhalf(Ap[l].d) * unhalf(Bp[l].d));
        }
        C[ldc * j + i] = hsum(acc);
        END_KERNEL()
    }

    static inline float unhalf(unsigned short d) {
        return GGML_FP16_TO_FP32(d);
    }

    const block_q8_0 *const A;
    const block_q8_0 *const B;
    float *const C;
    const int k;
    const int lda;
    const int ldb;
    const int ldc;
    const int ith;
    const int nth;
};

} // namespace

bool llamafile_sgemm_q0q0s_dotprod(int m, int n, int k, const block_q8_0 *A, int lda,
                                   const block_q8_0 *B, int ldb, float *C, int ldc, int ith,
                                   int nth, int task) {
    if (task != GGML_TASK_TYPE_COMPUTE)
        return true;
    GEMMERQ0ARM tb{k, A, lda, B, ldb, C, ldc, ith, nth};
    tb.matmul(m, n);
    return true;
}

#endif // __aarch64__
