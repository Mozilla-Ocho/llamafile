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

#include "llama.cpp/ggml.h"

#include "hsum.h"
#include "kernel.h"
#include "varith.h"

namespace {

class SGEMMER {
  public:
    SGEMMER(int k, const TA *A, int lda, const TB *B, int ldb, TC *C, int ldc, int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n) {
        mnpack(0, m, 0, n);
    }

  private:
    dontinline void mnpack(int m0, int m, int n0, int n) {
        int mc, nc, mp, np;
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        if (VECTOR_REGISTERS >= 32 && m - m0 >= 8 && n - n0 >= 3) {
            mc = 8;
            nc = 3;
            gemm<8, 3>(m0, m, n0, n);
        } else if (m - m0 >= 4 && n - n0 >= 3) {
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n);
        } else if (n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n);
        } else if (m - m0 >= 4) {
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n);
        } else {
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, mp, np, n);
        mnpack(mp, m, np, n);
    }

    template <int RM, int RN> dontinline void gemm(int m0, int m, int n0, int n) {
        int ytiles = (m - m0) / RM;
        int xtiles = (n - n0) / RN;
        int tiles = xtiles * ytiles;
        int duty = (tiles + nth - 1) / nth;
        int start = duty * ith;
        int end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int job = start; job < end; ++job) {
            int ii = m0 + job / xtiles * RM;
            int jj = n0 + job % xtiles * RN;
            D Cv[RN][RM] = {0};
            for (int l = 0; l < k; l += KN)
                for (int j = 0; j < RN; ++j)
                    for (int i = 0; i < RM; ++i)
                        Cv[j][i] = madd(load(A + lda * (ii + i) + l), //
                                        load(B + ldb * (jj + j) + l), //
                                        Cv[j][i]);
            TC Cd[RN][RM];
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    Cd[j][i] = hsum(Cv[j][i]);
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = Cd[j][i];
        }
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
