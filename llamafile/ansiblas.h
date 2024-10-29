// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#pragma once

#include <cmath>
#include <unistd.h>

#include "llama.cpp/cores.h"

namespace {
namespace ansiBLAS {

static constexpr int KN = 8;

union Vector {
    double v[KN];
};

inline Vector load(const float *p) {
    Vector x;
    for (int i = 0; i < KN; ++i) {
        x.v[i] = p[i];
    }
    return x;
}

inline Vector madd(Vector x, Vector y, Vector s) {
    for (int i = 0; i < KN; ++i) {
        s.v[i] = fma(x.v[i], y.v[i], s.v[i]);
    }
    return s;
}

inline float hsum(Vector x) {
    double s = 0;
    for (int i = 0; i < KN; ++i) {
        s += x.v[i];
    }
    return s;
}

struct ansiBLAS {
  public:
    ansiBLAS(int k, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int ith,
             int nth)
        : k(k), A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n) {
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(int m0, int m, int n0, int n) {
        int mc, nc, mp, np;
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        if (m - m0 >= 4 && n - n0 >= 3) {
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
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    void gemm(int m0, int m, int n0, int n) {
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
            Vector Cv[RN][RM] = {0};
            Vector Ce[RN][RM] = {0};
            for (int l = 0; l < k; l += KN)
                for (int j = 0; j < RN; ++j)
                    for (int i = 0; i < RM; ++i)
                        Cv[j][i] = madd(load(A + lda * (ii + i) + l), //
                                        load(B + ldb * (jj + j) + l), //
                                        Cv[j][i]);
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    const int k;
    const float *const A;
    const int lda;
    const float *const B;
    const int ldb;
    float *const C;
    const int ldc;
    const int ith;
    const int nth;
};

void sgemm(int m, int n, int k, //
           const float *A, int lda, //
           const float *B, int ldb, //
           float *C, int ldc) {
    static int nth = cpu_get_num_math();
#pragma omp parallel for
    for (int ith = 0; ith < nth; ++ith) {
        ansiBLAS tb{k, A, lda, B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
    }
}

} // namespace ansiBLAS
} // namespace
