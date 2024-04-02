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
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        int mc, nc, mp, np;
        if (VECTOR_REGISTERS >= 32 && n - n0 >= 5 && m - m0 >= 5) {
            mc = 5;
            nc = 5;
            gemm5x5(m0, m, n0, n);
        } else if (n - n0 >= 4 && m - m0 >= 3) {
            mc = 3;
            nc = 4;
            gemm3x4(m0, m, n0, n);
        } else if (n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm1x4(m0, m, n0, n);
        } else if (m - m0 >= 4) {
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

    dontinline void gemm5x5(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(5, 5)
        D c00 = {0};
        D c01 = {0};
        D c02 = {0};
        D c03 = {0};
        D c04 = {0};
        D c10 = {0};
        D c11 = {0};
        D c12 = {0};
        D c13 = {0};
        D c14 = {0};
        D c20 = {0};
        D c21 = {0};
        D c22 = {0};
        D c23 = {0};
        D c24 = {0};
        D c30 = {0};
        D c31 = {0};
        D c32 = {0};
        D c33 = {0};
        D c34 = {0};
        D c40 = {0};
        D c41 = {0};
        D c42 = {0};
        D c43 = {0};
        D c44 = {0};
        for (int l = 0; l < k; l += KN) {
            V k0 = load(B + ldb * (j + 0) + l);
            V k1 = load(B + ldb * (j + 1) + l);
            V k2 = load(B + ldb * (j + 2) + l);
            V k3 = load(B + ldb * (j + 3) + l);
            V k4 = load(B + ldb * (j + 4) + l);
            V a0 = load(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
            c04 = madd(a0, k4, c04);
            V a1 = load(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            c11 = madd(a1, k1, c11);
            c12 = madd(a1, k2, c12);
            c13 = madd(a1, k3, c13);
            c14 = madd(a1, k4, c14);
            V a2 = load(A + lda * (i + 2) + l);
            c20 = madd(a2, k0, c20);
            c21 = madd(a2, k1, c21);
            c22 = madd(a2, k2, c22);
            c23 = madd(a2, k3, c23);
            c24 = madd(a2, k4, c24);
            V a3 = load(A + lda * (i + 3) + l);
            c30 = madd(a3, k0, c30);
            c31 = madd(a3, k1, c31);
            c32 = madd(a3, k2, c32);
            c33 = madd(a3, k3, c33);
            c34 = madd(a3, k4, c34);
            V a4 = load(A + lda * (i + 4) + l);
            c40 = madd(a4, k0, c40);
            c41 = madd(a4, k1, c41);
            c42 = madd(a4, k2, c42);
            c43 = madd(a4, k3, c43);
            c44 = madd(a4, k4, c44);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        C[ldc * (j + 0) + (i + 4)] = hsum(c40);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 1) + (i + 3)] = hsum(c31);
        C[ldc * (j + 1) + (i + 4)] = hsum(c41);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        C[ldc * (j + 2) + (i + 3)] = hsum(c32);
        C[ldc * (j + 2) + (i + 4)] = hsum(c42);
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum(c23);
        C[ldc * (j + 3) + (i + 3)] = hsum(c33);
        C[ldc * (j + 3) + (i + 4)] = hsum(c43);
        C[ldc * (j + 4) + (i + 0)] = hsum(c04);
        C[ldc * (j + 4) + (i + 1)] = hsum(c14);
        C[ldc * (j + 4) + (i + 2)] = hsum(c24);
        C[ldc * (j + 4) + (i + 3)] = hsum(c34);
        C[ldc * (j + 4) + (i + 4)] = hsum(c44);
        END_KERNEL()
    }

    dontinline void gemm3x4(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(3, 4)
        D c00 = {0};
        D c01 = {0};
        D c02 = {0};
        D c03 = {0};
        D c10 = {0};
        D c11 = {0};
        D c12 = {0};
        D c13 = {0};
        D c20 = {0};
        D c21 = {0};
        D c22 = {0};
        D c23 = {0};
        for (int l = 0; l < k; l += KN) {
            V k0 = load(B + ldb * (j + 0) + l);
            V k1 = load(B + ldb * (j + 1) + l);
            V k2 = load(B + ldb * (j + 2) + l);
            V k3 = load(B + ldb * (j + 3) + l);
            V a0 = load(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
            V a1 = load(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            c11 = madd(a1, k1, c11);
            c12 = madd(a1, k2, c12);
            c13 = madd(a1, k3, c13);
            V a2 = load(A + lda * (i + 2) + l);
            c20 = madd(a2, k0, c20);
            c21 = madd(a2, k1, c21);
            c22 = madd(a2, k2, c22);
            c23 = madd(a2, k3, c23);
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
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum(c23);
        END_KERNEL()
    }

    dontinline void gemm1x4(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 4)
        D c00 = {0}, e00 = {0};
        D c01 = {0}, e01 = {0};
        D c02 = {0}, e02 = {0};
        D c03 = {0}, e03 = {0};
        for (int l = 0; l < k; l += KN) {
            V a = load(A + lda * (i + 0) + l);
            c00 = madder(a, load(B + ldb * (j + 0) + l), c00, &e00);
            c01 = madder(a, load(B + ldb * (j + 1) + l), c01, &e01);
            c02 = madder(a, load(B + ldb * (j + 2) + l), c02, &e02);
            c03 = madder(a, load(B + ldb * (j + 3) + l), c03, &e03);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        END_KERNEL()
    }

    dontinline void gemm4x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 1)
        D c00 = {0}, e00 = {0};
        D c10 = {0}, e10 = {0};
        D c20 = {0}, e20 = {0};
        D c30 = {0}, e30 = {0};
        for (int l = 0; l < k; l += KN) {
            V b = load(B + ldb * (j + 0) + l);
            c00 = madder(load(A + lda * (i + 0) + l), b, c00, &e00);
            c10 = madder(load(A + lda * (i + 1) + l), b, c10, &e10);
            c20 = madder(load(A + lda * (i + 2) + l), b, c20, &e20);
            c30 = madder(load(A + lda * (i + 3) + l), b, c30, &e30);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        END_KERNEL()
    }

    dontinline void gemm1x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 1)
        D c = {0}, e = {0};
        for (int l = 0; l < k; l += KN)
            c = madder(load(A + lda * i + l), load(B + ldb * j + l), c, &e);
        C[ldc * j + i] = hsum(c);
        END_KERNEL()
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
