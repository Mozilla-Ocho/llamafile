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
#include "madd.h"

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
        if (VECTOR_REGISTERS >= 32 && m - m0 >= 5 && n - n0 >= 5) {
            mc = 5;
            nc = 5;
            gemm5x5(m0, m, n0, n);
        } else if (m - m0 >= 3 && n - n0 >= 4) {
            mc = 3;
            nc = 4;
            gemm3x4(m0, m, n0, n);
        } else if (m - m0 >= 1 && n - n0 >= 4) {
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

    dontinline void gemm5x5(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(5, 5)
        V c00 = zero();
        V c01 = zero();
        V c02 = zero();
        V c03 = zero();
        V c04 = zero();
        V c10 = zero();
        V c11 = zero();
        V c12 = zero();
        V c13 = zero();
        V c14 = zero();
        V c20 = zero();
        V c21 = zero();
        V c22 = zero();
        V c23 = zero();
        V c24 = zero();
        V c30 = zero();
        V c31 = zero();
        V c32 = zero();
        V c33 = zero();
        V c34 = zero();
        V c40 = zero();
        V c41 = zero();
        V c42 = zero();
        V c43 = zero();
        V c44 = zero();
        for (int l = 0; l < k; l += KN) {
            V k0 = load_b(B + ldb * (j + 0) + l);
            V k1 = load_b(B + ldb * (j + 1) + l);
            V k2 = load_b(B + ldb * (j + 2) + l);
            V k3 = load_b(B + ldb * (j + 3) + l);
            V k4 = load_b(B + ldb * (j + 4) + l);
            V a0 = load_a(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
            c04 = madd(a0, k4, c04);
            V a1 = load_a(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            c11 = madd(a1, k1, c11);
            c12 = madd(a1, k2, c12);
            c13 = madd(a1, k3, c13);
            c14 = madd(a1, k4, c14);
            V a2 = load_a(A + lda * (i + 2) + l);
            c20 = madd(a2, k0, c20);
            c21 = madd(a2, k1, c21);
            c22 = madd(a2, k2, c22);
            c23 = madd(a2, k3, c23);
            c24 = madd(a2, k4, c24);
            V a3 = load_a(A + lda * (i + 3) + l);
            c30 = madd(a3, k0, c30);
            c31 = madd(a3, k1, c31);
            c32 = madd(a3, k2, c32);
            c33 = madd(a3, k3, c33);
            c34 = madd(a3, k4, c34);
            V a4 = load_a(A + lda * (i + 4) + l);
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
        V c00 = zero();
        V c01 = zero();
        V c02 = zero();
        V c03 = zero();
        V c10 = zero();
        V c11 = zero();
        V c12 = zero();
        V c13 = zero();
        V c20 = zero();
        V c21 = zero();
        V c22 = zero();
        V c23 = zero();
        for (int l = 0; l < k; l += KN) {
            V k0 = load_b(B + ldb * (j + 0) + l);
            V k1 = load_b(B + ldb * (j + 1) + l);
            V k2 = load_b(B + ldb * (j + 2) + l);
            V k3 = load_b(B + ldb * (j + 3) + l);
            V a0 = load_a(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
            V a1 = load_a(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            c11 = madd(a1, k1, c11);
            c12 = madd(a1, k2, c12);
            c13 = madd(a1, k3, c13);
            V a2 = load_a(A + lda * (i + 2) + l);
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
        V c00 = zero();
        V c01 = zero();
        V c02 = zero();
        V c03 = zero();
        for (int l = 0; l < k; l += KN) {
            V a0 = load_a(A + lda * (i + 0) + l);
            V k0 = load_b(B + ldb * (j + 0) + l);
            V k1 = load_b(B + ldb * (j + 1) + l);
            V k2 = load_b(B + ldb * (j + 2) + l);
            V k3 = load_b(B + ldb * (j + 3) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        END_KERNEL()
    }

    dontinline void gemm4x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 1)
        V c00 = zero();
        V c10 = zero();
        V c20 = zero();
        V c30 = zero();
        for (int l = 0; l < k; l += KN) {
            V k0 = load_b(B + ldb * (j + 0) + l);
            V a0 = load_a(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            V a1 = load_a(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            V a2 = load_a(A + lda * (i + 2) + l);
            c20 = madd(a2, k0, c20);
            V a3 = load_a(A + lda * (i + 3) + l);
            c30 = madd(a3, k0, c30);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        END_KERNEL()
    }

    dontinline void gemm1x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 1)
        V c = zero();
        for (int l = 0; l < k; l += KN)
            c = madd(load_a(A + lda * i + l), load_b(B + ldb * j + l), c);
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
