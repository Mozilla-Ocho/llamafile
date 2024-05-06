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

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>

// multiplies matrix on cpu with column major ordering
//
//     m×k * k×n → m×n
//     k×m * k×n → m×n if aᵀ
//     m×k * n×k → m×n if bᵀ
//     k×m * n×k → m×n if aᵀ and bᵀ
//
template <typename T, typename TA, typename TB, typename TC>
void gemm(bool aT, bool bT, //
          int m, int n, int k, T alpha, //
          const TA *A, int lda, //
          const TB *B, int ldb, T beta, //
          TC *C, int ldc) {
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(lda >= std::max(1, aT ? k : m));
    assert(ldb >= std::max(1, bT ? n : k));
    assert(ldc >= std::max(1, m));
    assert(1ll * lda * (aT ? m : k) <= INT_MAX);
    assert(1ll * ldb * (bT ? k : n) <= INT_MAX);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            T d = 0;
            for (int l = 0; l < k; ++l) {
                T a = A[aT ? lda * i + l : lda * l + i];
                T b = B[bT ? ldb * l + j : ldb * j + l];
                d = std::fma(a, b, d);
            }
            if (beta) {
                T c = C[ldc * j + i];
                C[ldc * j + i] = std::fma(alpha, d, beta * c);
            } else {
                C[ldc * j + i] = alpha * d;
            }
        }
}

// multiplies matrices on cpu with column major ordering
//
//     m×k * k×n → m×n
//     k×m * k×n → m×n if aᵀ
//     m×k * n×k → m×n if bᵀ
//     k×m * n×k → m×n if aᵀ and bᵀ
//
template <typename T, typename TA, typename TB, typename TC>
void gsbe(bool aT, bool bT, //
          int m, int n, int k, T alpha, //
          const TA *A, int lda, long long sta, //
          const TB *B, int ldb, long long stb, T beta, //
          TC *C, int ldc, long long stc, int batches) {
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(lda >= std::max(1, aT ? k : m));
    assert(ldb >= std::max(1, bT ? n : k));
    assert(ldc >= std::max(1, m));
    assert(1ll * lda * (aT ? m : k) <= INT_MAX);
    assert(1ll * ldb * (bT ? k : n) <= INT_MAX);
    assert(std::max(0ll, stc) >= std::min(1ll * ldc * n, stc * 2));
    for (int z = 0; z < batches; ++z)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                T d = 0;
                for (int l = 0; l < k; ++l) {
                    T a = A[sta * z + (aT ? lda * i + l : lda * l + i)];
                    T b = B[stb * z + (bT ? ldb * l + j : ldb * j + l)];
                    d = std::fma(a, b, d);
                }
                if (beta) {
                    T c = C[stc * z + ldc * j + i];
                    C[stc * z + ldc * j + i] = std::fma(alpha, d, beta * c);
                } else {
                    C[stc * z + ldc * j + i] = alpha * d;
                }
            }
}
