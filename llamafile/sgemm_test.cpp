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

#ifdef __x86_64__

#include "gemm.h"
#include "llama.cpp/ggml.h"
#include "llamafile.h"

int rand32(void) {
    static unsigned long long lcg = 1;
    lcg *= 6364136223846793005;
    lcg += 1442695040888963407;
    return lcg >> 32;
}

float float01(unsigned x) { // (0,1)
    return 1.f / 8388608 * ((x >> 9) + .5f);
}

float numba(void) { // (-1,1)
    return float01(rand32()) * 2 - 1;
}

template <typename T> void randomize(T *A, int n) {
    for (int i = 0; i < n; ++i)
        A[i] = numba();
}

template <typename T> void broadcast(T *A, int n, T x) {
    for (int i = 0; i < n; ++i)
        A[i] = x;
}

int main(int argc, char *argv[]) {
    float tolerance = 1e-5;
    int n = 32;
    int m = 32;
    int k = 1024;
    int l = 0;
    int lda = k + l;
    int ldb = k + l;
    int ldc = m + l;
    float *A = new float[lda * m];
    float *B = new float[ldb * n];
    float *C = new float[ldc * n];
    float *G = new float[ldc * n];
    randomize(A, lda * m);
    randomize(B, ldb * n);
    broadcast(C, ldc * n, NAN);
    broadcast(G, ldc * n, NAN);
    gemm<double>(true, false, m, n, k, 1.f, A, lda, B, ldb, 0.f, G, ldc);
    if (!llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 1, GGML_TASK_TYPE_COMPUTE,
                         GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32))
        return 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            auto g = G[ldc * j + i];
            auto c = C[ldc * j + i];
            if (isnan(g))
                return 3;
            if (isnan(c))
                return 4;
            auto diff = g - c;
            if (diff < 0)
                diff = -diff;
            if (diff > tolerance)
                return 5;
        }
    }
}

#else
int main(int argc, char *argv[]) {
}
#endif // __x86_64__
