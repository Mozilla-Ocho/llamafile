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

#include "bench.h"
#include "float.h"
#include "gemm.h"
#include "llama.cpp/ggml.h"
#include "llamafile.h"
#include "numba.h"

#define ITERATIONS 5

int main(int argc, char *argv[]) {
    float tolerance = 2e-5;
    int m = 510;
    int n = 513;
    int k = 512;
    int l = 3;
    int lda = k + l;
    int ldb = k + l;
    int ldc = m + l;
    float *A = new float[lda * m];
    float *B = new float[ldb * n];
    float *C = new float[ldc * n];
    float *G = new float[ldc * n];
    broadcast(A, lda * m, NAN);
    broadcast(B, ldb * n, NAN);
    broadcast(C, ldc * n, NAN);
    broadcast(G, ldc * n, NAN);
    randomize(k, m, A, lda);
    randomize(k, n, B, ldb);

    BENCH(gemm(true, false, m, n, k, 1., A, lda, B, ldb, 0., G, ldc));
    BENCH(llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 1, GGML_TASK_TYPE_COMPUTE,
                          GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32));

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            auto g = G[ldc * j + i];
            auto c = C[ldc * j + i];
            if (flt::isnan(g)) {
                fprintf(stderr, "%s:%d: found nan in reference matrix: i=%d j=%d\n", __FILE__,
                        __LINE__, i, j);
                return 3;
            }
            if (flt::isnan(c)) {
                fprintf(stderr, "%s:%d: found nan in output matrix: i=%d j=%d\n", __FILE__,
                        __LINE__, i, j);
                return 4;
            }
            auto d = g - c;
            if (d < 0)
                d = -d;
            if (d > tolerance) {
                fprintf(stderr, "%s:%d: difference exceeds tolerance: %g vs. %g (%g) i=%d j=%d\n",
                        __FILE__, __LINE__, g, c, d, i, j);
                return 5;
            }
        }

    delete[] G;
    delete[] C;
    delete[] B;
    delete[] A;
}
