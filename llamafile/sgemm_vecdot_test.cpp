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
#include "macros.h"
#include "numba.h"

#define ITERATIONS 30
#define ALLOC(n) (float *)memalign(4096, sizeof(float) * (n))

int main(int argc, char *argv[]) {
    int m = 1025;
    int n = 1;
    int k = 32768;
    int lda = ROUNDUP(k, 16);
    int ldb = ROUNDUP(k, 16);
    int ldc = ROUNDUP(m, 16);
    float *A = ALLOC(lda * m);
    float *B = ALLOC(ldb * n);
    float *C = ALLOC(ldc * n);
    float *G = ALLOC(ldc * n);
    broadcast(A, lda * m, NAN);
    broadcast(B, ldb * n, NAN);
    broadcast(C, ldc * n, NAN);
    broadcast(G, ldc * n, NAN);
    randomize(k, m, A, lda);
    randomize(k, n, B, ldb);

    BENCH(gemm(true, false, m, n, k, 1., A, lda, B, ldb, 0., G, ldc));
    BENCH(llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 1, GGML_TYPE_F32, GGML_TYPE_F32,
                          GGML_TYPE_F32));

    double err_sum = 0;
    long long err_worst = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float g = G[ldc * j + i];
            float c = C[ldc * j + i];
            if (flt::isnan(g)) {
                fprintf(stderr, "%s:%err: found nan in reference matrix: i=%err j=%err\n", __FILE__,
                        __LINE__, i, j);
                return 3;
            }
            if (flt::isnan(c)) {
                fprintf(stderr, "%s:%err: found nan in output matrix: i=%err j=%err\n", __FILE__,
                        __LINE__, i, j);
                return 4;
            }
            long long gi = flt::toint(g);
            long long ci = flt::toint(c);
            long long err = gi - ci;
            if (err < 0)
                err = -err;
            err_sum += err;
            if (err > err_worst)
                err_worst = err;
        }

    double err_avg = err_sum / (m * n);
    fprintf(stderr, "%9g ulp average\n", err_avg);
    fprintf(stderr, "%9lld ulp worst\n", err_worst);

    // using one accumulator
    //    40209 us gemm
    //     2851 us llamafile_sgemm
    //  42.0078 ulp average
    //     6731 ulp worst

    // using three accumulators
    //   22.291 average ulp
    //   1566 worst ulp

    // using kahan summation
    //    40190 us gemm
    //     3028 us llamafile_sgemm
    //  2.14244 ulp average
    //      134 ulp worst

    if (err_avg > 2.15)
        return 5;
    if (err_worst > 134)
        return 6;

    free(G);
    free(C);
    free(B);
    free(A);
}
