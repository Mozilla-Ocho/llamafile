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

#define ITERATIONS 30

int main(int argc, char *argv[]) {
    int m = 1025;
    int n = 1;
    int k = 32768;
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
    fprintf(stderr, "note: %g average ulp\n", err_avg);
    fprintf(stderr, "note: %lld worst ulp\n", err_worst);

    // using one accumulator
    //   52.7393 average ulp
    //   9052 worst ulp

    // using three accumulators
    //   22.291 average ulp
    //   1566 worst ulp

    // using kahan summation
    //   2.14244 average ulp
    //   134 worst ulp

    if (err_avg > 30)
        return 5;
    if (err_worst > 3000)
        return 6;

    delete[] G;
    delete[] C;
    delete[] B;
    delete[] A;
}
