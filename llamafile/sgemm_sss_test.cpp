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

#include "ansiblas.h"
#include "bench.h"
#include "float.h"
#include "llama.cpp/ggml.h"
#include "llamafile.h"
#include "macros.h"
#include "numba.h"
#include <cassert>

#define ITERATIONS 5
#define ALLOC(n) (float *)memalign(4096, sizeof(float) * (n))

void llamafile_sgemm_openmp(long m, long n, long k, const void *A, long lda, const void *B,
                            long ldb, void *C, long ldc, int task, int Atype, int Btype, int Ctype,
                            int precision) {
    int nth = sysconf(_SC_NPROCESSORS_ONLN);
#pragma omp parallel for
    for (int ith = 0; ith < nth; ++ith) {
        bool res = llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task, Atype, Btype,
                                   Ctype, precision);
        assert(res);
    }
}

int test(void) {
    float tolerance = 2e-5;
    int m = 510;
    int n = 513;
    int k = 512 * 8;
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

    BENCH(ansiBLAS::sgemm(m, n, k, A, lda, B, ldb, G, ldc));
    BENCH(llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 1, GGML_TASK_TYPE_COMPUTE,
                          GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_PREC_DEFAULT));

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
    fprintf(stderr, "%12g ulp average\n", err_avg);
    fprintf(stderr, "%12lld ulp worst\n", err_worst);

    free(G);
    free(C);
    free(B);
    free(A);

    return 0;
}

int main(int argc, char *argv[]) {
    int rc;

    llamafile_trapping_enabled(+1);

    printf("\nFLAG_precise = false;\n");
    FLAG_precise = false;
    if ((rc = test()))
        return rc;

    printf("\nFLAG_precise = true;\n");
    FLAG_precise = true;
    if ((rc = test()))
        return rc;
}
