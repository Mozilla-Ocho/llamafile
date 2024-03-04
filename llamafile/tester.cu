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

#include "tester.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

const char *is_self_testing;

long long micros(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000 + (ts.tv_nsec + 999) / 1000;
}

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

void fill(int m, int n, half *A, int lda) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[1ll * lda * j + i] = numba();
}

void fill(int m, int n, float *A, int lda) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[1ll * lda * j + i] = numba();
}

void broadcast(int m, int n, half *A, int lda, half x) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[1ll * lda * j + i] = x;
}

void broadcast(int m, int n, float *A, int lda, float x) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[1ll * lda * j + i] = x;
}

void dgemm(bool aᵀ, bool bᵀ, //
           int m, int n, int k, float α, //
           const float *A, int lda, //
           const float *B, int ldb, float β, //
           float *C, int ldc) {
    ASSERT(m >= 0 && n >= 0 && k >= 0);
    ASSERT(lda >= std::max(1, aᵀ ? k : m));
    ASSERT(ldb >= std::max(1, bᵀ ? n : k));
    ASSERT(ldc >= std::max(1, m));
#pragma omp parallel for collapse(2) if (m * n * k > 30000)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double dot = 0;
            for (int l = 0; l < k; ++l)
                dot += (aᵀ ? A[1ll * lda * i + l] : A[1ll * lda * l + i]) *
                       (bᵀ ? B[1ll * ldb * l + j] : B[1ll * ldb * j + l]);
            C[1ll * ldc * j + i] = α * dot + β * C[1ll * ldc * j + i];
        }
}

void dgemm(bool aᵀ, bool bᵀ, //
           int m, int n, int k, float α, //
           const half *A, int lda, //
           const half *B, int ldb, float β, //
           float *C, int ldc) {
    ASSERT(m >= 0 && n >= 0 && k >= 0);
    ASSERT(lda >= std::max(1, aᵀ ? k : m));
    ASSERT(ldb >= std::max(1, bᵀ ? n : k));
    ASSERT(ldc >= std::max(1, m));
#pragma omp parallel for collapse(2) if (m * n * k > 30000)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double dot = 0;
            for (int l = 0; l < k; ++l)
                dot += __half2float(aᵀ ? A[1ll * lda * i + l] : A[1ll * lda * l + i]) *
                       __half2float(bᵀ ? B[1ll * ldb * l + j] : B[1ll * ldb * j + l]);
            C[1ll * ldc * j + i] = α * dot + β * C[1ll * ldc * j + i];
        }
}

void dgemm(bool aᵀ, bool bᵀ, //
           int m, int n, int k, float α, //
           const half *A, int lda, //
           const half *B, int ldb, float β, //
           half *C, int ldc) {
    ASSERT(m >= 0 && n >= 0 && k >= 0);
    ASSERT(lda >= std::max(1, aᵀ ? k : m));
    ASSERT(ldb >= std::max(1, bᵀ ? n : k));
    ASSERT(ldc >= std::max(1, m));
#pragma omp parallel for collapse(2) if (m * n * k > 30000)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float dot = 0;
            for (int l = 0; l < k; ++l)
                dot += __half2float(aᵀ ? A[1ll * lda * i + l] : A[1ll * lda * l + i]) *
                       __half2float(bᵀ ? B[1ll * ldb * l + j] : B[1ll * ldb * j + l]);
            C[1ll * ldc * j + i] = α * dot + β * __half2float(C[1ll * ldc * j + i]);
        }
}

template <typename T> double diff(int m, int n, const T *Wan, int lda, const T *Got, int ldb) {
    double s = 0;
    int got_nans = 0;
    int wan_nans = 0;
    if (!m || !n)
        return 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            if (IsNan(Wan[1ll * lda * j + i]))
                ++wan_nans;
            else if (IsNan(Got[1ll * ldb * j + i]))
                ++got_nans;
            else
                s += std::fabs(static_cast<double>(Wan[1ll * lda * j + i]) -
                               static_cast<double>(Got[1ll * ldb * j + i]));
    if (got_nans)
        fprintf(stderr, "WARNING: got %d NaNs!\n", got_nans);
    if (wan_nans)
        fprintf(stderr, "WARNING: want array has %d NaNs!\n", wan_nans);
    return s / (m * n);
}

template <typename T>
void show(FILE *f, int max, int m, int n, const T *A, int lda, const T *B, int ldb) {
    flockfile(f);
    fprintf(f, "      ");
    for (int i = 0; i < n && i < max; ++i)
        fprintf(f, "%13d", i);
    fprintf(f, "\n");
    for (int i = 0; i < m; ++i) {
        if (i == max) {
            fprintf(f, "...\n");
            break;
        }
        fprintf(f, "%5d ", i);
        for (int j = 0; j < n; ++j) {
            if (j == max) {
                fprintf(f, " ...");
                break;
            }
            char ba[32], bb[32];
            snprintf(ba, 32, "%13.7f", static_cast<double>(A[1ll * lda * j + i]));
            snprintf(bb, 32, "%13.7f", static_cast<double>(B[1ll * ldb * j + i]));
            for (int k = 0; ba[k] && bb[k]; ++k) {
                if (ba[k] != bb[k])
                    fputs_unlocked("\33[31m", f);
                fputc_unlocked(ba[k], f);
                if (ba[k] != bb[k])
                    fputs_unlocked("\33[0m", f);
            }
        }
        fprintf(f, "\n");
    }
    funlockfile(f);
}

template <typename T>
void misfit(FILE *f, int max, int m, int n, const T *A, int lda, const T *B, int ldb,
            const char *file, int line, double sad, double tol) {
    fprintf(f, "%s:%d: sad %.17g exceeds %g (%s)\nwant\n", file, line, sad, tol,
            is_self_testing ? is_self_testing : "n/a");
    show(f, max, m, n, A, lda, B, ldb);
    fprintf(f, "got\n");
    show(f, max, m, n, B, ldb, A, lda);
    fprintf(f, "\n");
}

template <typename T>
void checker(double tol, int m, int n, const T *A, int lda, const T *B, int ldb, const char *file,
             int line) {
    ASSERT(lda >= std::max(1, m));
    ASSERT(ldb >= std::max(1, m));
    double sad = diff(m, n, A, lda, B, ldb);
    if (sad <= tol) {
        if (!is_self_testing)
            printf("         %g error\n", sad);
    } else {
        flockfile(stderr);
        misfit(stderr, 16, m, n, A, lda, B, ldb, file, line, sad, tol);
        const char *path = "/tmp/llamafile_tester.log";
        FILE *f = fopen(path, "w");
        if (f) {
            misfit(f, 10000, m, n, A, lda, B, ldb, file, line, sad, tol);
            printf("see also %s\n", path);
        }
        exit(1);
    }
}

void check(double tol, int m, int n, const float *A, int lda, const float *B, int ldb,
           const char *file, int line) {
    checker(tol, m, n, A, lda, B, ldb, file, line);
}

void check(double tol, int m, int n, const half *A, int lda, const half *B, int ldb,
           const char *file, int line) {
    checker(tol, m, n, A, lda, B, ldb, file, line);
}

void run(const char *description) {
    printf("\n%s\n", description);
}

void bench(long long start, const char *description) {
    printf("%8lld µs %s\n", (micros() - start + ITERATIONS - 1) / ITERATIONS, description);
}

void passert(const char *file, int line, const char *expr) {
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
}
