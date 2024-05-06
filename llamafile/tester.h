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

#include "cuda.h"
#include "float.h"
#include "gemm.h"
#include "half.h"
#include "macros.h"
#include "naive.h"
#include "tinyblas.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>

//
//                 _   _          ___ _      _   ___
//                | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                 \__|_|_||_\_, |___/____/_/ \_\___/
//                           |__/
//
//                  BASIC LINEAR ALGEBRA SUBPROGRAMS
//

// c++ testing utilities for floating point math on cpu and gpu
//
// this file contains tools for testing matrix multiplication, measuring
// errors, and visualizing what went wrong in using ansi terminal codes.

#define ITERATIONS 20
#define TOMBSTONE 1.666f

#ifdef __HIP__
#define cublas hipblas
#endif

struct ErrorReport {
    long long worst;
    double avg;
    double sad;
    int infs;
    int flips;
    int zeroes;
    int denormals;
    int nans;
    double worsta;
    double worstb;
    unsigned long long worstabin;
    unsigned long long worstbbin;
};

extern const int kPageSize;
extern std::recursive_mutex g_log_lock;
extern thread_local const char *is_self_testing;

long long micros(void);
void *cudaMallocManagedOrDie(size_t);
void cudaFreeOrDie(void *);
void show_cuda_device(int);
void show_cuda_devices();
void test_matmul(std::function<void(int, int, int, int, float, float)>);

template <typename T> struct cuda_memory {
    const size_t size;
    T *const p;
    explicit cuda_memory(int len)
#ifdef __SANITIZE_ADDRESS__
        : size(len * sizeof(T)),
#else
        : size(ROUNDUP(len * sizeof(T) + 64, kPageSize) - 64),
#endif
          p(static_cast<T *>(cudaMallocManagedOrDie(size))) {
        broadcast(p, size / sizeof(T), NAN);
    }
    ~cuda_memory() {
        cudaFreeOrDie(p);
    }
};

template <typename T> ErrorReport diff(int m, int n, const T *Wan, int lda, const T *Got, int ldb) {
    double sad = 0;
    double worsta = 0;
    double worstb = 0;
    long long ulp = 0;
    long long worst = 0;
    int infs = 0;
    int flips = 0;
    int zeroes = 0;
    int got_nans = 0;
    int wan_nans = 0;
    int denormals = 0;
    int considered = 0;
    unsigned long long worstabin = 0;
    unsigned long long worstbbin = 0;
    if (m && n) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                T x = Wan[lda * j + i];
                T y = Got[ldb * j + i];
                if (flt::isnan(x)) {
                    ++wan_nans;
                } else if (flt::isnan(y)) {
                    ++got_nans;
                } else {
                    zeroes -= !x;
                    zeroes += !y;
                    infs -= flt::isinf(x);
                    infs += flt::isinf(y);
                    denormals -= flt::isdenormal(x);
                    denormals += flt::isdenormal(y);
                    flips += flt::sign(x) != flt::sign(y);
                    long long xi = flt::toint(x);
                    long long yi = flt::toint(y);
                    if (flt::sign(x) == flt::sign(y)) {
                        ++considered;
                        long long bad = std::abs(xi - yi);
                        ulp += bad;
                        if (bad > worst) {
                            worst = bad;
                            worsta = x;
                            worstb = y;
                            worstabin = xi;
                            worstbbin = yi;
                        }
                        sad += std::fabs(static_cast<double>(x) - static_cast<double>(y));
                    }
                }
            }
        if (got_nans)
            fprintf(stderr, "WARNING: got %d NaNs!\n", got_nans);
        if (wan_nans)
            fprintf(stderr, "WARNING: want array has %d NaNs!\n", wan_nans);
    }
    ErrorReport errors;
    errors.nans = got_nans + wan_nans;
    errors.avg = ulp / considered;
    errors.sad = sad / considered;
    errors.denormals = denormals;
    errors.zeroes = zeroes;
    errors.worst = worst;
    errors.flips = flips;
    errors.infs = infs;
    errors.worsta = worsta;
    errors.worstb = worstb;
    errors.worstabin = worstabin;
    errors.worstbbin = worstbbin;
    return errors;
}

template <typename T>
void show(FILE *f, int max, int m, int n, const T *A, int lda, const T *B, int ldb) {
    std::unique_lock<std::recursive_mutex> lock(g_log_lock);
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
            snprintf(ba, 32, "%13.7f", static_cast<double>(A[lda * j + i]));
            snprintf(bb, 32, "%13.7f", static_cast<double>(B[ldb * j + i]));
            for (int k = 0; ba[k] && bb[k]; ++k) {
                if (ba[k] != bb[k])
                    fputs("\33[31m", f);
                fputc(ba[k], f);
                if (ba[k] != bb[k])
                    fputs("\33[0m", f);
            }
        }
        fprintf(f, "\n");
    }
}

template <typename T>
void misfit(FILE *f, int max, int m, int n, int k, const T *A, int lda, const T *B, int ldb,
            const char *file, int line, double tol, ErrorReport &errors) {
    fprintf(f, "%s:%d: worst matrix error exceeds k=%d tolerance of %g ulp %s\n", file, line, k,
            tol * sqrt(k), is_self_testing ? is_self_testing : "");
    fprintf(f,
            "         %lld ulp - %g (%llx) vs. %g (%llx)\n"
            "         sad=%g nans=%d infs=%d flips=%d zeroes=%d denormals=%d\n",
            errors.worst, errors.worsta, errors.worstabin, errors.worstb, errors.worstbbin,
            errors.sad, errors.nans, errors.infs, errors.flips, errors.zeroes, errors.denormals);
    fprintf(f, "want\n");
    show(f, max, m, n, A, lda, B, ldb);
    fprintf(f, "got\n");
    show(f, max, m, n, B, ldb, A, lda);
    fprintf(f, "\n");
}

template <typename T>
void check(double tol, //
           int m, int n, int k, //
           const T *A, int lda, //
           const T *B, int ldb, //
           const char *file, int line) {
    assert(lda >= std::max(1, m));
    assert(ldb >= std::max(1, m));
    ErrorReport errors = diff(m, n, A, lda, B, ldb);
    if (!errors.nans && errors.sad <= tol && errors.flips < m * n * .01) {
        if (!is_self_testing)
            printf("         %lld ulp - %g (%llx) vs. %g (%llx)\n"
                   "         sad=%g nans=%d infs=%d flips=%d zeroes=%d denormals=%d\n",
                   errors.worst, errors.worsta, errors.worstabin, errors.worstb, errors.worstbbin,
                   errors.sad, errors.nans, errors.infs, errors.flips, errors.zeroes,
                   errors.denormals);
    } else {
        // std::unique_lock<std::recursive_mutex> lock(g_log_lock);
        misfit(stderr, 16, m, n, k, A, lda, B, ldb, file, line, tol, errors);
        const char *path = "/tmp/wompwomp.log";
        FILE *f = fopen(path, "w");
        if (f) {
            misfit(f, 10000, m, n, k, A, lda, B, ldb, file, line, tol, errors);
            fclose(f);
            fprintf(stderr, "see also %s\n", path);
        }
        fflush(stderr);
        (void)cudaDeviceReset();
        _Exit(1);
    }
}

#define CHECK(tol, m, n, k, A, lda, B, ldb) check(tol, m, n, k, A, lda, B, ldb, __FILE__, __LINE__)

#define CUDA_OR_DIE(x) \
    do { \
        cudaError_t err_ = (x); \
        if (err_ != cudaSuccess) \
            cuda_die(#x, __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err_)); \
    } while (0)

#define CUBLAS_OR_DIE(x) \
    do { \
        cublasStatus_t status_ = (x); \
        if (status_ != CUBLAS_STATUS_SUCCESS) \
            cuda_die(#x, __FUNCTION__, __FILE__, __LINE__, cublasGetStatusString(status_)); \
    } while (0)

#define TINYBLAS_OR_DIE(x) \
    do { \
        tinyblasStatus_t status_ = (x); \
        if (status_ != TINYBLAS_STATUS_SUCCESS) \
            cuda_die(#x, __FUNCTION__, __FILE__, __LINE__, tinyblasGetStatusString(status_)); \
    } while (0)

#define BENCH_CUDA(x) \
    do { \
        x; \
        if (!is_self_testing) { \
            cudaEvent_t start_, stop_; \
            CUDA_OR_DIE(cudaEventCreate(&stop_)); \
            CUDA_OR_DIE(cudaEventCreate(&start_)); \
            CUDA_OR_DIE(cudaEventRecord(start_, stream)); \
            for (int i_ = 0; i_ < ITERATIONS; ++i_) \
                x; \
            float msecTotal_ = 0; \
            CUDA_OR_DIE(cudaEventRecord(stop_, stream)); \
            CUDA_OR_DIE(cudaEventSynchronize(stop_)); \
            CUDA_OR_DIE(cudaEventElapsedTime(&msecTotal_, start_, stop_)); \
            printf("%8d us %s(%d, %d, %d) %g gigaflops\n", \
                   static_cast<int>(std::ceil(msecTotal_ / ITERATIONS * 1000)), __FUNCTION__, m, \
                   n, k, 1000. / (msecTotal_ / ITERATIONS) * m * n * k * 1e-9); \
            CUDA_OR_DIE(cudaEventDestroy(start_)); \
            CUDA_OR_DIE(cudaEventDestroy(stop_)); \
        } \
    } while (0)

#define RUN(x) \
    printf("                           \r\n%s\n", #x); \
    x; \
    printf("                           \r")

[[noreturn]] void cuda_die(const char *, const char *, const char *, int, const char *);

template <typename T> struct cublas_data_type;
template <> struct cublas_data_type<half> {
    static constexpr cudaDataType_t id = CUDA_R_16F;
};
template <> struct cublas_data_type<float> {
    static constexpr cudaDataType_t id = CUDA_R_32F;
};
template <> struct cublas_data_type<double> {
    static constexpr cudaDataType_t id = CUDA_R_64F;
};

template <typename T> struct cublas_compute_type;
template <> struct cublas_compute_type<half> {
    static constexpr cublasComputeType_t id = CUBLAS_COMPUTE_16F;
};
template <> struct cublas_compute_type<float> {
    static constexpr cublasComputeType_t id = CUBLAS_COMPUTE_32F;
};
template <> struct cublas_compute_type<double> {
    static constexpr cublasComputeType_t id = CUBLAS_COMPUTE_64F;
};

template <typename T> struct tinyblas_data_type;
template <> struct tinyblas_data_type<half> {
    static constexpr tinyblasDataType_t id = TINYBLAS_R_16F;
};
template <> struct tinyblas_data_type<float> {
    static constexpr tinyblasDataType_t id = TINYBLAS_R_32F;
};

template <typename T> struct tinyblas_compute_type;
template <> struct tinyblas_compute_type<half> {
    static constexpr tinyblasComputeType_t id = TINYBLAS_COMPUTE_16F;
};
template <> struct tinyblas_compute_type<float> {
    static constexpr tinyblasComputeType_t id = TINYBLAS_COMPUTE_32F;
};

// multiplies matrix with column major ordering
//
//     m×k * k×n → m×n
//     k×m * k×n → m×n if aᵀ
//     m×k * n×k → m×n if bᵀ
//     k×m * n×k → m×n if aᵀ and bᵀ
//
template <typename WORD, typename SRC, typename DST>
void gemmref(bool aT, bool bT, //
             int m, int n, int k, WORD alpha, //
             const SRC *A, int lda, //
             const SRC *B, int ldb, WORD beta, //
             DST *C, int ldc) {
    cudaStream_t stream;
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(lda >= std::max(1, aT ? k : m));
    assert(ldb >= std::max(1, bT ? n : k));
    assert(ldc >= std::max(1, m));
    CUDA_OR_DIE(cudaSetDevice(0));
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    BENCH_CUDA(
        CUDA_OR_DIE(naive::gemm(stream, aT, bT, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
}

// multiplies matrix with column major ordering
//
//     m×k * k×n → m×n
//     k×m * k×n → m×n if aᵀ
//     m×k * n×k → m×n if bᵀ
//     k×m * n×k → m×n if aᵀ and bᵀ
//
template <typename WORD, typename SRC, typename DST>
void gsberef(bool aT, bool bT, //
             int m, int n, int k, WORD alpha, //
             const SRC *A, int lda, long long sta, //
             const SRC *B, int ldb, long long stb, WORD beta, //
             DST *C, int ldc, long long stc, int batches) {
    cudaStream_t stream;
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(lda >= std::max(1, aT ? k : m));
    assert(ldb >= std::max(1, bT ? n : k));
    assert(ldc >= std::max(1, m));
    assert(std::max(0ll, stc) >= std::min(1ll * ldc * n, stc * 2));
    CUDA_OR_DIE(cudaSetDevice(0));
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    BENCH_CUDA(CUDA_OR_DIE(naive::gsbe(stream, aT, bT, m, n, k, alpha, A, lda, sta, B, ldb, stb,
                                       beta, C, ldc, stc, batches)));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
}

template <typename WORD, typename SRC, typename DST>
void cublas(bool aT, bool bT, //
            int m, int n, int k, WORD alpha, //
            const SRC *A, int lda, //
            const SRC *B, int ldb, WORD beta, //
            DST *C, int ldc) {
    cudaStream_t stream;
    cublasHandle_t blas;
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(lda >= std::max(1, aT ? k : m));
    assert(ldb >= std::max(1, bT ? n : k));
    assert(ldc >= std::max(1, m));
    CUDA_OR_DIE(cudaSetDevice(0));
    CUBLAS_OR_DIE(cublasCreate(&blas));
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    CUBLAS_OR_DIE(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
#ifdef __HIP__
    CUBLAS_OR_DIE(hipblasSetAtomicsMode(blas, HIPBLAS_ATOMICS_NOT_ALLOWED));
#endif
    CUBLAS_OR_DIE(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
    CUBLAS_OR_DIE(cublasSetStream(blas, stream));
    BENCH_CUDA(CUBLAS_OR_DIE(cublasGemmEx(
        blas, aT ? CUBLAS_OP_T : CUBLAS_OP_N, bT ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A,
        cublas_data_type<SRC>::id, lda, B, cublas_data_type<SRC>::id, ldb, &beta, C,
        cublas_data_type<DST>::id, ldc, cublas_compute_type<WORD>::id, CUBLAS_GEMM_DEFAULT)));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    CUBLAS_OR_DIE(cublasDestroy(blas));
}

template <typename WORD, typename SRC, typename DST>
void cublasGSBE(bool aT, bool bT, //
                int m, int n, int k, WORD alpha, //
                const SRC *A, int lda, long long sta, //
                const SRC *B, int ldb, long long stb, WORD beta, //
                DST *C, int ldc, long long stc, int batches) {
    cudaStream_t stream;
    cublasHandle_t blas;
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(lda >= std::max(1, aT ? k : m));
    assert(ldb >= std::max(1, bT ? n : k));
    assert(ldc >= std::max(1, m));
    CUDA_OR_DIE(cudaSetDevice(0));
    CUBLAS_OR_DIE(cublasCreate(&blas));
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    CUBLAS_OR_DIE(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
#ifdef __HIP__
    CUBLAS_OR_DIE(hipblasSetAtomicsMode(blas, HIPBLAS_ATOMICS_NOT_ALLOWED));
#endif
    CUBLAS_OR_DIE(cublasSetStream(blas, stream));
    BENCH_CUDA(CUBLAS_OR_DIE(cublasGemmStridedBatchedEx(
        blas, aT ? CUBLAS_OP_T : CUBLAS_OP_N, bT ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A,
        cublas_data_type<SRC>::id, lda, sta, B, cublas_data_type<SRC>::id, ldb, stb, &beta, C,
        cublas_data_type<DST>::id, ldc, stc, batches, cublas_compute_type<WORD>::id,
        CUBLAS_GEMM_DEFAULT)));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    CUBLAS_OR_DIE(cublasDestroy(blas));
}

template <typename WORD, typename SRC, typename DST>
void tinyblas(bool aT, bool bT, //
              int m, int n, int k, WORD alpha, //
              const SRC *A, int lda, //
              const SRC *B, int ldb, WORD beta, //
              DST *C, int ldc) {
    cudaStream_t stream;
    tinyblasHandle_t blas;
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(lda >= std::max(1, aT ? k : m));
    assert(ldb >= std::max(1, bT ? n : k));
    assert(ldc >= std::max(1, m));
    CUDA_OR_DIE(cudaSetDevice(0));
    TINYBLAS_OR_DIE(tinyblasCreate(&blas));
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    TINYBLAS_OR_DIE(tinyblasSetStream(blas, stream));
    BENCH_CUDA(TINYBLAS_OR_DIE(tinyblasGemmEx(
        blas, aT ? TINYBLAS_OP_T : TINYBLAS_OP_N, bT ? TINYBLAS_OP_T : TINYBLAS_OP_N, m, n, k,
        &alpha, A, tinyblas_data_type<SRC>::id, lda, B, tinyblas_data_type<SRC>::id, ldb, &beta, C,
        tinyblas_data_type<DST>::id, ldc, tinyblas_compute_type<WORD>::id, TINYBLAS_GEMM_DEFAULT)));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    TINYBLAS_OR_DIE(tinyblasDestroy(blas));
}

template <typename WORD, typename SRC, typename DST>
void tinyblasGSBE(bool aT, bool bT, //
                  int m, int n, int k, WORD alpha, //
                  const SRC *A, int lda, long long sta, //
                  const SRC *B, int ldb, long long stb, WORD beta, //
                  DST *C, int ldc, long long stc, int batches) {
    cudaStream_t stream;
    tinyblasHandle_t blas;
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(lda >= std::max(1, aT ? k : m));
    assert(ldb >= std::max(1, bT ? n : k));
    assert(ldc >= std::max(1, m));
    CUDA_OR_DIE(cudaSetDevice(0));
    TINYBLAS_OR_DIE(tinyblasCreate(&blas));
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    TINYBLAS_OR_DIE(tinyblasSetStream(blas, stream));
    BENCH_CUDA(TINYBLAS_OR_DIE(tinyblasGemmStridedBatchedEx(
        blas, aT ? TINYBLAS_OP_T : TINYBLAS_OP_N, bT ? TINYBLAS_OP_T : TINYBLAS_OP_N, m, n, k,
        &alpha, A, tinyblas_data_type<SRC>::id, lda, sta, B, tinyblas_data_type<SRC>::id, ldb, stb,
        &beta, C, tinyblas_data_type<DST>::id, ldc, stc, batches, tinyblas_compute_type<WORD>::id,
        TINYBLAS_GEMM_DEFAULT)));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    TINYBLAS_OR_DIE(tinyblasDestroy(blas));
}
