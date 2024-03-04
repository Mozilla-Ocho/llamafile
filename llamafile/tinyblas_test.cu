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

#include "tinyblas.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "tester.h"

#define TOMBSTONE 1.666
#define PRECISION_F16 0.2
#define PRECISION_F32 9e-6

#ifdef __NVCC__
#include <cublas_v2.h>
#include <cuda_runtime.h>
#else
#define HIPBLAS_V2
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#define cudaFree hipFree
#define cudaSuccess hipSuccess
#define cudaError_t hipError_t
#define cudaEvent_t hipEvent_t
#define cudaStream_t hipStream_t
#define cublasSgemm hipblasSgemm
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy
#define cublasSetStream hipblasSetStream
#define cudaStreamCreate hipStreamCreate
#define cudaMallocManaged hipMallocManaged
#define cudaStreamDestroy hipStreamDestroy
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamSynchronize hipStreamSynchronize
#define cublasGetStatusString hipblasStatusToString
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaDeviceReset hipDeviceReset
#define cublasGemmEx hipblasGemmEx
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUDA_R_16F HIPBLAS_R_16F
#define CUDA_R_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#endif

#define ARRAYLEN(A) ((sizeof(A) / sizeof(*(A))) / ((unsigned)!(sizeof(A) % sizeof(*(A)))))

#define CUDA_OR_DIE(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) \
            die(#x, __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err)); \
    } while (0)

#define CUBLAS_OR_DIE(x) \
    do { \
        cublasStatus_t status = (x); \
        if (status != CUBLAS_STATUS_SUCCESS) \
            die(#x, __FUNCTION__, __FILE__, __LINE__, cublasGetStatusString(status)); \
    } while (0)

#define TINYBLAS_OR_DIE(x) \
    do { \
        tinyblasStatus_t status = (x); \
        if (status != TINYBLAS_STATUS_SUCCESS) \
            die(#x, __FUNCTION__, __FILE__, __LINE__, tinyblasGetStatusString(status)); \
    } while (0)

static const int kDims[] = {1, 2, 23, 77, 15, 127, 128, 129};
static const float kAlphas[] = {1, .5};
static const float kBetas[] = {0, .1};

[[noreturn]]
void die(const char *stmt, const char *func, const char *file, int line, const char *msg) {
    int id = -1;
    (void)cudaGetDevice(&id);
    fprintf(stderr, "CUDA error: %s\n", msg);
    fprintf(stderr, "  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    fprintf(stderr, "  %s\n", stmt);
    (void)cudaDeviceReset();
    exit(1);
}

void checkFloat01() {
    for (int i = 0; i < 10000; ++i) {
        float x = float01(rand32());
        ASSERT(x >= -1);
        ASSERT(x <= +1);
    }
}

void checkCublasWorksSSS() {

    cublasHandle_t hand;
    CUBLAS_OR_DIE(cublasCreate(&hand));

    cudaStream_t stream;
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    CUBLAS_OR_DIE(cublasSetStream(hand, stream));

    float *A, *B, *C, *G;
    long long maxdim = 4096;
    CUDA_OR_DIE(cudaMallocManaged(&A, maxdim * maxdim * sizeof(float)));
    CUDA_OR_DIE(cudaMallocManaged(&B, maxdim * maxdim * sizeof(float)));
    CUDA_OR_DIE(cudaMallocManaged(&C, maxdim * maxdim * sizeof(float)));
    CUDA_OR_DIE(cudaMallocManaged(&G, maxdim * maxdim * sizeof(float)));

    std::atomic_llong t = ATOMIC_VAR_INIT(0);
    for (int ai = 0; ai < ARRAYLEN(kAlphas); ++ai)
        for (int bi = 0; bi < ARRAYLEN(kBetas); ++bi)
            for (int mi = 0; mi < ARRAYLEN(kDims); ++mi)
                for (int ni = 0; ni < ARRAYLEN(kDims); ++ni)
                    for (int ki = 0; ki < ARRAYLEN(kDims); ++ki) {
                        float α = kAlphas[ai];
                        float β = kBetas[bi];
                        int m = kDims[mi];
                        int n = kDims[ARRAYLEN(kDims) - 1 - ni];
                        int k = kDims[ki];
                        int lda = m;
                        int ldb = k;
                        int ldc = m;
                        char name[192];
                        snprintf(name, sizeof(name), "testing %4d %4d %4d α=%g β=%g", m, n, k, α,
                                 β);
                        if (t++ % 7 == 0)
                            fprintf(stderr, "%s\r", name);
                        is_self_testing = name;
                        fill(m, k, A, lda);
                        fill(k, n, B, ldb);
                        broadcast(m, n, G, n, TOMBSTONE);
                        broadcast(m, n, C, n, TOMBSTONE);
                        dgemm(false, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
                        CUBLAS_OR_DIE(cublasSgemm(hand, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &α, A,
                                                  lda, B, ldb, &β, C, ldc));
                        CUDA_OR_DIE(cudaStreamSynchronize(stream));
                        CHECK(PRECISION_F32, m, n, G, ldc, C, ldc);
                        is_self_testing = 0;
                    }

    float α = 1;
    float β = 0;
    int m = 4096;
    int n = 1024;
    int k = 577;
    int lda = k;
    int ldb = k;
    int ldc = m;
    fill(k, m, A, lda);
    fill(k, n, B, ldb);
    fill(m, n, C, ldc);
    fill(m, n, G, ldc);
    broadcast(m, n, G, n, TOMBSTONE);
    broadcast(m, n, C, n, TOMBSTONE);
    dgemm(true, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
    CUBLAS_OR_DIE(
        cublasSgemm(hand, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &α, A, lda, B, ldb, &β, C, ldc));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CHECK(PRECISION_F32, m, n, G, ldc, C, ldc);
    cudaEvent_t start, stop;
    CUDA_OR_DIE(cudaEventCreate(&stop));
    CUDA_OR_DIE(cudaEventCreate(&start));
    CUDA_OR_DIE(cudaEventRecord(start, NULL));
    for (int i = 0; i < ITERATIONS; ++i)
        CUBLAS_OR_DIE(
            cublasSgemm(hand, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &α, A, lda, B, ldb, &β, C, ldc));
    float msecTotal = 0;
    CUDA_OR_DIE(cudaEventRecord(stop, NULL));
    CUDA_OR_DIE(cudaEventSynchronize(stop));
    CUDA_OR_DIE(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("%8lld µs cublasSgemm(%d, %d, %d)\n",
           (long long)std::ceil(msecTotal / ITERATIONS * 1000), m, n, k);
    CUDA_OR_DIE(cudaEventDestroy(start));
    CUDA_OR_DIE(cudaEventDestroy(stop));

    CUDA_OR_DIE(cudaFree(G));
    CUDA_OR_DIE(cudaFree(C));
    CUDA_OR_DIE(cudaFree(B));
    CUDA_OR_DIE(cudaFree(A));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    CUBLAS_OR_DIE(cublasDestroy(hand));
}

void checkTinyblasWorksSSS() {

    tinyblasHandle_t hand;
    TINYBLAS_OR_DIE(tinyblasCreate(&hand));

    cudaStream_t stream;
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    TINYBLAS_OR_DIE(tinyblasSetStream(hand, stream));

    float *A, *B, *C, *G;
    long long maxdim = 4096;
    CUDA_OR_DIE(cudaMallocManaged(&A, maxdim * maxdim * sizeof(float)));
    CUDA_OR_DIE(cudaMallocManaged(&B, maxdim * maxdim * sizeof(float)));
    CUDA_OR_DIE(cudaMallocManaged(&C, maxdim * maxdim * sizeof(float)));
    CUDA_OR_DIE(cudaMallocManaged(&G, maxdim * maxdim * sizeof(float)));

    std::atomic_llong t = ATOMIC_VAR_INIT(0);
    for (int ai = 0; ai < ARRAYLEN(kAlphas); ++ai)
        for (int bi = 0; bi < ARRAYLEN(kBetas); ++bi)
            for (int mi = 0; mi < ARRAYLEN(kDims); ++mi)
                for (int ni = 0; ni < ARRAYLEN(kDims); ++ni)
                    for (int ki = 0; ki < ARRAYLEN(kDims); ++ki) {
                        float α = kAlphas[ai];
                        float β = kBetas[bi];
                        int m = kDims[mi];
                        int n = kDims[ARRAYLEN(kDims) - 1 - ni];
                        int k = kDims[ki];
                        int lda = m;
                        int ldb = k;
                        int ldc = m;
                        char name[192];
                        snprintf(name, sizeof(name), "testing %4d %4d %4d α=%g β=%g", m, n, k, α,
                                 β);
                        if (t++ % 7 == 0)
                            fprintf(stderr, "%s\r", name);
                        is_self_testing = name;
                        fill(m, k, A, lda);
                        fill(k, n, B, ldb);
                        broadcast(m, n, G, n, TOMBSTONE);
                        broadcast(m, n, C, n, TOMBSTONE);
                        dgemm(false, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
                        TINYBLAS_OR_DIE(tinyblasSgemm(hand, TINYBLAS_OP_N, TINYBLAS_OP_N, m, n, k,
                                                      &α, A, lda, B, ldb, &β, C, ldc));
                        CUDA_OR_DIE(cudaStreamSynchronize(stream));
                        CHECK(PRECISION_F32, m, n, G, ldc, C, ldc);
                        is_self_testing = 0;
                    }

    float α = 1;
    float β = 0;
    int m = 4096;
    int n = 1024;
    int k = 577;
    int lda = k;
    int ldb = k;
    int ldc = m;
    fill(k, m, A, lda);
    fill(k, n, B, ldb);
    fill(m, n, C, ldc);
    fill(m, n, G, ldc);
    broadcast(m, n, G, n, TOMBSTONE);
    broadcast(m, n, C, n, TOMBSTONE);
    dgemm(true, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
    TINYBLAS_OR_DIE(
        tinyblasSgemm(hand, TINYBLAS_OP_T, TINYBLAS_OP_N, m, n, k, &α, A, lda, B, ldb, &β, C, ldc));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CHECK(PRECISION_F32, m, n, G, ldc, C, ldc);
    cudaEvent_t start, stop;
    CUDA_OR_DIE(cudaEventCreate(&stop));
    CUDA_OR_DIE(cudaEventCreate(&start));
    CUDA_OR_DIE(cudaEventRecord(start, NULL));
    for (int i = 0; i < ITERATIONS; ++i)
        TINYBLAS_OR_DIE(tinyblasSgemm(hand, TINYBLAS_OP_T, TINYBLAS_OP_N, m, n, k, &α, A, lda, B,
                                      ldb, &β, C, ldc));
    float msecTotal = 0;
    CUDA_OR_DIE(cudaEventRecord(stop, NULL));
    CUDA_OR_DIE(cudaEventSynchronize(stop));
    CUDA_OR_DIE(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("%8lld µs tinyblasSgemm(%d, %d, %d)\n",
           (long long)std::ceil(msecTotal / ITERATIONS * 1000), m, n, k);
    CUDA_OR_DIE(cudaEventDestroy(start));
    CUDA_OR_DIE(cudaEventDestroy(stop));

    CUDA_OR_DIE(cudaFree(G));
    CUDA_OR_DIE(cudaFree(C));
    CUDA_OR_DIE(cudaFree(B));
    CUDA_OR_DIE(cudaFree(A));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    TINYBLAS_OR_DIE(tinyblasDestroy(hand));
}

void checkCublasWorksHHS() {

    cublasHandle_t hand;
    CUBLAS_OR_DIE(cublasCreate(&hand));

    cudaStream_t stream;
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    CUBLAS_OR_DIE(cublasSetStream(hand, stream));

    half *A, *B;
    float *C, *G;
    long long maxdim = 4096;
    CUDA_OR_DIE(cudaMallocManaged(&A, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&B, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&C, maxdim * maxdim * sizeof(float)));
    CUDA_OR_DIE(cudaMallocManaged(&G, maxdim * maxdim * sizeof(float)));

    std::atomic_llong t = ATOMIC_VAR_INIT(0);
    for (int ai = 0; ai < ARRAYLEN(kAlphas); ++ai)
        for (int bi = 0; bi < ARRAYLEN(kBetas); ++bi)
            for (int mi = 0; mi < ARRAYLEN(kDims); ++mi)
                for (int ni = 0; ni < ARRAYLEN(kDims); ++ni)
                    for (int ki = 0; ki < ARRAYLEN(kDims); ++ki) {
                        float α = kAlphas[ai];
                        float β = kBetas[bi];
                        int m = kDims[mi];
                        int n = kDims[ARRAYLEN(kDims) - 1 - ni];
                        int k = kDims[ki];
                        int lda = m;
                        int ldb = k;
                        int ldc = m;

                        char name[192];
                        snprintf(name, sizeof(name), "testing %4d %4d %4d α=%g β=%g", m, n, k, α,
                                 β);
                        if (t++ % 7 == 0)
                            fprintf(stderr, "%s\r", name);
                        is_self_testing = name;

                        fill(m, k, A, lda);
                        fill(k, n, B, ldb);
                        broadcast(m, n, G, n, TOMBSTONE);
                        broadcast(m, n, C, n, TOMBSTONE);
                        dgemm(false, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);

                        CUBLAS_OR_DIE(cublasGemmEx(hand, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &α, A,
                                                   CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &β, C,
                                                   CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F,
                                                   CUBLAS_GEMM_DEFAULT));
                        CUDA_OR_DIE(cudaStreamSynchronize(stream));

                        CHECK(PRECISION_F32, m, n, G, ldc, C, ldc);
                        is_self_testing = 0;
                    }

    float α = 1;
    float β = 0;
    int m = 4096;
    int n = 1024;
    int k = 577;
    int lda = k;
    int ldb = k;
    int ldc = m;
    fill(k, m, A, lda);
    fill(k, n, B, ldb);
    fill(m, n, C, ldc);
    fill(m, n, G, ldc);
    broadcast(m, n, G, n, TOMBSTONE);
    broadcast(m, n, C, n, TOMBSTONE);
    dgemm(true, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
    CUBLAS_OR_DIE(cublasGemmEx(hand, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &α, A, CUDA_R_16F, lda, B,
                               CUDA_R_16F, ldb, &β, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CHECK(PRECISION_F32, m, n, G, ldc, C, ldc);
    cudaEvent_t start, stop;
    CUDA_OR_DIE(cudaEventCreate(&stop));
    CUDA_OR_DIE(cudaEventCreate(&start));
    CUDA_OR_DIE(cudaEventRecord(start, NULL));
    for (int i = 0; i < ITERATIONS; ++i)
        CUBLAS_OR_DIE(cublasGemmEx(hand, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &α, A, CUDA_R_16F, lda,
                                   B, CUDA_R_16F, ldb, &β, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F,
                                   CUBLAS_GEMM_DEFAULT));
    float msecTotal = 0;
    CUDA_OR_DIE(cudaEventRecord(stop, NULL));
    CUDA_OR_DIE(cudaEventSynchronize(stop));
    CUDA_OR_DIE(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("%8lld µs cublasGemmExHHS(%d, %d, %d)\n",
           (long long)std::ceil(msecTotal / ITERATIONS * 1000), m, n, k);
    CUDA_OR_DIE(cudaEventDestroy(start));
    CUDA_OR_DIE(cudaEventDestroy(stop));

    CUDA_OR_DIE(cudaFree(G));
    CUDA_OR_DIE(cudaFree(C));
    CUDA_OR_DIE(cudaFree(B));
    CUDA_OR_DIE(cudaFree(A));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    CUBLAS_OR_DIE(cublasDestroy(hand));
}

void checkTinyblasWorksHHS() {

    tinyblasHandle_t hand;
    TINYBLAS_OR_DIE(tinyblasCreate(&hand));

    cudaStream_t stream;
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    TINYBLAS_OR_DIE(tinyblasSetStream(hand, stream));

    half *A, *B;
    float *C, *G;
    long long maxdim = 4096;
    CUDA_OR_DIE(cudaMallocManaged(&A, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&B, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&C, maxdim * maxdim * sizeof(float)));
    CUDA_OR_DIE(cudaMallocManaged(&G, maxdim * maxdim * sizeof(float)));

    std::atomic_llong t = ATOMIC_VAR_INIT(0);
    for (int ai = 0; ai < ARRAYLEN(kAlphas); ++ai)
        for (int bi = 0; bi < ARRAYLEN(kBetas); ++bi)
            for (int mi = 0; mi < ARRAYLEN(kDims); ++mi)
                for (int ni = 0; ni < ARRAYLEN(kDims); ++ni)
                    for (int ki = 0; ki < ARRAYLEN(kDims); ++ki) {
                        float α = kAlphas[ai];
                        float β = kBetas[bi];
                        int m = kDims[mi];
                        int n = kDims[ARRAYLEN(kDims) - 1 - ni];
                        int k = kDims[ki];
                        int lda = k;
                        int ldb = k;
                        int ldc = m;
                        char name[192];
                        snprintf(name, sizeof(name), "testing %4d %4d %4d α=%g β=%g", m, n, k, α,
                                 β);
                        if (t++ % 7 == 0)
                            fprintf(stderr, "%s\r", name);
                        is_self_testing = name;
                        fill(m, k, A, lda);
                        fill(k, n, B, ldb);
                        broadcast(m, n, G, n, TOMBSTONE);
                        broadcast(m, n, C, n, TOMBSTONE);
                        dgemm(true, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
                        TINYBLAS_OR_DIE(tinyblasGemmEx(
                            hand, TINYBLAS_OP_T, TINYBLAS_OP_N, m, n, k, &α, A, TINYBLAS_R_16F, lda,
                            B, TINYBLAS_R_16F, ldb, &β, C, TINYBLAS_R_32F, ldc,
                            TINYBLAS_COMPUTE_32F, TINYBLAS_GEMM_DEFAULT));
                        CUDA_OR_DIE(cudaStreamSynchronize(stream));
                        CHECK(PRECISION_F32, m, n, G, ldc, C, ldc);
                        is_self_testing = 0;
                    }

    float α = 1;
    float β = 0;
    int m = 4096;
    int n = 1024;
    int k = 577;
    int lda = k;
    int ldb = k;
    int ldc = m;
    fill(k, m, A, lda);
    fill(k, n, B, ldb);
    fill(m, n, C, ldc);
    fill(m, n, G, ldc);
    broadcast(m, n, G, n, TOMBSTONE);
    broadcast(m, n, C, n, TOMBSTONE);
    dgemm(true, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
    TINYBLAS_OR_DIE(tinyblasGemmEx(
        hand, TINYBLAS_OP_T, TINYBLAS_OP_N, m, n, k, &α, A, TINYBLAS_R_16F, lda, B, TINYBLAS_R_16F,
        ldb, &β, C, TINYBLAS_R_32F, ldc, TINYBLAS_COMPUTE_32F, TINYBLAS_GEMM_DEFAULT));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CHECK(PRECISION_F32, m, n, G, ldc, C, ldc);
    cudaEvent_t start, stop;
    CUDA_OR_DIE(cudaEventCreate(&stop));
    CUDA_OR_DIE(cudaEventCreate(&start));
    CUDA_OR_DIE(cudaEventRecord(start, NULL));
    for (int i = 0; i < ITERATIONS; ++i)
        TINYBLAS_OR_DIE(tinyblasGemmEx(hand, TINYBLAS_OP_T, TINYBLAS_OP_N, m, n, k, &α, A,
                                       TINYBLAS_R_16F, lda, B, TINYBLAS_R_16F, ldb, &β, C,
                                       TINYBLAS_R_32F, ldc, TINYBLAS_COMPUTE_32F,
                                       TINYBLAS_GEMM_DEFAULT));
    float msecTotal = 0;
    CUDA_OR_DIE(cudaEventRecord(stop, NULL));
    CUDA_OR_DIE(cudaEventSynchronize(stop));
    CUDA_OR_DIE(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("%8lld µs tinyblasGemmExHHS(%d, %d, %d)\n",
           (long long)std::ceil(msecTotal / ITERATIONS * 1000), m, n, k);
    CUDA_OR_DIE(cudaEventDestroy(start));
    CUDA_OR_DIE(cudaEventDestroy(stop));

    CUDA_OR_DIE(cudaFree(G));
    CUDA_OR_DIE(cudaFree(C));
    CUDA_OR_DIE(cudaFree(B));
    CUDA_OR_DIE(cudaFree(A));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    TINYBLAS_OR_DIE(tinyblasDestroy(hand));
}

void checkCublasWorksHHH() {

    cublasHandle_t hand;
    CUBLAS_OR_DIE(cublasCreate(&hand));

    cudaStream_t stream;
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    CUBLAS_OR_DIE(cublasSetStream(hand, stream));

    half *A, *B, *C, *G;
    long long maxdim = 32000;
    CUDA_OR_DIE(cudaMallocManaged(&A, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&B, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&C, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&G, maxdim * maxdim * sizeof(half)));

    std::atomic_llong t = ATOMIC_VAR_INIT(0);
    for (int ai = 0; ai < ARRAYLEN(kAlphas); ++ai)
        for (int bi = 0; bi < ARRAYLEN(kBetas); ++bi)
            for (int mi = 0; mi < ARRAYLEN(kDims); ++mi)
                for (int ni = 0; ni < ARRAYLEN(kDims); ++ni)
                    for (int ki = 0; ki < ARRAYLEN(kDims); ++ki) {
                        half α = kAlphas[ai];
                        half β = kBetas[bi];
                        int m = kDims[mi];
                        int n = kDims[ARRAYLEN(kDims) - 1 - ni];
                        int k = kDims[ki];
                        int lda = m;
                        int ldb = k;
                        int ldc = m;
                        char name[192];
                        snprintf(name, sizeof(name), "testing %4d %4d %4d α=%g β=%g", m, n, k,
                                 __half2float(α), __half2float(β));
                        if (t++ % 7 == 0)
                            fprintf(stderr, "%s\r", name);
                        is_self_testing = name;
                        fill(m, k, A, lda);
                        fill(k, n, B, ldb);
                        broadcast(m, n, G, n, TOMBSTONE);
                        broadcast(m, n, C, n, TOMBSTONE);
                        dgemm(false, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
                        CUBLAS_OR_DIE(cublasGemmEx(hand, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &α, A,
                                                   CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &β, C,
                                                   CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F,
                                                   CUBLAS_GEMM_DEFAULT));
                        CUDA_OR_DIE(cudaStreamSynchronize(stream));
                        CHECK(PRECISION_F16, m, n, G, ldc, C, ldc);
                        is_self_testing = 0;
                    }

    half α = 1;
    half β = 0;
    int m = 4096;
    int n = 1024;
    int k = 577;
    int lda = k;
    int ldb = k;
    int ldc = m;
    fill(k, m, A, lda);
    fill(k, n, B, ldb);
    fill(m, n, C, ldc);
    fill(m, n, G, ldc);
    broadcast(m, n, G, n, TOMBSTONE);
    broadcast(m, n, C, n, TOMBSTONE);
    dgemm(true, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
    CUBLAS_OR_DIE(cublasGemmEx(hand, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &α, A, CUDA_R_16F, lda, B,
                               CUDA_R_16F, ldb, &β, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F,
                               CUBLAS_GEMM_DEFAULT));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CHECK(PRECISION_F16, m, n, G, ldc, C, ldc);
    cudaEvent_t start, stop;
    CUDA_OR_DIE(cudaEventCreate(&stop));
    CUDA_OR_DIE(cudaEventCreate(&start));
    CUDA_OR_DIE(cudaEventRecord(start, NULL));
    for (int i = 0; i < ITERATIONS; ++i)
        CUBLAS_OR_DIE(cublasGemmEx(hand, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &α, A, CUDA_R_16F, lda,
                                   B, CUDA_R_16F, ldb, &β, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F,
                                   CUBLAS_GEMM_DEFAULT));
    float msecTotal = 0;
    CUDA_OR_DIE(cudaEventRecord(stop, NULL));
    CUDA_OR_DIE(cudaEventSynchronize(stop));
    CUDA_OR_DIE(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("%8lld µs cublasGemmExHHH(%d, %d, %d)\n",
           (long long)std::ceil(msecTotal / ITERATIONS * 1000), m, n, k);
    CUDA_OR_DIE(cudaEventDestroy(start));
    CUDA_OR_DIE(cudaEventDestroy(stop));

    CUDA_OR_DIE(cudaFree(G));
    CUDA_OR_DIE(cudaFree(C));
    CUDA_OR_DIE(cudaFree(B));
    CUDA_OR_DIE(cudaFree(A));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    CUBLAS_OR_DIE(cublasDestroy(hand));
}

void checkTinyblasWorksHHH() {

    tinyblasHandle_t hand;
    TINYBLAS_OR_DIE(tinyblasCreate(&hand));

    cudaStream_t stream;
    CUDA_OR_DIE(cudaStreamCreate(&stream));
    TINYBLAS_OR_DIE(tinyblasSetStream(hand, stream));

    half *A, *B, *C, *G;
    long long maxdim = 32000;
    CUDA_OR_DIE(cudaMallocManaged(&A, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&B, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&C, maxdim * maxdim * sizeof(half)));
    CUDA_OR_DIE(cudaMallocManaged(&G, maxdim * maxdim * sizeof(half)));

    std::atomic_llong t = ATOMIC_VAR_INIT(0);
    for (int ai = 0; ai < ARRAYLEN(kAlphas); ++ai)
        for (int bi = 0; bi < ARRAYLEN(kBetas); ++bi)
            for (int mi = 0; mi < ARRAYLEN(kDims); ++mi)
                for (int ni = 0; ni < ARRAYLEN(kDims); ++ni)
                    for (int ki = 0; ki < ARRAYLEN(kDims); ++ki) {
                        half α = kAlphas[ai];
                        half β = kBetas[bi];
                        int m = kDims[mi];
                        int n = kDims[ARRAYLEN(kDims) - 1 - ni];
                        int k = kDims[ki];
                        int lda = m;
                        int ldb = k;
                        int ldc = m;
                        char name[192];
                        snprintf(name, sizeof(name), "testing %4d %4d %4d α=%g β=%g", m, n, k,
                                 __half2float(α), __half2float(β));
                        if (t++ % 7 == 0)
                            fprintf(stderr, "%s\r", name);
                        is_self_testing = name;
                        fill(m, k, A, lda);
                        fill(k, n, B, ldb);
                        broadcast(m, n, G, n, TOMBSTONE);
                        broadcast(m, n, C, n, TOMBSTONE);
                        dgemm(false, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
                        TINYBLAS_OR_DIE(tinyblasGemmEx(
                            hand, TINYBLAS_OP_N, TINYBLAS_OP_N, m, n, k, &α, A, TINYBLAS_R_16F, lda,
                            B, TINYBLAS_R_16F, ldb, &β, C, TINYBLAS_R_16F, ldc,
                            TINYBLAS_COMPUTE_16F, TINYBLAS_GEMM_DEFAULT));
                        CUDA_OR_DIE(cudaStreamSynchronize(stream));
                        CHECK(PRECISION_F16, m, n, G, ldc, C, ldc);
                        is_self_testing = 0;
                    }

    half α = 1;
    half β = 0;
    int m = 4096;
    int n = 1024;
    int k = 577;
    int lda = k;
    int ldb = k;
    int ldc = m;
    fill(k, m, A, lda);
    fill(k, n, B, ldb);
    fill(m, n, C, ldc);
    fill(m, n, G, ldc);
    broadcast(m, n, G, n, TOMBSTONE);
    broadcast(m, n, C, n, TOMBSTONE);
    dgemm(true, false, m, n, k, α, A, lda, B, ldb, β, G, ldc);
    TINYBLAS_OR_DIE(tinyblasGemmEx(
        hand, TINYBLAS_OP_T, TINYBLAS_OP_N, m, n, k, &α, A, TINYBLAS_R_16F, lda, B, TINYBLAS_R_16F,
        ldb, &β, C, TINYBLAS_R_16F, ldc, TINYBLAS_COMPUTE_16F, TINYBLAS_GEMM_DEFAULT));
    CUDA_OR_DIE(cudaStreamSynchronize(stream));
    CHECK(PRECISION_F16, m, n, G, ldc, C, ldc);
    cudaEvent_t start, stop;
    CUDA_OR_DIE(cudaEventCreate(&stop));
    CUDA_OR_DIE(cudaEventCreate(&start));
    CUDA_OR_DIE(cudaEventRecord(start, NULL));
    for (int i = 0; i < ITERATIONS; ++i)
        TINYBLAS_OR_DIE(tinyblasGemmEx(hand, TINYBLAS_OP_T, TINYBLAS_OP_N, m, n, k, &α, A,
                                       TINYBLAS_R_16F, lda, B, TINYBLAS_R_16F, ldb, &β, C,
                                       TINYBLAS_R_16F, ldc, TINYBLAS_COMPUTE_16F,
                                       TINYBLAS_GEMM_DEFAULT));
    float msecTotal = 0;
    CUDA_OR_DIE(cudaEventRecord(stop, NULL));
    CUDA_OR_DIE(cudaEventSynchronize(stop));
    CUDA_OR_DIE(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("%8lld µs tinyblasGemmExHHH(%d, %d, %d)\n",
           (long long)std::ceil(msecTotal / ITERATIONS * 1000), m, n, k);
    CUDA_OR_DIE(cudaEventDestroy(start));
    CUDA_OR_DIE(cudaEventDestroy(stop));

    CUDA_OR_DIE(cudaFree(G));
    CUDA_OR_DIE(cudaFree(C));
    CUDA_OR_DIE(cudaFree(B));
    CUDA_OR_DIE(cudaFree(A));
    CUDA_OR_DIE(cudaStreamDestroy(stream));
    TINYBLAS_OR_DIE(tinyblasDestroy(hand));
}

int main(int argc, char *argv[]) {
    RUN(checkFloat01());
    RUN(checkCublasWorksSSS());
    RUN(checkTinyblasWorksSSS());
    RUN(checkCublasWorksHHS());
    RUN(checkTinyblasWorksHHS());
    RUN(checkCublasWorksHHH());
    RUN(checkTinyblasWorksHHH());
}
