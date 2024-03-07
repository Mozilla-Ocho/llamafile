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

#include <algorithm>
#include <cstdlib>
#include <type_traits>

#ifndef __HIP__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaGetLastError hipGetLastError
#endif

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//

#define THREAD_COUNT ((BM * BN) / (TM * TN))
#define KERNEL __launch_bounds__(THREAD_COUNT)
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

struct tinyblasContext {
    cudaStream_t stream;
};

tinyblasStatus_t tinyblasCreate(tinyblasHandle_t *out_handle) {
    tinyblasHandle_t handle;
    if ((handle = (tinyblasHandle_t)malloc(sizeof(struct tinyblasContext)))) {
        *out_handle = handle;
        return TINYBLAS_STATUS_SUCCESS;
    } else {
        return TINYBLAS_STATUS_ALLOC_FAILED;
    }
}

tinyblasStatus_t tinyblasDestroy(tinyblasHandle_t handle) {
    free(handle);
    return TINYBLAS_STATUS_SUCCESS;
}

tinyblasStatus_t tinyblasSetStream(tinyblasHandle_t handle, void *stream) {
    handle->stream = (cudaStream_t)stream;
    return TINYBLAS_STATUS_SUCCESS;
}

tinyblasStatus_t tinyblasGetStream(tinyblasHandle_t handle, void **out_stream) {
    *out_stream = handle->stream;
    return TINYBLAS_STATUS_SUCCESS;
}

tinyblasStatus_t tinyblasSgemm(tinyblasHandle_t handle, //
                               tinyblasOperation_t transa, //
                               tinyblasOperation_t transb, //
                               int m, int n, int k, //
                               const float *alpha, //
                               const float *A, int lda, //
                               const float *B, int ldb, //
                               const float *beta, //
                               float *C, int ldc) {
    return tinyblasGemmEx(handle, transa, transb, m, n, k, alpha, A, TINYBLAS_R_32F, lda, B,
                          TINYBLAS_R_32F, ldb, beta, C, TINYBLAS_R_32F, ldc, TINYBLAS_COMPUTE_32F,
                          TINYBLAS_GEMM_DEFAULT);
}

const char *tinyblasGetStatusString(tinyblasStatus_t err) {
    switch (err) {
    case TINYBLAS_STATUS_SUCCESS:
        return "Success";
    case TINYBLAS_STATUS_ALLOC_FAILED:
        return "Alloc failed";
    case TINYBLAS_STATUS_INVALID_VALUE:
        return "Invalid value";
    case TINYBLAS_STATUS_NOT_SUPPORTED:
        return "Not supported";
    case TINYBLAS_STATUS_EXECUTION_FAILED:
        return "Execution failed";
    case TINYBLAS_STATUS_DIMENSION_OVERFLOW:
        return "Dimension overflow";
    default:
        return "Unknown error";
    }
}

template <int BM, int BN, int TM, int TN, typename WORD, typename SRC, typename DST>
static __device__ void matmul_block2d(tinyblasOperation_t transa, tinyblasOperation_t transb, int m,
                                      int n, int k, WORD alpha, const SRC *A, int lda, const SRC *B,
                                      int ldb, WORD beta, DST *C, int ldc) {

    constexpr int BK = THREAD_COUNT;
    static_assert(BM % TM == 0, "can't divide work for threads");
    static_assert(BN % TN == 0, "can't divide work for threads");
    static_assert(BM > 0 && BN > 0 && BK > 0 && TM > 0 && TN > 0,
                  "one of the constexpr configuration values was non-positive");
    static_assert((BK * BM * sizeof(SRC)) + (BK * BN * sizeof(SRC)) <= 65536,
                  "you're almost almost certainly using too much shared memory");

    const int th = threadIdx.x;
    const int ii = blockIdx.x * BM;
    const int jj = blockIdx.y * BN;
    const int ti = th / (BN / TN) * TM;
    const int tj = th % (BN / TN) * TN;

    __shared__ SRC As[BK * BM];
    __shared__ SRC Bs[BK * BN];

    WORD At[TM];
    WORD Bt[TN];
    WORD Cs[TM * TN] = {0};

    for (int ll = 0; ll < k; ll += BK) {

        for (int i = 0; i < BM; ++i)
            As[BM * th + i] = 0;
        for (int i = 0; i < BM && ll + th < k && ii + i < m; ++i)
            As[BM * th + i] = A[transa ? lda * (ii + i) + (ll + th) : lda * (ll + th) + (ii + i)];

        for (int j = 0; j < BN; ++j)
            Bs[BN * th + j] = 0;
        for (int j = 0; j < BN && ll + th < k && jj + j < n; ++j)
            Bs[BN * th + j] = B[transb ? ldb * (ll + th) + (jj + j) : ldb * (jj + j) + (ll + th)];

        __syncthreads();

        for (int l = 0; l < BK; ++l) {
            for (int j = 0; j < TM; ++j)
                At[j] = As[BM * l + ti + j];
            for (int h = 0; h < TN; ++h)
                Bt[h] = Bs[BN * l + tj + h];
            for (int j = 0; j < TM; ++j) {
                WORD a = At[j];
                for (int h = 0; h < TN; ++h) {
                    WORD b = Bt[h];
                    Cs[TN * j + h] += a * b;
                }
            }
        }

        __syncthreads();
    }

    if (alpha != (WORD)1)
        for (int i = 0; i < TM * TN; ++i)
            Cs[i] *= alpha;

    for (int j = 0; j < TN && jj + tj + j < n; ++j)
        for (int i = 0; i < TM && ii + ti + i < m; ++i)
            if (beta) {
                WORD c = C[ldc * (jj + tj + j) + (ii + ti + i)];
                C[ldc * (jj + tj + j) + (ii + ti + i)] = c * beta + Cs[TN * i + j];
            } else {
                C[ldc * (jj + tj + j) + (ii + ti + i)] = Cs[TN * i + j];
            }
}

template <int BM, int BN, int TM, int TN, typename WORD, typename SRC, typename DST>
static __global__ void KERNEL tinyblasGE_entry(tinyblasOperation_t transa,
                                               tinyblasOperation_t transb, int m, int n, int k,
                                               WORD alpha, const SRC *A, int lda, const SRC *B,
                                               int ldb, WORD beta, DST *C, int ldc) {
    matmul_block2d<BM, BN, TM, TN>(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename WORD, typename SRC, typename DST>
static tinyblasStatus_t tinyblasGE_launch(tinyblasHandle_t handle, tinyblasOperation_t transa,
                                          tinyblasOperation_t transb, int m, int n, int k,
                                          WORD alpha, const SRC *A, int lda, const SRC *B, int ldb,
                                          WORD beta, DST *C, int ldc) {
    constexpr int BM = 48;
    constexpr int BN = 32;
    constexpr int TM = 3;
    constexpr int TN = 8;
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN));
    tinyblasGE_entry<BM, BN, TM, TN><<<maxblocks, THREAD_COUNT, 0, handle->stream>>>(
        transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (cudaGetLastError() != cudaSuccess)
        return TINYBLAS_STATUS_EXECUTION_FAILED;
    return TINYBLAS_STATUS_SUCCESS;
}

tinyblasStatus_t tinyblasGemmEx(tinyblasHandle_t handle, //
                                tinyblasOperation_t transa, //
                                tinyblasOperation_t transb, //
                                int m, int n, int k, //
                                const void *alpha, //
                                const void *A, tinyblasDataType_t Atype, int lda, //
                                const void *B, tinyblasDataType_t Btype, int ldb, //
                                const void *beta, //
                                void *C, tinyblasDataType_t Ctype, int ldc, //
                                tinyblasComputeType_t computeType, //
                                tinyblasGemmAlgo_t algo) {

    if (m < 0 || n < 0 || k < 0)
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, transa ? k : m))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (ldb < std::max(1, transb ? n : k))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (ldc < std::max(1, m))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (1ll * lda * ((transa ? k : m) - 1) + ((transa ? m : k) - 1) > INT_MAX)
        return TINYBLAS_STATUS_DIMENSION_OVERFLOW;
    if (1ll * ldb * ((transb ? n : k) - 1) + ((transb ? k : n) - 1) > INT_MAX)
        return TINYBLAS_STATUS_DIMENSION_OVERFLOW;
    if (1ll * ldc * (n - 1) + (m - 1) > INT_MAX)
        return TINYBLAS_STATUS_DIMENSION_OVERFLOW;
    if (Atype != Btype)
        return TINYBLAS_STATUS_NOT_SUPPORTED;

    switch (Atype) {
    case TINYBLAS_R_16F:
        switch (Ctype) {
        case TINYBLAS_R_16F:
            switch (computeType) {
            case TINYBLAS_COMPUTE_16F:
                return tinyblasGE_launch(
                    handle, transa, transb, m, n, k, (float)*(const half *)alpha, (const half *)A,
                    lda, (const half *)B, ldb, (float)*(const half *)beta, (half *)C, ldc);
            case TINYBLAS_COMPUTE_32F:
                return tinyblasGE_launch(handle, transa, transb, m, n, k, *(const float *)alpha,
                                         (const half *)A, lda, (const half *)B, ldb,
                                         *(const float *)beta, (half *)C, ldc);
            default:
                return TINYBLAS_STATUS_INVALID_VALUE;
            }
        case TINYBLAS_R_32F:
            switch (computeType) {
            case TINYBLAS_COMPUTE_16F:
                return TINYBLAS_STATUS_NOT_SUPPORTED;
            case TINYBLAS_COMPUTE_32F:
                return tinyblasGE_launch(handle, transa, transb, m, n, k, *(const float *)alpha,
                                         (const half *)A, lda, (const half *)B, ldb,
                                         *(const float *)beta, (float *)C, ldc);
            default:
                return TINYBLAS_STATUS_INVALID_VALUE;
            }
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    case TINYBLAS_R_32F:
        switch (Ctype) {
        case TINYBLAS_R_16F:
            return TINYBLAS_STATUS_NOT_SUPPORTED;
        case TINYBLAS_R_32F:
            switch (computeType) {
            case TINYBLAS_COMPUTE_16F:
                return TINYBLAS_STATUS_NOT_SUPPORTED;
            case TINYBLAS_COMPUTE_32F:
                return tinyblasGE_launch(handle, transa, transb, m, n, k, *(const float *)alpha,
                                         (const float *)A, lda, (const float *)B, ldb,
                                         *(const float *)beta, (float *)C, ldc);
            default:
                return TINYBLAS_STATUS_INVALID_VALUE;
            }
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    default:
        return TINYBLAS_STATUS_INVALID_VALUE;
    }
}

template <int BM, int BN, int TM, int TN, typename WORD, typename SRC, typename DST>
static __global__ void KERNEL tinyblasGBE_entry(tinyblasOperation_t transa,
                                                tinyblasOperation_t transb, int m, int n, int k,
                                                WORD alpha, const SRC *const Aarray[], int lda,
                                                const SRC *const Barray[], int ldb, WORD beta,
                                                DST *const Carray[], int ldc, int batchCount) {
    for (int z = blockIdx.z; z < batchCount; z += gridDim.z)
        matmul_block2d<BM, BN, TM, TN>(transa, transb, m, n, k, alpha, Aarray[z], lda, Barray[z],
                                       ldb, beta, Carray[z], ldc);
}

template <typename WORD, typename SRC, typename DST>
static tinyblasStatus_t tinyblasGBE_launch(tinyblasHandle_t handle, tinyblasOperation_t transa,
                                           tinyblasOperation_t transb, int m, int n, int k,
                                           WORD alpha, const SRC *const *Aarray, int lda,
                                           const SRC *const *Barray, int ldb, WORD beta,
                                           DST *const *Carray, int ldc, int batchCount) {
    constexpr int BC = 32;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int TM = 2;
    constexpr int TN = 8;
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), BC);
    tinyblasGBE_entry<BM, BN, TM, TN><<<maxblocks, THREAD_COUNT, 0, handle->stream>>>(
        transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    if (cudaGetLastError() != cudaSuccess)
        return TINYBLAS_STATUS_EXECUTION_FAILED;
    return TINYBLAS_STATUS_SUCCESS;
}

tinyblasStatus_t tinyblasGemmBatchedEx(tinyblasHandle_t handle, //
                                       tinyblasOperation_t transa, //
                                       tinyblasOperation_t transb, //
                                       int m, int n, int k, //
                                       const void *alpha, //
                                       const void *const Aarray[], tinyblasDataType_t Atype,
                                       int lda, //
                                       const void *const Barray[], tinyblasDataType_t Btype,
                                       int ldb, //
                                       const void *beta, //
                                       void *const Carray[], tinyblasDataType_t Ctype, int ldc, //
                                       int batchCount, //
                                       tinyblasComputeType_t computeType, //
                                       tinyblasGemmAlgo_t algo) {

    if (m < 0 || n < 0 || k < 0)
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, transa ? k : m))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (ldb < std::max(1, transb ? n : k))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (ldc < std::max(1, m))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (1ll * lda * ((transa ? k : m) - 1) + ((transa ? m : k) - 1) > INT_MAX)
        return TINYBLAS_STATUS_DIMENSION_OVERFLOW;
    if (1ll * ldb * ((transb ? n : k) - 1) + ((transb ? k : n) - 1) > INT_MAX)
        return TINYBLAS_STATUS_DIMENSION_OVERFLOW;
    if (1ll * ldc * (n - 1) + (m - 1) > INT_MAX)
        return TINYBLAS_STATUS_DIMENSION_OVERFLOW;
    if (Atype != Btype)
        return TINYBLAS_STATUS_NOT_SUPPORTED;

    switch (Atype) {
    case TINYBLAS_R_16F:
        switch (Ctype) {
        case TINYBLAS_R_16F:
            switch (computeType) {
            case TINYBLAS_COMPUTE_16F:
                return tinyblasGBE_launch(
                    handle, transa, transb, m, n, k, (float)*(const half *)alpha,
                    (const half *const *)Aarray, lda, (const half *const *)Barray, ldb,
                    (float)*(const half *)beta, (half *const *)Carray, ldc, batchCount);
            case TINYBLAS_COMPUTE_32F:
                return tinyblasGBE_launch(handle, transa, transb, m, n, k, *(const float *)alpha,
                                          (const half *const *)Aarray, lda,
                                          (const half *const *)Barray, ldb, *(const float *)beta,
                                          (half *const *)Carray, ldc, batchCount);
            default:
                return TINYBLAS_STATUS_INVALID_VALUE;
            }
        case TINYBLAS_R_32F:
            switch (computeType) {
            case TINYBLAS_COMPUTE_16F:
                return TINYBLAS_STATUS_NOT_SUPPORTED;
            case TINYBLAS_COMPUTE_32F:
                return tinyblasGBE_launch(handle, transa, transb, m, n, k, *(const float *)alpha,
                                          (const half *const *)Aarray, lda,
                                          (const half *const *)Barray, ldb, *(const float *)beta,
                                          (float *const *)Carray, ldc, batchCount);
            default:
                return TINYBLAS_STATUS_INVALID_VALUE;
            }
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    case TINYBLAS_R_32F:
        switch (Ctype) {
        case TINYBLAS_R_16F:
            return TINYBLAS_STATUS_NOT_SUPPORTED;
        case TINYBLAS_R_32F:
            switch (computeType) {
            case TINYBLAS_COMPUTE_16F:
                return TINYBLAS_STATUS_NOT_SUPPORTED;
            case TINYBLAS_COMPUTE_32F:
                return tinyblasGBE_launch(handle, transa, transb, m, n, k, *(const float *)alpha,
                                          (const float *const *)Aarray, lda,
                                          (const float *const *)Barray, ldb, *(const float *)beta,
                                          (float *const *)Carray, ldc, batchCount);
            default:
                return TINYBLAS_STATUS_INVALID_VALUE;
            }
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    default:
        return TINYBLAS_STATUS_INVALID_VALUE;
    }
}

template <int BM, int BN, int TM, int TN, typename SRC, typename DST, typename WORD>
static __global__ void KERNEL tinyblasGSBE_entry(tinyblasOperation_t transa,
                                                 tinyblasOperation_t transb, int m, int n, int k,
                                                 WORD alpha, const SRC *A, int lda,
                                                 long long strideA, const SRC *B, int ldb,
                                                 long long strideB, WORD beta, DST *C, int ldc,
                                                 long long strideC, int batchCount) {
    matmul_block2d<BM, BN, TM, TN>(transa, transb, m, n, k, alpha, A + blockIdx.z * strideA, lda,
                                   B + blockIdx.z * strideB, ldb, beta, C + blockIdx.z * strideC,
                                   ldc);
}

template <typename WORD, typename SRC, typename DST>
static tinyblasStatus_t tinyblasGSBE_launch(tinyblasHandle_t handle, tinyblasOperation_t transa,
                                            tinyblasOperation_t transb, int m, int n, int k,
                                            WORD alpha, const SRC *A, int lda, long long strideA,
                                            const SRC *B, int ldb, long long strideB, WORD beta,
                                            DST *C, int ldc, long long strideC, int batchCount) {
    constexpr int BC = 32;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int TM = 2;
    constexpr int TN = 8;
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), BC);
    tinyblasGSBE_entry<BM, BN, TM, TN><<<maxblocks, THREAD_COUNT, 0, handle->stream>>>(
        transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC,
        batchCount);
    if (cudaGetLastError() != cudaSuccess)
        return TINYBLAS_STATUS_EXECUTION_FAILED;
    return TINYBLAS_STATUS_SUCCESS;
}

tinyblasStatus_t tinyblasGemmStridedBatchedEx(tinyblasHandle_t handle, //
                                              tinyblasOperation_t transa, //
                                              tinyblasOperation_t transb, //
                                              int m, int n, int k, //
                                              const void *alpha, //
                                              const void *A, tinyblasDataType_t Atype, int lda,
                                              long long strideA, //
                                              const void *B, tinyblasDataType_t Btype, int ldb,
                                              long long strideB, //
                                              const void *beta, //
                                              void *C, tinyblasDataType_t Ctype, int ldc,
                                              long long strideC, //
                                              int batchCount, //
                                              tinyblasComputeType_t computeType, //
                                              tinyblasGemmAlgo_t algo) {

    if (m < 0 || n < 0 || k < 0)
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, transa ? k : m))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (ldb < std::max(1, transb ? n : k))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (ldc < std::max(1, m))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (1ll * lda * ((transa ? k : m) - 1) + ((transa ? m : k) - 1) > INT_MAX)
        return TINYBLAS_STATUS_DIMENSION_OVERFLOW;
    if (1ll * ldb * ((transb ? n : k) - 1) + ((transb ? k : n) - 1) > INT_MAX)
        return TINYBLAS_STATUS_DIMENSION_OVERFLOW;
    if (1ll * ldc * (n - 1) + (m - 1) > INT_MAX)
        return TINYBLAS_STATUS_DIMENSION_OVERFLOW;
    if (Atype != Btype)
        return TINYBLAS_STATUS_NOT_SUPPORTED;

    switch (Atype) {
    case TINYBLAS_R_16F:
        switch (Ctype) {
        case TINYBLAS_R_16F:
            switch (computeType) {
            case TINYBLAS_COMPUTE_16F:
                return tinyblasGSBE_launch(
                    handle, transa, transb, m, n, k, (float)*(const half *)alpha, (const half *)A,
                    lda, strideA, (const half *)B, ldb, strideB, (float)*(const half *)beta,
                    (half *)C, ldc, strideC, batchCount);
            case TINYBLAS_COMPUTE_32F:
                return tinyblasGSBE_launch(handle, transa, transb, m, n, k, *(const float *)alpha,
                                           (const half *)A, lda, strideA, (const half *)B, ldb,
                                           strideB, *(const float *)beta, (half *)C, ldc, strideC,
                                           batchCount);
            default:
                return TINYBLAS_STATUS_INVALID_VALUE;
            }
        case TINYBLAS_R_32F:
            switch (computeType) {
            case TINYBLAS_COMPUTE_16F:
                return TINYBLAS_STATUS_NOT_SUPPORTED;
            case TINYBLAS_COMPUTE_32F:
                return tinyblasGSBE_launch(handle, transa, transb, m, n, k, *(const float *)alpha,
                                           (const half *)A, lda, strideA, (const half *)B, ldb,
                                           strideB, *(const float *)beta, (float *)C, ldc, strideC,
                                           batchCount);
            default:
                return TINYBLAS_STATUS_INVALID_VALUE;
            }
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    case TINYBLAS_R_32F:
        switch (Ctype) {
        case TINYBLAS_R_16F:
            return TINYBLAS_STATUS_NOT_SUPPORTED;
        case TINYBLAS_R_32F:
            switch (computeType) {
            case TINYBLAS_COMPUTE_16F:
                return TINYBLAS_STATUS_NOT_SUPPORTED;
            case TINYBLAS_COMPUTE_32F:
                return tinyblasGSBE_launch(handle, transa, transb, m, n, k, *(const float *)alpha,
                                           (const float *)A, lda, strideA, (const float *)B, ldb,
                                           strideB, *(const float *)beta, (float *)C, ldc, strideC,
                                           batchCount);
            default:
                return TINYBLAS_STATUS_INVALID_VALUE;
            }
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    default:
        return TINYBLAS_STATUS_INVALID_VALUE;
    }
}
