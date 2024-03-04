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

#ifdef __NVCC__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define tinyblasStream_t cudaStream_t
#else
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define tinyblasStream_t hipStream_t
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

#define SHARELEN (BK * BM + BK * BN)
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

struct tinyblasContext {
    tinyblasStream_t stream;
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
    handle->stream = (tinyblasStream_t)stream;
    return TINYBLAS_STATUS_SUCCESS;
}

tinyblasStatus_t tinyblasGetStream(tinyblasHandle_t handle, void **out_stream) {
    *out_stream = handle->stream;
    return TINYBLAS_STATUS_SUCCESS;
}

template <int BM, int BN, int BK, int TM, int TN, typename WORD, typename SRC, typename DST>
static __device__ void matmul_block2d(tinyblasOperation_t transa, tinyblasOperation_t transb, int m,
                                      int n, int k, int x, int y, WORD alpha, const SRC *A, int lda,
                                      const SRC *B, int ldb, WORD beta, DST *C, int ldc) {
    static_assert((BM % TM == 0) && (BN % TN == 0), "can't divide work for threads");
    static_assert((BM * BN) <= (BM * BK) + (BK * BN),
                  "didn't allocate enough shared mem for threads");
    const int ii0 = threadIdx.x / (BN / TN); // {0, ..., (BM/TM) - 1}
    const int ii1 = threadIdx.x % (BN / TN); // {0, ..., (BN/TN) - 1}
    extern __shared__ char svals[]; // shared across all threads in a block
    SRC *As /*[BK][BM]*/ = (SRC *)svals;
    SRC *Bs /*[BK][BN]*/ = (SRC *)svals + BM * BK;

    WORD Cs[TM][TN] = {0};
    SRC At[TM], Bt[TN];
    int i, h, j, l, blob;

    for (blob = 0; blob < k; blob += BK) {
        for (i = threadIdx.x; i < BK; i += blockDim.x)
            for (j = 0; j < BM; ++j)
                As[BM * i + j] = 0;
        for (i = threadIdx.x; i < BK && blob + i < k; i += blockDim.x)
            for (j = 0; j < BM && x + j < m; ++j)
                As[BM * i + j] = transa ? A[1ll * lda * (x + j) + (blob + i)]
                                        : A[1ll * lda * (blob + i) + (x + j)];
        for (i = threadIdx.x; i < BK; i += blockDim.x)
            for (j = 0; j < BN; ++j)
                Bs[BN * i + j] = 0;
        for (i = threadIdx.x; i < BK && blob + i < k; i += blockDim.x)
            for (j = 0; j < BN && y + j < n; ++j)
                Bs[BN * i + j] = transb ? B[1ll * ldb * (blob + i) + (y + j)]
                                        : B[1ll * ldb * (y + j) + (blob + i)];
        __syncthreads();

        for (l = 0; l < BK; ++l) {
            for (j = 0; j < TM; ++j)
                At[j] = As[BM * l + TM * ii0 + j];
            for (h = 0; h < TN; ++h)
                Bt[h] = Bs[BN * l + TN * ii1 + h];
            for (j = 0; j < TM; ++j) {
                WORD a = At[j];
                for (h = 0; h < TN; ++h) {
                    WORD b = Bt[h];
                    Cs[j][h] += a * b;
                }
            }
        }
        __syncthreads();
    }

    x += ii0 * TM;
    y += ii1 * TN;
    for (l = 0; l < TN && y + l < n; ++l)
        for (j = 0; j < TM && x + j < m; ++j)
            if (beta) {
                WORD c = C[1ll * ldc * (y + l) + (x + j)];
                C[1ll * ldc * (y + l) + (x + j)] = alpha * Cs[j][l] + beta * c;
            } else {
                C[1ll * ldc * (y + l) + (x + j)] = alpha * Cs[j][l];
            }
}

template <int BM, int BN, int BK, int TM, int TN>
static __global__ void tinyblasS_entry(tinyblasOperation_t transa, tinyblasOperation_t transb,
                                       int m, int n, int k, float alpha, const float *A, int lda,
                                       const float *B, int ldb, float beta, float *C, int ldc) {

    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;

    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-matrix of size TM * TN
    for (x = blockIdx.x * BM; x < m; x += jump1)
        for (y = blockIdx.y * BN; y < n; y += jump2)
            matmul_block2d<BM, BN, BK, TM, TN, float>(transa, transb, m, n, k, x, y, alpha, A, lda,
                                                      B, ldb, beta, C, ldc);
}

template <int BM, int BN, int BK, int TM, int TN>
static void tinyblasS_wrapper(tinyblasHandle_t handle, tinyblasOperation_t transa,
                              tinyblasOperation_t transb, int m, int n, int k, const float alpha,
                              const float *A, int lda, const float *B, int ldb, const float beta,
                              float *C, int ldc) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int sharedmem = sizeof(float) * SHARELEN;
    int maxthreads = (BM * BN) / (TM * TN);
    tinyblasS_entry<BM, BN, BK, TM, TN><<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
        transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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
    tinyblasS_wrapper<32, 32, 16, 2, 8>(handle, transa, transb, m, n, k, *alpha, A, lda, B, ldb,
                                        *beta, C, ldc);
    return TINYBLAS_STATUS_SUCCESS;
}

static bool is_valid(tinyblasOperation_t transa, //
                     tinyblasOperation_t transb, //
                     int m, int n, int k, //
                     int lda, int ldb, int ldc) {
    return m >= 0 && //
           n >= 0 && //
           k >= 0 && //
           lda >= std::max(1, transa ? k : m) && //
           ldb >= std::max(1, transb ? n : k) && //
           ldc >= std::max(1, m);
}

static bool is_supported(tinyblasDatatype_t Atype, tinyblasDatatype_t Btype) {
    return Atype == TINYBLAS_R_16F && //
           Btype == TINYBLAS_R_16F;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
template <int BM, int BN, int BK, int TM, int TN, typename SRC, typename DST, typename T>
static __global__ void tinyblasGE_entry(tinyblasOperation_t transa, tinyblasOperation_t transb,
                                        int m, int n, int k, T alpha, const SRC *A, int lda,
                                        const SRC *B, int ldb, T beta, DST *C, int ldc) {

    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;

    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    // each block handles a sub-matrix of C, of size BM * BN
    for (x = blockIdx.x * BM; x < m; x += jump1)
        for (y = blockIdx.y * BN; y < n; y += jump2)
            matmul_block2d<BM, BN, BK, TM, TN, float>(transa, transb, m, n, k, x, y, alpha, A, lda,
                                                      B, ldb, beta, C, ldc);
}

template <int BM, int BN, int BK, int TM, int TN, typename SRC>
static tinyblasStatus_t
tinyblasGE_wrapper(tinyblasHandle_t handle, tinyblasOperation_t transa, tinyblasOperation_t transb,
                   int m, int n, int k, const void *pAlpha, const SRC *A, int lda, const SRC *B,
                   int ldb, const void *pBeta, void *C, tinyblasDatatype_t Ctype, int ldc,
                   tinyblasComputeType_t computeType) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int sharedmem = sizeof(SRC) * SHARELEN;
    int maxthreads = (BM * BN) / (TM * TN);
    switch (Ctype) {
    case TINYBLAS_R_16F:
        switch (computeType) {
        case TINYBLAS_COMPUTE_16F:
            tinyblasGE_entry<BM, BN, BK, TM, TN>
                <<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
                    transa, transb, m, n, k, *(const half *)pAlpha, A, lda, B, ldb,
                    *(const half *)pBeta, (half *)C, ldc);
            return TINYBLAS_STATUS_SUCCESS;
        case TINYBLAS_COMPUTE_32F:
            tinyblasGE_entry<BM, BN, BK, TM, TN>
                <<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
                    transa, transb, m, n, k, *(const float *)pAlpha, A, lda, B, ldb,
                    *(const float *)pBeta, (half *)C, ldc);
            return TINYBLAS_STATUS_SUCCESS;
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    case TINYBLAS_R_32F:
        switch (computeType) {
        case TINYBLAS_COMPUTE_16F:
            return TINYBLAS_STATUS_NOT_SUPPORTED;
        case TINYBLAS_COMPUTE_32F:
            tinyblasGE_entry<BM, BN, BK, TM, TN>
                <<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
                    transa, transb, m, n, k, *(const float *)pAlpha, A, lda, B, ldb,
                    *(const float *)pBeta, (float *)C, ldc);
            return TINYBLAS_STATUS_SUCCESS;
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    default:
        return TINYBLAS_STATUS_INVALID_VALUE;
    }
}

tinyblasStatus_t tinyblasGemmEx(tinyblasHandle_t handle, //
                                tinyblasOperation_t transa, //
                                tinyblasOperation_t transb, //
                                int m, int n, int k, //
                                const void *pAlpha, //
                                const void *A, tinyblasDatatype_t Atype, int lda, //
                                const void *B, tinyblasDatatype_t Btype, int ldb, //
                                const void *pBeta, //
                                void *C, tinyblasDatatype_t Ctype, int ldc, //
                                tinyblasComputeType_t computeType, //
                                tinyblasGemmAlgo_t algo) {
    if (!is_valid(transa, transb, m, n, k, lda, ldb, ldc))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (!is_supported(Atype, Btype))
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    return tinyblasGE_wrapper<48, 32, 64, 3, 8>(handle, transa, transb, m, n, k, pAlpha,
                                                (const half *)A, lda, (const half *)B, ldb, pBeta,
                                                C, Ctype, ldc, computeType);
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex
template <int BM, int BN, int BK, int TM, int TN, typename SRC, typename DST, typename T>
static __global__ void tinyblasGBE_entry(tinyblasOperation_t transa, tinyblasOperation_t transb,
                                         int m, int n, int k, T alpha, const SRC *const Aarray[],
                                         int lda, const SRC *const Barray[], int ldb, T beta,
                                         DST *const Carray[], int ldc, int batchCount) {

    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;

    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    int z = blockIdx.z;
    const int jump3 = gridDim.z;

    // each block handles a sub-matrix of C, of size BM * BN
    for (z = blockIdx.z; z < batchCount; z += jump3)
        for (x = blockIdx.x * BM; x < m; x += jump1)
            for (y = blockIdx.y * BN; y < n; y += jump2)
                matmul_block2d<BM, BN, BK, TM, TN, float>(transa, transb, m, n, k, x, y, alpha,
                                                          Aarray[z], lda, Barray[z], ldb, beta,
                                                          Carray[z], ldc);
}

template <int BM, int BN, int BK, int TM, int TN, typename SRC>
static tinyblasStatus_t tinyblasGBE_wrapper(tinyblasHandle_t handle, tinyblasOperation_t transa,
                                            tinyblasOperation_t transb, int m, int n, int k,
                                            const void *pAlpha, const SRC *const Aarray[], int lda,
                                            const SRC *const Barray[], int ldb, const void *pBeta,
                                            void *const Carray[], tinyblasDatatype_t Ctype, int ldc,
                                            int batchCount, tinyblasComputeType_t computeType) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int sharedmem = sizeof(SRC) * SHARELEN;
    int maxthreads = (BM * BN) / (TM * TN);
    switch (Ctype) {
    case TINYBLAS_R_16F:
        switch (computeType) {
        case TINYBLAS_COMPUTE_16F:
            tinyblasGBE_entry<BM, BN, BK, TM, TN>
                <<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
                    transa, transb, m, n, k, *(const half *)pAlpha, Aarray, lda, Barray, ldb,
                    *(const half *)pBeta, (half *const *)Carray, ldc, batchCount);
            return TINYBLAS_STATUS_SUCCESS;
        case TINYBLAS_COMPUTE_32F:
            tinyblasGBE_entry<BM, BN, BK, TM, TN>
                <<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
                    transa, transb, m, n, k, *(const float *)pAlpha, Aarray, lda, Barray, ldb,
                    *(const float *)pBeta, (half *const *)Carray, ldc, batchCount);
            return TINYBLAS_STATUS_SUCCESS;
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    case TINYBLAS_R_32F:
        switch (computeType) {
        case TINYBLAS_COMPUTE_16F:
            return TINYBLAS_STATUS_NOT_SUPPORTED;
        case TINYBLAS_COMPUTE_32F:
            tinyblasGBE_entry<BM, BN, BK, TM, TN>
                <<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
                    transa, transb, m, n, k, *(const float *)pAlpha, Aarray, lda, Barray, ldb,
                    *(const float *)pBeta, (float *const *)Carray, ldc, batchCount);
            return TINYBLAS_STATUS_SUCCESS;
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    default:
        return TINYBLAS_STATUS_INVALID_VALUE;
    }
}

tinyblasStatus_t tinyblasGemmBatchedEx(tinyblasHandle_t handle, //
                                       tinyblasOperation_t transa, //
                                       tinyblasOperation_t transb, //
                                       int m, int n, int k, //
                                       const void *pAlpha, //
                                       const void *const Aarray[], tinyblasDatatype_t Atype,
                                       int lda, //
                                       const void *const Barray[], tinyblasDatatype_t Btype,
                                       int ldb, //
                                       const void *pBeta, //
                                       void *const Carray[], tinyblasDatatype_t Ctype, int ldc, //
                                       int batchCount, //
                                       tinyblasComputeType_t computeType, //
                                       tinyblasGemmAlgo_t algo) {
    if (!is_valid(transa, transb, m, n, k, lda, ldb, ldc))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (!is_supported(Atype, Btype))
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    return tinyblasGBE_wrapper<48, 32, 64, 3, 8>(handle, transa, transb, //
                                                 m, n, k, pAlpha, //
                                                 (const half *const *)Aarray, lda, //
                                                 (const half *const *)Barray, ldb, //
                                                 pBeta, Carray, Ctype, ldc, //
                                                 batchCount, computeType);
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
template <int BM, int BN, int BK, int TM, int TN, typename SRC, typename DST, typename T>
static __global__ void
tinyblasGSBE_entry(tinyblasOperation_t transa, tinyblasOperation_t transb, int m, int n, int k,
                   T alpha, const SRC *A, int lda, long long strideA, const SRC *B, int ldb,
                   long long strideB, T beta, DST *C, int ldc, long long strideC, int batchCount) {

    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;

    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    int z = blockIdx.z;
    const int jump3 = gridDim.z;

    // each block handles a sub-matrix of C, of size BM * BN
    for (z = blockIdx.z; z < batchCount; z += jump3)
        for (x = blockIdx.x * BM; x < m; x += jump1)
            for (y = blockIdx.y * BN; y < n; y += jump2)
                matmul_block2d<BM, BN, BK, TM, TN, float>(transa, transb, m, n, k, x, y, alpha,
                                                          A + z * strideA, lda, B + z * strideB,
                                                          ldb, beta, C + z * strideC, ldc);
}

template <int BM, int BN, int BK, int TM, int TN, typename SRC>
static tinyblasStatus_t tinyblasGSBE_wrapper(tinyblasHandle_t handle, tinyblasOperation_t transa,
                                             tinyblasOperation_t transb, int m, int n, int k,
                                             const void *pAlpha, const SRC *A, int lda,
                                             long long strideA, const SRC *B, int ldb,
                                             long long strideB, const void *pBeta, void *C,
                                             tinyblasDatatype_t Ctype, int ldc, long long strideC,
                                             int batchCount, tinyblasComputeType_t computeType) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int sharedmem = sizeof(SRC) * SHARELEN;
    int maxthreads = (BM * BN) / (TM * TN);
    switch (Ctype) {
    case TINYBLAS_R_16F:
        switch (computeType) {
        case TINYBLAS_COMPUTE_16F:
            tinyblasGSBE_entry<BM, BN, BK, TM, TN>
                <<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
                    transa, transb, m, n, k, *(const half *)pAlpha, A, lda, strideA, B, ldb,
                    strideB, *(const half *)pBeta, (half *)C, ldc, strideC, batchCount);
            return TINYBLAS_STATUS_SUCCESS;
        case TINYBLAS_COMPUTE_32F:
            tinyblasGSBE_entry<BM, BN, BK, TM, TN>
                <<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
                    transa, transb, m, n, k, *(const float *)pAlpha, A, lda, strideA, B, ldb,
                    strideB, *(const float *)pBeta, (half *)C, ldc, strideC, batchCount);
            return TINYBLAS_STATUS_SUCCESS;
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    case TINYBLAS_R_32F:
        switch (computeType) {
        case TINYBLAS_COMPUTE_16F:
            return TINYBLAS_STATUS_NOT_SUPPORTED;
        case TINYBLAS_COMPUTE_32F:
            tinyblasGSBE_entry<BM, BN, BK, TM, TN>
                <<<maxblocks, maxthreads, sharedmem, handle->stream>>>(
                    transa, transb, m, n, k, *(const float *)pAlpha, A, lda, strideA, B, ldb,
                    strideB, *(const float *)pBeta, (float *)C, ldc, strideC, batchCount);
            return TINYBLAS_STATUS_SUCCESS;
        default:
            return TINYBLAS_STATUS_INVALID_VALUE;
        }
    default:
        return TINYBLAS_STATUS_INVALID_VALUE;
    }
}

tinyblasStatus_t tinyblasGemmStridedBatchedEx(tinyblasHandle_t handle, //
                                              tinyblasOperation_t transa, //
                                              tinyblasOperation_t transb, //
                                              int m, int n, int k, //
                                              const void *pAlpha, //
                                              const void *A, tinyblasDatatype_t Atype, int lda,
                                              long long strideA, //
                                              const void *B, tinyblasDatatype_t Btype, int ldb,
                                              long long strideB, //
                                              const void *pBeta, //
                                              void *C, tinyblasDatatype_t Ctype, int ldc,
                                              long long strideC, //
                                              int batchCount, //
                                              tinyblasComputeType_t computeType, //
                                              tinyblasGemmAlgo_t algo) {
    if (!is_valid(transa, transb, m, n, k, lda, ldb, ldc))
        return TINYBLAS_STATUS_INVALID_VALUE;
    if (!is_supported(Atype, Btype))
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    return tinyblasGSBE_wrapper<32, 4, 64, 1, 2>(
        handle, transa, transb, m, n, k, pAlpha, (const half *)A, lda, strideA, (const half *)B,
        ldb, strideB, pBeta, C, Ctype, ldc, strideC, batchCount, computeType);
}

const char *tinyblasGetStatusString(tinyblasStatus_t err) {
    switch (err) {
    case TINYBLAS_STATUS_SUCCESS:
        return "TINYBLAS_STATUS_SUCCESS";
    case TINYBLAS_STATUS_ALLOC_FAILED:
        return "TINYBLAS_STATUS_ALLOC_FAILED";
    case TINYBLAS_STATUS_INVALID_VALUE:
        return "TINYBLAS_STATUS_INVALID_VALUE";
    case TINYBLAS_STATUS_NOT_SUPPORTED:
        return "TINYBLAS_STATUS_NOT_SUPPORTED";
    default:
        return "unknown tinyblas error";
    }
}
