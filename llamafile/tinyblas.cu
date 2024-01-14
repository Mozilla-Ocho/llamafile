// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2023 Mozilla Foundation
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

#define READ(A, trans, ld, i, j)                                        \
    (((trans) == TINYBLAS_OP_N) ? (A)[(i) + (j) * (ld)] : (A)[(j) + (i) * (ld)])
#define READ16(A, trans, ld, i, j) __half2float(READ(A, trans, ld, i, j))

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template<int BM, int BN, int BK>
static __device__ void matmul32_block2d(int m, int n, int k, int x, int y,
                                        const float *A, int lda, float *As,
                                        const float *B, int ldb, float *Bs,
                                        void *C, int ldc, float *Cs) {
    assert(blockDim.x == BK);
    static_assert(BK == BM, "");
    static_assert(BN <= BM, "");
    const int i = threadIdx.x;
    int j, l, blob;
    // within each block
    // we first zero out Cs
    for (j = 0; j < BN; ++j) Cs[j] = 0;

    for (blob = 0; blob < k; blob += BK) {
        if (i < BK) {
            if ((blob + i) < k) {
                // we copy into As from A
                for (j = 0; j < BM && x + j < m; ++j) {
                    As[(j * BK) + i] =
                        READ(A, TINYBLAS_OP_T, lda, x + j, blob + i);
                }
                for (; j < BM; ++j) As[(j * BK) + i] = 0;
                // we copy into Bs from B
                for (j = 0; j < BN && y + j < n; ++j) {
                    Bs[(i * BN) + j] =
                        READ(B, TINYBLAS_OP_N, ldb, blob + i, y + j);
                }
                for (; j < BN; ++j) Bs[(i * BN) + j] = 0;
            } else {  // UNLIKELY
                for (j = 0; j < BM; ++j) As[(j * BK) + i] = 0;
                for (j = 0; j < BN; ++j) Bs[(i * BN) + j] = 0;
            }
        }
        __syncthreads();


        // We matmul the blobs, basically Cs += matmul(As, Bs)
        for (j = 0; j < BN; ++j) {
            for (l = 0; l < BK; ++l) {
                Cs[j] += As[(i * BK) + l] * Bs[(l * BN) + j];
            }
        }
        __syncthreads();
    }

    for (j = 0; j < BN;  ++j) {
        As[(i*BN) + j] = Cs[j];
    }

    // We write Cs out into C
    if (y + i < n && i < BN) {
        for (j = 0; j < BM && x + j < m; ++j) {
            *((float *)C + (x + j) + (y + i) * ldc) = As[j*BN + i];
        }
    }
    __syncthreads();
}

template<int BM, int BN, int BK>
static __global__ void tinyblasS_entry(int m, int n, int k,
                                       const float *A, int lda,
                                       const float *B, int ldb,
                                       float       *C, int ldc) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;
    float Cs[BN];  // only within a particular thread

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-row of size BN
    for (x = blockIdx.x * BM; x < m; x += jump1) {
        for (y = blockIdx.y * BN; y < n; y += jump2) {
            matmul32_block2d<BM, BN, BK>(m, n, k, x, y,  //
                             A, lda, As,     //
                             B, ldb, Bs,     //
                             C, ldc, Cs);
        }
    }
}

static bool check_args(tinyblasOperation_t transa, tinyblasOperation_t transb,
                       const void *pAlpha, cudaDataType_t Atype,
                       cudaDataType_t Btype, const void *pBeta,
                       cudaDataType_t Ctype, tinyblasComputeType_t computeType) {
    return (transa == TINYBLAS_OP_T &&
            transb == TINYBLAS_OP_N &&
            Atype == CUDA_R_16F &&
            Btype == CUDA_R_16F &&
            (Ctype == CUDA_R_16F ||
             Ctype == CUDA_R_32F) &&
            ((computeType == TINYBLAS_COMPUTE_16F &&
              __half2float(*(half *)pAlpha) == 1.0f &&
              __half2float(*(half *)pBeta) == 0.0f) ||
             (computeType == TINYBLAS_COMPUTE_32F &&
              *(float *)pAlpha == 1.0f &&
              *(float *)pBeta == 0.0f)));
}

template <int BM, int BN, int BK>
static void tinyblasS_wrapper(tinyblasHandle_t stream, int m, int n, int k,
                              const float *A, int lda, const float *B, int ldb,
                              float *C, int ldc) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int maxthreads = BK;

    tinyblasS_entry<BM, BN, BK>
        <<<maxblocks, maxthreads, (sizeof(float) * (BM * BK + BK * BN)),
           stream>>>(m, n, k, A, lda, B, ldb, C, ldc);
}

tinyblasStatus_t tinyblasSgemm(tinyblasHandle_t stream,
                               tinyblasOperation_t transa,
                               tinyblasOperation_t transb,
                               int m, int n, int k,
                               const float *alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               const float *beta,
                               float       *C, int ldc) {
    if (transa != TINYBLAS_OP_T || transb != TINYBLAS_OP_N ||
        *alpha != 1.0f || *beta != 0.0f) {
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    }

    tinyblasS_wrapper<48, 12, 48>(stream, m, n, k, A, lda, B, ldb, C, ldc);
    return TINYBLAS_STATUS_SUCCESS;
}

template<int BM, int BN, int BK>
static __device__ void matmul_block2d(int m, int n, int k, int x, int y,
                                      const half *A, int lda, float *As,
                                      const half *B, int ldb, float *Bs,
                                      void *C, cudaDataType_t Ctype, int ldc,
                                      float *Cs) {
    assert(blockDim.x == BK);
    static_assert(BK == BM, "");
    static_assert(BN <= BM, "");
    const int i = threadIdx.x;
    int j, l, blob;
    // within each block
    // we first zero out Cs
    for (j = 0; j < BN; ++j) Cs[j] = 0;

    for (blob = 0; blob < k; blob += BK) {
        if (i < BK) {
            if ((blob + i) < k) {
                // we copy into As from A
                for (j = 0; j < BM && x + j < m; ++j) {
                    As[(j * BK) + i] =
                        READ16(A, TINYBLAS_OP_T, lda, x + j, blob + i);
                }
                for (; j < BM; ++j) As[(j * BK) + i] = 0;
                // we copy into Bs from B
                for (j = 0; j < BN && y + j < n; ++j) {
                    Bs[(i * BN) + j] =
                        READ16(B, TINYBLAS_OP_N, ldb, blob + i, y + j);
                }
                for (; j < BN; ++j) Bs[(i * BN) + j] = 0;
            } else {  // UNLIKELY
                for (j = 0; j < BM; ++j) As[(j * BK) + i] = 0;
                for (j = 0; j < BN; ++j) Bs[(i * BN) + j] = 0;
            }
        }
        __syncthreads();

        // We matmul the blobs, basically Cs += matmul(As, Bs)
        for (j = 0; j < BN; ++j) {
            for (l = 0; l < BK; ++l) {
                Cs[j] += As[(i * BK) + l] * Bs[(l * BN) + j];
            }
        }
        __syncthreads();
    }

    for (j = 0; j < BN;  ++j) {
        As[(i*BN) + j] = Cs[j];
    }

    // We write Cs out into C
    if (y + i < n && i < BN) {
        if (Ctype == CUDA_R_16F) {
            for (j = 0; j < BM && x + j < m; ++j) {
                *((half *)C + (x + j) + (y + i) * ldc) = __float2half(As[j*BN + i]);
            }
        } else {
            for (j = 0; j < BM && x + j < m; ++j) {
                *((float *)C + (x + j) + (y + i) * ldc) = As[j*BN + i];
            }
        }
    }
    __syncthreads();
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
template<int BM, int BN, int BK>
static __global__ void tinyblasGE_entry(int m, int n, int k, const half *A,
                                        int lda, const half *B, int ldb,
                                        void *C, cudaDataType_t Ctype,
                                        int ldc) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;
    float Cs[BN];  // only within a particular thread

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-row of size BN
    for (x = blockIdx.x * BM; x < m; x += jump1) {
        for (y = blockIdx.y * BN; y < n; y += jump2) {
            matmul_block2d<BM, BN, BK>(m, n, k, x, y,  //
                                       A, lda, As,     //
                                       B, ldb, Bs,     //
                                       C, Ctype, ldc, Cs);
        }
    }
}

template <int BM, int BN, int BK>
static void tinyblasGE_wrapper(tinyblasHandle_t stream, int m, int n, int k,
                               const half *A, int lda, const half *B, int ldb,
                               void *C, cudaDataType_t Ctype, int ldc) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int maxthreads = BK;

    tinyblasGE_entry<BM, BN, BK>
        <<<maxblocks, maxthreads, (sizeof(float) * (BM * BK + BK * BN)),
           stream>>>(m, n, k, A, lda, B, ldb, C, Ctype, ldc);
}

tinyblasStatus_t tinyblasGemmEx(tinyblasHandle_t stream,
                                tinyblasOperation_t transa,
                                tinyblasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const void    *alpha,
                                const void     *A,
                                cudaDataType_t Atype,
                                int lda,
                                const void     *B,
                                cudaDataType_t Btype,
                                int ldb,
                                const void    *beta,
                                void           *C,
                                cudaDataType_t Ctype,
                                int ldc,
                                tinyblasComputeType_t computeType,
                                tinyblasGemmAlgo_t algo) {
    if (!check_args(transa, transb, alpha, Atype, Btype, beta, Ctype,
                    computeType)) {
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    }

    tinyblasGE_wrapper<48, 12, 48>(stream, m, n, k, (const half *)A, lda,
                                   (const half *)B, ldb, C, Ctype, ldc);
    return TINYBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex

template<int BM, int BN, int BK>
static __global__ void tinyblasGBE_entry(int m, int n, int k,
                                         const half *const Aarray[], int lda,
                                         const half *const Barray[], int ldb,
                                         void *const Carray[],
                                         cudaDataType_t Ctype, int ldc,
                                         int batchCount) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;
    int z = blockIdx.z;
    const int jump3 = gridDim.z;

    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;
    float Cs[BN];  // only within a particular thread

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-row of size BN
    for (z = blockIdx.z; z < batchCount; z += jump3) {
        for (x = blockIdx.x * BM; x < m; x += jump1) {
            for (y = blockIdx.y * BN; y < n; y += jump2) {
                matmul_block2d<BM, BN, BK>(m, n, k, x, y,       //
                                           Aarray[z], lda, As,  //
                                           Barray[z], ldb, Bs,  //
                                           Carray[z], Ctype, ldc, Cs);
            }
        }
    }
}

template<int BM, int BN, int BK>
static void tinyblasGBE_wrapper(tinyblasHandle_t stream, int m, int n, int k,
                                const half *const Aarray[], int lda,
                                const half *const Barray[], int ldb,
                                void *const Carray[], cudaDataType_t Ctype,
                                int ldc, int batchCount) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int maxthreads = BK;

    tinyblasGBE_entry<BM, BN, BK>
        <<<maxblocks, maxthreads, (sizeof(float) * (BM * BK + BK * BN)),
           stream>>>(m, n, k, Aarray, lda, Barray,
                     ldb, Carray, Ctype, ldc, batchCount);
}

tinyblasStatus_t tinyblasGemmBatchedEx(tinyblasHandle_t stream,
                                       tinyblasOperation_t transa,
                                       tinyblasOperation_t transb,
                                       int m,
                                       int n,
                                       int k,
                                       const void    *alpha,
                                       const void     *const Aarray[],
                                       cudaDataType_t Atype,
                                       int lda,
                                       const void     *const Barray[],
                                       cudaDataType_t Btype,
                                       int ldb,
                                       const void    *beta,
                                       void           *const Carray[],
                                       cudaDataType_t Ctype,
                                       int ldc,
                                       int batchCount,
                                       tinyblasComputeType_t computeType,
                                       tinyblasGemmAlgo_t algo) {
    if (!check_args(transa, transb, alpha, Atype, Btype, beta, Ctype,
                    computeType)) {
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    }

    tinyblasGBE_wrapper<48, 12, 48>(stream, m, n, k, (const half **)Aarray, lda,
                                    (const half **)Barray, ldb, Carray, Ctype,
                                    ldc, batchCount);
    return TINYBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
template<int BM, int BN, int BK>
static __global__ void tinyblasGSBE_entry(int m, int n, int k,
                                          const half      *A,
                                          int             lda,
                                          long long int   strideA,
                                          const half      *B,
                                          int             ldb,
                                          long long int   strideB,
                                          void            *C,
                                          cudaDataType_t  Ctype,
                                          int             ldc,
                                          long long int   strideC,
                                          int batchCount) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;
    int z = blockIdx.z;
    const int jump3 = gridDim.z;

    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;
    float Cs[BN];  // only within a particular thread

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-row of size BN
    for (z = blockIdx.z; z < batchCount; z += jump3) {
        for (x = blockIdx.x * BM; x < m; x += jump1) {
            for (y = blockIdx.y * BN; y < n; y += jump2) {
                matmul_block2d<BM, BN, BK>(
                    m, n, k, x, y,             //
                    A + z * strideA, lda, As,  //
                    B + z * strideB, ldb, Bs,  //
                    (Ctype == CUDA_R_16F ? (void *)((half *)C + z * strideC)
                                         : (void *)((float *)C + z * strideC)),
                    Ctype, ldc, Cs);
            }
        }
    }
}

template <int BM, int BN, int BK>
static void tinyblasGSBE_wrapper(tinyblasHandle_t stream, int m, int n, int k,
                                 const half *A, int lda, long long int strideA,
                                 const half *B, int ldb, long long int strideB,
                                 void *C, cudaDataType_t Ctype, int ldc,
                                 long long int strideC, int batchCount) {
    // call the entry function
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int maxthreads = BK;

    tinyblasGSBE_entry<BM, BN, BK>
        <<<maxblocks, maxthreads, (sizeof(float) * (BM * BK + BK * BN)),
           stream>>>(m, n, k,                 //
                     A, lda, strideA,         //
                     B, ldb, strideB,         //
                     C, Ctype, ldc, strideC,  //
                     batchCount);
}

tinyblasStatus_t tinyblasGemmStridedBatchedEx(tinyblasHandle_t stream,
                                              tinyblasOperation_t transa,
                                              tinyblasOperation_t transb,
                                              int m, int n, int k,
                                              const void    *pAlpha,
                                              const void     *A,
                                              cudaDataType_t Atype,
                                              int lda,
                                              long long int strideA,
                                              const void     *B,
                                              cudaDataType_t Btype,
                                              int ldb,
                                              long long int strideB,
                                              const void    *pBeta,
                                              void           *C,
                                              cudaDataType_t Ctype,
                                              int ldc,
                                              long long int strideC,
                                              int batchCount,
                                              tinyblasComputeType_t computeType,
                                              tinyblasGemmAlgo_t algo) {
    if (!check_args(transa, transb, pAlpha, Atype, Btype, pBeta, Ctype,
                    computeType)) {
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    }

    tinyblasGSBE_wrapper<64, 4, 64>(stream, m, n, k, (const half *)A, lda, strideA,
                                     (const half *)B, ldb, strideB, C, Ctype,
                                     ldc, strideC, batchCount);

    return TINYBLAS_STATUS_SUCCESS;
}

const char *tinyblasGetStatusString(tinyblasStatus_t err) {
    switch (err) {
        case TINYBLAS_STATUS_SUCCESS:
            return "TINYBLAS_STATUS_SUCCESS";
        case TINYBLAS_STATUS_NOT_INITIALIZED:
            return "TINYBLAS_STATUS_NOT_INITIALIZED";
        case TINYBLAS_STATUS_ALLOC_FAILED:
            return "TINYBLAS_STATUS_ALLOC_FAILED";
        case TINYBLAS_STATUS_INVALID_VALUE:
            return "TINYBLAS_STATUS_INVALID_VALUE";
        case TINYBLAS_STATUS_ARCH_MISMATCH:
            return "TINYBLAS_STATUS_ARCH_MISMATCH";
        case TINYBLAS_STATUS_MAPPING_ERROR:
            return "TINYBLAS_STATUS_MAPPING_ERROR";
        case TINYBLAS_STATUS_EXECUTION_FAILED:
            return "TINYBLAS_STATUS_EXECUTION_FAILED";
        case TINYBLAS_STATUS_INTERNAL_ERROR:
            return "TINYBLAS_STATUS_INTERNAL_ERROR";
        case TINYBLAS_STATUS_NOT_SUPPORTED:
            return "TINYBLAS_STATUS_NOT_SUPPORTED";
        default:
            return "unknown error";
    }
}
