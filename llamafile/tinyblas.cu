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
    (((trans) == CUBLAS_OP_N) ? (A)[(i) + (j) * (ld)] : (A)[(j) + (i) * (ld)])
#define READ16(A, trans, ld, i, j) __half2float(READ(A, trans, ld, i, j))

#define BM 64
#define BN 32
#define BK BM
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static __device__ void matmul32_block2d(int m, int n, int k, int x, int y,
                                      const float *A, int lda, float *As,
                                      const float *B, int ldb, float *Bs,
                                      void *C, int ldc, float *Cs) {
    const int i = threadIdx.x;
    int j, l, blob;
    // within each block
    // we first zero out Cs
    for (j = 0; j < BN; ++j) Cs[j] = 0;

    for (blob = 0; blob < k; blob += BK) {
        // we copy into As from A
        if (i < BM && (x + i) < m) {
            for (j = 0; j < BK && blob + j < k; ++j) {
                As[(i * BK) + j] = READ(A, CUBLAS_OP_T, lda, x + i, blob + j);
            }
            for (; j < BK; ++j) As[(i * BK) + j] = 0;
        } else { // UNLIKELY
            for (j = 0; j < BK; ++j) As[(i * BK) + j] = 0;
        }

        // we copy into Bs from B
        if (i < BK && (blob + i) < k) {
            for (j = 0; j < BN && y + j < n; ++j) {
                Bs[(i * BN) + j] = READ(B, CUBLAS_OP_N, ldb, blob + i, y + j);
            }
            for (; j < BN; ++j) Bs[(i * BN) + j] = 0;
        } else { // UNLIKELY
            for (j = 0; j < BN; ++j) Bs[(i * BN) + j] = 0;
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

    // We write Cs out into C
    if (x + i < m) {
        for (j = 0; j < BN && y + j < n; ++j) {
            *((float *)C + (x + i) + (y + j) * ldc) = Cs[j];
        }
    }
    __syncthreads();
}

static __global__ void tinyblasS_entry(int m, int n, int k,
                                       const float *A, int lda,
                                       const float *B, int ldb,
                                       float       *C, int ldc) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    assert(blockDim.x == BM);
    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;
    float Cs[BN];  // only within a particular thread

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-row of size BN
    for (x = blockIdx.x * BM; x < m; x += jump1) {
        for (y = blockIdx.y * BN; y < n; y += jump2) {
            matmul32_block2d(m, n, k, x, y,  //
                           A, lda, As,     //
                           B, ldb, Bs,     //
                           C, ldc, Cs);
        }
    }
}

static bool check_args(cublasOperation_t transa, cublasOperation_t transb,
                       const void *pAlpha, cudaDataType_t Atype,
                       cudaDataType_t Btype, const void *pBeta,
                       cudaDataType_t Ctype, cublasComputeType_t computeType) {
    return (transa == CUBLAS_OP_T &&
            transb == CUBLAS_OP_N &&
            Atype == CUDA_R_16F &&
            Btype == CUDA_R_16F &&
            (Ctype == CUDA_R_16F ||
             Ctype == CUDA_R_32F) &&
            ((computeType == CUBLAS_COMPUTE_16F &&
              __half2float(*(half *)pAlpha) == 1.0f &&
              __half2float(*(half *)pBeta) == 0.0f) ||
             (computeType == CUBLAS_COMPUTE_32F &&
              *(float *)pAlpha == 1.0f &&
              *(float *)pBeta == 0.0f)));
}

cublasStatus_t tinyblasSgemm(cudaStream_t stream,
                             cublasOperation_t transa,
                             cublasOperation_t transb,
                             int m, int n, int k,
                             const float *alpha,
                             const float *A, int lda,
                             const float *B, int ldb,
                             const float *beta,
                             float       *C, int ldc) {
    if (transa != CUBLAS_OP_T || transb != CUBLAS_OP_N ||
        *alpha != 1.0f || *beta != 0.0f) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int maxthreads = BM;

    tinyblasS_entry<<<maxblocks, maxthreads,
                       (sizeof(float) * (BM * BK + BK * BN)), stream>>>(
        m, n, k, A, lda, B, ldb, C, ldc);
    return CUBLAS_STATUS_SUCCESS;
}

static __device__ void matmul_block2d(int m, int n, int k, int x, int y,
                                      const half *A, int lda, float *As,
                                      const half *B, int ldb, float *Bs,
                                      void *C, cudaDataType_t Ctype, int ldc,
                                      float *Cs) {
    const int i = threadIdx.x;
    int j, l, blob;
    // within each block
    // we first zero out Cs
    for (j = 0; j < BN; ++j) Cs[j] = 0;

    for (blob = 0; blob < k; blob += BK) {
        // we copy into As from A
        if (i < BM && (x + i) < m) {
            for (j = 0; j < BK && blob + j < k; ++j) {
                As[(i * BK) + j] = READ16(A, CUBLAS_OP_T, lda, x + i, blob + j);
            }
            for (; j < BK; ++j) As[(i * BK) + j] = 0;
        } else { // UNLIKELY
            for (j = 0; j < BK; ++j) As[(i * BK) + j] = 0;
        }

        // we copy into Bs from B
        if (i < BK && (blob + i) < k) {
            for (j = 0; j < BN && y + j < n; ++j) {
                Bs[(i * BN) + j] = READ16(B, CUBLAS_OP_N, ldb, blob + i, y + j);
            }
            for (; j < BN; ++j) Bs[(i * BN) + j] = 0;
        } else { // UNLIKELY
            for (j = 0; j < BN; ++j) Bs[(i * BN) + j] = 0;
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

    // We write Cs out into C
    if (x + i < m) {
        if (Ctype == CUDA_R_16F) {
            for (j = 0; j < BN && y + j < n; ++j) {
                *((half *)C + (x + i) + (y + j) * ldc) = __float2half(Cs[j]);
            }
        } else {
            for (j = 0; j < BN && y + j < n; ++j) {
                *((float *)C + (x + i) + (y + j) * ldc) = Cs[j];
            }
        }
    }
    __syncthreads();
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
static __global__ void tinyblasGE_entry(int m, int n, int k, const half *A,
                                        int lda, const half *B, int ldb,
                                        void *C, cudaDataType_t Ctype,
                                        int ldc) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    assert(blockDim.x == BM);
    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;
    float Cs[BN];  // only within a particular thread

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-row of size BN
    for (x = blockIdx.x * BM; x < m; x += jump1) {
        for (y = blockIdx.y * BN; y < n; y += jump2) {
            matmul_block2d(m, n, k, x, y,  //
                           A, lda, As,     //
                           B, ldb, Bs,     //
                           C, Ctype, ldc, Cs);
        }
    }
}

cublasStatus_t tinyblasGemmEx(cudaStream_t stream,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
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
                              cublasComputeType_t computeType,
                              cublasGemmAlgo_t algo) {
    if (!check_args(transa, transb, alpha, Atype, Btype, beta, Ctype,
                    computeType)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int maxthreads = BM;

    tinyblasGE_entry<<<maxblocks, maxthreads,
                       (sizeof(float) * (BM * BK + BK * BN)), stream>>>(
        m, n, k, (const half *)A, lda, (const half *)B, ldb, C, Ctype, ldc);
    return CUBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex

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

    assert(blockDim.x == BM);
    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;
    float Cs[BN];  // only within a particular thread

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-row of size BN
    for (z = blockIdx.z; z < batchCount; z += jump3) {
        for (x = blockIdx.x * BM; x < m; x += jump1) {
            for (y = blockIdx.y * BN; y < n; y += jump2) {
                matmul_block2d(m, n, k, x, y,       //
                               Aarray[z], lda, As,  //
                               Barray[z], ldb, Bs,  //
                               Carray[z], Ctype, ldc, Cs);
            }
        }
    }
}

cublasStatus_t tinyblasGemmBatchedEx(cudaStream_t stream,
                                     cublasOperation_t transa,
                                     cublasOperation_t transb,
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
                                     cublasComputeType_t computeType,
                                     cublasGemmAlgo_t algo) {
    if (!check_args(transa, transb, alpha, Atype, Btype, beta, Ctype,
                    computeType)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int maxthreads = BM;

    tinyblasGBE_entry<<<maxblocks, maxthreads,
                       (sizeof(float) * (BM * BK + BK * BN)), stream>>>(
        m, n, k, (const half **)Aarray, lda, (const half **)Barray, ldb,
        Carray, Ctype, ldc, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex

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

    assert(blockDim.x == BM);
    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;
    float Cs[BN];  // only within a particular thread

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-row of size BN
    for (z = blockIdx.z; z < batchCount; z += jump3) {
        for (x = blockIdx.x * BM; x < m; x += jump1) {
            for (y = blockIdx.y * BN; y < n; y += jump2) {
                matmul_block2d(m, n, k, x, y,       //
                               A + z * strideA, lda, As,  //
                               B + z * strideB, ldb, Bs,  //
                              (Ctype == CUDA_R_16F
                               ? (void *)((half *)C + z * strideC)
                               : (void *)((float *)C + z * strideC)),
                              Ctype, ldc, Cs);
            }
        }
    }
}

cublasStatus_t tinyblasGemmStridedBatchedEx(cudaStream_t stream,
                                            cublasOperation_t transa,
                                            cublasOperation_t transb,
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
                                            cublasComputeType_t computeType,
                                            cublasGemmAlgo_t algo) {
    if (!check_args(transa, transb, pAlpha, Atype, Btype, pBeta, Ctype,
                    computeType)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    // call the entry function
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int maxthreads = BM;

    tinyblasGSBE_entry<<<maxblocks, maxthreads,
                       (sizeof(float) * (BM * BK + BK * BN)), stream>>>(
        m, n, k, (const half*)A, lda, strideA, (const half*)B, ldb, strideB,
        C, Ctype, ldc, strideC, batchCount);

    return CUBLAS_STATUS_SUCCESS;
}
