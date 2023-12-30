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

static __device__ __forceinline__ void matmul_single(int m, int n, int k, 
                                                     int i, int j,
                                                     const half *A, int lda,
                                                     const half *B, int ldb,
                                                     void       *C,
                                                     cudaDataType_t Ctype,
                                                     int ldc) {
    float sum = 0.0f;
    for (int l = 0; l < k; ++l) {
        sum += READ16(A, CUBLAS_OP_T, lda, i, l) *
               READ16(B, CUBLAS_OP_N, ldb, l, j);
    }
    if (Ctype == CUDA_R_16F) {
        *((half *)C + i + j * ldc) = __float2half(sum);
    } else {
        *((float *)C + i + j * ldc) = sum;
    }
}

static __device__ __forceinline__ void matmul_single32(int m, int n, int k, int i, int j,
                                                       const float *A, int lda,
                                                       const float *B, int ldb,
                                                       float       *C, int ldc) {
    float sum = 0.0f;
    float *cptr = C + i + j * ldc;
    for (int l = 0; l < k; ++l) {
        sum += READ(A, CUBLAS_OP_T, lda, i, l) *
               READ(B, CUBLAS_OP_N, ldb, l, j);
    }
    *cptr = sum;
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

static __global__ void tinyblasS_entry(int m, int n, int k,
                                       const float *A, int lda,
                                       const float *B, int ldb,
                                       float       *C, int ldc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int jump = blockDim.x * gridDim.x;
    int y = threadIdx.y;
    int jump2 = blockDim.y;

    for (; x < m; x += jump) {
        for (y=threadIdx.y; y < n; y += jump2) {
            matmul_single32(m, n, k, x, y, A, lda, B, ldb, C, ldc);
        }
    }
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

    int numSMs, devId;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    int maxblocks = 16 * numSMs;
    dim3 maxthreads(16, 64, 1);

    tinyblasS_entry<<<maxblocks, maxthreads, 0, stream>>>(
        m, n, k, A, lda, B, ldb, C, ldc);
    return CUBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
#define BM 64
#define BN 32
#define BK BM
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static __global__ void tinyblasGE_entry(int m, int n, int k, const half *A,
                                        int lda, const half *B, int ldb,
                                        void *C, cudaDataType_t Ctype,
                                        int ldc) {
    int x = blockIdx.x * BM;
    int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    int jump2 = gridDim.x * BN;

    assert(blockDim.x == BM);
    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;
    float Cs[BN];  // only within a particular thread

    int i, j, l;
    int blob;

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-row of size BN
    for (x = blockIdx.x * BM; x < m; x += jump1) {
        for (y = blockIdx.y * BN; y < n; y += jump2) {
            // within each block
            // we first zero out Cs
            for (j = 0; j < BN; ++j) Cs[j] = 0;

            for (blob = 0; blob < k; blob += BK) {
                // we copy into As from A
                for (i = threadIdx.x; i < BM && (x + i) < m; i += blockDim.x) {
                    for (j = 0; j < BK && blob + j < k; ++j) {
                        As[(i * BK) + j] =
                            READ16(A, CUBLAS_OP_T, lda, x + i, blob + j);
                    }
                    for (; j < BK; ++j) As[(i * BK) + j] = 0;
                }

                // we copy into Bs from B
                for (i = threadIdx.x; i < BK && (blob + i) < k;
                     i += blockDim.x) {
                    for (j = 0; j < BN && y + j < n; ++j) {
                        Bs[(i * BN) + j] =
                            READ16(B, CUBLAS_OP_N, ldb, blob + i, y + j);
                    }
                    for (; j < BN; ++j) Bs[(i * BN) + j] = 0;
                }
                __syncthreads();

                // We matmul the blobs, basically Cs += matmul(As, Bs)
                // (thread-local stuff can happen here? bounds-check?)
                i = threadIdx.x;
                for (j = 0; j < BN; ++j) {
                    for (l = 0; l < BK; ++l) {
                        Cs[j] += As[(i * BK) + l] * Bs[(l * BN) + j];
                    }
                }
                __syncthreads();
            }

            // We write Cs out into C
            i = threadIdx.x;
            for (j = 0; j < BN && y + j < n; ++j) {
                *((half *)C + (x + i) + (y + j) * ldc) = __float2half(Cs[j]);
            }
            __syncthreads();
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
                                         const half *const  Aarray[], int lda,
                                         const half *const  Barray[], int ldb,
                                         void *const        Carray[],
                                         cudaDataType_t     Ctype,    int ldc,
                                         int batchCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int jump = blockDim.x * gridDim.x;
    int jump2 = blockDim.y * gridDim.y;
    int jump3 = blockDim.z;

    for (; x < batchCount; x += jump) {
        for (y=blockIdx.y * blockDim.y + threadIdx.y; y < m; y += jump2) {
            for (z=threadIdx.z; z < n; z += jump3) {
                matmul_single(m, n, k, y, z, Aarray[x], lda, Barray[x], ldb, Carray[x], Ctype, ldc);
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

    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    int numSMs, devId;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    dim3 maxblocks(numSMs, 16, 1);
    dim3 maxthreads(4, 4, 64);

    tinyblasGBE_entry<<<maxblocks, maxthreads, 0, stream>>>(
        m, n, k, (const half **)Aarray, lda, (const half **)Barray, ldb,
        Carray, Ctype, ldc, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex

#define STRIDE0(A, i, stride) ((A) + (i) * (stride))
#define STRIDE(A, type, i, stride)                      \
    ((type) == CUDA_R_16F                               \
     ? (void *)STRIDE0((half *)(A), (i), (stride))      \
     : (void *)STRIDE0((float *)(A), (i), (stride)))
#define STRIDE_CONST(A, type, i, stride)                                \
    ((type) == CUDA_R_16F                                               \
     ? (const void *)STRIDE0((const half *)(A), (i), (stride))          \
     : (const void *)STRIDE0((const float *)(A), (i), (stride)))

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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;
    int jump = blockDim.x * gridDim.x;
    int jump2 = blockDim.y * gridDim.y;
    int jump3 = blockDim.z;

    for (; x < batchCount; x += jump) {
        for (y=blockIdx.y * blockDim.y + threadIdx.y; y < m; y += jump2) {
            for (z=threadIdx.z; z < n; z += jump3) {
                matmul_single(m, n, k, y, z, A + x * strideA, lda, B + x * strideB, ldb,
                              (Ctype == CUDA_R_16F
                               ? (void *)((half *)C + x * strideC)
                               : (void *)((float *)C + x * strideC)),
                              Ctype, ldc);
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

    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    int numSMs, devId;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    dim3 maxblocks(numSMs, 16, 1);
    dim3 maxthreads(4, 4, 64);

    // call the entry function
    tinyblasGSBE_entry<<<maxblocks, maxthreads, 0, stream>>>(
        m, n, k, (const half*)A, lda, strideA, (const half*)B, ldb, strideB,
        C, Ctype, ldc, strideC, batchCount);

    return CUBLAS_STATUS_SUCCESS;
}
