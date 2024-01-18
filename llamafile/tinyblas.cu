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

template<int BM, int BN, int BK, int TM, int TN>
static __device__ void matmul32_block2d(int m, int n, int k, int x, int y,
                                        const float *A, int lda, float *As,
                                        const float *B, int ldb, float *Bs,
                                        void *C, int ldc) {
    const int ii0 = threadIdx.x / (BN / TN); /* {0, ..., (BM/TM) - 1} */
    const int ii1 = threadIdx.x % (BN / TN); /* {0, ..., (BN/TN) - 1} */

    float Cs[TM * TN];
    float At[TM];
    float Bt[TN];
    int i, h, j, l, blob;
    // within each block
    // we first zero out Cs
    for (j = 0; j < TM * TN; ++j) Cs[j] = 0;

    i = threadIdx.x;
    for (blob = 0; blob < k; blob += BK) {
        for (i = threadIdx.x; i < BK; i += blockDim.x) {
            for (j = 0; j < BM; ++j) As[(i * BM) + j] = 0;
            if ((blob + i) < k) {
                // we copy into As from A
                for (j = 0; j < BM && x + j < m; ++j) {
                    As[(i * BM) + j] =
                        READ(A, TINYBLAS_OP_T, lda, x + j, blob + i);
                }
            }
        }
        __syncthreads();
        
        for (i = threadIdx.x; i < BK; i += blockDim.x) {
            for (j = 0; j < BN; ++j) Bs[(i * BN) + j] = 0;
            if ((blob + i) < k) {
                // we copy into Bs from B
                for (j = 0; j < BN && y + j < n; ++j) {
                    Bs[(i * BN) + j] =
                        READ(B, TINYBLAS_OP_N, ldb, blob + i, y + j);
                }
            }
        }
        __syncthreads();


        // We matmul the blobs, basically Cs += matmul(As, Bs)
        for (l = 0; l < BK; ++l) {
            for (j = 0; j < TM; ++j) At[j] = As[(l * BM) + ii0 * TM + j];
            for (h = 0; h < TN; ++h) Bt[h] = Bs[(l * BN) + ii1 * TN + h];
            for (j = 0; j < TM; ++j) {
                for (h = 0; h < TN; ++h) {
                    Cs[j * TN + h] += At[j] * Bt[h];
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // We write Cs out into C
    x += ii0 * TM;
    y += ii1 * TN;
    for (j = 0; j < TM && x + j < m; ++j) {
        for (l = 0; l < TN && y + l < n; ++l) {
            *((float *)C + (x + j) + (y + l) * ldc) = Cs[j * TN + l];
        }
    }
    __syncthreads();
}

template<int BM, int BN, int BK, int TM, int TN>
static __global__ void tinyblasS_entry(int m, int n, int k,
                                       const float *A, int lda,
                                       const float *B, int ldb,
                                       float       *C, int ldc) {
    assert(blockDim.x == BK);
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    extern __shared__ float svals[];  // shared across all threads in a block
    float *As = svals;
    float *Bs = svals + BM * BK;

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-matrix of size TM * TN
    for (x = blockIdx.x * BM; x < m; x += jump1) {
        for (y = blockIdx.y * BN; y < n; y += jump2) {
            matmul32_block2d<BM, BN, BK, TM, TN>(m, n, k, x, y,  //
                             A, lda, As,     //
                             B, ldb, Bs,     //
                             C, ldc);
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

template <int BM, int BN, int BK, int TM, int TN>
static void tinyblasS_wrapper(tinyblasHandle_t stream, int m, int n, int k,
                              const float *A, int lda, const float *B, int ldb,
                              float *C, int ldc) {
    static_assert(BN <= BM, "threads can't read columns properly");
    static_assert((BM % TM == 0) && (BN % TN == 0),
                  "can't divide work for threads");
    static_assert(BK == ((BM * BN) / (TM * TN)),
                  "threads can't load memory properly");
    static_assert((BM * BN) <= (BM * BK) + (BK * BN),
                  "didn't allocate enough shared mem for threads");
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int maxthreads = ((BM * BN) / (TM * TN));

    tinyblasS_entry<BM, BN, BK, TM, TN>
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

    tinyblasS_wrapper<48, 24, 64, 6, 3>(stream, m, n, k, A, lda, B, ldb, C, ldc);
    return TINYBLAS_STATUS_SUCCESS;
}

template<typename SRC, typename DST>
__device__ __forceinline__ DST typechange(SRC x) {
    static_assert(std::is_same<DST, SRC>::value, "write a specialization");
    if (std::is_same<DST, SRC>::value) {
        return x;
    } else {
        return (DST)(0);
    }
}

template<>
__device__ __forceinline__ float typechange(half x) {
    return __half2float(x);
}

template<>
__device__ __forceinline__ half typechange(float x) {
    return __float2half(x);
}

template<int BM, int BN, int BK, int TM, int TN, typename DST>
static __device__ void matmul_block2d(int m, int n, int k, int x, int y,
                                      const half *A, int lda, half *As,
                                      const half *B, int ldb, half *Bs,
                                      void *C, int ldc) {
    const int ii0 = threadIdx.x / (BN / TN); /* {0, ..., (BM/TM) - 1} */
    const int ii1 = threadIdx.x % (BN / TN); /* {0, ..., (BN/TN) - 1} */

    half Cs[TM * TN];
    half At[TM];
    half Bt[TN];
    int i, h, j, l, blob;
    // within each block
    // we first zero out Cs
    for (j = 0; j < TM * TN; ++j) Cs[j] = 0;

    i = threadIdx.x;
    for (blob = 0; blob < k; blob += BK) {
        for (i = threadIdx.x; i < BK; i += blockDim.x) {
            for (j = 0; j < BM; ++j) As[(i * BM) + j] = 0;
            if ((blob + i) < k) {
                // we copy into As from A
                for (j = 0; j < BM && x + j < m; ++j) {
                    As[(i * BM) + j] =
                        READ(A, TINYBLAS_OP_T, lda, x + j, blob + i);
                }
            }
        }
        __syncthreads();
        
        for (i = threadIdx.x; i < BK; i += blockDim.x) {
            for (j = 0; j < BN; ++j) Bs[(i * BN) + j] = 0;
            if ((blob + i) < k) {
                // we copy into Bs from B
                for (j = 0; j < BN && y + j < n; ++j) {
                    Bs[(i * BN) + j] =
                        READ(B, TINYBLAS_OP_N, ldb, blob + i, y + j);
                }
            }
        }
        __syncthreads();


        // We matmul the blobs, basically Cs += matmul(As, Bs)
        for (l = 0; l < BK; ++l) {
            for (j = 0; j < TM; ++j) At[j] = As[(l * BM) + ii0 * TM + j];
            for (h = 0; h < TN; ++h) Bt[h] = Bs[(l * BN) + ii1 * TN + h];
            for (j = 0; j < TM; ++j) {
                for (h = 0; h < TN; ++h) {
                    Cs[j * TN + h] += At[j] * Bt[h];
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // We write Cs out into C
    x += ii0 * TM;
    y += ii1 * TN;
    for (j = 0; j < TM && x + j < m; ++j) {
        for (l = 0; l < TN && y + l < n; ++l) {
            *((DST *)C + (x + j) + (y + l) * ldc) = typechange<half, DST>(Cs[j * TN + l]);
        }
    }
    __syncthreads();
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
template<int BM, int BN, int BK, int TM, int TN, typename DST>
static __global__ void tinyblasGE_entry(int m, int n, int k, const half *A,
                                        int lda, const half *B, int ldb,
                                        void *C, int ldc) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    extern __shared__ float svals[];  // shared across all threads in a block
    half *As = (half *) svals;
    half *Bs = (half *) svals + BM * BK;

    // each block handles a sub-matrix of C, of size BM * BN
    for (x = blockIdx.x * BM; x < m; x += jump1) {
        for (y = blockIdx.y * BN; y < n; y += jump2) {
            matmul_block2d<BM, BN, BK, TM, TN, DST>(m, n, k, x, y,  //
                                       A, lda, As,     //
                                       B, ldb, Bs,     //
                                       C, ldc);
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
static void tinyblasGE_wrapper(tinyblasHandle_t stream, int m, int n, int k,
                               const half *A, int lda, const half *B, int ldb,
                               void *C, cudaDataType_t Ctype, int ldc) {
    static_assert(BN <= BM, "threads can't read columns properly");
    static_assert((BM % TM == 0) && (BN % TN == 0),
                  "can't divide work for threads");
    static_assert((BM * BN) <= (BM * BK) + (BK * BN),
                  "didn't allocate enough shared mem for threads");
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int maxthreads = ((BM * BN) / (TM * TN));

    if (Ctype == CUDA_R_16F) {
        tinyblasGE_entry<BM, BN, BK, TM, TN, half>
            <<<maxblocks, maxthreads, (sizeof(half) * (BM * BK + BK * BN)),
               stream>>>(m, n, k, A, lda, B, ldb, C, ldc);
    } else {
        tinyblasGE_entry<BM, BN, BK, TM, TN, float>
            <<<maxblocks, maxthreads, (sizeof(half) * (BM * BK + BK * BN)),
               stream>>>(m, n, k, A, lda, B, ldb, C, ldc);
    }
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

    tinyblasGE_wrapper<48, 32, 32, 6, 8>(stream, m, n, k, (const half *)A, lda,
                               (const half *)B, ldb, C, Ctype, ldc);
    return TINYBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex

template<int BM, int BN, int BK, int TM, int TN, typename DST>
static __global__ void tinyblasGBE_entry(int m, int n, int k,
                                         const half *const Aarray[], int lda,
                                         const half *const Barray[], int ldb,
                                         void *const Carray[], int ldc,
                                         int batchCount) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;
    int z = blockIdx.z;
    const int jump3 = gridDim.z;

    extern __shared__ float svals[];  // shared across all threads in a block
    half *As = (half *) svals;
    half *Bs = (half *) svals + BM * BK;

    // each block handles a sub-matrix of C, of size BM * BN
    for (z = blockIdx.z; z < batchCount; z += jump3) {
        for (x = blockIdx.x * BM; x < m; x += jump1) {
            for (y = blockIdx.y * BN; y < n; y += jump2) {
                matmul_block2d<BM, BN, BK, TM, TN, DST>(m, n, k, x, y,       //
                                           Aarray[z], lda, As,  //
                                           Barray[z], ldb, Bs,  //
                                           Carray[z], ldc);
            }
        }
    }
}

template<int BM, int BN, int BK, int TM, int TN>
static void tinyblasGBE_wrapper(tinyblasHandle_t stream, int m, int n, int k,
                                const half *const Aarray[], int lda,
                                const half *const Barray[], int ldb,
                                void *const Carray[], cudaDataType_t Ctype,
                                int ldc, int batchCount) {
    static_assert(BN <= BM, "threads can't read columns properly");
    static_assert((BM % TM == 0) && (BN % TN == 0),
                  "can't divide work for threads");
    static_assert((BM * BN) <= (BM * BK) + (BK * BN),
                  "didn't allocate enough shared mem for threads");
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int maxthreads = ((BM * BN) / (TM * TN));

    if (Ctype == CUDA_R_16F) {
        tinyblasGBE_entry<BM, BN, BK, TM, TN, half>
            <<<maxblocks, maxthreads, (sizeof(half) * (BM * BK + BK * BN)),
               stream>>>(m, n, k, Aarray, lda, Barray, ldb, Carray, ldc,
                         batchCount);
    } else {
        tinyblasGBE_entry<BM, BN, BK, TM, TN, float>
            <<<maxblocks, maxthreads, (sizeof(half) * (BM * BK + BK * BN)),
               stream>>>(m, n, k, Aarray, lda, Barray, ldb, Carray, ldc,
                         batchCount);
    }
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

    tinyblasGBE_wrapper<48, 32, 32, 6, 8>(stream, m, n, k, (const half **)Aarray, lda,
                                    (const half **)Barray, ldb, Carray, Ctype,
                                    ldc, batchCount);
    return TINYBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
template<int BM, int BN, int BK, int TM, int TN, typename DST>
static __global__ void tinyblasGSBE_entry(int m, int n, int k,
                                          const half      *A,
                                          int             lda,
                                          long long int   strideA,
                                          const half      *B,
                                          int             ldb,
                                          long long int   strideB,
                                          void            *C,
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
    half *As = (half *) svals;
    half *Bs = (half *) svals + BM * BK;

    // each block handles a sub-matrix of C, of size BM * BN
    for (z = blockIdx.z; z < batchCount; z += jump3) {
        for (x = blockIdx.x * BM; x < m; x += jump1) {
            for (y = blockIdx.y * BN; y < n; y += jump2) {
                matmul_block2d<BM, BN, BK, TM, TN, DST>(
                    m, n, k, x, y,             //
                    A + z * strideA, lda, As,  //
                    B + z * strideB, ldb, Bs,  //
                    (void *)((DST *)C + z * strideC), ldc);
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
static void tinyblasGSBE_wrapper(tinyblasHandle_t stream, int m, int n, int k,
                                 const half *A, int lda, long long int strideA,
                                 const half *B, int ldb, long long int strideB,
                                 void *C, cudaDataType_t Ctype, int ldc,
                                 long long int strideC, int batchCount) {
    static_assert(BN <= BM, "threads can't read columns properly");
    static_assert((BM % TM == 0) && (BN % TN == 0),
                  "can't divide work for threads");
    static_assert((BM * BN) <= (BM * BK) + (BK * BN),
                  "didn't allocate enough shared mem for threads");
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int maxthreads = ((BM * BN) / (TM * TN));

    if (Ctype == CUDA_R_16F) {
        tinyblasGSBE_entry<BM, BN, BK, TM, TN, half>
            <<<maxblocks, maxthreads, (sizeof(half) * (BM * BK + BK * BN)),
               stream>>>(m, n, k,          //
                         A, lda, strideA,  //
                         B, ldb, strideB,  //
                         C, ldc, strideC,  //
                         batchCount);
    } else {
        tinyblasGSBE_entry<BM, BN, BK, TM, TN, float>
            <<<maxblocks, maxthreads, (sizeof(half) * (BM * BK + BK * BN)),
               stream>>>(m, n, k,          //
                         A, lda, strideA,  //
                         B, ldb, strideB,  //
                         C, ldc, strideC,  //
                         batchCount);
    }
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

    tinyblasGSBE_wrapper<32, 4, 64, 1, 2>(stream, m, n, k, (const half *)A, lda, strideA,
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
