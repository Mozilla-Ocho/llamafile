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

#define READ(A, trans, ld, i, j) \
    (((trans) == TINYBLAS_OP_N) ? (A)[(i) + (j) * (ld)] : (A)[(j) + (i) * (ld)])
#define READ16(A, trans, ld, i, j) __half2float(READ(A, trans, ld, i, j))

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <typename SRC, typename DST>
__device__ __forceinline__ DST typechange(SRC x) {
    static_assert(std::is_same<DST, SRC>::value, "write a specialization");
    if (std::is_same<DST, SRC>::value) {
        return x;
    } else {
        return (DST)(0);
    }
}

template <>
__device__ __forceinline__ float typechange(half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ half typechange(float x) {
    return __float2half(x);
}

template <int BM, int BN, int BK, int TM, int TN, typename SRC, typename DST>
static __device__ void matmul_block2d(tinyblasOperation_t transa,  //
                                      tinyblasOperation_t transb,  //
                                      int m, int n, int k,         //
                                      int x, int y,                //
                                      const DST alpha,             //
                                      const SRC *A, int lda,       //
                                      const SRC *B, int ldb,       //
                                      const DST beta,              //
                                      DST *C, int ldc) {
    static_assert((BM % TM == 0) && (BN % TN == 0),
                  "can't divide work for threads");
    static_assert(BK == ((BM * BN) / (TM * TN)),  // optional
                  "threads can't load memory properly");
    static_assert((BM * BN) <= (BM * BK) + (BK * BN),
                  "didn't allocate enough shared mem for threads");
    const int ii0 = threadIdx.x / (BN / TN); /* {0, ..., (BM/TM) - 1} */
    const int ii1 = threadIdx.x % (BN / TN); /* {0, ..., (BN/TN) - 1} */
    extern __shared__ float svals[];  // shared across all threads in a block
    SRC *As = (SRC *)svals;
    SRC *Bs = (SRC *)svals + BM * BK;

    SRC Cs[TM * TN];
    SRC At[TM];
    SRC Bt[TN];
    int i, h, j, l, blob;
    // within each block
    // we first zero out Cs
    for (j = 0; j < TM * TN; ++j) Cs[j] = 0;

    for (blob = 0; blob < k; blob += BK) {
        for (i = threadIdx.x; i < BK; i += blockDim.x) {
            for (j = 0; j < BM + BN; ++j) {
                As[(j * BK) + i] = 0;
            }
        }
        __syncthreads();

        for (i = threadIdx.x; i < BK && blob + i < k; i += blockDim.x) {
            // we copy into As from A
            for (j = 0; j < BM && x + j < m; ++j) {
                As[(i * BM) + j] = READ(A, transa, lda, x + j, blob + i);
            }
        }
        for (i = threadIdx.x; i < BK && blob + i < k; i += blockDim.x) {
            // we copy into Bs from B
            for (j = 0; j < BN && y + j < n; ++j) {
                Bs[(i * BN) + j] = READ(B, transb, ldb, blob + i, y + j);
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
            *(C + (x + j) + (y + l) * ldc) =
                alpha * typechange<SRC, DST>(Cs[j * TN + l]) +
                beta * (*(C + (x + j) + (y + l) * ldc));
        }
    }
    __syncthreads();
}

template <int BM, int BN, int BK, int TM, int TN>
static __global__ void tinyblasS_entry(tinyblasOperation_t transa,  //
                                       tinyblasOperation_t transb,  //
                                       int m, int n, int k,         //
                                       const float alpha,           //
                                       const float *A, int lda,     //
                                       const float *B, int ldb,     //
                                       const float beta,            //
                                       float *C, int ldc) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    // each block handles a sub-matrix of C, of size BM * BN
    // each thread handles a sub-matrix of size TM * TN
    for (x = blockIdx.x * BM; x < m; x += jump1) {
        for (y = blockIdx.y * BN; y < n; y += jump2) {
            matmul_block2d<BM, BN, BK, TM, TN>(transa, transb, m, n, k, x, y,
                                               alpha, A, lda, B, ldb, beta, C,
                                               ldc);
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
static void tinyblasS_wrapper(tinyblasHandle_t stream,     //
                              tinyblasOperation_t transa,  //
                              tinyblasOperation_t transb,  //
                              int m, int n, int k,         //
                              const float alpha,           //
                              const float *A, int lda,     //
                              const float *B, int ldb,     //
                              const float beta,            //
                              float *C, int ldc) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int maxthreads = ((BM * BN) / (TM * TN));

    tinyblasS_entry<BM, BN, BK, TM, TN>
        <<<maxblocks, maxthreads, (sizeof(float) * (BM * BK + BK * BN)),
           stream>>>(transa, transb, m, n, k, A, lda, B, ldb, C, ldc);
}

tinyblasStatus_t tinyblasSgemm(tinyblasHandle_t stream,
                               tinyblasOperation_t transa,
                               tinyblasOperation_t transb, int m, int n, int k,
                               const float *alpha, const float *A, int lda,
                               const float *B, int ldb, const float *beta,
                               float *C, int ldc) {
    tinyblasS_wrapper<32, 8, 128, 1, 2>(stream, transa, transb, m, n, k, *alpha,
                                        A, lda, B, ldb, *beta, C, ldc);
    return TINYBLAS_STATUS_SUCCESS;
}

static inline bool check_args(cudaDataType_t Atype, cudaDataType_t Btype,
                              cudaDataType_t Ctype,
                              tinyblasComputeType_t computeType) {
    return (Atype == CUDA_R_16F && Btype == CUDA_R_16F &&
            (Ctype == CUDA_R_16F || Ctype == CUDA_R_32F) &&
            (computeType == TINYBLAS_COMPUTE_16F ||
             computeType == TINYBLAS_COMPUTE_32F));
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
template <int BM, int BN, int BK, int TM, int TN, typename SRC, typename DST>
static __global__ void tinyblasGE_entry(tinyblasOperation_t transa,  //
                                        tinyblasOperation_t transb,  //
                                        int m, int n, int k,         //
                                        const DST alpha,             //
                                        const SRC *A, int lda,       //
                                        const SRC *B, int ldb,       //
                                        const DST beta,              //
                                        DST *C, int ldc) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;

    // each block handles a sub-matrix of C, of size BM * BN
    for (x = blockIdx.x * BM; x < m; x += jump1) {
        for (y = blockIdx.y * BN; y < n; y += jump2) {
            matmul_block2d<BM, BN, BK, TM, TN, SRC, DST>(
                transa, transb, m, n, k, x, y,  //
                alpha, A, lda, B, ldb,          //
                beta, C, ldc);
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN, typename SRC>
static void tinyblasGE_wrapper(tinyblasHandle_t stream,     //
                               tinyblasOperation_t transa,  //
                               tinyblasOperation_t transb,  //
                               int m, int n, int k,         //
                               const float alpha,           //
                               const SRC *A, int lda,       //
                               const SRC *B, int ldb,       //
                               const float beta,            //
                               void *C, cudaDataType_t Ctype, int ldc) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 1);
    int maxthreads = ((BM * BN) / (TM * TN));

    if (Ctype == CUDA_R_16F) {
        tinyblasGE_entry<BM, BN, BK, TM, TN, SRC, half>
            <<<maxblocks, maxthreads, (sizeof(SRC) * (BM * BK + BK * BN)),
               stream>>>(transa, transb, m, n, k, __float2half(alpha), A, lda,
                         B, ldb, __float2half(beta), (half *)C, ldc);
    } else {
        tinyblasGE_entry<BM, BN, BK, TM, TN, SRC, float>
            <<<maxblocks, maxthreads, (sizeof(SRC) * (BM * BK + BK * BN)),
               stream>>>(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,
                         (float *)C, ldc);
    }
}

tinyblasStatus_t tinyblasGemmEx(tinyblasHandle_t stream,
                                tinyblasOperation_t transa,
                                tinyblasOperation_t transb, int m, int n, int k,
                                const void *alpha, const void *A,
                                cudaDataType_t Atype, int lda, const void *B,
                                cudaDataType_t Btype, int ldb, const void *beta,
                                void *C, cudaDataType_t Ctype, int ldc,
                                tinyblasComputeType_t computeType,
                                tinyblasGemmAlgo_t algo) {
    if (!check_args(Atype, Btype, Ctype, computeType)) {
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    }

    float alpha0, beta0;

    if (computeType == CUBLAS_COMPUTE_32F) {
        alpha0 = *(float *)alpha;
        beta0 = *(float *)beta;
    } else /* computeType == CUBLAS_COMPUTE_16F */ {
        alpha0 = __half2float(*(half *)alpha);
        beta0 = __half2float(*(half *)beta);
    }

    tinyblasGE_wrapper<48, 32, 64, 3, 8, half>(stream, transa, transb, m, n, k,
                                               alpha0, (const half *)A, lda,
                                               (const half *)B, ldb,  //
                                               beta0, C, Ctype, ldc);
    return TINYBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex

template <int BM, int BN, int BK, int TM, int TN, typename SRC, typename DST>
static __global__ void tinyblasGBE_entry(tinyblasOperation_t transa,          //
                                         tinyblasOperation_t transb,          //
                                         int m, int n, int k,                 //
                                         const DST alpha,                     //
                                         const SRC *const Aarray[], int lda,  //
                                         const SRC *const Barray[], int ldb,  //
                                         const DST beta,                      //
                                         void *const Carray[], int ldc,
                                         int batchCount) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;
    int z = blockIdx.z;
    const int jump3 = gridDim.z;

    // each block handles a sub-matrix of C, of size BM * BN
    for (z = blockIdx.z; z < batchCount; z += jump3) {
        for (x = blockIdx.x * BM; x < m; x += jump1) {
            for (y = blockIdx.y * BN; y < n; y += jump2) {
                matmul_block2d<BM, BN, BK, TM, TN, SRC, DST>(
                    transa, transb, m, n, k, x, y, alpha, Aarray[z], lda,
                    Barray[z], ldb, beta, (DST *)(Carray[z]), ldc);
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN, typename SRC>
static void tinyblasGBE_wrapper(tinyblasHandle_t stream,
                                tinyblasOperation_t transa,          //
                                tinyblasOperation_t transb,          //
                                int m, int n, int k,                 //
                                const float alpha,                   //
                                const SRC *const Aarray[], int lda,  //
                                const SRC *const Barray[], int ldb,  //
                                const float beta,                    //
                                void *const Carray[], cudaDataType_t Ctype,
                                int ldc, int batchCount) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int maxthreads = ((BM * BN) / (TM * TN));

    if (Ctype == CUDA_R_16F) {
        tinyblasGBE_entry<BM, BN, BK, TM, TN, SRC>
            <<<maxblocks, maxthreads, (sizeof(SRC) * (BM * BK + BK * BN)),
               stream>>>(transa, transb, m, n, k, __float2half(alpha), Aarray,
                         lda, Barray, ldb, __float2half(beta), Carray, ldc,
                         batchCount);
    } else {
        tinyblasGBE_entry<BM, BN, BK, TM, TN, float>
            <<<maxblocks, maxthreads, (sizeof(SRC) * (BM * BK + BK * BN)),
               stream>>>(transa, transb, m, n, k, alpha, Aarray, lda, Barray,
                         ldb, beta, Carray, ldc, batchCount);
    }
}

tinyblasStatus_t tinyblasGemmBatchedEx(
    tinyblasHandle_t stream,                                                //
    tinyblasOperation_t transa, tinyblasOperation_t transb,                 //
    int m, int n, int k, const void *alpha,                                 //
    const void *const Aarray[], cudaDataType_t Atype, int lda,              //
    const void *const Barray[], cudaDataType_t Btype, int ldb,              //
    const void *beta, void *const Carray[], cudaDataType_t Ctype, int ldc,  //
    int batchCount, tinyblasComputeType_t computeType,
    tinyblasGemmAlgo_t algo) {
    if (!check_args(Atype, Btype, Ctype, computeType)) {
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    }
    float alpha0, beta0;

    if (computeType == CUBLAS_COMPUTE_32F) {
        alpha0 = *(float *)alpha;
        beta0 = *(float *)beta;
    } else /* computeType == CUBLAS_COMPUTE_16F */ {
        alpha0 = __half2float(*(half *)alpha);
        beta0 = __half2float(*(half *)beta);
    }

    tinyblasGBE_wrapper<48, 32, 64, 3, 8, half>(
        stream, transa, transb, m, n, k, (const half **)Aarray, lda,
        (const half **)Barray, ldb, Carray, Ctype, ldc, batchCount);
    return TINYBLAS_STATUS_SUCCESS;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
template <int BM, int BN, int BK, int TM, int TN, typename SRC, typename DST>
static __global__ void tinyblasGSBE_entry(
    tinyblasOperation_t transa, tinyblasOperation_t transb,  //
    int m, int n, int k,                                     //
    const DST alpha,                                         //
    const SRC *A, int lda, long long int strideA,            //
    const SRC *B, int ldb, long long int strideB,            //
    const DST beta,                                          //
    void *C, int ldc, long long int strideC,                 //
    int batchCount) {
    int x = blockIdx.x * BM;
    const int jump1 = gridDim.x * BM;
    int y = blockIdx.y * BN;
    const int jump2 = gridDim.y * BN;
    int z = blockIdx.z;
    const int jump3 = gridDim.z;

    // each block handles a sub-matrix of C, of size BM * BN
    for (z = blockIdx.z; z < batchCount; z += jump3) {
        for (x = blockIdx.x * BM; x < m; x += jump1) {
            for (y = blockIdx.y * BN; y < n; y += jump2) {
                matmul_block2d<BM, BN, BK, TM, TN, SRC, DST>(
                    transa, transb,        //
                    m, n, k, x, y,         //
                    alpha,                 //
                    A + z * strideA, lda,  //
                    B + z * strideB, ldb,  //
                    beta,                  //
                    ((DST *)C + z * strideC), ldc);
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN, typename SRC>
static void tinyblasGSBE_wrapper(tinyblasHandle_t stream,                 //
                                 tinyblasOperation_t transa,              //
                                 tinyblasOperation_t transb,              //
                                 int m, int n, int k,                     //
                                 const float alpha,                       //
                                 const SRC *A, int lda,                   //
                                 long long int strideA,                   //
                                 const SRC *B, int ldb,                   //
                                 long long int strideB,                   //
                                 const float beta,                        //
                                 void *C, cudaDataType_t Ctype, int ldc,  //
                                 long long int strideC, int batchCount) {
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN), 32);
    int maxthreads = ((BM * BN) / (TM * TN));

    if (Ctype == CUDA_R_16F) {
        tinyblasGSBE_entry<BM, BN, BK, TM, TN, SRC, half>
            <<<maxblocks, maxthreads, (sizeof(SRC) * (BM * BK + BK * BN)),
               stream>>>(transa, transb,       //
                         m, n, k,              //
                         __float2half(alpha),  //
                         A, lda, strideA,      //
                         B, ldb, strideB,      //
                         __half2float(beta),   //
                         C, ldc, strideC,      //
                         batchCount);
    } else {
        tinyblasGSBE_entry<BM, BN, BK, TM, TN, SRC, float>
            <<<maxblocks, maxthreads, (sizeof(SRC) * (BM * BK + BK * BN)),
               stream>>>(transa, transb,   //
                         m, n, k,          //
                         alpha,            //
                         A, lda, strideA,  //
                         B, ldb, strideB,  //
                         beta,             //
                         C, ldc, strideC,  //
                         batchCount);
    }
}

tinyblasStatus_t tinyblasGemmStridedBatchedEx(
    tinyblasHandle_t stream, tinyblasOperation_t transa,
    tinyblasOperation_t transb, int m, int n, int k, const void *pAlpha,
    const void *A, cudaDataType_t Atype, int lda, long long int strideA,
    const void *B, cudaDataType_t Btype, int ldb, long long int strideB,
    const void *pBeta, void *C, cudaDataType_t Ctype, int ldc,
    long long int strideC, int batchCount, tinyblasComputeType_t computeType,
    tinyblasGemmAlgo_t algo) {
    if (!check_args(Atype, Btype, Ctype, computeType)) {
        return TINYBLAS_STATUS_NOT_SUPPORTED;
    }
    float alpha0, beta0;

    if (computeType == CUBLAS_COMPUTE_32F) {
        alpha0 = *(float *)alpha;
        beta0 = *(float *)beta;
    } else /* computeType == CUBLAS_COMPUTE_16F */ {
        alpha0 = __half2float(*(half *)alpha);
        beta0 = __half2float(*(half *)beta);
    }

    tinyblasGSBE_wrapper<32, 4, 64, 1, 2, half>(
        stream, transa, transb, m, n, k, alpha0, (const half *)A, lda, strideA,
        (const half *)B, ldb, strideB, beta0, C, Ctype, ldc, strideC,
        batchCount);

    return TINYBLAS_STATUS_SUCCESS;
}

const char *tinyblasGetStatusString(tinyblasStatus_t err) {
    switch (err) {
        case TINYBLAS_STATUS_SUCCESS:
            return "TINYBLAS_STATUS_SUCCESS";
        case TINYBLAS_STATUS_NOT_SUPPORTED:
            return "TINYBLAS_STATUS_NOT_SUPPORTED";
        default:
            return "unknown tinyblas error";
    }
}
