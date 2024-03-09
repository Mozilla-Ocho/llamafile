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

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// In this file you'll find GPU subroutines implementing general matrix
// multiplication, that are API compatible with NVIDIA's cuBLAS library
// and nearly as fast[1] too. This is important because how can we call
// our software open source if it spends most of its time inside NVIDIA
// proprietary blobs like cuBLAS? tinyBLAS provides a free, open, libre
// alternative to cuBLAS that's orders of a magnitude tinier (cuBLAS is
// a 500mb DSO) and goes fast enough that you won't sacrifice much. AMD
// users might even prefer tinyBLAS, since outperforming hipBLAS is not
// very difficult for large matrices. tinyBLAS also has better accuracy
// too, since hipBLAS uses tricks that cause sign flips, and denormals.
//
// TODO(jart): make tinyBLAS go fast for skinny matrices
//
// [1] S. Boehm, ‘How to Optimize a CUDA Matmul Kernel for cuBLAS-like
//     Performance’, 2022. [Online]. Available:
//     https://siboehm.com/articles/22/CUDA-MMM. [Accessed:
//     05-Mar-2024].

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

#define WARPSIZE 32
#define THREAD_COUNT ((BM * BN) / (TM * TN))
#define KERNEL __launch_bounds__(THREAD_COUNT)
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

////////////////////////////////////////////////////////////////////////////////////////////////////
// tinyBLAS block tiling outer product GEMM kernel

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

////////////////////////////////////////////////////////////////////////////////////////////////////
// tinyBLAS warp block tiling outer product GEMM kernel

enum Mode {
    GENERAL,
    SIMPLE,
};

template <enum Mode MODE, int BM, int BN, int BK, int WM, int WN, int WNI, int TM, int TN, int TT,
          typename WORD, typename SRC, typename DST>
static __device__ void matmul_warp2d(tinyblasOperation_t aT, //
                                     tinyblasOperation_t bT, //
                                     int m, int n, int k, WORD alpha, //
                                     const SRC *A, int lda, //
                                     const SRC *B, int ldb, WORD beta, //
                                     DST *C, int ldc) {

    const int warpIdx = threadIdx.x / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    constexpr int WARPS = TT / WARPSIZE;
    constexpr int WMI = (WM * WN) / (WARPSIZE * TM * TN * WNI);
    constexpr int WSUBM = WM / WMI;
    constexpr int WSUBN = WN / WNI;
    constexpr int VE = sizeof(float4) / sizeof(SRC);

    const int threadIdxInWarp = threadIdx.x % WARPSIZE;
    const int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    // want to tune these magnums?
    // use llamafile/pick_a_warp_kernel.c
    static_assert(!(BN % WN) && !(BM % WM), "");
    static_assert((BN / WN) * (BM / WM) == WARPS, "");
    static_assert(!((WM * WN) % (WARPSIZE * TM * TN * WNI)), "");
    static_assert(BN % (sizeof(float4) * TN) == 0, "");
    static_assert(BM % (sizeof(float4) * TM) == 0, "");
    static_assert(!(WM % WMI) && !(WN % WNI), "");
    static_assert(!((BM * BK) % (VE * TT)), "");
    static_assert(!((BN * BK) % (VE * TT)), "");

    __shared__ SRC As[BK * BM];
    __shared__ SRC Bs[BK * BN];

    WORD Ar[WMI * TM] = {0};
    WORD Br[WNI * TN] = {0};
    WORD Cr[WMI * TM * WNI * TN] = {0};

    for (int bkIdx = 0; bkIdx < k; bkIdx += BK) {

        for (int h = 0; h < BM; h += (TT * VE) / BK)
            for (int v = 0; v < VE; ++v) {
                int l = bkIdx + threadIdx.x % (BK / VE) * VE + v;
                int i = blockIdx.y * BM + threadIdx.x / (BK / VE) + h;
                As[BM * (threadIdx.x % (BK / VE) * VE + v) + (threadIdx.x / (BK / VE) + h)] =
                    aT && MODE <= GENERAL
                        ? ((MODE == SIMPLE ? i < m : (l < k && i < m)) ? A[lda * l + i] : (SRC)0)
                        : ((MODE == SIMPLE ? i < m : (l < k && i < m)) ? A[lda * i + l] : (SRC)0);
            }

        for (int h = 0; h < BK; h += TT / (BN / VE))
            for (int v = 0; v < VE; ++v) {
                int l = bkIdx + threadIdx.x / (BN / VE) + h;
                int j = blockIdx.x * BN + threadIdx.x % (BN / VE) * VE + v;
                Bs[BN * (threadIdx.x / (BN / VE) + h) + (threadIdx.x % (BN / VE) * VE + v)] =
                    bT || MODE >= SIMPLE
                        ? (MODE == SIMPLE || (l < k && j < n) ? B[ldb * j + l] : (SRC)0)
                        : (MODE == SIMPLE || (l < k && j < n) ? B[ldb * l + j] : (SRC)0);
            }

        __syncthreads();

        for (int l = 0; l < BK; ++l) {
            for (int ii = 0; ii < WMI; ++ii)
                for (int i = 0; i < TM; ++i)
                    Ar[TM * ii + i] =
                        As[BM * l + WM * warpRow + WSUBM * ii + TM * threadRowInWarp + i];
            for (int jj = 0; jj < WNI; ++jj)
                for (int j = 0; j < TN; ++j)
                    Br[TN * jj + j] =
                        Bs[BN * l + WN * warpCol + WSUBN * jj + TN * threadColInWarp + j];
            for (int ii = 0; ii < WMI; ++ii)
                for (int jj = 0; jj < WNI; ++jj)
                    for (int i = 0; i < TM; ++i)
                        for (int j = 0; j < TN; ++j)
                            Cr[(WNI * TN) * (TM * ii + i) + (TN * jj) + j] +=
                                Ar[TM * ii + i] * Br[TN * jj + j];
        }

        __syncthreads();
    }

    for (int wSubRowIdx = 0; wSubRowIdx < WMI; ++wSubRowIdx)
        for (int wSubColIdx = 0; wSubColIdx < WNI; ++wSubColIdx)
            for (int resIdxM = 0; resIdxM < TM; resIdxM += 1)
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 1) {
                    int row = (BM * blockIdx.y + WM * warpRow) + (WSUBM * wSubRowIdx) +
                              (threadRowInWarp * TM + resIdxM);
                    int col = (BN * blockIdx.x + WN * warpCol) + (WSUBN * wSubColIdx) +
                              (threadColInWarp * TN + resIdxN);
                    if (MODE == SIMPLE) {
                        if (row < m)
                            C[ldc * row + col] = Cr[(WNI * TN) * (TM * wSubRowIdx + resIdxM) +
                                                    TN * wSubColIdx + resIdxN];
                    } else {
                        if (row < m && col < n)
                            C[ldc * row + col] =
                                alpha * Cr[(WNI * TN) * (TM * wSubRowIdx + resIdxM) +
                                           TN * wSubColIdx + resIdxN] +
                                beta * (WORD)C[ldc * row + col];
                    }
                }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// tinyBLAS canonical cuBLAS-like interface

struct tinyblasContext {
    cudaStream_t stream;
};

/**
 * Creates new tinyBLAS handle.
 *
 * Before calling tinyBLAS GEMM functions a handle must first be
 * created, using this function. It should be freed later, using
 * tinyblasDestroy(). After a handle is created the caller needs
 * tinyblasSetStream() to specify the CUDA stream.
 *
 * @param out_handle receives pointer to newly created handle
 * @return TINYBLAS_STATUS_SUCCESS on success otherwise error
 */
tinyblasStatus_t tinyblasCreate(tinyblasHandle_t *out_handle) {
    tinyblasHandle_t handle;
    if ((handle = (tinyblasHandle_t)malloc(sizeof(struct tinyblasContext)))) {
        *out_handle = handle;
        return TINYBLAS_STATUS_SUCCESS;
    } else {
        return TINYBLAS_STATUS_ALLOC_FAILED;
    }
}

/**
 * Destroys tinyBLAS handle.
 *
 * @param handle is pointer to handle created by tinyblasCreate()
 * @return TINYBLAS_STATUS_SUCCESS on success otherwise error
 */
tinyblasStatus_t tinyblasDestroy(tinyblasHandle_t handle) {
    free(handle);
    return TINYBLAS_STATUS_SUCCESS;
}

/**
 * Associates CUDA handle with tinyBLAS handle.
 *
 * The provided stream will be used when tinyBLAS launches kernels.
 *
 * @param handle is pointer to handle created by tinyblasCreate()
 * @param stream is pointer to stream created by cudaStreamCreate()
 * @return TINYBLAS_STATUS_SUCCESS on success otherwise error
 */
tinyblasStatus_t tinyblasSetStream(tinyblasHandle_t handle, void *stream) {
    handle->stream = (cudaStream_t)stream;
    return TINYBLAS_STATUS_SUCCESS;
}

/**
 * Gets CUDA stream associated with tinyBLAS handle.
 *
 * @param handle is pointer to handle created by tinyblasCreate()
 * @param out_stream receives pointer to any cudaStream_t object
 * @return TINYBLAS_STATUS_SUCCESS on success otherwise error
 */
tinyblasStatus_t tinyblasGetStream(tinyblasHandle_t handle, void **out_stream) {
    *out_stream = handle->stream;
    return TINYBLAS_STATUS_SUCCESS;
}

/**
 * Returns string describing tinyBLAS status code.
 */
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

/**
 * Performs single-precision general matrix multiplication.
 *
 * This is a column major GEMM subroutine for computing C = α*A*B + β*C.
 *
 * @param handle was created by tinyblasCreate()
 * @param transa if `A` should be transposed
 * @param transb if `B` should be transposed
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param alpha points to scalar that's multiplied against input
 * @param A is input array of first matrix
 * @param lda is row stride of `A`
 * @param B is input array of second matrix
 * @param ldb is row stride of `B`
 * @param beta points to scalar that's multiplied against existing output
 * @param C is input/output array of output matrix
 * @param ldc is row stride of `C`
 */
tinyblasStatus_t tinyblasSgemm(tinyblasHandle_t handle, tinyblasOperation_t transa,
                               tinyblasOperation_t transb, int m, int n, int k, const float *alpha,
                               const float *A, int lda, const float *B, int ldb, const float *beta,
                               float *C, int ldc) {
    return tinyblasGemmEx(handle, transa, transb, m, n, k, alpha, A, TINYBLAS_R_32F, lda, B,
                          TINYBLAS_R_32F, ldb, beta, C, TINYBLAS_R_32F, ldc, TINYBLAS_COMPUTE_32F,
                          TINYBLAS_GEMM_DEFAULT);
}

template <enum Mode MODE, int BM, int BN, int BK, int WM, int WN, int WNI, int TM, int TN, int TT,
          typename WORD, typename SRC, typename DST>
static __global__ void __launch_bounds__(TT) tinyblasGE_entry(tinyblasOperation_t aT, //
                                                              tinyblasOperation_t bT, //
                                                              int m, int n, int k, WORD alpha, //
                                                              const SRC *A, int lda, //
                                                              const SRC *B, int ldb, //
                                                              WORD beta, DST *C, int ldc) {
    matmul_warp2d<MODE, BM, BN, BK, WM, WN, WNI, TM, TN, TT>(aT, bT, m, n, k, alpha, A, lda, B, ldb,
                                                             beta, C, ldc);
}

template <typename WORD, typename SRC, typename DST>
static tinyblasStatus_t tinyblasGE_launcher(tinyblasHandle_t handle, tinyblasOperation_t aT,
                                            tinyblasOperation_t bT, int m, int n, int k, WORD alpha,
                                            const SRC *A, int lda, const SRC *B, int ldb, WORD beta,
                                            DST *C, int ldc) {
    const int TT = 256, BM = 128, BN = 64, BK = 64, WM = 128, WN = 8, WNI = 1, TM = 8, TN = 4;
    dim3 maxblocks(CEIL_DIV(m, BM), CEIL_DIV(n, BN));
    if (!aT && bT && !(n % BN) && !(k % BK) && alpha == (WORD)1 && beta == (WORD)0) {
        dim3 blocks(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
        tinyblasGE_entry<SIMPLE, BM, BN, BK, WM, WN, WNI, TM, TN, TT>
            <<<blocks, TT, 0, handle->stream>>>(aT, bT, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                                ldc);
    } else {
        dim3 blocks(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
        tinyblasGE_entry<GENERAL, BM, BN, BK, WM, WN, WNI, TM, TN, TT>
            <<<blocks, TT, 0, handle->stream>>>(aT, bT, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                                ldc);
    }
    if (cudaGetLastError() != cudaSuccess)
        return TINYBLAS_STATUS_EXECUTION_FAILED;
    return TINYBLAS_STATUS_SUCCESS;
}

template <typename WORD, typename SRC, typename DST>
static tinyblasStatus_t tinyblasGE_launch(tinyblasHandle_t handle, tinyblasOperation_t aT,
                                          tinyblasOperation_t bT, int m, int n, int k, WORD alpha,
                                          const SRC *A, int lda, const SRC *B, int ldb, WORD beta,
                                          DST *C, int ldc) {
    return tinyblasGE_launcher(handle, bT, aT, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
}

/**
 * Performs extended general matrix multiplication.
 *
 * This is a column major GEMM subroutine for computing C = α*A*B + β*C.
 *
 * @param handle was created by tinyblasCreate()
 * @param transa if `A` should be transposed
 * @param transb if `B` should be transposed
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param alpha points to scalar that's multiplied against input
 * @param A is input array of first matrix
 * @param Atype is data type of `C`
 * @param lda is row stride of `A`
 * @param B is input array of second matrix
 * @param Btype is data type of `C`
 * @param ldb is row stride of `B`
 * @param beta points to scalar that's multiplied against existing output
 * @param C is input/output array of output matrix
 * @param Ctype is data type of `C`
 * @param ldc is row stride of `C`
 * @param computeType is data type of `alpha`, `beta`, and dot product
 * @param algo specifies algorithm to use
 */
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
    if (algo != TINYBLAS_GEMM_DEFAULT)
        return TINYBLAS_STATUS_INVALID_VALUE;
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

tinyblasStatus_t tinyblasGemmBatchedEx(tinyblasHandle_t handle, tinyblasOperation_t transa,
                                       tinyblasOperation_t transb, int m, int n, int k,
                                       const void *alpha, const void *const Aarray[],
                                       tinyblasDataType_t Atype, int lda,
                                       const void *const Barray[], tinyblasDataType_t Btype,
                                       int ldb, const void *beta, void *const Carray[],
                                       tinyblasDataType_t Ctype, int ldc, int batchCount,
                                       tinyblasComputeType_t computeType, tinyblasGemmAlgo_t algo) {

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
    if (algo != TINYBLAS_GEMM_DEFAULT)
        return TINYBLAS_STATUS_INVALID_VALUE;
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
    if (algo != TINYBLAS_GEMM_DEFAULT)
        return TINYBLAS_STATUS_INVALID_VALUE;
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
