#pragma once

#ifdef GGML_USE_HIPBLAS
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

enum tinyblasOperation_t {
  TINYBLAS_OP_N,
  TINYBLAS_OP_T,
  TINYBLAS_OP_C,
};

enum tinyblasStatus_t {
  TINYBLAS_STATUS_SUCCESS,
  TINYBLAS_STATUS_NOT_SUPPORTED,
};

enum tinyblasComputeType_t {
  TINYBLAS_COMPUTE_16F,
  TINYBLAS_COMPUTE_32F,
};

enum tinyblasGemmAlgo_t {
  TINYBLAS_GEMM_DEFAULT_TENSOR_OP,
};

#define tinyblasHandle_t cudaStream_t

const char *tinyblasGetStatusString(tinyblasStatus_t);

tinyblasStatus_t tinyblasSgemm(tinyblasHandle_t handle,
                               tinyblasOperation_t transa,
                               tinyblasOperation_t transb,
                               int m, int n, int k,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *B, int ldb,
                               const float           *beta,
                               float           *C, int ldc);

tinyblasStatus_t tinyblasGemmEx(tinyblasHandle_t handle,
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
                                tinyblasGemmAlgo_t algo);

tinyblasStatus_t tinyblasGemmBatchedEx(tinyblasHandle_t handle,
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
                                       tinyblasGemmAlgo_t algo);

tinyblasStatus_t tinyblasGemmStridedBatchedEx(tinyblasHandle_t handle,
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
                                              tinyblasGemmAlgo_t algo);
