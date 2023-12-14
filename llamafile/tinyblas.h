#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

enum cublasOperation_t {
  CUBLAS_OP_N = 111,
  CUBLAS_OP_T = 112,
  CUBLAS_OP_C = 113,
};

enum cublasStatus_t {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_SUPPORTED = 7,
};

enum cublasComputeType_t {
  CUBLAS_COMPUTE_16F = 150,
};

enum cublasGemmAlgo_t {
  CUBLAS_GEMM_DEFAULT_TENSOR_OP  = 160,
};

#define cublasHandle_t cudaStream_t

cublasStatus_t tinyblasSgemm(cublasHandle_t handle,
                             cublasOperation_t transa,
                             cublasOperation_t transb,
                             int m, int n, int k,
                             const float           *alpha,
                             const float           *A, int lda,
                             const float           *B, int ldb,
                             const float           *beta,
                             float           *C, int ldc);

cublasStatus_t tinyblasGemmEx(cublasHandle_t handle,
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
                              cublasGemmAlgo_t algo);

cublasStatus_t tinyblasGemmBatchedEx(cublasHandle_t handle,
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
                                     cublasGemmAlgo_t algo);

cublasStatus_t tinyblasGemmStridedBatchedEx(cublasHandle_t handle,
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
                                            cublasGemmAlgo_t algo);
