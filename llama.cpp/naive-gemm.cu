#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define MULZERO(X, Y) (fabs((X)) > 0 ? (X) * (Y) : 0.0)
#define READ0(A, trans, ld, i, j) \
  (((trans) == CUBLAS_OP_N) ? (A)[(i) + (j) * (ld)] : (A)[(j) + (i) * (ld)])
#define READ(A, type, trans, ld, i, j) \
  ((type) == CUDA_R_16F                                         \
   ? __half2float(READ0((half *)(A), (trans), (ld), (i), (j)))  \
   : READ0((float *)(A), (trans), (ld), (i), (j)))

static __device__ __forceinline__ void matmul(cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              float               alpha,
                                              const void          *A,
                                              cublasDataType_t    Atype,
                                              int lda,
                                              const void          *B,
                                              cublasDataType_t    Btype,
                                              int ldb,
                                              float               beta,
                                              void                *C,
                                              cublasDataType_t    Ctype,
                                              int ldc) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0;
      for (int l = 0; l < k; ++l) {
        sum += READ(A, Atype, transa, lda, i, l) *
               READ(B, Btype, transb, ldb, l, j);
      }
      if (Ctype == CUDA_R_16F) {
        half *cptr = (half *)C + i + ldc * j;
        *cptr = __float2half(MULZERO(alpha, sum) +
                             MULZERO(beta, __half2float(*cptr)));
      } else {
        float *cptr = (float *)C + i + ldc * j;
        *cptr = MULZERO(alpha, sum) + MULZERO(beta, *cptr);
      }
    }
  }
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex

#define STRIDE0(A, i, stride) ((A) + (i) * (stride))
#define STRIDE(A, type, i, stride)                            \
  ((type) == CUDA_R_16F                                       \
   ? (void *)STRIDE0((half *)(A), (i), (stride))              \
   : (void *)STRIDE0((float *)(A), (i), (stride)))
#define STRIDE_CONST(A, type, i, stride)                      \
  ((type) == CUDA_R_16F                                       \
   ? (const void *)STRIDE0((const half *)(A), (i), (stride))  \
   : (const void *)STRIDE0((const float *)(A), (i), (stride)))

static __global__ void cublasGSBE_entry(cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m, int n, int k,
                                        float           alpha,
                                        const void      *A,
                                        cudaDataType_t  Atype,
                                        int             lda,
                                        long long int   strideA,
                                        const void      *B,
                                        cudaDataType_t  Btype,
                                        int             ldb,
                                        long long int   strideB,
                                        float           beta,
                                        void            *C,
                                        cudaDataType_t  Ctype,
                                        int             ldc,
                                        long long int   strideC,
                                        int batchCount) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int jump = blockDim.x * gridDim.x;

  const void *A_x;
  const void *B_x;
  void *C_x;

  for (; x < batchCount; x += jump) {
    A_x = STRIDE_CONST(A, Atype, x, strideA);
    B_x = STRIDE_CONST(B, Btype, x, strideB);
    C_x = STRIDE(C, Ctype, x, strideC);
    matmul(transa, transb, m, n, k, alpha, A_x, Atype, lda, B_x, Btype, ldb,
           beta, C_x, Ctype, ldc);
  }
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
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
  if ((Atype != CUDA_R_16F && Atype != CUDA_R_32F) ||
      (Btype != CUDA_R_16F && Btype != CUDA_R_32F) ||
      (Ctype != CUDA_R_16F && Ctype != CUDA_R_32F) ||
      (transa != CUBLAS_OP_N && transa != CUBLAS_OP_T) ||
      (transb != CUBLAS_OP_N && transb != CUBLAS_OP_T)) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }

  float alpha, beta;
  switch (computeType) {
    case CUBLAS_COMPUTE_16F:
      beta = __half2float(*(half *)pBeta);
      alpha = __half2float(*(half *)pAlpha);
      break;
    case CUBLAS_COMPUTE_32F:
      beta = *(float *)pBeta;
      alpha = *(float *)pAlpha;
      break;
    default:
      return CUBLAS_STATUS_NOT_SUPPORTED;
  }

  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  int numSMs, devId;
  cudaGetDevice(&devId);
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
  int maxblocks = 16 * numSMs;
  int maxthreads = 128;

  // call the entry function
  cublasGSBE_entry<<<maxblocks, maxthreads>>>(transa, transb, m, n, k, alpha,
                                              A, Atype, lda, strideA, B, Btype,
                                              ldb, strideB, beta, C, Ctype,
                                              ldc, strideC, batchCount);

  return CUBLAS_STATUS_SUCCESS;
}
