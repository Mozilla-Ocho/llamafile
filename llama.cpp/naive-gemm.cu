#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define READ(A, trans, ld, i, j) \
  __half2float(((trans) == CUBLAS_OP_N) \
                 ? (A)[(i) + (j) * (ld)] : (A)[(j) + (i) * (ld)])

static __device__ __forceinline__ void matmul(int m, int n, int k,
                                              const half *A, int lda,
                                              const half *B, int ldb,
                                              half       *C, int ldc) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0;
      half *cptr = C + i + j * ldc;
      for (int l = 0; l < k; ++l) {
        sum += READ(A, CUBLAS_OP_T, lda, i, l) *
               READ(B, CUBLAS_OP_N, ldb, l, j);
      }
      *cptr = __float2half(sum);
    }
  }
}

static __global__ void wrap_matmul(int m, int n, int k, const half *A, int lda,
                                   const half *B, int ldb, half *C, int ldc) {
  matmul(m, n, k, A, lda, B, ldb, C, ldc);
}

static bool check_args(cublasOperation_t transa, cublasOperation_t transb,
                       const void *pAlpha, cudaDataType_t Atype,
                       cudaDataType_t Btype, const void *pBeta,
                       cudaDataType_t Ctype, cublasComputeType_t computeType) {
  return transa == CUBLAS_OP_T && transb == CUBLAS_OP_N &&
    Atype == CUDA_R_16F && Btype == CUDA_R_16F && Ctype == CUDA_R_16F &&
    computeType == CUBLAS_COMPUTE_16F &&
    __half2float(*(half *)pAlpha) == 1.0f &&
    __half2float(*(half *)pBeta) == 0.0f;
}

// https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex

cublasStatus_t cublasGemmEx(cublasHandle_t handle,
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

  cudaStream_t stream;
  cublasGetStream(handle, &stream);
  wrap_matmul<<<1, 1, 0, stream>>>(
      m, n, k, (const half*)A, lda, (const half *)B, ldb, (half *)C, ldc);
  return CUBLAS_STATUS_SUCCESS;
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

static __global__ void cublasGSBE_entry(int m, int n, int k,
                                        const half      *A,
                                        int             lda,
                                        long long int   strideA,
                                        const half      *B,
                                        int             ldb,
                                        long long int   strideB,
                                        half            *C,
                                        int             ldc,
                                        long long int   strideC,
                                        int batchCount) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int jump = blockDim.x * gridDim.x;

  for (; x < batchCount; x += jump) {
    matmul(m, n, k, A + x * strideA, lda, B + x * strideB, ldb, C + x * strideC,
           ldc);
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
  if (!check_args(transa, transb, pAlpha, Atype, Btype, pBeta, Ctype,
                  computeType)) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }

  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  int numSMs, devId;
  cudaGetDevice(&devId);
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
  int maxblocks = 16 * numSMs;
  int maxthreads = 128;

  // call the entry function
  cublasGSBE_entry<<<maxblocks, maxthreads, 0, stream>>>(
      m, n, k, (const half*)A, lda, strideA, (const half*)B, ldb, strideB,
      (half *)C, ldc, strideC, batchCount);

  return CUBLAS_STATUS_SUCCESS;
}
