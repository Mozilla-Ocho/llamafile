#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// XXX: this currently needs to be hard-coded
//      be sure to pass llamafile -t 16
#define NTHREADS 96

void llama_matmul_avx2_fp32(const float *A, const float *B, float *C, const int M, const int N,
                            const int K, const long lda, const long ldb, const long ldc,
                            const int ith);
void llama_matmul_avx2_fp16(const uint16_t *A, const uint16_t *B, float *C, const int M,
                            const int N, const int K, const long lda, const long ldb,
                            const long ldc, const int ith);
void llama_matmul_avx2_bf16(const uint16_t *A, const uint16_t *B, float *C, const int M,
                            const int N, const int K, const long lda, const long ldb,
                            const long ldc, const int ith);

#ifdef __cplusplus
}
#endif
