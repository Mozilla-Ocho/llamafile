#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// XXX: this currently needs to be hard-coded
//      be sure to pass llamafile -t 16
#define NTHREADS 16

void llama_matmul_avx2_bf16(const uint16_t *A, const uint16_t *B, float *C, const int M,
                            const int N, const int K, const int ith);

#ifdef __cplusplus
}
#endif
