#pragma once
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm_sss_avx(int, int, int, const float *, int, const float *, int, float *, int,
                             int, int, int);
bool llamafile_sgemm_sss_fma(int, int, int, const float *, int, const float *, int, float *, int,
                             int, int, int);
bool llamafile_sgemm_sss_avx512f(int, int, int, const float *, int, const float *, int, float *,
                                 int, int, int, int);

bool llamafile_sgemm_hss_f16c(int, int, int, const unsigned short *, int, const float *, int,
                              float *, int, int, int, int);
bool llamafile_sgemm_hss_avx512f(int, int, int, const unsigned short *, int, const float *, int,
                                 float *, int, int, int, int);

bool llamafile_sgemm_qqs_avx512vnni(int, int, int, const block_q8_0 *, int, const block_q8_0 *, int,
                                    float *, int, int, int, int);
bool llamafile_sgemm_qqs_avxvnni(int, int, int, const block_q8_0 *, int, const block_q8_0 *, int,
                                 float *, int, int, int, int);
bool llamafile_sgemm_qqs_fma(int, int, int, const block_q8_0 *, int, const block_q8_0 *, int,
                             float *, int, int, int, int);

bool llamafile_sgemm_eqs_avx512vnni(int, int, int, const block_q4_0 *, int, const block_q8_0 *, int,
                                    float *, int, int, int, int);
bool llamafile_sgemm_eqs_avxvnni(int, int, int, const block_q4_0 *, int, const block_q8_0 *, int,
                                 float *, int, int, int, int);
bool llamafile_sgemm_eqs_fma(int, int, int, const block_q4_0 *, int, const block_q8_0 *, int,
                             float *, int, int, int, int);

#ifdef __cplusplus
}
#endif
