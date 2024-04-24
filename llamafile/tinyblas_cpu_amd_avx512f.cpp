#ifdef __x86_64__
#define llamafile_sgemm llamafile_sgemm_amd_avx512f
#define llamafile_mixmul llamafile_mixmul_amd_avx512f
#include "tinyblas_cpu.inc"
#endif // __x86_64__
