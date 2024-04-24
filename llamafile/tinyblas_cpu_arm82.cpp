#ifdef __aarch64__
#define llamafile_sgemm llamafile_sgemm_arm82
#define llamafile_mixmul llamafile_mixmul_arm82
#include "tinyblas_cpu.inc"
#endif // __aarch64__
