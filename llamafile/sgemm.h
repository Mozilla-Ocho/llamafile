#pragma once
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_compute_params;

bool llamafile_sgemm_unsupported(long, long, long, const void *, long, const void *, long, void *,
                                 long, int, int, int, int, int, int, int);
bool llamafile_sgemm_amd_avx(long, long, long, const void *, long, const void *, long, void *, long,
                             int, int, int, int, int, int, int);
bool llamafile_sgemm_amd_fma(long, long, long, const void *, long, const void *, long, void *, long,
                             int, int, int, int, int, int, int);
bool llamafile_sgemm_amd_avx2(long, long, long, const void *, long, const void *, long, void *,
                              long, int, int, int, int, int, int, int);
bool llamafile_sgemm_amd_avxvnni(long, long, long, const void *, long, const void *, long, void *,
                                 long, int, int, int, int, int, int, int);
bool llamafile_sgemm_amd_avx512f(long, long, long, const void *, long, const void *, long, void *,
                                 long, int, int, int, int, int, int, int);
bool llamafile_sgemm_amd_zen4(long, long, long, const void *, long, const void *, long, void *,
                              long, int, int, int, int, int, int, int);
bool llamafile_sgemm_arm80(long, long, long, const void *, long, const void *, long, void *, long,
                           int, int, int, int, int, int, int);
bool llamafile_sgemm_arm82(long, long, long, const void *, long, const void *, long, void *, long,
                           int, int, int, int, int, int, int);

bool llamafile_mixmul_unsupported(struct ggml_compute_params *, const struct ggml_tensor *,
                                  const struct ggml_tensor *, const struct ggml_tensor *,
                                  struct ggml_tensor *);
bool llamafile_mixmul_amd_avx(struct ggml_compute_params *, const struct ggml_tensor *,
                              const struct ggml_tensor *, const struct ggml_tensor *,
                              struct ggml_tensor *);
bool llamafile_mixmul_amd_fma(struct ggml_compute_params *, const struct ggml_tensor *,
                              const struct ggml_tensor *, const struct ggml_tensor *,
                              struct ggml_tensor *);
bool llamafile_mixmul_amd_avx2(struct ggml_compute_params *, const struct ggml_tensor *,
                               const struct ggml_tensor *, const struct ggml_tensor *,
                               struct ggml_tensor *);
bool llamafile_mixmul_amd_avxvnni(struct ggml_compute_params *, const struct ggml_tensor *,
                                  const struct ggml_tensor *, const struct ggml_tensor *,
                                  struct ggml_tensor *);
bool llamafile_mixmul_amd_avx512f(struct ggml_compute_params *, const struct ggml_tensor *,
                                  const struct ggml_tensor *, const struct ggml_tensor *,
                                  struct ggml_tensor *);
bool llamafile_mixmul_amd_zen4(struct ggml_compute_params *, const struct ggml_tensor *,
                               const struct ggml_tensor *, const struct ggml_tensor *,
                               struct ggml_tensor *);
bool llamafile_mixmul_arm80(struct ggml_compute_params *, const struct ggml_tensor *,
                            const struct ggml_tensor *, const struct ggml_tensor *,
                            struct ggml_tensor *);
bool llamafile_mixmul_arm82(struct ggml_compute_params *, const struct ggml_tensor *,
                            const struct ggml_tensor *, const struct ggml_tensor *,
                            struct ggml_tensor *);

#ifdef __cplusplus
}
#endif
