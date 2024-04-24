#pragma once
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_compute_params;

bool llamafile_sgemm_unsupported(int, int, int, const void *, int, const void *, int, void *, int,
                                 int, int, int, int, int, int);
bool llamafile_sgemm_amd_avx(int, int, int, const void *, int, const void *, int, void *, int, int,
                             int, int, int, int, int);
bool llamafile_sgemm_amd_fma(int, int, int, const void *, int, const void *, int, void *, int, int,
                             int, int, int, int, int);
bool llamafile_sgemm_amd_avx2(int, int, int, const void *, int, const void *, int, void *, int, int,
                              int, int, int, int, int);
bool llamafile_sgemm_amd_avxvnni(int, int, int, const void *, int, const void *, int, void *, int,
                                 int, int, int, int, int, int);
bool llamafile_sgemm_amd_avx512f(int, int, int, const void *, int, const void *, int, void *, int,
                                 int, int, int, int, int, int);
bool llamafile_sgemm_amd_zen4(int, int, int, const void *, int, const void *, int, void *, int, int,
                              int, int, int, int, int);
bool llamafile_sgemm_arm80(int, int, int, const void *, int, const void *, int, void *, int, int,
                           int, int, int, int, int);
bool llamafile_sgemm_arm82(int, int, int, const void *, int, const void *, int, void *, int, int,
                           int, int, int, int, int);

bool llamafile_mixmul_unsupported(const struct ggml_compute_params *, const struct ggml_tensor *,
                                  const struct ggml_tensor *, const struct ggml_tensor *,
                                  struct ggml_tensor *);
bool llamafile_mixmul_amd_avx(const struct ggml_compute_params *, const struct ggml_tensor *,
                              const struct ggml_tensor *, const struct ggml_tensor *,
                              struct ggml_tensor *);
bool llamafile_mixmul_amd_fma(const struct ggml_compute_params *, const struct ggml_tensor *,
                              const struct ggml_tensor *, const struct ggml_tensor *,
                              struct ggml_tensor *);
bool llamafile_mixmul_amd_avx2(const struct ggml_compute_params *, const struct ggml_tensor *,
                               const struct ggml_tensor *, const struct ggml_tensor *,
                               struct ggml_tensor *);
bool llamafile_mixmul_amd_avxvnni(const struct ggml_compute_params *, const struct ggml_tensor *,
                                  const struct ggml_tensor *, const struct ggml_tensor *,
                                  struct ggml_tensor *);
bool llamafile_mixmul_amd_avx512f(const struct ggml_compute_params *, const struct ggml_tensor *,
                                  const struct ggml_tensor *, const struct ggml_tensor *,
                                  struct ggml_tensor *);
bool llamafile_mixmul_amd_zen4(const struct ggml_compute_params *, const struct ggml_tensor *,
                               const struct ggml_tensor *, const struct ggml_tensor *,
                               struct ggml_tensor *);
bool llamafile_mixmul_arm80(const struct ggml_compute_params *, const struct ggml_tensor *,
                            const struct ggml_tensor *, const struct ggml_tensor *,
                            struct ggml_tensor *);
bool llamafile_mixmul_arm82(const struct ggml_compute_params *, const struct ggml_tensor *,
                            const struct ggml_tensor *, const struct ggml_tensor *,
                            struct ggml_tensor *);

#ifdef __cplusplus
}
#endif
