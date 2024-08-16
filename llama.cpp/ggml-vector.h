#pragma once
#include "ggml.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef double ggml_float;

void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n);
void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n);
void ggml_bf16_to_fp32_row(const ggml_bf16_t * x, float * y, int64_t n);
void ggml_fp32_to_bf16_row(const float * x, ggml_bf16_t * y, int64_t n);
void ggml_fp32_to_bf16_row_ref(const float * x, ggml_bf16_t * y, int64_t n);
void ggml_vec_dot_f32(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
void ggml_vec_dot_f16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
void ggml_vec_dot_bf16(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
void ggml_vec_dot_f16_unroll(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
void ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
void ggml_vec_set_i8(const int n, int8_t * x, const int8_t v);
void ggml_vec_set_i16(const int n, int16_t * x, const int16_t v);
void ggml_vec_set_i32(const int n, int32_t * x, const int32_t v);
void ggml_vec_set_f16(const int n, ggml_fp16_t * x, const int32_t v);
void ggml_vec_set_bf16(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y);
void ggml_vec_add1_f32(const int n, float * z, const float * x, const float   v);
void ggml_vec_acc_f32 (const int n, float * y, const float * x);
void ggml_vec_acc1_f32(const int n, float * y, const float   v);
void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y);
void ggml_vec_set_f32 (const int n, float * x, const float   v);
void ggml_vec_cpy_f32 (const int n, float * y, const float * x);
void ggml_vec_neg_f32 (const int n, float * y, const float * x);
void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y);
void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y);
void ggml_vec_scale_f32(const int n, float * y, const float   v);
void ggml_vec_scale_f16(const int n, ggml_fp16_t * y, const float v);
void ggml_vec_mad_f32(const int n, float * y, const float * x, const float v);
void ggml_vec_mad_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
void ggml_vec_norm_f32 (const int n, float * s, const float * x);
void ggml_vec_sqr_f32  (const int n, float * y, const float * x);
void ggml_vec_sqrt_f32 (const int n, float * y, const float * x);
void ggml_vec_log_f32  (const int n, float * y, const float * x);
void ggml_vec_abs_f32  (const int n, float * y, const float * x);
void ggml_vec_sgn_f32  (const int n, float * y, const float * x);
void ggml_vec_step_f32 (const int n, float * y, const float * x);
void ggml_vec_tanh_f32 (const int n, float * y, const float * x);
void ggml_vec_elu_f32  (const int n, float * y, const float * x);
void ggml_vec_relu_f32 (const int n, float * y, const float * x);
void ggml_vec_leaky_relu_f32 (const int n, float * y, const float * x, const float ns);
void ggml_vec_hardswish_f32 (const int n, float * y, const float * x);
void ggml_vec_hardsigmoid_f32 (const int n, float * y, const float * x);
void ggml_vec_gelu_f32(const int n, float * y, const float * x);
void ggml_vec_gelu_quick_f32(const int n, float * y, const float * x);
void ggml_vec_silu_f32(const int n, float * y, const float * x);
float ggml_silu_backward_f32(float x, float dy);
void ggml_vec_silu_backward_f32(const int n, float * dx, const float * x, const float * dy);
void ggml_vec_sum_f32(const int n, float * s, const float * x);
void ggml_vec_sum_f32_ggf(const int n, ggml_float * s, const float * x);
void ggml_vec_sum_f16_ggf(const int n, float * s, const ggml_fp16_t * x);
void ggml_vec_sum_bf16_ggf(const int n, float * s, const ggml_bf16_t * x);
void ggml_vec_max_f32(const int n, float * s, const float * x);
void ggml_vec_argmax_f32(const int n, int * s, const float * x);
ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max);
void ggml_vec_norm_inv_f32(const int n, float * s, const float * x);
void ggml_vec_sigmoid_f32 (const int n, float * y, const float * x);

#ifdef __cplusplus
}
#endif
