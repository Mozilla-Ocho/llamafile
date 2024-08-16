#include <cosmo.h>
#include <sys/auxv.h>
#include <libc/sysv/consts/hwcap.h>
#include "ggml-vector.h"

extern "C" void ggml_fp16_to_fp32_row_amd_avx512bf16(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_amd_avx512vl(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_amd_avx512(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_amd_avx2(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_amd_f16c(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_amd_fma(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_amd_avx(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_amd_ssse3(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_amd_k8(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_arm82(const ggml_fp16_t * x, float * y, int64_t n);
extern "C" void ggml_fp16_to_fp32_row_arm80(const ggml_fp16_t * x, float * y, int64_t n);

extern "C" void ggml_fp32_to_fp16_row_amd_avx512bf16(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_amd_avx512vl(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_amd_avx512(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_amd_avx2(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_amd_f16c(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_amd_fma(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_amd_avx(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_amd_ssse3(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_amd_k8(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_arm82(const float * x, ggml_fp16_t * y, int64_t n);
extern "C" void ggml_fp32_to_fp16_row_arm80(const float * x, ggml_fp16_t * y, int64_t n);

extern "C" void ggml_bf16_to_fp32_row_amd_avx512bf16(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_amd_avx512vl(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_amd_avx512(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_amd_avx2(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_amd_f16c(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_amd_fma(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_amd_avx(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_amd_ssse3(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_amd_k8(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_arm82(const ggml_bf16_t * x, float * y, int64_t n);
extern "C" void ggml_bf16_to_fp32_row_arm80(const ggml_bf16_t * x, float * y, int64_t n);

extern "C" void ggml_fp32_to_bf16_row_amd_avx512bf16(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_amd_avx512vl(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_amd_avx512(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_amd_avx2(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_amd_f16c(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_amd_fma(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_amd_avx(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_amd_ssse3(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_amd_k8(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_arm82(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_arm80(const float * x, ggml_bf16_t * y, int64_t n);

extern "C" void ggml_fp32_to_bf16_row_ref_amd_avx512bf16(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_amd_avx512vl(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_amd_avx512(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_amd_avx2(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_amd_f16c(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_amd_fma(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_amd_avx(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_amd_ssse3(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_amd_k8(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_arm82(const float * x, ggml_bf16_t * y, int64_t n);
extern "C" void ggml_fp32_to_bf16_row_ref_arm80(const float * x, ggml_bf16_t * y, int64_t n);

extern "C" void ggml_vec_dot_f32_amd_avx512bf16(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_amd_avx512vl(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_amd_avx512(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_amd_avx2(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_amd_f16c(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_amd_fma(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_amd_avx(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_amd_ssse3(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_amd_k8(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_arm82(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f32_arm80(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);

extern "C" void ggml_vec_dot_f16_amd_avx512bf16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_amd_avx512vl(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_amd_avx512(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_amd_avx2(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_amd_f16c(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_amd_fma(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_amd_avx(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_amd_ssse3(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_amd_k8(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_arm82(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_f16_arm80(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);

extern "C" void ggml_vec_dot_bf16_amd_avx512bf16(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_amd_avx512vl(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_amd_avx512(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_amd_avx2(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_amd_f16c(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_amd_fma(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_amd_avx(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_amd_ssse3(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_amd_k8(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_arm82(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);
extern "C" void ggml_vec_dot_bf16_arm80(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc);

extern "C" void ggml_vec_dot_f16_unroll_amd_avx512bf16(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_amd_avx512vl(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_amd_avx512(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_amd_avx2(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_amd_f16c(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_amd_fma(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_amd_avx(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_amd_ssse3(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_amd_k8(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_arm82(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);
extern "C" void ggml_vec_dot_f16_unroll_arm80(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y);

extern "C" void ggml_vec_mad_f32_unroll_amd_avx512bf16(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_amd_avx512vl(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_amd_avx512(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_amd_avx2(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_amd_f16c(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_amd_fma(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_amd_avx(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_amd_ssse3(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_amd_k8(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_arm82(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);
extern "C" void ggml_vec_mad_f32_unroll_arm80(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv);

extern "C" void ggml_vec_set_i8_amd_avx512bf16(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_amd_avx512vl(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_amd_avx512(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_amd_avx2(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_amd_f16c(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_amd_fma(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_amd_avx(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_amd_ssse3(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_amd_k8(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_arm82(const int n, int8_t * x, const int8_t v);
extern "C" void ggml_vec_set_i8_arm80(const int n, int8_t * x, const int8_t v);

extern "C" void ggml_vec_set_i16_amd_avx512bf16(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_amd_avx512vl(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_amd_avx512(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_amd_avx2(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_amd_f16c(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_amd_fma(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_amd_avx(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_amd_ssse3(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_amd_k8(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_arm82(const int n, int16_t * x, const int16_t v);
extern "C" void ggml_vec_set_i16_arm80(const int n, int16_t * x, const int16_t v);

extern "C" void ggml_vec_set_i32_amd_avx512bf16(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_amd_avx512vl(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_amd_avx512(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_amd_avx2(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_amd_f16c(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_amd_fma(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_amd_avx(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_amd_ssse3(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_amd_k8(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_arm82(const int n, int32_t * x, const int32_t v);
extern "C" void ggml_vec_set_i32_arm80(const int n, int32_t * x, const int32_t v);

extern "C" void ggml_vec_set_f16_amd_avx512bf16(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_amd_avx512vl(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_amd_avx512(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_amd_avx2(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_amd_f16c(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_amd_fma(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_amd_avx(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_amd_ssse3(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_amd_k8(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_arm82(const int n, ggml_fp16_t * x, const int32_t v);
extern "C" void ggml_vec_set_f16_arm80(const int n, ggml_fp16_t * x, const int32_t v);

extern "C" void ggml_vec_set_bf16_amd_avx512bf16(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_amd_avx512vl(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_amd_avx512(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_amd_avx2(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_amd_f16c(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_amd_fma(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_amd_avx(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_amd_ssse3(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_amd_k8(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_arm82(const int n, ggml_bf16_t * x, const ggml_bf16_t v);
extern "C" void ggml_vec_set_bf16_arm80(const int n, ggml_bf16_t * x, const ggml_bf16_t v);

extern "C" void ggml_vec_add_f32_amd_avx512bf16 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_amd_avx512vl (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_amd_avx512 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_amd_avx2 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_amd_f16c (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_amd_fma (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_amd_avx (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_amd_ssse3 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_amd_k8 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_arm82 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_add_f32_arm80 (const int n, float * z, const float * x, const float * y);

extern "C" void ggml_vec_add1_f32_amd_avx512bf16(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_amd_avx512vl(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_amd_avx512(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_amd_avx2(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_amd_f16c(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_amd_fma(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_amd_avx(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_amd_ssse3(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_amd_k8(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_arm82(const int n, float * z, const float * x, const float   v);
extern "C" void ggml_vec_add1_f32_arm80(const int n, float * z, const float * x, const float   v);

extern "C" void ggml_vec_acc_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_acc_f32_arm80 (const int n, float * y, const float * x);

extern "C" void ggml_vec_acc1_f32_amd_avx512bf16(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_amd_avx512vl(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_amd_avx512(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_amd_avx2(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_amd_f16c(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_amd_fma(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_amd_avx(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_amd_ssse3(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_amd_k8(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_arm82(const int n, float * y, const float   v);
extern "C" void ggml_vec_acc1_f32_arm80(const int n, float * y, const float   v);

extern "C" void ggml_vec_sub_f32_amd_avx512bf16 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_amd_avx512vl (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_amd_avx512 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_amd_avx2 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_amd_f16c (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_amd_fma (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_amd_avx (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_amd_ssse3 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_amd_k8 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_arm82 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_sub_f32_arm80 (const int n, float * z, const float * x, const float * y);

extern "C" void ggml_vec_set_f32_amd_avx512bf16 (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_amd_avx512vl (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_amd_avx512 (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_amd_avx2 (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_amd_f16c (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_amd_fma (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_amd_avx (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_amd_ssse3 (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_amd_k8 (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_arm82 (const int n, float * x, const float   v);
extern "C" void ggml_vec_set_f32_arm80 (const int n, float * x, const float   v);

extern "C" void ggml_vec_cpy_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_cpy_f32_arm80 (const int n, float * y, const float * x);

extern "C" void ggml_vec_neg_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_neg_f32_arm80 (const int n, float * y, const float * x);

extern "C" void ggml_vec_mul_f32_amd_avx512bf16 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_amd_avx512vl (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_amd_avx512 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_amd_avx2 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_amd_f16c (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_amd_fma (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_amd_avx (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_amd_ssse3 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_amd_k8 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_arm82 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_mul_f32_arm80 (const int n, float * z, const float * x, const float * y);

extern "C" void ggml_vec_div_f32_amd_avx512bf16 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_amd_avx512vl (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_amd_avx512 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_amd_avx2 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_amd_f16c (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_amd_fma (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_amd_avx (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_amd_ssse3 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_amd_k8 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_arm82 (const int n, float * z, const float * x, const float * y);
extern "C" void ggml_vec_div_f32_arm80 (const int n, float * z, const float * x, const float * y);

extern "C" void ggml_vec_scale_f32_amd_avx512bf16(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_amd_avx512vl(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_amd_avx512(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_amd_avx2(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_amd_f16c(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_amd_fma(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_amd_avx(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_amd_ssse3(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_amd_k8(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_arm82(const int n, float * y, const float   v);
extern "C" void ggml_vec_scale_f32_arm80(const int n, float * y, const float   v);

extern "C" void ggml_vec_scale_f16_amd_avx512bf16(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_amd_avx512vl(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_amd_avx512(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_amd_avx2(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_amd_f16c(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_amd_fma(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_amd_avx(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_amd_ssse3(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_amd_k8(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_arm82(const int n, ggml_fp16_t * y, const float v);
extern "C" void ggml_vec_scale_f16_arm80(const int n, ggml_fp16_t * y, const float v);

extern "C" void ggml_vec_mad_f32_amd_avx512bf16(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_amd_avx512vl(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_amd_avx512(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_amd_avx2(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_amd_f16c(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_amd_fma(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_amd_avx(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_amd_ssse3(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_amd_k8(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_arm82(const int n, float * y, const float * x, const float v);
extern "C" void ggml_vec_mad_f32_arm80(const int n, float * y, const float * x, const float v);

extern "C" void ggml_vec_mad_f16_amd_avx512bf16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_amd_avx512vl(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_amd_avx512(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_amd_avx2(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_amd_f16c(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_amd_fma(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_amd_avx(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_amd_ssse3(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_amd_k8(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_arm82(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);
extern "C" void ggml_vec_mad_f16_arm80(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v);

extern "C" void ggml_vec_norm_f32_amd_avx512bf16 (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_amd_avx512vl (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_amd_avx512 (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_amd_avx2 (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_amd_f16c (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_amd_fma (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_amd_avx (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_amd_ssse3 (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_amd_k8 (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_arm82 (const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_f32_arm80 (const int n, float * s, const float * x);

extern "C" void ggml_vec_sqr_f32_amd_avx512bf16  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_amd_avx512vl  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_amd_avx512  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_amd_avx2  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_amd_f16c  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_amd_fma  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_amd_avx  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_amd_ssse3  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_amd_k8  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_arm82  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqr_f32_arm80  (const int n, float * y, const float * x);

extern "C" void ggml_vec_sqrt_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sqrt_f32_arm80 (const int n, float * y, const float * x);

extern "C" void ggml_vec_log_f32_amd_avx512bf16  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_amd_avx512vl  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_amd_avx512  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_amd_avx2  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_amd_f16c  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_amd_fma  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_amd_avx  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_amd_ssse3  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_amd_k8  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_arm82  (const int n, float * y, const float * x);
extern "C" void ggml_vec_log_f32_arm80  (const int n, float * y, const float * x);

extern "C" void ggml_vec_abs_f32_amd_avx512bf16  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_amd_avx512vl  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_amd_avx512  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_amd_avx2  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_amd_f16c  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_amd_fma  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_amd_avx  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_amd_ssse3  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_amd_k8  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_arm82  (const int n, float * y, const float * x);
extern "C" void ggml_vec_abs_f32_arm80  (const int n, float * y, const float * x);

extern "C" void ggml_vec_sgn_f32_amd_avx512bf16  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_amd_avx512vl  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_amd_avx512  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_amd_avx2  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_amd_f16c  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_amd_fma  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_amd_avx  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_amd_ssse3  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_amd_k8  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_arm82  (const int n, float * y, const float * x);
extern "C" void ggml_vec_sgn_f32_arm80  (const int n, float * y, const float * x);

extern "C" void ggml_vec_step_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_step_f32_arm80 (const int n, float * y, const float * x);

extern "C" void ggml_vec_tanh_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_tanh_f32_arm80 (const int n, float * y, const float * x);

extern "C" void ggml_vec_elu_f32_amd_avx512bf16  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_amd_avx512vl  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_amd_avx512  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_amd_avx2  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_amd_f16c  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_amd_fma  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_amd_avx  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_amd_ssse3  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_amd_k8  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_arm82  (const int n, float * y, const float * x);
extern "C" void ggml_vec_elu_f32_arm80  (const int n, float * y, const float * x);

extern "C" void ggml_vec_relu_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_relu_f32_arm80 (const int n, float * y, const float * x);

extern "C" void ggml_vec_leaky_relu_f32_amd_avx512bf16 (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_amd_avx512vl (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_amd_avx512 (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_amd_avx2 (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_amd_f16c (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_amd_fma (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_amd_avx (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_amd_ssse3 (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_amd_k8 (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_arm82 (const int n, float * y, const float * x, const float ns);
extern "C" void ggml_vec_leaky_relu_f32_arm80 (const int n, float * y, const float * x, const float ns);

extern "C" void ggml_vec_hardswish_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardswish_f32_arm80 (const int n, float * y, const float * x);

extern "C" void ggml_vec_hardsigmoid_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_hardsigmoid_f32_arm80 (const int n, float * y, const float * x);

extern "C" void ggml_vec_gelu_f32_amd_avx512bf16(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_amd_avx512vl(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_amd_avx512(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_amd_avx2(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_amd_f16c(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_amd_fma(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_amd_avx(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_amd_ssse3(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_amd_k8(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_arm82(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_f32_arm80(const int n, float * y, const float * x);

extern "C" void ggml_vec_gelu_quick_f32_amd_avx512bf16(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_amd_avx512vl(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_amd_avx512(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_amd_avx2(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_amd_f16c(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_amd_fma(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_amd_avx(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_amd_ssse3(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_amd_k8(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_arm82(const int n, float * y, const float * x);
extern "C" void ggml_vec_gelu_quick_f32_arm80(const int n, float * y, const float * x);

extern "C" void ggml_vec_silu_f32_amd_avx512bf16(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_amd_avx512vl(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_amd_avx512(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_amd_avx2(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_amd_f16c(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_amd_fma(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_amd_avx(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_amd_ssse3(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_amd_k8(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_arm82(const int n, float * y, const float * x);
extern "C" void ggml_vec_silu_f32_arm80(const int n, float * y, const float * x);

extern "C" float ggml_silu_backward_f32_amd_avx512bf16(float x, float dy);
extern "C" float ggml_silu_backward_f32_amd_avx512vl(float x, float dy);
extern "C" float ggml_silu_backward_f32_amd_avx512(float x, float dy);
extern "C" float ggml_silu_backward_f32_amd_avx2(float x, float dy);
extern "C" float ggml_silu_backward_f32_amd_f16c(float x, float dy);
extern "C" float ggml_silu_backward_f32_amd_fma(float x, float dy);
extern "C" float ggml_silu_backward_f32_amd_avx(float x, float dy);
extern "C" float ggml_silu_backward_f32_amd_ssse3(float x, float dy);
extern "C" float ggml_silu_backward_f32_amd_k8(float x, float dy);
extern "C" float ggml_silu_backward_f32_arm82(float x, float dy);
extern "C" float ggml_silu_backward_f32_arm80(float x, float dy);

extern "C" void ggml_vec_silu_backward_f32_amd_avx512bf16(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_amd_avx512vl(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_amd_avx512(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_amd_avx2(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_amd_f16c(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_amd_fma(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_amd_avx(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_amd_ssse3(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_amd_k8(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_arm82(const int n, float * dx, const float * x, const float * dy);
extern "C" void ggml_vec_silu_backward_f32_arm80(const int n, float * dx, const float * x, const float * dy);

extern "C" void ggml_vec_sum_f32_amd_avx512bf16(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_amd_avx512vl(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_amd_avx512(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_amd_avx2(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_amd_f16c(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_amd_fma(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_amd_avx(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_amd_ssse3(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_amd_k8(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_arm82(const int n, float * s, const float * x);
extern "C" void ggml_vec_sum_f32_arm80(const int n, float * s, const float * x);

extern "C" void ggml_vec_sum_f32_ggf_amd_avx512bf16(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_amd_avx512vl(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_amd_avx512(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_amd_avx2(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_amd_f16c(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_amd_fma(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_amd_avx(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_amd_ssse3(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_amd_k8(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_arm82(const int n, ggml_float * s, const float * x);
extern "C" void ggml_vec_sum_f32_ggf_arm80(const int n, ggml_float * s, const float * x);

extern "C" void ggml_vec_sum_f16_ggf_amd_avx512bf16(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_amd_avx512vl(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_amd_avx512(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_amd_avx2(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_amd_f16c(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_amd_fma(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_amd_avx(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_amd_ssse3(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_amd_k8(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_arm82(const int n, float * s, const ggml_fp16_t * x);
extern "C" void ggml_vec_sum_f16_ggf_arm80(const int n, float * s, const ggml_fp16_t * x);

extern "C" void ggml_vec_sum_bf16_ggf_amd_avx512bf16(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_amd_avx512vl(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_amd_avx512(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_amd_avx2(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_amd_f16c(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_amd_fma(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_amd_avx(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_amd_ssse3(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_amd_k8(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_arm82(const int n, float * s, const ggml_bf16_t * x);
extern "C" void ggml_vec_sum_bf16_ggf_arm80(const int n, float * s, const ggml_bf16_t * x);

extern "C" void ggml_vec_max_f32_amd_avx512bf16(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_amd_avx512vl(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_amd_avx512(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_amd_avx2(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_amd_f16c(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_amd_fma(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_amd_avx(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_amd_ssse3(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_amd_k8(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_arm82(const int n, float * s, const float * x);
extern "C" void ggml_vec_max_f32_arm80(const int n, float * s, const float * x);

extern "C" void ggml_vec_argmax_f32_amd_avx512bf16(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_amd_avx512vl(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_amd_avx512(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_amd_avx2(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_amd_f16c(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_amd_fma(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_amd_avx(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_amd_ssse3(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_amd_k8(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_arm82(const int n, int * s, const float * x);
extern "C" void ggml_vec_argmax_f32_arm80(const int n, int * s, const float * x);

extern "C" ggml_float ggml_vec_soft_max_f32_amd_avx512bf16(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_amd_avx512vl(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_amd_avx512(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_amd_avx2(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_amd_f16c(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_amd_fma(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_amd_avx(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_amd_ssse3(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_amd_k8(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_arm82(const int n, float * y, const float * x, float max);
extern "C" ggml_float ggml_vec_soft_max_f32_arm80(const int n, float * y, const float * x, float max);

extern "C" void ggml_vec_norm_inv_f32_amd_avx512bf16(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_amd_avx512vl(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_amd_avx512(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_amd_avx2(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_amd_f16c(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_amd_fma(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_amd_avx(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_amd_ssse3(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_amd_k8(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_arm82(const int n, float * s, const float * x);
extern "C" void ggml_vec_norm_inv_f32_arm80(const int n, float * s, const float * x);

extern "C" void ggml_vec_sigmoid_f32_amd_avx512bf16 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_amd_avx512vl (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_amd_avx512 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_amd_avx2 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_amd_f16c (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_amd_fma (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_amd_avx (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_amd_ssse3 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_amd_k8 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_arm82 (const int n, float * y, const float * x);
extern "C" void ggml_vec_sigmoid_f32_arm80 (const int n, float * y, const float * x);

static const struct VectorFuncs {
    typeof(ggml_fp16_to_fp32_row) *ptr_ggml_fp16_to_fp32_row;
    typeof(ggml_fp32_to_fp16_row) *ptr_ggml_fp32_to_fp16_row;
    typeof(ggml_bf16_to_fp32_row) *ptr_ggml_bf16_to_fp32_row;
    typeof(ggml_fp32_to_bf16_row) *ptr_ggml_fp32_to_bf16_row;
    typeof(ggml_fp32_to_bf16_row_ref) *ptr_ggml_fp32_to_bf16_row_ref;
    typeof(ggml_vec_dot_f32) *ptr_ggml_vec_dot_f32;
    typeof(ggml_vec_dot_f16) *ptr_ggml_vec_dot_f16;
    typeof(ggml_vec_dot_bf16) *ptr_ggml_vec_dot_bf16;
    typeof(ggml_vec_dot_f16_unroll) *ptr_ggml_vec_dot_f16_unroll;
    typeof(ggml_vec_mad_f32_unroll) *ptr_ggml_vec_mad_f32_unroll;
    typeof(ggml_vec_set_i8) *ptr_ggml_vec_set_i8;
    typeof(ggml_vec_set_i16) *ptr_ggml_vec_set_i16;
    typeof(ggml_vec_set_i32) *ptr_ggml_vec_set_i32;
    typeof(ggml_vec_set_f16) *ptr_ggml_vec_set_f16;
    typeof(ggml_vec_set_bf16) *ptr_ggml_vec_set_bf16;
    typeof(ggml_vec_add_f32) *ptr_ggml_vec_add_f32;
    typeof(ggml_vec_add1_f32) *ptr_ggml_vec_add1_f32;
    typeof(ggml_vec_acc_f32) *ptr_ggml_vec_acc_f32;
    typeof(ggml_vec_acc1_f32) *ptr_ggml_vec_acc1_f32;
    typeof(ggml_vec_sub_f32) *ptr_ggml_vec_sub_f32;
    typeof(ggml_vec_set_f32) *ptr_ggml_vec_set_f32;
    typeof(ggml_vec_cpy_f32) *ptr_ggml_vec_cpy_f32;
    typeof(ggml_vec_neg_f32) *ptr_ggml_vec_neg_f32;
    typeof(ggml_vec_mul_f32) *ptr_ggml_vec_mul_f32;
    typeof(ggml_vec_div_f32) *ptr_ggml_vec_div_f32;
    typeof(ggml_vec_scale_f32) *ptr_ggml_vec_scale_f32;
    typeof(ggml_vec_scale_f16) *ptr_ggml_vec_scale_f16;
    typeof(ggml_vec_mad_f32) *ptr_ggml_vec_mad_f32;
    typeof(ggml_vec_mad_f16) *ptr_ggml_vec_mad_f16;
    typeof(ggml_vec_norm_f32) *ptr_ggml_vec_norm_f32;
    typeof(ggml_vec_sqr_f32) *ptr_ggml_vec_sqr_f32;
    typeof(ggml_vec_sqrt_f32) *ptr_ggml_vec_sqrt_f32;
    typeof(ggml_vec_log_f32) *ptr_ggml_vec_log_f32;
    typeof(ggml_vec_abs_f32) *ptr_ggml_vec_abs_f32;
    typeof(ggml_vec_sgn_f32) *ptr_ggml_vec_sgn_f32;
    typeof(ggml_vec_step_f32) *ptr_ggml_vec_step_f32;
    typeof(ggml_vec_tanh_f32) *ptr_ggml_vec_tanh_f32;
    typeof(ggml_vec_elu_f32) *ptr_ggml_vec_elu_f32;
    typeof(ggml_vec_relu_f32) *ptr_ggml_vec_relu_f32;
    typeof(ggml_vec_leaky_relu_f32) *ptr_ggml_vec_leaky_relu_f32;
    typeof(ggml_vec_hardswish_f32) *ptr_ggml_vec_hardswish_f32;
    typeof(ggml_vec_hardsigmoid_f32) *ptr_ggml_vec_hardsigmoid_f32;
    typeof(ggml_vec_gelu_f32) *ptr_ggml_vec_gelu_f32;
    typeof(ggml_vec_gelu_quick_f32) *ptr_ggml_vec_gelu_quick_f32;
    typeof(ggml_vec_silu_f32) *ptr_ggml_vec_silu_f32;
    typeof(ggml_silu_backward_f32) *ptr_ggml_silu_backward_f32;
    typeof(ggml_vec_silu_backward_f32) *ptr_ggml_vec_silu_backward_f32;
    typeof(ggml_vec_sum_f32) *ptr_ggml_vec_sum_f32;
    typeof(ggml_vec_sum_f32_ggf) *ptr_ggml_vec_sum_f32_ggf;
    typeof(ggml_vec_sum_f16_ggf) *ptr_ggml_vec_sum_f16_ggf;
    typeof(ggml_vec_sum_bf16_ggf) *ptr_ggml_vec_sum_bf16_ggf;
    typeof(ggml_vec_max_f32) *ptr_ggml_vec_max_f32;
    typeof(ggml_vec_argmax_f32) *ptr_ggml_vec_argmax_f32;
    typeof(ggml_vec_soft_max_f32) *ptr_ggml_vec_soft_max_f32;
    typeof(ggml_vec_norm_inv_f32) *ptr_ggml_vec_norm_inv_f32;
    typeof(ggml_vec_sigmoid_f32) *ptr_ggml_vec_sigmoid_f32;

    VectorFuncs() {
#ifdef __x86_64__
        if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) && X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL) && X86_HAVE(AVX512_BF16)) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_amd_avx512bf16;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_amd_avx512bf16;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_amd_avx512bf16;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_amd_avx512bf16;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_amd_avx512bf16;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_amd_avx512bf16;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_amd_avx512bf16;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_amd_avx512bf16;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_amd_avx512bf16;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_amd_avx512bf16;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_amd_avx512bf16;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_amd_avx512bf16;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_amd_avx512bf16;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_amd_avx512bf16;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_amd_avx512bf16;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_amd_avx512bf16;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_amd_avx512bf16;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_amd_avx512bf16;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_amd_avx512bf16;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_amd_avx512bf16;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_amd_avx512bf16;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_amd_avx512bf16;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_amd_avx512bf16;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_amd_avx512bf16;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_amd_avx512bf16;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_amd_avx512bf16;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_amd_avx512bf16;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_amd_avx512bf16;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_amd_avx512bf16;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_amd_avx512bf16;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_amd_avx512bf16;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_amd_avx512bf16;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_amd_avx512bf16;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_amd_avx512bf16;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_amd_avx512bf16;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_amd_avx512bf16;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_amd_avx512bf16;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_amd_avx512bf16;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_amd_avx512bf16;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_amd_avx512bf16;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_amd_avx512bf16;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_amd_avx512bf16;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_amd_avx512bf16;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_amd_avx512bf16;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_amd_avx512bf16;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_amd_avx512bf16;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_amd_avx512bf16;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_amd_avx512bf16;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_amd_avx512bf16;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_amd_avx512bf16;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_amd_avx512bf16;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_amd_avx512bf16;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_amd_avx512bf16;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_amd_avx512bf16;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_amd_avx512bf16;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_amd_avx512bf16;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) && X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL)) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_amd_avx512vl;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_amd_avx512vl;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_amd_avx512vl;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_amd_avx512vl;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_amd_avx512vl;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_amd_avx512vl;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_amd_avx512vl;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_amd_avx512vl;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_amd_avx512vl;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_amd_avx512vl;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_amd_avx512vl;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_amd_avx512vl;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_amd_avx512vl;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_amd_avx512vl;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_amd_avx512vl;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_amd_avx512vl;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_amd_avx512vl;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_amd_avx512vl;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_amd_avx512vl;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_amd_avx512vl;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_amd_avx512vl;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_amd_avx512vl;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_amd_avx512vl;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_amd_avx512vl;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_amd_avx512vl;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_amd_avx512vl;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_amd_avx512vl;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_amd_avx512vl;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_amd_avx512vl;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_amd_avx512vl;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_amd_avx512vl;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_amd_avx512vl;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_amd_avx512vl;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_amd_avx512vl;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_amd_avx512vl;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_amd_avx512vl;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_amd_avx512vl;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_amd_avx512vl;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_amd_avx512vl;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_amd_avx512vl;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_amd_avx512vl;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_amd_avx512vl;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_amd_avx512vl;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_amd_avx512vl;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_amd_avx512vl;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_amd_avx512vl;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_amd_avx512vl;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_amd_avx512vl;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_amd_avx512vl;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_amd_avx512vl;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_amd_avx512vl;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_amd_avx512vl;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_amd_avx512vl;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_amd_avx512vl;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_amd_avx512vl;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_amd_avx512vl;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F)) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_amd_avx512;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_amd_avx512;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_amd_avx512;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_amd_avx512;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_amd_avx512;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_amd_avx512;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_amd_avx512;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_amd_avx512;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_amd_avx512;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_amd_avx512;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_amd_avx512;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_amd_avx512;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_amd_avx512;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_amd_avx512;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_amd_avx512;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_amd_avx512;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_amd_avx512;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_amd_avx512;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_amd_avx512;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_amd_avx512;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_amd_avx512;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_amd_avx512;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_amd_avx512;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_amd_avx512;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_amd_avx512;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_amd_avx512;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_amd_avx512;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_amd_avx512;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_amd_avx512;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_amd_avx512;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_amd_avx512;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_amd_avx512;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_amd_avx512;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_amd_avx512;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_amd_avx512;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_amd_avx512;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_amd_avx512;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_amd_avx512;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_amd_avx512;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_amd_avx512;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_amd_avx512;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_amd_avx512;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_amd_avx512;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_amd_avx512;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_amd_avx512;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_amd_avx512;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_amd_avx512;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_amd_avx512;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_amd_avx512;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_amd_avx512;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_amd_avx512;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_amd_avx512;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_amd_avx512;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_amd_avx512;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_amd_avx512;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_amd_avx512;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2)) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_amd_avx2;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_amd_avx2;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_amd_avx2;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_amd_avx2;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_amd_avx2;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_amd_avx2;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_amd_avx2;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_amd_avx2;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_amd_avx2;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_amd_avx2;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_amd_avx2;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_amd_avx2;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_amd_avx2;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_amd_avx2;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_amd_avx2;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_amd_avx2;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_amd_avx2;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_amd_avx2;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_amd_avx2;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_amd_avx2;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_amd_avx2;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_amd_avx2;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_amd_avx2;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_amd_avx2;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_amd_avx2;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_amd_avx2;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_amd_avx2;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_amd_avx2;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_amd_avx2;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_amd_avx2;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_amd_avx2;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_amd_avx2;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_amd_avx2;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_amd_avx2;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_amd_avx2;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_amd_avx2;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_amd_avx2;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_amd_avx2;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_amd_avx2;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_amd_avx2;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_amd_avx2;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_amd_avx2;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_amd_avx2;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_amd_avx2;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_amd_avx2;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_amd_avx2;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_amd_avx2;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_amd_avx2;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_amd_avx2;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_amd_avx2;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_amd_avx2;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_amd_avx2;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_amd_avx2;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_amd_avx2;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_amd_avx2;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_amd_avx2;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(AVX) && X86_HAVE(F16C)) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_amd_f16c;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_amd_f16c;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_amd_f16c;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_amd_f16c;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_amd_f16c;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_amd_f16c;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_amd_f16c;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_amd_f16c;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_amd_f16c;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_amd_f16c;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_amd_f16c;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_amd_f16c;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_amd_f16c;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_amd_f16c;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_amd_f16c;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_amd_f16c;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_amd_f16c;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_amd_f16c;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_amd_f16c;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_amd_f16c;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_amd_f16c;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_amd_f16c;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_amd_f16c;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_amd_f16c;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_amd_f16c;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_amd_f16c;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_amd_f16c;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_amd_f16c;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_amd_f16c;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_amd_f16c;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_amd_f16c;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_amd_f16c;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_amd_f16c;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_amd_f16c;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_amd_f16c;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_amd_f16c;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_amd_f16c;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_amd_f16c;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_amd_f16c;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_amd_f16c;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_amd_f16c;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_amd_f16c;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_amd_f16c;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_amd_f16c;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_amd_f16c;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_amd_f16c;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_amd_f16c;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_amd_f16c;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_amd_f16c;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_amd_f16c;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_amd_f16c;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_amd_f16c;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_amd_f16c;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_amd_f16c;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_amd_f16c;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_amd_f16c;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(AVX) && X86_HAVE(FMA)) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_amd_fma;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_amd_fma;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_amd_fma;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_amd_fma;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_amd_fma;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_amd_fma;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_amd_fma;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_amd_fma;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_amd_fma;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_amd_fma;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_amd_fma;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_amd_fma;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_amd_fma;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_amd_fma;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_amd_fma;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_amd_fma;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_amd_fma;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_amd_fma;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_amd_fma;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_amd_fma;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_amd_fma;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_amd_fma;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_amd_fma;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_amd_fma;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_amd_fma;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_amd_fma;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_amd_fma;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_amd_fma;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_amd_fma;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_amd_fma;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_amd_fma;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_amd_fma;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_amd_fma;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_amd_fma;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_amd_fma;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_amd_fma;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_amd_fma;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_amd_fma;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_amd_fma;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_amd_fma;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_amd_fma;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_amd_fma;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_amd_fma;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_amd_fma;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_amd_fma;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_amd_fma;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_amd_fma;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_amd_fma;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_amd_fma;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_amd_fma;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_amd_fma;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_amd_fma;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_amd_fma;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_amd_fma;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_amd_fma;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_amd_fma;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(AVX)) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_amd_avx;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_amd_avx;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_amd_avx;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_amd_avx;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_amd_avx;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_amd_avx;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_amd_avx;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_amd_avx;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_amd_avx;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_amd_avx;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_amd_avx;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_amd_avx;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_amd_avx;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_amd_avx;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_amd_avx;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_amd_avx;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_amd_avx;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_amd_avx;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_amd_avx;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_amd_avx;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_amd_avx;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_amd_avx;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_amd_avx;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_amd_avx;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_amd_avx;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_amd_avx;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_amd_avx;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_amd_avx;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_amd_avx;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_amd_avx;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_amd_avx;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_amd_avx;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_amd_avx;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_amd_avx;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_amd_avx;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_amd_avx;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_amd_avx;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_amd_avx;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_amd_avx;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_amd_avx;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_amd_avx;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_amd_avx;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_amd_avx;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_amd_avx;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_amd_avx;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_amd_avx;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_amd_avx;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_amd_avx;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_amd_avx;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_amd_avx;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_amd_avx;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_amd_avx;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_amd_avx;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_amd_avx;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_amd_avx;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_amd_avx;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(SSSE3)) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_amd_ssse3;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_amd_ssse3;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_amd_ssse3;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_amd_ssse3;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_amd_ssse3;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_amd_ssse3;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_amd_ssse3;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_amd_ssse3;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_amd_ssse3;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_amd_ssse3;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_amd_ssse3;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_amd_ssse3;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_amd_ssse3;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_amd_ssse3;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_amd_ssse3;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_amd_ssse3;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_amd_ssse3;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_amd_ssse3;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_amd_ssse3;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_amd_ssse3;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_amd_ssse3;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_amd_ssse3;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_amd_ssse3;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_amd_ssse3;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_amd_ssse3;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_amd_ssse3;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_amd_ssse3;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_amd_ssse3;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_amd_ssse3;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_amd_ssse3;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_amd_ssse3;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_amd_ssse3;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_amd_ssse3;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_amd_ssse3;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_amd_ssse3;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_amd_ssse3;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_amd_ssse3;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_amd_ssse3;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_amd_ssse3;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_amd_ssse3;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_amd_ssse3;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_amd_ssse3;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_amd_ssse3;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_amd_ssse3;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_amd_ssse3;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_amd_ssse3;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_amd_ssse3;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_amd_ssse3;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_amd_ssse3;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_amd_ssse3;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_amd_ssse3;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_amd_ssse3;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_amd_ssse3;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_amd_ssse3;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_amd_ssse3;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_amd_ssse3;
            return;
        }
#endif
#ifdef __x86_64__
        if (1) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_amd_k8;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_amd_k8;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_amd_k8;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_amd_k8;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_amd_k8;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_amd_k8;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_amd_k8;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_amd_k8;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_amd_k8;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_amd_k8;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_amd_k8;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_amd_k8;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_amd_k8;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_amd_k8;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_amd_k8;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_amd_k8;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_amd_k8;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_amd_k8;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_amd_k8;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_amd_k8;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_amd_k8;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_amd_k8;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_amd_k8;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_amd_k8;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_amd_k8;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_amd_k8;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_amd_k8;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_amd_k8;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_amd_k8;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_amd_k8;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_amd_k8;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_amd_k8;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_amd_k8;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_amd_k8;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_amd_k8;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_amd_k8;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_amd_k8;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_amd_k8;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_amd_k8;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_amd_k8;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_amd_k8;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_amd_k8;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_amd_k8;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_amd_k8;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_amd_k8;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_amd_k8;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_amd_k8;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_amd_k8;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_amd_k8;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_amd_k8;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_amd_k8;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_amd_k8;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_amd_k8;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_amd_k8;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_amd_k8;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_amd_k8;
            return;
        }
#endif
#ifdef __aarch64__
        if ((getauxval(AT_HWCAP) & HWCAP_FPHP) && (getauxval(AT_HWCAP) & HWCAP_ASIMDHP)) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_arm82;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_arm82;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_arm82;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_arm82;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_arm82;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_arm82;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_arm82;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_arm82;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_arm82;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_arm82;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_arm82;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_arm82;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_arm82;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_arm82;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_arm82;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_arm82;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_arm82;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_arm82;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_arm82;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_arm82;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_arm82;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_arm82;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_arm82;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_arm82;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_arm82;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_arm82;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_arm82;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_arm82;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_arm82;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_arm82;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_arm82;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_arm82;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_arm82;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_arm82;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_arm82;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_arm82;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_arm82;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_arm82;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_arm82;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_arm82;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_arm82;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_arm82;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_arm82;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_arm82;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_arm82;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_arm82;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_arm82;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_arm82;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_arm82;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_arm82;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_arm82;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_arm82;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_arm82;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_arm82;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_arm82;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_arm82;
            return;
        }
#endif
#ifdef __aarch64__
        if (1) {
            ptr_ggml_fp16_to_fp32_row = ggml_fp16_to_fp32_row_arm80;
            ptr_ggml_fp32_to_fp16_row = ggml_fp32_to_fp16_row_arm80;
            ptr_ggml_bf16_to_fp32_row = ggml_bf16_to_fp32_row_arm80;
            ptr_ggml_fp32_to_bf16_row = ggml_fp32_to_bf16_row_arm80;
            ptr_ggml_fp32_to_bf16_row_ref = ggml_fp32_to_bf16_row_ref_arm80;
            ptr_ggml_vec_dot_f32 = ggml_vec_dot_f32_arm80;
            ptr_ggml_vec_dot_f16 = ggml_vec_dot_f16_arm80;
            ptr_ggml_vec_dot_bf16 = ggml_vec_dot_bf16_arm80;
            ptr_ggml_vec_dot_f16_unroll = ggml_vec_dot_f16_unroll_arm80;
            ptr_ggml_vec_mad_f32_unroll = ggml_vec_mad_f32_unroll_arm80;
            ptr_ggml_vec_set_i8 = ggml_vec_set_i8_arm80;
            ptr_ggml_vec_set_i16 = ggml_vec_set_i16_arm80;
            ptr_ggml_vec_set_i32 = ggml_vec_set_i32_arm80;
            ptr_ggml_vec_set_f16 = ggml_vec_set_f16_arm80;
            ptr_ggml_vec_set_bf16 = ggml_vec_set_bf16_arm80;
            ptr_ggml_vec_add_f32 = ggml_vec_add_f32_arm80;
            ptr_ggml_vec_add1_f32 = ggml_vec_add1_f32_arm80;
            ptr_ggml_vec_acc_f32 = ggml_vec_acc_f32_arm80;
            ptr_ggml_vec_acc1_f32 = ggml_vec_acc1_f32_arm80;
            ptr_ggml_vec_sub_f32 = ggml_vec_sub_f32_arm80;
            ptr_ggml_vec_set_f32 = ggml_vec_set_f32_arm80;
            ptr_ggml_vec_cpy_f32 = ggml_vec_cpy_f32_arm80;
            ptr_ggml_vec_neg_f32 = ggml_vec_neg_f32_arm80;
            ptr_ggml_vec_mul_f32 = ggml_vec_mul_f32_arm80;
            ptr_ggml_vec_div_f32 = ggml_vec_div_f32_arm80;
            ptr_ggml_vec_scale_f32 = ggml_vec_scale_f32_arm80;
            ptr_ggml_vec_scale_f16 = ggml_vec_scale_f16_arm80;
            ptr_ggml_vec_mad_f32 = ggml_vec_mad_f32_arm80;
            ptr_ggml_vec_mad_f16 = ggml_vec_mad_f16_arm80;
            ptr_ggml_vec_norm_f32 = ggml_vec_norm_f32_arm80;
            ptr_ggml_vec_sqr_f32 = ggml_vec_sqr_f32_arm80;
            ptr_ggml_vec_sqrt_f32 = ggml_vec_sqrt_f32_arm80;
            ptr_ggml_vec_log_f32 = ggml_vec_log_f32_arm80;
            ptr_ggml_vec_abs_f32 = ggml_vec_abs_f32_arm80;
            ptr_ggml_vec_sgn_f32 = ggml_vec_sgn_f32_arm80;
            ptr_ggml_vec_step_f32 = ggml_vec_step_f32_arm80;
            ptr_ggml_vec_tanh_f32 = ggml_vec_tanh_f32_arm80;
            ptr_ggml_vec_elu_f32 = ggml_vec_elu_f32_arm80;
            ptr_ggml_vec_relu_f32 = ggml_vec_relu_f32_arm80;
            ptr_ggml_vec_leaky_relu_f32 = ggml_vec_leaky_relu_f32_arm80;
            ptr_ggml_vec_hardswish_f32 = ggml_vec_hardswish_f32_arm80;
            ptr_ggml_vec_hardsigmoid_f32 = ggml_vec_hardsigmoid_f32_arm80;
            ptr_ggml_vec_gelu_f32 = ggml_vec_gelu_f32_arm80;
            ptr_ggml_vec_gelu_quick_f32 = ggml_vec_gelu_quick_f32_arm80;
            ptr_ggml_vec_silu_f32 = ggml_vec_silu_f32_arm80;
            ptr_ggml_silu_backward_f32 = ggml_silu_backward_f32_arm80;
            ptr_ggml_vec_silu_backward_f32 = ggml_vec_silu_backward_f32_arm80;
            ptr_ggml_vec_sum_f32 = ggml_vec_sum_f32_arm80;
            ptr_ggml_vec_sum_f32_ggf = ggml_vec_sum_f32_ggf_arm80;
            ptr_ggml_vec_sum_f16_ggf = ggml_vec_sum_f16_ggf_arm80;
            ptr_ggml_vec_sum_bf16_ggf = ggml_vec_sum_bf16_ggf_arm80;
            ptr_ggml_vec_max_f32 = ggml_vec_max_f32_arm80;
            ptr_ggml_vec_argmax_f32 = ggml_vec_argmax_f32_arm80;
            ptr_ggml_vec_soft_max_f32 = ggml_vec_soft_max_f32_arm80;
            ptr_ggml_vec_norm_inv_f32 = ggml_vec_norm_inv_f32_arm80;
            ptr_ggml_vec_sigmoid_f32 = ggml_vec_sigmoid_f32_arm80;
            return;
        }
#endif
    }
} funcs;

void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n) {
  return funcs.ptr_ggml_fp16_to_fp32_row(x, y, n);
}

void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n) {
  return funcs.ptr_ggml_fp32_to_fp16_row(x, y, n);
}

void ggml_bf16_to_fp32_row(const ggml_bf16_t * x, float * y, int64_t n) {
  return funcs.ptr_ggml_bf16_to_fp32_row(x, y, n);
}

void ggml_fp32_to_bf16_row(const float * x, ggml_bf16_t * y, int64_t n) {
  return funcs.ptr_ggml_fp32_to_bf16_row(x, y, n);
}

void ggml_fp32_to_bf16_row_ref(const float * x, ggml_bf16_t * y, int64_t n) {
  return funcs.ptr_ggml_fp32_to_bf16_row_ref(x, y, n);
}

void ggml_vec_dot_f32(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_f32(n, s, bs, x, bx, y, by, nrc);
}

void ggml_vec_dot_f16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_f16(n, s, bs, x, bx, y, by, nrc);
}

void ggml_vec_dot_bf16(int n, float * s, size_t bs, ggml_bf16_t * x, size_t bx, ggml_bf16_t * y, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_bf16(n, s, bs, x, bx, y, by, nrc);
}

void ggml_vec_dot_f16_unroll(const int n, const int xs, float * s, void * xv, ggml_fp16_t * y) {
  return funcs.ptr_ggml_vec_dot_f16_unroll(n, xs, s, xv, y);
}

void ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float * y, const float * xv, const float * vv) {
  return funcs.ptr_ggml_vec_mad_f32_unroll(n, xs, vs, y, xv, vv);
}

void ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) {
  return funcs.ptr_ggml_vec_set_i8(n, x, v);
}

void ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) {
  return funcs.ptr_ggml_vec_set_i16(n, x, v);
}

void ggml_vec_set_i32(const int n, int32_t * x, const int32_t v) {
  return funcs.ptr_ggml_vec_set_i32(n, x, v);
}

void ggml_vec_set_f16(const int n, ggml_fp16_t * x, const int32_t v) {
  return funcs.ptr_ggml_vec_set_f16(n, x, v);
}

void ggml_vec_set_bf16(const int n, ggml_bf16_t * x, const ggml_bf16_t v) {
  return funcs.ptr_ggml_vec_set_bf16(n, x, v);
}

void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) {
  return funcs.ptr_ggml_vec_add_f32(n, z, x, y);
}

void ggml_vec_add1_f32(const int n, float * z, const float * x, const float   v) {
  return funcs.ptr_ggml_vec_add1_f32(n, z, x, v);
}

void ggml_vec_acc_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_acc_f32(n, y, x);
}

void ggml_vec_acc1_f32(const int n, float * y, const float   v) {
  return funcs.ptr_ggml_vec_acc1_f32(n, y, v);
}

void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) {
  return funcs.ptr_ggml_vec_sub_f32(n, z, x, y);
}

void ggml_vec_set_f32 (const int n, float * x, const float   v) {
  return funcs.ptr_ggml_vec_set_f32(n, x, v);
}

void ggml_vec_cpy_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_cpy_f32(n, y, x);
}

void ggml_vec_neg_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_neg_f32(n, y, x);
}

void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) {
  return funcs.ptr_ggml_vec_mul_f32(n, z, x, y);
}

void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) {
  return funcs.ptr_ggml_vec_div_f32(n, z, x, y);
}

void ggml_vec_scale_f32(const int n, float * y, const float   v) {
  return funcs.ptr_ggml_vec_scale_f32(n, y, v);
}

void ggml_vec_scale_f16(const int n, ggml_fp16_t * y, const float v) {
  return funcs.ptr_ggml_vec_scale_f16(n, y, v);
}

void ggml_vec_mad_f32(const int n, float * y, const float * x, const float v) {
  return funcs.ptr_ggml_vec_mad_f32(n, y, x, v);
}

void ggml_vec_mad_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float v) {
  return funcs.ptr_ggml_vec_mad_f16(n, y, x, v);
}

void ggml_vec_norm_f32 (const int n, float * s, const float * x) {
  return funcs.ptr_ggml_vec_norm_f32(n, s, x);
}

void ggml_vec_sqr_f32  (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_sqr_f32(n, y, x);
}

void ggml_vec_sqrt_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_sqrt_f32(n, y, x);
}

void ggml_vec_log_f32  (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_log_f32(n, y, x);
}

void ggml_vec_abs_f32  (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_abs_f32(n, y, x);
}

void ggml_vec_sgn_f32  (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_sgn_f32(n, y, x);
}

void ggml_vec_step_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_step_f32(n, y, x);
}

void ggml_vec_tanh_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_tanh_f32(n, y, x);
}

void ggml_vec_elu_f32  (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_elu_f32(n, y, x);
}

void ggml_vec_relu_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_relu_f32(n, y, x);
}

void ggml_vec_leaky_relu_f32 (const int n, float * y, const float * x, const float ns) {
  return funcs.ptr_ggml_vec_leaky_relu_f32(n, y, x, ns);
}

void ggml_vec_hardswish_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_hardswish_f32(n, y, x);
}

void ggml_vec_hardsigmoid_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_hardsigmoid_f32(n, y, x);
}

void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_gelu_f32(n, y, x);
}

void ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_gelu_quick_f32(n, y, x);
}

void ggml_vec_silu_f32(const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_silu_f32(n, y, x);
}

float ggml_silu_backward_f32(float x, float dy) {
  return funcs.ptr_ggml_silu_backward_f32(x, dy);
}

void ggml_vec_silu_backward_f32(const int n, float * dx, const float * x, const float * dy) {
  return funcs.ptr_ggml_vec_silu_backward_f32(n, dx, x, dy);
}

void ggml_vec_sum_f32(const int n, float * s, const float * x) {
  return funcs.ptr_ggml_vec_sum_f32(n, s, x);
}

void ggml_vec_sum_f32_ggf(const int n, ggml_float * s, const float * x) {
  return funcs.ptr_ggml_vec_sum_f32_ggf(n, s, x);
}

void ggml_vec_sum_f16_ggf(const int n, float * s, const ggml_fp16_t * x) {
  return funcs.ptr_ggml_vec_sum_f16_ggf(n, s, x);
}

void ggml_vec_sum_bf16_ggf(const int n, float * s, const ggml_bf16_t * x) {
  return funcs.ptr_ggml_vec_sum_bf16_ggf(n, s, x);
}

void ggml_vec_max_f32(const int n, float * s, const float * x) {
  return funcs.ptr_ggml_vec_max_f32(n, s, x);
}

void ggml_vec_argmax_f32(const int n, int * s, const float * x) {
  return funcs.ptr_ggml_vec_argmax_f32(n, s, x);
}

ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
  return funcs.ptr_ggml_vec_soft_max_f32(n, y, x, max);
}

void ggml_vec_norm_inv_f32(const int n, float * s, const float * x) {
  return funcs.ptr_ggml_vec_norm_inv_f32(n, s, x);
}

void ggml_vec_sigmoid_f32 (const int n, float * y, const float * x) {
  return funcs.ptr_ggml_vec_sigmoid_f32(n, y, x);
}

