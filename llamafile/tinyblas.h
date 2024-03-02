// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#ifdef GGML_USE_HIPBLAS
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

enum tinyblasOperation_t {
    TINYBLAS_OP_N,
    TINYBLAS_OP_T,
};

enum tinyblasStatus_t {
    TINYBLAS_STATUS_SUCCESS,
    TINYBLAS_STATUS_NOT_SUPPORTED,
};

enum tinyblasComputeType_t {
    TINYBLAS_COMPUTE_16F,
    TINYBLAS_COMPUTE_32F,
};

enum tinyblasGemmAlgo_t {
    TINYBLAS_GEMM_DEFAULT_TENSOR_OP,
};

typedef cudaStream_t tinyblasHandle_t;

const char *tinyblasGetStatusString(tinyblasStatus_t);

tinyblasStatus_t tinyblasSgemm(tinyblasHandle_t, tinyblasOperation_t, tinyblasOperation_t, int, int,
                               int, const float *, const float *, int, const float *, int,
                               const float *, float *, int);

tinyblasStatus_t tinyblasGemmEx(tinyblasHandle_t, tinyblasOperation_t, tinyblasOperation_t, int,
                                int, int, const void *, const void *, cudaDataType_t, int,
                                const void *, cudaDataType_t, int, const void *, void *,
                                cudaDataType_t, int, tinyblasComputeType_t, tinyblasGemmAlgo_t);

tinyblasStatus_t tinyblasGemmBatchedEx(tinyblasHandle_t, tinyblasOperation_t, tinyblasOperation_t,
                                       int, int, int, const void *, const void *const[],
                                       cudaDataType_t, int, const void *const[], cudaDataType_t,
                                       int, const void *, void *const[], cudaDataType_t, int, int,
                                       tinyblasComputeType_t, tinyblasGemmAlgo_t);

tinyblasStatus_t tinyblasGemmStridedBatchedEx(tinyblasHandle_t, tinyblasOperation_t,
                                              tinyblasOperation_t, int, int, int, const void *,
                                              const void *, cudaDataType_t, int, long long,
                                              const void *, cudaDataType_t, int, long long,
                                              const void *, void *, cudaDataType_t, int, long long,
                                              int, tinyblasComputeType_t, tinyblasGemmAlgo_t);
