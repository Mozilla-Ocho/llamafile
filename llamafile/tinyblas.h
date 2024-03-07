// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

enum tinyblasOperation_t {
    TINYBLAS_OP_N,
    TINYBLAS_OP_T,
};

enum tinyblasDataType_t {
    TINYBLAS_R_32F,
    TINYBLAS_R_16F,
};

enum tinyblasComputeType_t {
    TINYBLAS_COMPUTE_32F,
    TINYBLAS_COMPUTE_16F,
};

enum tinyblasGemmAlgo_t {
    TINYBLAS_GEMM_DEFAULT,
};

enum tinyblasStatus_t {
    TINYBLAS_STATUS_SUCCESS,
    TINYBLAS_STATUS_ALLOC_FAILED,
    TINYBLAS_STATUS_INVALID_VALUE,
    TINYBLAS_STATUS_NOT_SUPPORTED,
    TINYBLAS_STATUS_EXECUTION_FAILED,
    TINYBLAS_STATUS_DIMENSION_OVERFLOW,
};

struct tinyblasContext;
typedef struct tinyblasContext *tinyblasHandle_t;

const char *tinyblasGetStatusString(tinyblasStatus_t);

tinyblasStatus_t tinyblasCreate(tinyblasHandle_t *);
tinyblasStatus_t tinyblasDestroy(tinyblasHandle_t);
tinyblasStatus_t tinyblasSetStream(tinyblasHandle_t, void *);
tinyblasStatus_t tinyblasGetStream(tinyblasHandle_t, void **);

tinyblasStatus_t tinyblasSgemm(tinyblasHandle_t, tinyblasOperation_t, tinyblasOperation_t, int, int,
                               int, const float *, const float *, int, const float *, int,
                               const float *, float *, int);

tinyblasStatus_t tinyblasGemmEx(tinyblasHandle_t, tinyblasOperation_t, tinyblasOperation_t, int,
                                int, int, const void *, const void *, tinyblasDataType_t, int,
                                const void *, tinyblasDataType_t, int, const void *, void *,
                                tinyblasDataType_t, int, tinyblasComputeType_t, tinyblasGemmAlgo_t);

tinyblasStatus_t tinyblasGemmBatchedEx(tinyblasHandle_t, tinyblasOperation_t, tinyblasOperation_t,
                                       int, int, int, const void *, const void *const[],
                                       tinyblasDataType_t, int, const void *const[],
                                       tinyblasDataType_t, int, const void *, void *const[],
                                       tinyblasDataType_t, int, int, tinyblasComputeType_t,
                                       tinyblasGemmAlgo_t);

tinyblasStatus_t tinyblasGemmStridedBatchedEx(tinyblasHandle_t, tinyblasOperation_t,
                                              tinyblasOperation_t, int, int, int, const void *,
                                              const void *, tinyblasDataType_t, int, long long,
                                              const void *, tinyblasDataType_t, int, long long,
                                              const void *, void *, tinyblasDataType_t, int,
                                              long long, int, tinyblasComputeType_t,
                                              tinyblasGemmAlgo_t);
