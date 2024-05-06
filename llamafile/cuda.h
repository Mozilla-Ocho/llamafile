// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#ifndef __HIP__
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define __shfl(var, srcLane, warpSize) __shfl_sync(-1u, var, srcLane, warpSize)
#define __shfl_down(var, srcLane, warpSize) __shfl_down_sync(-1u, var, srcLane, warpSize)
#else
#define HIPBLAS_V2
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#define cudaFree hipFree
#define cublasMath_t int
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaSuccess hipSuccess
#define cudaError_t hipError_t
#define cudaEvent_t hipEvent_t
#define cudaFreeHost hipHostFree
#define cudaStream_t hipStream_t
#define cublasSgemm hipblasSgemm
#define cudaSetDevice hipSetDevice
#define cublasCreate hipblasCreate
#define cudaDataType_t hipDataType
#define cublasDestroy hipblasDestroy
#define cudaDeviceProp hipDeviceProp_t
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetLastError hipGetLastError
#define cudaDeviceReset hipDeviceReset
#define cudaEventDestroy hipEventDestroy
#define cublasSetStream hipblasSetStream
#define cudaStreamCreate hipStreamCreate
#define cudaMallocManaged hipMallocManaged
#define cudaStreamDestroy hipStreamDestroy
#define cublasComputeType_t hipblasComputeType_t
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#define cublasGetStatusString hipblasStatusToString
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocNonCoherent)
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaDeviceReset hipDeviceReset
#define cublasGemmEx hipblasGemmEx_v2
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx_v2
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_64F HIP_R_64F
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_64F HIPBLAS_COMPUTE_64F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION 0
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUBLAS_PEDANTIC_MATH 0
#define CUBLAS_DEFAULT_MATH 0
#endif

#define WARPSIZE 32
