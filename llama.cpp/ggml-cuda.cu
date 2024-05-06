// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#include <algorithm>
#include <stdarg.h>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-cuda.h"

#if defined(GGML_USE_HIPBLAS)
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#endif
#include "ggml-common.h"

#if defined(GGML_USE_HIPBLAS)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifdef GGML_USE_TINYBLAS
#include "tinyblas.cu"
#define CUBLAS_COMPUTE_16F TINYBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F TINYBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F TINYBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT TINYBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP TINYBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N TINYBLAS_OP_N
#define CUBLAS_OP_T TINYBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS TINYBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUDA_R_16F TINYBLAS_R_16F
#define CUDA_R_32F TINYBLAS_R_32F
#define cublasComputeType_t tinyblasComputeType_t
#define cublasCreate tinyblasCreate
#define cublasDestroy tinyblasDestroy
#define cublasGemmEx tinyblasGemmEx
#define cublasGemmBatchedEx tinyblasGemmBatchedEx
#define cublasGemmStridedBatchedEx tinyblasGemmStridedBatchedEx
#define cublasHandle_t tinyblasHandle_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cublasSetStream tinyblasSetStream
#define cublasSgemm tinyblasSgemm
#define cublasStatus_t tinyblasStatus_t
#define cudaDataType_t tinyblasDataType_t
#define cublasGetStatusString tinyblasGetStatusString

#else
#include <hipblas/hipblas.h>
#ifdef __HIP_PLATFORM_AMD__
// for rocblas_initialize()
#include "rocblas/rocblas.h"
#endif // __HIP_PLATFORM_AMD__
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUDA_R_16F  HIPBLAS_R_16F
#define CUDA_R_32F  HIPBLAS_R_32F
#define cublasComputeType_t hipblasDatatype_t //deprecated, new hipblasComputeType_t not in 5.6
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasStatus_t hipblasStatus_t
#define cudaDataType_t hipblasDatatype_t //deprecated, new hipblasDatatype not in 5.6
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#endif

#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventDestroy hipEventDestroy
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#ifdef GGML_HIP_UMA
#define cudaMalloc hipMallocManaged
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size)
#else
#define cudaMalloc hipMalloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#endif
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaSetDevice hipSetDevice
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamFireAndForget hipStreamFireAndForget
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamPerThread hipStreamPerThread
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent(stream, event, flags) hipStreamWaitEvent(stream, event, flags)
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define __trap abort

#elif defined(GGML_USE_TINYBLAS)
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "tinyblas.cu"

#define CUBLAS_COMPUTE_16F TINYBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F TINYBLAS_COMPUTE_32F
#define CUBLAS_OP_N TINYBLAS_OP_N
#define CUBLAS_OP_T TINYBLAS_OP_T
#define CUDA_R_16F TINYBLAS_R_16F
#define CUDA_R_32F TINYBLAS_R_32F
#define CUBLAS_GEMM_DEFAULT TINYBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP TINYBLAS_GEMM_DEFAULT
#define CUBLAS_STATUS_SUCCESS TINYBLAS_STATUS_SUCCESS
#define cublasGemmAlgo_t tinyblasGemmAlgo_t
#define cublasOperation_t tinyblasOperation_t
#define cublasComputeType_t tinyblasComputeType_t
#define cublasHandle_t tinyblasHandle_t
#define cublasStatus_t tinyblasStatus_t
#define cublasSgemm tinyblasSgemm
#define cublasGemmEx tinyblasGemmEx
#define cublasCreate tinyblasCreate
#define cublasDestroy tinyblasDestroy
#define cublasSetStream tinyblasSetStream
#define cublasGemmBatchedEx tinyblasGemmBatchedEx
#define cublasGemmStridedBatchedEx tinyblasGemmStridedBatchedEx
#define cublasGetStatusString tinyblasGetStatusString
#define cudaDataType_t tinyblasDataType_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS

#else
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020

#endif // defined(GGML_USE_HIPBLAS)

#include "ggml-backend-impl.h"

static const struct ggml_backend_api *g_backend;
#define exit g_backend->exit
#define getenv g_backend->getenv
#define FLAG_log_disable (*g_backend->FLAG_log_disable)
#define ggml_backend_register g_backend->ggml_backend_register
#define ggml_is_quantized g_backend->ggml_is_quantized
#define ggml_type_size g_backend->ggml_type_size
#define ggml_blck_size g_backend->ggml_blck_size
#define ggml_is_transposed g_backend->ggml_is_transposed
#define ggml_nbytes g_backend->ggml_nbytes
#define ggml_get_unary_op g_backend->ggml_get_unary_op
#define ggml_nelements g_backend->ggml_nelements
#define ggml_nrows g_backend->ggml_nrows
#define ggml_is_permuted g_backend->ggml_is_permuted
#define ggml_is_contiguous g_backend->ggml_is_contiguous
#define ggml_op_name g_backend->ggml_op_name
#define ggml_type_name g_backend->ggml_type_name
#define ggml_element_size g_backend->ggml_element_size
#define ggml_row_size g_backend->ggml_row_size
#define ggml_element_size g_backend->ggml_element_size
#define ggml_element_size g_backend->ggml_element_size
#define ggml_rope_yarn_corr_dims g_backend->ggml_rope_yarn_corr_dims
#define ggml_backend_buft_alloc_buffer g_backend->ggml_backend_buft_alloc_buffer
#define ggml_backend_buft_get_alloc_size g_backend->ggml_backend_buft_get_alloc_size
#define ggml_backend_cpu_buffer_type g_backend->ggml_backend_cpu_buffer_type
#define ggml_backend_buffer_init g_backend->ggml_backend_buffer_init
#define ggml_backend_cpu_buffer_from_ptr g_backend->ggml_backend_cpu_buffer_from_ptr
#define ggml_guid_matches g_backend->ggml_guid_matches
#define ggml_is_empty g_backend->ggml_is_empty
#define ggml_op_desc g_backend->ggml_op_desc

// printf() and fprintf() runtime bridge
// this is needed so text gets printed on windows
// it also helps ensure the atomicity of log lines
static void ggml_cuda_print(const char *fmt, ...) {
#define GGML_CUDA_PRINT_BUFSIZ 512
#define fflush(_) (void)0
#define printf(...) ggml_cuda_print(__VA_ARGS__)
#define fprintf(_, ...) ggml_cuda_print(__VA_ARGS__)
    int len;
    va_list va;
    char buf[GGML_CUDA_PRINT_BUFSIZ];
    va_start(va, fmt);
    len = vsnprintf(buf, GGML_CUDA_PRINT_BUFSIZ, fmt, va);
    va_end(va);
    if (len < 0)
        len = strnlen(buf, GGML_CUDA_PRINT_BUFSIZ);
    if (len >= GGML_CUDA_PRINT_BUFSIZ) {
        len = GGML_CUDA_PRINT_BUFSIZ;
        buf[len - 4] = '.';
        buf[len - 3] = '.';
        buf[len - 2] = '.';
        buf[len - 1] = '\n';
    }
    g_backend->write(2, buf, len);
}

#ifdef GGML_USE_TINYBLAS
#define BLAS_NAME "tinyBLAS"
#else
#define BLAS_NAME GGML_CUBLAS_NAME
#endif

GGML_CALL bool ggml_cuda_link(const struct ggml_backend_api *backend_api) {
    g_backend = backend_api;
    if (!FLAG_log_disable)
        fprintf(stderr, "%s: welcome to " GGML_CUDA_NAME " SDK with " BLAS_NAME "\n", __func__);
#ifdef __HIP_PLATFORM_AMD__
    // cargo culting workaround below
#ifndef GGML_USE_TINYBLAS
    rocblas_initialize();
    cudaDeviceSynchronize();
#endif
#endif
    int device_count;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
#define CUDART_HMAX   11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)
#define CUDART_HMASK  12000 // CUDA 12.0, min. ver. for half2 -> uint mask comparisons

#define CC_PASCAL     600
#define MIN_CC_DP4A   610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define CC_VOLTA      700
#define CC_AMPERE     800
#define CC_OFFSET_AMD 1000000
#define CC_RDNA1      (CC_OFFSET_AMD + 1010)
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)
#define CC_RDNA3      (CC_OFFSET_AMD + 1100)

// define this if you want to always fallback to MMQ kernels and not use cuBLAS for matrix multiplication
// on modern hardware, using cuBLAS is recommended as it utilizes F16 tensor cores which are very performant
// for large computational tasks. the drawback is that this requires some extra amount of VRAM:
// -  7B quantum model: +100-200 MB
// - 13B quantum model: +200-400 MB
//
//#define GGML_CUDA_FORCE_MMQ

// TODO: improve this to be correct for more hardware
//       for example, currently fails for GeForce GTX 1660 which is TURING arch (> VOLTA) but does not have tensor cores
#if !defined(GGML_CUDA_FORCE_MMQ)
#define CUDA_USE_TENSOR_CORES
#endif

#define MMVQ_MAX_BATCH_SIZE  8 // max batch size to use MMVQ kernels
#define  MMQ_MAX_BATCH_SIZE 32 // max batch size to use MMQ kernels when tensor cores are available

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define GGML_CUDA_MAX_STREAMS 8

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

#define CUDA_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)

#if CUDART_VERSION >= 12000 || defined(GGML_USE_TINYBLAS)
    static const char * cublas_get_error_str(const cublasStatus_t err) {
        return cublasGetStatusString(err);
    }
#else
    static const char * cublas_get_error_str(const cublasStatus_t err) {
        switch (err) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            default: return "unknown error";
        }
    }
#endif // CUDART_VERSION >= 12000

#define CUBLAS_CHECK(err) CUDA_CHECK_GEN(err, CUBLAS_STATUS_SUCCESS, cublas_get_error_str)

#if !defined(GGML_USE_HIPBLAS)
static const char * cu_get_error_str(CUresult err) {
    const char * err_str;
    cuGetErrorString(err, &err_str);
    return err_str;
}
#define CU_CHECK(err) CUDA_CHECK_GEN(err, CUDA_SUCCESS, cu_get_error_str)
#endif

#if CUDART_VERSION >= 11100
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif // CUDART_VERSION >= 11100

#ifdef GGML_CUDA_F16
typedef half dfloat; // dequantize float
typedef half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
#endif //GGML_CUDA_F16

[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    (printf)("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
           file_name, line, function_name, arch);
    GGML_UNUSED(arch_list);
#else
    (printf)("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ASSERT(false && "NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
        a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
    }
    return a;
}

static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
#pragma unroll
   for (int mask = 16; mask > 0; mask >>= 1) {
       a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, mask, 32));
   }
   return a;
#else
   GGML_UNUSED(a);
   NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

static __device__ __forceinline__ half ggml_cuda_hmax(const half a, const half b) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))

#if CUDART_VERSION >= CUDART_HMAX
    return __hmax(a, b);
#else
    return __half2float(a) > __half2float(b) ? a : b;
#endif // CUDART_VERSION >= CUDART_HMAX

#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && CUDART_VERSION < CUDART_HMAX
}
static __device__ __forceinline__ half2 ggml_cuda_hmax2(const half2 a, const half2 b) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))

#if CUDART_VERSION >= CUDART_HMAX
    return __hmax2(a, b);
#else
    half2 ret;
    reinterpret_cast<half&>(ret.x) =  __low2float(a) >  __low2float(b) ?  __low2half(a) :  __low2half(b);
    reinterpret_cast<half&>(ret.y) = __high2float(a) > __high2float(b) ? __high2half(a) : __high2half(b);
    return ret;
#endif // CUDART_VERSION >= CUDART_HMAX

#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && CUDART_VERSION < CUDART_HMAX
}

static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
#pragma unroll
   for (int mask = 16; mask > 0; mask >>= 1) {
       x = ggml_cuda_hmax2(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
   }
   return x;
#else
   GGML_UNUSED(x);
   NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
}

#if CUDART_VERSION < CUDART_HMASK
static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float( __low2half(a)) > float( __low2half(b)));
    const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
    return mask_low | mask_high;
}
#endif // CUDART_VERSION < 12000

#if defined(GGML_USE_HIPBLAS)
#define __CUDA_ARCH__ 1300

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define RDNA3
#endif

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2
#endif

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        if(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;
}

static __device__ __forceinline__ int __dp4a(const int a, const int b, int c) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx1030__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(__gfx1010__) || defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;
}
#endif // defined(GGML_USE_HIPBLAS)

#define FP16_AVAILABLE     defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) ? \
    defined(RDNA1) || defined(RDNA2) || defined(RDNA3) : __CUDA_ARCH__ >= CC_PASCAL

#define FP16_MMA_AVAILABLE !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_VOLTA

#if FP16_MMA_AVAILABLE
#include <mma.h>
#endif

// TODO: move to ggml-common.h
static const __device__ int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, dfloat2 & v);


//////////////////////

struct ggml_cuda_device_info {
    int device_count;

    struct cuda_device_info {
        int     cc;                 // compute capability
        int     nsm;                // number of streaming multiprocessors
        size_t  smpb;               // max. shared memory per block
        bool    vmm;                // virtual memory support
        size_t  vmm_granularity;    // granularity of virtual memory
        size_t  total_vram;
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

    std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};
};

const ggml_cuda_device_info & ggml_cuda_info();

void ggml_cuda_set_device(int device);
int ggml_cuda_get_device();

struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool & pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_cuda_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc &&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc &&) = delete;
};


// backend interface

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]; // events for synchronizing multiple GPUs
};

struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaEvent_t copy_event = nullptr;

    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

    explicit ggml_backend_cuda_context(int device) :
        device(device),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_context() {
        if (copy_event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(copy_event));
        }
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
                if (streams[i][j] != nullptr) {
                    CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
                }
            }
            if (cublas_handles[i] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
            }
        }
    }

    cudaStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
        }
        return streams[device][stream];
    }

    cudaStream_t stream() {
        return stream(device, 0);
    }

    cublasHandle_t cublas_handle(int device) {
        if (cublas_handles[device] == nullptr) {
            ggml_cuda_set_device(device);
            CUBLAS_CHECK(cublasCreate(&cublas_handles[device]));
            CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH));
        }
        return cublas_handles[device];
    }

    cublasHandle_t cublas_handle() {
        return cublas_handle(device);
    }

    // pool
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES];

    static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device);

    ggml_cuda_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(device);
        }
        return *pools[device];
    }

    ggml_cuda_pool & pool() {
        return pool(device);
    }
};

#define CUDA_ACC_BLOCK_SIZE 256

void ggml_cuda_op_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_ALIBI_BLOCK_SIZE 32

void ggml_cuda_op_alibi(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_ARANGE_BLOCK_SIZE 256

void ggml_cuda_op_arange(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_CLAMP_BLOCK_SIZE 256

void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

#define CUDA_CPY_BLOCK_SIZE 32

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32

void ggml_cuda_op_diag_mask_inf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dmmv = dequantize_mul_mat_vec

// TODO: remove this?
#ifndef GGML_CUDA_DMMV_X
#define GGML_CUDA_DMMV_X 32
#endif

#ifndef GGML_CUDA_MMV_Y
#define GGML_CUDA_MMV_Y 1
#endif

void ggml_cuda_op_dequantize_mul_mat_vec(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_GET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_IM2COL_BLOCK_SIZE 256

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_supports_mmq(enum ggml_type type);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_PAD_BLOCK_SIZE 256

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_POOL2D_BLOCK_SIZE 256

void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_QUANTIZE_BLOCK_SIZE 256

void quantize_row_q8_1_cuda(const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream);

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE 256

void ggml_cuda_op_timestep_embedding(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_UPSCALE_BLOCK_SIZE 256

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {16.0f, 16.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}


// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = __dp4a(vi0, u[2*i+0], sumi);
        sumi = __dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = __dp4a(vi0, u[2*i+0], sumi);
        sumi = __dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = __dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = __dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = __dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = __dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);

#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = __dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * sumi;
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = __dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * __dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float & d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;

        const int sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = __dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m    = __dp4a(m,    u[i], sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const float2 dm2f = __half22float2(dm2);

    return d8 * (dm2f.x*sumi_d - dm2f.y*sumi_m);
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = __dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;

#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = __dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;

#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = __dp4a(v0i, u[2*i+0], __dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+0], __dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;

#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = __dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;

#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = __dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = __dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = __dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = __dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc[i0/2+0]*sumi_d.x + sc[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;

#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}


static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#ifndef GGML_QKK_64
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);

#else

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    const uint16_t * a = (const uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    const float dall = bq4_K->dm[0];
    const float dmin = bq4_K->dm[1];

    const float d8_1 = __low2float(bq8_1[0].ds);
    const float d8_2 = __low2float(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * q4 = (const int *)bq4_K->qs + (iqs/2);
    const int v1 = q4[0];
    const int v2 = q4[4];

    const int dot1 = __dp4a(ui2, v2 & 0x0f0f0f0f, __dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    const int dot2 = __dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, __dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    const int dot3 = __dp4a(0x01010101, ui2, __dp4a(0x01010101, ui1, 0));
    const int dot4 = __dp4a(0x01010101, ui4, __dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    return dall * sumf_d - dmin * sumf_m;

#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A

#endif
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#ifndef GGML_QKK_64
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#else

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    const int8_t * s = bq5_K->scales;

    const float d = bq5_K->d;

    const float d8_1 = __low2half(bq8_1[0].ds);
    const float d8_2 = __low2half(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * ql = (const int *)bq5_K->qs + (iqs/2);
    const int vl1 = ql[0];
    const int vl2 = ql[4];

    const int step = 4 * (iqs/2); // 0, 4, 8, 12
    const int im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    const int in = step%8; // 0, 4, 0, 4
    const int vh = (*((const int *)(bq5_K->qh + in))) >> im;

    const int v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    const int v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    const int v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    const int v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    const float sumf_d = d8_1 * (__dp4a(ui1, v1, 0) * s[0] + __dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (__dp4a(ui3, v3, 0) * s[2] + __dp4a(ui4, v4, 0) * s[3]);

    return d * sumf_d;

#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A

#endif
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if QK_K == 256
    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq;

#if QR2_XXS == 8
    const int ib32 = iqs;
    const uint16_t * q2 = bq2->qs + 4*ib32;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
        const uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        for (int j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    const float d = (float)bq2->d * (0.5f + aux32) * __low2float(bq8_1[ib32].ds) * 0.25f;
    return d * sumi;
#else
    // iqs is 0...15
    const int ib32 = iqs/2;
    const int il = iqs%2;
    const uint16_t * q2 = bq2->qs + 4*ib32;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid1 = (const uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)bq2->d * (0.5f + (aux32 >> 28)) * __low2float(bq8_1[ib32].ds) * 0.25f;
    const uint8_t signs1 = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    const uint8_t signs2 = ksigns_iq2xs[(aux32 >> (14*il + 7)) & 127];
    const int8_t * q8 = bq8_1[ib32].qs + 16*il;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 8; ++j) {
        sumi1 += q8[j+0] * grid1[j] * (signs1 & kmask_iq2xs[j] ? -1 : 1);
        sumi2 += q8[j+8] * grid2[j] * (signs2 & kmask_iq2xs[j] ? -1 : 1);
    }
    return d * (sumi1 + sumi2);
#endif
#else
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
#if QK_K == 256
    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq;

    const int ib32 = iqs;
    const uint16_t * q2 = bq2->qs + 4*ib32;
    const int8_t   * q8 = bq8_1[ib32].qs;
    const uint8_t ls1 = bq2->scales[ib32] & 0xf;
    const uint8_t ls2 = bq2->scales[ib32] >>  4;
    int sumi1 = 0;
    for (int l = 0; l < 2; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2xs_grid + (q2[l] & 511));
        const uint32_t * signs = (const uint32_t *)(ksigns64 + (q2[l] >> 9));
        const int grid_l = __vsub4(grid[0] ^ signs[0], signs[0]);
        const int grid_h = __vsub4(grid[1] ^ signs[1], signs[1]);
        sumi1 = __dp4a(grid_l, *((const int *)q8 + 0), sumi1);
        sumi1 = __dp4a(grid_h, *((const int *)q8 + 1), sumi1);
        q8 += 8;
    }
    int sumi2 = 0;
    for (int l = 2; l < 4; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2xs_grid + (q2[l] & 511));
        const uint32_t * signs = (const uint32_t *)(ksigns64 + (q2[l] >> 9));
        const int grid_l = __vsub4(grid[0] ^ signs[0], signs[0]);
        const int grid_h = __vsub4(grid[1] ^ signs[1], signs[1]);
        sumi2 = __dp4a(grid_l, *((const int *)q8 + 0), sumi2);
        sumi2 = __dp4a(grid_h, *((const int *)q8 + 1), sumi2);
        q8 += 8;
    }
    const float d = (float)bq2->d * __low2float(bq8_1[ib32].ds) * 0.25f;
    return d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#else
    GGML_UNUSED(ksigns64);
    NO_DEVICE_CODE;
#endif
#else
    GGML_UNUSED(ksigns64);
    NO_DEVICE_CODE;
#endif
}

// TODO
static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
#if QK_K == 256
    const block_iq2_s * bq2 = (const block_iq2_s *) vbq;

    const int ib32 = iqs;
    const int8_t  * q8 = bq8_1[ib32].qs;
    const uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    const uint8_t ls1 = bq2->scales[ib32] & 0xf;
    const uint8_t ls2 = bq2->scales[ib32] >>  4;
    int sumi1 = 0;
    for (int l = 0; l < 2; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        const uint32_t signs0 = __vcmpeq4(((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        const uint32_t signs1 = __vcmpeq4(((signs[l] >>  4) * 0x01010101) & 0x08040201, 0x08040201);
        const int grid_l = __vsub4(grid[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid[1] ^ signs1, signs1);
        sumi1 = __dp4a(grid_l, *((const int *)q8 + 0), sumi1);
        sumi1 = __dp4a(grid_h, *((const int *)q8 + 1), sumi1);
        q8 += 8;
    }
    int sumi2 = 0;
    for (int l = 2; l < 4; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        const uint32_t signs0 = __vcmpeq4(((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        const uint32_t signs1 = __vcmpeq4(((signs[l] >>  4) * 0x01010101) & 0x08040201, 0x08040201);
        const int grid_l = __vsub4(grid[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid[1] ^ signs1, signs1);
        sumi2 = __dp4a(grid_l, *((const int *)q8 + 0), sumi2);
        sumi2 = __dp4a(grid_h, *((const int *)q8 + 1), sumi2);
        q8 += 8;
    }
    const float d = (float)bq2->d * __low2float(bq8_1[ib32].ds) * 0.25f;
    return d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#else
    GGML_UNUSED(ksigns64);
    NO_DEVICE_CODE;
#endif
#else
    GGML_UNUSED(ksigns64);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
#if QK_K == 256
    const block_iq3_xxs * bq2 = (const block_iq3_xxs *) vbq;

    const int ib32 = iqs;
    const uint8_t  * q3 = bq2->qs + 8*ib32;
    const uint16_t * gas = (const uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    const int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const uint32_t * grid1 = iq3xxs_grid + q3[2*l+0];
        const uint32_t * grid2 = iq3xxs_grid + q3[2*l+1];
        const uint32_t * signs = (const uint32_t *)(ksigns64 + (aux32 & 127));
        const int grid_l = __vsub4(grid1[0] ^ signs[0], signs[0]);
        const int grid_h = __vsub4(grid2[0] ^ signs[1], signs[1]);
        sumi = __dp4a(grid_l, *((int *)q8+0), sumi);
        sumi = __dp4a(grid_h, *((int *)q8+1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    const float d = (float)bq2->d * (0.5f + aux32) * __low2float(bq8_1[ib32].ds) * 0.5f;
    return d * sumi;
#else
    NO_DEVICE_CODE;
#endif
#else
    NO_DEVICE_CODE;
#endif
}

// TODO: don't use lookup table for signs
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
#if QK_K == 256
    const block_iq3_s * bq2 = (const block_iq3_s *) vbq;

    const int ib32 = iqs;
    const uint8_t  * qs = bq2->qs + 8*ib32;
    const int8_t   * q8 = bq8_1[ib32].qs;
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const uint32_t * grid1 = iq3s_grid + (qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256));
        const uint32_t * grid2 = iq3s_grid + (qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256));
        uint32_t signs0 = __vcmpeq4(((bq2->signs[4*ib32+l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        uint32_t signs1 = __vcmpeq4(((bq2->signs[4*ib32+l] >>  4) * 0x01010101) & 0x08040201, 0x08040201);
        const int grid_l = __vsub4(grid1[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid2[0] ^ signs1, signs1);
        sumi = __dp4a(grid_l, *((int *)q8+0), sumi);
        sumi = __dp4a(grid_h, *((int *)q8+1), sumi);
        q8 += 8;
    }
    const float d = (float)bq2->d * (1 + 2*((bq2->scales[ib32/2] >> 4*(ib32%2)) & 0xf)) * __low2float(bq8_1[ib32].ds);
    return d * sumi;
#else
    NO_DEVICE_CODE;
#endif
#else
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if QK_K == 256
    const block_iq1_s * bq1 = (const block_iq1_s *) vbq;

    const int ib32 = iqs;
    int sumi = 0;
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    const int * q8 = (const int *)bq8_1[ib32].qs;
    for (int l = 0; l < 4; ++l) {
        const int * grid = (const int *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        int grid0 = grid[0] & 0x0f0f0f0f;
        int grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = __dp4a(q8[2*l+1], grid1, __dp4a(q8[2*l+0], grid0, sumi));
    }
#else
    const int8_t * q8 = bq8_1[ib32].qs;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        for (int j = 0; j < 4; ++j) {
            sumi += q8[j] * (grid[j] & 0xf) + q8[j+4] * (grid[j] >> 4);
        }
        q8 += 8;
    }
#endif
    const float delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    const float d1q = (float)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    const float d = d1q * __low2float (bq8_1[ib32].ds);
    const float m = d1q * __high2float(bq8_1[ib32].ds);
    return d * sumi + m * delta;
#else
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if QK_K == 256
    const block_iq1_m * bq1 = (const block_iq1_m *) vbq;

    const int ib32 = iqs;
    int   sumi[2] = {0, 0};
    float sumf[2] = {0.f, 0.f};
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    const int * q8 = (const int *)bq8_1[ib32].qs;
    for (int l = 0; l < 4; ++l) {
        const int * grid = (const int *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        int grid0 = grid[0] & 0x0f0f0f0f;
        int grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l/2] = __dp4a(q8[2*l+1], grid1, __dp4a(q8[2*l+0], grid0, sumi[l/2]));
        const float delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        const int sumy = __dp4a(q8[2*l+1], 0x01010101, __dp4a(q8[2*l+0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }
#else
    const int8_t * q8 = bq8_1[ib32].qs;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        int sumy = 0;
        for (int j = 0; j < 4; ++j) {
            sumi[l/2] += q8[j] * (grid[j] & 0xf) + q8[j+4] * (grid[j] >> 4);
            sumy += q8[j] + q8[j+4];
        }
        const float delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        sumf[l/2] += delta*sumy;
        q8 += 8;
    }
#endif
    iq1m_scale_t scale;
    const uint16_t * sc = (const uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = (float)scale.f16 * __low2float (bq8_1[ib32].ds);
    return d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#else
    NO_DEVICE_CODE;
#endif
}

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
static __device__ __forceinline__ void get_int_from_table_16(const uint32_t & q4, const uint8_t * values,
        int & val1, int & val2) {

    uint32_t aux32; const uint8_t * q8 = (const uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = values[q8[0]] | (values[q8[1]] << 8);
    uint16_t v2 = values[q8[2]] | (values[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = values[q8[0]] | (values[q8[1]] << 8);
    v2 = values[q8[2]] | (values[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}
#endif

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_iq4_nl * bq = (const block_iq4_nl *) vbq;

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    const uint16_t * q4 = (const uint16_t *)bq->qs + 2*iqs;
    const int32_t  * q8 = (const int32_t  *)bq8_1->qs + iqs;

    const uint8_t * values = (const uint8_t *)kvalues_iq4nl;

    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_table_16(aux, values, v1, v2);
        sumi1 = __dp4a(v1, q8[l+0], sumi1);
        sumi2 = __dp4a(v2, q8[l+4], sumi2);
    }

#else
    const uint8_t * q4 = bq->qs + 4*iqs;
    const int8_t  * q8 = bq8_1->qs + 4*iqs;

    int sumi1 = 0, sumi2 = 0;
    for (int l = 0; l < 4*VDR_Q4_0_Q8_1_MMVQ; ++l) {
        sumi1 += q8[l+ 0] * kvalues_iq4nl[q4[l] & 0xf];
        sumi2 += q8[l+16] * kvalues_iq4nl[q4[l] >>  4];
    }
#endif
    const float d = (float)bq->d * __low2float(bq8_1->ds);
    return d * (sumi1 + sumi2);
}

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#if QK_K == 256
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq;
    const uint8_t * values = (const uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    const int ib32 = iqs;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    const int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)bq4->d * (ls - 32) * __low2float(bq8_1[ib32].ds);
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        get_int_from_table_16(q4[j], values, v1, v2);
        sumi1 = __dp4a(v1, q8[j+0], sumi1);
        sumi2 = __dp4a(v2, q8[j+4], sumi2);
    }
    return d * (sumi1 + sumi2);

#else
    NO_DEVICE_CODE;
#endif
#else
    return vec_dot_iq4_xs_q8_1(vbq, bq8_1, iqs);
#endif
}

static __global__ void acc_f32(const float * x, const float * y, float * dst, const int ne,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, int offset) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne) {
        return;
    }
    int src1_idx = i - offset;
    int oz = src1_idx / nb2;
    int oy = (src1_idx - (oz * nb2)) / nb1;
    int ox = src1_idx % nb1;
    if (src1_idx >= 0 && ox < ne10 && oy < ne11 && oz < ne12) {
        dst[i] = x[i] + y[ox + oy * ne10 + oz * ne10 * ne11];
    } else {
        dst[i] = x[i];
    }
}

static void acc_f32_cuda(const float * x, const float * y, float * dst, const int n_elements,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, const int offset, cudaStream_t stream) {
    int num_blocks = (n_elements + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;
    acc_f32<<<num_blocks, CUDA_ACC_BLOCK_SIZE, 0, stream>>>(x, y, dst, n_elements, ne10, ne11, ne12, nb1, nb2, offset);
}

void ggml_cuda_op_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[3] == 1); // just 3D tensors supported

    int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
    int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
    // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
    int offset = dst->op_params[3] / 4; // offset in bytes

    acc_f32_cuda(src0_d, src1_d, dst_d, ggml_nelements(dst), src1->ne[0], src1->ne[1], src1->ne[2], nb1, nb2, offset, stream);
}

static __global__ void alibi_f32(const float * x, float * dst, const int ncols, const int k_rows,
                                 const int n_heads_log2_floor, const float m0, const float m1) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;

    const int k = row/k_rows;

    float m_k;
    if (k < n_heads_log2_floor) {
        m_k = powf(m0, k + 1);
    } else {
        m_k = powf(m1, 2 * (k - n_heads_log2_floor) + 1);
    }

    dst[i] = col * m_k + x[i];
}

static void alibi_f32_cuda(const float * x, float * dst, const int ncols, const int nrows,
                           const int k_rows, const int n_heads_log2_floor, const float m0,
                           const float m1, cudaStream_t stream) {
    const dim3 block_dims(CUDA_ALIBI_BLOCK_SIZE, 1, 1);
    const int num_blocks_x = (ncols + CUDA_ALIBI_BLOCK_SIZE - 1) / (CUDA_ALIBI_BLOCK_SIZE);
    const dim3 block_nums(num_blocks_x, nrows, 1);
    alibi_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, k_rows, n_heads_log2_floor, m0, m1);
}

void ggml_cuda_op_alibi(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t nrows = ggml_nrows(src0);

    //const int n_past = ((int32_t *) dst->op_params)[0];
    const int n_head = ((int32_t *) dst->op_params)[1];
    float max_bias;
    memcpy(&max_bias, (int32_t *) dst->op_params + 2, sizeof(float));

    //GGML_ASSERT(ne01 + n_past == ne00);
    GGML_ASSERT(n_head == ne02);

    const int n_heads_log2_floor = 1 << (int) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

    alibi_f32_cuda(src0_d, dst_d, ne00, nrows, ne01, n_heads_log2_floor, m0, m1, stream);
}

static __global__ void arange_f32(float * dst, const int ne0, const float start, const float step) {
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    dst[nidx] = start + step * nidx;
}

static void arange_f32_cuda(float * dst, const int ne0, const float start, const float step, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_ARANGE_BLOCK_SIZE - 1) / CUDA_ARANGE_BLOCK_SIZE;
    arange_f32<<<num_blocks, CUDA_ARANGE_BLOCK_SIZE, 0, stream>>>(dst, ne0, start,  step);
}

void ggml_cuda_op_arange(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float start;
    float stop;
    float step;
    memcpy(&start, (float *)dst->op_params + 0, sizeof(float));
    memcpy(&stop,  (float *)dst->op_params + 1, sizeof(float));
    memcpy(&step,  (float *)dst->op_params + 2, sizeof(float));

    int64_t steps = (int64_t)ceil((stop - start) / step);
    GGML_ASSERT(ggml_nelements(dst) == steps);

    arange_f32_cuda(dst_d, dst->ne[0], start, step, stream);
}

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.y;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void argsort_f32_i32_cuda(const float * x, int * dst, const int ncols, const int nrows, ggml_sort_order order, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ASSERT(false);
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_cuda(src0_d, (int *)dst_d, ncols, nrows, order, stream);
}

static __device__ __forceinline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

static __device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_bin_bcast(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s00,*/ int s01, int s02, int s03,
        /*int s10,*/ int s11, int s12, int s13) {
    const int i0s = blockDim.x*blockIdx.x + threadIdx.x;
    const int i1 = (blockDim.y*blockIdx.y + threadIdx.y);
    const int i2 = (blockDim.z*blockIdx.z + threadIdx.z) / ne3;
    const int i3 = (blockDim.z*blockIdx.z + threadIdx.z) % ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x*gridDim.x) {
        const int i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
    }
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_bin_bcast_unravel(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s00,*/ int s01, int s02, int s03,
        /*int s10,*/ int s11, int s12, int s13) {

    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    const int i3 = i/(ne2*ne1*ne0);
    const int i2 = (i/(ne1*ne0)) % ne2;
    const int i1 = (i/ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    const int i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
}

template<float (*bin_op)(const float, const float)>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {

        GGML_TENSOR_BINARY_OP_LOCALS

        int nr0 = ne10/ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};

        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};

        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
            for (int i = 0; i < 4; i++) {
                if (nr[i] != 1) {
                    break;
                }
                if (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }

        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            //int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
            //int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
            //int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
            //int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            size_t s00 = nb00 / sizeof(src0_t);
            size_t s01 = nb01 / sizeof(src0_t);
            size_t s02 = nb02 / sizeof(src0_t);
            size_t s03 = nb03 / sizeof(src0_t);

            GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

            GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

            GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

            GGML_ASSERT(s0 == 1);
            GGML_ASSERT(s00 == 1);
            GGML_ASSERT(s10 == 1);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            dim3 block_dims;
            block_dims.x = std::min<unsigned int>(hne0, block_size);
            block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
            block_dims.z = std::min(std::min<unsigned int>(ne2*ne3, block_size / block_dims.x / block_dims.y), 64U);

            dim3 block_nums(
                (hne0 + block_dims.x - 1) / block_dims.x,
                (ne1 + block_dims.y - 1) / block_dims.y,
                (ne2*ne3 + block_dims.z - 1) / block_dims.z
            );

            if (block_nums.z > 65535) {
                // this is the maximum number of blocks in z dimension, fallback to 1D grid kernel
                int block_num = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                k_bin_bcast_unravel<bin_op><<<block_num, block_size, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00, */ s01, s02, s03,
                    /* s10, */ s11, s12, s13);
            } else {
                k_bin_bcast<bin_op><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00, */ s01, s02, s03,
                    /* s10, */ s11, s12, s13);
            }
        }
    }
};

template<class op>
static void ggml_cuda_op_bin_bcast(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const void * src0_dd, const void * src1_dd, void * dst_dd, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (half *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ASSERT(false);
    }
}

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat>>(dst, dst->src[0], dst, nullptr, dst->src[0]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

static __global__ void clamp_f32(const float * x, float * dst, const float min, const float max, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = x[i] < min ? min : (x[i] > max ? max : x[i]);
}

static void clamp_f32_cuda(const float * x, float * dst, const float min, const float max, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
    clamp_f32<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max, k);
}


void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    clamp_f32_cuda(src0_d, dst_d, min, max, ggml_nelements(src0), stream);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void concat_f32(const float * x,const float * y, float * dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    // operation
    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;
    if (blockIdx.z < ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float * x, const float * y, float * dst, const int ne0, int ne1, int ne2, int ne02, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    concat_f32<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    for (int i3 = 0; i3 < dst->ne[3]; i3++) {
        concat_f32_cuda(src0_d + i3 * (src0->nb[3] / 4), src1_d + i3 * (src1->nb[3] / 4), dst_d + i3 * (dst->nb[3] / 4), dst->ne[0], dst->ne[1], dst->ne[2], src0->ne[2], stream);
    }
}

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i = (int64_t)2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const int64_t ib = i/qk; // block index
    const int64_t iqs = (i%qk)/qr; // quant index
    const int64_t iybs = i - i%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED(vx);
    GGML_UNUSED(y);
    GGML_UNUSED(k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t tid = threadIdx.x;
#if QK_K == 256
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#else
    const int64_t is = tid/16;  // 0 or 1
    const int64_t il = tid%16;  // 0...15
    const uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;
    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

#if QK_K == 256
    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#else
    const int64_t tid = threadIdx.x;
    const int64_t is  = tid/16;  // 0 or 1
    const int64_t il  = tid%16;  // 0...15
    const int64_t im  = il/8;    // 0...1
    const int64_t in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    const uint8_t q = x[i].qs[il] >> (2*is);
    const uint8_t h = x[i].hmask[in] >> (2*is + im);
    const float   d = (float)x[i].d;

    if (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } else {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

#if QK_K == 256
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = blockIdx.x;

#if QK_K == 256
    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
#else
    const int64_t tid = threadIdx.x;
    const uint8_t * q = x[i].qs;
    dst_t * y = yy + i*QK_K;
    const float d = (float)x[i].dm[0];
    const float m = (float)x[i].dm[1];
    y[tid+ 0] = d * (x[i].scales[0] & 0xF) * (q[tid] & 0xF) - m * (x[i].scales[0] >> 4);
    y[tid+32] = d * (x[i].scales[1] & 0xF) * (q[tid] >>  4) - m * (x[i].scales[1] >> 4);
#endif
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = blockIdx.x;

#if QK_K == 256
    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#else
    const int64_t tid = threadIdx.x;
    const uint8_t q = x[i].qs[tid];
    const int64_t im = tid/8;  // 0...3
    const int64_t in = tid%8;  // 0...7
    const int64_t is = tid/16; // 0 or 1
    const uint8_t h = x[i].qh[in] >> im;
    const float d = x[i].d;
    dst_t * y = yy + i*QK_K + tid;
    y[ 0] = d * x[i].scales[is+0] * ((q & 0xF) - ((h >> 0) & 1 ? 0 : 16));
    y[32] = d * x[i].scales[is+2] * ((q >>  4) - ((h >> 4) & 1 ? 0 : 16));
#endif
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;
#if QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#else

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/16;         // 0 or 1
    const int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    const float d = x[i].d;

    const uint8_t   ql = x[i].ql[16*ip + il];
    const uint8_t   qh = x[i].qh[il] >> (2*ip);
    const int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t tid = threadIdx.x;
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#else
    NO_DEVICE_CODE;
#endif

}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t tid = threadIdx.x;
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#else
    NO_DEVICE_CODE;
#endif

}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t tid = threadIdx.x;
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#else
    NO_DEVICE_CODE;
#endif

}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t tid = threadIdx.x;
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#else
    NO_DEVICE_CODE;
#endif

}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t tid = threadIdx.x;
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#else
    NO_DEVICE_CODE;
#endif

}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = threadIdx.x;
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#else
    NO_DEVICE_CODE;
#endif

}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = threadIdx.x;
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#else
    NO_DEVICE_CODE;
#endif

}


template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}

#if QK_K != 64
template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}
#endif

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE);
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q2_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q3_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q5_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q6_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
#if QK_K == 64
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
#else
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template <typename src_t, typename dst_t>
static __global__ void convert_unary(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const src_t * x = (src_t *) vx;

    y[i] = x[i];
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    int id;
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            CUDA_CHECK(cudaGetDevice(&id));
            if (ggml_cuda_info().devices[id].cc >= CC_PASCAL) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half>;
        default:
            return nullptr;
    }
}

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

static __device__ void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

static __device__ void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    half * dsti = (half *) cdsti;

    *dsti = __float2half(*xi);
}

static __device__ void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const half * xi = (const half *) cxi;
    half * dsti = (half *) cdsti;

    *dsti = *xi;
}

static __device__ void cpy_1_f16_f32(const char * cxi, char * cdsti) {
    const half * xi = (const half *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

template <cpy_kernel_t cpy_1>
static __global__ void cpy_f32_f16(const char * cx, char * cdst, const int ne,
                                   const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                                   const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q8_0 * dsti = (block_q8_0 *) cdsti;

    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = xi[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = xi[j]*id;

        dsti->qs[j] = roundf(x0);
    }
}

static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_0 * dsti = (block_q4_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_1 * dsti = (block_q4_1 *) cdsti;

    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = xi[j];

        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->dm.x = d;
    dsti->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (xi[0       + j] - vmin)*id;
        const float x1 = (xi[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q5_0 * dsti = (block_q5_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        dsti->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q5_1 * dsti = (block_q5_1 *) cdsti;

    float min = xi[0];
    float max = xi[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = xi[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->dm.x = d;
    dsti->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (xi[0       + j] - min)*id;
        const float x1 = (xi[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dsti->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}


static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = xi[0        + j]*id;
        const float x1 = xi[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = xi[0        + j]*xi[0        + j];
        const float w1 = xi[QK4_NL/2 + j]*xi[QK4_NL/2 + j];
        sumqx += w0*v0*xi[j] + w1*v1*xi[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    dsti->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_f32_q(const char * cx, char * cdst, const int ne,
                                 const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                                 const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                 const int nb12, const int nb13) {
    const int i = (blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int i03 = i/(ne00 * ne01 * ne02);
    const int i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int i13 = i/(ne10 * ne11 * ne12);
    const int i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = (i10/qk)*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

static void ggml_cpy_f16_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f16_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q8_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    cpy_f32_q<cpy_blck_f32_q8_0, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    cpy_f32_q<cpy_blck_f32_q4_0, QK4_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_1_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    cpy_f32_q<cpy_blck_f32_q4_1, QK4_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ne / QK5_0;
    cpy_f32_q<cpy_blck_f32_q5_0, QK5_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_1_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ne / QK5_1;
    cpy_f32_q<cpy_blck_f32_q5_1, QK5_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_iq4_nl_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ne / QK4_NL;
    cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f16_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f16_f16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX);
    GGML_ASSERT(ggml_nbytes(src1) <= INT_MAX);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    //GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    //GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    cudaStream_t main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else {
        fprintf(stderr, "%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ASSERT(false);
    }
}

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_cuda_cpy(ctx, src0, dst);
}

static __global__ void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past) {
    const int col = blockDim.y*blockIdx.y + threadIdx.y;
    const int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void diag_mask_inf_f32_cuda(const float * x, float * dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream) {
    const dim3 block_dims(1, CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
    const dim3 block_nums(nrows_x, block_num_x, 1);
    diag_mask_inf_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x, rows_per_channel, n_past);
}

void ggml_cuda_op_diag_mask_inf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int nrows0 = ggml_nrows(src0);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_cuda(src0_d, dst_d, ne00, nrows0, ne01, n_past, stream);
}

#ifndef K_QUANTS_PER_ITERATION
#define K_QUANTS_PER_ITERATION 2
#else
static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#endif

static __global__ void dequantize_mul_mat_vec_q2_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...15
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2
    const int q_offset = 32*im + l0;
    const int s_offset = 8*im;
    const int y_offset = 128*im + l0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

        const uint32_t * a = (const uint32_t *)(x[i].scales + s_offset);
        aux[0] = a[0] & 0x0f0f0f0f;
        aux[1] = a[1] & 0x0f0f0f0f;
        aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
        aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                  + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                  + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                  + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                  + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                  + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                  + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                  +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
            sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                  + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

        }
        tmp += dall * sum1 - dmin * sum2;

    }
#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    const uint8_t * d = (const uint8_t *)uaux;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint32_t * s = (const uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        const float2 dall = __half22float2(x[i].dm);

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x * sum1 - dall.y * sum2;
    }
#endif

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

static __global__ void dequantize_mul_mat_vec_q3_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0....15 or 0...7

    const uint8_t m = 1 << (4*im);

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2
    const int q_offset =  32*im + l0;
    const int y_offset = 128*im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4*im;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;
        const uint8_t * h = x[i].hmask + l0;

        const uint16_t * a = (const uint16_t *)x[i].scales;
        utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
        utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
        utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
        utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < n; ++l) {
            sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                 + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                 + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                 + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
            sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                 + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                 + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
        }
        tmp += d * sum;

    }
#else

    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    const int in = offset/8;                                 // 0 or 1
    const int im = offset%8;                                 // 0...7

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint8_t * s = x[i].scales;

        const float dall = (float)x[i].d;

        float sum = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t hl = x[i].hmask[im+l] >> in;
            const uint8_t ql = q[l];
            sum += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

static __global__ void dequantize_mul_mat_vec_q4_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il  = tid/step;                            // 0...3
    const int ir  = tid - step*il;                       // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y1 = yy + i*QK_K + y_offset;
        const float   * y2 = y1 + 128;

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
        const uint32_t * q1 = (const uint32_t *)(x[i].qs + q_offset);
        const uint32_t * q2 = q1 + 16;

        q32[0] = q1[0] & 0x0f0f0f0f;
        q32[1] = q1[0] & 0xf0f0f0f0;
        q32[2] = q2[0] & 0x0f0f0f0f;
        q32[3] = q2[0] & 0xf0f0f0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 4; ++l) {
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+ 4];
            s.z += y2[l] * q4[l+8]; s.w += y2[l+32] * q4[l+12];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
#else
        const uint16_t * q1 = (const uint16_t *)(x[i].qs + q_offset);
        const uint16_t * q2 = q1 + 32;

        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[0] & 0xf0f0;
        q16[2] = q2[0] & 0x0f0f;
        q16[3] = q2[0] & 0xf0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 2; ++l) {
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+2];
            s.z += y2[l] * q4[l+4]; s.w += y2[l+32] * q4[l+6];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
#endif

    }
#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const float d = (float)x[i].dm[0];
        const float m = (float)x[i].dm[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static __global__ void dequantize_mul_mat_vec_q5_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols) {

    const int row = blockIdx.x;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/2;  // 0...15
    const int ix  = threadIdx.x%2;

    const int il  = tid/4;     // 0...3
    const int ir  = tid - 4*il;// 0...3
    const int n   = 2;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1  = 1 << (2*im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * ql1 = x[i].qs + q_offset;
        const uint8_t * qh  = x[i].qh + l0;
        const float   * y1  = yy + i*QK_K + y_offset;
        const float   * y2  = y1 + 128;

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        float4 sum = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        const uint16_t * q1 = (const uint16_t *)ql1;
        const uint16_t * q2 = q1 + 32;
        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[8] & 0x0f0f;
        q16[2] = (q1[0] >> 4) & 0x0f0f;
        q16[3] = (q1[8] >> 4) & 0x0f0f;
        q16[4] = q2[0] & 0x0f0f;
        q16[5] = q2[8] & 0x0f0f;
        q16[6] = (q2[0] >> 4) & 0x0f0f;
        q16[7] = (q2[8] >> 4) & 0x0f0f;
        for (int l = 0; l < n; ++l) {
            sum.x += y1[l+ 0] * (q4[l +0] + (qh[l+ 0] & (hm1 << 0) ? 16 : 0))
                   + y1[l+16] * (q4[l +2] + (qh[l+16] & (hm1 << 0) ? 16 : 0));
            sum.y += y1[l+32] * (q4[l +4] + (qh[l+ 0] & (hm1 << 1) ? 16 : 0))
                   + y1[l+48] * (q4[l +6] + (qh[l+16] & (hm1 << 1) ? 16 : 0));
            sum.z += y2[l+ 0] * (q4[l +8] + (qh[l+ 0] & (hm2 << 0) ? 16 : 0))
                   + y2[l+16] * (q4[l+10] + (qh[l+16] & (hm2 << 0) ? 16 : 0));
            sum.w += y2[l+32] * (q4[l+12] + (qh[l+ 0] & (hm2 << 1) ? 16 : 0))
                   + y2[l+48] * (q4[l+14] + (qh[l+16] & (hm2 << 1) ? 16 : 0));
            smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                  + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
        }
        tmp += dall * (sum.x * sc[0] + sum.y * sc[1] + sum.z * sc[4] + sum.w * sc[5]) - dmin * smin;
    }

#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);
    const int step = tid * K_QUANTS_PER_ITERATION;
    const int im = step/8;
    const int in = step%8;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const int8_t  * s = x[i].scales;
        const float   * y = yy + i*QK_K + step;
        const float     d = x[i].d;
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            const uint8_t h = x[i].qh[in+j] >> im;
            sum += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

static __global__ void dequantize_mul_mat_vec_q6_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

#if QK_K == 256

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif
    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = x[i].d;

#if K_QUANTS_PER_ITERATION == 1
        float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                  + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                  + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                  + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                  + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                  + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                  + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                  +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
        tmp += sum;
#else
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                 + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                 + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                 + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        tmp += sum;
#endif

    }

#else

    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = x[i].ql + step;
        const uint8_t * qh = x[i].qh + step;
        const int8_t  * s  = x[i].scales;

        const float d = x[i+0].d;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static __device__ void convert_f16(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const half * x = (const half *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x = x[ib + iqs + 0];
    v.y = x[ib + iqs + 1];
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_mul_mat_vec(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int64_t row = (int64_t)blockIdx.x*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_CUDA_F16
    half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_CUDA_F16

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int64_t ib = ((int64_t)row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_CUDA_F16
            tmp += __hmul2(v, {
                y[iybs + iqs + j/qr + 0],
                y[iybs + iqs + j/qr + y_offset]
            });
#else
            tmp += v.x * y[iybs + iqs + j/qr + 0];
            tmp += v.y * y[iybs + iqs + j/qr + y_offset];
#endif // GGML_CUDA_F16
        }
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (tid == 0) {
#ifdef GGML_CUDA_F16
        dst[row] = tmp.x + tmp.y;
#else
        dst[row] = tmp;
#endif // GGML_CUDA_F16
    }
}

static void dequantize_mul_mat_vec_q4_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q4_1_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q5_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q5_1_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q8_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q2_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q2_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q3_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q3_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q4_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q4_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q5_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const dim3 block_dims(32, 1, 1);
    dequantize_mul_mat_vec_q5_k<<<nrows, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q6_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q6_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void convert_mul_mat_vec_f16_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<1, 1, convert_f16>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

void ggml_cuda_op_dequantize_mul_mat_vec(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    GGML_UNUSED(ctx);
    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_CUDA_F16
    ggml_cuda_pool_alloc<half> src1_dfloat_a(ctx.pool());
    half * src1_dfloat = nullptr; // dfloat == half

    bool src1_convert_f16 =
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_F16;

    if (src1_convert_f16) {
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        to_fp16_cuda(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#else
    const dfloat * src1_dfloat = (const dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_CUDA_F16

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            dequantize_mul_mat_vec_q4_0_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_mul_mat_vec_q8_0_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_q2_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_q3_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_q4_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_q5_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_q6_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_F16:
            convert_mul_mat_vec_f16_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

template<int D, int parallel_blocks> // D == head size
__launch_bounds__(((D + WARP_SIZE - 1) / WARP_SIZE)*WARP_SIZE, 1)
static __global__ void flash_attn_vec_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#if FP16_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic = blockIdx.x / parallel_blocks; // Index of the Q/QKV column to work on.
    const int ip = blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.y              + nb01*ic);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const half   * V_h   = (const half   *) (V    + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask + ne11*ic;

    const int stride_KV  = nb11 / sizeof(half);
    const int stride_KV2 = nb11 / sizeof(half2);

    constexpr int nwarps = (D + WARP_SIZE - 1) / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nwarps*WARP_SIZE);

    __shared__ half KQ[nwarps*WARP_SIZE];
    KQ[tid] = -INFINITY;
    half2 * KQ2 = (half2 *) KQ;

    half kqmax = -HALF_MAX_HALF;
    half kqsum = 0.0f;

    __shared__ half kqmax_shared[WARP_SIZE];
    __shared__ half kqsum_shared[WARP_SIZE];
    if (threadIdx.y == 0) {
        kqmax_shared[threadIdx.x] = -HALF_MAX_HALF;
        kqsum_shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Convert Q to half2 and store in registers:
    half2 Q_h2[(D/2 + WARP_SIZE - 1) / WARP_SIZE];
#pragma unroll
    for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
        const int i = i0 + threadIdx.x;
        if (i0 + WARP_SIZE > D/2 && i >= D/2) {
            break;
        }

        Q_h2[i0/WARP_SIZE] = make_half2(scale, scale) * make_half2(Q_f2[i].x, Q_f2[i].y);
    }

    half2 VKQ = make_half2(0.0f, 0.0f); // Each thread calculates a single VKQ value.

    const int k_start  = parallel_blocks == 1 ? 0 : ip*D;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*D) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        half kqmax_new = kqmax;
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

            half2 sum2 = make_half2(0.0f, 0.0f);
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;
                if (k_KQ_0 + WARP_SIZE > D/2 && k_KQ >= D/2) {
                    break;
                }

                const half2 K_ik = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
                sum2 += K_ik * Q_h2[k_KQ_0/WARP_SIZE];
            }

            sum2 = warp_reduce_sum(sum2);
            half sum = __low2half(sum2) + __high2half(sum2);
            sum += mask ? maskh[k_VKQ_0 + i_KQ] : __float2half(0.0f);
            kqmax_new = ggml_cuda_hmax(kqmax_new, sum);
            if (threadIdx.x == 0) {
                KQ[i_KQ] = sum;
            }
        }

        kqmax_new = warp_reduce_max(kqmax_new);
        if (threadIdx.x == 0) {
            kqmax_shared[threadIdx.y] = kqmax_new;
        }
        __syncthreads();
        kqmax_new = kqmax_shared[threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);

        const half KQ_max_scale = hexp(kqmax - kqmax_new);
        kqmax = kqmax_new;

        const half val = hexp(KQ[tid] - kqmax);
        kqsum = kqsum*KQ_max_scale + val;
        KQ[tid] = val;

        VKQ *= __half2half2(KQ_max_scale);

        __syncthreads();

        if (tid < D) {
#pragma unroll
            for (int k0 = 0; k0 < D; k0 += 2) {
                if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                    break;
                }

                half2 V_k;
                reinterpret_cast<half&>(V_k.x) = V_h[(k_VKQ_0 + k0 + 0)*stride_KV + tid];
                reinterpret_cast<half&>(V_k.y) = V_h[(k_VKQ_0 + k0 + 1)*stride_KV + tid];
                VKQ += V_k*KQ2[k0/2];
            }
        }

        __syncthreads();
    }

    if (tid >= D) {
        kqsum = 0.0f;
    }

    kqsum = warp_reduce_sum(kqsum);
    if (threadIdx.x == 0) {
        kqsum_shared[threadIdx.y] = kqsum;
    }
    __syncthreads();
    kqsum = kqsum_shared[threadIdx.x];
    kqsum = warp_reduce_sum(kqsum);

    if (tid >= D) {
        return;
    }

    half dst_val = (__low2half(VKQ) + __high2half(VKQ));
    if (parallel_blocks == 1) {
        dst_val /= kqsum;
    }
    dst[D*gridDim.y*blockIdx.x + D*blockIdx.y + tid] = dst_val;

    if (parallel_blocks == 1 || tid != 0) {
        return;
    }
    dst_meta[ic*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax, kqsum);
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// D == head size, VKQ_stride == num VKQ rows calculated in parallel:
template<int D, int ncols, int nwarps, int VKQ_stride, int parallel_blocks, typename KQ_acc_t>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#if FP16_MMA_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = ncols*(blockIdx.x / parallel_blocks); // Index of the first Q/QKV column to work on.
    const int ip  =        blockIdx.x % parallel_blocks;  // Index in group of blocks running for the same column in parallel.

    static_assert(D <= FATTN_KQ_STRIDE, "D must be <= FATTN_KQ_STRIDE.");
    static_assert(ncols == 8 || ncols % 16 == 0, "ncols must be 8 or a multiple of 16.");
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0, "If ncols == 8 then D % frag_m must be 0.");
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::row_major> frag_a_K;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_a_V;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_b;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, KQ_acc_t>                      frag_c_KQ;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, half>                          frag_c_VKQ;

    constexpr int KQ_stride_tc  = nwarps*frag_m; // Number of KQ rows calculated in parallel.
    constexpr int VKQ_ratio = KQ_stride_tc/VKQ_stride; // Number of parallel VKQ accumulators needed to keep all warps busy.
    static_assert(VKQ_ratio <= nwarps, "VKQ_ratio must be <= nwarps.");

    // Pad internal representation of KQ, KQV to reduce shared memory bank conflicts:
    constexpr int D_padded = D + 8;
    constexpr int kqs_padded = FATTN_KQ_STRIDE + 8;
    constexpr int kqar = sizeof(KQ_acc_t)/sizeof(half);

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f   = (const float *) (Q + nb02* blockIdx.y              + nb01*ic0);
    const half  * K_h   = (const half  *) (K + nb12*(blockIdx.y / gqa_ratio));
    const half  * V_h   = (const half  *) (V + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half  * maskh = (const half  *)  mask + (nb31/sizeof(half))* ic0;
    const half2 * mask2 = (const half2 *)  mask + (nb31/sizeof(half))*(ic0/2);

    const int stride_Q  = nb01 / sizeof(float);
    const int stride_KV = nb11 / sizeof(half);

    frag_b Q_b[D/16][ncols/frag_n];

    // A single buffer for temporarily holding tiles of KQ and VKQ parts:
    constexpr int mem_KQ = ncols*kqs_padded*kqar;
    constexpr int mem_VKQ_parts = VKQ_ratio*ncols*D_padded;
    __shared__ half KQ[mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts];
    float * KQ_f = (float *) KQ;
    half2 * KQ2 = (half2 *) KQ;

    float    KQ_rowsum_f[ncols/nwarps] = {0.0f};
    float       KQ_max_f[ncols/nwarps];
    float KQ_max_scale_f[ncols/nwarps] = {0.0f};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_f[j] = -FLT_MAX/2.0f;
    }

    half2    KQ_rowsum_h2[ncols/nwarps] = {{0.0f, 0.0f}};
    half2       KQ_max_h2[ncols/nwarps];
    half2 KQ_max_scale_h2[ncols/nwarps] = {{0.0f, 0.0f}};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_h2[j] = make_half2(-HALF_MAX_HALF, -HALF_MAX_HALF);
    }

    __shared__ half VKQ[ncols*D_padded]; // Accumulator for final VKQ slice.
    half2 * VKQ2 = (half2 *) VKQ;
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D/2 && i >= D/2) {
                break;
            }
            VKQ2[j*(D_padded/2) + i] = make_half2(0.0f, 0.0f);
        }
    }

    // Convert Q to half and apply scale, temporarily store in KQ:
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D && i >= D) {
                break;
            }
            KQ[j*D_padded + i] = ic0 + j < ne01 ? Q_f[j*stride_Q + i] * scale : 0.0f;
        }
    }

    __syncthreads();

    // Load Q into tensor core fragments/registers since it will be used frequently:
#pragma unroll
    for (int i0 = 0; i0 < D; i0 += 16) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
            nvcuda::wmma::load_matrix_sync(Q_b[i0/16][j0/frag_n], KQ + j0*D_padded + i0, D_padded);
        }
    }

    __syncthreads();

    // Iterate over ne11 == previous tokens:
    for (int k_VKQ_0 = ip*FATTN_KQ_STRIDE; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE) {
        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE; i_KQ_0 += KQ_stride_tc) {
            frag_c_KQ KQ_c[ncols/frag_n];
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(KQ_c[j], 0.0f);
            }
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 16) {
                frag_a_K K_a;
                nvcuda::wmma::load_matrix_sync(K_a, K_h + (k_VKQ_0 + i_KQ_0 + frag_m*threadIdx.y)*stride_KV + k_KQ_0, stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    nvcuda::wmma::mma_sync(KQ_c[j], K_a, Q_b[k_KQ_0/16][j], KQ_c[j]);
                }
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync((KQ_acc_t *) KQ + j0*kqs_padded + i_KQ_0 + frag_m*threadIdx.y, KQ_c[j0/frag_n], kqs_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (std::is_same<KQ_acc_t, float>::value) {
                float KQ_f_tmp[FATTN_KQ_STRIDE / WARP_SIZE];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/WARP_SIZE] = KQ_f[j*kqs_padded + k];
                }

                float KQ_max_new = KQ_max_f[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/WARP_SIZE] += mask ? __half2float(maskh[j*(nb31/sizeof(half)) + k_VKQ_0 + k]) : 0.0f;
                    KQ_max_new = max(KQ_max_new, KQ_f_tmp[k0/WARP_SIZE]);
                }
                KQ_max_new = warp_reduce_max(KQ_max_new);

                const float diff = KQ_max_f[j0/nwarps] - KQ_max_new;
                KQ_max_scale_f[j0/nwarps] = expf(diff);
                if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_max_scale_f[j0/nwarps] = 0.0f;
                }
                KQ_max_f[j0/nwarps] = KQ_max_new;

                float KQ_rowsum_add = 0.0f;
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    const float diff = KQ_f_tmp[k0/WARP_SIZE] - KQ_max_f[j0/nwarps];
                    KQ_f_tmp[k0/WARP_SIZE] = expf(diff);
                    if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                        KQ_f_tmp[k0/WARP_SIZE] = 0.0f;
                    }
                    KQ_rowsum_add += KQ_f_tmp[k0/WARP_SIZE];
                    KQ[j*(kqar*kqs_padded) + k] = KQ_f_tmp[k0/WARP_SIZE];
                }
                KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_f[j0/nwarps] = KQ_max_scale_f[j0/nwarps]*KQ_rowsum_f[j0/nwarps] + KQ_rowsum_add;
            } else {
                half2 KQ2_tmp[FATTN_KQ_STRIDE/(2*WARP_SIZE)];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/WARP_SIZE] = KQ2[j*(kqs_padded/2) + k];
                }

                half2 KQ_max_new = KQ_max_h2[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/WARP_SIZE] += mask ? mask2[(j*ne11 + k_VKQ_0)/2 + k] : make_half2(0.0f, 0.0f);
                    KQ_max_new = ggml_cuda_hmax2(KQ_max_new, KQ2_tmp[k0/WARP_SIZE]);
                }
                KQ_max_new = __half2half2(warp_reduce_max(ggml_cuda_hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))));
                const half2 diff = KQ_max_h2[j0/nwarps] - KQ_max_new;
                KQ_max_scale_h2[j0/nwarps] = h2exp(diff);
                const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                *((uint32_t *) &KQ_max_scale_h2[j0/nwarps]) &= ftz_mask;
                KQ_max_h2[j0/nwarps] = KQ_max_new;

                half2 KQ_rowsum_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    const half2 diff = KQ2_tmp[k0/WARP_SIZE] - KQ_max_h2[j0/nwarps];
                    KQ2_tmp[k0/WARP_SIZE] = h2exp(diff);
                    const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                    *((uint32_t *) &KQ2_tmp[k0/WARP_SIZE]) &= ftz_mask;
                    KQ_rowsum_add += KQ2_tmp[k0/WARP_SIZE];
                    KQ2[j*(kqs_padded/2) + k] = KQ2_tmp[k0/WARP_SIZE];
                }
                KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_h2[j0/nwarps] = KQ_max_scale_h2[j0/nwarps]*KQ_rowsum_h2[j0/nwarps] + KQ_rowsum_add;
            }
        }

        __syncthreads();

        frag_b KQ_b[FATTN_KQ_STRIDE/(VKQ_ratio*16)][ncols/frag_n];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio)*16;
                nvcuda::wmma::load_matrix_sync(
                    KQ_b[k0/(VKQ_ratio*16)][j0/frag_n],
                    KQ + j0*(kqar*kqs_padded) + k,
                    kqar*kqs_padded);
            }
        }

        frag_c_VKQ VKQ_c[D/VKQ_stride][ncols/frag_n];
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(VKQ_c[i_VKQ_0/VKQ_stride][j], 0.0f);
            }

#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio)*16;

                frag_a_V v_a;
                nvcuda::wmma::load_matrix_sync(v_a, V_h + (k_VKQ_0 + k)*stride_KV + i_VKQ_0 + frag_m*(threadIdx.y/VKQ_ratio), stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    nvcuda::wmma::mma_sync(VKQ_c[i_VKQ_0/VKQ_stride][j], v_a, KQ_b[k0/(VKQ_ratio*16)][j], VKQ_c[i_VKQ_0/VKQ_stride][j]);
                }
            }
        }

        __syncthreads();

        const int offset_k = (threadIdx.y % VKQ_ratio) * (ncols*D_padded);
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += VKQ_stride) {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync(
                    KQ + offset_k + j0*D_padded + i_KQ_0 + frag_m*(threadIdx.y/VKQ_ratio),
                    VKQ_c[i_KQ_0/VKQ_stride][j0/frag_n],
                    D_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            half2 VKQ_scale;
            if (std::is_same<KQ_acc_t, float>::value) {
                VKQ_scale = make_half2(KQ_max_scale_f[j0/nwarps], KQ_max_scale_f[j0/nwarps]);
            } else {
                VKQ_scale = KQ_max_scale_h2[j0/nwarps];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                if (i0 + WARP_SIZE > D/2 && i >= D/2) {
                    break;
                }

                half2 VKQ_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int l = 0; l < VKQ_ratio; ++l) {
                    VKQ_add += KQ2[l*(ncols*D_padded/2) + j*(D_padded/2) + i];
                }
                VKQ2[j*(D_padded/2) + i] = VKQ_scale*VKQ2[j*(D_padded/2) + i] + VKQ_add;
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j_VKQ = j0 + threadIdx.y;
        if (ic0 + j_VKQ >= ne01) {
            return;
        }
        const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;

        float KQ_rowsum_j;
        if (std::is_same<KQ_acc_t, float>::value) {
            KQ_rowsum_j = KQ_rowsum_f[j0/nwarps];
        } else {
            KQ_rowsum_j = __low2float(KQ_rowsum_h2[j0/nwarps]) + __high2float(KQ_rowsum_h2[j0/nwarps]);
        }

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D && i >= D) {
                break;
            }
            float dst_val = VKQ[j_VKQ*D_padded + i];
            if (parallel_blocks == 1) {
                dst_val /= KQ_rowsum_j;
            }
            dst[j_dst*gridDim.y*D + blockIdx.y*D + i] = dst_val;
        }

        if (parallel_blocks == 1 || threadIdx.x != 0) {
            continue;
        }

        float2 dst_meta_val;
        if (std::is_same<KQ_acc_t, float>::value) {
            dst_meta_val.x = KQ_max_f[j0/nwarps];
        } else {
            dst_meta_val.x = __low2float(KQ_max_h2[j0/nwarps]);
        }
        dst_meta_val.y = KQ_rowsum_j;
        dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = dst_meta_val;
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_MMA_AVAILABLE
}

template<int D, int parallel_blocks> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst) {
#if FP16_AVAILABLE
    VKQ_parts += parallel_blocks*D * gridDim.y*blockIdx.x;
    VKQ_meta  += parallel_blocks   * gridDim.y*blockIdx.x;
    dst       +=                 D * gridDim.y*blockIdx.x;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float2 meta[parallel_blocks];
    if (tid < 2*parallel_blocks) {
        ((float *) meta)[threadIdx.x] = ((const float *)VKQ_meta) [blockIdx.y*(2*parallel_blocks) + tid];
    }

    __syncthreads();

    float kqmax = meta[0].x;
#pragma unroll
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
#pragma unroll
    for (int l = 0; l < parallel_blocks; ++l) {
        const float diff = meta[l].x - kqmax;
        const float KQ_max_scale = expf(diff);
        const uint32_t ftz_mask = 0xFFFFFFFF * (diff > SOFTMAX_FTZ_THRESHOLD);
        *((uint32_t *) &KQ_max_scale) &= ftz_mask;

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*gridDim.y*D + blockIdx.y*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[blockIdx.y*D + tid] = VKQ_numerator / VKQ_denominator;
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

constexpr int get_max_power_of_2(int x) {
    return x % 2 == 0 ? 2*get_max_power_of_2(x/2) : 1;
}

static_assert(get_max_power_of_2(1) == 1, "Test failed.");
static_assert(get_max_power_of_2(2) == 2, "Test failed.");
static_assert(get_max_power_of_2(4) == 4, "Test failed.");
static_assert(get_max_power_of_2(6) == 2, "Test failed.");

// Number of VKQ rows calculated in parallel:
constexpr int get_VKQ_stride(int D, int nwarps, int frag_m) {
    return (get_max_power_of_2(D/frag_m) < nwarps ? get_max_power_of_2(D/frag_m) : nwarps)*frag_m;
}

static_assert(get_VKQ_stride(128, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride(128, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride(128, 4, 32) == 128, "Test failed.");
static_assert(get_VKQ_stride( 64, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride( 64, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 64, 4, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 80, 1, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 2, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 4, 16) ==  16, "Test failed.");

template <int D, int parallel_blocks> void launch_fattn_vec_f16(
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V, ggml_tensor * KQV, const ggml_tensor * mask,
        ggml_cuda_pool & pool, cudaStream_t main_stream
) {
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }

    constexpr int  nwarps = (D + WARP_SIZE - 1) / WARP_SIZE;
    const     dim3 block_dim(WARP_SIZE, nwarps, 1);
    const     dim3 blocks_num(parallel_blocks*Q->ne[1], Q->ne[2], Q->ne[3]);
    const     int  shmem = 0;

    float scale;
    memcpy(&scale, KQV->op_params, sizeof(float));

    flash_attn_vec_ext_f16<D, parallel_blocks>
        <<<blocks_num, block_dim, shmem, main_stream>>> (
                (const char *) Q->data,
                (const char *) K->data,
                (const char *) V->data,
                mask ? ((const char *) mask->data) : nullptr,
                parallel_blocks == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                scale,
                Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                Q->nb[1], Q->nb[2], Q->nb[3],
                K->nb[1], K->nb[2], K->nb[3],
                KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                );
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks == 1) {
        return;
    }

    const dim3 block_dim_combine(D, 1, 1);
    const dim3 blocks_num_combine(Q->ne[1], blocks_num.y, blocks_num.z);
    const int  shmem_combine = 0;

    flash_attn_combine_results<D, parallel_blocks>
        <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
        (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
    CUDA_CHECK(cudaGetLastError());
}

template <int D, int cols_per_block, int nwarps, int parallel_blocks, typename KQ_acc_t> void launch_fattn_f16_impl(
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V, ggml_tensor * KQV, const ggml_tensor * mask,
        ggml_cuda_pool & pool, cudaStream_t main_stream
) {
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }

    constexpr int  frag_m = (cols_per_block) == 8 && (D) % 32 == 0 ? 32 : 16;
    const     dim3 block_dim(WARP_SIZE, nwarps, 1);
    const     dim3 blocks_num(parallel_blocks*(Q->ne[1] + cols_per_block - 1) / cols_per_block, Q->ne[2], Q->ne[3]);
    const     int  shmem = 0;

    float scale;
    memcpy(&scale, KQV->op_params, sizeof(float));

    flash_attn_ext_f16<D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t>
        <<<blocks_num, block_dim, shmem, main_stream>>> (
                (const char *) Q->data,
                (const char *) K->data,
                (const char *) V->data,
                mask ? ((const char *) mask->data) : nullptr,
                (parallel_blocks) == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                scale,
                Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                Q->nb[1], Q->nb[2], Q->nb[3],
                K->nb[1], K->nb[2], K->nb[3],
                KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                );
    CUDA_CHECK(cudaGetLastError());

    if ((parallel_blocks) == 1) {
        return;
    }

    const dim3 block_dim_combine(D, 1, 1);
    const dim3 blocks_num_combine(Q->ne[1], blocks_num.y, blocks_num.z);
    const int  shmem_combine = 0;

    flash_attn_combine_results<D, parallel_blocks>
        <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
        (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
    CUDA_CHECK(cudaGetLastError());
}

template <int D, int cols_per_block, int nwarps, typename KQ_acc_t> void launch_fattn_f16(
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V, ggml_tensor * KQV, const ggml_tensor * mask,
        const int nsm, ggml_cuda_pool & pool, cudaStream_t main_stream
) {
    const int blocks_num_pb1 = ((Q->ne[1] + cols_per_block - 1) / cols_per_block)*Q->ne[2]*Q->ne[3];

    if (4*blocks_num_pb1 < 2*nsm) {
        launch_fattn_f16_impl<D, cols_per_block, nwarps, 4, KQ_acc_t>(Q, K, V, KQV, mask, pool, main_stream);
        return;
    }
    if (2*blocks_num_pb1 < 2*nsm) {
        launch_fattn_f16_impl<D, cols_per_block, nwarps, 2, KQ_acc_t>(Q, K, V, KQV, mask, pool, main_stream);
        return;
    }
    launch_fattn_f16_impl<D, cols_per_block, nwarps, 1, KQ_acc_t>(Q, K, V, KQV, mask, pool, main_stream);
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_tensor * mask = dst->src[3];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(K->type == GGML_TYPE_F16);
    GGML_ASSERT(V->type == GGML_TYPE_F16);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!mask || mask->ne[1] >= GGML_PAD(Q->ne[1], 16) &&
                                "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

    ggml_cuda_set_device(ctx.device);

    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    const int32_t precision = KQV->op_params[1];

    if (precision != GGML_PREC_DEFAULT) {
        if (Q->ne[1] <= 32 || Q->ne[0] > 128) {
            constexpr int cols_per_block = 16;
            constexpr int nwarps         =  4;
            switch (Q->ne[0]) {
                case 64:
                    launch_fattn_f16< 64, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 80:
                    launch_fattn_f16< 80, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 96:
                    launch_fattn_f16< 96, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 112:
                    launch_fattn_f16<112, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 128:
                    launch_fattn_f16<128, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 256:
                    launch_fattn_f16<256, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                default:
                    GGML_ASSERT(false);
                    break;
            }
        } else {
            constexpr int cols_per_block = 32;
            constexpr int nwarps         =  4;
            switch (Q->ne[0]) {
                case 64:
                    launch_fattn_f16< 64, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 80:
                    launch_fattn_f16< 80, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 96:
                    launch_fattn_f16< 96, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 112:
                    launch_fattn_f16<112, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 128:
                    launch_fattn_f16<128, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                // case 256:
                //     launch_fattn_f16<256, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                //     break;
                default:
                    GGML_ASSERT(false);
                    break;
            }
        }
        return;
    }

    if (Q->ne[1] == 1 && Q->ne[0] % (2*WARP_SIZE) == 0) {
        constexpr int parallel_blocks = 4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_vec_f16< 64, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_vec_f16<128, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            case 256:
                launch_fattn_vec_f16<256, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    if (Q->ne[1] <= 8 && Q->ne[0] % WARP_SIZE == 0) {
        constexpr int cols_per_block = 8;
        constexpr int nwarps         = 4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_f16< 64, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 96:
                launch_fattn_f16< 96, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_f16<128, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 256:
                launch_fattn_f16<256, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 16;
        constexpr int nwarps         =  4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_f16< 64, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 80:
                launch_fattn_f16< 80, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 96:
                launch_fattn_f16< 96, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 112:
                launch_fattn_f16<112, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_f16<128, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 256:
                launch_fattn_f16<256, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    constexpr int cols_per_block = 32;
    constexpr int nwarps         =  4;
    switch (Q->ne[0]) {
        case 64:
            launch_fattn_f16< 64, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 80:
            launch_fattn_f16< 80, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 96:
            launch_fattn_f16< 96, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 112:
            launch_fattn_f16<112, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 128:
            launch_fattn_f16<128, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 256:
            launch_fattn_f16<256, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
    return;
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12/*, size_t s13*/) {

    const int i00 = (blockIdx.x*blockDim.x + threadIdx.x)*2;
    const int i10 = blockDim.y*blockIdx.y + threadIdx.y;
    const int i11 = (blockIdx.z*blockDim.z + threadIdx.z)/ne12;
    const int i12 = (blockIdx.z*blockDim.z + threadIdx.z)%ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0]        = v.x;
    dst_row[iybs + iqs + y_offset] = v.y;
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12/*, size_t s13*/) {

    const int i00 = blockIdx.x*blockDim.x + threadIdx.x;
    const int i10 = blockDim.y*blockIdx.y + threadIdx.y;
    const int i11 = (blockIdx.z*blockDim.z + threadIdx.z)/ne12;
    const int i12 = (blockIdx.z*blockDim.z + threadIdx.z)%ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = src0_row[i00];
}

template<int qk, int qr, dequantize_kernel_t dq>
static void get_rows_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                            const void * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(block_num_x, ne10, ne11*ne12);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
            src0_dd, src1_dd, dst_dd,
            ne00, /*ne01, ne02, ne03,*/
            /*ne10, ne11,*/ ne12, /*ne13,*/
            /* s0,*/ s1, s2, s3,
            /* nb00,*/ nb01, nb02, nb03,
            s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

template<typename src0_t>
static void get_rows_cuda_float(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                const src0_t * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ne10, ne11*ne12);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    k_get_rows_float<<<block_nums, block_dims, 0, stream>>>(
            src0_dd, src1_dd, dst_dd,
            ne00, /*ne01, ne02, ne03,*/
            /*ne10, ne11,*/ ne12, /*ne13,*/
            /* s0,*/ s1, s2, s3,
            /* nb00,*/ nb01, nb02, nb03,
            s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();


    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) src1_d;

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float(src0, src1, dst, (const half *)src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda<QK4_0, QR4_0, dequantize_q4_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda<QK4_1, QR4_1, dequantize_q4_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda<QK5_0, QR5_0, dequantize_q5_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda<QK5_1, QR5_1, dequantize_q5_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda<QK8_0, QR8_0, dequantize_q8_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        default:
            // TODO: k-quants
            fprintf(stderr, "%s: unsupported type: %s\n", __func__, ggml_type_name(src0->type));
            GGML_ASSERT(false);
            break;
    }
}

template <typename T>
static  __global__ void im2col_kernel(
        const float * x, T * dst, int64_t batch_offset,
        int64_t offset_delta, int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH, int64_t pelements, int64_t CHW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= pelements) {
        return;
    }

    const int64_t  ksize = OW * (KH > 1 ? KW : 1);
    const int64_t  kx = i / ksize;
    const int64_t  kd = kx * ksize;
    const int64_t  ky = (i - kd) / OW;
    const int64_t  ix = i % OW;

    const int64_t  oh = blockIdx.y;
    const int64_t  batch = blockIdx.z / IC;
    const int64_t  ic = blockIdx.z % IC;

    const int64_t iiw = ix * s0 + kx * d0 - p0;
    const int64_t iih = oh * s1 + ky * d1 - p1;

    const int64_t offset_dst =
        ((batch * OH + oh) * OW + ix) * CHW +
        (ic * (KW * KH) + ky * KW + kx);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        dst[offset_dst] = 0.0f;
    } else {
        const int64_t offset_src = ic * offset_delta + batch * batch_offset;
        dst[offset_dst] = x[offset_src + iih * IW + iiw];
    }
}

template <typename T>
static void im2col_cuda(const float * x, T* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {
    const int parallel_elements = OW * KW * KH;
    const int num_blocks = (parallel_elements + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    dim3 block_nums(num_blocks, OH, batch * IC);
    im2col_kernel<<<block_nums, CUDA_IM2COL_BLOCK_SIZE, 0, stream>>>(x, dst, batch_offset, offset_delta, IC, IW, IH, OH, OW, KW, KH, parallel_elements, (IC * KH * KW), s0, s1, p0, p1, d0, d1);
}

static void im2col_cuda_f16(const float * x, half * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<half>(x, dst, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, offset_delta, s0, s1, p0, p1, d0, d1, stream);
}

static void im2col_cuda_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, offset_delta, s0, s1, p0, p1, d0, d1, stream);
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const size_t delta_offset = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t batch = src1->ne[3];
    const size_t batch_offset = src1->nb[3] / 4; // nb is byte offset, src is type float32

    if(dst->type == GGML_TYPE_F16) {
        im2col_cuda_f16(src1_d, (half *) dst_d, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_cuda_f32(src1_d, (float *) dst_d, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
    }
}

typedef void (*allocate_tiles_cuda_t)(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc);
typedef void (*load_tiles_cuda_t)(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row);
typedef float (*vec_dot_q_mul_mat_cuda_t)(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ms, const int & i, const int & j, const int & k);
typedef void (*dot_kernel_k_t)(const void * __restrict__ vx, const int ib, const int iqs, const float * __restrict__ y, float & v);

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);
    GGML_UNUSED(x_sc);

    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI4_0) + mmq_y/QI4_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);
    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (const block_q4_0 *) vx;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (const float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_0) % WARP_SIZE];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int   tile_x_qs[mmq_y * (WARP_SIZE) +     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI4_1) + mmq_y/QI4_1];

    *x_ql = tile_x_qs;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (const block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_1) % WARP_SIZE];
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int  tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI5_0) + mmq_y/QI5_0];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (const block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_0) % WARP_SIZE];
    }

    return vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}


template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_1) + mmq_y/QI5_1];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset < nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (const block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_1) % WARP_SIZE];
    }

    return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q8_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI8_0) + mmq_y/QI8_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (const block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q2_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI2_K) + mmq_y/QI2_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (const block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI2_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const int kbx = k / QI2_K;
    const int ky  = (k % QI2_K) * QR2_K;
    const float * y_df = (const float *) y_ds;

    int v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    const int kqsx = i * (WARP_SIZE + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    const int shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    for (int l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    const uint8_t * scales = ((const uint8_t *) &x_sc[i * (WARP_SIZE/4) + i/4 + kbx*4]) + ky/4;

    const int index_y = j * WARP_SIZE + (QR2_K*k) % WARP_SIZE;
    return vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q3_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI3_K) + mmq_y/QI3_K];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE/2)     + mmq_y/2];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (const block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const int kbx  = k / QI3_K;
    const int ky  = (k % QI3_K) * QR3_K;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

    int v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        const int kqsx = i * (WARP_SIZE + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        const int shift = 2 * ((ky % 32) / 8);
        const int vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        const int vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        const int vlh = (vh << 2) & 0x04040404;

        v[l] = __vsubss4(vll, vlh);
    }

    const int index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    return vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI4_K) + mmq_y/QI4_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (const block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#else
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    const int index_y = j * WARP_SIZE + (QR4_K*k) % WARP_SIZE;
    return vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_K) + mmq_y/QI5_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (const block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    const int index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    const int index_y = j * WARP_SIZE             + (QR5_K*k) % WARP_SIZE;
    return vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q6_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI6_K) + mmq_y/QI6_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (const block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/8]);

    const int index_x = i * (QR6_K*WARP_SIZE + 1) +  QR6_K*k;
    const int index_y = j * WARP_SIZE             + (QR6_K*k) % WARP_SIZE;
    return vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}

#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_0_AMPERE 4
#define  MMQ_Y_Q4_0_AMPERE 32
#define NWARPS_Q4_0_AMPERE 4
#else
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#endif
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

template <int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x, int mmq_y, int nwarps,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void mul_mat_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE / qi;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = blockIdx.x*mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = blockIdx.y*mmq_x;
    const int & col_y_0 = col_dst_0;

    int   * tile_x_ql = nullptr;
    half2 * tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    __shared__ int    tile_y_qs[mmq_x * WARP_SIZE];
    __shared__ half2  tile_y_ds[mmq_x * WARP_SIZE/QI8_1];

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0*blocks_per_row_x + ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   threadIdx.y, nrows_x-row_x_0-1, threadIdx.x, blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr; ++ir) {
            const int kqs = ir*WARP_SIZE + threadIdx.x;
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); // to prevent out-of-bounds memory accesses

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                const int index_y = (threadIdx.y + i) * WARP_SIZE + kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE/QI8_1)) % mmq_x;
                const int kby = threadIdx.x % (WARP_SIZE/QI8_1);
                const int col_y_eff = min(col_y_0 + ids, ncols_y-1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const half2 * dsi_src = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(WARP_SIZE/QI8_1) + kby].ds;
                half2       * dsi_dst = &tile_y_ds[ids * (WARP_SIZE/QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = __low2float(*dsi_src);
                }
            }

            __syncthreads();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE) {
                        sum[i/WARP_SIZE][j/nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                            threadIdx.x + i, threadIdx.y + j, k);
                    }
                }
            }

            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + threadIdx.y;

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + threadIdx.x + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q4_0_RDNA2;
    const int mmq_y  =  MMQ_Y_Q4_0_RDNA2;
    const int nwarps = NWARPS_Q4_0_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q4_0_RDNA1;
    const int mmq_y  =  MMQ_Y_Q4_0_RDNA1;
    const int nwarps = NWARPS_Q4_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q4_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    const int nwarps = NWARPS_Q4_0_AMPERE;

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q4_0_PASCAL;
    const int mmq_y  =  MMQ_Y_Q4_0_PASCAL;
    const int nwarps = NWARPS_Q4_0_PASCAL;

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q4_0_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_1_AMPERE 4
#define  MMQ_Y_Q4_1_AMPERE 32
#define NWARPS_Q4_1_AMPERE 4
#else
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#endif
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_1_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_1_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q4_1_RDNA2;
    const int mmq_y  =  MMQ_Y_Q4_1_RDNA2;
    const int nwarps = NWARPS_Q4_1_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q4_1_RDNA1;
    const int mmq_y  =  MMQ_Y_Q4_1_RDNA1;
    const int nwarps = NWARPS_Q4_1_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q4_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    const int nwarps = NWARPS_Q4_1_AMPERE;

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q4_1_PASCAL;
    const int mmq_y  =  MMQ_Y_Q4_1_PASCAL;
    const int nwarps = NWARPS_Q4_1_PASCAL;

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q4_1_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_0_AMPERE 4
#define  MMQ_Y_Q5_0_AMPERE 32
#define NWARPS_Q5_0_AMPERE 4
#else
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#endif
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q5_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q5_0_RDNA2;
    const int mmq_y  =  MMQ_Y_Q5_0_RDNA2;
    const int nwarps = NWARPS_Q5_0_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q5_0_RDNA1;
    const int mmq_y  =  MMQ_Y_Q5_0_RDNA1;
    const int nwarps = NWARPS_Q5_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q5_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    const int nwarps = NWARPS_Q5_0_AMPERE;

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q5_0_PASCAL;
    const int mmq_y  =  MMQ_Y_Q5_0_PASCAL;
    const int nwarps = NWARPS_Q5_0_PASCAL;

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q5_0_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_1_AMPERE 4
#define  MMQ_Y_Q5_1_AMPERE 32
#define NWARPS_Q5_1_AMPERE 4
#else
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#endif
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q5_1_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q5_1_RDNA2;
    const int mmq_y  =  MMQ_Y_Q5_1_RDNA2;
    const int nwarps = NWARPS_Q5_1_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q5_1_RDNA1;
    const int mmq_y  =  MMQ_Y_Q5_1_RDNA1;
    const int nwarps = NWARPS_Q5_1_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q5_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    const int nwarps = NWARPS_Q5_1_AMPERE;

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q5_1_PASCAL;
    const int mmq_y  =  MMQ_Y_Q5_1_PASCAL;
    const int nwarps = NWARPS_Q5_1_PASCAL;

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q5_1_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q8_0_AMPERE 4
#define  MMQ_Y_Q8_0_AMPERE 32
#define NWARPS_Q8_0_AMPERE 4
#else
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#endif
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q8_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q8_0_RDNA2;
    const int mmq_y  =  MMQ_Y_Q8_0_RDNA2;
    const int nwarps = NWARPS_Q8_0_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q8_0_RDNA1;
    const int mmq_y  =  MMQ_Y_Q8_0_RDNA1;
    const int nwarps = NWARPS_Q8_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q8_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    const int nwarps = NWARPS_Q8_0_AMPERE;

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q8_0_PASCAL;
    const int mmq_y  =  MMQ_Y_Q8_0_PASCAL;
    const int nwarps = NWARPS_Q8_0_PASCAL;

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q8_0_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q2_K_AMPERE 4
#define  MMQ_Y_Q2_K_AMPERE 32
#define NWARPS_Q2_K_AMPERE 4
#else
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#endif
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q2_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q2_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q2_K_RDNA2;
    const int nwarps = NWARPS_Q2_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q2_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q2_K_RDNA1;
    const int nwarps = NWARPS_Q2_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q2_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    const int nwarps = NWARPS_Q2_K_AMPERE;

    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q2_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q2_K_PASCAL;
    const int nwarps = NWARPS_Q2_K_PASCAL;

    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q2_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q3_K_AMPERE 4
#define  MMQ_Y_Q3_K_AMPERE 32
#define NWARPS_Q3_K_AMPERE 4
#else
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#endif
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q3_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q3_K_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q3_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q3_K_RDNA2;
    const int nwarps = NWARPS_Q3_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q3_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q3_K_RDNA1;
    const int nwarps = NWARPS_Q3_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q3_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    const int nwarps = NWARPS_Q3_K_AMPERE;

    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q3_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q3_K_PASCAL;
    const int nwarps = NWARPS_Q3_K_PASCAL;

    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q3_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_K_AMPERE 4
#define  MMQ_Y_Q4_K_AMPERE 32
#define NWARPS_Q4_K_AMPERE 4
#else
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_K_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q4_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q4_K_RDNA2;
    const int nwarps = NWARPS_Q4_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q4_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q4_K_RDNA1;
    const int nwarps = NWARPS_Q4_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q4_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    const int nwarps = NWARPS_Q4_K_AMPERE;

    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q4_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q4_K_PASCAL;
    const int nwarps = NWARPS_Q4_K_PASCAL;

    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q4_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_K_AMPERE 4
#define  MMQ_Y_Q5_K_AMPERE 32
#define NWARPS_Q5_K_AMPERE 4
#else
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#endif
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q5_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q5_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q5_K_RDNA2;
    const int nwarps = NWARPS_Q5_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q5_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q5_K_RDNA1;
    const int nwarps = NWARPS_Q5_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q5_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    const int nwarps = NWARPS_Q5_K_AMPERE;

    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q5_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q5_K_PASCAL;
    const int nwarps = NWARPS_Q5_K_PASCAL;

    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q5_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q6_K_AMPERE 4
#define  MMQ_Y_Q6_K_AMPERE 32
#define NWARPS_Q6_K_AMPERE 4
#else
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#endif
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q6_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q6_K_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q6_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q6_K_RDNA2;
    const int nwarps = NWARPS_Q6_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q6_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q6_K_RDNA1;
    const int nwarps = NWARPS_Q6_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q6_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    const int nwarps = NWARPS_Q6_K_AMPERE;

    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q6_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q6_K_PASCAL;
    const int nwarps = NWARPS_Q6_K_PASCAL;

    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q6_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

static void ggml_mul_mat_q4_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q4_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q8_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q8_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q8_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q2_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q2_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q2_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q3_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

#if QK_K == 256

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q3_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q3_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
#endif
}

static void ggml_mul_mat_q4_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q6_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q6_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q6_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}

bool ggml_cuda_supports_mmq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);

template <int ncols_y, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__((ncols_y <= 4 ? 4 : 2)*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int nwarps              = 1;
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int nwarps              = ncols_y <= 4 ? 4 : 2;
    constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    &x[kbx + (row0 + i)*blocks_per_row_x], &y[j*blocks_per_col_y + kby], kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block) {
            dst[j*nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
static void mul_mat_vec_q_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % qk == 0);
    GGML_ASSERT(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    int64_t nwarps = 1;
    int64_t rows_per_cuda_block = 1;

    if (ggml_cuda_info().devices[id].cc < CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        switch(ncols_y) {
            case 1:
                nwarps = 4;
                rows_per_cuda_block = 1;
                break;
            case 2:
            case 3:
            case 4:
                nwarps = 4;
                rows_per_cuda_block = 2;
                break;
            case 5:
            case 6:
            case 7:
            case 8:
                nwarps = 2;
                rows_per_cuda_block = 2;
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
    }
    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (ncols_y) {
        case 1:
            mul_mat_vec_q<1, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 2:
            mul_mat_vec_q<2, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 3:
            mul_mat_vec_q<3, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 4:
            mul_mat_vec_q<4, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 5:
            mul_mat_vec_q<5, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 6:
            mul_mat_vec_q<6, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 7:
            mul_mat_vec_q<7, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 8:
            mul_mat_vec_q<8, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
}

static void mul_mat_vec_q4_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q4_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q5_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q5_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q8_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q2_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q3_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q4_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q5_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q6_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq2_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq2_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq2_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq3_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq1_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq1_m_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI1_S, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq4_nl_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq4_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq3_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_iq2_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_iq2_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_iq2_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_iq3_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_iq1_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_iq1_m_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_iq4_nl_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_iq4_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_iq3_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}

template <int block_size>
static __global__ void norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = (x[row*ncols + col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    int start = blockIdx.x * group_size;
    int end = start + group_size;

    start += threadIdx.x;

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    float variance = tmp / group_size;
    float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size>
static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * x[row*ncols + col];
    }
}

static void norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

static void group_norm_f32_cuda(const float * x, float * dst, const int num_groups, const int group_size, const int ne_elements, cudaStream_t stream) {
    static const float eps = 1e-6f;
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];
    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    rms_norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
}

static __global__ void pad_f32(const float * x, float * dst, const int ne0, const int ne00, const int ne01, const int ne02, const int ne03) {
    // blockIdx.z: idx of ne2*ne3, aka ne02*ne03
    // blockIdx.y: idx of ne1
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;
    if (nidx < ne00 && blockIdx.y < ne01 && blockIdx.z < ne02*ne03) {
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        dst[offset_dst] = 0.0f;
    }
}

static void pad_f32_cuda(const float * x, float * dst,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int ne0, const int ne1, const int ne2, const int ne3, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2*ne3);
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(x, dst, ne0, ne00, ne01, ne02, ne03);
}

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    pad_f32_cuda(src0_d, dst_d,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

template <typename Ti, typename To>
static  __global__ void pool2d_nchw_kernel(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int I_HW = ih * iw;
    const int O_HW = oh * ow;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / ow;
    const int cur_ow = idx % O_HW % ow;
    const Ti* i_ptr = src + nc * I_HW;
    To* o_ptr = dst + nc * O_HW;
    const int start_h = cur_oh * sh - ph;
    const int bh = max(0, start_h);
    const int eh = min(ih, start_h + kh);
    const int start_w = cur_ow * sw - pw;
    const int bw = max(0, start_w);
    const int ew = min(iw, start_w + kw);
    const To scale = 1. / (kh * kw);
    To res = 0;

    switch (op) {
        case GGML_OP_POOL_AVG: res = 0; break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: assert(false);
    }

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
#if __CUDA_ARCH__ >= 350
            Ti cur = __ldg(i_ptr + i * iw + j);
#else
            Ti cur = i_ptr[i * iw + j];
#endif
            switch (op) {
                case GGML_OP_POOL_AVG: res += cur * scale; break;
                case GGML_OP_POOL_MAX: res = max(res, (To)cur); break;
                default: assert(false);
            }
        }
    }
    o_ptr[cur_oh * ow + cur_ow] = res;
}

static void pool2d_nchw_kernel_f32_f32_cuda(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {

    const int num_blocks = (parallel_elements + CUDA_POOL2D_BLOCK_SIZE - 1) / CUDA_POOL2D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool2d_nchw_kernel<<<block_nums, CUDA_POOL2D_BLOCK_SIZE, 0, stream>>>(ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    const int64_t IH = src0->ne[1];
    const int64_t IW = src0->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const int parallel_elements = N * OC * OH * OW;

    pool2d_nchw_kernel_f32_f32_cuda(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0, parallel_elements, src0_d, dst_d, op, stream);
}

static __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int64_t kx, const int64_t kx_padded) {
    const int64_t ix = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int64_t iy = (int64_t)blockDim.y*blockIdx.y + threadIdx.y;

    const int64_t i_padded = (int64_t)iy*kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib = i_padded / QK8_1; // block index
    const int64_t iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

void quantize_row_q8_1_cuda(const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream) {
    const int64_t block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ky, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}


struct rope_corr_dims {
    float v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static __device__ void rope_yarn(
    float theta_extrap, float freq_scale, rope_corr_dims corr_dims, int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta
) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

// rope == RoPE == rotary positional embedding
template<typename T, bool has_pos>
static __global__ void rope(
    const T * x, T * dst, int ncols, const int32_t * pos, float freq_scale, int p_delta_rows, float freq_base,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims
) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
    const int i2 = row/p_delta_rows;

    const int p = has_pos ? pos[i2] : 0;
    const float theta_base = p*powf(freq_base, -float(col)/ncols);

    float cos_theta, sin_theta;
    rope_yarn(theta_base, freq_scale, corr_dims, col, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

template<typename T, bool has_pos>
static __global__ void rope_neox(
    const T * x, T * dst, int ncols, int n_dims, const int32_t * pos, float freq_scale, int p_delta_rows,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims, float theta_scale, float inv_ndims
) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int ib = col / n_dims;
    const int ic = col % n_dims;

    if (ib > 0) {
        const int i = row*ncols + ib*n_dims + ic;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int i  = row*ncols + ib*n_dims + ic/2;
    const int i2 = row/p_delta_rows;

    float cur_rot = inv_ndims * ic - ib;

    const int p = has_pos ? pos[i2] : 0;
    const float theta_base = p*freq_scale*powf(theta_scale, col/2.0f);

    float cos_theta, sin_theta;
    rope_yarn(theta_base, freq_scale, corr_dims, cur_rot, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + n_dims/2];

    dst[i + 0]        = x0*cos_theta - x1*sin_theta;
    dst[i + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

static __global__ void rope_glm_f32(
    const float * x, float * dst, int ncols, const int32_t * pos, float freq_scale, int p_delta_rows, float freq_base,
    int n_ctx
) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    const int half_n_dims = ncols/4;

    if (col >= half_n_dims) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;
    const int i2 = row/p_delta_rows;

    const float col_theta_scale = powf(freq_base, -2.0f*col/ncols);
     // FIXME: this is likely wrong
    const int p = pos != nullptr ? pos[i2] : 0;

    const float theta = min(p, n_ctx - 2)*freq_scale*col_theta_scale;
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + half_n_dims];

    dst[i + 0]           = x0*cos_theta - x1*sin_theta;
    dst[i + half_n_dims] = x0*sin_theta + x1*cos_theta;

    const float block_theta = ((float)max(p - n_ctx - 2, 0))*col_theta_scale;
    const float sin_block_theta = sinf(block_theta);
    const float cos_block_theta = cosf(block_theta);

    const float x2 = x[i + half_n_dims * 2];
    const float x3 = x[i + half_n_dims * 3];

    dst[i + half_n_dims * 2] = x2*cos_block_theta - x3*sin_block_theta;
    dst[i + half_n_dims * 3] = x2*sin_block_theta + x3*cos_block_theta;
}


template<typename T>
static void rope_cuda(
    const T * x, T * dst, int ncols, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream
) {
    GGML_ASSERT(ncols % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nrows, num_blocks_x, 1);
    if (pos == nullptr) {
        rope<T, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ncols, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims
        );
    } else {
        rope<T, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ncols, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims
        );
    }
}

template<typename T>
static void rope_neox_cuda(
    const T * x, T * dst, int ncols, int n_dims, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream
) {
    GGML_ASSERT(ncols % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nrows, num_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    const float inv_ndims = -1.0f / n_dims;

    if (pos == nullptr) {
        rope_neox<T, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ncols, n_dims, pos, freq_scale, p_delta_rows, ext_factor, attn_factor, corr_dims,
            theta_scale, inv_ndims
        );
    } else {
        rope_neox<T, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ncols, n_dims, pos, freq_scale, p_delta_rows, ext_factor, attn_factor, corr_dims,
            theta_scale, inv_ndims
        );
    }
}

static void rope_glm_f32_cuda(
    const float * x, float * dst, int ncols, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, int n_ctx, cudaStream_t stream
) {
    GGML_ASSERT(ncols % 4 == 0);
    const dim3 block_dims(CUDA_ROPE_BLOCK_SIZE/4, 1, 1);
    const int num_blocks_x = (ncols + CUDA_ROPE_BLOCK_SIZE - 1) / CUDA_ROPE_BLOCK_SIZE;
    const dim3 block_nums(num_blocks_x, nrows, 1);
    rope_glm_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, pos, freq_scale, p_delta_rows, freq_base, n_ctx);
}

static void rope_cuda_f16(
    const half * x, half * dst, int ncols, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream) {

    rope_cuda<half>(x, dst, ncols, nrows, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, stream);
}

static void rope_cuda_f32(
    const float * x, float * dst, int ncols, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream) {

    rope_cuda<float>(x, dst, ncols, nrows, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, stream);
}

static void rope_neox_cuda_f16(
    const half * x, half * dst, int ncols, int n_dims, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream) {

    rope_neox_cuda<half>(x, dst, ncols, n_dims, nrows, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, stream);
}

static void rope_neox_cuda_f32(
    const float * x, float * dst, int ncols, int n_dims, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream
) {

    rope_neox_cuda<float>(x, dst, ncols, n_dims, nrows, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, stream);
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t nrows = ggml_nrows(src0);

    //const int n_past      = ((int32_t *) dst->op_params)[0];
    const int n_dims      = ((int32_t *) dst->op_params)[1];
    const int mode        = ((int32_t *) dst->op_params)[2];
    const int n_ctx       = ((int32_t *) dst->op_params)[3];
    const int n_orig_ctx  = ((int32_t *) dst->op_params)[4];

    // RoPE alteration for extended context
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    const int32_t * pos = nullptr;
    if ((mode & 1) == 0) {
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        GGML_ASSERT(src1->ne[0] == ne2);
        pos = (const int32_t *) src1_d;
    }

    const bool is_neox = mode & 2;
    const bool is_glm  = mode & 4;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_glm) {
        GGML_ASSERT(false);
        rope_glm_f32_cuda(src0_d, dst_d, ne00, nrows, pos, freq_scale, ne01, freq_base, n_ctx, stream);
    } else if (is_neox) {
        if (src0->type == GGML_TYPE_F32) {
            rope_neox_cuda_f32(
                (const float *)src0_d, (float *)dst_d, ne00, n_dims, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, stream
            );
        } else if (src0->type == GGML_TYPE_F16) {
            rope_neox_cuda_f16(
                (const half *)src0_d, (half *)dst_d, ne00, n_dims, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, stream
            );
        } else {
            GGML_ASSERT(false);
        }
    } else {
        if (src0->type == GGML_TYPE_F32) {
            rope_cuda_f32(
                (const float *)src0_d, (float *)dst_d, ne00, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, stream
            );
        } else if (src0->type == GGML_TYPE_F16) {
            rope_cuda_f16(
                (const half *)src0_d, (half *)dst_d, ne00, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, stream
            );
        } else {
            GGML_ASSERT(false);
        }
    }
}

static __global__ void scale_f32(const float * x, float * dst, const float scale, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i];
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, k);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    scale_f32_cuda(src0_d, dst_d, scale, ggml_nelements(src0), stream);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float) val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

template <bool vals_smem, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(const float * x, const T * mask, const T * pos, float * dst, const int ncols_par, const int nrows_y, const float scale, const float max_bias, const float m0, const float m1, uint32_t n_head_log2) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;
    const int rowy = rowx % nrows_y; // broadcast the mask in the row dimension

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    float slope = 0.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        const int h = rowx/nrows_y; // head index

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = powf(base, exp);
    }

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = vals_smem ? buf_iw + WARP_SIZE : dst + (int64_t)rowx*ncols;

    float max_val = -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        const int64_t iy = (int64_t)rowy*ncols + col;

        const float val = x[ix]*scale + (mask ? t2f32(mask[iy]) : 0.0f) + (pos ? slope*t2f32(pos[col]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        const int64_t idst = (int64_t)rowx*ncols + col;
        dst[idst] = vals[col] * inv_sum;
    }
}

template<typename T>
static void soft_max_f32_cuda(const float * x, const T * mask, const T * pos, float * dst, const int ncols_x, const int nrows_x, const int nrows_y, const float scale, const float max_bias, cudaStream_t stream) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(nrows_x, 1, 1);
    const size_t shmem = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

    const uint32_t n_head_kv   = nrows_x/nrows_y;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head_kv));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    if (shmem < ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        switch (ncols_x) {
            case 32:
                soft_max_f32<true, 32, 32><<<block_nums, block_dims, shmem, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 64:
                soft_max_f32<true, 64, 64><<<block_nums, block_dims, shmem, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 128:
                soft_max_f32<true, 128, 128><<<block_nums, block_dims, shmem, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 256:
                soft_max_f32<true, 256, 256><<<block_nums, block_dims, shmem, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 512:
                soft_max_f32<true, 512, 512><<<block_nums, block_dims, shmem, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 1024:
                soft_max_f32<true, 1024, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 2048:
                soft_max_f32<true, 2048, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            case 4096:
                soft_max_f32<true, 4096, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
            default:
                soft_max_f32<true, 0, 0><<<block_nums, block_dims, shmem, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
                break;
        }
    } else {
        const size_t shmem_low = WARP_SIZE*sizeof(float);
        soft_max_f32<false, 0, 0><<<block_nums, block_dims, shmem_low, stream>>>(x, mask, pos, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
    }
}

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const void  * src1_d = src1 ? (const void *)src1->data : nullptr;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional
    GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F16 || src2->type == GGML_TYPE_F32); // src2 contains positions and it is optional

    const int64_t ne00    = src0->ne[0];
    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    // positions tensor
    void * src2_d = nullptr;

    const bool use_src2 = src2 != nullptr;

    if (use_src2) {
        src2_d = (void *)src2->data;
    }

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16) || (src2 && src2->type == GGML_TYPE_F16);

    if (use_f16) {
        const half * src1_dd = (const half *)src1_d;
        const half * src2_dd = (const half *)src2_d;

        soft_max_f32_cuda(src0_d, src1_dd, src2_dd, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, stream);
    } else {
        const float * src1_dd = (const float *)src1_d;
        const float * src2_dd = (const float *)src2_d;

        soft_max_f32_cuda(src0_d, src1_dd, src2_dd, dst_d, ne00, nrows_x, nrows_y, scale, max_bias, stream);
    }
}

static __global__ void k_sum_rows_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum);

    if (col == 0) {
        dst[row] = sum;
    }
}

static void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    k_sum_rows_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));


    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    sum_rows_f32_cuda(src0_d, dst_d, ncols, nrows, stream);
}

static __global__ void timestep_embedding_f32(const float * timesteps, float * dst, const int nb1, const int dim, const int max_period) {
    // blockIDx.y: idx of timesteps->ne[0]
    // blockIDx.x: idx of ((dim + 1) / 2) / BLOCK_SIZE
    int i = blockIdx.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    float * embed_data = (float *)((char *)dst +  i*nb1);

    if (dim % 2 != 0 && j == ((dim + 1) / 2)) {
        embed_data[dim] = 0.f;
    }

    int half = dim / 2;
    if (j >= half) {
        return;
    }

    float timestep = timesteps[i];
    float freq = (float)expf(-logf(max_period) * j / half);
    float arg = timestep * freq;
    embed_data[j] = cosf(arg);
    embed_data[j + half] = sinf(arg);
}

static void timestep_embedding_f32_cuda(const float * x, float * dst, const int ne00, const int nb1,
                                        const int dim, const int max_period, cudaStream_t stream) {
    int half_ceil = (dim + 1) / 2;
    int num_blocks = (half_ceil + CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne00, 1);
    timestep_embedding_f32<<<gridDim, CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE, 0, stream>>>(x, dst, nb1, dim, max_period);
}

void ggml_cuda_op_timestep_embedding(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];

    timestep_embedding_f32_cuda(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}

static __global__ void gelu_f32(const float * x, float * dst, const int k) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    float xi = x[i];
    dst[i] = 0.5f*xi*(1.0f + tanhf(SQRT_2_OVER_PI*xi*(1.0f + GELU_COEF_A*xi*xi)));
}

static __global__ void gelu_quick_f32(const float * x, float * dst, int k) {
    const float GELU_QUICK_COEF = -1.702f;
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x[i])));
}

static __global__ void silu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + expf(-x[i]));
}

static __global__ void tanh_f32(const float * x, float * dst, int k) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = tanhf(x[i]);
}

static __global__ void relu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0);
}

static __global__ void hardsigmoid_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
}

static __global__ void hardswish_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
}

static __global__ void leaky_relu_f32(const float * x, float * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0) + fminf(x[i], 0.0f) * negative_slope;
}

static __global__ void sqr_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

static void gelu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void gelu_quick_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_quick_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void silu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void tanh_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_TANH_BLOCK_SIZE - 1) / CUDA_TANH_BLOCK_SIZE;
    tanh_f32<<<num_blocks, CUDA_TANH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void relu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void hardsigmoid_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSIGMOID_BLOCK_SIZE - 1) / CUDA_HARDSIGMOID_BLOCK_SIZE;
    hardsigmoid_f32<<<num_blocks, CUDA_HARDSIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void hardswish_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSWISH_BLOCK_SIZE - 1) / CUDA_HARDSWISH_BLOCK_SIZE;
    hardswish_f32<<<num_blocks, CUDA_HARDSWISH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void leaky_relu_f32_cuda(const float * x, float * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

static void sqr_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQR_BLOCK_SIZE - 1) / CUDA_SQR_BLOCK_SIZE;
    sqr_f32<<<num_blocks, CUDA_SQR_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    silu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_quick_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    tanh_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    relu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardsigmoid_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardswish_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    leaky_relu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), negative_slope, stream);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqr_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

static __global__ void upscale_f32(const float * x, float * dst, const int ne00, const int ne00xne01, const int scale_factor) {
    // blockIdx.z: idx of ne02*ne03
    // blockIdx.y: idx of ne01*scale_factor， aka ne1
    // blockIDx.x: idx of ne00*scale_factor / BLOCK_SIZE
    // ne00xne01: ne00 * ne01
    int ne0 = ne00 * scale_factor;
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    // operation
    int i00 = nidx / scale_factor;
    int i01 = blockIdx.y / scale_factor;
    int offset_src =
        i00 +
        i01 * ne00 +
        blockIdx.z * ne00xne01;
    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;
    dst[offset_dst] = x[offset_src];
}

static void upscale_f32_cuda(const float * x, float * dst, const int ne00, const int ne01, const int ne02, const int ne03,
                             const int scale_factor, cudaStream_t stream) {
    int ne0 = (ne00 * scale_factor);
    int num_blocks = (ne0 + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;
    dim3 gridDim(num_blocks, (ne01 * scale_factor), ne02*ne03);
    upscale_f32<<<gridDim, CUDA_UPSCALE_BLOCK_SIZE, 0, stream>>>(x, dst, ne00, ne00 * ne01, scale_factor);
}

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    const int scale_factor = dst->op_params[0];

    upscale_f32_cuda(src0_d, dst_d, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], scale_factor, stream);
}
#include "ggml-cuda.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    cudaGetDevice(&id);

    fprintf(stderr, "CUDA error: %s\n", msg);
    fprintf(stderr, "  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    fprintf(stderr, "  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ASSERT(!"CUDA error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static ggml_cuda_device_info ggml_cuda_init() {
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
#ifndef GGML_USE_TINYBLAS
    rocblas_initialize();
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
#endif

    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
    if (!FLAG_log_disable)
#if defined(GGML_CUDA_FORCE_MMQ)
    fprintf(stderr, "%s: GGML_CUDA_FORCE_MMQ:   yes\n", __func__);
#else
    fprintf(stderr, "%s: GGML_CUDA_FORCE_MMQ:   no\n", __func__);
#endif
    if (!FLAG_log_disable)
#if defined(CUDA_USE_TENSOR_CORES)
    fprintf(stderr, "%s: CUDA_USE_TENSOR_CORES: yes\n", __func__);
#else
    fprintf(stderr, "%s: CUDA_USE_TENSOR_CORES: no\n", __func__);
#endif
    if (!FLAG_log_disable)
    fprintf(stderr, "%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, info.device_count);
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if !defined(GGML_USE_HIPBLAS)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // !defined(GGML_USE_HIPBLAS)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        if (!FLAG_log_disable)
        fprintf(stderr, "  Device %d: %s, compute capability %d.%d, VMM: %s\n", id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].cc = 100*prop.major + 10*prop.minor + CC_OFFSET_AMD;
#else
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].smpb = prop.sharedMemPerBlock;
        info.devices[id].nsm  = prop.multiProcessorCount;
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc((void **) &ptr, look_ahead_size));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        fprintf(stderr, "%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(pool_size/1024/1024), (uint32_t)(size/1024/1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        fprintf(stderr, "WARNING: cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if !defined(GGML_USE_HIPBLAS)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CU_CHECK(cuMemMap(pool_addr + pool_size, reserve_size, 0, handle, 0));

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess(pool_addr + pool_size, reserve_size, &access, 1));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) (pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) (pool_addr + pool_used));
    }
};
#endif // !defined(GGML_USE_HIPBLAS)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) {
#if !defined(GGML_USE_HIPBLAS)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

GGML_CALL static const char * ggml_backend_cuda_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_buffer_get_name;
}

GGML_CALL static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
}

GGML_CALL static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

GGML_CALL static const char * ggml_backend_cuda_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1); // cudaMalloc returns null for size 0

    void * dev_ptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size/1024.0/1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

GGML_CALL static bool ggml_backend_cuda_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) {
        return false;
    }

    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return buft_ctx->device == cuda_ctx->device;
}

static ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_cuda_buffer_type_supports_backend,
    /* .is_host          = */ NULL,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (tensor_split[id] < (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            if (min_compute_capability > ggml_cuda_info().devices[id].cc) {
                min_compute_capability = ggml_cuda_info().devices[id].cc;
            }
            if (max_compute_capability < ggml_cuda_info().devices[id].cc) {
                max_compute_capability = ggml_cuda_info().devices[id].cc;
            }
        }
    }

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return max_compute_capability >= CC_RDNA2 ? 128 : 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
            return max_compute_capability >= CC_RDNA2 ? 128 : 32;
        case GGML_TYPE_Q3_K:
            return min_compute_capability < CC_RDNA2 ? 128 : 64;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= CC_RDNA2 ? 128 : 64;
        default:
            GGML_ASSERT(false);
    }
#else
    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ASSERT(false);
    }
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;

    if (id == ggml_backend_cuda_get_device_count() - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_cuda_split_buffer_type_context {
    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
                for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
                    if (extra->events[id][is] != nullptr) {
                        CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
                    }
                }
                if (extra->data_device[id] != nullptr) {
                    CUDA_CHECK(cudaFree(extra->data_device[id]));
                }
            }
            delete extra;
        }
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};

GGML_CALL static const char * ggml_backend_cuda_split_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buffer);
}

static bool ggml_backend_buffer_is_cuda_split(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_split_buffer_get_name;
    GGML_UNUSED(ggml_backend_buffer_is_cuda_split); // only used in debug builds currently, avoid unused function warning in release builds
}

GGML_CALL static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if cudaMalloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_cuda_set_device(id);
        char * buf;
        CUDA_CHECK(cudaMalloc(&buf, size));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
        }

        extra->data_device[id] = buf;

        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
            CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
        }
    }
    tensor->extra = extra;
}

GGML_CALL static void ggml_backend_cuda_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

GGML_CALL static void ggml_backend_cuda_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

GGML_CALL static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_split_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

GGML_CALL static const char * ggml_backend_cuda_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buft);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

GGML_CALL static bool ggml_backend_cuda_split_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_cuda(backend);

    GGML_UNUSED(buft);
}

GGML_CALL static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_cuda_split_buffer_type_supports_backend,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<std::array<float, GGML_CUDA_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_CUDA_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_cuda_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    if (it != buft_map.end()) {
        return &it->second;
    }

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .context = */ new ggml_backend_cuda_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    return &result.first->second;
}

// host buffer type

GGML_CALL static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_cuda_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        fprintf(stderr, "%s: warning: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
            size/1024.0/1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_cuda_host_buffer_name;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .supports_backend = */ ggml_backend_cpu_buffer_type()->iface.supports_backend,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static __global__ void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y) {

    const half * x = (const half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;

        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __global__ void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x, const int channel_x_divisor) {

    const half * x = (const half *) vx;

    const int row_x     = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel   = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / channel_x_divisor;

    const int nrows_y   = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel*nrows_y + row_y;

        const float xi = __half2float(x[ix]);

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static void ggml_mul_mat_p021_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x,
    const int nchannels_x, const int nchannels_y, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_p021_f16_f32<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, nchannels_x, nchannels_y);
}

static void ggml_mul_mat_vec_nc_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int row_stride_x,
    const int nchannels_x, const int nchannels_y, const int channel_stride_x, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_vec_nc_f16_f32<<<block_nums, block_dims, 0, stream>>>
        (vx, y, dst, ncols_x, nrows_x, row_stride_x, channel_stride_x, nchannels_y/nchannels_x);
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src->buffer));
    char * src_ptr = (char *) src->data;
    char * dst_ptr = (char *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int compute_capability = ggml_cuda_info().devices[id].cc;

    if (compute_capability >= CC_VOLTA && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();
        ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

        const half alpha_f16 = 1.0f;
        const half beta_f16 = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f16, src0_ptr,       CUDA_R_16F, ne00,
                                src1_ptr,       CUDA_R_16F, ne10,
                    &beta_f16,   dst_f16.get(), CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}

static void ggml_cuda_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_CUDA_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);

        for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
            if (id == id_other) {
                continue;
            }
            if (id != main_device && id_other != main_device) {
                continue;
            }

            int can_access_peer;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (can_access_peer) {
                if (enable_peer_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
                    if (err != cudaErrorPeerAccessAlreadyEnabled) {
                        CUDA_CHECK(err);
                    }
                } else {
                    cudaError_t err = cudaDeviceDisablePeerAccess(id_other);
                    if (err != cudaErrorPeerAccessNotEnabled) {
                        CUDA_CHECK(err);
                    }
                }
            }
        }
    }

    ggml_cuda_set_device(main_device);
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;

    GGML_UNUSED(main_device);
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    const bool convert_src1_to_q8_1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_cuda_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    ggml_tensor_extra_gpu * src0_extra = split ? (ggml_tensor_extra_gpu *) src0->extra : nullptr;


    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_cuda_pool_alloc<char>  src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (id != 0) {
                dev[id].row_low  = ne01*tensor_split[id];
                if (dev[id].row_low < ne01) {
                    dev[id].row_low -= dev[id].row_low % rounding;
                }
            }

            if (id != ggml_backend_cuda_get_device_count() - 1) {
                dev[id].row_high  = ne01*tensor_split[id + 1];
                if (dev[id].row_high < ne01) {
                    dev[id].row_high -= dev[id].row_high % rounding;
                }
            }
        }
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = split ? (char *) src0_extra->data_device[id] : (char *) src0->data;
        } else {
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (convert_src1_to_q8_1) {
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                quantize_row_q8_1_cuda(dev[id].src1_ddf, dev[id].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low)*ne1 : ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaEventRecord(src0_extra->events[ctx.device][0], ctx.stream()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_CUDA_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            // wait for main GPU data if necessary
            if (split && (id != ctx.device || is != 0)) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[ctx.device][0], 0));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[id].src0_dd + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (convert_src1_to_q8_1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddq_i, id, src1_ddq_i_source, ctx.device,
                                                            src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ASSERT(false);
                }

                if (convert_src1_to_q8_1 && !src1_is_contiguous) {
                    quantize_row_q8_1_cuda(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, src1_padded_col_size, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[id].row_low;
#if !defined(GGML_USE_HIPBLAS)
                        // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
                        cudaMemcpy3DPeerParms p = {};
                        p.dstDevice = ctx.device;
                        p.dstPtr = make_cudaPitchedPtr(dhf_dst_i, ne0*sizeof(float), row_diff, src1_ncols);
                        p.srcDevice = id;
                        p.srcPtr = make_cudaPitchedPtr(dst_dd_i, row_diff*sizeof(float), row_diff, src1_ncols);
                        p.extent = make_cudaExtent(row_diff*sizeof(float), src1_ncols, 1);
                        CUDA_CHECK(cudaMemcpy3DPeerAsync(&p, stream));
#else
                        // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
                        CUDA_CHECK(cudaMemcpy2DAsync(dhf_dst_i, ne0*sizeof(float),
                                                        dst_dd_i, row_diff*sizeof(float),
                                                        row_diff*sizeof(float), src1_ncols,
                                                        cudaMemcpyDeviceToDevice, stream));
#endif
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != ctx.device || is != 0)) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_backend_cuda_get_device_count() > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_CUDA_MAX_STREAMS ? is_max : GGML_CUDA_MAX_STREAMS;

        ggml_cuda_set_device(ctx.device);
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (dev[id].row_low == dev[id].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), src0_extra->events[id][is], 0));
            }
        }
    }
}

static void ggml_cuda_mul_mat_vec_p021(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}

static void ggml_cuda_mul_mat_vec_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(half);
    const int64_t channel_stride_x = nb02 / sizeof(half);

    ggml_mul_mat_vec_nc_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, main_stream);
}

static __global__ void k_compute_batched_ptrs(
        const half * src0_as_f16, const half * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    int64_t i03 = i13 / r3;
    int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);

    cudaStream_t main_stream = ctx.stream();

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    void * src0_ddq = src0->data;
    half * src0_f16 = (half *) src0_ddq;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    // convert src1 to fp16
    ggml_cuda_pool_alloc<half> src1_f16_alloc(ctx.pool());
    if (src1->type != GGML_TYPE_F16) {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        to_fp16_cuda(src1_ddf, src1_f16_alloc.get(), ne_src1, main_stream);
    }
    half * src1_f16 = src1->type == GGML_TYPE_F16 ? (half *) src1_ddf : src1_f16_alloc.get();

    ggml_cuda_pool_alloc<half> dst_f16(ctx.pool());
    char * dst_t;

    cublasComputeType_t cu_compute_type = CUBLAS_COMPUTE_16F;
    cudaDataType_t      cu_data_type    = CUDA_R_16F;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const half  alpha_f16 = 1.0f;
    const half  beta_f16  = 0.0f;

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f16;
    const void * beta  = &beta_f16;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        dst_t = (char *) dst_f16.alloc(ne_dst);

        nbd2 /= sizeof(float) / sizeof(half);
        nbd3 /= sizeof(float) / sizeof(half);
    } else {
        dst_t = (char *) dst_ddf;

        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type    = CUDA_R_32F;

        alpha = &alpha_f32;
        beta  = &beta_f32;
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

#if 0
    // use cublasGemmEx
    {
        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                int i03 = i13 / r3;
                int i02 = i12 / r2;

                CUBLAS_CHECK(
                        cublasGemmEx(g_cublas_handles[g_main_device], CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            alpha, (const char *) src0_as_f16 + i02*src0->nb[2]   + i03*src0->nb[3]  , CUDA_R_16F,   nb01/sizeof(half),
                                   (const char *) src1_as_f16 + i12*src1->nb[2]/2 + i13*src1->nb[3]/2, CUDA_R_16F,   nb11/sizeof(float),
                            beta,  (      char *)       dst_t + i12*nbd2          + i13*nbd3,          cu_data_type, ne01,
                            cu_compute_type,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
    }
#else
    if (r2 == 1 && r3 == 1 && src0->nb[2]*src0->ne[2] == src0->nb[3] && src1->nb[2]*src1->ne[2] == src1->nb[3]) {
        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const char *) src0_f16, CUDA_R_16F,   nb01/nb00, nb02/nb00,  // strideA
                       (const char *) src1_f16, CUDA_R_16F,   nb11/nb10, nb12/nb10,  // strideB
                beta,  (      char *)    dst_t, cu_data_type, ne01,       nb2/nb0,   // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        dim3 block_dims(ne13, ne12);
        k_compute_batched_ptrs<<<1, block_dims, 0, main_stream>>>(
                src0_f16, src1_f16, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                src1->type == GGML_TYPE_F16 ? nb12 : nb12/2,
                src1->type == GGML_TYPE_F16 ? nb13 : nb13/2,
                nbd2, nbd3,
                r2, r3);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), CUDA_R_16F,   nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), CUDA_R_16F,   nb11/nb10,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type, ne01,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
#endif

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const bool split = ggml_backend_buffer_is_cuda_split(src0->buffer);

    int64_t min_compute_capability = INT_MAX;

    bool any_pascal_with_slow_fp16 = false;
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_cuda_info().devices[id].cc) {
                min_compute_capability = ggml_cuda_info().devices[id].cc;
            }
            if (ggml_cuda_info().devices[id].cc == 610) {
                any_pascal_with_slow_fp16 = true;
            }
        }
    } else {
        min_compute_capability    = ggml_cuda_info().devices[ctx.device].cc;
        any_pascal_with_slow_fp16 = ggml_cuda_info().devices[ctx.device].cc == 610;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % GGML_CUDA_DMMV_X == 0 && src1->ne[1] == 1;

    bool          use_mul_mat_vec_q =  ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    bool              use_mul_mat_q =  ggml_cuda_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)

    const bool fp16_performance_good = min_compute_capability >= CC_RDNA1;

#ifdef CUDA_USE_TENSOR_CORES
    use_mul_mat_q = use_mul_mat_q && min_compute_capability < CC_RDNA3;
#endif // CUDA_USE_TENSOR_CORES

#else

    // fp16 performance is good on Volta or newer and on P100 (compute capability 6.0)
    const bool fp16_performance_good = min_compute_capability >= CC_PASCAL && !any_pascal_with_slow_fp16;

    // mmvq and mmq need the __dp4a instruction which on NVIDIA is only available for CC >= 6.1
    use_mul_mat_vec_q = use_mul_mat_vec_q && min_compute_capability >= MIN_CC_DP4A;
    use_mul_mat_q     = use_mul_mat_q     && min_compute_capability >= MIN_CC_DP4A;

#ifdef CUDA_USE_TENSOR_CORES
    // when tensor cores are available, use them for large batch size
    // ref: https://github.com/ggerganov/llama.cpp/pull/3776
    use_mul_mat_q     = use_mul_mat_q     && (!fp16_performance_good || src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // CUDA_USE_TENSOR_CORES

#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)

    // if mmvq is available it's a better choice than dmmv:
#ifndef GGML_CUDA_FORCE_DMMV
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
#endif // GGML_CUDA_FORCE_DMMV

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (!split && !fp16_performance_good && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // KQ single-batch
        ggml_cuda_mul_mat_vec_p021(ctx, src0, src1, dst);
    } else if (!split && !fp16_performance_good && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1) {
        // KQV single-batch
        ggml_cuda_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || fp16_performance_good) && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, false);
    } else if (use_mul_mat_vec_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, true);
    } else if (use_mul_mat_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, true);
    } else {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, false);
    }
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static __global__ void k_copy_src1_to_contiguous(const char * __restrict__ src1_original, char * __restrict__ src1_contiguous,
                                                 int * __restrict__ cur_src1_row, mmid_row_mapping * __restrict__ row_mapping,
                                                 const char * __restrict ids, int64_t i02, size_t ids_nb1, size_t ids_nb0,
                                                 int64_t ne11, int64_t ne10,
                                                 size_t nb11, size_t nb12) {
    int32_t iid1 = blockIdx.x;
    int32_t id = blockIdx.y;

    const int32_t row_id_i = *(const int32_t *) (ids + iid1*ids_nb1 + id*ids_nb0);

    if (row_id_i != i02) {
        return;
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    __shared__ int src1_row;
    if (threadIdx.x == 0) {
        src1_row = atomicAdd(cur_src1_row, 1);
        row_mapping[src1_row] = {id, iid1};
    }
    __syncthreads();

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);
    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);

    for (int i = threadIdx.x; i < ne10; i += blockDim.x) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

static __global__ void k_copy_dst_from_contiguous(char * __restrict__ dst_original, const char * __restrict__ dst_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne0,
                                                  size_t nb1, size_t nb2) {
    int32_t i = blockIdx.x;

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

    for (int j = threadIdx.x; j < ne0; j += blockDim.x) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(!ggml_backend_buffer_is_cuda_split(src0->buffer) && "mul_mat_id does not support split buffers");

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *)  dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);

                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

                src0_row.data = src0_original + i02*nb02;
                src1_row.data = src1_original + i11*nb11 + i12*nb12;
                dst_row.data  =  dst_original + i1*nb1   + i2*nb2;

                ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
        ggml_cuda_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            int64_t num_src1_rows = 0;

            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);

                    GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

                    if (row_id_i != i02) {
                        continue;
                    }

                    num_src1_rows++;
                }
            }

            if (num_src1_rows == 0) {
                continue;
            }

            ggml_cuda_pool_alloc<int> dev_cur_src1_row(ctx.pool(), 1);
            ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), num_src1_rows);
            CUDA_CHECK(cudaMemsetAsync(dev_cur_src1_row.get(), 0, sizeof(int), stream));

            {
                dim3 block_dims(std::min((unsigned int)ne10, 768u));
                dim3 grid_dims(ids->ne[1], n_ids);
                k_copy_src1_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                        src1_original, src1_contiguous.get(),
                        dev_cur_src1_row.get(), dev_row_mapping.get(),
                        ids_dev, i02, ids->nb[1], ids->nb[0],
                        ne11, ne10,
                        nb11, nb12);
                CUDA_CHECK(cudaGetLastError());
            }

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);

            src1_row.ne[1] = num_src1_rows;
            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row);

            {
                dim3 block_dims(std::min((unsigned int)ne0, 768u));
                dim3 grid_dims(num_src1_rows);
                k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                        dst_original, dst_contiguous.get(),
                        dev_row_mapping.get(),
                        ne0,
                        nb1, nb2);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // why is this here instead of mul_mat?
    if (dst->src[0] != nullptr && ggml_backend_buffer_is_cuda_split(dst->src[0]->buffer)) {
        ggml_cuda_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                fprintf(stderr, "%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
                return false;
            } else {
                ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ALIBI:
            ggml_cuda_op_alibi(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

GGML_CALL static const char * ggml_backend_cuda_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_cuda_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return ggml_backend_cuda_buffer_type(cuda_ctx->device);
}

GGML_CALL static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

GGML_CALL static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

GGML_CALL static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_backend_is_cuda(backend_src) || ggml_backend_is_cuda(backend_dst));

    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_buffer_is_cuda(src->buffer)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *)backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *)backend_dst->context;

    if (backend_src != backend_dst) {
        ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *)buf_src->context;
        ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *)buf_dst->context;

        GGML_ASSERT(cuda_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cuda_ctx_dst->device == buf_ctx_dst->device);

        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_dst->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif
        }

        // record event on src stream
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_dst->stream()));
    }
    return true;
}

GGML_CALL static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

GGML_CALL static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) || ggml_backend_buffer_is_cuda_split(node->src[j]->buffer));
            }
        }
#endif

        bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
        if (!ok) {
            fprintf(stderr, "%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_cuda_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a;
                struct ggml_tensor * b;
                if (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } else {
                    a = op->src[2];
                    b = op->src[1];
                }
                if (a->ne[3] != b->ne[3]) {
                    return false;
                }
                ggml_type a_type = a->type;
                if (a_type == GGML_TYPE_IQ2_XXS || a_type == GGML_TYPE_IQ2_XS || a_type == GGML_TYPE_IQ3_XXS ||
                    a_type == GGML_TYPE_IQ1_S   || a_type == GGML_TYPE_IQ4_NL || a_type == GGML_TYPE_IQ3_S   ||
                    a_type == GGML_TYPE_IQ1_M   || a_type == GGML_TYPE_IQ2_S  || a_type == GGML_TYPE_IQ4_XS) {
                    if (b->ne[1] == 1 && ggml_nrows(b) > 1) {
                        return false;
                    }
                }
                return true;
            } break;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ALIBI:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_FLASH_ATTN_EXT:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_cuda_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
           (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

    GGML_UNUSED(backend);
}

GGML_CALL static ggml_backend_event_t ggml_backend_cuda_event_new(ggml_backend_t backend) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .backend = */ backend,
        /* .context = */ event,
    };
#endif
}

GGML_CALL static void ggml_backend_cuda_event_free(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));

    delete event;
}

GGML_CALL static void ggml_backend_cuda_event_record(ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)event->backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

GGML_CALL static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(event->backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ASSERT(false);
    }
}

GGML_CALL static void ggml_backend_cuda_event_synchronize(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .get_default_buffer_type = */ ggml_backend_cuda_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .supports_op             = */ ggml_backend_cuda_supports_op,
    /* .offload_op              = */ ggml_backend_cuda_offload_op,
    /* .event_new               = */ ggml_backend_cuda_event_new,
    /* .event_free              = */ ggml_backend_cuda_event_free,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .event_synchronize       = */ ggml_backend_cuda_event_synchronize,
};

GGML_CALL static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        fprintf(stderr, "%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cuda_guid(),
        /* .interface = */ ggml_backend_cuda_interface,
        /* .context   = */ ctx
    };

    return cuda_backend;
}

GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

GGML_CALL int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11100
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();

        fprintf(stderr, "%s: warning: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                size/1024.0/1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    return false;
#endif
}

GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
    }
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_cuda_init(const char * params, void * user_data) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init((int) (intptr_t) user_data);
    return cuda_backend;

    GGML_UNUSED(params);
}

extern "C" GGML_API GGML_CALL int ggml_backend_cuda_reg_devices();

GGML_API GGML_CALL int ggml_backend_cuda_reg_devices() {
    int device_count = ggml_backend_cuda_get_device_count();
    //int device_count = 1; // DEBUG: some tools require delaying CUDA initialization
    for (int i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_CUDA_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_cuda_init, ggml_backend_cuda_buffer_type(i), (void *) (intptr_t) i);
    }
    return device_count;
}
