#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

struct ggml_cuda_device_properties {
    char name[256];
    size_t totalGlobalMem;
    int multiProcessorCount;
    int major;
    int minor;
    char compute[8];
};

GGML_API GGML_CALL bool ggml_cuda_link(const struct ggml_backend_api * backend_api);

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_properties(int device, struct ggml_cuda_device_properties * properties);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

GGML_API GGML_CALL int ggml_backend_cuda_reg_devices(); // [jart]

#ifdef  __cplusplus
}
#endif
