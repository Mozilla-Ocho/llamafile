// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

// ggml-backend internal header

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    //
    // Backend buffer
    //

    // buffer type
    typedef void * ggml_backend_buffer_type_context_t;
    struct ggml_backend_buffer_type_i {
        ggml_backend_buffer_t (*GGML_ABI alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
        size_t                (*GGML_ABI get_alignment)   (ggml_backend_buffer_type_t buft); // tensor alignment
        size_t                (*GGML_ABI get_alloc_size)  (ggml_backend_buffer_type_t buft, struct ggml_tensor * tensor); // data size needed to allocate the tensor, including padding
        bool                  (*GGML_ABI supports_backend)(ggml_backend_buffer_type_t buft, ggml_backend_t backend); // check if the buffer type is usable by the backend
        // check if tensor data is in host memory
        // should be equivalent to supports_backend(buft, ggml_backend_cpu_init())
        bool                  (*GGML_ABI is_host)         (ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_buffer_type_context_t context;
    };

    // buffer
    typedef void * ggml_backend_buffer_context_t;

    struct ggml_backend_buffer_i {
        void   (*GGML_ABI free_buffer)    (ggml_backend_buffer_t buffer);
        //void     (*GGML_ABI reset)      (ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
        void * (*GGML_ABI get_base)       (ggml_backend_buffer_t buffer);
        void   (*GGML_ABI init_tensor)    (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        void   (*GGML_ABI set_tensor)     (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void   (*GGML_ABI get_tensor)     (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        // (optional) copy tensor between different buffer-type, allow for single-copy tranfers
        void   (*GGML_ABI cpy_tensor_from)(ggml_backend_buffer_t buffer, struct ggml_tensor * src, struct ggml_tensor * dst);
        void   (*GGML_ABI cpy_tensor_to)  (ggml_backend_buffer_t buffer, struct ggml_tensor * src, struct ggml_tensor * dst);
        void   (*GGML_ABI clear)          (ggml_backend_buffer_t buffer, uint8_t value);
    };

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        ggml_backend_buffer_context_t context;
        size_t size;
    };

    ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t      buft,
            struct ggml_backend_buffer_i           iface,
                   ggml_backend_buffer_context_t   context,
                   size_t                          size) GGML_ABI;


    //
    // Backend
    //

    typedef void * ggml_backend_context_t;

    struct ggml_backend_i {
        const char * (*GGML_ABI get_name)(ggml_backend_t backend);

        void (*GGML_ABI free)(ggml_backend_t backend);

        // buffer allocation
        ggml_backend_buffer_type_t (*GGML_ABI get_default_buffer_type)(ggml_backend_t backend);

        // (optional) asynchroneous tensor data access
        void (*GGML_ABI set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*GGML_ABI get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

        // (optional) asynchroneous tensor copy
        void (*GGML_ABI cpy_tensor_from_async)(ggml_backend_t backend, struct ggml_tensor * src, struct ggml_tensor * dst);
        void (*GGML_ABI cpy_tensor_to_async)  (ggml_backend_t backend, struct ggml_tensor * src, struct ggml_tensor * dst);

        void (*GGML_ABI synchronize)(ggml_backend_t backend);

        // compute graph with a plan
        ggml_backend_graph_plan_t (*GGML_ABI graph_plan_create) (ggml_backend_t backend, struct ggml_cgraph * cgraph);
        void                      (*GGML_ABI graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        void                      (*GGML_ABI graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph without a plan
        void (*GGML_ABI graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*GGML_ABI supports_op)(ggml_backend_t backend, const struct ggml_tensor * op);
    };

    struct ggml_backend {
        struct ggml_backend_i iface;

        ggml_backend_context_t context;
    };


    //
    // Backend registry
    //

    typedef ggml_backend_t (*ggml_backend_init_fn)(const char * params, void * user_data);

    void ggml_backend_register(const char * name, ggml_backend_init_fn init_fn, ggml_backend_buffer_type_t default_buffer_type, void * user_data);

    //
    // GGML Backend API
    //
    // This struct includes all functions that a backend module needs
    // the application to define.
    //

    struct ggml_backend_api {
        void (*GGML_ABI free)(void *);
        void *(*GGML_ABI malloc)(size_t);
        ggml_backend_buffer_t (*GGML_ABI ggml_backend_buffer_init)(ggml_backend_buffer_type_t, struct ggml_backend_buffer_i, ggml_backend_buffer_context_t, size_t);
        ggml_backend_buffer_t (*GGML_ABI ggml_backend_cpu_buffer_from_ptr)(void *, size_t);
        ggml_backend_buffer_type_t (*GGML_ABI ggml_backend_cpu_buffer_type)(void);
        size_t (*GGML_ABI ggml_backend_buft_get_alloc_size)(ggml_backend_buffer_type_t, struct ggml_tensor *);
        ggml_backend_buffer_t (*GGML_ABI ggml_backend_buft_alloc_buffer)(ggml_backend_buffer_type_t, size_t);
        bool (*GGML_ABI ggml_backend_is_cpu)(ggml_backend_t);
        void (*GGML_ABI ggml_backend_tensor_get)(const struct ggml_tensor *, void *, size_t, size_t);
        void (*GGML_ABI ggml_backend_tensor_set)(struct ggml_tensor *, const void *, size_t, size_t);
        bool (*GGML_ABI ggml_is_quantized)(enum ggml_type);
        size_t (*GGML_ABI ggml_type_size)(enum ggml_type);
        int (*GGML_ABI ggml_blck_size)(enum ggml_type);
        bool (*GGML_ABI ggml_is_transposed)(const struct ggml_tensor *);
        size_t (*GGML_ABI ggml_nbytes)(const struct ggml_tensor *);
        enum ggml_unary_op (*GGML_ABI ggml_get_unary_op)(const struct ggml_tensor *);
        int64_t (*GGML_ABI ggml_nelements)(const struct ggml_tensor *);
        int64_t (*GGML_ABI ggml_nrows)(const struct ggml_tensor *);
        bool (*GGML_ABI ggml_is_permuted)(const struct ggml_tensor *);
        bool (*GGML_ABI ggml_is_contiguous)(const struct ggml_tensor *);
        const char *(*GGML_ABI ggml_op_name)(enum ggml_op);
        const char *(*GGML_ABI ggml_type_name)(enum ggml_type);
        size_t (*GGML_ABI ggml_element_size)(const struct ggml_tensor *);
        size_t (*GGML_ABI ggml_row_size)(enum ggml_type, int64_t);
        void (*GGML_ABI ggml_rope_yarn_corr_dims)(int, int, float, float, float, float[2]);
        const char *(*GGML_ABI ggml_op_desc)(const struct ggml_tensor *);
    };

#ifdef  __cplusplus
}
#endif
