// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

// ggml-backend internal header

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
        const char *          (*GGML_CALL get_name)        (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*GGML_CALL alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*GGML_CALL get_alignment)   (ggml_backend_buffer_type_t buft);
        // max buffer size that can be allocated
        size_t                (*GGML_CALL get_max_size)    (ggml_backend_buffer_type_t buft);
        // data size needed to allocate the tensor, including padding
        size_t                (*GGML_CALL get_alloc_size)  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
        // check if tensor data is in host memory
        bool                  (*GGML_CALL is_host)         (ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_buffer_type_context_t context;
    };

    // buffer
    typedef void * ggml_backend_buffer_context_t;

    struct ggml_backend_buffer_i {
        const char * (*GGML_CALL get_name)   (ggml_backend_buffer_t buffer);
        void         (*GGML_CALL free_buffer)(ggml_backend_buffer_t buffer);
        void *       (*GGML_CALL get_base)   (ggml_backend_buffer_t buffer);
        void         (*GGML_CALL init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        void         (*GGML_CALL set_tensor) (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*GGML_CALL get_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool         (*GGML_CALL cpy_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
        void         (*GGML_CALL clear)      (ggml_backend_buffer_t buffer, uint8_t value);
        void         (*GGML_CALL reset)      (ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
    };

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        ggml_backend_buffer_context_t context;
        size_t size;
        enum ggml_backend_buffer_usage usage;
    };

    GGML_CALL ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t      buft,
            struct ggml_backend_buffer_i           iface,
                   ggml_backend_buffer_context_t   context,
                   size_t                          size);

    // do not use directly, use ggml_backend_tensor_copy instead
    bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);

    // buffer that contains a collection of buffers
    GGML_CALL ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
    GGML_CALL bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    GGML_CALL void                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);

    //
    // Backend
    //

    typedef void * ggml_backend_context_t;

    struct ggml_backend_i {
        const char * (*GGML_CALL get_name)(ggml_backend_t backend);

        void (*GGML_CALL free)(ggml_backend_t backend);

        // buffer allocation
        ggml_backend_buffer_type_t (*GGML_CALL get_default_buffer_type)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*GGML_CALL set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*GGML_CALL get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*GGML_CALL cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations
        void (*GGML_CALL synchronize)(ggml_backend_t backend);

        // compute graph with a plan (not used currently)
        // create a new plan for a graph
        ggml_backend_graph_plan_t (*GGML_CALL graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*GGML_CALL graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*GGML_CALL graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum ggml_status          (*GGML_CALL graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph without a plan (async)
        enum ggml_status (*GGML_CALL graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // check if the backend can compute an operation
        bool (*GGML_CALL supports_op)(ggml_backend_t backend, const struct ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*GGML_CALL supports_buft)(ggml_backend_t backend, ggml_backend_buffer_type_t buft);

        // check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
        // these should be expensive operations with large batch sizes that may benefit from running on this backend
        // even if the weight has to be copied from the CPU temporarily
        bool (*GGML_CALL offload_op)(ggml_backend_t backend, const struct ggml_tensor * op);

        // (optional) event synchronization
        // create a new event that can record events on this backend instance
        ggml_backend_event_t (*GGML_CALL event_new)         (ggml_backend_t backend);
        void                 (*GGML_CALL event_free)        (ggml_backend_event_t event);
        // record an event on the backend instance that created it
        void                 (*GGML_CALL event_record)      (ggml_backend_event_t event);
        // wait for an event on on a different backend instance
        void                 (*GGML_CALL event_wait)        (ggml_backend_t backend, ggml_backend_event_t event);
        // block until an event is recorded
        void                 (*GGML_CALL event_synchronize) (ggml_backend_event_t event);
    };

    struct ggml_backend {
        ggml_guid_t guid;

        struct ggml_backend_i iface;
        ggml_backend_context_t context;
    };

    struct ggml_backend_event {
        ggml_backend_t backend;
        void * context;
    };

    //
    // Backend registry
    //

    typedef ggml_backend_t (*GGML_CALL ggml_backend_init_fn)(const char * params, void * user_data);

    GGML_CALL void ggml_backend_register(const char * name, ggml_backend_init_fn init_fn, ggml_backend_buffer_type_t default_buffer_type, void * user_data);

    //
    // GGML Backend API
    //
    // This struct includes all functions that a backend module needs
    // the application to define.
    //

    struct ggml_backend_api {
        bool *FLAG_log_disable;
        void (*GGML_CALL exit)(int);
        void (*GGML_CALL free)(void *);
        void *(*GGML_CALL malloc)(size_t);
        char *(*GGML_CALL getenv)(const char *);
        long (*GGML_CALL write)(int, const void *, long);
        void (*GGML_CALL ggml_backend_register)(const char *, ggml_backend_init_fn, ggml_backend_buffer_type_t, void *);
        ggml_backend_buffer_t (*GGML_CALL ggml_backend_buffer_init)(ggml_backend_buffer_type_t, struct ggml_backend_buffer_i, ggml_backend_buffer_context_t, size_t);
        ggml_backend_buffer_t (*GGML_CALL ggml_backend_cpu_buffer_from_ptr)(void *, size_t);
        ggml_backend_buffer_type_t (*GGML_CALL ggml_backend_cpu_buffer_type)(void);
        size_t (*GGML_CALL ggml_backend_buft_get_alloc_size)(ggml_backend_buffer_type_t, struct ggml_tensor *);
        ggml_backend_buffer_t (*GGML_CALL ggml_backend_buft_alloc_buffer)(ggml_backend_buffer_type_t, size_t);
        bool (*GGML_CALL ggml_backend_is_cpu)(ggml_backend_t);
        void (*GGML_CALL ggml_backend_tensor_get)(const struct ggml_tensor *, void *, size_t, size_t);
        void (*GGML_CALL ggml_backend_tensor_set)(struct ggml_tensor *, const void *, size_t, size_t);
        bool (*GGML_CALL ggml_is_quantized)(enum ggml_type);
        size_t (*GGML_CALL ggml_type_size)(enum ggml_type);
        int64_t (*GGML_CALL ggml_blck_size)(enum ggml_type);
        bool (*GGML_CALL ggml_is_transposed)(const struct ggml_tensor *);
        size_t (*GGML_CALL ggml_nbytes)(const struct ggml_tensor *);
        enum ggml_unary_op (*GGML_CALL ggml_get_unary_op)(const struct ggml_tensor *);
        int64_t (*GGML_CALL ggml_nelements)(const struct ggml_tensor *);
        int64_t (*GGML_CALL ggml_nrows)(const struct ggml_tensor *);
        bool (*GGML_CALL ggml_is_permuted)(const struct ggml_tensor *);
        bool (*GGML_CALL ggml_is_contiguous)(const struct ggml_tensor *);
        const char *(*GGML_CALL ggml_op_name)(enum ggml_op);
        const char *(*GGML_CALL ggml_type_name)(enum ggml_type);
        size_t (*GGML_CALL ggml_element_size)(const struct ggml_tensor *);
        size_t (*GGML_CALL ggml_row_size)(enum ggml_type, int64_t);
        void (*GGML_CALL ggml_rope_yarn_corr_dims)(int, int, float, float, float, float[2]);
        const char *(*GGML_CALL ggml_op_desc)(const struct ggml_tensor *);
        bool (*GGML_CALL ggml_backend_buffer_is_host)(ggml_backend_buffer_t);
        bool (*GGML_CALL ggml_guid_matches)(ggml_guid_t, ggml_guid_t);
        bool (*GGML_CALL ggml_is_empty)(const struct ggml_tensor *);
        enum ggml_backend_buffer_usage (*GGML_CALL ggml_backend_buffer_get_usage)(ggml_backend_buffer_t);
        bool (*GGML_CALL ggml_are_same_shape)(const struct ggml_tensor *, const struct ggml_tensor *);
        bool (*GGML_CALL ggml_is_contiguous_1)(const struct ggml_tensor *);
        bool (*GGML_CALL ggml_is_contiguous_2)(const struct ggml_tensor *);
    };

#ifdef  __cplusplus
}
#endif
