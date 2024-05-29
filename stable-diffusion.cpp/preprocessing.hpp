#ifndef __PREPROCESSING_HPP__
#define __PREPROCESSING_HPP__

#include "ggml_extend.hpp"
#define M_PI_ 3.14159265358979323846

void convolve(struct ggml_tensor* input, struct ggml_tensor* output, struct ggml_tensor* kernel, int padding) {
    struct ggml_init_params params;
    params.mem_size                 = 20 * 1024 * 1024;  // 10
    params.mem_buffer               = NULL;
    params.no_alloc                 = false;
    struct ggml_context* ctx0       = ggml_init(params);
    struct ggml_tensor* kernel_fp16 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, kernel->ne[0], kernel->ne[1], 1, 1);
    ggml_fp32_to_fp16_row((float*)kernel->data, (ggml_fp16_t*)kernel_fp16->data, ggml_nelements(kernel));
    ggml_tensor* h  = ggml_conv_2d(ctx0, kernel_fp16, input, 1, 1, padding, padding, 1, 1);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, h, output));
    ggml_graph_compute_with_ctx(ctx0, gf, 1);
    ggml_free(ctx0);
}

void gaussian_kernel(struct ggml_tensor* kernel) {
    int ks_mid   = kernel->ne[0] / 2;
    float sigma  = 1.4f;
    float normal = 1.f / (2.0f * M_PI_ * powf(sigma, 2.0f));
    for (int y = 0; y < kernel->ne[0]; y++) {
        float gx = -ks_mid + y;
        for (int x = 0; x < kernel->ne[1]; x++) {
            float gy = -ks_mid + x;
            float k_ = expf(-((gx * gx + gy * gy) / (2.0f * powf(sigma, 2.0f)))) * normal;
            ggml_tensor_set_f32(kernel, k_, x, y);
        }
    }
}

void grayscale(struct ggml_tensor* rgb_img, struct ggml_tensor* grayscale) {
    for (int iy = 0; iy < rgb_img->ne[1]; iy++) {
        for (int ix = 0; ix < rgb_img->ne[0]; ix++) {
            float r    = ggml_tensor_get_f32(rgb_img, ix, iy);
            float g    = ggml_tensor_get_f32(rgb_img, ix, iy, 1);
            float b    = ggml_tensor_get_f32(rgb_img, ix, iy, 2);
            float gray = 0.2989f * r + 0.5870f * g + 0.1140f * b;
            ggml_tensor_set_f32(grayscale, gray, ix, iy);
        }
    }
}

void prop_hypot(struct ggml_tensor* x, struct ggml_tensor* y, struct ggml_tensor* h) {
    int n_elements = ggml_nelements(h);
    float* dx      = (float*)x->data;
    float* dy      = (float*)y->data;
    float* dh      = (float*)h->data;
    for (int i = 0; i < n_elements; i++) {
        dh[i] = sqrtf(dx[i] * dx[i] + dy[i] * dy[i]);
    }
}

void prop_arctan2(struct ggml_tensor* x, struct ggml_tensor* y, struct ggml_tensor* h) {
    int n_elements = ggml_nelements(h);
    float* dx      = (float*)x->data;
    float* dy      = (float*)y->data;
    float* dh      = (float*)h->data;
    for (int i = 0; i < n_elements; i++) {
        dh[i] = atan2f(dy[i], dx[i]);
    }
}

void normalize_tensor(struct ggml_tensor* g) {
    int n_elements = ggml_nelements(g);
    float* dg      = (float*)g->data;
    float max      = -INFINITY;
    for (int i = 0; i < n_elements; i++) {
        max = dg[i] > max ? dg[i] : max;
    }
    max = 1.0f / max;
    for (int i = 0; i < n_elements; i++) {
        dg[i] *= max;
    }
}

void non_max_supression(struct ggml_tensor* result, struct ggml_tensor* G, struct ggml_tensor* D) {
    for (int iy = 1; iy < result->ne[1] - 1; iy++) {
        for (int ix = 1; ix < result->ne[0] - 1; ix++) {
            float angle = ggml_tensor_get_f32(D, ix, iy) * 180.0f / M_PI_;
            angle       = angle < 0.0f ? angle += 180.0f : angle;
            float q     = 1.0f;
            float r     = 1.0f;

            // angle 0
            if ((0 >= angle && angle < 22.5f) || (157.5f >= angle && angle <= 180)) {
                q = ggml_tensor_get_f32(G, ix, iy + 1);
                r = ggml_tensor_get_f32(G, ix, iy - 1);
            }
            // angle 45
            else if (22.5f >= angle && angle < 67.5f) {
                q = ggml_tensor_get_f32(G, ix + 1, iy - 1);
                r = ggml_tensor_get_f32(G, ix - 1, iy + 1);
            }
            // angle 90
            else if (67.5f >= angle && angle < 112.5) {
                q = ggml_tensor_get_f32(G, ix + 1, iy);
                r = ggml_tensor_get_f32(G, ix - 1, iy);
            }
            // angle 135
            else if (112.5 >= angle && angle < 157.5f) {
                q = ggml_tensor_get_f32(G, ix - 1, iy - 1);
                r = ggml_tensor_get_f32(G, ix + 1, iy + 1);
            }

            float cur = ggml_tensor_get_f32(G, ix, iy);
            if ((cur >= q) && (cur >= r)) {
                ggml_tensor_set_f32(result, cur, ix, iy);
            } else {
                ggml_tensor_set_f32(result, 0.0f, ix, iy);
            }
        }
    }
}

void threshold_hystersis(struct ggml_tensor* img, float high_threshold, float low_threshold, float weak, float strong) {
    int n_elements = ggml_nelements(img);
    float* imd     = (float*)img->data;
    float max      = -INFINITY;
    for (int i = 0; i < n_elements; i++) {
        max = imd[i] > max ? imd[i] : max;
    }
    float ht = max * high_threshold;
    float lt = ht * low_threshold;
    for (int i = 0; i < n_elements; i++) {
        float img_v = imd[i];
        if (img_v >= ht) {  // strong pixel
            imd[i] = strong;
        } else if (img_v <= ht && img_v >= lt) {  // strong pixel
            imd[i] = weak;
        }
    }

    for (int iy = 0; iy < img->ne[1]; iy++) {
        for (int ix = 0; ix < img->ne[0]; ix++) {
            if (ix >= 3 && ix <= img->ne[0] - 3 && iy >= 3 && iy <= img->ne[1] - 3) {
                ggml_tensor_set_f32(img, ggml_tensor_get_f32(img, ix, iy), ix, iy);
            } else {
                ggml_tensor_set_f32(img, 0.0f, ix, iy);
            }
        }
    }

    // hysteresis
    for (int iy = 1; iy < img->ne[1] - 1; iy++) {
        for (int ix = 1; ix < img->ne[0] - 1; ix++) {
            float imd_v = ggml_tensor_get_f32(img, ix, iy);
            if (imd_v == weak) {
                if (ggml_tensor_get_f32(img, ix + 1, iy - 1) == strong || ggml_tensor_get_f32(img, ix + 1, iy) == strong ||
                    ggml_tensor_get_f32(img, ix, iy - 1) == strong || ggml_tensor_get_f32(img, ix, iy + 1) == strong ||
                    ggml_tensor_get_f32(img, ix - 1, iy - 1) == strong || ggml_tensor_get_f32(img, ix - 1, iy) == strong) {
                    ggml_tensor_set_f32(img, strong, ix, iy);
                } else {
                    ggml_tensor_set_f32(img, 0.0f, ix, iy);
                }
            }
        }
    }
}

uint8_t* preprocess_canny(uint8_t* img, int width, int height, float high_threshold, float low_threshold, float weak, float strong, bool inverse) {
    struct ggml_init_params params;
    params.mem_size               = static_cast<size_t>(10 * 1024 * 1024);  // 10
    params.mem_buffer             = NULL;
    params.no_alloc               = false;
    struct ggml_context* work_ctx = ggml_init(params);

    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    float kX[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1};

    float kY[9] = {
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1};

    // generate kernel
    int kernel_size             = 5;
    struct ggml_tensor* gkernel = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, kernel_size, kernel_size, 1, 1);
    struct ggml_tensor* sf_kx   = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 3, 3, 1, 1);
    memcpy(sf_kx->data, kX, ggml_nbytes(sf_kx));
    struct ggml_tensor* sf_ky = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 3, 3, 1, 1);
    memcpy(sf_ky->data, kY, ggml_nbytes(sf_ky));
    gaussian_kernel(gkernel);
    struct ggml_tensor* image      = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
    struct ggml_tensor* image_gray = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 1, 1);
    struct ggml_tensor* iX         = ggml_dup_tensor(work_ctx, image_gray);
    struct ggml_tensor* iY         = ggml_dup_tensor(work_ctx, image_gray);
    struct ggml_tensor* G          = ggml_dup_tensor(work_ctx, image_gray);
    struct ggml_tensor* tetha      = ggml_dup_tensor(work_ctx, image_gray);
    sd_image_to_tensor(img, image);
    grayscale(image, image_gray);
    convolve(image_gray, image_gray, gkernel, 2);
    convolve(image_gray, iX, sf_kx, 1);
    convolve(image_gray, iY, sf_ky, 1);
    prop_hypot(iX, iY, G);
    normalize_tensor(G);
    prop_arctan2(iX, iY, tetha);
    non_max_supression(image_gray, G, tetha);
    threshold_hystersis(image_gray, high_threshold, low_threshold, weak, strong);
    // to RGB channels
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            float gray = ggml_tensor_get_f32(image_gray, ix, iy);
            gray       = inverse ? 1.0f - gray : gray;
            ggml_tensor_set_f32(image, gray, ix, iy);
            ggml_tensor_set_f32(image, gray, ix, iy, 1);
            ggml_tensor_set_f32(image, gray, ix, iy, 2);
        }
    }
    free(img);
    uint8_t* output = sd_tensor_to_image(image);
    ggml_free(work_ctx);
    return output;
}

#endif  // __PREPROCESSING_HPP__