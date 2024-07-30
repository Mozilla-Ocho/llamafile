#ifndef __ESRGAN_HPP__
#define __ESRGAN_HPP__

#include "ggml_extend.hpp"
#include "model.h"

/*
    ===================================    ESRGAN  ===================================
    References:
    https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/rrdbnet_arch.py

*/

class ResidualDenseBlock : public GGMLBlock {
protected:
    int num_feat;
    int num_grow_ch;

public:
    ResidualDenseBlock(int num_feat = 64, int num_grow_ch = 32)
        : num_feat(num_feat), num_grow_ch(num_grow_ch) {
        blocks["conv1"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv2"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + num_grow_ch, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv3"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv4"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv5"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat + 4 * num_grow_ch, num_feat, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* lrelu(struct ggml_context* ctx, struct ggml_tensor* x) {
        return ggml_leaky_relu(ctx, x, 0.2f, true);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [n, num_feat, h, w]
        // return: [n, num_feat, h, w]

        auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
        auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"]);
        auto conv3 = std::dynamic_pointer_cast<Conv2d>(blocks["conv3"]);
        auto conv4 = std::dynamic_pointer_cast<Conv2d>(blocks["conv4"]);
        auto conv5 = std::dynamic_pointer_cast<Conv2d>(blocks["conv5"]);

        auto x1    = lrelu(ctx, conv1->forward(ctx, x));
        auto x_cat = ggml_concat(ctx, x, x1, 2);
        auto x2    = lrelu(ctx, conv2->forward(ctx, x_cat));
        x_cat      = ggml_concat(ctx, x_cat, x2, 2);
        auto x3    = lrelu(ctx, conv3->forward(ctx, x_cat));
        x_cat      = ggml_concat(ctx, x_cat, x3, 2);
        auto x4    = lrelu(ctx, conv4->forward(ctx, x_cat));
        x_cat      = ggml_concat(ctx, x_cat, x4, 2);
        auto x5    = conv5->forward(ctx, x_cat);

        x5 = ggml_add(ctx, ggml_scale(ctx, x5, 0.2f), x);
        return x5;
    }
};

class RRDB : public GGMLBlock {
public:
    RRDB(int num_feat, int num_grow_ch = 32) {
        blocks["rdb1"] = std::shared_ptr<GGMLBlock>(new ResidualDenseBlock(num_feat, num_grow_ch));
        blocks["rdb2"] = std::shared_ptr<GGMLBlock>(new ResidualDenseBlock(num_feat, num_grow_ch));
        blocks["rdb3"] = std::shared_ptr<GGMLBlock>(new ResidualDenseBlock(num_feat, num_grow_ch));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [n, num_feat, h, w]
        // return: [n, num_feat, h, w]

        auto rdb1 = std::dynamic_pointer_cast<ResidualDenseBlock>(blocks["rdb1"]);
        auto rdb2 = std::dynamic_pointer_cast<ResidualDenseBlock>(blocks["rdb2"]);
        auto rdb3 = std::dynamic_pointer_cast<ResidualDenseBlock>(blocks["rdb3"]);

        auto out = rdb1->forward(ctx, x);
        out      = rdb2->forward(ctx, out);
        out      = rdb3->forward(ctx, out);

        out = ggml_add(ctx, ggml_scale(ctx, out, 0.2f), x);
        return out;
    }
};

class RRDBNet : public GGMLBlock {
protected:
    int scale       = 4;  // default RealESRGAN_x4plus_anime_6B
    int num_block   = 6;  // default RealESRGAN_x4plus_anime_6B
    int num_in_ch   = 3;
    int num_out_ch  = 3;
    int num_feat    = 64;  // default RealESRGAN_x4plus_anime_6B
    int num_grow_ch = 32;  // default RealESRGAN_x4plus_anime_6B

public:
    RRDBNet() {
        blocks["conv_first"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_in_ch, num_feat, {3, 3}, {1, 1}, {1, 1}));
        for (int i = 0; i < num_block; i++) {
            std::string name = "body." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new RRDB(num_feat, num_grow_ch));
        }
        blocks["conv_body"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_feat, {3, 3}, {1, 1}, {1, 1}));
        // upsample
        blocks["conv_up1"]  = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_feat, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv_up2"]  = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_feat, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv_hr"]   = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_feat, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv_last"] = std::shared_ptr<GGMLBlock>(new Conv2d(num_feat, num_out_ch, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* lrelu(struct ggml_context* ctx, struct ggml_tensor* x) {
        return ggml_leaky_relu(ctx, x, 0.2f, true);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [n, num_in_ch, h, w]
        // return: [n, num_out_ch, h*4, w*4]
        auto conv_first = std::dynamic_pointer_cast<Conv2d>(blocks["conv_first"]);
        auto conv_body  = std::dynamic_pointer_cast<Conv2d>(blocks["conv_body"]);
        auto conv_up1   = std::dynamic_pointer_cast<Conv2d>(blocks["conv_up1"]);
        auto conv_up2   = std::dynamic_pointer_cast<Conv2d>(blocks["conv_up2"]);
        auto conv_hr    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_hr"]);
        auto conv_last  = std::dynamic_pointer_cast<Conv2d>(blocks["conv_last"]);

        auto feat      = conv_first->forward(ctx, x);
        auto body_feat = feat;
        for (int i = 0; i < num_block; i++) {
            std::string name = "body." + std::to_string(i);
            auto block       = std::dynamic_pointer_cast<RRDB>(blocks[name]);

            body_feat = block->forward(ctx, body_feat);
        }
        body_feat = conv_body->forward(ctx, body_feat);
        feat      = ggml_add(ctx, feat, body_feat);
        // upsample
        feat     = lrelu(ctx, conv_up1->forward(ctx, ggml_upscale(ctx, feat, 2)));
        feat     = lrelu(ctx, conv_up2->forward(ctx, ggml_upscale(ctx, feat, 2)));
        auto out = conv_last->forward(ctx, lrelu(ctx, conv_hr->forward(ctx, feat)));
        return out;
    }
};

struct ESRGAN : public GGMLRunner {
    RRDBNet rrdb_net;
    int scale     = 4;
    int tile_size = 128;  // avoid cuda OOM for 4gb VRAM

    ESRGAN(ggml_backend_t backend,
           ggml_type wtype)
        : GGMLRunner(backend, wtype) {
        rrdb_net.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "esrgan";
    }

    bool load_from_file(const std::string& file_path) {
        LOG_INFO("loading esrgan from '%s'", file_path.c_str());

        alloc_params_buffer();
        std::map<std::string, ggml_tensor*> esrgan_tensors;
        rrdb_net.get_param_tensors(esrgan_tensors);

        ModelLoader model_loader;
        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init esrgan model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool success = model_loader.load_tensors(esrgan_tensors, backend);

        if (!success) {
            LOG_ERROR("load esrgan tensors from model loader failed");
            return false;
        }

        LOG_INFO("esrgan model loaded");
        return success;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x) {
        struct ggml_cgraph* gf  = ggml_new_graph(compute_ctx);
        x                       = to_backend(x);
        struct ggml_tensor* out = rrdb_net.forward(compute_ctx, x);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* x,
                 ggml_tensor** output,
                 ggml_context* output_ctx = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x);
        };
        GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

#endif  // __ESRGAN_HPP__