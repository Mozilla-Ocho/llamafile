#ifndef __TAE_HPP__
#define __TAE_HPP__

#include "ggml_extend.hpp"

#include "model.h"

/*
    ===================================    TinyAutoEncoder  ===================================
    References:
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py
    https://github.com/madebyollin/taesd/blob/main/taesd.py

*/

class TAEBlock : public UnaryBlock {
protected:
    int n_in;
    int n_out;

public:
    TAEBlock(int n_in, int n_out)
        : n_in(n_in), n_out(n_out) {
        blocks["conv.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_in, n_out, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_out, n_out, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_out, n_out, {3, 3}, {1, 1}, {1, 1}));
        if (n_in != n_out) {
            blocks["skip"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_in, n_out, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [n, n_in, h, w]
        // return: [n, n_out, h, w]

        auto conv_0 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.0"]);
        auto conv_2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.2"]);
        auto conv_4 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.4"]);

        auto h = conv_0->forward(ctx, x);
        h      = ggml_relu_inplace(ctx, h);
        h      = conv_2->forward(ctx, h);
        h      = ggml_relu_inplace(ctx, h);
        h      = conv_4->forward(ctx, h);

        if (n_in != n_out) {
            auto skip = std::dynamic_pointer_cast<Conv2d>(blocks["skip"]);
            LOG_DEBUG("skip");
            x = skip->forward(ctx, x);
        }

        h = ggml_add(ctx, h, x);
        h = ggml_relu_inplace(ctx, h);
        return h;
    }
};

class TinyEncoder : public UnaryBlock {
    int in_channels = 3;
    int channels    = 64;
    int z_channels  = 4;
    int num_blocks  = 3;

public:
    TinyEncoder() {
        int index                       = 0;
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, channels, {3, 3}, {1, 1}, {1, 1}));
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, z_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [n, in_channels, h, w]
        // return: [n, z_channels, h/8, w/8]

        for (int i = 0; i < num_blocks * 3 + 6; i++) {
            auto block = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(i)]);

            x = block->forward(ctx, x);
        }

        return x;
    }
};

class TinyDecoder : public UnaryBlock {
    int z_channels   = 4;
    int channels     = 64;
    int out_channels = 3;
    int num_blocks   = 3;

public:
    TinyDecoder(int index = 0) {
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(z_channels, channels, {3, 3}, {1, 1}, {1, 1}));
        index++;  // nn.ReLU()

        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }
        index++;  // nn.Upsample()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));

        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }
        index++;  // nn.Upsample()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));

        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }
        index++;  // nn.Upsample()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [n, z_channels, h, w]
        // return: [n, out_channels, h*8, w*8]

        auto h = ggml_scale(ctx, z, 1.0f / 3.0f);
        h      = ggml_tanh_inplace(ctx, h);
        h      = ggml_scale(ctx, h, 3.0f);

        for (int i = 0; i < num_blocks * 3 + 10; i++) {
            if (blocks.find(std::to_string(i)) == blocks.end()) {
                if (i == 1) {
                    h = ggml_relu_inplace(ctx, h);
                } else {
                    h = ggml_upscale(ctx, h, 2);
                }
                continue;
            }
            auto block = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(i)]);

            h = block->forward(ctx, h);
        }

        return h;
    }
};

class TAESD : public GGMLBlock {
protected:
    bool decode_only;

public:
    TAESD(bool decode_only = true)
        : decode_only(decode_only) {
        blocks["decoder.layers"] = std::shared_ptr<GGMLBlock>(new TinyDecoder());

        if (!decode_only) {
            blocks["encoder.layers"] = std::shared_ptr<GGMLBlock>(new TinyEncoder());
        }
    }

    struct ggml_tensor* decode(struct ggml_context* ctx, struct ggml_tensor* z) {
        auto decoder = std::dynamic_pointer_cast<TinyDecoder>(blocks["decoder.layers"]);
        return decoder->forward(ctx, z);
    }

    struct ggml_tensor* encode(struct ggml_context* ctx, struct ggml_tensor* x) {
        auto encoder = std::dynamic_pointer_cast<TinyEncoder>(blocks["encoder.layers"]);
        return encoder->forward(ctx, x);
    }
};

struct TinyAutoEncoder : public GGMLRunner {
    TAESD taesd;
    bool decode_only = false;

    TinyAutoEncoder(ggml_backend_t backend,
                    ggml_type wtype,
                    bool decoder_only = true)
        : decode_only(decoder_only),
          taesd(decode_only),
          GGMLRunner(backend, wtype) {
        taesd.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "taesd";
    }

    bool load_from_file(const std::string& file_path) {
        LOG_INFO("loading taesd from '%s', decode_only = %s", file_path.c_str(), decode_only ? "true" : "false");
        alloc_params_buffer();
        std::map<std::string, ggml_tensor*> taesd_tensors;
        taesd.get_param_tensors(taesd_tensors);
        std::set<std::string> ignore_tensors;
        if (decode_only) {
            ignore_tensors.insert("encoder.");
        }

        ModelLoader model_loader;
        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init taesd model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool success = model_loader.load_tensors(taesd_tensors, backend, ignore_tensors);

        if (!success) {
            LOG_ERROR("load tae tensors from model loader failed");
            return false;
        }

        LOG_INFO("taesd model loaded");
        return success;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        struct ggml_cgraph* gf  = ggml_new_graph(compute_ctx);
        z                       = to_backend(z);
        struct ggml_tensor* out = decode_graph ? taesd.decode(compute_ctx, z) : taesd.encode(compute_ctx, z);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* z,
                 bool decode_graph,
                 struct ggml_tensor** output,
                 struct ggml_context* output_ctx = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(z, decode_graph);
        };

        GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

#endif  // __TAE_HPP__