#ifndef __UNET_HPP__
#define __UNET_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"
#include "model.h"

/*==================================================== UnetModel =====================================================*/

#define UNET_GRAPH_SIZE 10240

class SpatialVideoTransformer : public SpatialTransformer {
protected:
    int64_t time_depth;
    int64_t max_time_embed_period;

public:
    SpatialVideoTransformer(int64_t in_channels,
                            int64_t n_head,
                            int64_t d_head,
                            int64_t depth,
                            int64_t context_dim,
                            int64_t time_depth            = 1,
                            int64_t max_time_embed_period = 10000)
        : SpatialTransformer(in_channels, n_head, d_head, depth, context_dim),
          max_time_embed_period(max_time_embed_period) {
        // We will convert unet transformer linear to conv2d 1x1 when loading the weights, so use_linear is always False
        // use_spatial_context is always True
        // merge_strategy is always learned_with_images
        // merge_factor is loaded from weights
        // time_context_dim is always None
        // ff_in is always True
        // disable_self_attn is always False
        // disable_temporal_crossattention is always False

        int64_t inner_dim = n_head * d_head;

        GGML_ASSERT(depth == time_depth);
        GGML_ASSERT(in_channels == inner_dim);

        int64_t time_mix_d_head    = d_head;
        int64_t n_time_mix_heads   = n_head;
        int64_t time_mix_inner_dim = time_mix_d_head * n_time_mix_heads;  // equal to inner_dim
        int64_t time_context_dim   = context_dim;

        for (int i = 0; i < time_depth; i++) {
            std::string name = "time_stack." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new BasicTransformerBlock(inner_dim,
                                                                                    n_time_mix_heads,
                                                                                    time_mix_d_head,
                                                                                    time_context_dim,
                                                                                    true));
        }

        int64_t time_embed_dim     = in_channels * 4;
        blocks["time_pos_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, time_embed_dim));
        // time_pos_embed.1 is nn.SiLU()
        blocks["time_pos_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, in_channels));

        blocks["time_mixer"] = std::shared_ptr<GGMLBlock>(new AlphaBlender());
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* context,
                                int timesteps) {
        // x: [N, in_channels, h, w] aka [b*t, in_channels, h, w], t == timesteps
        // context: [N, max_position(aka n_context), hidden_size(aka context_dim)] aka [b*t, n_context, context_dim], t == timesteps
        // t_emb: [N, in_channels] aka [b*t, in_channels]
        // timesteps is num_frames
        // time_context is always None
        // image_only_indicator is always tensor([0.])
        // transformer_options is not used
        // GGML_ASSERT(ggml_n_dims(context) == 3);

        auto norm             = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        auto proj_in          = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in"]);
        auto proj_out         = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);
        auto time_pos_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["time_pos_embed.0"]);
        auto time_pos_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["time_pos_embed.2"]);
        auto time_mixer       = std::dynamic_pointer_cast<AlphaBlender>(blocks["time_mixer"]);

        auto x_in         = x;
        int64_t n         = x->ne[3];
        int64_t h         = x->ne[1];
        int64_t w         = x->ne[0];
        int64_t inner_dim = n_head * d_head;

        GGML_ASSERT(n == timesteps);  // We compute cond and uncond separately, so batch_size==1

        auto time_context    = context;  // [b*t, n_context, context_dim]
        auto spatial_context = context;
        // time_context_first_timestep = time_context[::timesteps]
        auto time_context_first_timestep = ggml_view_3d(ctx,
                                                        time_context,
                                                        time_context->ne[0],
                                                        time_context->ne[1],
                                                        1,
                                                        time_context->nb[1],
                                                        time_context->nb[2],
                                                        0);  // [b, n_context, context_dim]
        time_context                     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                              time_context_first_timestep->ne[0],
                                                              time_context_first_timestep->ne[1],
                                                              time_context_first_timestep->ne[2] * h * w);
        time_context                     = ggml_repeat(ctx, time_context_first_timestep, time_context);  // [b*h*w, n_context, context_dim]

        x = norm->forward(ctx, x);
        x = proj_in->forward(ctx, x);  // [N, inner_dim, h, w]

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [N, h, w, inner_dim]
        x = ggml_reshape_3d(ctx, x, inner_dim, w * h, n);      // [N, h * w, inner_dim]

        auto num_frames = ggml_arange(ctx, 0, timesteps, 1);
        // since b is 1, no need to do repeat
        auto t_emb = ggml_nn_timestep_embedding(ctx, num_frames, in_channels, max_time_embed_period);  // [N, in_channels]

        auto emb = time_pos_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx, emb);
        emb      = time_pos_embed_2->forward(ctx, emb);                   // [N, in_channels]
        emb      = ggml_reshape_3d(ctx, emb, emb->ne[0], 1, emb->ne[1]);  // [N, 1, in_channels]

        for (int i = 0; i < depth; i++) {
            std::string transformer_name = "transformer_blocks." + std::to_string(i);
            std::string time_stack_name  = "time_stack." + std::to_string(i);

            auto block     = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[transformer_name]);
            auto mix_block = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[time_stack_name]);

            x = block->forward(ctx, x, spatial_context);  // [N, h * w, inner_dim]

            // in_channels == inner_dim
            auto x_mix = x;
            x_mix      = ggml_add(ctx, x_mix, emb);  // [N, h * w, inner_dim]

            int64_t N = x_mix->ne[2];
            int64_t T = timesteps;
            int64_t B = N / T;
            int64_t S = x_mix->ne[1];
            int64_t C = x_mix->ne[0];

            x_mix = ggml_reshape_4d(ctx, x_mix, C, S, T, B);               // (b t) s c -> b t s c
            x_mix = ggml_cont(ctx, ggml_permute(ctx, x_mix, 0, 2, 1, 3));  // b t s c -> b s t c
            x_mix = ggml_reshape_3d(ctx, x_mix, C, T, S * B);              // b s t c -> (b s) t c

            x_mix = mix_block->forward(ctx, x_mix, time_context);  // [B * h * w, T, inner_dim]

            x_mix = ggml_reshape_4d(ctx, x_mix, C, T, S, B);               // (b s) t c -> b s t c
            x_mix = ggml_cont(ctx, ggml_permute(ctx, x_mix, 0, 2, 1, 3));  // b s t c -> b t s c
            x_mix = ggml_reshape_3d(ctx, x_mix, C, S, T * B);              // b t s c -> (b t) s c

            x = time_mixer->forward(ctx, x, x_mix);  // [N, h * w, inner_dim]
        }

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [N, inner_dim, h * w]
        x = ggml_reshape_4d(ctx, x, w, h, inner_dim, n);       // [N, inner_dim, h, w]

        // proj_out
        x = proj_out->forward(ctx, x);  // [N, in_channels, h, w]

        x = ggml_add(ctx, x, x_in);
        return x;
    }
};

// ldm.modules.diffusionmodules.openaimodel.UNetModel
class UnetModelBlock : public GGMLBlock {
protected:
    SDVersion version = VERSION_1_x;
    // network hparams
    int in_channels                        = 4;
    int out_channels                       = 4;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult          = {1, 2, 4, 4};
    std::vector<int> transformer_depth     = {1, 1, 1, 1};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = 8;
    int num_head_channels                  = -1;   // channels // num_heads
    int context_dim                        = 768;  // 1024 for VERSION_2_x, 2048 for VERSION_XL

public:
    int model_channels  = 320;
    int adm_in_channels = 2816;  // only for VERSION_XL/SVD

    UnetModelBlock(SDVersion version = VERSION_1_x)
        : version(version) {
        if (version == VERSION_2_x) {
            context_dim       = 1024;
            num_head_channels = 64;
            num_heads         = -1;
        } else if (version == VERSION_XL) {
            context_dim           = 2048;
            attention_resolutions = {4, 2};
            channel_mult          = {1, 2, 4};
            transformer_depth     = {1, 2, 10};
            num_head_channels     = 64;
            num_heads             = -1;
        } else if (version == VERSION_SVD) {
            in_channels       = 8;
            out_channels      = 4;
            context_dim       = 1024;
            adm_in_channels   = 768;
            num_head_channels = 64;
            num_heads         = -1;
        }
        // dims is always 2
        // use_temporal_attention is always True for SVD

        blocks["time_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(model_channels, time_embed_dim));
        // time_embed_1 is nn.SiLU()
        blocks["time_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));

        if (version == VERSION_XL || version == VERSION_SVD) {
            blocks["label_emb.0.0"] = std::shared_ptr<GGMLBlock>(new Linear(adm_in_channels, time_embed_dim));
            // label_emb_1 is nn.SiLU()
            blocks["label_emb.0.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));
        }

        // input_blocks
        blocks["input_blocks.0.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, model_channels, {3, 3}, {1, 1}, {1, 1}));

        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch              = model_channels;
        int input_block_idx = 0;
        int ds              = 1;

        auto get_resblock = [&](int64_t channels, int64_t emb_channels, int64_t out_channels) -> ResBlock* {
            if (version == VERSION_SVD) {
                return new VideoResBlock(channels, emb_channels, out_channels);
            } else {
                return new ResBlock(channels, emb_channels, out_channels);
            }
        };

        auto get_attention_layer = [&](int64_t in_channels,
                                       int64_t n_head,
                                       int64_t d_head,
                                       int64_t depth,
                                       int64_t context_dim) -> SpatialTransformer* {
            if (version == VERSION_SVD) {
                return new SpatialVideoTransformer(in_channels, n_head, d_head, depth, context_dim);
            } else {
                return new SpatialTransformer(in_channels, n_head, d_head, depth, context_dim);
            }
        };

        size_t len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, mult * model_channels));

                ch = mult * model_channels;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    blocks[name]     = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                      n_head,
                                                                                      d_head,
                                                                                      transformer_depth[i],
                                                                                      context_dim));
                }
                input_block_chans.push_back(ch);
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(ch, ch));

                input_block_chans.push_back(ch);
                ds *= 2;
            }
        }

        // middle blocks
        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        blocks["middle_block.0"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));
        blocks["middle_block.1"] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                  n_head,
                                                                                  d_head,
                                                                                  transformer_depth[transformer_depth.size() - 1],
                                                                                  context_dim));
        blocks["middle_block.2"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));

        // output_blocks
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks + 1; j++) {
                int ich = input_block_chans.back();
                input_block_chans.pop_back();

                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch + ich, time_embed_dim, mult * model_channels));

                ch                = mult * model_channels;
                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";
                    blocks[name]     = std::shared_ptr<GGMLBlock>(get_attention_layer(ch, n_head, d_head, transformer_depth[i], context_dim));

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    blocks[name]     = std::shared_ptr<GGMLBlock>(new UpSampleBlock(ch, ch));

                    ds /= 2;
                }

                output_block_idx += 1;
            }
        }

        // out
        blocks["out.0"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(ch));  // ch == model_channels
        // out_1 is nn.SiLU()
        blocks["out.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(model_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* resblock_forward(std::string name,
                                         struct ggml_context* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* emb,
                                         int num_video_frames) {
        if (version == VERSION_SVD) {
            auto block = std::dynamic_pointer_cast<VideoResBlock>(blocks[name]);

            return block->forward(ctx, x, emb, num_video_frames);
        } else {
            auto block = std::dynamic_pointer_cast<ResBlock>(blocks[name]);

            return block->forward(ctx, x, emb);
        }
    }

    struct ggml_tensor* attention_layer_forward(std::string name,
                                                struct ggml_context* ctx,
                                                struct ggml_tensor* x,
                                                struct ggml_tensor* context,
                                                int timesteps) {
        if (version == VERSION_SVD) {
            auto block = std::dynamic_pointer_cast<SpatialVideoTransformer>(blocks[name]);

            return block->forward(ctx, x, context, timesteps);
        } else {
            auto block = std::dynamic_pointer_cast<SpatialTransformer>(blocks[name]);

            return block->forward(ctx, x, context);
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* timesteps,
                                struct ggml_tensor* context,
                                struct ggml_tensor* c_concat              = NULL,
                                struct ggml_tensor* y                     = NULL,
                                int num_video_frames                      = -1,
                                std::vector<struct ggml_tensor*> controls = {},
                                float control_strength                    = 0.f) {
        // x: [N, in_channels, h, w] or [N, in_channels/2, h, w]
        // timesteps: [N,]
        // context: [N, max_position, hidden_size] or [1, max_position, hidden_size]. for example, [N, 77, 768]
        // c_concat: [N, in_channels, h, w] or [1, in_channels, h, w]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        // return: [N, out_channels, h, w]
        if (context != NULL) {
            if (context->ne[2] != x->ne[3]) {
                context = ggml_repeat(ctx, context, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, context->ne[0], context->ne[1], x->ne[3]));
            }
        }

        if (c_concat != NULL) {
            if (c_concat->ne[3] != x->ne[3]) {
                c_concat = ggml_repeat(ctx, c_concat, x);
            }
            x = ggml_concat(ctx, x, c_concat);
        }

        if (y != NULL) {
            if (y->ne[1] != x->ne[3]) {
                y = ggml_repeat(ctx, y, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, y->ne[0], x->ne[3]));
            }
        }

        auto time_embed_0     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.0"]);
        auto time_embed_2     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.2"]);
        auto input_blocks_0_0 = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks.0.0"]);

        auto out_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out.0"]);
        auto out_2 = std::dynamic_pointer_cast<Conv2d>(blocks["out.2"]);

        auto t_emb = ggml_nn_timestep_embedding(ctx, timesteps, model_channels);  // [N, model_channels]

        auto emb = time_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx, emb);
        emb      = time_embed_2->forward(ctx, emb);  // [N, time_embed_dim]

        // SDXL/SVD
        if (y != NULL) {
            auto label_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.0"]);
            auto label_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.2"]);

            auto label_emb = label_embed_0->forward(ctx, y);
            label_emb      = ggml_silu_inplace(ctx, label_emb);
            label_emb      = label_embed_2->forward(ctx, label_emb);  // [N, time_embed_dim]

            emb = ggml_add(ctx, emb, label_emb);  // [N, time_embed_dim]
        }

        // input_blocks
        std::vector<struct ggml_tensor*> hs;

        // input block 0
        auto h = input_blocks_0_0->forward(ctx, x);

        ggml_set_name(h, "bench-start");
        hs.push_back(h);
        // input block 1-11
        size_t len_mults    = channel_mult.size();
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                h                = resblock_forward(name, ctx, h, emb, num_video_frames);  // [N, mult*model_channels, h, w]
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    h                = attention_layer_forward(name, ctx, h, context, num_video_frames);  // [N, mult*model_channels, h, w]
                }
                hs.push_back(h);
            }
            if (i != len_mults - 1) {
                ds *= 2;
                input_block_idx += 1;

                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                auto block       = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = block->forward(ctx, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]
                hs.push_back(h);
            }
        }
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        h = resblock_forward("middle_block.0", ctx, h, emb, num_video_frames);             // [N, 4*model_channels, h/8, w/8]
        h = attention_layer_forward("middle_block.1", ctx, h, context, num_video_frames);  // [N, 4*model_channels, h/8, w/8]
        h = resblock_forward("middle_block.2", ctx, h, emb, num_video_frames);             // [N, 4*model_channels, h/8, w/8]

        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[controls.size() - 1], control_strength);
            h       = ggml_add(ctx, h, cs);  // middle control
        }
        int control_offset = controls.size() - 2;

        // output_blocks
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                auto h_skip = hs.back();
                hs.pop_back();

                if (controls.size() > 0) {
                    auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
                    h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
                    control_offset--;
                }

                h = ggml_concat(ctx, h, h_skip);

                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";

                h = resblock_forward(name, ctx, h, emb, num_video_frames);

                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";

                    h = attention_layer_forward(name, ctx, h, context, num_video_frames);

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    auto block       = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);

                    h = block->forward(ctx, h);

                    ds /= 2;
                }

                output_block_idx += 1;
            }
        }

        // out
        h = out_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        h = out_2->forward(ctx, h);
        ggml_set_name(h, "bench-end");
        return h;  // [N, out_channels, h, w]
    }
};

struct UNetModel : public GGMLModule {
    SDVersion version = VERSION_1_x;
    UnetModelBlock unet;

    UNetModel(ggml_backend_t backend,
              ggml_type wtype,
              SDVersion version = VERSION_1_x)
        : GGMLModule(backend, wtype), unet(version) {
        unet.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "unet";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        unet.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* c_concat              = NULL,
                                    struct ggml_tensor* y                     = NULL,
                                    int num_video_frames                      = -1,
                                    std::vector<struct ggml_tensor*> controls = {},
                                    float control_strength                    = 0.f) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, UNET_GRAPH_SIZE, false);

        if (num_video_frames == -1) {
            num_video_frames = x->ne[3];
        }

        x         = to_backend(x);
        context   = to_backend(context);
        y         = to_backend(y);
        timesteps = to_backend(timesteps);

        for (int i = 0; i < controls.size(); i++) {
            controls[i] = to_backend(controls[i]);
        }

        struct ggml_tensor* out = unet.forward(compute_ctx,
                                               x,
                                               timesteps,
                                               context,
                                               c_concat,
                                               y,
                                               num_video_frames,
                                               controls,
                                               control_strength);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 77, 768]) or [1, max_position, hidden_size]
        // c_concat: [N, in_channels, h, w] or [1, in_channels, h, w]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, context, c_concat, y, num_video_frames, controls, control_strength);
        };

        GGMLModule::compute(get_graph, n_threads, false, output, output_ctx);
    }

    void test() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        struct ggml_context* work_ctx = ggml_init(params);
        GGML_ASSERT(work_ctx != NULL);

        {
            // CPU, num_video_frames = 1, x{num_video_frames, 8, 8, 8}: Pass
            // CUDA, num_video_frames = 1, x{num_video_frames, 8, 8, 8}: Pass
            // CPU, num_video_frames = 3, x{num_video_frames, 8, 8, 8}: Wrong result
            // CUDA, num_video_frames = 3, x{num_video_frames, 8, 8, 8}: nan
            int num_video_frames = 3;

            auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 8, num_video_frames);
            std::vector<float> timesteps_vec(num_video_frames, 999.f);
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            ggml_set_f32(x, 0.5f);
            // print_ggml_tensor(x);

            auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 1024, 1, num_video_frames);
            ggml_set_f32(context, 0.5f);
            // print_ggml_tensor(context);

            auto y = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 768, num_video_frames);
            ggml_set_f32(y, 0.5f);
            // print_ggml_tensor(y);

            struct ggml_tensor* out = NULL;

            int t0 = ggml_time_ms();
            compute(8, x, timesteps, context, NULL, y, num_video_frames, {}, 0.f, &out, work_ctx);
            int t1 = ggml_time_ms();

            print_ggml_tensor(out);
            LOG_DEBUG("unet test done in %dms", t1 - t0);
        }
    };
};

#endif  // __UNET_HPP__