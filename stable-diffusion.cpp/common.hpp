#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include "ggml_extend.hpp"

class DownSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;
    bool vae_downsample;

public:
    DownSampleBlock(int channels,
                    int out_channels,
                    bool vae_downsample = false)
        : channels(channels),
          out_channels(out_channels),
          vae_downsample(vae_downsample) {
        if (vae_downsample) {
            blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {0, 0}));
        } else {
            blocks["op"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {1, 1}));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        if (vae_downsample) {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

            x = ggml_pad(ctx, x, 1, 1, 0, 0);
            x = conv->forward(ctx, x);
        } else {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["op"]);

            x = conv->forward(ctx, x);
        }
        return x;  // [N, out_channels, h/2, w/2]
    }
};

class UpSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;

public:
    UpSampleBlock(int channels,
                  int out_channels)
        : channels(channels),
          out_channels(out_channels) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

        x = ggml_upscale(ctx, x, 2);  // [N, channels, h*2, w*2]
        x = conv->forward(ctx, x);    // [N, out_channels, h*2, w*2]
        return x;
    }
};

class ResBlock : public GGMLBlock {
protected:
    // network hparams
    int64_t channels;      // model_channels * (1, 1, 1, 2, 2, 4, 4, 4)
    int64_t emb_channels;  // time_embed_dim
    int64_t out_channels;  // mult * model_channels
    std::pair<int, int> kernel_size;
    int dims;
    bool skip_t_emb;
    bool exchange_temb_dims;

    std::shared_ptr<GGMLBlock> conv_nd(int dims,
                                       int64_t in_channels,
                                       int64_t out_channels,
                                       std::pair<int, int> kernel_size,
                                       std::pair<int, int> padding) {
        GGML_ASSERT(dims == 2 || dims == 3);
        if (dims == 3) {
            return std::shared_ptr<GGMLBlock>(new Conv3dnx1x1(in_channels, out_channels, kernel_size.first, 1, padding.first));
        } else {
            return std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, kernel_size, {1, 1}, padding));
        }
    }

public:
    ResBlock(int64_t channels,
             int64_t emb_channels,
             int64_t out_channels,
             std::pair<int, int> kernel_size = {3, 3},
             int dims                        = 2,
             bool exchange_temb_dims         = false,
             bool skip_t_emb                 = false)
        : channels(channels),
          emb_channels(emb_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          dims(dims),
          skip_t_emb(skip_t_emb),
          exchange_temb_dims(exchange_temb_dims) {
        std::pair<int, int> padding = {kernel_size.first / 2, kernel_size.second / 2};
        blocks["in_layers.0"]       = std::shared_ptr<GGMLBlock>(new GroupNorm32(channels));
        // in_layer_1 is nn.SILU()
        blocks["in_layers.2"] = conv_nd(dims, channels, out_channels, kernel_size, padding);

        if (!skip_t_emb) {
            // emb_layer_0 is nn.SILU()
            blocks["emb_layers.1"] = std::shared_ptr<GGMLBlock>(new Linear(emb_channels, out_channels));
        }

        blocks["out_layers.0"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(out_channels));
        // out_layer_1 is nn.SILU()
        // out_layer_2 is nn.Dropout(), skip for inference
        blocks["out_layers.3"] = conv_nd(dims, out_channels, out_channels, kernel_size, padding);

        if (out_channels != channels) {
            blocks["skip_connection"] = conv_nd(dims, channels, out_channels, {1, 1}, {0, 0});
        }
    }

    virtual struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* emb = NULL) {
        // For dims==3, we reduce dimension from 5d to 4d by merging h and w, in order not to change ggml
        // [N, c, t, h, w] => [N, c, t, h * w]
        // x: [N, channels, h, w] if dims == 2 else [N, channels, t, h, w]
        // emb: [N, emb_channels] if dims == 2 else [N, t, emb_channels]
        auto in_layers_0  = std::dynamic_pointer_cast<GroupNorm32>(blocks["in_layers.0"]);
        auto in_layers_2  = std::dynamic_pointer_cast<UnaryBlock>(blocks["in_layers.2"]);
        auto out_layers_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out_layers.0"]);
        auto out_layers_3 = std::dynamic_pointer_cast<UnaryBlock>(blocks["out_layers.3"]);

        if (emb == NULL) {
            GGML_ASSERT(skip_t_emb);
        }

        // in_layers
        auto h = in_layers_0->forward(ctx, x);
        h      = ggml_silu_inplace(ctx, h);
        h      = in_layers_2->forward(ctx, h);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]

        // emb_layers
        if (!skip_t_emb) {
            auto emb_layer_1 = std::dynamic_pointer_cast<Linear>(blocks["emb_layers.1"]);

            auto emb_out = ggml_silu(ctx, emb);
            emb_out      = emb_layer_1->forward(ctx, emb_out);  // [N, out_channels] if dims == 2 else [N, t, out_channels]

            if (dims == 2) {
                emb_out = ggml_reshape_4d(ctx, emb_out, 1, 1, emb_out->ne[0], emb_out->ne[1]);  // [N, out_channels, 1, 1]
            } else {
                emb_out = ggml_reshape_4d(ctx, emb_out, 1, emb_out->ne[0], emb_out->ne[1], emb_out->ne[2]);  // [N, t, out_channels, 1]
                if (exchange_temb_dims) {
                    // emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                    emb_out = ggml_cont(ctx, ggml_permute(ctx, emb_out, 0, 2, 1, 3));  // [N, out_channels, t, 1]
                }
            }

            h = ggml_add(ctx, h, emb_out);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
        }

        // out_layers
        h = out_layers_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        h = out_layers_3->forward(ctx, h);

        // skip connection
        if (out_channels != channels) {
            auto skip_connection = std::dynamic_pointer_cast<UnaryBlock>(blocks["skip_connection"]);
            x                    = skip_connection->forward(ctx, x);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
        }

        h = ggml_add(ctx, h, x);
        return h;  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
    }
};

class GEGLU : public GGMLBlock {
protected:
    int64_t dim_in;
    int64_t dim_out;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["proj.weight"] = ggml_new_tensor_2d(ctx, wtype, dim_in, dim_out * 2);
        params["proj.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_out * 2);
    }

public:
    GEGLU(int64_t dim_in, int64_t dim_out)
        : dim_in(dim_in), dim_out(dim_out) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [ne3, ne2, ne1, dim_in]
        // return: [ne3, ne2, ne1, dim_out]
        struct ggml_tensor* w = params["proj.weight"];
        struct ggml_tensor* b = params["proj.bias"];

        auto x_w    = ggml_view_2d(ctx, w, w->ne[0], w->ne[1] / 2, w->nb[1], 0);                        // [dim_out, dim_in]
        auto x_b    = ggml_view_1d(ctx, b, b->ne[0] / 2, 0);                                            // [dim_out, dim_in]
        auto gate_w = ggml_view_2d(ctx, w, w->ne[0], w->ne[1] / 2, w->nb[1], w->nb[1] * w->ne[1] / 2);  // [dim_out, ]
        auto gate_b = ggml_view_1d(ctx, b, b->ne[0] / 2, b->nb[0] * b->ne[0] / 2);                      // [dim_out, ]

        auto x_in = x;
        x         = ggml_nn_linear(ctx, x_in, x_w, x_b);        // [ne3, ne2, ne1, dim_out]
        auto gate = ggml_nn_linear(ctx, x_in, gate_w, gate_b);  // [ne3, ne2, ne1, dim_out]

        gate = ggml_gelu_inplace(ctx, gate);

        x = ggml_mul(ctx, x, gate);  // [ne3, ne2, ne1, dim_out]

        return x;
    }
};

class FeedForward : public GGMLBlock {
public:
    FeedForward(int64_t dim,
                int64_t dim_out,
                int64_t mult = 4) {
        int64_t inner_dim = dim * mult;

        blocks["net.0"] = std::shared_ptr<GGMLBlock>(new GEGLU(dim, inner_dim));
        // net_1 is nn.Dropout(), skip for inference
        blocks["net.2"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim_out));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [ne3, ne2, ne1, dim]
        // return: [ne3, ne2, ne1, dim_out]

        auto net_0 = std::dynamic_pointer_cast<GEGLU>(blocks["net.0"]);
        auto net_2 = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);

        x = net_0->forward(ctx, x);  // [ne3, ne2, ne1, inner_dim]
        x = net_2->forward(ctx, x);  // [ne3, ne2, ne1, dim_out]
        return x;
    }
};

class CrossAttention : public GGMLBlock {
protected:
    int64_t query_dim;
    int64_t context_dim;
    int64_t n_head;
    int64_t d_head;

public:
    CrossAttention(int64_t query_dim,
                   int64_t context_dim,
                   int64_t n_head,
                   int64_t d_head)
        : n_head(n_head),
          d_head(d_head),
          query_dim(query_dim),
          context_dim(context_dim) {
        int64_t inner_dim = d_head * n_head;

        blocks["to_q"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, false));
        blocks["to_k"] = std::shared_ptr<GGMLBlock>(new Linear(context_dim, inner_dim, false));
        blocks["to_v"] = std::shared_ptr<GGMLBlock>(new Linear(context_dim, inner_dim, false));

        blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, query_dim));
        // to_out_1 is nn.Dropout(), skip for inference
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* context) {
        // x: [N, n_token, query_dim]
        // context: [N, n_context, context_dim]
        // return: [N, n_token, query_dim]
        auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
        auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
        auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
        auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

        int64_t n         = x->ne[2];
        int64_t n_token   = x->ne[1];
        int64_t n_context = context->ne[1];
        int64_t inner_dim = d_head * n_head;

        auto q = to_q->forward(ctx, x);                                 // [N, n_token, inner_dim]
        q      = ggml_reshape_4d(ctx, q, d_head, n_head, n_token, n);   // [N, n_token, n_head, d_head]
        q      = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));      // [N, n_head, n_token, d_head]
        q      = ggml_reshape_3d(ctx, q, d_head, n_token, n_head * n);  // [N * n_head, n_token, d_head]

        auto k = to_k->forward(ctx, context);                             // [N, n_context, inner_dim]
        k      = ggml_reshape_4d(ctx, k, d_head, n_head, n_context, n);   // [N, n_context, n_head, d_head]
        k      = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));        // [N, n_head, n_context, d_head]
        k      = ggml_reshape_3d(ctx, k, d_head, n_context, n_head * n);  // [N * n_head, n_context, d_head]

        auto v = to_v->forward(ctx, context);                             // [N, n_context, inner_dim]
        v      = ggml_reshape_4d(ctx, v, d_head, n_head, n_context, n);   // [N, n_context, n_head, d_head]
        v      = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));        // [N, n_head, d_head, n_context]
        v      = ggml_reshape_3d(ctx, v, n_context, d_head, n_head * n);  // [N * n_head, d_head, n_context]

        auto kqv = ggml_nn_attention(ctx, q, k, v, false);  // [N * n_head, n_token, d_head]
        kqv      = ggml_reshape_4d(ctx, kqv, d_head, n_token, n_head, n);
        kqv      = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [N, n_token, n_head, d_head]

        x = ggml_reshape_3d(ctx, kqv, d_head * n_head, n_token, n);  // [N, n_token, inner_dim]

        x = to_out_0->forward(ctx, x);  // [N, n_token, query_dim]
        return x;
    }
};

class BasicTransformerBlock : public GGMLBlock {
protected:
    int64_t n_head;
    int64_t d_head;
    bool ff_in;

public:
    BasicTransformerBlock(int64_t dim,
                          int64_t n_head,
                          int64_t d_head,
                          int64_t context_dim,
                          bool ff_in = false)
        : n_head(n_head), d_head(d_head), ff_in(ff_in) {
        // disable_self_attn is always False
        // disable_temporal_crossattention is always False
        // switch_temporal_ca_to_sa is always False
        // inner_dim is always None or equal to dim
        // gated_ff is always True
        blocks["attn1"] = std::shared_ptr<GGMLBlock>(new CrossAttention(dim, dim, n_head, d_head));
        blocks["attn2"] = std::shared_ptr<GGMLBlock>(new CrossAttention(dim, context_dim, n_head, d_head));
        blocks["ff"]    = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim));
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));

        if (ff_in) {
            blocks["norm_in"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
            blocks["ff_in"]   = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* context) {
        // x: [N, n_token, query_dim]
        // context: [N, n_context, context_dim]
        // return: [N, n_token, query_dim]

        auto attn1 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn1"]);
        auto attn2 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn2"]);
        auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);
        auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto norm3 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"]);

        if (ff_in) {
            auto norm_in = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_in"]);
            auto ff_in   = std::dynamic_pointer_cast<FeedForward>(blocks["ff_in"]);

            auto x_skip = x;
            x           = norm_in->forward(ctx, x);
            x           = ff_in->forward(ctx, x);
            // self.is_res is always True
            x = ggml_add(ctx, x, x_skip);
        }

        auto r = x;
        x      = norm1->forward(ctx, x);
        x      = attn1->forward(ctx, x, x);  // self-attention
        x      = ggml_add(ctx, x, r);
        r      = x;
        x      = norm2->forward(ctx, x);
        x      = attn2->forward(ctx, x, context);  // cross-attention
        x      = ggml_add(ctx, x, r);
        r      = x;
        x      = norm3->forward(ctx, x);
        x      = ff->forward(ctx, x);
        x      = ggml_add(ctx, x, r);

        return x;
    }
};

class SpatialTransformer : public GGMLBlock {
protected:
    int64_t in_channels;  // mult * model_channels
    int64_t n_head;
    int64_t d_head;
    int64_t depth       = 1;    // 1
    int64_t context_dim = 768;  // hidden_size, 1024 for VERSION_2_x

public:
    SpatialTransformer(int64_t in_channels,
                       int64_t n_head,
                       int64_t d_head,
                       int64_t depth,
                       int64_t context_dim)
        : in_channels(in_channels),
          n_head(n_head),
          d_head(d_head),
          depth(depth),
          context_dim(context_dim) {
        // We will convert unet transformer linear to conv2d 1x1 when loading the weights, so use_linear is always False
        // disable_self_attn is always False
        int64_t inner_dim = n_head * d_head;  // in_channels
        blocks["norm"]    = std::shared_ptr<GGMLBlock>(new GroupNorm32(in_channels));
        blocks["proj_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, inner_dim, {1, 1}));

        for (int i = 0; i < depth; i++) {
            std::string name = "transformer_blocks." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new BasicTransformerBlock(inner_dim, n_head, d_head, context_dim));
        }

        blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(inner_dim, in_channels, {1, 1}));
    }

    virtual struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* context) {
        // x: [N, in_channels, h, w]
        // context: [N, max_position(aka n_token), hidden_size(aka context_dim)]
        auto norm     = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        auto proj_in  = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in"]);
        auto proj_out = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);

        auto x_in         = x;
        int64_t n         = x->ne[3];
        int64_t h         = x->ne[1];
        int64_t w         = x->ne[0];
        int64_t inner_dim = n_head * d_head;

        x = norm->forward(ctx, x);
        x = proj_in->forward(ctx, x);  // [N, inner_dim, h, w]

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [N, h, w, inner_dim]
        x = ggml_reshape_3d(ctx, x, inner_dim, w * h, n);      // [N, h * w, inner_dim]

        for (int i = 0; i < depth; i++) {
            std::string name       = "transformer_blocks." + std::to_string(i);
            auto transformer_block = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[name]);

            x = transformer_block->forward(ctx, x, context);
        }

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [N, inner_dim, h * w]
        x = ggml_reshape_4d(ctx, x, w, h, inner_dim, n);       // [N, inner_dim, h, w]

        // proj_out
        x = proj_out->forward(ctx, x);  // [N, in_channels, h, w]

        x = ggml_add(ctx, x, x_in);
        return x;
    }
};

class AlphaBlender : public GGMLBlock {
protected:
    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["mix_factor"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    }

    float get_alpha() {
        // image_only_indicator is always tensor([0.]) and since mix_factor.shape is [1,]
        // so learned_with_images is same as learned
        float alpha = ggml_backend_tensor_get_f32(params["mix_factor"]);
        return sigmoid(alpha);
    }

public:
    AlphaBlender() {
        // merge_strategy is always learned_with_images
        // for inference, we don't need to set alpha
        // since mix_factor.shape is [1,], we don't need rearrange using  rearrange_pattern
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x_spatial,
                                struct ggml_tensor* x_temporal) {
        // image_only_indicator is always tensor([0.])
        float alpha = get_alpha();
        auto x      = ggml_add(ctx,
                               ggml_scale(ctx, x_spatial, alpha),
                               ggml_scale(ctx, x_temporal, 1.0f - alpha));
        return x;
    }
};

class VideoResBlock : public ResBlock {
public:
    VideoResBlock(int channels,
                  int emb_channels,
                  int out_channels,
                  std::pair<int, int> kernel_size = {3, 3},
                  int64_t video_kernel_size       = 3,
                  int dims                        = 2)  // always 2
        : ResBlock(channels, emb_channels, out_channels, kernel_size, dims) {
        blocks["time_stack"] = std::shared_ptr<GGMLBlock>(new ResBlock(out_channels, emb_channels, out_channels, kernel_size, 3, true));
        blocks["time_mixer"] = std::shared_ptr<GGMLBlock>(new AlphaBlender());
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* emb,
                                int num_video_frames) {
        // x: [N, channels, h, w] aka [b*t, channels, h, w]
        // emb: [N, emb_channels] aka [b*t, emb_channels]
        // image_only_indicator is always tensor([0.])
        auto time_stack = std::dynamic_pointer_cast<ResBlock>(blocks["time_stack"]);
        auto time_mixer = std::dynamic_pointer_cast<AlphaBlender>(blocks["time_mixer"]);

        x = ResBlock::forward(ctx, x, emb);

        int64_t T = num_video_frames;
        int64_t B = x->ne[3] / T;
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];

        x          = ggml_reshape_4d(ctx, x, W * H, C, T, B);           // (b t) c h w -> b t c (h w)
        x          = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // b t c (h w) -> b c t (h w)
        auto x_mix = x;

        emb = ggml_reshape_4d(ctx, emb, emb->ne[0], T, B, emb->ne[3]);  // (b t) ... -> b t ...

        x = time_stack->forward(ctx, x, emb);  // b t c (h w)

        x = time_mixer->forward(ctx, x_mix, x);  // b t c (h w)

        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // b c t (h w) -> b t c (h w)
        x = ggml_reshape_4d(ctx, x, W, H, C, T * B);           // b t c (h w) -> (b t) c h w

        return x;
    }
};

#endif  // __COMMON_HPP__