#ifndef __MMDIT_HPP__
#define __MMDIT_HPP__

#include "ggml_extend.hpp"
#include "model.h"

#define MMDIT_GRAPH_SIZE 10240

struct Mlp : public GGMLBlock {
public:
    Mlp(int64_t in_features,
        int64_t hidden_features = -1,
        int64_t out_features    = -1,
        bool bias               = true) {
        // act_layer is always lambda: nn.GELU(approximate="tanh")
        // norm_layer is always None
        // use_conv is always False
        if (hidden_features == -1) {
            hidden_features = in_features;
        }
        if (out_features == -1) {
            out_features = in_features;
        }
        blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(in_features, hidden_features, bias));
        blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_features, out_features, bias));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, in_features]
        auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

        x = fc1->forward(ctx, x);
        x = ggml_gelu_inplace(ctx, x);
        x = fc2->forward(ctx, x);
        return x;
    }
};

struct PatchEmbed : public GGMLBlock {
    // 2D Image to Patch Embedding
protected:
    bool flatten;
    bool dynamic_img_pad;
    int patch_size;

public:
    PatchEmbed(int64_t img_size     = 224,
               int patch_size       = 16,
               int64_t in_chans     = 3,
               int64_t embed_dim    = 1536,
               bool bias            = true,
               bool flatten         = true,
               bool dynamic_img_pad = true)
        : patch_size(patch_size),
          flatten(flatten),
          dynamic_img_pad(dynamic_img_pad) {
        // img_size is always None
        // patch_size is always 2
        // in_chans is always 16
        // norm_layer is always False
        // strict_img_size is always true, but not used

        blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_chans,
                                                               embed_dim,
                                                               {patch_size, patch_size},
                                                               {patch_size, patch_size},
                                                               {0, 0},
                                                               {1, 1},
                                                               bias));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, C, H, W]
        // return: [N, H*W, embed_dim]
        auto proj = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

        if (dynamic_img_pad) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int pad_h = (patch_size - H % patch_size) % patch_size;
            int pad_w = (patch_size - W % patch_size) % patch_size;
            x         = ggml_pad(ctx, x, pad_w, pad_h, 0, 0);  // TODO: reflect pad mode
        }
        x = proj->forward(ctx, x);

        if (flatten) {
            x = ggml_reshape_3d(ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        }
        return x;
    }
};

struct TimestepEmbedder : public GGMLBlock {
    // Embeds scalar timesteps into vector representations.
protected:
    int64_t frequency_embedding_size;

public:
    TimestepEmbedder(int64_t hidden_size,
                     int64_t frequency_embedding_size = 256)
        : frequency_embedding_size(frequency_embedding_size) {
        blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(frequency_embedding_size, hidden_size));
        blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* t) {
        // t: [N, ]
        // return: [N, hidden_size]
        auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
        auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

        auto t_freq = ggml_nn_timestep_embedding(ctx, t, frequency_embedding_size);  // [N, frequency_embedding_size]

        auto t_emb = mlp_0->forward(ctx, t_freq);
        t_emb      = ggml_silu_inplace(ctx, t_emb);
        t_emb      = mlp_2->forward(ctx, t_emb);
        return t_emb;
    }
};

struct VectorEmbedder : public GGMLBlock {
    // Embeds a flat vector of dimension input_dim
public:
    VectorEmbedder(int64_t input_dim,
                   int64_t hidden_size) {
        blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(input_dim, hidden_size));
        blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, input_dim]
        // return: [N, hidden_size]
        auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
        auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

        x = mlp_0->forward(ctx, x);
        x = ggml_silu_inplace(ctx, x);
        x = mlp_2->forward(ctx, x);
        return x;
    }
};

__STATIC_INLINE__ std::vector<struct ggml_tensor*> split_qkv(struct ggml_context* ctx,
                                                             struct ggml_tensor* qkv) {
    // qkv: [N, L, 3*C]
    // return: ([N, L, C], [N, L, C], [N, L, C])
    qkv = ggml_reshape_4d(ctx, qkv, qkv->ne[0] / 3, 3, qkv->ne[1], qkv->ne[2]);  // [N, L, 3, C]
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 3, 1, 2));                    // [3, N, L, C]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 0);  // [N, L, C]
    auto k         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 1);  // [N, L, C]
    auto v         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 2);  // [N, L, C]
    return {q, k, v};
}

class SelfAttention : public GGMLBlock {
public:
    int64_t num_heads;
    bool pre_only;

public:
    SelfAttention(int64_t dim,
                  int64_t num_heads = 8,
                  bool qkv_bias     = false,
                  bool pre_only     = false)
        : num_heads(num_heads), pre_only(pre_only) {
        // qk_norm is always None
        blocks["qkv"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
        if (!pre_only) {
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
        }
    }

    std::vector<struct ggml_tensor*> pre_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
        auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);

        auto qkv = qkv_proj->forward(ctx, x);
        return split_qkv(ctx, qkv);
    }

    struct ggml_tensor* post_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
        GGML_ASSERT(!pre_only);

        auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

        x = proj->forward(ctx, x);  // [N, n_token, dim]
        return x;
    }

    // x: [N, n_token, dim]
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        auto qkv = pre_attention(ctx, x);
        x        = ggml_nn_attention_ext(ctx, qkv[0], qkv[1], qkv[2], num_heads);  // [N, n_token, dim]
        x        = post_attention(ctx, x);                                         // [N, n_token, dim]
        return x;
    }
};

__STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                               struct ggml_tensor* x,
                                               struct ggml_tensor* shift,
                                               struct ggml_tensor* scale) {
    // x: [N, L, C]
    // scale: [N, C]
    // shift: [N, C]
    scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
    shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C]
    x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    x     = ggml_add(ctx, x, shift);
    return x;
}

struct DismantledBlock : public GGMLBlock {
    // A DiT block with gated adaptive layer norm (adaLN) conditioning.
public:
    int64_t num_heads;
    bool pre_only;

public:
    DismantledBlock(int64_t hidden_size,
                    int64_t num_heads,
                    float mlp_ratio = 4.0,
                    bool qkv_bias   = false,
                    bool pre_only   = false)
        : num_heads(num_heads), pre_only(pre_only) {
        // rmsnorm is always Flase
        // scale_mod_only is always Flase
        // swiglu is always Flase
        // qk_norm is always Flase
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
        blocks["attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias, pre_only));

        if (!pre_only) {
            blocks["norm2"]        = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
            int64_t mlp_hidden_dim = (int64_t)(hidden_size * mlp_ratio);
            blocks["mlp"]          = std::shared_ptr<GGMLBlock>(new Mlp(hidden_size, mlp_hidden_dim));
        }

        int64_t n_mods = 6;
        if (pre_only) {
            n_mods = 2;
        }
        blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, n_mods * hidden_size));
    }

    std::pair<std::vector<struct ggml_tensor*>, std::vector<struct ggml_tensor*>> pre_attention(struct ggml_context* ctx,
                                                                                                struct ggml_tensor* x,
                                                                                                struct ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int64_t n_mods = 6;
        if (pre_only) {
            n_mods = 2;
        }
        auto m = adaLN_modulation_1->forward(ctx, ggml_silu(ctx, c));  // [N, n_mods * hidden_size]
        m      = ggml_reshape_3d(ctx, m, c->ne[0], n_mods, c->ne[1]);  // [N, n_mods, hidden_size]
        m      = ggml_cont(ctx, ggml_permute(ctx, m, 0, 2, 1, 3));     // [n_mods, N, hidden_size]

        int64_t offset = m->nb[1] * m->ne[1];
        auto shift_msa = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 0);  // [N, hidden_size]
        auto scale_msa = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 1);  // [N, hidden_size]
        if (!pre_only) {
            auto gate_msa  = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 2);  // [N, hidden_size]
            auto shift_mlp = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 3);  // [N, hidden_size]
            auto scale_mlp = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 4);  // [N, hidden_size]
            auto gate_mlp  = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 5);  // [N, hidden_size]

            auto attn_in = modulate(ctx, norm1->forward(ctx, x), shift_msa, scale_msa);

            auto qkv = attn->pre_attention(ctx, attn_in);

            return {qkv, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp}};
        } else {
            auto attn_in = modulate(ctx, norm1->forward(ctx, x), shift_msa, scale_msa);
            auto qkv     = attn->pre_attention(ctx, attn_in);

            return {qkv, {NULL, NULL, NULL, NULL, NULL}};
        }
    }

    struct ggml_tensor* post_attention(struct ggml_context* ctx,
                                       struct ggml_tensor* attn_out,
                                       struct ggml_tensor* x,
                                       struct ggml_tensor* gate_msa,
                                       struct ggml_tensor* shift_mlp,
                                       struct ggml_tensor* scale_mlp,
                                       struct ggml_tensor* gate_mlp) {
        // attn_out: [N, n_token, hidden_size]
        // x: [N, n_token, hidden_size]
        // gate_msa: [N, hidden_size]
        // shift_mlp: [N, hidden_size]
        // scale_mlp: [N, hidden_size]
        // gate_mlp: [N, hidden_size]
        // return: [N, n_token, hidden_size]
        GGML_ASSERT(!pre_only);

        auto attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto mlp   = std::dynamic_pointer_cast<Mlp>(blocks["mlp"]);

        gate_msa = ggml_reshape_3d(ctx, gate_msa, gate_msa->ne[0], 1, gate_msa->ne[1]);  // [N, 1, hidden_size]
        gate_mlp = ggml_reshape_3d(ctx, gate_mlp, gate_mlp->ne[0], 1, gate_mlp->ne[1]);  // [N, 1, hidden_size]

        attn_out = attn->post_attention(ctx, attn_out);

        x            = ggml_add(ctx, x, ggml_mul(ctx, attn_out, gate_msa));
        auto mlp_out = mlp->forward(ctx, modulate(ctx, norm2->forward(ctx, x), shift_mlp, scale_mlp));
        x            = ggml_add(ctx, x, ggml_mul(ctx, mlp_out, gate_mlp));

        return x;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        // return: [N, n_token, hidden_size]

        auto attn = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);

        auto qkv_intermediates = pre_attention(ctx, x, c);
        auto qkv               = qkv_intermediates.first;
        auto intermediates     = qkv_intermediates.second;

        auto attn_out = ggml_nn_attention_ext(ctx, qkv[0], qkv[1], qkv[2], num_heads);  // [N, n_token, dim]
        x             = post_attention(ctx,
                                       attn_out,
                                       intermediates[0],
                                       intermediates[1],
                                       intermediates[2],
                                       intermediates[3],
                                       intermediates[4]);
        return x;  // [N, n_token, dim]
    }
};

__STATIC_INLINE__ std::pair<struct ggml_tensor*, struct ggml_tensor*> block_mixing(struct ggml_context* ctx,
                                                                                   struct ggml_tensor* context,
                                                                                   struct ggml_tensor* x,
                                                                                   struct ggml_tensor* c,
                                                                                   std::shared_ptr<DismantledBlock> context_block,
                                                                                   std::shared_ptr<DismantledBlock> x_block) {
    // context: [N, n_context, hidden_size]
    // x: [N, n_token, hidden_size]
    // c: [N, hidden_size]
    auto context_qkv_intermediates = context_block->pre_attention(ctx, context, c);
    auto context_qkv               = context_qkv_intermediates.first;
    auto context_intermediates     = context_qkv_intermediates.second;

    auto x_qkv_intermediates = x_block->pre_attention(ctx, x, c);
    auto x_qkv               = x_qkv_intermediates.first;
    auto x_intermediates     = x_qkv_intermediates.second;

    std::vector<struct ggml_tensor*> qkv;
    for (int i = 0; i < 3; i++) {
        qkv.push_back(ggml_concat(ctx, context_qkv[i], x_qkv[i], 1));
    }

    auto attn         = ggml_nn_attention_ext(ctx, qkv[0], qkv[1], qkv[2], x_block->num_heads);  // [N, n_context + n_token, hidden_size]
    attn              = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));                     // [n_context + n_token, N, hidden_size]
    auto context_attn = ggml_view_3d(ctx,
                                     attn,
                                     attn->ne[0],
                                     attn->ne[1],
                                     context->ne[1],
                                     attn->nb[1],
                                     attn->nb[2],
                                     0);                                              // [n_context, N, hidden_size]
    context_attn      = ggml_cont(ctx, ggml_permute(ctx, context_attn, 0, 2, 1, 3));  // [N, n_context, hidden_size]
    auto x_attn       = ggml_view_3d(ctx,
                                     attn,
                                     attn->ne[0],
                                     attn->ne[1],
                                     x->ne[1],
                                     attn->nb[1],
                                     attn->nb[2],
                                     attn->nb[2] * context->ne[1]);             // [n_token, N, hidden_size]
    x_attn            = ggml_cont(ctx, ggml_permute(ctx, x_attn, 0, 2, 1, 3));  // [N, n_token, hidden_size]

    if (!context_block->pre_only) {
        context = context_block->post_attention(ctx,
                                                context_attn,
                                                context_intermediates[0],
                                                context_intermediates[1],
                                                context_intermediates[2],
                                                context_intermediates[3],
                                                context_intermediates[4]);
    } else {
        context = NULL;
    }

    x = x_block->post_attention(ctx,
                                x_attn,
                                x_intermediates[0],
                                x_intermediates[1],
                                x_intermediates[2],
                                x_intermediates[3],
                                x_intermediates[4]);

    return {context, x};
}

struct JointBlock : public GGMLBlock {
public:
    JointBlock(int64_t hidden_size,
               int64_t num_heads,
               float mlp_ratio = 4.0,
               bool qkv_bias   = false,
               bool pre_only   = false) {
        // qk_norm is always Flase
        blocks["context_block"] = std::shared_ptr<GGMLBlock>(new DismantledBlock(hidden_size, num_heads, mlp_ratio, qkv_bias, pre_only));
        blocks["x_block"]       = std::shared_ptr<GGMLBlock>(new DismantledBlock(hidden_size, num_heads, mlp_ratio, qkv_bias, false));
    }

    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(struct ggml_context* ctx,
                                                                struct ggml_tensor* context,
                                                                struct ggml_tensor* x,
                                                                struct ggml_tensor* c) {
        auto context_block = std::dynamic_pointer_cast<DismantledBlock>(blocks["context_block"]);
        auto x_block       = std::dynamic_pointer_cast<DismantledBlock>(blocks["x_block"]);

        return block_mixing(ctx, context, x, c, context_block, x_block);
    }
};

struct FinalLayer : public GGMLBlock {
    // The final layer of DiT.
public:
    FinalLayer(int64_t hidden_size,
               int64_t patch_size,
               int64_t out_channels) {
        // total_out_channels is always None
        blocks["norm_final"]         = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
        blocks["linear"]             = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, patch_size * patch_size * out_channels));
        blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 2 * hidden_size));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        // return: [N, n_token, patch_size * patch_size * out_channels]
        auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
        auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        auto m = adaLN_modulation_1->forward(ctx, ggml_silu(ctx, c));  // [N, 2 * hidden_size]
        m      = ggml_reshape_3d(ctx, m, c->ne[0], 2, c->ne[1]);       // [N, 2, hidden_size]
        m      = ggml_cont(ctx, ggml_permute(ctx, m, 0, 2, 1, 3));     // [2, N, hidden_size]

        int64_t offset = m->nb[1] * m->ne[1];
        auto shift     = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 0);  // [N, hidden_size]
        auto scale     = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 1);  // [N, hidden_size]

        x = modulate(ctx, norm_final->forward(ctx, x), shift, scale);
        x = linear->forward(ctx, x);

        return x;
    }
};

struct MMDiT : public GGMLBlock {
    // Diffusion model with a Transformer backbone.
protected:
    SDVersion version          = VERSION_3_2B;
    int64_t input_size         = -1;
    int64_t patch_size         = 2;
    int64_t in_channels        = 16;
    int64_t depth              = 24;
    float mlp_ratio            = 4.0f;
    int64_t adm_in_channels    = 2048;
    int64_t out_channels       = 16;
    int64_t pos_embed_max_size = 192;
    int64_t num_patchs         = 36864;  // 192 * 192
    int64_t context_size       = 4096;
    int64_t hidden_size;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["pos_embed"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, num_patchs, 1);
    }

public:
    MMDiT(SDVersion version = VERSION_3_2B)
        : version(version) {
        // input_size is always None
        // learn_sigma is always False
        // register_length is alwalys 0
        // rmsnorm is alwalys False
        // scale_mod_only is alwalys False
        // swiglu is alwalys False
        // qk_norm is always None
        // qkv_bias is always True
        // context_processor_layers is always None
        // pos_embed_scaling_factor is not used
        // pos_embed_offset is not used
        // context_embedder_config is always {'target': 'torch.nn.Linear', 'params': {'in_features': 4096, 'out_features': 1536}}
        if (version == VERSION_3_2B) {
            input_size         = -1;
            patch_size         = 2;
            in_channels        = 16;
            depth              = 24;
            mlp_ratio          = 4.0f;
            adm_in_channels    = 2048;
            out_channels       = 16;
            pos_embed_max_size = 192;
            num_patchs         = 36864;  // 192 * 192
            context_size       = 4096;
        }
        int64_t default_out_channels = in_channels;
        hidden_size                  = 64 * depth;
        int64_t num_heads            = depth;

        blocks["x_embedder"] = std::shared_ptr<GGMLBlock>(new PatchEmbed(input_size, patch_size, in_channels, hidden_size, true));
        blocks["t_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedder(hidden_size));

        if (adm_in_channels != -1) {
            blocks["y_embedder"] = std::shared_ptr<GGMLBlock>(new VectorEmbedder(adm_in_channels, hidden_size));
        }

        blocks["context_embedder"] = std::shared_ptr<GGMLBlock>(new Linear(4096, 1536));

        for (int i = 0; i < depth; i++) {
            blocks["joint_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new JointBlock(hidden_size,
                                                                                                    num_heads,
                                                                                                    mlp_ratio,
                                                                                                    true,
                                                                                                    i == depth - 1));
        }

        blocks["final_layer"] = std::shared_ptr<GGMLBlock>(new FinalLayer(hidden_size, patch_size, out_channels));
    }

    struct ggml_tensor* cropped_pos_embed(struct ggml_context* ctx,
                                          int64_t h,
                                          int64_t w) {
        auto pos_embed = params["pos_embed"];

        h = (h + 1) / patch_size;
        w = (w + 1) / patch_size;

        GGML_ASSERT(h <= pos_embed_max_size && h > 0);
        GGML_ASSERT(w <= pos_embed_max_size && w > 0);

        int64_t top  = (pos_embed_max_size - h) / 2;
        int64_t left = (pos_embed_max_size - w) / 2;

        auto spatial_pos_embed = ggml_reshape_3d(ctx, pos_embed, hidden_size, pos_embed_max_size, pos_embed_max_size);

        // spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = ggml_view_3d(ctx,
                                         spatial_pos_embed,
                                         hidden_size,
                                         pos_embed_max_size,
                                         h,
                                         spatial_pos_embed->nb[1],
                                         spatial_pos_embed->nb[2],
                                         spatial_pos_embed->nb[2] * top);                      // [h, pos_embed_max_size, hidden_size]
        spatial_pos_embed = ggml_cont(ctx, ggml_permute(ctx, spatial_pos_embed, 0, 2, 1, 3));  // [pos_embed_max_size, h, hidden_size]
        spatial_pos_embed = ggml_view_3d(ctx,
                                         spatial_pos_embed,
                                         hidden_size,
                                         h,
                                         w,
                                         spatial_pos_embed->nb[1],
                                         spatial_pos_embed->nb[2],
                                         spatial_pos_embed->nb[2] * left);                     // [w, h, hidden_size]
        spatial_pos_embed = ggml_cont(ctx, ggml_permute(ctx, spatial_pos_embed, 0, 2, 1, 3));  // [h, w, hidden_size]
        spatial_pos_embed = ggml_reshape_3d(ctx, spatial_pos_embed, hidden_size, h * w, 1);    // [1, h*w, hidden_size]
        return spatial_pos_embed;
    }

    struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                                   struct ggml_tensor* x,
                                   int64_t h,
                                   int64_t w) {
        // x: [N, H*W, patch_size * patch_size * C]
        // return: [N, C, H, W]
        int64_t n = x->ne[2];
        int64_t c = out_channels;
        int64_t p = patch_size;
        h         = (h + 1) / p;
        w         = (w + 1) / p;

        GGML_ASSERT(h * w == x->ne[1]);

        x = ggml_reshape_4d(ctx, x, c, p * p, w * h, n);       // [N, H*W, P*P, C]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));  // [N, C, H*W, P*P]
        x = ggml_reshape_4d(ctx, x, p, p, w, h * c * n);       // [N*C*H, W, P, P]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*H, P, W, P]
        x = ggml_reshape_4d(ctx, x, p * w, p * h, c, n);       // [N, C, H*P, W*P]
        return x;
    }

    struct ggml_tensor* forward_core_with_concat(struct ggml_context* ctx,
                                                 struct ggml_tensor* x,
                                                 struct ggml_tensor* c_mod,
                                                 struct ggml_tensor* context) {
        // x: [N, H*W, hidden_size]
        // context: [N, n_context, d_context]
        // c: [N, hidden_size]
        // return: [N, N*W, patch_size * patch_size * out_channels]
        auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

        for (int i = 0; i < depth; i++) {
            auto block = std::dynamic_pointer_cast<JointBlock>(blocks["joint_blocks." + std::to_string(i)]);

            auto context_x = block->forward(ctx, context, x, c_mod);
            context        = context_x.first;
            x              = context_x.second;
        }

        x = final_layer->forward(ctx, x, c_mod);  // (N, T, patch_size ** 2 * out_channels)

        return x;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* t,
                                struct ggml_tensor* y       = NULL,
                                struct ggml_tensor* context = NULL) {
        // Forward pass of DiT.
        // x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        // t: (N,) tensor of diffusion timesteps
        // y: (N, adm_in_channels) tensor of class labels
        // context: (N, L, D)
        // return: (N, C, H, W)
        auto x_embedder = std::dynamic_pointer_cast<PatchEmbed>(blocks["x_embedder"]);
        auto t_embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);

        int64_t w = x->ne[0];
        int64_t h = x->ne[1];

        auto patch_embed = x_embedder->forward(ctx, x);            // [N, H*W, hidden_size]
        auto pos_embed   = cropped_pos_embed(ctx, h, w);           // [1, H*W, hidden_size]
        x                = ggml_add(ctx, patch_embed, pos_embed);  // [N, H*W, hidden_size]

        auto c = t_embedder->forward(ctx, t);  // [N, hidden_size]
        if (y != NULL && adm_in_channels != -1) {
            auto y_embedder = std::dynamic_pointer_cast<VectorEmbedder>(blocks["y_embedder"]);

            y = y_embedder->forward(ctx, y);  // [N, hidden_size]
            c = ggml_add(ctx, c, y);
        }

        if (context != NULL) {
            auto context_embedder = std::dynamic_pointer_cast<Linear>(blocks["context_embedder"]);

            context = context_embedder->forward(ctx, context);  // [N, L, D] aka [N, L, 1536]
        }

        x = forward_core_with_concat(ctx, x, c, context);  // (N, H*W, patch_size ** 2 * out_channels)

        x = unpatchify(ctx, x, h, w);  // [N, C, H, W]

        return x;
    }
};

struct MMDiTRunner : public GGMLRunner {
    MMDiT mmdit;

    MMDiTRunner(ggml_backend_t backend,
                ggml_type wtype,
                SDVersion version = VERSION_3_2B)
        : GGMLRunner(backend, wtype), mmdit(version) {
        mmdit.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "mmdit";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        mmdit.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* y) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, MMDIT_GRAPH_SIZE, false);

        x         = to_backend(x);
        context   = to_backend(context);
        y         = to_backend(y);
        timesteps = to_backend(timesteps);

        struct ggml_tensor* out = mmdit.forward(compute_ctx,
                                                x,
                                                timesteps,
                                                y,
                                                context);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* y,
                 struct ggml_tensor** output     = NULL,
                 struct ggml_context* output_ctx = NULL) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 154, 4096]) or [1, max_position, hidden_size]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, context, y);
        };

        GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }

    void test() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        struct ggml_context* work_ctx = ggml_init(params);
        GGML_ASSERT(work_ctx != NULL);

        {
            // cpu f16: pass
            // cpu f32: pass
            // cuda f16: pass
            // cuda f32: pass
            auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 128, 128, 16, 1);
            std::vector<float> timesteps_vec(1, 999.f);
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            ggml_set_f32(x, 0.01f);
            // print_ggml_tensor(x);

            auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 4096, 154, 1);
            ggml_set_f32(context, 0.01f);
            // print_ggml_tensor(context);

            auto y = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 2048, 1);
            ggml_set_f32(y, 0.01f);
            // print_ggml_tensor(y);

            struct ggml_tensor* out = NULL;

            int t0 = ggml_time_ms();
            compute(8, x, timesteps, context, y, &out, work_ctx);
            int t1 = ggml_time_ms();

            print_ggml_tensor(out);
            LOG_DEBUG("mmdit test done in %dms", t1 - t0);
        }
    }

    static void load_from_file_and_test(const std::string& file_path) {
        // ggml_backend_t backend    = ggml_backend_cuda_init(0);
        ggml_backend_t backend             = ggml_backend_cpu_init();
        ggml_type model_data_type          = GGML_TYPE_F16;
        std::shared_ptr<MMDiTRunner> mmdit = std::shared_ptr<MMDiTRunner>(new MMDiTRunner(backend, model_data_type));
        {
            LOG_INFO("loading from '%s'", file_path.c_str());

            mmdit->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            mmdit->get_param_tensors(tensors, "model.diffusion_model");

            ModelLoader model_loader;
            if (!model_loader.init_from_file(file_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            bool success = model_loader.load_tensors(tensors, backend);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("mmdit model loaded");
        }
        mmdit->test();
    }
};

#endif