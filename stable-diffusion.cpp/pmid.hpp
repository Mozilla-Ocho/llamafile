#ifndef __PMI_HPP__
#define __PMI_HPP__

#include "ggml_extend.hpp"

#include "clip.hpp"
#include "lora.hpp"

struct FuseBlock : public GGMLBlock {
    // network hparams
    int in_dim;
    int out_dim;
    int hidden_dim;
    bool use_residue;

public:
    FuseBlock(int i_d, int o_d, int h_d, bool use_residue = true)
        : in_dim(i_d), out_dim(o_d), hidden_dim(h_d), use_residue(use_residue) {
        blocks["fc1"]       = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
        blocks["fc2"]       = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, out_dim, true));
        blocks["layernorm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(in_dim));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]

        auto fc1        = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2        = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);
        auto layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["layernorm"]);

        struct ggml_tensor* r = x;
        // x = ggml_nn_layer_norm(ctx, x, ln_w, ln_b);
        x = layer_norm->forward(ctx, x);
        // x = ggml_add(ctx, ggml_mul_mat(ctx, fc1_w, x),  fc1_b);
        x = fc1->forward(ctx, x);
        x = ggml_gelu_inplace(ctx, x);
        x = fc2->forward(ctx, x);
        // x = ggml_add(ctx, ggml_mul_mat(ctx, fc2_w, x),  fc2_b);
        if (use_residue)
            x = ggml_add(ctx, x, r);
        return x;
    }
};

struct FuseModule : public GGMLBlock {
    // network hparams
    int embed_dim;

public:
    FuseModule(int imb_d)
        : embed_dim(imb_d) {
        blocks["mlp1"]       = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d * 2, imb_d, imb_d, false));
        blocks["mlp2"]       = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d, imb_d, imb_d, true));
        blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(embed_dim));
    }

    struct ggml_tensor* fuse_fn(struct ggml_context* ctx,
                                struct ggml_tensor* prompt_embeds,
                                struct ggml_tensor* id_embeds) {
        auto mlp1       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp1"]);
        auto mlp2       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp2"]);
        auto layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);

        auto prompt_embeds0 = ggml_cont(ctx, ggml_permute(ctx, prompt_embeds, 2, 0, 1, 3));
        auto id_embeds0     = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        // concat is along dim 2
        auto stacked_id_embeds = ggml_concat(ctx, prompt_embeds0, id_embeds0);
        stacked_id_embeds      = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 1, 2, 0, 3));

        // stacked_id_embeds = mlp1.forward(ctx, stacked_id_embeds);
        // stacked_id_embeds = ggml_add(ctx, stacked_id_embeds, prompt_embeds);
        // stacked_id_embeds = mlp2.forward(ctx, stacked_id_embeds);
        // stacked_id_embeds = ggml_nn_layer_norm(ctx, stacked_id_embeds, ln_w, ln_b);

        stacked_id_embeds = mlp1->forward(ctx, stacked_id_embeds);
        stacked_id_embeds = ggml_add(ctx, stacked_id_embeds, prompt_embeds);
        stacked_id_embeds = mlp2->forward(ctx, stacked_id_embeds);
        stacked_id_embeds = layer_norm->forward(ctx, stacked_id_embeds);

        return stacked_id_embeds;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* prompt_embeds,
                                struct ggml_tensor* id_embeds,
                                struct ggml_tensor* class_tokens_mask,
                                struct ggml_tensor* class_tokens_mask_pos,
                                struct ggml_tensor* left,
                                struct ggml_tensor* right) {
        // x: [N, channels, h, w]

        struct ggml_tensor* valid_id_embeds = id_embeds;
        // # slice out the image token embeddings
        // print_ggml_tensor(class_tokens_mask_pos, false);
        ggml_set_name(class_tokens_mask_pos, "class_tokens_mask_pos");
        ggml_set_name(prompt_embeds, "prompt_embeds");
        // print_ggml_tensor(valid_id_embeds, true, "valid_id_embeds");
        // print_ggml_tensor(class_tokens_mask_pos, true, "class_tokens_mask_pos");
        struct ggml_tensor* image_token_embeds = ggml_get_rows(ctx, prompt_embeds, class_tokens_mask_pos);
        ggml_set_name(image_token_embeds, "image_token_embeds");
        struct ggml_tensor* stacked_id_embeds = fuse_fn(ctx, image_token_embeds, valid_id_embeds);

        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        if (left && right) {
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        } else if (left) {
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
        } else if (right) {
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        }
        stacked_id_embeds                         = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        class_tokens_mask                         = ggml_cont(ctx, ggml_transpose(ctx, class_tokens_mask));
        class_tokens_mask                         = ggml_repeat(ctx, class_tokens_mask, prompt_embeds);
        prompt_embeds                             = ggml_mul(ctx, prompt_embeds, class_tokens_mask);
        struct ggml_tensor* updated_prompt_embeds = ggml_add(ctx, prompt_embeds, stacked_id_embeds);
        ggml_set_name(updated_prompt_embeds, "updated_prompt_embeds");
        return updated_prompt_embeds;
    }
};

struct PhotoMakerIDEncoderBlock : public CLIPVisionModelProjection {
    PhotoMakerIDEncoderBlock()
        : CLIPVisionModelProjection(OPENAI_CLIP_VIT_L_14) {
        blocks["visual_projection_2"] = std::shared_ptr<GGMLBlock>(new Linear(1024, 1280, false));
        blocks["fuse_module"]         = std::shared_ptr<GGMLBlock>(new FuseModule(2048));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* id_pixel_values,
                                struct ggml_tensor* prompt_embeds,
                                struct ggml_tensor* class_tokens_mask,
                                struct ggml_tensor* class_tokens_mask_pos,
                                struct ggml_tensor* left,
                                struct ggml_tensor* right) {
        // x: [N, channels, h, w]
        auto vision_model        = std::dynamic_pointer_cast<CLIPVisionModel>(blocks["vision_model"]);
        auto visual_projection   = std::dynamic_pointer_cast<CLIPProjection>(blocks["visual_projection"]);
        auto visual_projection_2 = std::dynamic_pointer_cast<Linear>(blocks["visual_projection_2"]);
        auto fuse_module         = std::dynamic_pointer_cast<FuseModule>(blocks["fuse_module"]);

        struct ggml_tensor* shared_id_embeds = vision_model->forward(ctx, id_pixel_values);          // [N, hidden_size]
        struct ggml_tensor* id_embeds        = visual_projection->forward(ctx, shared_id_embeds);    // [N, proj_dim(768)]
        struct ggml_tensor* id_embeds_2      = visual_projection_2->forward(ctx, shared_id_embeds);  // [N, 1280]

        id_embeds   = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        id_embeds_2 = ggml_cont(ctx, ggml_permute(ctx, id_embeds_2, 2, 0, 1, 3));

        id_embeds = ggml_concat(ctx, id_embeds, id_embeds_2);  // [batch_size, seq_length, 1, 2048] check whether concat at dim 2 is right
        id_embeds = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 1, 2, 0, 3));

        struct ggml_tensor* updated_prompt_embeds = fuse_module->forward(ctx,
                                                                         prompt_embeds,
                                                                         id_embeds,
                                                                         class_tokens_mask,
                                                                         class_tokens_mask_pos,
                                                                         left, right);
        return updated_prompt_embeds;
    }
};

struct PhotoMakerIDEncoder : public GGMLModule {
public:
    SDVersion version = VERSION_XL;
    PhotoMakerIDEncoderBlock id_encoder;
    float style_strength;

    std::vector<float> ctm;
    std::vector<ggml_fp16_t> ctmf16;
    std::vector<int> ctmpos;

    std::vector<ggml_fp16_t> zeros_left_16;
    std::vector<float> zeros_left;
    std::vector<ggml_fp16_t> zeros_right_16;
    std::vector<float> zeros_right;

public:
    PhotoMakerIDEncoder(ggml_backend_t backend, ggml_type wtype, SDVersion version = VERSION_XL, float sty = 20.f)
        : GGMLModule(backend, wtype),
          version(version),
          style_strength(sty) {
        id_encoder.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "pmid";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        id_encoder.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(  // struct ggml_allocr* allocr,
        struct ggml_tensor* id_pixel_values,
        struct ggml_tensor* prompt_embeds,
        std::vector<bool>& class_tokens_mask) {
        ctm.clear();
        ctmf16.clear();
        ctmpos.clear();
        zeros_left.clear();
        zeros_left_16.clear();
        zeros_right.clear();
        zeros_right_16.clear();

        ggml_context* ctx0 = compute_ctx;

        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        int64_t hidden_size = prompt_embeds->ne[0];
        int64_t seq_length  = prompt_embeds->ne[1];
        ggml_type type      = GGML_TYPE_F32;

        struct ggml_tensor* class_tokens_mask_d = ggml_new_tensor_1d(ctx0, type, class_tokens_mask.size());

        struct ggml_tensor* id_pixel_values_d = to_backend(id_pixel_values);
        struct ggml_tensor* prompt_embeds_d   = to_backend(prompt_embeds);

        struct ggml_tensor* left  = NULL;
        struct ggml_tensor* right = NULL;
        for (int i = 0; i < class_tokens_mask.size(); i++) {
            if (class_tokens_mask[i]) {
                ctm.push_back(0.f);                        // here use 0.f instead of 1.f to make a scale mask
                ctmf16.push_back(ggml_fp32_to_fp16(0.f));  // here use 0.f instead of 1.f to make a scale mask
                ctmpos.push_back(i);
            } else {
                ctm.push_back(1.f);                        // here use 1.f instead of 0.f to make a scale mask
                ctmf16.push_back(ggml_fp32_to_fp16(1.f));  // here use 0.f instead of 1.f to make a scale mask
            }
        }
        if (ctmpos[0] > 0) {
            left = ggml_new_tensor_3d(ctx0, type, hidden_size, 1, ctmpos[0]);
        }
        if (ctmpos[ctmpos.size() - 1] < seq_length - 1) {
            right = ggml_new_tensor_3d(ctx0, type,
                                       hidden_size, 1, seq_length - ctmpos[ctmpos.size() - 1] - 1);
        }
        struct ggml_tensor* class_tokens_mask_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctmpos.size());

        {
            if (type == GGML_TYPE_F16)
                set_backend_tensor_data(class_tokens_mask_d, ctmf16.data());
            else
                set_backend_tensor_data(class_tokens_mask_d, ctm.data());
            set_backend_tensor_data(class_tokens_mask_pos, ctmpos.data());
            if (left) {
                if (type == GGML_TYPE_F16) {
                    for (int i = 0; i < ggml_nelements(left); ++i)
                        zeros_left_16.push_back(ggml_fp32_to_fp16(0.f));
                    set_backend_tensor_data(left, zeros_left_16.data());
                } else {
                    for (int i = 0; i < ggml_nelements(left); ++i)
                        zeros_left.push_back(0.f);
                    set_backend_tensor_data(left, zeros_left.data());
                }
            }
            if (right) {
                if (type == GGML_TYPE_F16) {
                    for (int i = 0; i < ggml_nelements(right); ++i)
                        zeros_right_16.push_back(ggml_fp32_to_fp16(0.f));
                    set_backend_tensor_data(right, zeros_right_16.data());
                } else {
                    for (int i = 0; i < ggml_nelements(right); ++i)
                        zeros_right.push_back(0.f);
                    set_backend_tensor_data(right, zeros_right.data());
                }
            }
        }
        struct ggml_tensor* updated_prompt_embeds = id_encoder.forward(ctx0,
                                                                       id_pixel_values_d,
                                                                       prompt_embeds_d,
                                                                       class_tokens_mask_d,
                                                                       class_tokens_mask_pos,
                                                                       left, right);
        ggml_build_forward_expand(gf, updated_prompt_embeds);

        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* id_pixel_values,
                 struct ggml_tensor* prompt_embeds,
                 std::vector<bool>& class_tokens_mask,
                 struct ggml_tensor** updated_prompt_embeds,
                 ggml_context* output_ctx) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            // return build_graph(compute_allocr, id_pixel_values, prompt_embeds, class_tokens_mask);
            return build_graph(id_pixel_values, prompt_embeds, class_tokens_mask);
        };

        // GGMLModule::compute(get_graph, n_threads, updated_prompt_embeds);
        GGMLModule::compute(get_graph, n_threads, true, updated_prompt_embeds, output_ctx);
    }
};

#endif  // __PMI_HPP__
