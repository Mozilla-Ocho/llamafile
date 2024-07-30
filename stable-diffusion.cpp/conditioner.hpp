#ifndef __CONDITIONER_HPP__
#define __CONDITIONER_HPP__

#include "clip.hpp"
#include "t5.hpp"

struct SDCondition {
    struct ggml_tensor* c_crossattn = NULL;  // aka context
    struct ggml_tensor* c_vector    = NULL;  // aka y
    struct ggml_tensor* c_concat    = NULL;

    SDCondition() = default;
    SDCondition(struct ggml_tensor* c_crossattn, struct ggml_tensor* c_vector, struct ggml_tensor* c_concat) :
    c_crossattn(c_crossattn), c_vector(c_vector), c_concat(c_concat) {}
};

struct Conditioner {
    virtual SDCondition get_learned_condition(ggml_context* work_ctx,
                                              int n_threads,
                                              const std::string& text,
                                              int clip_skip,
                                              int width,
                                              int height,
                                              int adm_in_channels        = -1,
                                              bool force_zero_embeddings = false)                                             = 0;
    virtual void alloc_params_buffer()                                                                                        = 0;
    virtual void free_params_buffer()                                                                                         = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors)                                       = 0;
    virtual size_t get_params_buffer_size()                                                                                   = 0;
    virtual std::tuple<SDCondition, std::vector<bool>> get_learned_condition_with_trigger(ggml_context* work_ctx,
                                                                                          int n_threads,
                                                                                          const std::string& text,
                                                                                          int clip_skip,
                                                                                          int width,
                                                                                          int height,
                                                                                          int num_input_imgs,
                                                                                          int adm_in_channels        = -1,
                                                                                          bool force_zero_embeddings = false) = 0;
    virtual std::string remove_trigger_from_prompt(ggml_context* work_ctx,
                                                   const std::string& prompt)                                                 = 0;
};

// ldm.modules.encoders.modules.FrozenCLIPEmbedder
// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/sd_hijack_clip.py#L283
struct FrozenCLIPEmbedderWithCustomWords : public Conditioner {
    SDVersion version = VERSION_1_x;
    CLIPTokenizer tokenizer;
    ggml_type wtype;
    std::shared_ptr<CLIPTextModelRunner> text_model;
    std::shared_ptr<CLIPTextModelRunner> text_model2;

    std::string trigger_word = "img";  // should be user settable
    std::string embd_dir;
    int32_t num_custom_embeddings = 0;
    std::vector<uint8_t> token_embed_custom;
    std::vector<std::string> readed_embeddings;

    FrozenCLIPEmbedderWithCustomWords(ggml_backend_t backend,
                                      ggml_type wtype,
                                      const std::string& embd_dir,
                                      SDVersion version = VERSION_1_x,
                                      int clip_skip     = -1)
        : version(version), tokenizer(version == VERSION_2_x ? 0 : 49407), embd_dir(embd_dir), wtype(wtype) {
        if (clip_skip <= 0) {
            clip_skip = 1;
            if (version == VERSION_2_x || version == VERSION_XL) {
                clip_skip = 2;
            }
        }
        if (version == VERSION_1_x) {
            text_model = std::make_shared<CLIPTextModelRunner>(backend, wtype, OPENAI_CLIP_VIT_L_14, clip_skip);
        } else if (version == VERSION_2_x) {
            text_model = std::make_shared<CLIPTextModelRunner>(backend, wtype, OPEN_CLIP_VIT_H_14, clip_skip);
        } else if (version == VERSION_XL) {
            text_model  = std::make_shared<CLIPTextModelRunner>(backend, wtype, OPENAI_CLIP_VIT_L_14, clip_skip, false);
            text_model2 = std::make_shared<CLIPTextModelRunner>(backend, wtype, OPEN_CLIP_VIT_BIGG_14, clip_skip, false);
        }
    }

    void set_clip_skip(int clip_skip) {
        text_model->set_clip_skip(clip_skip);
        if (version == VERSION_XL) {
            text_model2->set_clip_skip(clip_skip);
        }
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        text_model->get_param_tensors(tensors, "cond_stage_model.transformer.text_model");
        if (version == VERSION_XL) {
            text_model2->get_param_tensors(tensors, "cond_stage_model.1.transformer.text_model");
        }
    }

    void alloc_params_buffer() {
        text_model->alloc_params_buffer();
        if (version == VERSION_XL) {
            text_model2->alloc_params_buffer();
        }
    }

    void free_params_buffer() {
        text_model->free_params_buffer();
        if (version == VERSION_XL) {
            text_model2->free_params_buffer();
        }
    }

    size_t get_params_buffer_size() {
        size_t buffer_size = text_model->get_params_buffer_size();
        if (version == VERSION_XL) {
            buffer_size += text_model2->get_params_buffer_size();
        }
        return buffer_size;
    }

    bool load_embedding(std::string embd_name, std::string embd_path, std::vector<int32_t>& bpe_tokens) {
        // the order matters
        ModelLoader model_loader;
        if (!model_loader.init_from_file(embd_path)) {
            LOG_ERROR("embedding '%s' failed", embd_name.c_str());
            return false;
        }
        if (std::find(readed_embeddings.begin(), readed_embeddings.end(), embd_name) != readed_embeddings.end()) {
            LOG_DEBUG("embedding already read in: %s", embd_name.c_str());
            return true;
        }
        struct ggml_init_params params;
        params.mem_size               = 10 * 1024 * 1024;  // max for custom embeddings 10 MB
        params.mem_buffer             = NULL;
        params.no_alloc               = false;
        struct ggml_context* embd_ctx = ggml_init(params);
        struct ggml_tensor* embd      = NULL;
        int64_t hidden_size           = text_model->model.hidden_size;
        auto on_load                  = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) {
            if (tensor_storage.ne[0] != hidden_size) {
                LOG_DEBUG("embedding wrong hidden size, got %i, expected %i", tensor_storage.ne[0], hidden_size);
                return false;
            }
            embd        = ggml_new_tensor_2d(embd_ctx, wtype, hidden_size, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
            *dst_tensor = embd;
            return true;
        };
        model_loader.load_tensors(on_load, NULL);
        readed_embeddings.push_back(embd_name);
        token_embed_custom.resize(token_embed_custom.size() + ggml_nbytes(embd));
        memcpy((void*)(token_embed_custom.data() + num_custom_embeddings * hidden_size * ggml_type_size(wtype)),
               embd->data,
               ggml_nbytes(embd));
        for (int i = 0; i < embd->ne[1]; i++) {
            bpe_tokens.push_back(text_model->model.vocab_size + num_custom_embeddings);
            // LOG_DEBUG("new custom token: %i", text_model.vocab_size + num_custom_embeddings);
            num_custom_embeddings++;
        }
        LOG_DEBUG("embedding '%s' applied, custom embeddings: %i", embd_name.c_str(), num_custom_embeddings);
        return true;
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<bool>>
    tokenize_with_trigger_token(std::string text,
                                int num_input_imgs,
                                int32_t image_token,
                                bool padding = false) {
        return tokenize_with_trigger_token(text, num_input_imgs, image_token,
                                           text_model->model.n_token, padding);
    }

    std::vector<int> convert_token_to_id(std::string text) {
        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            size_t word_end       = str.find(",");
            std::string embd_name = word_end == std::string::npos ? str : str.substr(0, word_end);
            embd_name             = trim(embd_name);
            std::string embd_path = get_full_path(embd_dir, embd_name + ".pt");
            if (embd_path.size() == 0) {
                embd_path = get_full_path(embd_dir, embd_name + ".ckpt");
            }
            if (embd_path.size() == 0) {
                embd_path = get_full_path(embd_dir, embd_name + ".safetensors");
            }
            if (embd_path.size() > 0) {
                if (load_embedding(embd_name, embd_path, bpe_tokens)) {
                    if (word_end != std::string::npos) {
                        str = str.substr(word_end);
                    } else {
                        str = "";
                    }
                    return true;
                }
            }
            return false;
        };
        std::vector<int> curr_tokens = tokenizer.encode(text, on_new_token_cb);
        return curr_tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        return tokenizer.decode(tokens);
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<bool>>
    tokenize_with_trigger_token(std::string text,
                                int num_input_imgs,
                                int32_t image_token,
                                size_t max_length = 0,
                                bool padding      = false) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            size_t word_end       = str.find(",");
            std::string embd_name = word_end == std::string::npos ? str : str.substr(0, word_end);
            embd_name             = trim(embd_name);
            std::string embd_path = get_full_path(embd_dir, embd_name + ".pt");
            if (embd_path.size() == 0) {
                embd_path = get_full_path(embd_dir, embd_name + ".ckpt");
            }
            if (embd_path.size() == 0) {
                embd_path = get_full_path(embd_dir, embd_name + ".safetensors");
            }
            if (embd_path.size() > 0) {
                if (load_embedding(embd_name, embd_path, bpe_tokens)) {
                    if (word_end != std::string::npos) {
                        str = str.substr(word_end);
                    } else {
                        str = "";
                    }
                    return true;
                }
            }
            return false;
        };

        std::vector<int> tokens;
        std::vector<float> weights;
        std::vector<bool> class_token_mask;
        int32_t class_idx = -1, tokens_acc = 0;
        for (const auto& item : parsed_attention) {
            std::vector<int> class_token_index;
            std::vector<int> clean_input_ids;
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            // printf(" %s: %f \n", curr_text.c_str(), curr_weight);
            std::vector<int> curr_tokens = tokenizer.encode(curr_text, on_new_token_cb);
            int32_t clean_index          = 0;
            for (uint32_t i = 0; i < curr_tokens.size(); i++) {
                int token_id = curr_tokens[i];
                if (token_id == image_token)
                    class_token_index.push_back(clean_index - 1);
                else {
                    clean_input_ids.push_back(token_id);
                    clean_index++;
                }
            }
            // GGML_ASSERT(class_token_index.size() == 1); // PhotoMaker currently does not support multiple
            //     trigger words in a single prompt.
            if (class_token_index.size() == 1) {
                // Expand the class word token and corresponding mask
                int class_token = clean_input_ids[class_token_index[0]];
                class_idx       = tokens_acc + class_token_index[0];
                std::vector<int> clean_input_ids_tmp;
                for (uint32_t i = 0; i < class_token_index[0]; i++)
                    clean_input_ids_tmp.push_back(clean_input_ids[i]);
                for (uint32_t i = 0; i < num_input_imgs; i++)
                    clean_input_ids_tmp.push_back(class_token);
                for (uint32_t i = class_token_index[0] + 1; i < clean_input_ids.size(); i++)
                    clean_input_ids_tmp.push_back(clean_input_ids[i]);
                clean_input_ids.clear();
                clean_input_ids = clean_input_ids_tmp;
            }
            tokens_acc += clean_index;
            tokens.insert(tokens.end(), clean_input_ids.begin(), clean_input_ids.end());
            weights.insert(weights.end(), clean_input_ids.size(), curr_weight);
        }
        tokens.insert(tokens.begin(), tokenizer.BOS_TOKEN_ID);
        weights.insert(weights.begin(), 1.0);

        tokenizer.pad_tokens(tokens, weights, max_length, padding);

        for (uint32_t i = 0; i < tokens.size(); i++) {
            if (class_idx + 1 <= i && i < class_idx + 1 + num_input_imgs)
                class_token_mask.push_back(true);
            else
                class_token_mask.push_back(false);
        }

        // printf("[");
        // for (int i = 0; i < tokens.size(); i++) {
        //     printf("%d, ", class_token_mask[i] ? 1 : 0);
        // }
        // printf("]\n");

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return std::make_tuple(tokens, weights, class_token_mask);
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             bool padding = false) {
        return tokenize(text, text_model->model.n_token, padding);
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             size_t max_length = 0,
                                                             bool padding      = false) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            size_t word_end       = str.find(",");
            std::string embd_name = word_end == std::string::npos ? str : str.substr(0, word_end);
            embd_name             = trim(embd_name);
            std::string embd_path = get_full_path(embd_dir, embd_name + ".pt");
            if (embd_path.size() == 0) {
                embd_path = get_full_path(embd_dir, embd_name + ".ckpt");
            }
            if (embd_path.size() == 0) {
                embd_path = get_full_path(embd_dir, embd_name + ".safetensors");
            }
            if (embd_path.size() > 0) {
                if (load_embedding(embd_name, embd_path, bpe_tokens)) {
                    if (word_end != std::string::npos) {
                        str = str.substr(word_end);
                    } else {
                        str = "";
                    }
                    return true;
                }
            }
            return false;
        };

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer.encode(curr_text, on_new_token_cb);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        tokenizer.pad_tokens(tokens, weights, max_length, padding);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights};
    }

    SDCondition get_learned_condition_common(ggml_context* work_ctx,
                                             int n_threads,
                                             std::vector<int>& tokens,
                                             std::vector<float>& weights,
                                             int clip_skip,
                                             int width,
                                             int height,
                                             int adm_in_channels        = -1,
                                             bool force_zero_embeddings = false) {
        set_clip_skip(clip_skip);
        int64_t t0                               = ggml_time_ms();
        struct ggml_tensor* hidden_states        = NULL;  // [N, n_token, hidden_size]
        struct ggml_tensor* chunk_hidden_states  = NULL;  // [n_token, hidden_size] or [n_token, hidden_size + hidden_size2]
        struct ggml_tensor* chunk_hidden_states1 = NULL;  // [n_token, hidden_size]
        struct ggml_tensor* chunk_hidden_states2 = NULL;  // [n_token, hidden_size2]
        struct ggml_tensor* pooled               = NULL;
        std::vector<float> hidden_states_vec;

        size_t chunk_len   = 77;
        size_t chunk_count = tokens.size() / chunk_len;
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            std::vector<int> chunk_tokens(tokens.begin() + chunk_idx * chunk_len,
                                          tokens.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_weights(weights.begin() + chunk_idx * chunk_len,
                                             weights.begin() + (chunk_idx + 1) * chunk_len);

            auto input_ids                 = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
            struct ggml_tensor* input_ids2 = NULL;
            size_t max_token_idx           = 0;
            if (version == VERSION_XL) {
                auto it = std::find(chunk_tokens.begin(), chunk_tokens.end(), tokenizer.EOS_TOKEN_ID);
                if (it != chunk_tokens.end()) {
                    std::fill(std::next(it), chunk_tokens.end(), 0);
                }

                max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

                input_ids2 = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);

                // for (int i = 0; i < chunk_tokens.size(); i++) {
                //     printf("%d ", chunk_tokens[i]);
                // }
                // printf("\n");
            }

            {
                text_model->compute(n_threads,
                                    input_ids,
                                    num_custom_embeddings,
                                    token_embed_custom.data(),
                                    max_token_idx,
                                    false,
                                    &chunk_hidden_states1,
                                    work_ctx);
                if (version == VERSION_XL) {
                    text_model2->compute(n_threads,
                                         input_ids2,
                                         0,
                                         NULL,
                                         max_token_idx,
                                         false,
                                         &chunk_hidden_states2, work_ctx);
                    // concat
                    chunk_hidden_states = ggml_tensor_concat(work_ctx, chunk_hidden_states1, chunk_hidden_states2, 0);

                    if (chunk_idx == 0) {
                        text_model2->compute(n_threads,
                                             input_ids2,
                                             0,
                                             NULL,
                                             max_token_idx,
                                             true,
                                             &pooled,
                                             work_ctx);
                    }
                } else {
                    chunk_hidden_states = chunk_hidden_states1;
                }
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            ggml_tensor* result = ggml_dup_tensor(work_ctx, chunk_hidden_states);
            {
                float original_mean = ggml_tensor_mean(chunk_hidden_states);
                for (int i2 = 0; i2 < chunk_hidden_states->ne[2]; i2++) {
                    for (int i1 = 0; i1 < chunk_hidden_states->ne[1]; i1++) {
                        for (int i0 = 0; i0 < chunk_hidden_states->ne[0]; i0++) {
                            float value = ggml_tensor_get_f32(chunk_hidden_states, i0, i1, i2);
                            value *= chunk_weights[i1];
                            ggml_tensor_set_f32(result, value, i0, i1, i2);
                        }
                    }
                }
                float new_mean = ggml_tensor_mean(result);
                ggml_tensor_scale(result, (original_mean / new_mean));
            }
            if (force_zero_embeddings) {
                float* vec = (float*)result->data;
                for (int i = 0; i < ggml_nelements(result); i++) {
                    vec[i] = 0;
                }
            }
            hidden_states_vec.insert(hidden_states_vec.end(), (float*)result->data, ((float*)result->data) + ggml_nelements(result));
        }

        hidden_states = vector_to_ggml_tensor(work_ctx, hidden_states_vec);
        hidden_states = ggml_reshape_2d(work_ctx,
                                        hidden_states,
                                        chunk_hidden_states->ne[0],
                                        ggml_nelements(hidden_states) / chunk_hidden_states->ne[0]);

        ggml_tensor* vec = NULL;
        if (version == VERSION_XL) {
            int out_dim = 256;
            vec         = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, adm_in_channels);
            // [0:1280]
            size_t offset = 0;
            memcpy(vec->data, pooled->data, ggml_nbytes(pooled));
            offset += ggml_nbytes(pooled);

            // original_size_as_tuple
            float orig_width             = (float)width;
            float orig_height            = (float)height;
            std::vector<float> timesteps = {orig_height, orig_width};

            ggml_tensor* embed_view = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // crop_coords_top_left
            float crop_coord_top  = 0.f;
            float crop_coord_left = 0.f;
            timesteps             = {crop_coord_top, crop_coord_left};
            embed_view            = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // target_size_as_tuple
            float target_width  = (float)width;
            float target_height = (float)height;
            timesteps           = {target_height, target_width};
            embed_view          = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            GGML_ASSERT(offset == ggml_nbytes(vec));
        }
        // print_ggml_tensor(result);
        return SDCondition(hidden_states, vec, NULL);
    }

    std::tuple<SDCondition, std::vector<bool>>
    get_learned_condition_with_trigger(ggml_context* work_ctx,
                                       int n_threads,
                                       const std::string& text,
                                       int clip_skip,
                                       int width,
                                       int height,
                                       int num_input_imgs,
                                       int adm_in_channels        = -1,
                                       bool force_zero_embeddings = false) {
        auto image_tokens = convert_token_to_id(trigger_word);
        // if(image_tokens.size() == 1){
        //     printf(" image token id is: %d \n", image_tokens[0]);
        // }
        GGML_ASSERT(image_tokens.size() == 1);
        auto tokens_and_weights     = tokenize_with_trigger_token(text,
                                                                  num_input_imgs,
                                                                  image_tokens[0],
                                                                  true);
        std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
        std::vector<float>& weights = std::get<1>(tokens_and_weights);
        std::vector<bool>& clsm     = std::get<2>(tokens_and_weights);
        // printf("tokens: \n");
        // for(int i = 0; i < tokens.size(); ++i)
        //    printf("%d ", tokens[i]);
        // printf("\n");
        // printf("clsm: \n");
        // for(int i = 0; i < clsm.size(); ++i)
        //    printf("%d ", clsm[i]?1:0);
        // printf("\n");
        auto cond = get_learned_condition_common(work_ctx, n_threads, tokens, weights, clip_skip, width, height, adm_in_channels, force_zero_embeddings);
        return std::make_tuple(cond, clsm);
    }

    std::string remove_trigger_from_prompt(ggml_context* work_ctx,
                                           const std::string& prompt) {
        auto image_tokens = convert_token_to_id(trigger_word);
        GGML_ASSERT(image_tokens.size() == 1);
        auto tokens_and_weights  = tokenize(prompt, false);
        std::vector<int>& tokens = tokens_and_weights.first;
        auto it                  = std::find(tokens.begin(), tokens.end(), image_tokens[0]);
        GGML_ASSERT(it != tokens.end());  // prompt must have trigger word
        tokens.erase(it);
        return decode(tokens);
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const std::string& text,
                                      int clip_skip,
                                      int width,
                                      int height,
                                      int adm_in_channels        = -1,
                                      bool force_zero_embeddings = false) {
        auto tokens_and_weights     = tokenize(text, true);
        std::vector<int>& tokens    = tokens_and_weights.first;
        std::vector<float>& weights = tokens_and_weights.second;
        return get_learned_condition_common(work_ctx, n_threads, tokens, weights, clip_skip, width, height, adm_in_channels, force_zero_embeddings);
    }
};

struct FrozenCLIPVisionEmbedder : public GGMLRunner {
    CLIPVisionModelProjection vision_model;

    FrozenCLIPVisionEmbedder(ggml_backend_t backend, ggml_type wtype)
        : vision_model(OPEN_CLIP_VIT_H_14, true), GGMLRunner(backend, wtype) {
        vision_model.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "clip_vision";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        vision_model.get_param_tensors(tensors, "cond_stage_model.transformer");
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* pixel_values) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        pixel_values = to_backend(pixel_values);

        struct ggml_tensor* hidden_states = vision_model.forward(compute_ctx, pixel_values);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    void compute(const int n_threads,
                 ggml_tensor* pixel_values,
                 ggml_tensor** output,
                 ggml_context* output_ctx) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(pixel_values);
        };
        GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
    }
};

struct SD3CLIPEmbedder : public Conditioner {
    ggml_type wtype;
    CLIPTokenizer clip_l_tokenizer;
    CLIPTokenizer clip_g_tokenizer;
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<CLIPTextModelRunner> clip_l;
    std::shared_ptr<CLIPTextModelRunner> clip_g;
    std::shared_ptr<T5Runner> t5;

    SD3CLIPEmbedder(ggml_backend_t backend,
                    ggml_type wtype,
                    int clip_skip = -1)
        : wtype(wtype), clip_g_tokenizer(0) {
        if (clip_skip <= 0) {
            clip_skip = 2;
        }
        clip_l = std::make_shared<CLIPTextModelRunner>(backend, wtype, OPENAI_CLIP_VIT_L_14, clip_skip, false);
        clip_g = std::make_shared<CLIPTextModelRunner>(backend, wtype, OPEN_CLIP_VIT_BIGG_14, clip_skip, false);
        t5     = std::make_shared<T5Runner>(backend, wtype);
    }

    void set_clip_skip(int clip_skip) {
        clip_l->set_clip_skip(clip_skip);
        clip_g->set_clip_skip(clip_skip);
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        clip_l->get_param_tensors(tensors, "text_encoders.clip_l.transformer.text_model");
        clip_g->get_param_tensors(tensors, "text_encoders.clip_g.transformer.text_model");
        t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
    }

    void alloc_params_buffer() {
        clip_l->alloc_params_buffer();
        clip_g->alloc_params_buffer();
        t5->alloc_params_buffer();
    }

    void free_params_buffer() {
        clip_l->free_params_buffer();
        clip_g->free_params_buffer();
        t5->free_params_buffer();
    }

    size_t get_params_buffer_size() {
        size_t buffer_size = clip_l->get_params_buffer_size();
        buffer_size += clip_g->get_params_buffer_size();
        buffer_size += t5->get_params_buffer_size();
        return buffer_size;
    }

    std::vector<std::pair<std::vector<int>, std::vector<float>>> tokenize(std::string text,
                                                                          size_t max_length = 0,
                                                                          bool padding      = false) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            return false;
        };

        std::vector<int> clip_l_tokens;
        std::vector<float> clip_l_weights;
        std::vector<int> clip_g_tokens;
        std::vector<float> clip_g_weights;
        std::vector<int> t5_tokens;
        std::vector<float> t5_weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;

            std::vector<int> curr_tokens = clip_l_tokenizer.encode(curr_text, on_new_token_cb);
            clip_l_tokens.insert(clip_l_tokens.end(), curr_tokens.begin(), curr_tokens.end());
            clip_l_weights.insert(clip_l_weights.end(), curr_tokens.size(), curr_weight);

            curr_tokens = clip_g_tokenizer.encode(curr_text, on_new_token_cb);
            clip_g_tokens.insert(clip_g_tokens.end(), curr_tokens.begin(), curr_tokens.end());
            clip_g_weights.insert(clip_g_weights.end(), curr_tokens.size(), curr_weight);

            curr_tokens = t5_tokenizer.Encode(curr_text, true);
            t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
            t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
        }

        clip_l_tokenizer.pad_tokens(clip_l_tokens, clip_l_weights, max_length, padding);
        clip_g_tokenizer.pad_tokens(clip_g_tokens, clip_g_weights, max_length, padding);
        t5_tokenizer.pad_tokens(t5_tokens, t5_weights, max_length, padding);

        // for (int i = 0; i < clip_l_tokens.size(); i++) {
        //     std::cout << clip_l_tokens[i] << ":" << clip_l_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        // for (int i = 0; i < clip_g_tokens.size(); i++) {
        //     std::cout << clip_g_tokens[i] << ":" << clip_g_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        // for (int i = 0; i < t5_tokens.size(); i++) {
        //     std::cout << t5_tokens[i] << ":" << t5_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {{clip_l_tokens, clip_l_weights}, {clip_g_tokens, clip_g_weights}, {t5_tokens, t5_weights}};
    }

    SDCondition get_learned_condition_common(ggml_context* work_ctx,
                                             int n_threads,
                                             std::vector<std::pair<std::vector<int>, std::vector<float>>> token_and_weights,
                                             int clip_skip,
                                             bool force_zero_embeddings = false) {
        set_clip_skip(clip_skip);
        auto& clip_l_tokens  = token_and_weights[0].first;
        auto& clip_l_weights = token_and_weights[0].second;
        auto& clip_g_tokens  = token_and_weights[1].first;
        auto& clip_g_weights = token_and_weights[1].second;
        auto& t5_tokens      = token_and_weights[2].first;
        auto& t5_weights     = token_and_weights[2].second;

        int64_t t0                                 = ggml_time_ms();
        struct ggml_tensor* hidden_states          = NULL;  // [N, n_token*2, 4096]
        struct ggml_tensor* chunk_hidden_states    = NULL;  // [n_token*2, 4096]
        struct ggml_tensor* chunk_hidden_states_l  = NULL;  // [n_token, hidden_size_l]
        struct ggml_tensor* chunk_hidden_states_g  = NULL;  // [n_token, hidden_size_g]
        struct ggml_tensor* chunk_hidden_states_t5 = NULL;  // [n_token, hidden_size_t5]
        struct ggml_tensor* pooled                 = NULL;
        struct ggml_tensor* pooled_l               = NULL;  // [768,]
        struct ggml_tensor* pooled_g               = NULL;  // [1280,]
        std::vector<float> hidden_states_vec;

        size_t chunk_len   = 77;
        size_t chunk_count = clip_l_tokens.size() / chunk_len;
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            // clip_l
            {
                std::vector<int> chunk_tokens(clip_l_tokens.begin() + chunk_idx * chunk_len,
                                              clip_l_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(clip_l_weights.begin() + chunk_idx * chunk_len,
                                                 clip_l_weights.begin() + (chunk_idx + 1) * chunk_len);

                auto input_ids       = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
                size_t max_token_idx = 0;

                clip_l->compute(n_threads,
                                input_ids,
                                0,
                                NULL,
                                max_token_idx,
                                false,
                                &chunk_hidden_states_l,
                                work_ctx);
                {
                    auto tensor         = chunk_hidden_states_l;
                    float original_mean = ggml_tensor_mean(tensor);
                    for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                        for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                                float value = ggml_tensor_get_f32(tensor, i0, i1, i2);
                                value *= chunk_weights[i1];
                                ggml_tensor_set_f32(tensor, value, i0, i1, i2);
                            }
                        }
                    }
                    float new_mean = ggml_tensor_mean(tensor);
                    ggml_tensor_scale(tensor, (original_mean / new_mean));
                }

                if (chunk_idx == 0) {
                    // auto it = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_l_tokenizer.EOS_TOKEN_ID);
                    // max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);
                    // clip_l->compute(n_threads,
                    //                 input_ids,
                    //                 0,
                    //                 NULL,
                    //                 max_token_idx,
                    //                 true,
                    //                 &pooled_l,
                    //                 work_ctx);

                    // clip_l.transformer.text_model.text_projection no in file, ignore
                    // TODO: use torch.eye(embed_dim) as default clip_l.transformer.text_model.text_projection
                    pooled_l = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 768);
                    ggml_set_f32(pooled_l, 0.f);
                }
            }

            // clip_g
            {
                std::vector<int> chunk_tokens(clip_g_tokens.begin() + chunk_idx * chunk_len,
                                              clip_g_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(clip_g_weights.begin() + chunk_idx * chunk_len,
                                                 clip_g_weights.begin() + (chunk_idx + 1) * chunk_len);

                auto input_ids       = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
                size_t max_token_idx = 0;

                clip_g->compute(n_threads,
                                input_ids,
                                0,
                                NULL,
                                max_token_idx,
                                false,
                                &chunk_hidden_states_g,
                                work_ctx);

                {
                    auto tensor         = chunk_hidden_states_g;
                    float original_mean = ggml_tensor_mean(tensor);
                    for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                        for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                                float value = ggml_tensor_get_f32(tensor, i0, i1, i2);
                                value *= chunk_weights[i1];
                                ggml_tensor_set_f32(tensor, value, i0, i1, i2);
                            }
                        }
                    }
                    float new_mean = ggml_tensor_mean(tensor);
                    ggml_tensor_scale(tensor, (original_mean / new_mean));
                }

                if (chunk_idx == 0) {
                    // auto it = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_g_tokenizer.EOS_TOKEN_ID);
                    // max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);
                    // clip_g->compute(n_threads,
                    //                 input_ids,
                    //                 0,
                    //                 NULL,
                    //                 max_token_idx,
                    //                 true,
                    //                 &pooled_g,
                    //                 work_ctx);
                    // clip_l.transformer.text_model.text_projection no in file, ignore pooled_g too

                    // TODO: fix pooled_g
                    pooled_g = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1280);
                    ggml_set_f32(pooled_g, 0.f);
                }
            }

            // t5
            {
                std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                              t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                                 t5_weights.begin() + (chunk_idx + 1) * chunk_len);

                auto input_ids = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);

                t5->compute(n_threads,
                            input_ids,
                            &chunk_hidden_states_t5,
                            work_ctx);
                {
                    auto tensor         = chunk_hidden_states_t5;
                    float original_mean = ggml_tensor_mean(tensor);
                    for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                        for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                                float value = ggml_tensor_get_f32(tensor, i0, i1, i2);
                                value *= chunk_weights[i1];
                                ggml_tensor_set_f32(tensor, value, i0, i1, i2);
                            }
                        }
                    }
                    float new_mean = ggml_tensor_mean(tensor);
                    ggml_tensor_scale(tensor, (original_mean / new_mean));
                }
            }

            auto chunk_hidden_states_lg_pad = ggml_new_tensor_3d(work_ctx,
                                                                 chunk_hidden_states_l->type,
                                                                 4096,
                                                                 chunk_hidden_states_l->ne[1],
                                                                 chunk_hidden_states_l->ne[2]);  // [n_token, 4096]

            for (int i2 = 0; i2 < chunk_hidden_states_lg_pad->ne[2]; i2++) {
                for (int i1 = 0; i1 < chunk_hidden_states_lg_pad->ne[1]; i1++) {
                    for (int i0 = 0; i0 < chunk_hidden_states_lg_pad->ne[0]; i0++) {
                        float value = 0.f;
                        if (i0 < chunk_hidden_states_l->ne[0]) {
                            value = ggml_tensor_get_f32(chunk_hidden_states_l, i0, i1, i2);
                        } else if (i0 < chunk_hidden_states_l->ne[0] + chunk_hidden_states_g->ne[0]) {
                            value = ggml_tensor_get_f32(chunk_hidden_states_g, i0 - chunk_hidden_states_l->ne[0], i1, i2);
                        }
                        ggml_tensor_set_f32(chunk_hidden_states_lg_pad, value, i0, i1, i2);
                    }
                }
            }

            chunk_hidden_states = ggml_tensor_concat(work_ctx, chunk_hidden_states_lg_pad, chunk_hidden_states_t5, 1);  // [n_token*2, 4096]

            if (chunk_idx == 0) {
                pooled = ggml_tensor_concat(work_ctx, pooled_l, pooled_g, 0);  // [768 + 1280]
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            if (force_zero_embeddings) {
                float* vec = (float*)chunk_hidden_states->data;
                for (int i = 0; i < ggml_nelements(chunk_hidden_states); i++) {
                    vec[i] = 0;
                }
            }

            hidden_states_vec.insert(hidden_states_vec.end(),
                                     (float*)chunk_hidden_states->data,
                                     ((float*)chunk_hidden_states->data) + ggml_nelements(chunk_hidden_states));
        }

        hidden_states = vector_to_ggml_tensor(work_ctx, hidden_states_vec);
        hidden_states = ggml_reshape_2d(work_ctx,
                                        hidden_states,
                                        chunk_hidden_states->ne[0],
                                        ggml_nelements(hidden_states) / chunk_hidden_states->ne[0]);
        return SDCondition(hidden_states, pooled, NULL);
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const std::string& text,
                                      int clip_skip,
                                      int width,
                                      int height,
                                      int adm_in_channels        = -1,
                                      bool force_zero_embeddings = false) {
        auto tokens_and_weights = tokenize(text, 77, true);
        return get_learned_condition_common(work_ctx, n_threads, tokens_and_weights, clip_skip, force_zero_embeddings);
    }

    std::tuple<SDCondition, std::vector<bool>> get_learned_condition_with_trigger(ggml_context* work_ctx,
                                                                                  int n_threads,
                                                                                  const std::string& text,
                                                                                  int clip_skip,
                                                                                  int width,
                                                                                  int height,
                                                                                  int num_input_imgs,
                                                                                  int adm_in_channels        = -1,
                                                                                  bool force_zero_embeddings = false) {
        GGML_ASSERT(0 && "Not implemented yet!");
    }

    std::string remove_trigger_from_prompt(ggml_context* work_ctx,
                                           const std::string& prompt) {
        GGML_ASSERT(0 && "Not implemented yet!");
    }
};

#endif