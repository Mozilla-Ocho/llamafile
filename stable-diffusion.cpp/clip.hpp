#ifndef __CLIP_HPP__
#define __CLIP_HPP__

#include "ggml_extend.hpp"
#include "model.h"

/*================================================== CLIPTokenizer ===================================================*/

std::pair<std::unordered_map<std::string, float>, std::string> extract_and_remove_lora(std::string text) {
    std::regex re("<lora:([^:]+):([^>]+)>");
    std::smatch matches;
    std::unordered_map<std::string, float> filename2multiplier;

    while (std::regex_search(text, matches, re)) {
        std::string filename = matches[1].str();
        float multiplier     = std::stof(matches[2].str());

        text = std::regex_replace(text, re, "", std::regex_constants::format_first_only);

        if (multiplier == 0.f) {
            continue;
        }

        if (filename2multiplier.find(filename) == filename2multiplier.end()) {
            filename2multiplier[filename] = multiplier;
        } else {
            filename2multiplier[filename] += multiplier;
        }
    }

    return std::make_pair(filename2multiplier, text);
}

std::vector<std::pair<int, std::u32string>> bytes_to_unicode() {
    std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
    std::set<int> byte_set;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 161; b <= 172; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 174; b <= 255; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(n + 256)));
            ++n;
        }
    }
    // LOG_DEBUG("byte_unicode_pairs %d", byte_unicode_pairs.size());
    return byte_unicode_pairs;
}

// Ref: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

typedef std::function<bool(std::string&, std::vector<int32_t>&)> on_new_token_cb_t;

class CLIPTokenizer {
private:
    std::map<int, std::u32string> byte_encoder;
    std::map<std::u32string, int> byte_decoder;
    std::map<std::u32string, int> encoder;
    std::map<int, std::u32string> decoder;
    std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
    std::regex pat;
    int encoder_len;
    int bpe_len;

public:
    const std::string UNK_TOKEN = "<|endoftext|>";
    const std::string BOS_TOKEN = "<|startoftext|>";
    const std::string EOS_TOKEN = "<|endoftext|>";
    const std::string PAD_TOEKN = "<|endoftext|>";

    const int UNK_TOKEN_ID = 49407;
    const int BOS_TOKEN_ID = 49406;
    const int EOS_TOKEN_ID = 49407;
    const int PAD_TOKEN_ID = 49407;

private:
    static std::string strip(const std::string& str) {
        std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
        std::string::size_type end   = str.find_last_not_of(" \t\n\r\v\f");

        if (start == std::string::npos) {
            // String contains only whitespace characters
            return "";
        }

        return str.substr(start, end - start + 1);
    }

    static std::string whitespace_clean(std::string text) {
        text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
        text = strip(text);
        return text;
    }

    static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords) {
        std::set<std::pair<std::u32string, std::u32string>> pairs;
        if (subwords.size() == 0) {
            return pairs;
        }
        std::u32string prev_subword = subwords[0];
        for (int i = 1; i < subwords.size(); i++) {
            std::u32string subword = subwords[i];
            std::pair<std::u32string, std::u32string> pair(prev_subword, subword);
            pairs.insert(pair);
            prev_subword = subword;
        }
        return pairs;
    }

public:
    CLIPTokenizer(int pad_token_id = 49407, const std::string& merges_utf8_str = "")
        : PAD_TOKEN_ID(pad_token_id) {
        if (merges_utf8_str.size() > 0) {
            load_from_merges(merges_utf8_str);
        } else {
            load_from_merges(ModelLoader::load_merges());
        }
    }

    void load_from_merges(const std::string& merges_utf8_str) {
        auto byte_unicode_pairs = bytes_to_unicode();
        // printf("byte_unicode_pairs have %lu pairs \n", byte_unicode_pairs.size());
        byte_encoder = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
        for (auto& pair : byte_unicode_pairs) {
            byte_decoder[pair.second] = pair.first;
        }
        // for (auto & pair: byte_unicode_pairs) {
        //     std::cout << pair.first << ": " << pair.second << std::endl;
        // }
        std::vector<std::u32string> merges;
        size_t start = 0;
        size_t pos;
        std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
        while ((pos = merges_utf32_str.find('\n', start)) != std::string::npos) {
            merges.push_back(merges_utf32_str.substr(start, pos - start));
            start = pos + 1;
        }
        // LOG_DEBUG("merges size %llu", merges.size());
        GGML_ASSERT(merges.size() == 48895);
        merges = std::vector<std::u32string>(merges.begin() + 1, merges.end());
        std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
        for (const auto& merge : merges) {
            size_t space_pos = merge.find(' ');
            merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
            // LOG_DEBUG("%s", utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
            // printf("%s :: %s | %s \n", utf32_to_utf8(merge).c_str(), utf32_to_utf8(merge.substr(0, space_pos)).c_str(),
            //                     utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
        }
        std::vector<std::u32string> vocab;
        for (const auto& pair : byte_unicode_pairs) {
            vocab.push_back(pair.second);
        }
        for (const auto& pair : byte_unicode_pairs) {
            vocab.push_back(pair.second + utf8_to_utf32("</w>"));
        }
        for (const auto& merge : merge_pairs) {
            vocab.push_back(merge.first + merge.second);
        }
        vocab.push_back(utf8_to_utf32("<|startoftext|>"));
        vocab.push_back(utf8_to_utf32("<|endoftext|>"));
        LOG_DEBUG("vocab size: %llu", vocab.size());
        int i = 0;
        for (const auto& token : vocab) {
            encoder[token] = i;
            decoder[i]     = token;
            i++;
        }
        encoder_len = i;

        auto it = encoder.find(utf8_to_utf32("img</w>"));
        if (it != encoder.end()) {
            LOG_DEBUG(" trigger word img already in vocab");
        } else {
            LOG_DEBUG(" trigger word img not in vocab yet");
        }

        int rank = 0;
        for (const auto& merge : merge_pairs) {
            bpe_ranks[merge] = rank++;
        }
        bpe_len = rank;
    };

    void add_token(const std::string& text) {
        std::u32string token = utf8_to_utf32(text);
        auto it              = encoder.find(token);
        if (it != encoder.end()) {
            encoder[token]       = encoder_len;
            decoder[encoder_len] = token;
            encoder_len++;
        }
    }

    std::u32string bpe(const std::u32string& token) {
        std::vector<std::u32string> word;

        for (int i = 0; i < token.size() - 1; i++) {
            word.emplace_back(1, token[i]);
        }
        word.push_back(token.substr(token.size() - 1) + utf8_to_utf32("</w>"));

        std::set<std::pair<std::u32string, std::u32string>> pairs = get_pairs(word);

        if (pairs.empty()) {
            return token + utf8_to_utf32("</w>");
        }

        while (true) {
            auto min_pair_iter = std::min_element(pairs.begin(),
                                                  pairs.end(),
                                                  [&](const std::pair<std::u32string, std::u32string>& a,
                                                      const std::pair<std::u32string, std::u32string>& b) {
                                                      if (bpe_ranks.find(a) == bpe_ranks.end()) {
                                                          return false;
                                                      } else if (bpe_ranks.find(b) == bpe_ranks.end()) {
                                                          return true;
                                                      }
                                                      return bpe_ranks.at(a) < bpe_ranks.at(b);
                                                  });

            const std::pair<std::u32string, std::u32string>& bigram = *min_pair_iter;

            if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
                break;
            }

            std::u32string first  = bigram.first;
            std::u32string second = bigram.second;
            std::vector<std::u32string> new_word;
            int32_t i = 0;

            while (i < word.size()) {
                auto it = std::find(word.begin() + i, word.end(), first);
                if (it == word.end()) {
                    new_word.insert(new_word.end(), word.begin() + i, word.end());
                    break;
                }
                new_word.insert(new_word.end(), word.begin() + i, it);
                i = static_cast<int32_t>(std::distance(word.begin(), it));

                if (word[i] == first && i < static_cast<int32_t>(word.size()) - 1 && word[i + 1] == second) {
                    new_word.push_back(first + second);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i += 1;
                }
            }

            word = new_word;

            if (word.size() == 1) {
                break;
            }
            pairs = get_pairs(word);
        }

        std::u32string result;
        for (int i = 0; i < word.size(); i++) {
            result += word[i];
            if (i != word.size() - 1) {
                result += utf8_to_utf32(" ");
            }
        }

        return result;
    }

    std::vector<int> tokenize(std::string text,
                              on_new_token_cb_t on_new_token_cb,
                              size_t max_length = 0,
                              bool padding      = false) {
        std::vector<int32_t> tokens = encode(text, on_new_token_cb);

        tokens.insert(tokens.begin(), BOS_TOKEN_ID);
        if (max_length > 0) {
            if (tokens.size() > max_length - 1) {
                tokens.resize(max_length - 1);
                tokens.push_back(EOS_TOKEN_ID);
            } else {
                tokens.push_back(EOS_TOKEN_ID);
                if (padding) {
                    tokens.insert(tokens.end(), max_length - tokens.size(), PAD_TOKEN_ID);
                }
            }
        }

        return tokens;
    }

    void pad_tokens(std::vector<int>& tokens,
                    std::vector<float>& weights,
                    size_t max_length = 0,
                    bool padding      = false) {
        if (max_length > 0 && padding) {
            size_t n = std::ceil(tokens.size() * 1.0 / (max_length - 2));
            if (n == 0) {
                n = 1;
            }
            size_t length = max_length * n;
            LOG_DEBUG("token length: %llu", length);
            std::vector<int> new_tokens;
            std::vector<float> new_weights;
            new_tokens.push_back(BOS_TOKEN_ID);
            new_weights.push_back(1.0);
            int token_idx = 0;
            for (int i = 1; i < length; i++) {
                if (token_idx >= tokens.size()) {
                    break;
                }
                if (i % max_length == 0) {
                    new_tokens.push_back(BOS_TOKEN_ID);
                    new_weights.push_back(1.0);
                } else if (i % max_length == max_length - 1) {
                    new_tokens.push_back(EOS_TOKEN_ID);
                    new_weights.push_back(1.0);
                } else {
                    new_tokens.push_back(tokens[token_idx]);
                    new_weights.push_back(weights[token_idx]);
                    token_idx++;
                }
            }

            new_tokens.push_back(EOS_TOKEN_ID);
            new_weights.push_back(1.0);
            tokens  = new_tokens;
            weights = new_weights;

            if (padding) {
                tokens.insert(tokens.end(), length - tokens.size(), PAD_TOKEN_ID);
                weights.insert(weights.end(), length - weights.size(), 1.0);
            }
        }
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string text = "";
        for (int t : tokens) {
            if (t == 49406 || t == 49407)
                continue;
            std::u32string ts = decoder[t];
            // printf("%d, %s \n", t,  utf32_to_utf8(ts).c_str());
            std::string s = utf32_to_utf8(ts);
            if (s.length() >= 4 && ends_with(s, "</w>")) {
                text += " " + s.replace(s.length() - 4, s.length() - 1, "");
            } else {
                text += " " + s;
            }
        }
        // std::vector<unsigned char> bytes;
        // for (auto c : text){
        //     bytes.push_back(byte_decoder[c]);
        // }

        // std::string s((char *)bytes.data());
        // std::string s = "";
        return trim(text);
    }

    std::vector<int> encode(std::string text, on_new_token_cb_t on_new_token_cb) {
        std::string original_text = text;
        std::vector<int32_t> bpe_tokens;
        text = whitespace_clean(text);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });

        std::regex pat(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
                       std::regex::icase);

        std::smatch matches;
        std::string str = text;
        std::vector<std::string> token_strs;
        while (std::regex_search(str, matches, pat)) {
            bool skip = on_new_token_cb(str, bpe_tokens);
            if (skip) {
                continue;
            }
            for (auto& token : matches) {
                std::string token_str = token.str();
                std::u32string utf32_token;
                for (int i = 0; i < token_str.length(); i++) {
                    char b = token_str[i];
                    utf32_token += byte_encoder[b];
                }
                auto bpe_strs = bpe(utf32_token);
                size_t start  = 0;
                size_t pos;
                while ((pos = bpe_strs.find(' ', start)) != std::u32string::npos) {
                    auto bpe_str = bpe_strs.substr(start, pos - start);
                    bpe_tokens.push_back(encoder[bpe_str]);
                    token_strs.push_back(utf32_to_utf8(bpe_str));

                    start = pos + 1;
                }
                auto bpe_str = bpe_strs.substr(start, bpe_strs.size() - start);
                bpe_tokens.push_back(encoder[bpe_str]);
                token_strs.push_back(utf32_to_utf8(bpe_str));
            }
            str = matches.suffix();
        }
        std::stringstream ss;
        ss << "[";
        for (auto token : token_strs) {
            ss << "\"" << token << "\", ";
        }
        ss << "]";
        // LOG_DEBUG("split prompt \"%s\" to tokens %s", original_text.c_str(), ss.str().c_str());
        // printf("split prompt \"%s\" to tokens %s \n", original_text.c_str(), ss.str().c_str());
        return bpe_tokens;
    }
};

/*================================================ FrozenCLIPEmbedder ================================================*/

// Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py

struct CLIPMLP : public GGMLBlock {
protected:
    bool use_gelu;

public:
    CLIPMLP(int64_t d_model, int64_t intermediate_size) {
        blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(d_model, intermediate_size));
        blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, d_model));

        if (d_model == 1024 || d_model == 1280) {  // SD 2.x
            use_gelu = true;
        } else {  // SD 1.x
            use_gelu = false;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, d_model]
        auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

        x = fc1->forward(ctx, x);
        if (use_gelu) {
            x = ggml_gelu_inplace(ctx, x);
        } else {
            x = ggml_gelu_quick_inplace(ctx, x);
        }
        x = fc2->forward(ctx, x);
        return x;
    }
};

struct CLIPLayer : public GGMLBlock {
protected:
    int64_t d_model;  // hidden_size/embed_dim
    int64_t n_head;
    int64_t intermediate_size;

public:
    CLIPLayer(int64_t d_model,
              int64_t n_head,
              int64_t intermediate_size)
        : d_model(d_model),
          n_head(n_head),
          intermediate_size(intermediate_size) {
        blocks["self_attn"] = std::shared_ptr<GGMLBlock>(new MultiheadAttention(d_model, n_head, true, true));

        blocks["layer_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));
        blocks["layer_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));

        blocks["mlp"] = std::shared_ptr<GGMLBlock>(new CLIPMLP(d_model, intermediate_size));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, bool mask = true) {
        // x: [N, n_token, d_model]
        auto self_attn   = std::dynamic_pointer_cast<MultiheadAttention>(blocks["self_attn"]);
        auto layer_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm1"]);
        auto layer_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm2"]);
        auto mlp         = std::dynamic_pointer_cast<CLIPMLP>(blocks["mlp"]);

        x = ggml_add(ctx, x, self_attn->forward(ctx, layer_norm1->forward(ctx, x), mask));
        x = ggml_add(ctx, x, mlp->forward(ctx, layer_norm2->forward(ctx, x)));
        return x;
    }
};

struct CLIPEncoder : public GGMLBlock {
protected:
    int64_t n_layer;

public:
    CLIPEncoder(int64_t n_layer,
                int64_t d_model,
                int64_t n_head,
                int64_t intermediate_size)
        : n_layer(n_layer) {
        for (int i = 0; i < n_layer; i++) {
            std::string name = "layers." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new CLIPLayer(d_model, n_head, intermediate_size));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int clip_skip = -1, bool mask = true) {
        // x: [N, n_token, d_model]
        int layer_idx = n_layer - 1;
        // LOG_DEBUG("clip_skip %d", clip_skip);
        if (clip_skip > 0) {
            layer_idx = n_layer - clip_skip;
        }

        for (int i = 0; i < n_layer; i++) {
            // LOG_DEBUG("layer %d", i);
            if (i == layer_idx + 1) {
                break;
            }
            std::string name = "layers." + std::to_string(i);
            auto layer       = std::dynamic_pointer_cast<CLIPLayer>(blocks[name]);
            x                = layer->forward(ctx, x, mask);  // [N, n_token, d_model]
            // LOG_DEBUG("layer %d", i);
        }
        return x;
    }
};

class CLIPEmbeddings : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t vocab_size;
    int64_t num_positions;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["token_embedding.weight"]    = ggml_new_tensor_2d(ctx, wtype, embed_dim, vocab_size);
        params["position_embedding.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, num_positions);
    }

public:
    CLIPEmbeddings(int64_t embed_dim,
                   int64_t vocab_size    = 49408,
                   int64_t num_positions = 77)
        : embed_dim(embed_dim),
          vocab_size(vocab_size),
          num_positions(num_positions) {
    }

    struct ggml_tensor* get_token_embed_weight() {
        return params["token_embedding.weight"];
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                struct ggml_tensor* custom_embed_weight) {
        // input_ids: [N, n_token]
        auto token_embed_weight    = params["token_embedding.weight"];
        auto position_embed_weight = params["position_embedding.weight"];

        GGML_ASSERT(input_ids->ne[0] == position_embed_weight->ne[1]);
        input_ids            = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        auto token_embedding = ggml_get_rows(ctx, custom_embed_weight != NULL ? custom_embed_weight : token_embed_weight, input_ids);
        token_embedding      = ggml_reshape_3d(ctx, token_embedding, token_embedding->ne[0], token_embedding->ne[1], token_embedding->ne[3]);

        // token_embedding + position_embedding
        auto x = ggml_add(ctx,
                          token_embedding,
                          position_embed_weight);  // [N, n_token, embed_dim]
        return x;
    }
};

class CLIPVisionEmbeddings : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t num_channels;
    int64_t patch_size;
    int64_t image_size;
    int64_t num_patches;
    int64_t num_positions;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["patch_embedding.weight"]    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, patch_size, patch_size, num_channels, embed_dim);
        params["class_embedding"]           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim);
        params["position_embedding.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, num_positions);
    }

public:
    CLIPVisionEmbeddings(int64_t embed_dim,
                         int64_t num_channels = 3,
                         int64_t patch_size   = 14,
                         int64_t image_size   = 224)
        : embed_dim(embed_dim),
          num_channels(num_channels),
          patch_size(patch_size),
          image_size(image_size) {
        num_patches   = (image_size / patch_size) * (image_size / patch_size);
        num_positions = num_patches + 1;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* pixel_values) {
        // pixel_values: [N, num_channels, image_size, image_size]
        // return: [N, num_positions, embed_dim]
        GGML_ASSERT(pixel_values->ne[0] == image_size && pixel_values->ne[1] == image_size && pixel_values->ne[2] == num_channels);

        auto patch_embed_weight    = params["patch_embedding.weight"];
        auto class_embed_weight    = params["class_embedding"];
        auto position_embed_weight = params["position_embedding.weight"];

        // concat(patch_embedding, class_embedding) + position_embedding
        struct ggml_tensor* patch_embedding;
        int64_t N       = pixel_values->ne[3];
        patch_embedding = ggml_nn_conv_2d(ctx, pixel_values, patch_embed_weight, NULL, patch_size, patch_size);  // [N, embed_dim, image_size // pacht_size, image_size // pacht_size]
        patch_embedding = ggml_reshape_3d(ctx, patch_embedding, num_patches, embed_dim, N);                      // [N, embed_dim, num_patches]
        patch_embedding = ggml_cont(ctx, ggml_permute(ctx, patch_embedding, 1, 0, 2, 3));                        // [N, num_patches, embed_dim]
        patch_embedding = ggml_reshape_4d(ctx, patch_embedding, 1, embed_dim, num_patches, N);                   // [N, num_patches, embed_dim, 1]

        struct ggml_tensor* class_embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, N);
        class_embedding                     = ggml_repeat(ctx, class_embed_weight, class_embedding);      // [N, embed_dim]
        class_embedding                     = ggml_reshape_4d(ctx, class_embedding, 1, embed_dim, 1, N);  // [N, 1, embed_dim, 1]

        struct ggml_tensor* x = ggml_concat(ctx, class_embedding, patch_embedding, 2);  // [N, num_positions, embed_dim, 1]
        x                     = ggml_reshape_3d(ctx, x, embed_dim, num_positions, N);   // [N, num_positions, embed_dim]
        x                     = ggml_add(ctx, x, position_embed_weight);
        return x;  // [N, num_positions, embed_dim]
    }
};

// OPENAI_CLIP_VIT_L_14: https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
// OPEN_CLIP_VIT_H_14: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json
// OPEN_CLIP_VIT_BIGG_14: https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/blob/main/config.json (CLIPTextModelWithProjection)

enum CLIPVersion {
    OPENAI_CLIP_VIT_L_14,   // SD 1.x and SDXL
    OPEN_CLIP_VIT_H_14,     // SD 2.x
    OPEN_CLIP_VIT_BIGG_14,  // SDXL
};

class CLIPTextModel : public GGMLBlock {
protected:
    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        if (version == OPEN_CLIP_VIT_BIGG_14) {
            params["text_projection"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, projection_dim, hidden_size);
        }
    }

public:
    CLIPVersion version = OPENAI_CLIP_VIT_L_14;
    // network hparams
    int32_t vocab_size        = 49408;
    int32_t n_token           = 77;  // max_position_embeddings
    int32_t hidden_size       = 768;
    int32_t intermediate_size = 3072;
    int32_t n_head            = 12;
    int32_t n_layer           = 12;    // num_hidden_layers
    int32_t projection_dim    = 1280;  // only for OPEN_CLIP_VIT_BIGG_14
    int32_t clip_skip         = -1;
    bool with_final_ln        = true;

    CLIPTextModel(CLIPVersion version = OPENAI_CLIP_VIT_L_14,
                  int clip_skip_value = -1,
                  bool with_final_ln  = true)
        : version(version), with_final_ln(with_final_ln) {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size       = 1024;
            intermediate_size = 4096;
            n_head            = 16;
            n_layer           = 24;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) {  // CLIPTextModelWithProjection
            hidden_size       = 1280;
            intermediate_size = 5120;
            n_head            = 20;
            n_layer           = 32;
        }
        set_clip_skip(clip_skip_value);

        blocks["embeddings"]       = std::shared_ptr<GGMLBlock>(new CLIPEmbeddings(hidden_size, vocab_size, n_token));
        blocks["encoder"]          = std::shared_ptr<GGMLBlock>(new CLIPEncoder(n_layer, hidden_size, n_head, intermediate_size));
        blocks["final_layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size));
    }

    void set_clip_skip(int skip) {
        if (skip <= 0) {
            return;
        }
        clip_skip = skip;
    }

    struct ggml_tensor* get_token_embed_weight() {
        auto embeddings = std::dynamic_pointer_cast<CLIPEmbeddings>(blocks["embeddings"]);
        return embeddings->get_token_embed_weight();
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                struct ggml_tensor* tkn_embeddings,
                                size_t max_token_idx = 0,
                                bool return_pooled   = false) {
        // input_ids: [N, n_token]
        auto embeddings       = std::dynamic_pointer_cast<CLIPEmbeddings>(blocks["embeddings"]);
        auto encoder          = std::dynamic_pointer_cast<CLIPEncoder>(blocks["encoder"]);
        auto final_layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["final_layer_norm"]);

        auto x = embeddings->forward(ctx, input_ids, tkn_embeddings);  // [N, n_token, hidden_size]
        x      = encoder->forward(ctx, x, return_pooled ? -1 : clip_skip, true);
        if (return_pooled || with_final_ln) {
            x = final_layer_norm->forward(ctx, x);
        }

        if (return_pooled) {
            auto text_projection = params["text_projection"];
            ggml_tensor* pooled  = ggml_view_1d(ctx, x, hidden_size, x->nb[1] * max_token_idx);
            pooled               = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, text_projection)), pooled);
            return pooled;
        }

        return x;  // [N, n_token, hidden_size]
    }
};

class CLIPVisionModel : public GGMLBlock {
public:
    // network hparams
    int32_t num_channels      = 3;
    int32_t patch_size        = 14;
    int32_t image_size        = 224;
    int32_t num_positions     = 257;  // (image_size / patch_size)^2 + 1
    int32_t hidden_size       = 1024;
    int32_t intermediate_size = 4096;
    int32_t n_head            = 16;
    int32_t n_layer           = 24;

public:
    CLIPVisionModel(CLIPVersion version = OPENAI_CLIP_VIT_L_14) {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size       = 1280;
            intermediate_size = 5120;
            n_head            = 16;
            n_layer           = 32;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) {
            hidden_size       = 1664;
            intermediate_size = 8192;
            n_head            = 16;
            n_layer           = 48;
        }

        blocks["embeddings"]     = std::shared_ptr<GGMLBlock>(new CLIPVisionEmbeddings(hidden_size, num_channels, patch_size, image_size));
        blocks["pre_layernorm"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size));
        blocks["encoder"]        = std::shared_ptr<GGMLBlock>(new CLIPEncoder(n_layer, hidden_size, n_head, intermediate_size));
        blocks["post_layernorm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* pixel_values, bool return_pooled = true) {
        // pixel_values: [N, num_channels, image_size, image_size]
        auto embeddings     = std::dynamic_pointer_cast<CLIPVisionEmbeddings>(blocks["embeddings"]);
        auto pre_layernorm  = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_layernorm"]);
        auto encoder        = std::dynamic_pointer_cast<CLIPEncoder>(blocks["encoder"]);
        auto post_layernorm = std::dynamic_pointer_cast<LayerNorm>(blocks["post_layernorm"]);

        auto x = embeddings->forward(ctx, pixel_values);  // [N, num_positions, embed_dim]
        x      = pre_layernorm->forward(ctx, x);
        x      = encoder->forward(ctx, x, -1, false);
        x      = post_layernorm->forward(ctx, x);  // [N, n_token, hidden_size]

        GGML_ASSERT(x->ne[3] == 1);
        if (return_pooled) {
            ggml_tensor* pooled = ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0));
            return pooled;  // [N, hidden_size]
        } else {
            return x;  // [N, n_token, hidden_size]
        }
    }
};

class CLIPProjection : public UnaryBlock {
protected:
    int64_t in_features;
    int64_t out_features;
    bool transpose_weight;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        if (transpose_weight) {
            LOG_ERROR("transpose_weight");
            params["weight"] = ggml_new_tensor_2d(ctx, wtype, out_features, in_features);
        } else {
            params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        }
    }

public:
    CLIPProjection(int64_t in_features,
                   int64_t out_features,
                   bool transpose_weight = false)
        : in_features(in_features),
          out_features(out_features),
          transpose_weight(transpose_weight) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        if (transpose_weight) {
            w = ggml_cont(ctx, ggml_transpose(ctx, w));
        }
        return ggml_nn_linear(ctx, x, w, NULL);
    }
};

class CLIPVisionModelProjection : public GGMLBlock {
public:
    int32_t hidden_size    = 1024;
    int32_t projection_dim = 768;
    int32_t image_size     = 224;

public:
    CLIPVisionModelProjection(CLIPVersion version   = OPENAI_CLIP_VIT_L_14,
                              bool transpose_proj_w = false) {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size    = 1280;
            projection_dim = 1024;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) {
            hidden_size = 1664;
        }

        blocks["vision_model"]      = std::shared_ptr<GGMLBlock>(new CLIPVisionModel(version));
        blocks["visual_projection"] = std::shared_ptr<GGMLBlock>(new CLIPProjection(hidden_size, projection_dim, transpose_proj_w));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* pixel_values) {
        // pixel_values: [N, num_channels, image_size, image_size]
        // return: [N, projection_dim]
        auto vision_model      = std::dynamic_pointer_cast<CLIPVisionModel>(blocks["vision_model"]);
        auto visual_projection = std::dynamic_pointer_cast<CLIPProjection>(blocks["visual_projection"]);

        auto x = vision_model->forward(ctx, pixel_values);  // [N, hidden_size]
        x      = visual_projection->forward(ctx, x);        // [N, projection_dim]

        return x;  // [N, projection_dim]
    }
};

struct CLIPTextModelRunner : public GGMLRunner {
    CLIPTextModel model;

    CLIPTextModelRunner(ggml_backend_t backend,
                        ggml_type wtype,
                        CLIPVersion version = OPENAI_CLIP_VIT_L_14,
                        int clip_skip_value = 1,
                        bool with_final_ln  = true)
        : GGMLRunner(backend, wtype), model(version, clip_skip_value, with_final_ln) {
        model.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "clip";
    }

    void set_clip_skip(int clip_skip) {
        model.set_clip_skip(clip_skip);
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                struct ggml_tensor* embeddings,
                                size_t max_token_idx = 0,
                                bool return_pooled   = false) {
        size_t N       = input_ids->ne[1];
        size_t n_token = input_ids->ne[0];
        if (input_ids->ne[0] > model.n_token) {
            GGML_ASSERT(input_ids->ne[0] % model.n_token == 0);
            input_ids = ggml_reshape_2d(ctx, input_ids, model.n_token, input_ids->ne[0] / model.n_token);
        }

        return model.forward(ctx, input_ids, embeddings, max_token_idx, return_pooled);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids,
                                    int num_custom_embeddings    = 0,
                                    void* custom_embeddings_data = NULL,
                                    size_t max_token_idx         = 0,
                                    bool return_pooled           = false) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        input_ids = to_backend(input_ids);

        struct ggml_tensor* embeddings = NULL;

        if (num_custom_embeddings > 0 && custom_embeddings_data != NULL) {
            auto custom_embeddings = ggml_new_tensor_2d(compute_ctx,
                                                        wtype,
                                                        model.hidden_size,
                                                        num_custom_embeddings);
            set_backend_tensor_data(custom_embeddings, custom_embeddings_data);

            auto token_embed_weight = model.get_token_embed_weight();
            // concatenate custom embeddings
            embeddings = ggml_concat(compute_ctx, token_embed_weight, custom_embeddings, 1);
        }

        struct ggml_tensor* hidden_states = forward(compute_ctx, input_ids, embeddings, max_token_idx, return_pooled);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* input_ids,
                 int num_custom_embeddings,
                 void* custom_embeddings_data,
                 size_t max_token_idx,
                 bool return_pooled,
                 ggml_tensor** output,
                 ggml_context* output_ctx = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(input_ids, num_custom_embeddings, custom_embeddings_data, max_token_idx, return_pooled);
        };
        GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
    }
};

#endif  // __CLIP_HPP__
