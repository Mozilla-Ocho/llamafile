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

const std::string UNK_TOKEN = "<|endoftext|>";
const std::string BOS_TOKEN = "<|startoftext|>";
const std::string EOS_TOKEN = "<|endoftext|>";
const std::string PAD_TOEKN = "<|endoftext|>";

const int UNK_TOKEN_ID = 49407;
const int BOS_TOKEN_ID = 49406;
const int EOS_TOKEN_ID = 49407;
const int PAD_TOKEN_ID = 49407;

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
    SDVersion version = VERSION_1_x;
    std::map<int, std::u32string> byte_encoder;
    std::map<std::u32string, int> byte_decoder;
    std::map<std::u32string, int> encoder;
    std::map<int, std::u32string> decoder;
    std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
    std::regex pat;
    int encoder_len;
    int bpe_len;

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
    CLIPTokenizer(SDVersion version = VERSION_1_x)
        : version(version) {}

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
                    int pad_token_id = PAD_TOKEN_ID;
                    if (version == VERSION_2_x) {
                        pad_token_id = 0;
                    }
                    tokens.insert(tokens.end(), max_length - tokens.size(), pad_token_id);
                }
            }
        }

        return tokens;
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

// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/prompt_parser.py#L345
//
// Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
// Accepted tokens are:
//   (abc) - increases attention to abc by a multiplier of 1.1
//   (abc:3.12) - increases attention to abc by a multiplier of 3.12
//   [abc] - decreases attention to abc by a multiplier of 1.1
//   \( - literal character '('
//   \[ - literal character '['
//   \) - literal character ')'
//   \] - literal character ']'
//   \\ - literal character '\'
//   anything else - just text
//
// >>> parse_prompt_attention('normal text')
// [['normal text', 1.0]]
// >>> parse_prompt_attention('an (important) word')
// [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
// >>> parse_prompt_attention('(unbalanced')
// [['unbalanced', 1.1]]
// >>> parse_prompt_attention('\(literal\]')
// [['(literal]', 1.0]]
// >>> parse_prompt_attention('(unnecessary)(parens)')
// [['unnecessaryparens', 1.1]]
// >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
// [['a ', 1.0],
//  ['house', 1.5730000000000004],
//  [' ', 1.1],
//  ['on', 1.0],
//  [' a ', 1.1],
//  ['hill', 0.55],
//  [', sun, ', 1.1],
//  ['sky', 1.4641000000000006],
//  ['.', 1.1]]
std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    std::vector<std::pair<std::string, float>> res;
    std::vector<int> round_brackets;
    std::vector<int> square_brackets;

    float round_bracket_multiplier  = 1.1f;
    float square_bracket_multiplier = 1 / 1.1f;

    std::regex re_attention(R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|[^\\()\[\]:]+|:)");
    std::regex re_break(R"(\s*\bBREAK\b\s*)");

    auto multiply_range = [&](int start_position, float multiplier) {
        for (int p = start_position; p < res.size(); ++p) {
            res[p].second *= multiplier;
        }
    };

    std::smatch m;
    std::string remaining_text = text;

    while (std::regex_search(remaining_text, m, re_attention)) {
        std::string text   = m[0];
        std::string weight = m[1];

        if (text == "(") {
            round_brackets.push_back((int)res.size());
        } else if (text == "[") {
            square_brackets.push_back((int)res.size());
        } else if (!weight.empty()) {
            if (!round_brackets.empty()) {
                multiply_range(round_brackets.back(), std::stof(weight));
                round_brackets.pop_back();
            }
        } else if (text == ")" && !round_brackets.empty()) {
            multiply_range(round_brackets.back(), round_bracket_multiplier);
            round_brackets.pop_back();
        } else if (text == "]" && !square_brackets.empty()) {
            multiply_range(square_brackets.back(), square_bracket_multiplier);
            square_brackets.pop_back();
        } else if (text == "\\(") {
            res.push_back({text.substr(1), 1.0f});
        } else {
            res.push_back({text, 1.0f});
        }

        remaining_text = m.suffix();
    }

    for (int pos : round_brackets) {
        multiply_range(pos, round_bracket_multiplier);
    }

    for (int pos : square_brackets) {
        multiply_range(pos, square_bracket_multiplier);
    }

    if (res.empty()) {
        res.push_back({"", 1.0f});
    }

    int i = 0;
    while (i + 1 < res.size()) {
        if (res[i].second == res[i + 1].second) {
            res[i].first += res[i + 1].first;
            res.erase(res.begin() + i + 1);
        } else {
            ++i;
        }
    }

    return res;
}

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
        blocks["self_attn"] = std::shared_ptr<GGMLBlock>(new MultiheadAttention(d_model, n_head, true));

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

        struct ggml_tensor* x = ggml_concat(ctx, class_embedding, patch_embedding);    // [N, num_positions, embed_dim, 1]
        x                     = ggml_reshape_3d(ctx, x, embed_dim, num_positions, N);  // [N, num_positions, embed_dim]
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

// ldm.modules.encoders.modules.FrozenCLIPEmbedder
// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/sd_hijack_clip.py#L283
struct FrozenCLIPEmbedderWithCustomWords : public GGMLModule {
    SDVersion version = VERSION_1_x;
    CLIPTokenizer tokenizer;
    CLIPTextModel text_model;
    CLIPTextModel text_model2;

    std::string embd_dir;
    int32_t num_custom_embeddings = 0;
    std::vector<uint8_t> token_embed_custom;
    std::vector<std::string> readed_embeddings;

    FrozenCLIPEmbedderWithCustomWords(ggml_backend_t backend,
                                      ggml_type wtype,
                                      SDVersion version = VERSION_1_x,
                                      int clip_skip     = -1)
        : GGMLModule(backend, wtype), version(version), tokenizer(version) {
        if (clip_skip <= 0) {
            clip_skip = 1;
            if (version == VERSION_2_x || version == VERSION_XL) {
                clip_skip = 2;
            }
        }
        if (version == VERSION_1_x) {
            text_model = CLIPTextModel(OPENAI_CLIP_VIT_L_14, clip_skip);
            text_model.init(params_ctx, wtype);
        } else if (version == VERSION_2_x) {
            text_model = CLIPTextModel(OPEN_CLIP_VIT_H_14, clip_skip);
            text_model.init(params_ctx, wtype);
        } else if (version == VERSION_XL) {
            text_model  = CLIPTextModel(OPENAI_CLIP_VIT_L_14, clip_skip, false);
            text_model2 = CLIPTextModel(OPEN_CLIP_VIT_BIGG_14, clip_skip, false);
            text_model.init(params_ctx, wtype);
            text_model2.init(params_ctx, wtype);
        }
    }

    std::string get_desc() {
        return "clip";
    }

    void set_clip_skip(int clip_skip) {
        text_model.set_clip_skip(clip_skip);
        if (version == VERSION_XL) {
            text_model2.set_clip_skip(clip_skip);
        }
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        text_model.get_param_tensors(tensors, prefix + "transformer.text_model");
        if (version == VERSION_XL) {
            text_model2.get_param_tensors(tensors, prefix + "1.transformer.text_model");
        }
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
        auto on_load                  = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) {
            if (tensor_storage.ne[0] != text_model.hidden_size) {
                LOG_DEBUG("embedding wrong hidden size, got %i, expected %i", tensor_storage.ne[0], text_model.hidden_size);
                return false;
            }
            embd        = ggml_new_tensor_2d(embd_ctx, wtype, text_model.hidden_size, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
            *dst_tensor = embd;
            return true;
        };
        model_loader.load_tensors(on_load, NULL);
        readed_embeddings.push_back(embd_name);
        token_embed_custom.resize(token_embed_custom.size() + ggml_nbytes(embd));
        memcpy((void*)(token_embed_custom.data() + num_custom_embeddings * text_model.hidden_size * ggml_type_size(wtype)),
               embd->data,
               ggml_nbytes(embd));
        for (int i = 0; i < embd->ne[1]; i++) {
            bpe_tokens.push_back(text_model.vocab_size + num_custom_embeddings);
            // LOG_DEBUG("new custom token: %i", text_model.vocab_size + num_custom_embeddings);
            num_custom_embeddings++;
        }
        LOG_DEBUG("embedding '%s' applied, custom embeddings: %i", embd_name.c_str(), num_custom_embeddings);
        return true;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                struct ggml_tensor* input_ids2,
                                struct ggml_tensor* embeddings,
                                size_t max_token_idx = 0,
                                bool return_pooled   = false) {
        size_t N       = input_ids->ne[1];
        size_t n_token = input_ids->ne[0];
        if (input_ids != NULL && input_ids->ne[0] > text_model.n_token) {
            GGML_ASSERT(input_ids->ne[0] % text_model.n_token == 0);
            input_ids = ggml_reshape_2d(ctx, input_ids, text_model.n_token, input_ids->ne[0] / text_model.n_token);
        }
        if (input_ids2 != NULL && input_ids2->ne[0] > text_model2.n_token) {
            GGML_ASSERT(input_ids2->ne[0] % text_model2.n_token == 0);
            input_ids2 = ggml_reshape_2d(ctx, input_ids2, text_model2.n_token, input_ids2->ne[0] / text_model2.n_token);
        }

        if (return_pooled) {
            return text_model2.forward(ctx, input_ids2, NULL, max_token_idx, return_pooled);
        }

        auto hidden_states = text_model.forward(ctx, input_ids, embeddings);  // [N, n_token, hidden_size]
        // LOG_DEBUG("hidden_states: %d %d %d %d", hidden_states->ne[0], hidden_states->ne[1], hidden_states->ne[2], hidden_states->ne[3]);
        if (version == VERSION_XL) {
            hidden_states = ggml_reshape_4d(ctx,
                                            hidden_states,
                                            hidden_states->ne[0],
                                            hidden_states->ne[1],
                                            hidden_states->ne[2],
                                            hidden_states->ne[3]);
            hidden_states = ggml_cont(ctx, ggml_permute(ctx, hidden_states, 2, 0, 1, 3));

            auto hidden_states2 = text_model2.forward(ctx, input_ids2, NULL);  // [N, n_token, hidden_size2]
            // LOG_DEBUG("hidden_states: %d %d %d %d", hidden_states->ne[0], hidden_states->ne[1], hidden_states->ne[2], hidden_states->ne[3]);
            hidden_states2 = ggml_reshape_4d(ctx,
                                             hidden_states2,
                                             hidden_states2->ne[0],
                                             hidden_states2->ne[1],
                                             hidden_states2->ne[2],
                                             hidden_states2->ne[3]);
            hidden_states2 = ggml_cont(ctx, ggml_permute(ctx, hidden_states2, 2, 0, 1, 3));

            hidden_states = ggml_concat(ctx, hidden_states, hidden_states2);  // [N, n_token, hidden_size + hidden_size2]

            hidden_states = ggml_cont(ctx, ggml_permute(ctx, hidden_states, 1, 2, 0, 3));
        }
        hidden_states = ggml_reshape_3d(ctx, hidden_states, hidden_states->ne[0], n_token, N);
        // LOG_DEBUG("hidden_states: %d %d %d %d", hidden_states->ne[0], hidden_states->ne[1], hidden_states->ne[2], hidden_states->ne[3]);
        return hidden_states;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_ids2 = NULL,
                                    size_t max_token_idx           = 0,
                                    bool return_pooled             = false) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        input_ids2 = to_backend(input_ids2);
        if (!return_pooled) {
            input_ids = to_backend(input_ids);
        }

        struct ggml_tensor* embeddings = NULL;

        if (num_custom_embeddings > 0 && version != VERSION_XL) {
            auto custom_embeddings = ggml_new_tensor_3d(compute_ctx,
                                                        wtype,
                                                        text_model.hidden_size,
                                                        1,
                                                        num_custom_embeddings);
            set_backend_tensor_data(custom_embeddings, token_embed_custom.data());

            auto token_embed_weight = text_model.get_token_embed_weight();
            token_embed_weight      = ggml_reshape_3d(compute_ctx, token_embed_weight, token_embed_weight->ne[0], 1, token_embed_weight->ne[1]);
            // concatenate custom embeddings
            embeddings = ggml_concat(compute_ctx, token_embed_weight, custom_embeddings);
            embeddings = ggml_reshape_2d(compute_ctx, embeddings, embeddings->ne[0], embeddings->ne[2]);
        }

        struct ggml_tensor* hidden_states = forward(compute_ctx, input_ids, input_ids2, embeddings, max_token_idx, return_pooled);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* input_ids,
                 struct ggml_tensor* input_ids2,
                 size_t max_token_idx,
                 bool return_pooled,
                 ggml_tensor** output,
                 ggml_context* output_ctx = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(input_ids, input_ids2, max_token_idx, return_pooled);
        };
        GGMLModule::compute(get_graph, n_threads, true, output, output_ctx);
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             bool padding = false) {
        return tokenize(text, text_model.n_token, padding);
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<bool>>
    tokenize_with_trigger_token(std::string text,
                                int num_input_imgs,
                                int32_t image_token,
                                bool padding = false) {
        return tokenize_with_trigger_token(text, num_input_imgs, image_token,
                                           text_model.n_token, padding);
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
                int pad_token_id = PAD_TOKEN_ID;
                if (version == VERSION_2_x) {
                    pad_token_id = 0;
                }
                tokens.insert(tokens.end(), length - tokens.size(), pad_token_id);
                weights.insert(weights.end(), length - weights.size(), 1.0);
            }
        }
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
        tokens.insert(tokens.begin(), BOS_TOKEN_ID);
        weights.insert(weights.begin(), 1.0);

        pad_tokens(tokens, weights, max_length, padding);

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

        pad_tokens(tokens, weights, max_length, padding);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights};
    }
};

struct FrozenCLIPVisionEmbedder : public GGMLModule {
    CLIPVisionModelProjection vision_model;

    FrozenCLIPVisionEmbedder(ggml_backend_t backend, ggml_type wtype)
        : vision_model(OPEN_CLIP_VIT_H_14, true), GGMLModule(backend, wtype) {
        vision_model.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "clip_vision";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        vision_model.get_param_tensors(tensors, prefix + "transformer");
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
        GGMLModule::compute(get_graph, n_threads, true, output, output_ctx);
    }
};

#endif  // __CLIP_HPP__