#ifndef __T5_HPP__
#define __T5_HPP__

#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "darts.h"
#include "ggml_extend.hpp"
#include "llama.cpp/json.h"
#include "model.h"

// Port from: https://github.com/google/sentencepiece/blob/master/src/unigram_model.h
// and https://github.com/google/sentencepiece/blob/master/src/unigram_model.h.
// Original License: https://github.com/google/sentencepiece/blob/master/LICENSE
//
// Since tokenization is not the bottleneck in SD, performance was not a major consideration
// during the migration.
class MetaspacePreTokenizer {
private:
    std::string replacement;
    bool add_prefix_space;

public:
    MetaspacePreTokenizer(const std::string replacement = " ", bool add_prefix_space = true)
        : replacement(replacement), add_prefix_space(add_prefix_space) {}

    std::string tokenize(const std::string& input) const {
        std::string tokens;
        std::stringstream ss(input);

        if (add_prefix_space) {
            tokens += replacement;
        }

        std::string token;
        bool firstToken = true;
        while (std::getline(ss, token, ' ')) {
            if (!firstToken)
                tokens += replacement + token;
            else
                tokens += token;

            firstToken = false;
        }

        return tokens;
    }
};

using EncodeResult = std::vector<std::pair<std::string, int>>;
class T5UniGramTokenizer {
public:
    enum Status {
        OK,
        NO_PIECES_LOADED,
        NO_ENTRY_FOUND,
        BUILD_DOUBLE_ARRAY_FAILED,
        PIECE_ALREADY_DEFINED,
        INVLIAD_JSON
    };

protected:
    MetaspacePreTokenizer pre_tokenizer;

    // all <piece, score> pairs
    std::vector<std::pair<std::string, float>> piece_score_pairs;

    float min_score_ = 0.0;
    float max_score_ = 0.0;
    std::unique_ptr<Darts::DoubleArray> trie_;

    // Maximum size of the return value of Trie, which corresponds
    // to the maximum size of shared common prefix in the sentence pieces.
    int trie_results_size_;
    // unknown id.
    int unk_id_            = 2;
    std::string eos_token_ = "</s>";
    int eos_id_            = 1;
    int pad_id_            = 0;
    // status.
    Status status_ = OK;

    float kUnkPenalty = 10.0;

    std::string replacement;
    bool add_prefix_space = true;

    void InitializePieces(const std::string& json_str) {
        nlohmann::json data;

        try {
            data = nlohmann::json::parse(json_str);
        } catch (const nlohmann::json::parse_error& e) {
            status_ = INVLIAD_JSON;
            return;
        }
        if (!data.contains("model")) {
            status_ = INVLIAD_JSON;
            return;
        }
        nlohmann::json model = data["model"];
        if (!model.contains("vocab")) {
            status_ = INVLIAD_JSON;
            return;
        }
        if (model.contains("unk_id")) {
            unk_id_ = model["unk_id"];
        }

        replacement      = data["pre_tokenizer"]["replacement"];
        add_prefix_space = data["pre_tokenizer"]["add_prefix_space"];

        pre_tokenizer = MetaspacePreTokenizer(replacement, add_prefix_space);

        for (const auto& item : model["vocab"]) {
            if (item.size() != 2 || !item[0].is_string() || !item[1].is_number_float()) {
                status_ = INVLIAD_JSON;
                return;
            }
            std::string piece = item[0];
            float score       = item[1];
            piece_score_pairs.emplace_back(piece, score);
        }
    }

    // Builds a Trie index.
    void BuildTrie(std::vector<std::pair<std::string, int>>* pieces) {
        if (status_ != OK)
            return;

        if (pieces->empty()) {
            status_ = NO_PIECES_LOADED;
            return;
        }

        // sort by sentencepiece since DoubleArray::build()
        // only accepts sorted strings.
        sort(pieces->begin(), pieces->end());

        // Makes key/value set for DoubleArrayTrie.
        std::vector<const char*> key(pieces->size());
        std::vector<int> value(pieces->size());
        for (size_t i = 0; i < pieces->size(); ++i) {
            key[i]   = (*pieces)[i].first.data();  // sorted piece.
            value[i] = (*pieces)[i].second;        // vocab_id
        }

        trie_ = std::unique_ptr<Darts::DoubleArray>(new Darts::DoubleArray());
        if (trie_->build(key.size(), const_cast<char**>(&key[0]), nullptr,
                         &value[0]) != 0) {
            status_ = BUILD_DOUBLE_ARRAY_FAILED;
            return;
        }

        // Computes the maximum number of shared prefixes in the trie.
        const int kMaxTrieResultsSize = 1024;
        std::vector<Darts::DoubleArray::result_pair_type> results(
            kMaxTrieResultsSize);
        trie_results_size_ = 0;
        for (const auto& p : *pieces) {
            const int num_nodes = trie_->commonPrefixSearch(
                p.first.data(), results.data(), results.size(), p.first.size());
            trie_results_size_ = std::max(trie_results_size_, num_nodes);
        }

        if (trie_results_size_ == 0)
            status_ = NO_ENTRY_FOUND;
    }

    // Non-virtual (inlined) implementation for faster execution.
    inline float GetScoreInlined(int id) const {
        return piece_score_pairs[id].second;
    }

    inline bool IsUnusedInlined(int id) const {
        return false;  // TODO
    }

    inline bool IsUserDefinedInlined(int id) const {
        return false;  // TODO
    }

    inline size_t OneCharLen(const char* src) const {
        return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
    }

    // The optimized Viterbi encode.
    // Main differences from the original function:
    // 1. Memorizes the best path at each postion so far,
    // 2. No need to store the Lattice nodes,
    // 3. Works in utf-8 directly,
    // 4. Defines a new struct with fewer fields than Lattice,
    // 5. Does not depend on `class Lattice` nor call `SetSentence()`,
    // `PopulateNodes()`, or `Viterbi()`. It does everything in one function.
    // For detailed explanations please see the comments inside the function body.
    EncodeResult EncodeOptimized(const std::string& normalized) const {
        // An optimized Viterbi algorithm for unigram language models. Benchmarking
        // results show that it generates almost identical outputs and achieves 2.1x
        // speedup on average for 102 languages compared to the original
        // implementation. It's based on the following three ideas:
        //
        // 1. Because it uses the *unigram* model:
        //     best_score(x1, x2, …, xt) = best_score(x1, x2, …, x{t-1}) + score(xt)
        // Deciding the best path (and score) can be decoupled into two isolated
        // terms: (a) the best path ended before the last token `best_score(x1, x2, …,
        // x{t-1})`, and (b) the last token and its `score(xt)`. The two terms are
        // not related to each other at all.
        //
        // Therefore, we can compute once and store the *best_path ending at
        // each character position*. In this way, when we know best_path_ends_at[M],
        // we can reuse it to compute all the best_path_ends_at_[...] where the last
        // token starts at the same character position M.
        //
        // This improves the time complexity from O(n*k*k) to O(n*k) because it
        // eliminates the extra loop of recomputing the best path ending at the same
        // position, where n is the input length and k is the maximum number of tokens
        // that can be recognized starting at each position.
        //
        // 2. Again, because it uses the *unigram* model, we don’t need to actually
        // store the lattice nodes. We still recognize all the tokens and lattice
        // nodes from the input, but along identifying them, we use and discard them
        // on the fly. There is no need to actually store them for best path Viterbi
        // decoding. The only thing we need to store is the best_path ending at
        // each character position.
        //
        // This improvement reduces the things needed to store in memory from O(n*k)
        // to O(n), where n is the input length and k is the maximum number of tokens
        // that can be recognized starting at each position.
        //
        // It also avoids the need of dynamic-size lattice node pool, because the
        // number of things to store is fixed as n.
        //
        // 3. SentencePiece is designed to work with unicode, taking utf-8 encoding
        // inputs. In the original implementation, the lattice positions are based on
        // unicode positions. A mapping from unicode position to the utf-8 position is
        // maintained to recover the utf-8 string piece.
        //
        // We found that it is sufficient and beneficial to directly work with utf-8
        // positions:
        //
        // Firstly, it saves the conversion and mapping between unicode positions and
        // utf-8 positions.
        //
        // Secondly, it reduces the number of fields we need to maintain in the
        // node/path structure. Specifically, there are 8 fields defined in
        // `Lattice::Node` used by the original encoder, but here in the optimized
        // encoder we only need to define 3 fields in `BestPathNode`.

        if (status() != OK || normalized.empty()) {
            return {};
        }
        // Represents the last node of the best path.
        struct BestPathNode {
            int id = -1;  // The vocab id. (maybe -1 for UNK)
            float best_path_score =
                0;  // The total score of the best path ending at this node.
            int starts_at =
                -1;  // The starting position (in utf-8) of this node. The entire best
                     // path can be constructed by backtracking along this link.
        };
        const int size        = normalized.size();
        const float unk_score = min_score() - kUnkPenalty;
        // The ends are exclusive.
        std::vector<BestPathNode> best_path_ends_at(size + 1);
        // Generate lattice on-the-fly (not stored) and update best_path_ends_at.
        int starts_at = 0;
        while (starts_at < size) {
            std::size_t node_pos = 0;
            std::size_t key_pos  = starts_at;
            const auto best_path_score_till_here =
                best_path_ends_at[starts_at].best_path_score;
            bool has_single_node = false;
            const int mblen =
                std::min<int>(OneCharLen(normalized.data() + starts_at),
                              size - starts_at);
            while (key_pos < size) {
                const int ret =
                    trie_->traverse(normalized.data(), node_pos, key_pos, key_pos + 1);
                if (ret == -2)
                    break;
                if (ret >= 0) {
                    if (IsUnusedInlined(ret))
                        continue;
                    // Update the best path node.
                    auto& target_node = best_path_ends_at[key_pos];
                    const auto length = (key_pos - starts_at);
                    // User defined symbol receives extra bonus to always be selected.
                    const auto score = IsUserDefinedInlined(ret)
                                           ? (length * max_score_ - 0.1)
                                           : GetScoreInlined(ret);
                    const auto candidate_best_path_score =
                        score + best_path_score_till_here;
                    if (target_node.starts_at == -1 ||
                        candidate_best_path_score > target_node.best_path_score) {
                        target_node.best_path_score = candidate_best_path_score;
                        target_node.starts_at       = starts_at;
                        target_node.id              = ret;
                    }
                    if (!has_single_node && length == mblen) {
                        has_single_node = true;
                    }
                }
            }
            if (!has_single_node) {
                auto& target_node = best_path_ends_at[starts_at + mblen];
                const auto candidate_best_path_score =
                    unk_score + best_path_score_till_here;
                if (target_node.starts_at == -1 ||
                    candidate_best_path_score > target_node.best_path_score) {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at       = starts_at;
                    target_node.id              = unk_id_;
                }
            }
            // Move by one unicode character.
            starts_at += mblen;
        }
        // Backtrack to identify the best path.
        EncodeResult results;
        int ends_at = size;
        while (ends_at > 0) {
            const auto& node = best_path_ends_at[ends_at];
            results.emplace_back(
                normalized.substr(node.starts_at, ends_at - node.starts_at), node.id);
            ends_at = node.starts_at;
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

public:
    explicit T5UniGramTokenizer(const std::string& json_str = "") {
        if (json_str.size() != 0) {
            InitializePieces(json_str);
        } else {
            InitializePieces(ModelLoader::load_t5_tokenizer_json());
        }

        min_score_ = FLT_MAX;
        max_score_ = FLT_MIN;

        std::vector<std::pair<std::string, int>> pieces;
        for (int i = 0; i < piece_score_pairs.size(); i++) {
            const auto& sp = piece_score_pairs[i];

            min_score_ = std::min(min_score_, sp.second);
            max_score_ = std::max(max_score_, sp.second);

            pieces.emplace_back(sp.first, i);
        }

        BuildTrie(&pieces);
    }
    ~T5UniGramTokenizer(){};

    std::string Normalize(const std::string& input) const {
        // Ref: https://github.com/huggingface/tokenizers/blob/1ff56c0c70b045f0cd82da1af9ac08cd4c7a6f9f/bindings/python/py_src/tokenizers/implementations/sentencepiece_unigram.py#L29
        // TODO: nmt-nfkc
        std::string normalized = std::regex_replace(input, std::regex(" {2,}"), " ");
        return normalized;
    }

    std::vector<int> Encode(const std::string& input, bool append_eos_if_not_present = true) const {
        std::string normalized = Normalize(input);
        normalized             = pre_tokenizer.tokenize(normalized);
        EncodeResult result    = EncodeOptimized(normalized);
        if (result.size() > 0 && append_eos_if_not_present) {
            auto item = result[result.size() - 1];
            if (item.first != eos_token_) {
                result.emplace_back(eos_token_, eos_id_);
            }
        }
        std::vector<int> tokens;
        for (auto item : result) {
            tokens.push_back(item.second);
        }
        return tokens;
    }

    void pad_tokens(std::vector<int>& tokens,
                    std::vector<float>& weights,
                    size_t max_length = 0,
                    bool padding      = false) {
        if (max_length > 0 && padding) {
            size_t orig_token_num = tokens.size() - 1;
            size_t n              = std::ceil(orig_token_num * 1.0 / (max_length - 1));
            if (n == 0) {
                n = 1;
            }
            size_t length = max_length * n;
            LOG_DEBUG("token length: %llu", length);
            std::vector<int> new_tokens;
            std::vector<float> new_weights;
            int token_idx = 0;
            for (int i = 0; i < length; i++) {
                if (token_idx >= orig_token_num) {
                    break;
                }
                if (i % max_length == max_length - 1) {
                    new_tokens.push_back(eos_id_);
                    new_weights.push_back(1.0);
                } else {
                    new_tokens.push_back(tokens[token_idx]);
                    new_weights.push_back(weights[token_idx]);
                    token_idx++;
                }
            }

            new_tokens.push_back(eos_id_);
            new_weights.push_back(1.0);
            tokens  = new_tokens;
            weights = new_weights;

            if (padding) {
                int pad_token_id = pad_id_;
                tokens.insert(tokens.end(), length - tokens.size(), pad_token_id);
                weights.insert(weights.end(), length - weights.size(), 1.0);
            }
        }
    }

    // Returns the minimum score in sentence pieces.
    // min_score() - 10 is used for the cost of unknown sentence.
    float min_score() const { return min_score_; }

    // Returns the maximum score in sentence pieces.
    // max_score() is used for the cost of user defined symbols.
    float max_score() const { return max_score_; }

    Status status() const { return status_; }
};

class T5LayerNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    }

public:
    T5LayerNorm(int64_t hidden_size,
                float eps = 1e-06f)
        : hidden_size(hidden_size),
          eps(eps) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        x                     = ggml_norm_ext(ctx, x, eps, false);
        x                     = ggml_mul(ctx, x, w);
        return x;
    }
};

struct T5DenseActDense : public UnaryBlock {
public:
    T5DenseActDense(int64_t model_dim, int64_t ff_dim) {
        blocks["wi"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, ff_dim, false));
        blocks["wo"] = std::shared_ptr<GGMLBlock>(new Linear(ff_dim, model_dim, false));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, model_dim]
        auto wi = std::dynamic_pointer_cast<Linear>(blocks["wi"]);
        auto wo = std::dynamic_pointer_cast<Linear>(blocks["wo"]);

        x = wi->forward(ctx, x);
        x = ggml_relu_inplace(ctx, x);
        x = wo->forward(ctx, x);
        return x;
    }
};

struct T5DenseGatedActDense : public UnaryBlock {
public:
    T5DenseGatedActDense(int64_t model_dim, int64_t ff_dim) {
        blocks["wi_0"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, ff_dim, false));
        blocks["wi_1"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, ff_dim, false));
        blocks["wo"]   = std::shared_ptr<GGMLBlock>(new Linear(ff_dim, model_dim, false));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, model_dim]
        auto wi_0 = std::dynamic_pointer_cast<Linear>(blocks["wi_0"]);
        auto wi_1 = std::dynamic_pointer_cast<Linear>(blocks["wi_1"]);
        auto wo   = std::dynamic_pointer_cast<Linear>(blocks["wo"]);

        auto hidden_gelu   = ggml_gelu_inplace(ctx, wi_0->forward(ctx, x));
        auto hidden_linear = wi_1->forward(ctx, x);
        x                  = ggml_mul_inplace(ctx, hidden_gelu, hidden_linear);
        x                  = wo->forward(ctx, x);
        return x;
    }
};

struct T5LayerFF : public UnaryBlock {
public:
    T5LayerFF(int64_t model_dim, int64_t ff_dim) {
        blocks["DenseReluDense"] = std::shared_ptr<GGMLBlock>(new T5DenseGatedActDense(model_dim, ff_dim));
        blocks["layer_norm"]     = std::shared_ptr<GGMLBlock>(new T5LayerNorm(model_dim));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, model_dim]
        auto DenseReluDense = std::dynamic_pointer_cast<T5DenseGatedActDense>(blocks["DenseReluDense"]);
        auto layer_norm     = std::dynamic_pointer_cast<T5LayerNorm>(blocks["layer_norm"]);

        auto forwarded_states = layer_norm->forward(ctx, x);
        forwarded_states      = DenseReluDense->forward(ctx, forwarded_states);
        x                     = ggml_add_inplace(ctx, forwarded_states, x);
        return x;
    }
};

class T5Attention : public GGMLBlock {
protected:
    int64_t model_dim;
    int64_t inner_dim;
    int64_t num_heads;
    bool using_relative_attention_bias;
    int64_t relative_attention_num_buckets  = 32;
    int64_t relative_attention_max_distance = 128;

public:
    T5Attention(int64_t model_dim,
                int64_t inner_dim,
                int64_t num_heads,
                bool using_relative_attention_bias = false)
        : model_dim(model_dim),
          inner_dim(inner_dim),
          num_heads(num_heads),
          using_relative_attention_bias(using_relative_attention_bias) {
        blocks["q"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, inner_dim, false));
        blocks["k"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, inner_dim, false));
        blocks["v"] = std::shared_ptr<GGMLBlock>(new Linear(model_dim, inner_dim, false));
        blocks["o"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, model_dim, false));
        if (using_relative_attention_bias) {
            blocks["relative_attention_bias"] = std::shared_ptr<GGMLBlock>(new Embedding(relative_attention_num_buckets, num_heads));
        }
    }

    struct ggml_tensor* compute_bias(struct ggml_context* ctx,
                                     struct ggml_tensor* relative_position_bucket) {
        auto relative_attention_bias = std::dynamic_pointer_cast<Embedding>(blocks["relative_attention_bias"]);

        auto values = relative_attention_bias->forward(ctx, relative_position_bucket);  // shape (query_length, key_length, num_heads)
        values      = ggml_cont(ctx, ggml_permute(ctx, values, 2, 0, 1, 3));            // shape (1, num_heads, query_length, key_length)
        return values;
    }

    // x: [N, n_token, model_dim]
    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(struct ggml_context* ctx,
                                                                struct ggml_tensor* x,
                                                                struct ggml_tensor* past_bias                = NULL,
                                                                struct ggml_tensor* mask                     = NULL,
                                                                struct ggml_tensor* relative_position_bucket = NULL) {
        auto q_proj   = std::dynamic_pointer_cast<Linear>(blocks["q"]);
        auto k_proj   = std::dynamic_pointer_cast<Linear>(blocks["k"]);
        auto v_proj   = std::dynamic_pointer_cast<Linear>(blocks["v"]);
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);

        int64_t n_head = num_heads;
        int64_t d_head = inner_dim / n_head;

        auto q = q_proj->forward(ctx, x);
        auto k = k_proj->forward(ctx, x);
        auto v = v_proj->forward(ctx, x);

        if (using_relative_attention_bias && relative_position_bucket != NULL) {
            past_bias = compute_bias(ctx, relative_position_bucket);
        }
        if (past_bias != NULL) {
            if (mask != NULL) {
                mask = ggml_add(ctx, mask, past_bias);
            } else {
                mask = past_bias;
            }
        }

        k = ggml_scale_inplace(ctx, k, sqrt(d_head));

        x = ggml_nn_attention_ext(ctx, q, k, v, num_heads, mask);  // [N, n_token, d_head * n_head]

        x = out_proj->forward(ctx, x);  // [N, n_token, model_dim]
        return {x, past_bias};
    }
};

struct T5LayerSelfAttention : public GGMLBlock {
public:
    T5LayerSelfAttention(int64_t model_dim,
                         int64_t inner_dim,
                         int64_t ff_dim,
                         int64_t num_heads,
                         bool using_relative_attention_bias) {
        blocks["SelfAttention"] = std::shared_ptr<GGMLBlock>(new T5Attention(model_dim, inner_dim, num_heads, using_relative_attention_bias));
        blocks["layer_norm"]    = std::shared_ptr<GGMLBlock>(new T5LayerNorm(model_dim));
    }

    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(struct ggml_context* ctx,
                                                                struct ggml_tensor* x,
                                                                struct ggml_tensor* past_bias                = NULL,
                                                                struct ggml_tensor* mask                     = NULL,
                                                                struct ggml_tensor* relative_position_bucket = NULL) {
        // x: [N, n_token, model_dim]
        auto SelfAttention = std::dynamic_pointer_cast<T5Attention>(blocks["SelfAttention"]);
        auto layer_norm    = std::dynamic_pointer_cast<T5LayerNorm>(blocks["layer_norm"]);

        auto normed_hidden_state = layer_norm->forward(ctx, x);
        auto ret                 = SelfAttention->forward(ctx, normed_hidden_state, past_bias, mask, relative_position_bucket);
        auto output              = ret.first;
        past_bias                = ret.second;

        x = ggml_add_inplace(ctx, output, x);
        return {x, past_bias};
    }
};

struct T5Block : public GGMLBlock {
public:
    T5Block(int64_t model_dim, int64_t inner_dim, int64_t ff_dim, int64_t num_heads, bool using_relative_attention_bias) {
        blocks["layer.0"] = std::shared_ptr<GGMLBlock>(new T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, using_relative_attention_bias));
        blocks["layer.1"] = std::shared_ptr<GGMLBlock>(new T5LayerFF(model_dim, ff_dim));
    }

    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(struct ggml_context* ctx,
                                                                struct ggml_tensor* x,
                                                                struct ggml_tensor* past_bias                = NULL,
                                                                struct ggml_tensor* mask                     = NULL,
                                                                struct ggml_tensor* relative_position_bucket = NULL) {
        // x: [N, n_token, model_dim]
        auto layer_0 = std::dynamic_pointer_cast<T5LayerSelfAttention>(blocks["layer.0"]);
        auto layer_1 = std::dynamic_pointer_cast<T5LayerFF>(blocks["layer.1"]);

        auto ret  = layer_0->forward(ctx, x, past_bias, mask, relative_position_bucket);
        x         = ret.first;
        past_bias = ret.second;
        x         = layer_1->forward(ctx, x);
        return {x, past_bias};
    }
};

struct T5Stack : public GGMLBlock {
    int64_t num_layers;

public:
    T5Stack(int64_t num_layers,
            int64_t model_dim,
            int64_t inner_dim,
            int64_t ff_dim,
            int64_t num_heads)
        : num_layers(num_layers) {
        for (int i = 0; i < num_layers; i++) {
            blocks["block." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new T5Block(model_dim, inner_dim, ff_dim, num_heads, i == 0));
        }

        blocks["final_layer_norm"] = std::shared_ptr<GGMLBlock>(new T5LayerNorm(model_dim));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* past_bias                = NULL,
                                struct ggml_tensor* attention_mask           = NULL,
                                struct ggml_tensor* relative_position_bucket = NULL) {
        // x: [N, n_token, model_dim]
        for (int i = 0; i < num_layers; i++) {
            auto block = std::dynamic_pointer_cast<T5Block>(blocks["block." + std::to_string(i)]);

            auto ret  = block->forward(ctx, x, past_bias, attention_mask, relative_position_bucket);
            x         = ret.first;
            past_bias = ret.second;
        }

        auto final_layer_norm = std::dynamic_pointer_cast<T5LayerNorm>(blocks["final_layer_norm"]);

        x = final_layer_norm->forward(ctx, x);
        return x;
    }
};

struct T5 : public GGMLBlock {
public:
    T5(int64_t num_layers,
       int64_t model_dim,
       int64_t ff_dim,
       int64_t num_heads,
       int64_t vocab_size) {
        blocks["encoder"] = std::shared_ptr<GGMLBlock>(new T5Stack(num_layers, model_dim, model_dim, ff_dim, num_heads));
        blocks["shared"]  = std::shared_ptr<GGMLBlock>(new Embedding(vocab_size, model_dim));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                struct ggml_tensor* past_bias                = NULL,
                                struct ggml_tensor* attention_mask           = NULL,
                                struct ggml_tensor* relative_position_bucket = NULL) {
        // input_ids: [N, n_token]

        auto shared  = std::dynamic_pointer_cast<Embedding>(blocks["shared"]);
        auto encoder = std::dynamic_pointer_cast<T5Stack>(blocks["encoder"]);

        auto x = shared->forward(ctx, input_ids);
        x      = encoder->forward(ctx, x, past_bias, attention_mask, relative_position_bucket);
        return x;
    }
};

struct T5Runner : public GGMLRunner {
    T5 model;
    std::vector<int> relative_position_bucket_vec;

    T5Runner(ggml_backend_t backend,
             ggml_type wtype,
             int64_t num_layers = 24,
             int64_t model_dim  = 4096,
             int64_t ff_dim     = 10240,
             int64_t num_heads  = 64,
             int64_t vocab_size = 32128)
        : GGMLRunner(backend, wtype), model(num_layers, model_dim, ff_dim, num_heads, vocab_size) {
        model.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "t5";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                struct ggml_tensor* relative_position_bucket) {
        size_t N       = input_ids->ne[1];
        size_t n_token = input_ids->ne[0];

        auto hidden_states = model.forward(ctx, input_ids, NULL, NULL, relative_position_bucket);  // [N, n_token, model_dim]
        return hidden_states;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        input_ids = to_backend(input_ids);

        relative_position_bucket_vec = compute_relative_position_bucket(input_ids->ne[0], input_ids->ne[0]);

        // for (int i = 0; i < relative_position_bucket_vec.size(); i++) {
        //     if (i % 77 == 0) {
        //         printf("\n");
        //     }
        //     printf("%d ", relative_position_bucket_vec[i]);
        // }

        auto relative_position_bucket = ggml_new_tensor_2d(compute_ctx,
                                                           GGML_TYPE_I32,
                                                           input_ids->ne[0],
                                                           input_ids->ne[0]);
        set_backend_tensor_data(relative_position_bucket, relative_position_bucket_vec.data());

        struct ggml_tensor* hidden_states = forward(compute_ctx, input_ids, relative_position_bucket);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* input_ids,
                 ggml_tensor** output,
                 ggml_context* output_ctx = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(input_ids);
        };
        GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
    }

    static std::vector<int> _relative_position_bucket(const std::vector<int>& relative_position,
                                                      bool bidirectional = true,
                                                      int num_buckets    = 32,
                                                      int max_distance   = 128) {
        std::vector<int> relative_buckets(relative_position.size(), 0);
        std::vector<int> abs_relative_position = relative_position;

        if (bidirectional) {
            num_buckets = num_buckets / 2;
            for (size_t i = 0; i < relative_position.size(); ++i) {
                if (relative_position[i] > 0) {
                    relative_buckets[i] += num_buckets;
                }
                abs_relative_position[i] = std::abs(relative_position[i]);
            }
        } else {
            for (size_t i = 0; i < relative_position.size(); ++i) {
                abs_relative_position[i] = std::max(-relative_position[i], 0);
            }
        }

        int max_exact = num_buckets / 2;
        std::vector<int> relative_position_if_large(relative_position.size(), 0);

        for (size_t i = 0; i < relative_position.size(); ++i) {
            if (abs_relative_position[i] < max_exact) {
                relative_buckets[i] += abs_relative_position[i];
            } else {
                float log_pos                 = std::log(static_cast<float>(abs_relative_position[i]) / max_exact);
                float log_base                = std::log(static_cast<float>(max_distance) / max_exact);
                relative_position_if_large[i] = max_exact + static_cast<int>((log_pos / log_base) * (num_buckets - max_exact));
                relative_position_if_large[i] = std::min(relative_position_if_large[i], num_buckets - 1);
                relative_buckets[i] += relative_position_if_large[i];
            }
        }

        return relative_buckets;
    }

    std::vector<int> compute_relative_position_bucket(int query_length,
                                                      int key_length) {
        std::vector<int> context_position(query_length);
        std::vector<int> memory_position(key_length);

        for (int i = 0; i < query_length; ++i) {
            context_position[i] = i;
        }
        for (int i = 0; i < key_length; ++i) {
            memory_position[i] = i;
        }

        std::vector<std::vector<int>> relative_position(query_length, std::vector<int>(key_length, 0));
        for (int i = 0; i < query_length; ++i) {
            for (int j = 0; j < key_length; ++j) {
                relative_position[i][j] = memory_position[j] - context_position[i];
            }
        }

        std::vector<int> relative_position_bucket;
        for (int i = 0; i < query_length; ++i) {
            std::vector<int> result = _relative_position_bucket(relative_position[i], true);
            relative_position_bucket.insert(relative_position_bucket.end(), result.begin(), result.end());
        }

        return relative_position_bucket;
    }
};

struct T5Embedder {
    T5UniGramTokenizer tokenizer;
    T5Runner model;

    T5Embedder(ggml_backend_t backend,
               ggml_type wtype,
               int64_t num_layers = 24,
               int64_t model_dim  = 4096,
               int64_t ff_dim     = 10240,
               int64_t num_heads  = 64,
               int64_t vocab_size = 32128)
        : model(backend, wtype, num_layers, model_dim, ff_dim, num_heads, vocab_size) {
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    void alloc_params_buffer() {
        model.alloc_params_buffer();
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

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer.Encode(curr_text, false);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        int EOS_TOKEN_ID = 1;
        tokens.push_back(EOS_TOKEN_ID);
        weights.push_back(1.0);

        tokenizer.pad_tokens(tokens, weights, max_length, padding);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights};
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
            // cuda f16: nan
            // cuda f32: pass
            // cuda q8_0: nan
            // TODO: fix cuda nan
            std::string text("a lovely cat");
            auto tokens_and_weights     = tokenize(text, 77, true);
            std::vector<int>& tokens    = tokens_and_weights.first;
            std::vector<float>& weights = tokens_and_weights.second;
            for (auto token : tokens) {
                printf("%d ", token);
            }
            printf("\n");
            auto input_ids          = vector_to_ggml_tensor_i32(work_ctx, tokens);
            struct ggml_tensor* out = NULL;

            int t0 = ggml_time_ms();
            model.compute(8, input_ids, &out, work_ctx);
            int t1 = ggml_time_ms();

            print_ggml_tensor(out);
            LOG_DEBUG("t5 test done in %dms", t1 - t0);
        }
    }

    static void load_from_file_and_test(const std::string& file_path) {
        // ggml_backend_t backend    = ggml_backend_cuda_init(0);
        ggml_backend_t backend         = ggml_backend_cpu_init();
        ggml_type model_data_type      = GGML_TYPE_F32;
        std::shared_ptr<T5Embedder> t5 = std::shared_ptr<T5Embedder>(new T5Embedder(backend, model_data_type));
        {
            LOG_INFO("loading from '%s'", file_path.c_str());

            t5->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            t5->get_param_tensors(tensors, "");

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

            LOG_INFO("t5 model loaded");
        }
        t5->test();
    }
};

#endif  // __T5_HPP__
