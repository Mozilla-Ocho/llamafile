// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include "llama.cpp/common.h"
#include "llama.cpp/llama.h"
#include "llama.cpp/grammar-parser.h"
#include "llama.cpp/llava/llava.h"
#include "llama.cpp/stb_image.h"
#include "utils.h"
#include "oai.h"
#include "llamafile/micros.h"
#include "llamafile/llamafile.h"
#include "macsandbox.h"

// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
#include "httplib.h"
#include "llama.cpp/json.h"

#include <cstddef>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include <signal.h>
#include <libc/calls/pledge.h>
#include <tool/args/args.h>
#include <libc/dce.h>

double g_prompt_per_second_jart;

using json = nlohmann::json;

struct server_params
{
    std::string hostname = "127.0.0.1";
    std::vector<std::string> api_keys;
    std::string public_path = "/zip/llama.cpp/server/public";
    std::string chat_template = "";
    int32_t port = 8080;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
    bool nobrowser = false;
    bool slots_endpoint = true;
    bool metrics_endpoint = false;
};

bool server_verbose = false;
bool server_log_json = true;

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }
    return i;
}

enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

static bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text)
{
    if (!text.empty() && !stop.empty())
    {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--)
        {
            if (stop[char_index] == text_last_char)
            {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial))
                {
                    return text.size() - char_index - 1;
                }
            }
        }
    }
    return std::string::npos;
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context *ctx, Iter begin, Iter end)
{
    std::string ret;
    for (; begin != end; ++begin)
    {
        ret += llama_token_to_piece(ctx, *begin, false);
    }
    return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token, false);
    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }
    return out;
}

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context *ctx, const std::vector<completion_token_output> &probs)
{
    json out = json::array();
    for (const auto &prob : probs)
    {
        json probs_for_token = json::array();
        for (const auto &p : prob.probs)
        {
            std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
            probs_for_token.push_back(json
            {
                {"tok_str", tok_str},
                {"prob",    p.prob},
            });
        }
        std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
        out.push_back(json{
            {"content", tok_str},
            {"probs",   probs_for_token},
        });
    }
    return out;
}

struct llama_client_slot
{
    int id;
    int task_id = -1;

    struct slot_params params;

    slot_state state = IDLE;
    slot_command command = NONE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_past      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;
    int32_t n_predict   = -1;

    int32_t num_prompt_tokens           = 0;
    int32_t num_prompt_tokens_processed = 0;

    json prompt;
    std::string generated_text;
    llama_token sampled;
    std::vector<llama_token> cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    bool infill = false;
    bool embedding = false;
    bool has_next_token = true;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;

    bool oaicompat = false;
    std::string oaicompat_model;

    std::string stopping_word;

    // sampling
    struct llama_sampling_params sparams;
    llama_sampling_context *ctx_sampling = nullptr;

    int32_t ga_i = 0;   // group-attention state
    int32_t ga_n = 1;   // group-attention factor
    int32_t ga_w = 512; // group-attention width

    int32_t n_past_se = 0; // self-extend

    // multimodal
    std::vector<slot_image> images;

    // stats
    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_genereration;

    double t_prompt_processing; // ms
    double t_token_generation; // ms

    // multitasks
    int multitask_id = -1;

    void reset() {
        num_prompt_tokens      = 0;
        generated_text         = "";
        truncated              = false;
        stopped_eos            = false;
        stopped_word           = false;
        stopped_limit          = false;
        stopping_word          = "";
        n_past                 = 0;
        sent_count             = 0;
        sent_token_probs_index = 0;
        infill                 = false;
        ga_i                   = 0;
        n_past_se              = 0;

        generated_token_probs.clear();

        for (slot_image & img : images)
        {
            free(img.image_embedding);
            if (img.img_data) {
                clip_image_u8_free(img.img_data);
            }
            img.prefix_prompt = "";
        }

        images.clear();
    }

    bool has_budget(gpt_params &global_params) {
        if (params.n_predict == -1 && global_params.n_predict == -1)
        {
            return true; // limitless
        }

        n_remaining = -1;

        if (params.n_predict != -1)
        {
            n_remaining = params.n_predict - n_decoded;
        }
        else if (global_params.n_predict != -1)
        {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool available() const {
        return state == IDLE && command == NONE;
    }

    bool is_processing() const {
        return (state == IDLE && command == LOAD_PROMPT) || state == PROCESSING;
    }

    void add_token_string(const completion_token_output &token) {
        if (command == RELEASE)
        {
            return;
        }
        cache_tokens.push_back(token.tok);
        generated_token_probs.push_back(token);
    }

    void release() {
        if (state == PROCESSING)
        {
            t_token_generation = (ggml_time_us() - t_start_genereration) / 1e3;
            command = RELEASE;
        }
    }

    json get_formated_timings() {
        return json
        {
            {"prompt_n",               num_prompt_tokens_processed},
            {"prompt_ms",              t_prompt_processing},
            {"prompt_per_token_ms",    t_prompt_processing / num_prompt_tokens_processed},
            {"prompt_per_second",      1e3 / t_prompt_processing * num_prompt_tokens_processed},
            {"prompt_per_second_jart", g_prompt_per_second_jart},

            {"predicted_n",            n_decoded},
            {"predicted_ms",           t_token_generation},
            {"predicted_per_token_ms", t_token_generation / n_decoded},
            {"predicted_per_second",   1e3 / t_token_generation * n_decoded},
        };
    }

    void print_timings() const {
       char buffer[512];
        double t_token = t_prompt_processing / num_prompt_tokens_processed;
        double n_tokens_second = 1e3 / t_prompt_processing * num_prompt_tokens_processed;
        sprintf(buffer, "prompt eval time     = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)",
                t_prompt_processing, num_prompt_tokens_processed,
                t_token, n_tokens_second);
        LOG_INFO(buffer, {
            {"slot_id",                     id},
            {"task_id",                     task_id},
            {"t_prompt_processing",         t_prompt_processing},
            {"num_prompt_tokens_processed", num_prompt_tokens_processed},
            {"t_token",                     t_token},
            {"n_tokens_second",             n_tokens_second},
        });

        t_token = t_token_generation / n_decoded;
        n_tokens_second = 1e3 / t_token_generation * n_decoded;
        sprintf(buffer, "generation eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)",
                t_token_generation, n_decoded,
                t_token, n_tokens_second);
        LOG_INFO(buffer, {
            {"slot_id",            id},
            {"task_id",            task_id},
            {"t_token_generation", t_token_generation},
            {"n_decoded",          n_decoded},
            {"t_token",            t_token},
            {"n_tokens_second",    n_tokens_second},
        });

        sprintf(buffer, "          total time = %10.2f ms", t_prompt_processing + t_token_generation);
        LOG_INFO(buffer, {
            {"slot_id",             id},
            {"task_id",             task_id},
            {"t_prompt_processing", t_prompt_processing},
            {"t_token_generation",  t_token_generation},
            {"t_total",             t_prompt_processing + t_token_generation},
        });
    }
};

struct llama_metrics {
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t n_tokens_predicted_total        = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted       = 0;
    uint64_t t_tokens_generation      = 0;


    void on_prompt_eval(const llama_client_slot &slot) {
        n_prompt_tokens_processed_total += slot.num_prompt_tokens_processed;

        n_prompt_tokens_processed += slot.num_prompt_tokens_processed;
        t_prompt_processing       += slot.t_prompt_processing;
    }

    void on_prediction(const llama_client_slot &slot) {
        n_tokens_predicted_total += slot.n_decoded;

        n_tokens_predicted  += slot.n_decoded;
        t_tokens_generation += slot.t_token_generation;
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};

struct llama_server_context
{
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;

    clip_ctx *clp_ctx = nullptr;

    gpt_params params;

    llama_batch batch;

    bool multimodal         = false;
    bool clean_kv_cache     = true;
    bool all_slots_are_idle = false;
    bool add_bos_token      = true;

    int32_t n_ctx;  // total context for all clients / slots

    // system prompt
    bool system_need_update = false;

    std::string              system_prompt;
    std::vector<llama_token> system_tokens;

    std::string name_user;      // this should be the antiprompt
    std::string name_assistant;

    // slots / clients
    std::vector<llama_client_slot> slots;
    json default_generation_settings_for_props;

    llama_server_queue queue_tasks;
    llama_server_response queue_results;

    llama_metrics metrics;

    ~llama_server_context()
    {
        if (ctx)
        {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model)
        {
            llama_free_model(model);
            model = nullptr;
        }
    }

    bool load_model(const gpt_params &params_)
    {
        params = params_;
        if (!params.mmproj.empty()) {
            multimodal = true;
            LOG_INFO("Multi Modal Mode Enabled", {});
            clp_ctx = clip_model_load(params.mmproj.c_str(), /*verbosity=*/ 1);
            if(clp_ctx == nullptr) {
                LOG_ERROR("unable to load clip model", {{"model", params.mmproj}});
                return false;
            }

            if (params.n_ctx < 2048) { // request larger context for the image embedding
                params.n_ctx = 2048;
            }
        }

        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        if (model == nullptr)
        {
            LOG_ERROR("unable to load model", {{"model", params.model}});
            return false;
        }

        if (multimodal) {
            const int n_embd_clip = clip_n_mmproj_embd(clp_ctx);
            const int n_embd_llm  = llama_n_embd(model);
            if (n_embd_clip != n_embd_llm) {
                LOG_TEE("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_embd_clip, n_embd_llm);
                llama_free(ctx);
                llama_free_model(model);
                return false;
            }
        }

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_should_add_bos_token(model);
        GGML_ASSERT(llama_add_eos_token(model) != 1);

        return true;
    }

    void validate_model_chat_template(server_params & sparams) {
        llama_chat_message chat[] = {{"user", "test"}};
        std::vector<char> buf(1);
        int res = llama_chat_apply_template(model, nullptr, chat, 1, true, buf.data(), buf.size());
        if (res < 0) {
            LOG_ERROR("The chat template comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses", {});
            sparams.chat_template = "<|im_start|>"; // llama_chat_apply_template only checks if <|im_start|> exist in the template
        }
    }

    void initialize() {
        // create slots
        all_slots_are_idle = true;

        const int32_t n_ctx_slot = n_ctx / params.n_parallel;

        LOG_INFO("initializing slots", {{"n_slots", params.n_parallel}});
        for (int i = 0; i < params.n_parallel; i++)
        {
            llama_client_slot slot;

            slot.id = i;
            slot.n_ctx = n_ctx_slot;
            slot.n_predict = params.n_predict;

            LOG_INFO("new slot", {
                {"slot_id",    slot.id},
                {"n_ctx_slot", slot.n_ctx}
            });

            const int ga_n = params.grp_attn_n;
            const int ga_w = params.grp_attn_w;

            if (ga_n != 1) {
                GGML_ASSERT(ga_n > 0                    && "ga_n must be positive");                     // NOLINT
                GGML_ASSERT(ga_w % ga_n == 0            && "ga_w must be a multiple of ga_n");     // NOLINT
                //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of ga_w");    // NOLINT
                //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * ga_n"); // NOLINT

                LOG_INFO("slot self-extend", {
                    {"slot_id",   slot.id},
                    {"ga_n",      ga_n},
                    {"ga_w",      ga_w}
                });
            }

            slot.ga_i = 0;
            slot.ga_n = ga_n;
            slot.ga_w = ga_w;

            slot.reset();

            slots.push_back(slot);
        }

        default_generation_settings_for_props = get_formated_generation(slots.front());
        default_generation_settings_for_props["seed"] = -1;

        batch = llama_batch_init(n_ctx, 0, params.n_parallel);
    }

    std::vector<llama_token> tokenize(const json & json_prompt, bool add_special) const
    {
        // TODO: currently, we tokenize using special tokens by default
        //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
        //       but it's better compared to completely ignoring ChatML and other chat templates
        const bool TMP_FORCE_SPECIAL = true;

        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        std::vector<llama_token> prompt_tokens;

        if (json_prompt.is_array())
        {
            bool first = true;
            for (const auto& p : json_prompt)
            {
                if (p.is_string())
                {
                    auto s = p.template get<std::string>();
                    std::vector<llama_token> p;
                    if (first)
                    {
                        p = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
                        first = false;
                    }
                    else
                    {
                        p = ::llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
                    }
                    prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
                }
                else
                {
                    if (first)
                    {
                        first = false;
                    }
                    prompt_tokens.push_back(p.template get<llama_token>());
                }
            }
        }
        else
        {
            auto s = json_prompt.template get<std::string>();
            prompt_tokens = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
        }

        return prompt_tokens;
    }

    llama_client_slot* get_slot(int id) {
        int64_t t_last = ggml_time_us();
        llama_client_slot *last_used = nullptr;

        for (llama_client_slot & slot : slots)
        {
            if (slot.id == id && slot.available())
            {
                return &slot;
            }

            if (slot.available() && slot.t_last_used < t_last)
            {
                last_used = &slot;
                t_last = slot.t_last_used;
            }
        }

        return last_used;
    }

    bool launch_slot_with_data(llama_client_slot* &slot, json data) {
        slot_params default_params;
        llama_sampling_params default_sparams;

        if (data.count("__oaicompat") != 0) {
            slot->oaicompat = true;
            slot->oaicompat_model = json_value(data, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
        } else {
            slot->oaicompat = false;
            slot->oaicompat_model = "";
        }

        slot->params.stream             = json_value(data, "stream",            false);
        slot->params.cache_prompt       = json_value(data, "cache_prompt",      false);
        slot->params.n_predict          = json_value(data, "n_predict",         default_params.n_predict);
        slot->sparams.top_k             = json_value(data, "top_k",             default_sparams.top_k);
        slot->sparams.top_p             = json_value(data, "top_p",             default_sparams.top_p);
        slot->sparams.min_p             = json_value(data, "min_p",             default_sparams.min_p);
        slot->sparams.tfs_z             = json_value(data, "tfs_z",             default_sparams.tfs_z);
        slot->sparams.typical_p         = json_value(data, "typical_p",         default_sparams.typical_p);
        slot->sparams.temp              = json_value(data, "temperature",       default_sparams.temp);
        slot->sparams.dynatemp_range    = json_value(data, "dynatemp_range",    default_sparams.dynatemp_range);
        slot->sparams.dynatemp_exponent = json_value(data, "dynatemp_exponent", default_sparams.dynatemp_exponent);
        slot->sparams.penalty_last_n    = json_value(data, "repeat_last_n",     default_sparams.penalty_last_n);
        slot->sparams.penalty_repeat    = json_value(data, "repeat_penalty",    default_sparams.penalty_repeat);
        slot->sparams.penalty_freq      = json_value(data, "frequency_penalty", default_sparams.penalty_freq);
        slot->sparams.penalty_present   = json_value(data, "presence_penalty",  default_sparams.penalty_present);
        slot->sparams.mirostat          = json_value(data, "mirostat",          default_sparams.mirostat);
        slot->sparams.mirostat_tau      = json_value(data, "mirostat_tau",      default_sparams.mirostat_tau);
        slot->sparams.mirostat_eta      = json_value(data, "mirostat_eta",      default_sparams.mirostat_eta);
        slot->sparams.penalize_nl       = json_value(data, "penalize_nl",       default_sparams.penalize_nl);
        slot->params.n_keep             = json_value(data, "n_keep",            slot->params.n_keep);
        slot->params.seed               = json_value(data, "seed",              default_params.seed);
        slot->sparams.grammar           = json_value(data, "grammar",           default_sparams.grammar);
        slot->sparams.n_probs           = json_value(data, "n_probs",           default_sparams.n_probs);
        slot->sparams.min_keep          = json_value(data, "min_keep",          default_sparams.min_keep);

        if (slot->n_predict > 0 && slot->params.n_predict > slot->n_predict) {
            // Might be better to reject the request with a 400 ?
            LOG_WARNING("Max tokens to predict exceeds server configuration", {
                {"params.n_predict", slot->params.n_predict},
                {"slot.n_predict", slot->n_predict},
            });
            slot->params.n_predict = slot->n_predict;
        }

        // infill
        if (data.count("input_prefix") != 0)
        {
            slot->params.input_prefix = data["input_prefix"];
        }
        else
        {
            slot->params.input_prefix = "";
        }

        if (data.count("input_suffix") != 0)
        {
            slot->params.input_suffix = data["input_suffix"];
        }
        else
        {
            slot->params.input_suffix = "";
        }

        if (data.count("prompt") != 0)
        {
            slot->prompt = data["prompt"];
        }
        else
        {
            slot->prompt = "";
        }

        slot->sparams.penalty_prompt_tokens.clear();
        slot->sparams.use_penalty_prompt_tokens = false;
        const auto &penalty_prompt = data.find("penalty_prompt");
        if (penalty_prompt != data.end())
        {
            if (penalty_prompt->is_string())
            {
                const auto penalty_prompt_string = penalty_prompt->get<std::string>();
                auto penalty_tokens = llama_tokenize(model, penalty_prompt_string, false);
                slot->sparams.penalty_prompt_tokens.swap(penalty_tokens);
                if (slot->params.n_predict > 0)
                {
                    slot->sparams.penalty_prompt_tokens.reserve(slot->sparams.penalty_prompt_tokens.size() + slot->params.n_predict);
                }
                slot->sparams.use_penalty_prompt_tokens = true;
            }
            else if (penalty_prompt->is_array())
            {
                const auto n_tokens = penalty_prompt->size();
                slot->sparams.penalty_prompt_tokens.reserve(n_tokens + std::max(0, slot->params.n_predict));
                const int n_vocab = llama_n_vocab(model);
                for (const auto &penalty_token : *penalty_prompt)
                {
                    if (penalty_token.is_number_integer())
                    {
                        const auto tok = penalty_token.get<llama_token>();
                        if (tok >= 0 && tok < n_vocab)
                        {
                            slot->sparams.penalty_prompt_tokens.push_back(tok);
                        }
                    }
                }
                slot->sparams.use_penalty_prompt_tokens = true;
            }
        }

        slot->sparams.logit_bias.clear();

        if (json_value(data, "ignore_eos", false))
        {
            slot->sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
        }

        const auto &logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && logit_bias->is_array())
        {
            const int n_vocab = llama_n_vocab(model);
            for (const auto &el : *logit_bias)
            {
                if (el.is_array() && el.size() == 2)
                {
                    float bias;
                    if (el[1].is_number())
                    {
                        bias = el[1].get<float>();
                    }
                    else if (el[1].is_boolean() && !el[1].get<bool>())
                    {
                        bias = -INFINITY;
                    }
                    else
                    {
                        continue;
                    }

                    if (el[0].is_number_integer())
                    {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab)
                        {
                            slot->sparams.logit_bias[tok] = bias;
                        }
                    }
                    else if (el[0].is_string())
                    {
                        auto toks = llama_tokenize(model, el[0].get<std::string>(), false);
                        for (auto tok : toks)
                        {
                            slot->sparams.logit_bias[tok] = bias;
                        }
                    }
                }
            }
        }

        slot->params.antiprompt.clear();

        const auto &stop = data.find("stop");
        if (stop != data.end() && stop->is_array())
        {
            for (const auto &word : *stop)
            {
                if (!word.empty())
                {
                    slot->params.antiprompt.push_back(word);
                }
            }
        }

        const auto &samplers_sequence = data.find("samplers");
        if (samplers_sequence != data.end() && samplers_sequence->is_array())
        {
            std::vector<std::string> sampler_names;
            for (const auto &sampler_name : *samplers_sequence)
            {
                if (sampler_name.is_string())
                {
                    sampler_names.emplace_back(sampler_name);
                }
            }
            slot->sparams.samplers_sequence = sampler_types_from_names(sampler_names, false);
        }
        else
        {
            slot->sparams.samplers_sequence = default_sparams.samplers_sequence;
        }

        if (multimodal)
        {
            const auto &images_data = data.find("image_data");
            if (images_data != data.end() && images_data->is_array())
            {
                for (const auto &img : *images_data)
                {
                    const std::vector<uint8_t> image_buffer = base64_decode(img["data"].get<std::string>());

                    slot_image img_sl;
                    img_sl.id = img.count("id") != 0 ? img["id"].get<int>() : slot->images.size();
                    img_sl.img_data = clip_image_u8_init();
                    if (!clip_image_load_from_bytes(image_buffer.data(), image_buffer.size(), img_sl.img_data))
                    {
                        LOG_ERROR("failed to load image", {
                            {"slot_id",   slot->id},
                            {"img_sl_id", img_sl.id}
                        });
                        return false;
                    }
                    LOG_VERBOSE("image loaded", {
                        {"slot_id",   slot->id},
                        {"img_sl_id", img_sl.id}
                    });
                    img_sl.request_encode_image = true;
                    slot->images.push_back(img_sl);
                }
                // process prompt
                // example: system prompt [img-102] user [img-103] describe [img-134] -> [{id: 102, prefix: 'system prompt '}, {id: 103, prefix: ' user '}, {id: 134, prefix: ' describe '}]}
                if (slot->images.size() > 0 && !slot->prompt.is_array())
                {
                    std::string prompt = slot->prompt.get<std::string>();
                    size_t pos = 0, begin_prefix = 0;
                    std::string pattern = "[img-";
                    while ((pos = prompt.find(pattern, pos)) != std::string::npos) {
                        size_t end_prefix = pos;
                        pos += pattern.length();
                        size_t end_pos = prompt.find(']', pos);
                        if (end_pos != std::string::npos)
                        {
                            std::string image_id = prompt.substr(pos, end_pos - pos);
                            try
                            {
                                int img_id = std::stoi(image_id);
                                bool found = false;
                                for (slot_image &img : slot->images)
                                {
                                    if (img.id == img_id) {
                                        found = true;
                                        img.prefix_prompt = prompt.substr(begin_prefix, end_prefix - begin_prefix);
                                        begin_prefix = end_pos + 1;
                                        break;
                                    }
                                }
                                if (!found) {
                                    LOG_TEE("ERROR: Image with id: %i, not found.\n", img_id);
                                    slot->images.clear();
                                    return false;
                                }
                            } catch (const std::invalid_argument& e) {
                                LOG_TEE("Invalid image number id in prompt\n");
                                slot->images.clear();
                                return false;
                            }
                        }
                    }
                    slot->prompt = "";
                    slot->params.input_suffix = prompt.substr(begin_prefix);
                    slot->params.cache_prompt = false; // multimodal doesn't support cache prompt
                }
            }
        }

        if (slot->ctx_sampling != nullptr)
        {
            llama_sampling_free(slot->ctx_sampling);
        }
        slot->ctx_sampling = llama_sampling_init(slot->sparams);
        llama_set_rng_seed(ctx, slot->params.seed);
        slot->command = LOAD_PROMPT;

        all_slots_are_idle = false;

        LOG_INFO("slot is processing task", {
            {"slot_id", slot->id},
            {"task_id", slot->task_id},
        });

        return true;
    }

    void kv_cache_clear() {
        // clear the entire KV cache
        llama_kv_cache_clear(ctx);
        clean_kv_cache = false;
    }

    void update_system_prompt() {
        kv_cache_clear();
        system_tokens.clear();

        if (!system_prompt.empty()) {
            system_tokens = ::llama_tokenize(ctx, system_prompt, add_bos_token);

            llama_batch_clear(batch);

            for (int i = 0; i < (int)system_tokens.size(); ++i)
            {
                llama_batch_add(batch, system_tokens[i], i, { 0 }, false);
            }

            long t1 = micros();

            for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += params.n_batch)
            {
                const int32_t n_tokens = std::min(params.n_batch, (int32_t) (batch.n_tokens - i));
                llama_batch batch_view = {
                    n_tokens,
                    batch.token    + i,
                    nullptr,
                    batch.pos      + i,
                    batch.n_seq_id + i,
                    batch.seq_id   + i,
                    batch.logits   + i,
                    0, 0, 0, // unused
                };
                if (llama_decode(ctx, batch_view) != 0)
                {
                    LOG_TEE("%s: llama_decode() failed\n", __func__);
                    return;
                }
            }

            long t2 = micros();
            int n_tokens = batch.n_tokens;
            g_prompt_per_second_jart = 1e6 / (t2 - t1) * n_tokens;
            LOG_TEE("evaluated %d prompt tokens in %ld us at %g tok/sec\n",
                    n_tokens, t2 - t1, g_prompt_per_second_jart);

            // assign the system KV cache to all parallel sequences
            for (int32_t i = 1; i < params.n_parallel; ++i)
            {
                llama_kv_cache_seq_cp(ctx, 0, i, 0, system_tokens.size());
            }
        }

        LOG_TEE("system prompt updated\n");
        system_need_update = false;
    }

    void notify_system_prompt_changed() {
        // release all slots
        for (llama_client_slot &slot : slots)
        {
            slot.release();
        }

        system_need_update = true;
    }

    void process_system_prompt_data(const json &sys_props) {
        system_prompt  = sys_props.value("prompt", "");
        name_user      = sys_props.value("anti_prompt", "");
        name_assistant = sys_props.value("assistant_name", "");


        notify_system_prompt_changed();
    }

    static size_t find_stopping_strings(const std::string &text, const size_t last_token_size,
                                        const stop_type type, llama_client_slot &slot)
    {
        size_t stop_pos = std::string::npos;

        for (const std::string &word : slot.params.antiprompt)
        {
            size_t pos;
            if (type == STOP_FULL)
            {
                const size_t tmp = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
                pos = text.find(word, from_pos);
            }
            else
            {
                pos = find_partial_stop_string(word, text);
            }
            if (pos != std::string::npos &&
                (stop_pos == std::string::npos || pos < stop_pos))
            {
                if (type == STOP_FULL)
                {
                    slot.stopped_word = true;
                    slot.stopping_word = word;
                    slot.has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    bool process_token(completion_token_output &result, llama_client_slot &slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = llama_token_to_piece(ctx, result.tok);
        slot.sampled = result.tok;

        // search stop word and delete it
        slot.generated_text += token_str;
        slot.has_next_token = true;

        if (slot.ctx_sampling->params.use_penalty_prompt_tokens && result.tok != -1)
        {
            // we can change penalty_prompt_tokens because it is always created from scratch each request
            slot.ctx_sampling->params.penalty_prompt_tokens.push_back(result.tok);
        }

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = false;
        for (unsigned i = 1; i < 5 && i <= slot.generated_text.size(); ++i)
        {
            unsigned char c = slot.generated_text[slot.generated_text.size() - i];
            if ((c & 0xC0) == 0x80)
            {
                // continuation byte: 10xxxxxx
                continue;
            }
            if ((c & 0xE0) == 0xC0)
            {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            }
            else if ((c & 0xF0) == 0xE0)
            {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            }
            else if ((c & 0xF8) == 0xF0)
            {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }

        if (!incomplete)
        {
            size_t pos = std::min(slot.sent_count, slot.generated_text.size());
            const std::string str_test = slot.generated_text.substr(pos);
            bool is_stop_full = false;
            size_t stop_pos = find_stopping_strings(str_test, token_str.size(), STOP_FULL, slot);
            if (stop_pos != std::string::npos)
            {
                is_stop_full = true;
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.sent_count, slot.generated_text.size());
            }
            else
            {
                is_stop_full = false;
                stop_pos = find_stopping_strings(str_test, token_str.size(), STOP_PARTIAL, slot);
            }

            // check if there is any token to predict
            if (stop_pos == std::string::npos || (!slot.has_next_token && !is_stop_full && stop_pos > 0))
            {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.sent_count += result.text_to_send.size();
                // add the token to slot queue and cache
            }
            slot.add_token_string(result);
            if (slot.params.stream)
            {
                send_partial_response(slot, result);
            }
        }

        if (incomplete)
        {
            slot.has_next_token = true;
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params))
        {
            slot.stopped_limit = true;
            slot.has_next_token = false;
        }

        if (!slot.cache_tokens.empty() && llama_token_is_eog(model, result.tok))
        {
            slot.stopped_eos = true;
            slot.has_next_token = false;
            LOG_VERBOSE("eos token found", {});
        }

        LOG_VERBOSE("next token", {
                                      {"token", result.tok},
                                      {"token_text", tokens_to_output_formatted_string(ctx, result.tok)},
                                      {"has_next_token", slot.has_next_token},
                                      {"n_remain", slot.n_remaining},
                                      {"num_tokens_predicted", slot.n_decoded},
                                      {"stopped_eos", slot.stopped_eos},
                                      {"stopped_word", slot.stopped_word},
                                      {"stopped_limit", slot.stopped_limit},
                                      {"stopping_word", slot.stopping_word},
                                  });

        return slot.has_next_token; // continue
    }

    bool process_images(llama_client_slot &slot) const
    {
        for (slot_image &img : slot.images)
        {
            if (!img.request_encode_image)
            {
                continue;
            }

            if (!llava_image_embed_make_with_clip_img(clp_ctx, params.n_threads, img.img_data, &img.image_embedding, &img.image_tokens)) {
                LOG_TEE("Error processing the given image");
                return false;
            }


            img.request_encode_image = false;
        }

        return slot.images.size() > 0;
    }

    void send_error(task_server& task, const std::string &error)
    {
        LOG_TEE("task %i - error: %s\n", task.id, error.c_str());
        task_result res;
        res.id = task.id;
        res.multitask_id = task.multitask_id;
        res.stop = false;
        res.error = true;
        res.result_json = { { "content", error } };
        queue_results.send(res);
    }

    json get_formated_generation(llama_client_slot &slot)
    {
        const auto eos_bias = slot.sparams.logit_bias.find(llama_token_eos(model));
        const bool ignore_eos = eos_bias != slot.sparams.logit_bias.end() &&
                                eos_bias->second < 0.0f && std::isinf(eos_bias->second);
        std::vector<std::string> samplers_sequence;
        for (const auto &sampler_type : slot.sparams.samplers_sequence)
        {
            samplers_sequence.emplace_back(sampler_type_to_name_string(sampler_type));
        }

        return json {
            {"n_ctx",             slot.n_ctx},
            {"n_predict",         slot.n_predict},
            {"model",             params.model_alias},
            {"seed",              slot.params.seed},
            {"temperature",       slot.sparams.temp},
            {"dynatemp_range",    slot.sparams.dynatemp_range},
            {"dynatemp_exponent", slot.sparams.dynatemp_exponent},
            {"top_k",             slot.sparams.top_k},
            {"top_p",             slot.sparams.top_p},
            {"min_p",             slot.sparams.min_p},
            {"tfs_z",             slot.sparams.tfs_z},
            {"typical_p",         slot.sparams.typical_p},
            {"repeat_last_n",     slot.sparams.penalty_last_n},
            {"repeat_penalty",    slot.sparams.penalty_repeat},
            {"presence_penalty",  slot.sparams.penalty_present},
            {"frequency_penalty", slot.sparams.penalty_freq},
            {"penalty_prompt_tokens", slot.sparams.penalty_prompt_tokens},
            {"use_penalty_prompt_tokens", slot.sparams.use_penalty_prompt_tokens},
            {"mirostat",          slot.sparams.mirostat},
            {"mirostat_tau",      slot.sparams.mirostat_tau},
            {"mirostat_eta",      slot.sparams.mirostat_eta},
            {"penalize_nl",       slot.sparams.penalize_nl},
            {"stop",              slot.params.antiprompt},
            {"n_predict",         slot.params.n_predict},
            {"n_keep",            params.n_keep},
            {"ignore_eos",        ignore_eos},
            {"stream",            slot.params.stream},
            {"logit_bias",        slot.sparams.logit_bias},
            {"n_probs",           slot.sparams.n_probs},
            {"min_keep",          slot.sparams.min_keep},
            {"grammar",           slot.sparams.grammar},
            {"samplers",          samplers_sequence}
        };
    }

    void send_partial_response(llama_client_slot &slot, completion_token_output tkn)
    {
        task_result res;
        res.id = slot.task_id;
        res.multitask_id = slot.multitask_id;
        res.error = false;
        res.stop = false;

        res.result_json = json
        {
            {"content",    tkn.text_to_send},
            {"stop",       false},
            {"slot_id",    slot.id},
            {"multimodal", multimodal}
        };

        if (slot.sparams.n_probs > 0)
        {
            std::vector<completion_token_output> probs_output = {};
            const std::vector<llama_token> to_send_toks = llama_tokenize(ctx, tkn.text_to_send, false);
            size_t probs_pos      = std::min(slot.sent_token_probs_index,                       slot.generated_token_probs.size());
            size_t probs_stop_pos = std::min(slot.sent_token_probs_index + to_send_toks.size(), slot.generated_token_probs.size());
            if (probs_pos < probs_stop_pos)
            {
                probs_output = std::vector<completion_token_output>(slot.generated_token_probs.begin() + probs_pos, slot.generated_token_probs.begin() + probs_stop_pos);
            }
            slot.sent_token_probs_index = probs_stop_pos;
            res.result_json["completion_probabilities"] = probs_vector_to_json(ctx, probs_output);
        }

        if (slot.oaicompat)
        {
            res.result_json["oaicompat_token_ctr"] = slot.n_decoded;
            res.result_json["model"] = slot.oaicompat_model;
        }

        queue_results.send(res);
    }

    void send_final_response(llama_client_slot &slot)
    {
        task_result res;
        res.id = slot.task_id;
        res.multitask_id = slot.multitask_id;
        res.error = false;
        res.stop = true;

        res.result_json = json
        {
            {"content",             !slot.params.stream ? slot.generated_text : ""},
            {"slot_id",             slot.id},
            {"stop",                true},
            {"model",               params.model_alias},
            {"tokens_predicted",    slot.n_decoded},
            {"tokens_evaluated",    slot.num_prompt_tokens},
            {"generation_settings", get_formated_generation(slot)},
            {"prompt",              slot.prompt},
            {"truncated",           slot.truncated},
            {"stopped_eos",         slot.stopped_eos},
            {"stopped_word",        slot.stopped_word},
            {"stopped_limit",       slot.stopped_limit},
            {"stopping_word",       slot.stopping_word},
            {"tokens_cached",       slot.n_past},
            {"timings",             slot.get_formated_timings()}
        };

        if (slot.sparams.n_probs > 0)
        {
            std::vector<completion_token_output> probs = {};
            if (!slot.params.stream && slot.stopped_word)
            {
                const std::vector<llama_token> stop_word_toks = llama_tokenize(ctx, slot.stopping_word, false);
                probs = std::vector<completion_token_output>(slot.generated_token_probs.begin(), slot.generated_token_probs.end() - stop_word_toks.size());
            }
            else
            {
                probs = std::vector<completion_token_output>(
                                    slot.generated_token_probs.begin(),
                                    slot.generated_token_probs.end());
            }
            res.result_json["completion_probabilities"] = probs_vector_to_json(ctx, probs);
        }

        if (slot.oaicompat)
        {
            res.result_json["oaicompat_token_ctr"] = slot.n_decoded;
            res.result_json["model"] = slot.oaicompat_model;
        }

        queue_results.send(res);
    }

    void send_embedding(llama_client_slot &slot)
    {
        task_result res;
        res.id = slot.task_id;
        res.multitask_id = slot.multitask_id;
        res.error = false;
        res.stop = true;

        const int n_embd = llama_n_embd(model);
        if (!params.embedding)
        {
            LOG_WARNING("embedding disabled", {
                                                  {"params.embedding", params.embedding},
                                              });
            res.result_json = json
            {
                {"embedding", std::vector<float>(n_embd, 0.0f)},
            };
        }
        else
        {
            std::vector<float> embd_res(n_embd, 0.0f);

            for (int i = 0; i < batch.n_tokens; i++) {
                const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
                if (embd == NULL) {
                    embd = llama_get_embeddings_ith(ctx, i);
                }
                if (embd == NULL) {
                    LOG_ERROR("failed to get embeddings", {
                        {"token",  batch.token [i]},
                        {"seq_id", batch.seq_id[i][0]}
                    });
                    res.result_json = json {
                            {"embedding", std::vector<float>(n_embd, 0.0f)},
                    };
                    continue;
                }
                llama_embd_normalize(embd, embd_res.data(), n_embd);
                res.result_json = json {
                        {"embedding", embd_res},
                };
            }
        }
        queue_results.send(res);
    }

    void request_completion(int task_id, json data, bool infill, bool embedding, int multitask_id)
    {
        task_server task;
        task.id = task_id;
        task.target_id = 0;
        task.data = std::move(data);
        task.infill_mode = infill;
        task.embedding_mode = embedding;
        task.type = TASK_TYPE_COMPLETION;
        task.multitask_id = multitask_id;

        // when a completion task's prompt array is not a singleton, we split it into multiple requests
        // otherwise, it's a single-prompt task, we actually queue it
        // if there's numbers in the prompt array it will be treated as an array of tokens
        if (task.data.count("prompt") != 0 && task.data.at("prompt").size() > 1) {
            bool numbers = false;
            for (const auto& e : task.data.at("prompt")) {
                if (e.is_number()) {
                    numbers = true;
                    break;
                }
            }

            // NOTE: split_multiprompt_task() does not handle a mix of strings and numbers,
            // it will completely stall the server. I don't know where the bug for this is.
            //
            // if there are numbers, it needs to be treated like a single prompt,
            // queue_tasks handles a mix of strings and numbers just fine.
            if (numbers) {
                queue_tasks.post(task);
            } else {
                split_multiprompt_task(task_id, task);
            }
        } else {
            // an empty prompt can make slot become buggy
            if (task.data.contains("prompt") && task.data["prompt"].is_string() && task.data["prompt"].get<std::string>().empty()) {
                task.data["prompt"] = " "; // add a space so that we have one token
            }
            queue_tasks.post(task);
        }
    }

    // for multiple images processing
    bool ingest_images(llama_client_slot &slot, int n_batch)
    {
        int image_idx = 0;

        while (image_idx < (int) slot.images.size())
        {
            slot_image &img = slot.images[image_idx];

            long t1 = micros();

            // process prefix prompt
            for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch)
            {
                const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));
                llama_batch batch_view = {
                    n_tokens,
                    batch.token    + i,
                    nullptr,
                    batch.pos      + i,
                    batch.n_seq_id + i,
                    batch.seq_id   + i,
                    batch.logits   + i,
                    0, 0, 0, // unused
                };
                if (llama_decode(ctx, batch_view))
                {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return false;
                }
            }

            // process image with llm
            for (int i = 0; i < img.image_tokens; i += n_batch)
            {
                int n_eval = img.image_tokens - i;
                if (n_eval > n_batch)
                {
                    n_eval = n_batch;
                }

                const int n_embd = llama_n_embd(model);
                llama_batch batch_img = { n_eval, nullptr, (img.image_embedding + i * n_embd), nullptr, nullptr, nullptr, nullptr, slot.n_past, 1, 0, };
                if (llama_decode(ctx, batch_img))
                {
                    LOG_TEE("%s : failed to eval image\n", __func__);
                    return false;
                }
                slot.n_past += n_eval;
            }
            image_idx++;

            long t2 = micros();
            int n_tokens = batch.n_tokens + img.image_tokens;
            g_prompt_per_second_jart = 1e6 / (t2 - t1) * n_tokens;
            LOG_TEE("evaluated %d image tokens in %ld us at %g tok/sec\n",
                    n_tokens, t2 - t1, g_prompt_per_second_jart);

            llama_batch_clear(batch);

            // append prefix of next image
            const auto json_prompt = (image_idx >= (int) slot.images.size()) ?
                slot.params.input_suffix : // no more images, then process suffix prompt
                (json)(slot.images[image_idx].prefix_prompt);

            std::vector<llama_token> append_tokens = tokenize(json_prompt, false); // has next image
            for (int i = 0; i < (int) append_tokens.size(); ++i)
            {
                llama_batch_add(batch, append_tokens[i], system_tokens.size() + slot.n_past, { slot.id }, true);
                slot.n_past += 1;
            }
        }

        return true;
    }

    void request_cancel(int task_id)
    {
        task_server task;
        task.type = TASK_TYPE_CANCEL;
        task.target_id = task_id;
        queue_tasks.post(task);
    }

    void split_multiprompt_task(int multitask_id, task_server& multiprompt_task)
    {
        int prompt_count = multiprompt_task.data.at("prompt").size();
        if (prompt_count <= 1) {
            send_error(multiprompt_task, "error while handling multiple prompts");
            return;
        }

        // generate all the ID for subtask
        std::vector<int> subtask_ids(prompt_count);
        for (int i = 0; i < prompt_count; i++)
        {
            subtask_ids[i] = queue_tasks.get_new_id();
        }

        // queue up the multitask so we can track its subtask progression
        queue_tasks.add_multitask(multitask_id, subtask_ids);

        // add subtasks
        for (int i = 0; i < prompt_count; i++)
        {
            json subtask_data = multiprompt_task.data;
            subtask_data["prompt"] = subtask_data["prompt"][i];

            // subtasks inherit everything else (infill mode, embedding mode, etc.)
            request_completion(subtask_ids[i], subtask_data, multiprompt_task.infill_mode, multiprompt_task.embedding_mode, multitask_id);
        }
    }

    void process_single_task(task_server& task)
    {
        switch (task.type)
        {
            case TASK_TYPE_COMPLETION: {
                llama_client_slot *slot = get_slot(json_value(task.data, "slot_id", -1));
                if (slot == nullptr)
                {
                    // if no slot is available, we defer this task for processing later
                    LOG_VERBOSE("no slot is available", {{"task_id", task.id}});
                    queue_tasks.defer(task);
                    break;
                }

                if (task.data.contains("system_prompt"))
                {
                    if (!all_slots_are_idle) {
                        send_error(task, "system prompt can only be updated when all slots are idle");
                        break;
                    }
                    process_system_prompt_data(task.data["system_prompt"]);

                    // reset cache_tokens for all slots
                    for (llama_client_slot &slot : slots)
                    {
                        slot.cache_tokens.clear();
                        slot.n_past    = 0;
                        slot.n_past_se = 0;
                    }
                }

                slot->reset();

                slot->infill       = task.infill_mode;
                slot->embedding    = task.embedding_mode;
                slot->task_id      = task.id;
                slot->multitask_id = task.multitask_id;

                if (!launch_slot_with_data(slot, task.data))
                {
                    // send error result
                    send_error(task, "internal_error");
                    break;
                }
            } break;
            case TASK_TYPE_CANCEL: { // release slot linked with the task id
                for (auto & slot : slots)
                {
                    if (slot.task_id == task.target_id)
                    {
                        slot.release();
                        break;
                    }
                }
            } break;
            case TASK_TYPE_NEXT_RESPONSE: {
                // do nothing
            } break;
            case TASK_TYPE_METRICS: {
                json slots_data        = json::array();
                int n_idle_slots       = 0;
                int n_processing_slots = 0;

                for (llama_client_slot &slot: slots) {
                    json slot_data = get_formated_generation(slot);
                    slot_data["id"] = slot.id;
                    slot_data["task_id"] = slot.task_id;
                    slot_data["state"] = slot.state;
                    slot_data["prompt"] = slot.prompt;
                    slot_data["next_token"] = {
                            {"has_next_token", slot.has_next_token},
                            {"n_remain", slot.n_remaining},
                            {"num_tokens_predicted", slot.n_decoded},
                            {"stopped_eos", slot.stopped_eos},
                            {"stopped_word", slot.stopped_word},
                            {"stopped_limit", slot.stopped_limit},
                            {"stopping_word", slot.stopping_word},
                    };
                    if (slot_data["state"] == IDLE) {
                        n_idle_slots++;
                    } else {
                        n_processing_slots++;
                    }
                    slots_data.push_back(slot_data);
                }
                LOG_INFO("slot data", {
                    {"task_id",            task.id},
                    {"n_idle_slots",       n_idle_slots},
                    {"n_processing_slots", n_processing_slots}
                });
                LOG_VERBOSE("slot data", {
                    {"task_id",            task.id},
                    {"n_idle_slots",       n_idle_slots},
                    {"n_processing_slots", n_processing_slots},
                    {"slots",              slots_data}
                });
                task_result res;
                res.id = task.id;
                res.multitask_id = task.multitask_id;
                res.stop = true;
                res.error = false;
                res.result_json = {
                        { "idle",                            n_idle_slots       },
                        { "processing",                      n_processing_slots },
                        { "deferred",                        queue_tasks.queue_tasks_deferred.size() },

                        { "n_prompt_tokens_processed_total", metrics.n_prompt_tokens_processed_total},
                        { "n_tokens_predicted_total",        metrics.n_tokens_predicted_total},

                        { "n_prompt_tokens_processed",       metrics.n_prompt_tokens_processed},
                        { "t_prompt_processing",             metrics.t_prompt_processing},
                        { "n_tokens_predicted",              metrics.n_tokens_predicted},
                        { "t_tokens_generation",             metrics.t_tokens_generation},

                        { "kv_cache_tokens_count",          llama_get_kv_cache_token_count(ctx)},
                        { "kv_cache_used_cells",            llama_get_kv_cache_used_cells(ctx)},

                        { "slots",                          slots_data },
                };
                metrics.reset_bucket();
                queue_results.send(res);
            } break;
        }
    }

    void on_finish_multitask(task_multi& multitask)
    {
        // all subtasks done == multitask is done
        task_result result;
        result.id = multitask.id;
        result.stop = true;
        result.error = false;

        // collect json results into one json result
        std::vector<json> result_jsons;
        for (auto& subres : multitask.results)
        {
            result_jsons.push_back(subres.result_json);
            result.error = result.error && subres.error;
        }
        result.result_json = json{ { "results", result_jsons } };
        queue_results.send(result);
    }

    bool update_slots() {
        if (system_need_update)
        {
            LOG_INFO("updating system prompt", {});
            update_system_prompt();
        }

        llama_batch_clear(batch);

        if (all_slots_are_idle)
        {
            if (system_prompt.empty() && clean_kv_cache)
            {
                LOG_INFO("all slots are idle and system prompt is empty, clear the KV cache", {});
                kv_cache_clear();
            }
            return true;
        }

        LOG_VERBOSE("posting NEXT_RESPONSE", {});
        task_server task;
        task.type = TASK_TYPE_NEXT_RESPONSE;
        task.target_id = -1;
        queue_tasks.post(task);

        for (llama_client_slot &slot : slots)
        {
            if (slot.ga_n == 1)
            {
                if (slot.is_processing() && system_tokens.size() + slot.cache_tokens.size() >= (size_t) slot.n_ctx)
                {
                    // Shift context
                    const int n_keep    = slot.params.n_keep + add_bos_token;
                    const int n_left    = (int) system_tokens.size() + slot.n_past - n_keep;
                    const int n_discard = n_left / 2;

                    LOG_INFO("slot context shift", {
                        {"slot_id",         slot.id},
                        {"task_id",         slot.task_id},
                        {"n_keep",          n_keep},
                        {"n_left",          n_left},
                        {"n_discard",       n_discard},
                        {"n_ctx",           n_ctx},
                        {"n_past",          slot.n_past},
                        {"n_system_tokens", system_tokens.size()},
                        {"n_cache_tokens",  slot.cache_tokens.size()}
                    });
                    llama_kv_cache_seq_rm (ctx, slot.id, n_keep            , n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, slot.id, n_keep + n_discard, system_tokens.size() + slot.n_past, -n_discard);

                    for (size_t i = n_keep + n_discard; i < slot.cache_tokens.size(); i++)
                    {
                        slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
                    }

                    slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);

                    slot.n_past -= n_discard;

                    slot.truncated = true;
                }
            }
        }

        // decode any currently ongoing sequences
        LOG_VERBOSE("decoding ongoing sequences", {});
        for (auto & slot : slots)
        {
            // release the slot
            if (slot.command == RELEASE)
            {
                slot.state = IDLE;
                slot.command = NONE;
                slot.t_last_used = ggml_time_us();

                LOG_INFO("slot released", {
                    {"slot_id",         slot.id},
                    {"task_id",         slot.task_id},
                    {"n_ctx",           n_ctx},
                    {"n_past",          slot.n_past},
                    {"n_system_tokens", system_tokens.size()},
                    {"n_cache_tokens",  slot.cache_tokens.size()},
                    {"truncated",       slot.truncated}
                });
                queue_tasks.notify_slot_changed();

                continue;
            }

            if (slot.state == IDLE)
            {
                continue;
            }

            slot.i_batch = batch.n_tokens;

            const int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

            // TODO: we always have to take into account the "system_tokens"
            //       this is not great and needs to be improved somehow
            llama_batch_add(batch, slot.sampled, system_tokens.size() + slot_npast, { slot.id }, true);
            slot.n_past += 1;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        // assign workload to the slots
        if (params.cont_batching || batch.n_tokens == 0)
        {
            for (auto & slot : slots)
            {
                const bool has_prompt = slot.prompt.is_array() || (slot.prompt.is_string() && !slot.prompt.get<std::string>().empty()) || !slot.images.empty();

                // empty prompt passed -> release the slot and send empty response
                // note: infill mode allows empty prompt
                if (slot.state == IDLE && slot.command == LOAD_PROMPT && !has_prompt && !slot.infill)
                {
                    slot.release();
                    slot.print_timings();
                    send_final_response(slot);
                    continue;
                }

                // need process the prompt
                if (slot.state == IDLE && slot.command == LOAD_PROMPT)
                {
                    slot.state = PROCESSING;
                    slot.command = NONE;
                    std::vector<llama_token> prompt_tokens;
                    slot.t_start_process_prompt = ggml_time_us();
                    slot.t_start_genereration = 0;

                    if (slot.infill)
                    {
                        bool suff_rm_leading_spc = true;
                        if (params.input_suffix.find_first_of(' ') == 0 && params.input_suffix.size() > 1)
                        {
                            params.input_suffix.erase(0, 1);
                            suff_rm_leading_spc = false;
                        }
                        auto prefix_tokens = tokenize(slot.params.input_prefix, false);
                        auto suffix_tokens = tokenize(slot.params.input_suffix, false);

                        const int space_token = 29871; // TODO: this should not be hardcoded
                        if (suff_rm_leading_spc && !suffix_tokens.empty() && suffix_tokens[0] == space_token) {
                            suffix_tokens.erase(suffix_tokens.begin());
                        }

                        prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(model));
                        prefix_tokens.insert(prefix_tokens.begin(), llama_token_bos(model)); // always add BOS
                        prefix_tokens.insert(prefix_tokens.end(),   llama_token_suffix(model));
                        prefix_tokens.insert(prefix_tokens.end(),   suffix_tokens.begin(), suffix_tokens.end());
                        prefix_tokens.push_back(llama_token_middle(model));
                        prompt_tokens = prefix_tokens;
                    }
                    else
                    {
                        prompt_tokens = tokenize(slot.prompt, system_prompt.empty());  // add BOS if there isn't system prompt
                    }

                    slot.num_prompt_tokens = prompt_tokens.size();

                    if (slot.params.n_keep < 0)
                    {
                        slot.params.n_keep = slot.num_prompt_tokens;
                    }
                    slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                    // if input prompt is too big, truncate it
                    if (slot.num_prompt_tokens >= slot.n_ctx)
                    {
                        const int n_left = slot.n_ctx - slot.params.n_keep;
                        const int n_block_size = n_left / 2;
                        const int erased_blocks = (slot.num_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                        std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + slot.params.n_keep);
                        new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size, prompt_tokens.end());

                        LOG_VERBOSE("input truncated", {
                            {"n_ctx",  slot.n_ctx},
                            {"n_keep", slot.params.n_keep},
                            {"n_left", n_left},
                            {"new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend())},
                        });
                        slot.truncated = true;
                        prompt_tokens = new_tokens;

                        slot.num_prompt_tokens = prompt_tokens.size();
                        GGML_ASSERT(slot.num_prompt_tokens < slot.n_ctx);
                    }

                    if (!slot.params.cache_prompt)
                    {
                        llama_sampling_reset(slot.ctx_sampling);

                        slot.n_past = 0;
                        slot.n_past_se = 0;
                        slot.ga_i = 0;
                        slot.num_prompt_tokens_processed = slot.num_prompt_tokens;
                    }
                    else
                    {
                        // push the prompt into the sampling context (do not apply grammar)
                        for (auto &token : prompt_tokens)
                        {
                            llama_sampling_accept(slot.ctx_sampling, ctx, token, false);
                        }

                        slot.n_past = common_part(slot.cache_tokens, prompt_tokens);

                        // the last token of the cache is not in the KV cache until the next call to llama_decode
                        // (it was sampled, pushed into the "cache_tokens", but not yet put in the context)
                        if (slot.n_past > 0 && slot.n_past == (int32_t) slot.cache_tokens.size())
                        {
                            slot.n_past -= 1;
                        }

                        slot.num_prompt_tokens_processed = slot.num_prompt_tokens - slot.n_past;

                        if (slot.ga_n != 1)
                        {
                            int ga_i = 0;
                            int32_t ga_n = slot.ga_n;
                            int32_t ga_w = slot.ga_w;
                            int32_t slot_npast = 0;
                            for (int k = 0; k < slot.n_past; ++k)
                            {
                                while (slot_npast >= ga_i + ga_w) {
                                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                                    slot_npast -= bd;
                                    ga_i += ga_w/ga_n;
                                }
                                slot_npast++;
                            }
                            slot.n_past_se = slot_npast;
                            slot.ga_i = ga_i;
                        }

                        LOG_INFO("slot progression", {
                            { "slot_id", slot.id },
                            { "task_id", slot.task_id },
                            { "n_past",  slot.n_past },
                            { "num_prompt_tokens_processed", slot.num_prompt_tokens_processed }
                        });
                    }

                    slot.cache_tokens = prompt_tokens;

                    if (slot.n_past == slot.num_prompt_tokens && slot.n_past > 0)
                    {
                        // we have to evaluate at least 1 token to generate logits.
                        LOG_INFO("we have to evaluate at least 1 token to generate logits", {
                            { "slot_id", slot.id },
                            { "task_id", slot.task_id }
                        });
                        slot.n_past--;
                        if (slot.ga_i > 0)
                        {
                            slot.n_past_se--;
                        }
                    }

                    int p0 = (int) system_tokens.size() + slot.n_past;
                    LOG_INFO("kv cache rm [p0, end)", {
                        { "slot_id", slot.id },
                        { "task_id", slot.task_id },
                        { "p0",      p0 }
                    });
                    llama_kv_cache_seq_rm(ctx, slot.id, p0, -1);

                    LOG_VERBOSE("prompt ingested", {
                                                    {"n_past",  slot.n_past},
                                                    {"cached",  tokens_to_str(ctx, slot.cache_tokens.cbegin(), slot.cache_tokens.cbegin() + slot.n_past)},
                                                    {"to_eval", tokens_to_str(ctx, slot.cache_tokens.cbegin() + slot.n_past, slot.cache_tokens.cend())},
                                                });

                    const bool has_images = process_images(slot);

                    // process the prefix of first image
                    std::vector<llama_token> prefix_tokens = has_images ? tokenize(slot.images[0].prefix_prompt, add_bos_token) : prompt_tokens;

                    int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

                    int32_t ga_i = slot.ga_i;
                    int32_t ga_n = slot.ga_n;
                    int32_t ga_w = slot.ga_w;

                    for (; slot.n_past < (int) prefix_tokens.size(); ++slot.n_past)
                    {
                        if (slot.ga_n != 1)
                        {
                            while (slot_npast >= ga_i + ga_w) {
                                const int bd = (ga_w/ga_n)*(ga_n - 1);
                                slot_npast -= bd;
                                ga_i += ga_w/ga_n;
                            }
                        }
                        llama_batch_add(batch, prefix_tokens[slot.n_past], system_tokens.size() + slot_npast, {slot.id }, false);
                        slot_npast++;
                    }

                    if (has_images && !ingest_images(slot, n_batch))
                    {
                        LOG_ERROR("failed processing images", {
                            "slot_id", slot.id,
                            "task_id", slot.task_id,
                        });
                        // FIXME @phymbert: to be properly tested
                        //  early returning without changing the slot state will block the slot for ever
                        // no one at the moment is checking the return value
                        return false;
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0)
                    {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    slot.n_decoded = 0;
                    slot.i_batch   = batch.n_tokens - 1;
                }
            }
        }

        if (batch.n_tokens == 0)
        {
            all_slots_are_idle = true;
            return true;
        }

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch)
        {
            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            for (auto & slot : slots)
            {
                if (slot.ga_n != 1)
                {
                    // context extension via Self-Extend
                    while (slot.n_past_se >= slot.ga_i + slot.ga_w)
                    {
                        const int ib = (slot.ga_n * slot.ga_i) / slot.ga_w;
                        const int bd = (slot.ga_w / slot.ga_n) * (slot.ga_n - 1);
                        const int dd = (slot.ga_w / slot.ga_n) - ib * bd - slot.ga_w;

                        LOG_TEE("\n");
                        LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i, slot.n_past_se, ib * bd, slot.ga_i + ib * bd, slot.n_past_se + ib * bd);
                        LOG_TEE("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n, (slot.ga_i + ib * bd) / slot.ga_n, (slot.ga_i + ib * bd + slot.ga_w) / slot.ga_n);
                        LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd, slot.ga_i + ib * bd + slot.ga_w + dd, slot.n_past_se + ib * bd + dd);

                        llama_kv_cache_seq_add(ctx, slot.id, slot.ga_i, slot.n_past_se, ib * bd);
                        llama_kv_cache_seq_div(ctx, slot.id, slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w,slot.ga_n);
                        llama_kv_cache_seq_add(ctx, slot.id, slot.ga_i + ib * bd + slot.ga_w,slot.n_past_se + ib * bd, dd);

                        slot.n_past_se -= bd;

                        slot.ga_i += slot.ga_w / slot.ga_n;

                        LOG_TEE("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", slot.n_past_se + bd, slot.n_past_se, slot.ga_i);
                    }
                    slot.n_past_se += n_tokens;
                }
            }

            llama_batch batch_view =
            {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);

            if (ret != 0)
            {
                if (n_batch == 1 || ret < 0)
                {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_TEE("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return false;
                }

                LOG_TEE("%s : failed to find free space in the KV cache, retrying with smaller n_batch = %d\n", __func__, n_batch / 2);

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;
                continue;
            }

            for (auto & slot : slots)
            {
                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens))
                {
                    continue;
                }

                // prompt evaluated for embedding
                if (slot.embedding)
                {
                    send_embedding(slot);
                    slot.release();
                    slot.i_batch = -1;
                    continue;
                }

                completion_token_output result;
                const llama_token id = llama_sampling_sample(slot.ctx_sampling, ctx, NULL, slot.i_batch - i);

                llama_sampling_accept(slot.ctx_sampling, ctx, id, true);

                slot.n_decoded += 1;
                if (slot.n_decoded == 1)
                {
                    slot.t_start_genereration = ggml_time_us();
                    slot.t_prompt_processing = (slot.t_start_genereration - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                llama_token_data_array cur_p = { slot.ctx_sampling->cur.data(), slot.ctx_sampling->cur.size(), false };
                result.tok = id;

                const int32_t n_probs = slot.sparams.n_probs;
                if (slot.sparams.temp <= 0 && n_probs > 0)
                {
                    // for llama_sample_token_greedy we need to sort candidates
                    llama_sample_softmax(ctx, &cur_p);
                }

                for (size_t i = 0; i < std::min(cur_p.size, (size_t)n_probs); ++i)
                {
                    result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
                }

                if (!process_token(result, slot))
                {
                    slot.release();
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                }

                slot.i_batch = -1;
            }
        }

        LOG_VERBOSE("slots updated", {});
        return true;
    }

    void run_on_all_tasks_finished() {
        update_slots();
    }
};

static void server_print_usage(const char *argv0, const gpt_params &params,
                               const server_params &sparams)
{
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help                show this help message and exit\n");
    printf("  -v, --verbose             verbose output (default: %s)\n", server_verbose ? "enabled" : "disabled");
    printf("  -t N, --threads N         number of threads to use during computation (default: %d)\n", params.n_threads);
    printf("  -tb N, --threads-batch N  number of threads to use during batch and prompt processing (default: same as --threads)\n");
    printf("  -c N, --ctx-size N        size of the prompt context (default: %d)\n", params.n_ctx);
    printf("  --rope-scaling {none,linear,yarn}\n");
    printf("                            RoPE frequency scaling method, defaults to linear unless specified by the model\n");
    printf("  --rope-freq-base N        RoPE base frequency (default: loaded from model)\n");
    printf("  --rope-freq-scale N       RoPE frequency scaling factor, expands context by a factor of 1/N\n");
    printf("  --yarn-ext-factor N       YaRN: extrapolation mix factor (default: 1.0, 0.0 = full interpolation)\n");
    printf("  --yarn-attn-factor N      YaRN: scale sqrt(t) or attention magnitude (default: 1.0)\n");
    printf("  --yarn-beta-slow N        YaRN: high correction dim or alpha (default: %.1f)\n", params.yarn_beta_slow);
    printf("  --yarn-beta-fast N        YaRN: low correction dim or beta (default: %.1f)\n", params.yarn_beta_fast);
    printf("  -b N, --batch-size N      batch size for prompt processing (default: %d)\n", params.n_batch);
    printf("  --memory-f32              use f32 instead of f16 for memory key+value (default: disabled)\n");
    printf("                            not recommended: doubles context memory required and no measurable increase in quality\n");
    if (llama_supports_mlock())
    {
        printf("  --mlock                   force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_supports_mmap())
    {
        printf("  --no-mmap                 do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
    printf("  --numa TYPE               attempt optimizations that help on some NUMA systems\n");
    printf("                              - distribute: spread execution evenly over all nodes\n");
    printf("                              - isolate: only spawn threads on CPUs on the node that execution started on\n");
    printf("                              - numactl: use the CPU map provided my numactl\n");
    if (llama_supports_gpu_offload()) {
        printf("  -ngl N, --n-gpu-layers N\n");
        printf("                            number of layers to store in VRAM\n");
        printf("  -sm SPLIT_MODE, --split-mode SPLIT_MODE\n");
        printf("                            how to split the model across multiple GPUs, one of:\n");
        printf("                              - none: use one GPU only\n");
        printf("                              - layer (default): split layers and KV across GPUs\n");
        printf("                              - row: split rows across GPUs\n");
        printf("  -ts SPLIT --tensor-split SPLIT\n");
        printf("                            fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1\n");
        printf("  -mg i, --main-gpu i       the GPU to use for the model (with split-mode = none),\n");
        printf("                            or for intermediate results and KV (with split-mode = row)\n");
    }
    printf("  -m FNAME, --model FNAME\n");
    printf("                            model path (default: %s)\n", params.model.c_str());
    printf("  -a ALIAS, --alias ALIAS\n");
    printf("                            set an alias for the model, will be added as `model` field in completion response\n");
    printf("  --lora FNAME              apply LoRA adapter (implies --no-mmap)\n");
    printf("  --lora-base FNAME         optional model to use as a base for the layers modified by the LoRA adapter\n");
    printf("  --host                    ip address to listen (default  (default: %s)\n", sparams.hostname.c_str());
    printf("  --port PORT               port to listen (default  (default: %d)\n", sparams.port);
    printf("  --path PUBLIC_PATH        path from which to serve static files (default %s)\n", sparams.public_path.c_str());
    printf("  --api-key API_KEY         optional api key to enhance server security. If set, requests must include this key for access.\n");
    printf("  --api-key-file FNAME      path to file containing api keys delimited by new lines. If set, requests must include one of the keys for access.\n");
    printf("  -to N, --timeout N        server read/write timeout in seconds (default: %d)\n", sparams.read_timeout);
    printf("  --embedding               enable embedding vector output (default: %s)\n", params.embedding ? "enabled" : "disabled");
    printf("  -np N, --parallel N       number of slots for process requests (default: %d)\n", params.n_parallel);
    printf("  -cb, --cont-batching      enable continuous batching (a.k.a dynamic batching) (default: disabled)\n");
    printf("  -spf FNAME, --system-prompt-file FNAME\n");
    printf("                            set a file to load a system prompt (initial prompt of all slots), this is useful for chat applications.\n");
    printf("  -ctk TYPE, --cache-type-k TYPE\n");
    printf("                            KV cache data type for K (default: f16)\n");
    printf("  -ctv TYPE, --cache-type-v TYPE\n");
    printf("                            KV cache data type for V (default: f16)\n");
    printf("  --mmproj MMPROJ_FILE      path to a multimodal projector file for LLaVA.\n");
    printf("  --log-format              log output format: json or text (default: json)\n");
    printf("  --log-disable             disables logging to a file.\n");
    printf("  --slots-endpoint-disable  disables slots monitoring endpoint.\n");
    printf("  --metrics                 enable prometheus compatible metrics endpoint (default: %s).\n", sparams.metrics_endpoint ? "enabled" : "disabled");
    printf("\n");
    printf("  -n, --n-predict           maximum tokens to predict (default: %d)\n", params.n_predict);
    printf("  --override-kv KEY=TYPE:VALUE\n");
    printf("                            advanced option to override model metadata by key. may be specified multiple times.\n");
    printf("                            types: int, float, bool. example: --override-kv tokenizer.ggml.add_bos_token=bool:false\n");
    printf("  -gan N, --grp-attn-n N    set the group attention factor to extend context size through self-extend(default: 1=disabled), used together with group attention width `--grp-attn-w`");
    printf("  -gaw N, --grp-attn-w N    set the group attention width to extend context size through self-extend(default: 512), used together with group attention factor `--grp-attn-n`");
    printf("  --chat-template JINJA_TEMPLATE\n");
    printf("                            set custom jinja chat template (default: template taken from model's metadata)\n");
    printf("                            Note: only commonly used templates are accepted, since we don't have jinja parser\n");
    printf("\n");
}

static void server_params_parse(int argc, char **argv, server_params &sparams,
                                gpt_params &params, llama_server_context& llama)
{
    gpt_params default_params;
    server_params default_sparams;
    std::string arg;
    bool invalid_param = false;

    for (int i = 1; i < argc; i++)
    {
        arg = argv[i];
        if (arg == "--port")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.port = std::stoi(argv[i]);
        }
        else if (arg == "--host")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.hostname = argv[i];
        }
        else if (arg == "--path")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.public_path = argv[i];
        }
        else if (arg == "--api-key")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.api_keys.emplace_back(argv[i]);
        }
        else if (arg == "--api-key-file")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            std::ifstream key_file(argv[i]);
            if (!key_file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                invalid_param = true;
                break;
            }
            std::string key;
            while (std::getline(key_file, key)) {
               if (key.size() > 0) {
                   sparams.api_keys.push_back(key);
               }
            }
            key_file.close();
        }
        else if (arg == "--timeout" || arg == "-to")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.read_timeout = std::stoi(argv[i]);
            sparams.write_timeout = std::stoi(argv[i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        }
        else if (arg == "-a" || arg == "--alias")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.model_alias = argv[i];
        }
        else if (arg == "-h" || arg == "--help")
        {
            server_print_usage(argv[0], default_params, default_sparams);
            exit(0);
        }
        else if (arg == "-c" || arg == "--ctx-size" || arg == "--ctx_size")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_ctx = std::stoi(argv[i]);
        }
        else if (arg == "--rope-scaling")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            std::string value(argv[i]);
            /**/ if (value == "none")   { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
            else if (value == "yarn")   { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
            else { invalid_param = true; break; }
        }
        else if (arg == "--rope-freq-base")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.rope_freq_base = std::stof(argv[i]);
        }
        else if (arg == "--rope-freq-scale")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.rope_freq_scale = std::stof(argv[i]);
        }
        else if (arg == "--yarn-ext-factor")
        {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.yarn_ext_factor = std::stof(argv[i]);
        }
        else if (arg == "--yarn-attn-factor")
        {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.yarn_attn_factor = std::stof(argv[i]);
        }
        else if (arg == "--yarn-beta-fast")
        {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.yarn_beta_fast = std::stof(argv[i]);
        }
        else if (arg == "--yarn-beta-slow")
        {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.yarn_beta_slow = std::stof(argv[i]);
        }
        else if (arg == "--threads" || arg == "-t")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        }
        else if (arg == "--grp-attn-n" || arg == "-gan")
        {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }

            params.grp_attn_n = std::stoi(argv[i]);
        }
        else if (arg == "--grp-attn-w" || arg == "-gaw")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }

            params.grp_attn_w = std::stoi(argv[i]);
        }
        else if (arg == "--threads-batch" || arg == "-tb")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_threads_batch = std::stoi(argv[i]);
        }
        else if (arg == "-b" || arg == "--batch-size")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(argv[i]);
            params.n_batch = std::min(512, params.n_batch);
        }
        else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            if (llama_supports_gpu_offload()) {
                params.n_gpu_layers = std::stoi(argv[i]);
            } else {
                LOG_WARNING("Not compiled with GPU offload support, --n-gpu-layers option will be ignored. "
                        "See main README.md for information on enabling GPU BLAS support",
                        {{"n_gpu_layers", params.n_gpu_layers}});
            }
        }
        else if (arg == "--split-mode" || arg == "-sm")
        {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string arg_next = argv[i];
            if (arg_next == "none")
            {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            }
            else if (arg_next == "layer")
            {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            }
            else if (arg_next == "row")
            {
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            }
            else {
                invalid_param = true;
                break;
            }
        }
        else if (arg == "--tensor-split" || arg == "-ts")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            std::string arg_next = argv[i];

            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            GGML_ASSERT(split_arg.size() <= llama_max_devices());

            for (size_t i_device = 0; i_device < llama_max_devices(); ++i_device)
            {
                if (i_device < split_arg.size())
                {
                    params.tensor_split[i_device] = std::stof(split_arg[i_device]);
                }
                else
                {
                    params.tensor_split[i_device] = 0.0f;
                }
            }
        }
        else if (arg == "--main-gpu" || arg == "-mg")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.main_gpu = std::stoi(argv[i]);
        }
        else if (arg == "--lora")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_adapter.emplace_back(argv[i], 1.0f);
            params.use_mmap = false;
        }
        else if (arg == "--lora-scaled")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            const char * lora_adapter = argv[i];
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_adapter.emplace_back(lora_adapter, std::stof(argv[i]));
            params.use_mmap = false;
        }
        else if (arg == "--lora-base")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_base = argv[i];
        }
        else if (arg == "-v" || arg == "--verbose")
        {
#if SERVER_VERBOSE != 1
            LOG_WARNING("server.cpp is not built with verbose logging.", {});
#else
            server_verbose = true;
#endif
        }
        else if (arg == "--nobrowser")
        {
            sparams.nobrowser = true;
        }
        else if (arg == "--unsecure")
        {
            FLAG_unsecure = true;
        }
        else if (arg == "--server")
        {
        }
        else if (arg == "--fast")
        {
            FLAG_precise = false;
            FLAG_precision_specified = true;
        }
        else if (arg == "--precise")
        {
            FLAG_precise = true;
            FLAG_precision_specified = true;
        }
        else if (arg == "--trap")
        {
            FLAG_trap = true;
            FLAG_unsecure = true; // for better backtraces
            llamafile_trapping_enabled(+1);
        }
        else if (arg == "--nocompile")
        {
            FLAG_nocompile = true;
        }
        else if (arg == "--recompile")
        {
            FLAG_recompile = true;
        }
        else if (arg == "--gpu")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            FLAG_gpu = llamafile_gpu_parse(argv[i]);
            if (FLAG_gpu == -2)
            {
                fprintf(stderr, "error: invalid --gpu flag value: %s\n", argv[i]);
                exit(1);
            }
        }
        else if (arg == "--mlock")
        {
            params.use_mlock = true;
        }
        else if (arg == "--no-mmap")
        {
            params.use_mmap = false;
        }
        else if (arg == "--numa") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            } else {
                std::string value(argv[i]);
                /**/ if (value == "distribute" || value == "" ) { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
                else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
                else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
                else { invalid_param = true; break; }
            }
        }
        else if (arg == "--embedding")
        {
            params.embedding = true;
        }
        else if (arg == "-cb" || arg == "--cont-batching")
        {
            params.cont_batching = true;
        }
        else if (arg == "-np" || arg == "--parallel")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_parallel = std::stoi(argv[i]);
        } else if (arg == "-n" || arg == "--n-predict")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_predict = std::stoi(argv[i]);
        } else if (arg == "-spf" || arg == "--system-prompt-file")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                invalid_param = true;
                break;
            }
            std::string systm_content;
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(systm_content)
            );
            llama.process_system_prompt_data(json::parse(systm_content));
        }
        else if (arg == "-ctk" || arg == "--cache-type-k") {
            params.cache_type_k = argv[++i];
        }
        else if (arg == "-ctv" || arg == "--cache-type-v") {
            params.cache_type_v = argv[++i];
        }
        else if(arg == "--mmproj")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.mmproj = argv[i];
        }
        else if (arg == "--log-format")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            if (std::strcmp(argv[i], "json") == 0)
            {
                server_log_json = true;
            }
            else if (std::strcmp(argv[i], "text") == 0)
            {
                server_log_json = false;
            }
            else
            {
                invalid_param = true;
                break;
            }
        }
        else if (arg == "--log-disable")
        {
            log_set_target(stdout);
            LOG_INFO("logging to file is disabled.", {});
        }
        else if (arg == "--slots-endpoint-disable")
        {
            sparams.slots_endpoint = false;
        }
        else if (arg == "--metrics")
        {
            sparams.metrics_endpoint = true;
        }
        else if (arg == "--chat-template")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            if (!verify_custom_template(argv[i])) {
                fprintf(stderr, "error: the supplied chat template is not supported: %s\n", argv[i]);
                fprintf(stderr, "note: llama.cpp does not use jinja parser, we only support commonly used templates\n");
                invalid_param = true;
                break;
            }
            sparams.chat_template = argv[i];
        } else if (arg == "--override-kv") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            if (!parse_kv_override(argv[i], params.kv_overrides)) {
                fprintf(stderr, "error: Invalid type for KV override: %s\n", argv[i]);
                invalid_param = true;
                break;
            }
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            exit(1);
        }
    }

    params.embedding = true;  // [jart] #243 always enable embedding mode
    params.n_gpu_layers = llamafile_gpu_layers(params.n_gpu_layers);

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    if (invalid_param)
    {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        exit(1);
    }
}

/* llama.cpp completion api semantics */
static json format_partial_response(
    llama_server_context &llama, llama_client_slot *slot, const std::string &content, const std::vector<completion_token_output> &probs
) {
    json res = json
    {
        {"content",    content },
        {"stop",       false},
        {"slot_id",    slot->id },
        {"multimodal", llama.multimodal }
    };

    if (slot->sparams.n_probs > 0)
    {
        res["completion_probabilities"] = probs_vector_to_json(llama.ctx, probs);
    }

    return res;
}

static json format_tokenizer_response(const std::vector<llama_token> &tokens)
{
    return json {
        {"tokens", tokens}
    };
}

static json format_detokenized_response(std::string content)
{
    return json {
        {"content", content}
    };
}


static void log_server_request(const httplib::Request &req, const httplib::Response &res)
{
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health" || req.path == "/v1/completions")
    {
        return;
    }

    LOG_INFO("request", {
        {"remote_addr", req.remote_addr},
        {"remote_port", req.remote_port},
        {"status",      res.status},
        {"method",      req.method},
        {"path",        req.path},
        {"params",      req.params},
    });

    LOG_VERBOSE("request", {
        {"request",  req.body},
        {"response", res.body},
    });
}

struct token_translator
{
    llama_context * ctx;
    std::string operator()(llama_token tok)                    const { return llama_token_to_piece(ctx, tok, false); }
    std::string operator()(const completion_token_output &cto) const { return (*this)(cto.tok); }
};

static void append_to_generated_text_from_generated_token_probs(llama_server_context &llama, llama_client_slot *slot)
{
    auto & gtps = slot->generated_token_probs;
    auto translator = token_translator{llama.ctx};
    auto add_strlen = [=](size_t sum, const completion_token_output & cto) { return sum + translator(cto).size(); };
    const size_t len = std::accumulate(gtps.begin(), gtps.end(), size_t(0), add_strlen);
    if (slot->generated_text.capacity() < slot->generated_text.size() + len)
    {
        slot->generated_text.reserve(slot->generated_text.size() + len);
    }
    for (const completion_token_output & cto : gtps)
    {
        slot->generated_text += translator(cto);
    }
}

std::function<void(int)> shutdown_handler;
inline void signal_handler(int signal) { shutdown_handler(signal); }

static const char *sockaddr2str(const struct sockaddr *sa, char *buf, size_t size) {
  if (sa->sa_family == AF_INET) {
    return inet_ntop(AF_INET, &(((const struct sockaddr_in *)sa)->sin_addr),
                     buf, size);
  } else if (sa->sa_family == AF_INET6) {
    return inet_ntop(AF_INET6, &(((const struct sockaddr_in6 *)sa)->sin6_addr),
                     buf, size);
  } else {
    return 0;
  }
}

int server_cli(int argc, char **argv)
{
#if SERVER_VERBOSE != 1
    log_disable();
#endif
    // own arguments required by this example
    gpt_params params;
    server_params sparams;

    // struct that contains llama context and inference
    llama_server_context llama;

    server_params_parse(argc, argv, sparams, params, llama);

    if (params.model_alias == "unknown")
    {
        params.model_alias = params.model;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INFO("build info", {{"build", LLAMA_BUILD_NUMBER},
                            {"commit", LLAMA_COMMIT}});

    LOG_INFO("system info", {
                                {"n_threads", params.n_threads},
                                {"n_threads_batch", params.n_threads_batch},
                                {"total_threads", std::thread::hardware_concurrency()},
                                {"system_info", llama_print_system_info()},
                            });

    httplib::Server svr;

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr.set_default_headers({{"Server", "llama.cpp"}});

    // CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.set_header("Access-Control-Allow-Headers", "*");
    });

    svr.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
        server_state current_state = state.load();
        switch(current_state) {
            case SERVER_STATE_READY: {
                // request slots data using task queue
                task_server task;
                task.id   = llama.queue_tasks.get_new_id();
                task.type = TASK_TYPE_METRICS;
                task.target_id = -1;

                llama.queue_results.add_waiting_task_id(task.id);
                llama.queue_tasks.post(task);

                // get the result
                task_result result = llama.queue_results.recv(task.id);
                llama.queue_results.remove_waiting_task_id(task.id);

                int n_idle_slots       = result.result_json["idle"];
                int n_processing_slots = result.result_json["processing"];

                json health = {
                        {"status",           "ok"},
                        {"slots_idle",       n_idle_slots},
                        {"slots_processing", n_processing_slots}};
                res.status = 200; // HTTP OK
                if (sparams.slots_endpoint && req.has_param("include_slots")) {
                    health["slots"] = result.result_json["slots"];
                }

                if (n_idle_slots == 0) {
                    health["status"] = "no slot available";
                    if (req.has_param("fail_on_no_slot")) {
                        res.status = 503; // HTTP Service Unavailable
                    }
                }
                res.set_content(health.dump(), "application/json");
                break;
            }
            case SERVER_STATE_LOADING_MODEL:
                res.set_content(R"({"status": "loading model"})", "application/json");
                res.status = 503; // HTTP Service Unavailable
                break;
            case SERVER_STATE_ERROR:
                res.set_content(R"({"status": "error", "error": "Model failed to load"})", "application/json");
                res.status = 500; // HTTP Internal Server Error
                break;
        }
    });

    if (sparams.slots_endpoint) {
        svr.Get("/slots", [&](const httplib::Request&, httplib::Response& res) {
            // request slots data using task queue
            task_server task;
            task.id = llama.queue_tasks.get_new_id();
            task.type = TASK_TYPE_METRICS;
            task.target_id = -1;

            llama.queue_results.add_waiting_task_id(task.id);
            llama.queue_tasks.post(task);

            // get the result
            task_result result = llama.queue_results.recv(task.id);
            llama.queue_results.remove_waiting_task_id(task.id);

            res.set_content(result.result_json["slots"].dump(), "application/json");
            res.status = 200; // HTTP OK
        });
    }

    if (sparams.metrics_endpoint) {
        svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
            // request slots data using task queue
            task_server task;
            task.id = llama.queue_tasks.get_new_id();
            task.type = TASK_TYPE_METRICS;
            task.target_id = -1;

            llama.queue_results.add_waiting_task_id(task.id);
            llama.queue_tasks.post(task);

            // get the result
            task_result result = llama.queue_results.recv(task.id);
            llama.queue_results.remove_waiting_task_id(task.id);

            json data = result.result_json;

            uint64_t n_prompt_tokens_processed = data["n_prompt_tokens_processed"];
            uint64_t t_prompt_processing       = data["t_prompt_processing"];

            uint64_t n_tokens_predicted       = data["n_tokens_predicted"];
            uint64_t t_tokens_generation      = data["t_tokens_generation"];

            int32_t kv_cache_used_cells = data["kv_cache_used_cells"];

            // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
            json all_metrics_def = json {
                    {"counter", {{
                            {"name",  "prompt_tokens_total"},
                            {"help",  "Number of prompt tokens processed."},
                            {"value",  data["n_prompt_tokens_processed_total"]}
                    }, {
                            {"name",  "tokens_predicted_total"},
                            {"help",  "Number of generation tokens processed."},
                            {"value",  data["n_tokens_predicted_total"]}
                    }}},
                    {"gauge", {{
                            {"name",  "prompt_tokens_seconds"},
                            {"help",  "Average prompt throughput in tokens/s."},
                            {"value",  n_prompt_tokens_processed ? 1e3 / t_prompt_processing * n_prompt_tokens_processed : 0}
                    },{
                            {"name",  "predicted_tokens_seconds"},
                            {"help",  "Average generation throughput in tokens/s."},
                            {"value",  n_tokens_predicted ? 1e3 / t_tokens_generation * n_tokens_predicted : 0}
                     },{
                            {"name",  "kv_cache_usage_ratio"},
                            {"help",  "KV-cache usage. 1 means 100 percent usage."},
                            {"value",  1. * kv_cache_used_cells / params.n_ctx}
                     },{
                            {"name",  "kv_cache_tokens"},
                            {"help",  "KV-cache tokens."},
                            {"value",  data["kv_cache_tokens_count"]}
                    },{
                            {"name",  "requests_processing"},
                            {"help",  "Number of request processing."},
                            {"value",  data["processing"]}
                  },{
                            {"name",  "requests_deferred"},
                            {"help",  "Number of request deferred."},
                            {"value",  data["deferred"]}
                  }}}
            };

            std::stringstream prometheus;
            for (const auto& el : all_metrics_def.items()) {
                const auto& type = el.key();
                const auto& metrics_def = el.value();
                for (const auto& metric_def : metrics_def) {
                    std::string name = metric_def["name"];
                    std::string help = metric_def["help"];
                    prometheus << "# HELP llamacpp:" << name << " " << help                << "\n"
                               << "# TYPE llamacpp:" << name << " " << type                << "\n"
                               << "llamacpp:"        << name << " " << metric_def["value"] << "\n";
                }
            }

            res.set_content(prometheus.str(), "text/plain; version=0.0.4");
            res.status = 200; // HTTP OK
        });
    }

    svr.set_logger(log_server_request);

    svr.set_exception_handler([](const httplib::Request &, httplib::Response &res, std::exception_ptr ep)
            {
                const char fmt[] = "500 Internal Server Error\n%s";
                char buf[BUFSIZ];
                try
                {
                    std::rethrow_exception(std::move(ep));
                }
                catch (std::exception &e)
                {
                    snprintf(buf, sizeof(buf), fmt, e.what());
                }
                catch (...)
                {
                    snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
                }
                res.set_content(buf, "text/plain; charset=utf-8");
                res.status = 500;
            });

    svr.set_error_handler([](const httplib::Request &, httplib::Response &res)
            {
                if (res.status == 401)
                {
                    res.set_content("Unauthorized", "text/plain; charset=utf-8");
                }
                if (res.status == 400)
                {
                    res.set_content("Invalid request", "text/plain; charset=utf-8");
                }
                else if (res.status == 404)
                {
                    res.set_content("File Not Found", "text/plain; charset=utf-8");
                    res.status = 404;
                }
            });

    // load the model
    if (!llama.load_model(params))
    {
        state.store(SERVER_STATE_ERROR);
        return 1;
    } else {
        llama.initialize();
        state.store(SERVER_STATE_READY);
        LOG_INFO("model loaded", {});
    }

    // set timeouts and change hostname and port
    svr.set_read_timeout (sparams.read_timeout);
    svr.set_write_timeout(sparams.write_timeout);

    for (int i = 0;; ++i) {
        if (svr.bind_to_port(sparams.hostname, sparams.port)) {
            break;
        } else if (i < 10) {
            ++sparams.port;
        } else {
            fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n", sparams.hostname.c_str(), sparams.port);
            return 1;
        }
    }

    // Set the base directory for serving static files
    svr.set_base_dir(sparams.public_path);

    // to make it ctrl+clickable:
    const char *connect_host;
    if (sparams.hostname != "0.0.0.0") {
        connect_host = sparams.hostname.c_str();
        LOG_TEE("\nllama server listening at http://%s:%d\n\n",
                sparams.hostname.c_str(), sparams.port);
    } else {
        struct ifaddrs *ifaddrs;
        connect_host = "127.0.0.1";
        if (!getifaddrs(&ifaddrs)) {
            LOG_TEE("\n");
            for (struct ifaddrs *ifa = ifaddrs; ifa; ifa = ifa->ifa_next) {
                char buf[128];
                if (!(ifa->ifa_flags & IFF_UP)) continue;
                LOG_TEE("llama server listening at http://%s%s%s:%d\n",
                        ifa->ifa_addr->sa_family == AF_INET6 ? "[" : "",
                        sockaddr2str(ifa->ifa_addr, buf, sizeof(buf)),
                        ifa->ifa_addr->sa_family == AF_INET6 ? "]" : "",
                        sparams.port);
            }
            LOG_TEE("\n");
        } else {
            perror("getifaddrs");
            LOG_TEE("\nllama server listening at http://%s:%d\n\n",
                    sparams.hostname.c_str(), sparams.port);
        }
    }

    std::unordered_map<std::string, std::string> log_data;
    log_data["hostname"] = sparams.hostname;
    log_data["port"] = std::to_string(sparams.port);

    if (sparams.api_keys.size() == 1) {
        log_data["api_key"] = "api_key: ****" + sparams.api_keys[0].substr(sparams.api_keys[0].length() - 4);
    } else if (sparams.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(sparams.api_keys.size()) + " keys loaded";
    }

    // launch browser tab
    if (!sparams.nobrowser) {
        char url[128];
        snprintf(url, sizeof(url), "http://%s:%d/", connect_host, sparams.port);
        llamafile_launch_browser(url);
    }

    if (!FLAG_unsecure) {
        if (IsXnu()) {
            // Cosmopolitan libc explicitly does not support cosmo_dlopen on x64
            // macOS and mac_sandbox_init depends on cosmo_dlopen. We'll attempt
            // to enable sandboxing and ignore the failure on x64 macOS.
            // (Alternatively, the sandbox-exec(1) command could be used to
            // launch llamafile with a sandbox profile.) A goal of llamafile is
            // to work easily out-of-the-box and it already operates without a
            // sandbox on macOS so don't require --unsecure on x64 macOS to
            // workaround the failure. If support is added later, it will be
            // used without further changes. On Apple Silicon, if sandbox
            // initialization fails, exit and recommend using --unsecure to
            // disable sandboxing.
            std::string error_msg;
            int rv = mac_sandbox_init(error_msg);
            if (rv != 0) {
                fprintf(stderr, "Sandbox initialization failed:\n"
                    "rv: %d, error: %s.\n"
                    "Disable sandboxing with the --unsecure option.\n",
                    rv, error_msg.c_str());
                if (IsXnuSilicon()) {
                    exit(1); 
                } else {
                    LOG_WARNING("Sandbox initialization failed. Not supported "
                        "on x64 macOS. Continuing.", {});
                }
            }
        } else if (!llamafile_has_gpu()) {
            printf("In the sandboxing block!\n");
            // Enables pledge() security on Linux and OpenBSD.
            // - We do this *after* binding the server socket.
            // - We do this *after* opening the log file for writing.
            // - We do this *after* opening a tab in the user browser.
            // - We do this *before* loading any weights or graphdefs.
            // In effect, what this does is:
            // - Filesystem access is disabled entirely (except ZipOS).
            // - On Linux, network access is restricted to accept() only.
            // Cosmopolitan Libc implements pledge() on Linux using SECCOMP.
            char promises[32];
            if (IsOpenbsd()) {
                strlcpy(promises, "stdio inet", sizeof(promises));
            } else {
                strlcpy(promises, "stdio anet", sizeof(promises));
            }
            if (!startswith(sparams.public_path.c_str(), "/zip/")) {
                strlcat(promises, " rpath", sizeof(promises));
            }
            __pledge_mode = PLEDGE_PENALTY_RETURN_EPERM;
            if (pledge(0, 0)) {
                LOG_TEE("warning: this OS doesn't support pledge() security\n");
            } else if (pledge(promises, 0)) {
                perror("pledge");
                exit(1);
            }
        }
    }

    LOG_INFO("HTTP server listening", log_data);
    // run the HTTP server in a thread - see comment below
    std::thread t([&]()
            {
                if (!svr.listen_after_bind())
                {
                    state.store(SERVER_STATE_ERROR);
                    return 1;
                }

                return 0;
            });

    if (sparams.chat_template.empty()) { // custom chat template is not supplied
        // check if the template comes with the model is supported by us
        llama.validate_model_chat_template(sparams);
    }

    // Middleware for API key validation
    auto validate_api_key = [&sparams](const httplib::Request &req, httplib::Response &res) -> bool {
        // If API key is not set, skip validation
        if (sparams.api_keys.empty()) {
            return true;
        }

        // Check for API key in the header
        auto auth_header = req.get_header_value("Authorization");
        std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(sparams.api_keys.begin(), sparams.api_keys.end(), received_api_key) != sparams.api_keys.end()) {
                return true; // API key is valid
            }
        }

        // API key is invalid or not provided
        res.set_content("Unauthorized: Invalid API Key", "text/plain; charset=utf-8");
        res.status = 401; // Unauthorized

        LOG_WARNING("Unauthorized: Invalid API Key", {});

        return false;
    };

    svr.Get("/props", [&llama](const httplib::Request & req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                json data = {
                    { "user_name",      llama.name_user.c_str() },
                    { "assistant_name", llama.name_assistant.c_str() },
                    { "default_generation_settings", llama.default_generation_settings_for_props },
                    { "total_slots",    llama.params.n_parallel }
                };
                res.set_content(data.dump(), "application/json; charset=utf-8");
            });

    svr.Post("/completion", [&llama, &validate_api_key](const httplib::Request &req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                if (!validate_api_key(req, res)) {
                    return;
                }
                json data = json::parse(req.body);
                const int task_id = llama.queue_tasks.get_new_id();
                llama.queue_results.add_waiting_task_id(task_id);
                llama.request_completion(task_id, data, false, false, -1);
                if (!json_value(data, "stream", false)) {
                    std::string completion_text;
                    task_result result = llama.queue_results.recv(task_id);
                    if (!result.error && result.stop) {
                        res.set_content(result.result_json.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
                    }
                    else
                    {
                        res.status = 404;
                        res.set_content(result.result_json["content"], "text/plain; charset=utf-8");
                    }
                    llama.queue_results.remove_waiting_task_id(task_id);
                } else {
                    const auto chunked_content_provider = [task_id, &llama](size_t, httplib::DataSink & sink)
                    {
                        while (true)
                        {
                            task_result result = llama.queue_results.recv(task_id);
                            if (!result.error) {
                                const std::string str =
                                    "data: " +
                                    result.result_json.dump(-1, ' ', false, json::error_handler_t::replace) +
                                    "\n\n";
                                LOG_VERBOSE("data stream", {
                                    { "to_send", str }
                                });
                                if (!sink.write(str.c_str(), str.size()))
                                {
                                    llama.queue_results.remove_waiting_task_id(task_id);
                                    return false;
                                }
                                if (result.stop) {
                                    break;
                                }
                            } else {
                                const std::string str =
                                    "error: " +
                                    result.result_json.dump(-1, ' ', false, json::error_handler_t::replace) +
                                    "\n\n";
                                LOG_VERBOSE("data stream", {
                                    { "to_send", str }
                                });
                                if (!sink.write(str.c_str(), str.size()))
                                {
                                    llama.queue_results.remove_waiting_task_id(task_id);
                                    return false;
                                }
                                break;
                            }
                        }

                        llama.queue_results.remove_waiting_task_id(task_id);
                        sink.done();
                        return true;
                    };

                    auto on_complete = [task_id, &llama] (bool)
                    {
                        // cancel
                        llama.request_cancel(task_id);
                        llama.queue_results.remove_waiting_task_id(task_id);
                    };

                    res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
                }
            });

    svr.Get("/v1/models", [&params](const httplib::Request& req, httplib::Response& res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                std::time_t t = std::time(0);

                json models = {
                    {"object", "list"},
                    {"data", {
                        {
                            {"id", params.model_alias},
                            {"object", "model"},
                            {"created", t},
                            {"owned_by", "llamacpp"}
                        },
                    }}
                };

                res.set_content(models.dump(), "application/json; charset=utf-8");
            });


    // TODO: add mount point without "/v1" prefix -- how?
    svr.Post("/v1/chat/completions", [&llama, &validate_api_key, &sparams](const httplib::Request &req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                if (!validate_api_key(req, res)) {
                    return;
                }
                json data = oaicompat_completion_params_parse(llama.model, json::parse(req.body), sparams.chat_template);

                const int task_id = llama.queue_tasks.get_new_id();
                llama.queue_results.add_waiting_task_id(task_id);
                llama.request_completion(task_id, data, false, false, -1);

                if (!json_value(data, "stream", false)) {
                    std::string completion_text;
                    task_result result = llama.queue_results.recv(task_id);

                    if (!result.error && result.stop) {
                        json oaicompat_result = format_final_response_oaicompat(data, result);

                        res.set_content(oaicompat_result.dump(-1, ' ', false,
                                            json::error_handler_t::replace),
                                            "application/json; charset=utf-8");
                    } else {
                        res.status = 500;
                        res.set_content(result.result_json["content"], "text/plain; charset=utf-8");
                    }
                    llama.queue_results.remove_waiting_task_id(task_id);
                } else {
                    const auto chunked_content_provider = [task_id, &llama](size_t, httplib::DataSink &sink) {
                        while (true) {
                            task_result llama_result = llama.queue_results.recv(task_id);
                            if (!llama_result.error) {
                                std::vector<json> result_array = format_partial_response_oaicompat( llama_result);

                                for (auto it = result_array.begin(); it != result_array.end(); ++it)
                                {
                                    if (!it->empty()) {
                                        const std::string str =
                                            "data: " +
                                            it->dump(-1, ' ', false, json::error_handler_t::replace) +
                                            "\n\n";
                                        LOG_VERBOSE("data stream", {{"to_send", str}});
                                        if (!sink.write(str.c_str(), str.size())) {
                                            llama.queue_results.remove_waiting_task_id(task_id);
                                            return false;
                                        }
                                    }
                                }
                                if (llama_result.stop) {
                                    break;
                                }
                            } else {
                                const std::string str =
                                    "error: " +
                                    llama_result.result_json.dump(-1, ' ', false,
                                            json::error_handler_t::replace) +
                                    "\n\n";
                                LOG_VERBOSE("data stream", {{"to_send", str}});
                                if (!sink.write(str.c_str(), str.size())) {
                                    llama.queue_results.remove_waiting_task_id(task_id);
                                    return false;
                                }
                                break;
                            }
                        }
                        sink.done();
                        llama.queue_results.remove_waiting_task_id(task_id);
                        return true;
                    };

                    auto on_complete = [task_id, &llama](bool) {
                        // cancel request
                        llama.request_cancel(task_id);
                        llama.queue_results.remove_waiting_task_id(task_id);
                    };

                    res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
                }
            });

    svr.Post("/infill", [&llama, &validate_api_key](const httplib::Request &req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                if (!validate_api_key(req, res)) {
                    return;
                }
                json data = json::parse(req.body);
                const int task_id = llama.queue_tasks.get_new_id();
                llama.queue_results.add_waiting_task_id(task_id);
                llama.request_completion(task_id, data, true, false, -1);
                if (!json_value(data, "stream", false)) {
                    std::string completion_text;
                    task_result result = llama.queue_results.recv(task_id);
                    if (!result.error && result.stop)
                    {
                        res.set_content(result.result_json.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
                    }
                    else
                    {
                        res.status = 404;
                        res.set_content(result.result_json["content"], "text/plain; charset=utf-8");
                    }
                    llama.queue_results.remove_waiting_task_id(task_id);
                } else {
                    const auto chunked_content_provider = [task_id, &llama](size_t, httplib::DataSink & sink) {
                        while (true)
                        {
                            task_result result = llama.queue_results.recv(task_id);
                            if (!result.error) {
                                const std::string str =
                                "data: " +
                                result.result_json.dump(-1, ' ', false, json::error_handler_t::replace) +
                                "\n\n";
                                LOG_VERBOSE("data stream", {
                                    { "to_send", str }
                                });
                                if (!sink.write(str.c_str(), str.size()))
                                {
                                    llama.queue_results.remove_waiting_task_id(task_id);
                                    return false;
                                }
                                if (result.stop)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                break;
                            }
                        }

                        llama.queue_results.remove_waiting_task_id(task_id);
                        sink.done();
                        return true;
                    };

                    auto on_complete = [task_id, &llama] (bool)
                    {
                        // cancel
                        llama.request_cancel(task_id);
                    };

                    res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
                }
            });

    svr.Options(R"(/.*)", [](const httplib::Request &, httplib::Response &res)
                { return res.set_content("", "application/json; charset=utf-8"); });

    svr.Post("/tokenize", [&llama](const httplib::Request &req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                const json body = json::parse(req.body);
                std::vector<llama_token> tokens;
                if (body.count("content") != 0)
                {
                    tokens = llama.tokenize(body["content"], false);
                }
                const json data = format_tokenizer_response(tokens);
                return res.set_content(data.dump(), "application/json; charset=utf-8");
            });

    svr.Post("/detokenize", [&llama](const httplib::Request &req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                const json body = json::parse(req.body);
                std::string content;
                if (body.count("tokens") != 0)
                {
                    const std::vector<llama_token> tokens = body["tokens"];
                    content = tokens_to_str(llama.ctx, tokens.cbegin(), tokens.cend());
                }

                const json data = format_detokenized_response(content);
                return res.set_content(data.dump(), "application/json; charset=utf-8");
            });

    svr.Post("/embedding", [&llama](const httplib::Request &req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                const json body = json::parse(req.body);
                json prompt;
                if (body.count("content") != 0)
                {
                    prompt = body["content"];
                }
                else
                {
                    prompt = "";
                }

                json image_data;
                if (body.count("image_data") != 0) {
                    image_data = body["image_data"];
                }
                else
                {
                    image_data = "";
                }

                // create and queue the task
                const int task_id = llama.queue_tasks.get_new_id();
                llama.queue_results.add_waiting_task_id(task_id);
                llama.request_completion(task_id, { {"prompt", prompt}, { "n_predict", 0}, {"image_data", image_data} }, false, true, -1);

                // get the result
                task_result result = llama.queue_results.recv(task_id);
                llama.queue_results.remove_waiting_task_id(task_id);

                // send the result
                return res.set_content(result.result_json.dump(), "application/json; charset=utf-8");
            });

    svr.Post("/v1/embeddings", [&llama](const httplib::Request &req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                const json body = json::parse(req.body);

                json prompt;
                if (body.count("input") != 0)
                {
                    prompt = body["input"];
                    // batch
                    if(prompt.is_array()) {
                        json data = json::array();
                        int i = 0;
                        for (const json &elem : prompt) {
                            const int task_id = llama.queue_tasks.get_new_id();
                            llama.queue_results.add_waiting_task_id(task_id);
                            llama.request_completion(task_id, { {"prompt", elem}, { "n_predict", 0} }, false, true, -1);

                            // get the result
                            task_result result = llama.queue_results.recv(task_id);
                            llama.queue_results.remove_waiting_task_id(task_id);

                            json embedding = json{
                                {"embedding", json_value(result.result_json, "embedding", json::array())},
                                {"index", i++},
                                {"object", "embedding"}
                            };
                            data.push_back(embedding);
                        }
                        json result = format_embeddings_response_oaicompat(body, data);
                        return res.set_content(result.dump(), "application/json; charset=utf-8");
                    }
                }
                else
                {
                    prompt = "";
                }

                // create and queue the task
                const int task_id = llama.queue_tasks.get_new_id();
                llama.queue_results.add_waiting_task_id(task_id);
                llama.request_completion(task_id, { {"prompt", prompt}, { "n_predict", 0}}, false, true, -1);

                // get the result
                task_result result = llama.queue_results.recv(task_id);
                llama.queue_results.remove_waiting_task_id(task_id);

                json data = json::array({json{
                        {"embedding", json_value(result.result_json, "embedding", json::array())},
                        {"index", 0},
                        {"object", "embedding"}
                    }}
                );

                json root = format_embeddings_response_oaicompat(body, data);

                // send the result
                return res.set_content(root.dump(), "application/json; charset=utf-8");
            });

    // GG: if I put the main loop inside a thread, it crashes on the first request when build in Debug!?
    //     "Bus error: 10" - this is on macOS, it does not crash on Linux
    //std::thread t2([&]()
    /*{
        bool running = true;
        while (running)
        {
            running = llama.update_slots();
        }
    }*/
    //);

    llama.queue_tasks.on_new_task(std::bind(
        &llama_server_context::process_single_task, &llama, std::placeholders::_1));
    llama.queue_tasks.on_finish_multitask(std::bind(
        &llama_server_context::on_finish_multitask, &llama, std::placeholders::_1));
    llama.queue_tasks.on_all_tasks_finished(std::bind(
        &llama_server_context::run_on_all_tasks_finished, &llama));
    llama.queue_results.on_multitask_update(std::bind(
        &llama_server_queue::update_multitask,
        &llama.queue_tasks,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
    ));

    shutdown_handler = [&](int) {
        llama.queue_tasks.terminate();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    llama.queue_tasks.start_loop();
    svr.stop();
    t.join();

    llama_backend_free();
    return 0;
}
