// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

// Various helper functions and utilities

#pragma once

#include "llamafile/log.h"
#include "llama.cpp/cores.h"
#include "llama.h"

#include "sampling.h"
#include "llamafile/version.h"
#include "llamafile/llamafile.h"

#define LOG_NO_FILE_LINE_FUNCTION
#include "log.h"

#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>
#include <tuple>
#include <cosmo.h>

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

#define print_build_info() do {                                         \
        tinylog(__func__, ": llamafile version " LLAMAFILE_VERSION_STRING "\n", NULL); \
} while(0)

#define DEFAULT_MODEL_PATH "models/7B/ggml-model-f16.gguf"

struct llama_lora_adapter_info {
    std::string path;
    float scale;
};

struct llama_lora_adapter_container : llama_lora_adapter_info {
    struct llama_lora_adapter * adapter;
};

// build info
extern int LLAMA_BUILD_NUMBER;
extern char const * LLAMA_COMMIT;
extern char const * LLAMA_COMPILER;
extern char const * LLAMA_BUILD_TARGET;

struct llama_control_vector_load_info;

//
// CPU utils
//

int32_t cpu_get_num_physical_cores();

//
// CLI argument parsing
//

// dimensionality reduction methods, used by cvector-generator
enum dimre_method {
    DIMRE_METHOD_PCA,
    DIMRE_METHOD_MEAN,
};

struct gpt_params {
    uint32_t seed                 = LLAMA_DEFAULT_SEED; // RNG seed

    int32_t n_threads             = cpu_get_num_math();
    int32_t n_threads_draft       =    -1;
    int32_t n_threads_batch       =    -1; // number of threads to use for batch processing (-1 = use n_threads)
    int32_t n_threads_batch_draft =    -1;
    int32_t n_predict             =    -1; // new tokens to predict
    int32_t n_ctx                 =  8192; // context size [jart]
    int32_t n_batch               =  2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_ubatch              =   512; // physical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                =     0; // number of tokens to keep from initial prompt
    int32_t n_draft               =     5; // number of tokens to draft during speculative decoding
    int32_t n_chunks              =    -1; // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel            =     1; // number of parallel sequences to decode
    int32_t n_sequences           =     1; // number of sequences to decode
    float   p_split               =  0.1f; // speculative decoding split probability
    int32_t n_gpu_layers          =    -1; // number of layers to store in VRAM (-1 - use default)
    int32_t n_gpu_layers_draft    =    -1; // number of layers to store in VRAM for the draft model (-1 - use default)
    int32_t main_gpu              =     0; // the GPU that is used for scratch and small tensors
    float   tensor_split[128]     =   {0}; // how split tensors should be distributed across GPUs
    int32_t grp_attn_n            =     1; // group-attention factor
    int32_t grp_attn_w            =   512; // group-attention width
    int32_t n_print               =    -1; // print token count every n tokens (-1 = disabled)
    float   rope_freq_base        =  0.0f; // RoPE base frequency
    float   rope_freq_scale       =  0.0f; // RoPE frequency scaling factor
    float   yarn_ext_factor       = -1.0f; // YaRN extrapolation mix factor
    float   yarn_attn_factor      =  1.0f; // YaRN magnitude scaling factor
    float   yarn_beta_fast        = 32.0f; // YaRN low correction dim
    float   yarn_beta_slow        =  1.0f; // YaRN high correction dim
    int32_t yarn_orig_ctx         =     0; // YaRN original context length
    float   defrag_thold          = -1.0f; // KV cache defragmentation threshold

    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data                 = nullptr;

    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

    enum llama_split_mode        split_mode        = LLAMA_SPLIT_MODE_LAYER; // how to split the model across GPUs
    enum llama_rope_scaling_type rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    enum llama_pooling_type      pooling_type      = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling type for embeddings
    enum llama_attention_type    attention_type    = LLAMA_ATTENTION_TYPE_UNSPECIFIED; // attention type for embeddings

    // // sampling parameters
    struct llama_sampling_params sparams;

    std::string model                = ""; // model path
    std::string model_draft          = ""; // draft model for speculative decoding
    std::string model_alias          = "unknown"; // model alias
    std::string model_url            = ""; // model url to download
    std::string hf_token             = ""; // HF token
    std::string hf_repo              = ""; // HF repo
    std::string hf_file              = ""; // HF file
    std::string prompt               = "";
    std::string prompt_file          = ""; // store the external prompt file name
    std::string path_prompt_cache    = ""; // path to file for saving/loading prompt eval state
    std::string input_prefix         = ""; // string to prefix user inputs with
    std::string input_suffix         = ""; // string to suffix user inputs with
    std::string logdir               = ""; // directory in which to save YAML log files
    std::string lookup_cache_static  = ""; // path of static ngram cache file for lookup decoding
    std::string lookup_cache_dynamic = ""; // path of dynamic ngram cache file for lookup decoding
    std::string logits_file          = ""; // file for saving *all* logits
    std::string rpc_servers          = ""; // comma separated list of RPC servers

    std::vector<std::string> in_files;   // all input files
    std::vector<std::string> antiprompt; // strings upon which more user input is prompted (a.k.a. reverse prompts)
    std::vector<llama_model_kv_override> kv_overrides;

    bool lora_init_without_apply = false; // only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_lora_adapter_apply)
    std::vector<llama_lora_adapter_info> lora_adapters; // lora adapter path with user defined scale

    std::vector<llama_control_vector_load_info> control_vectors; // control vector with user defined scale

    int32_t verbosity                  = 0;
    int32_t control_vector_layer_start = -1; // layer range for control vector
    int32_t control_vector_layer_end   = -1; // layer range for control vector

    int32_t ppl_stride      = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int32_t ppl_output_type = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                     //                                       (which is more convenient to use for plotting)
                                     //
    bool   hellaswag        = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks  = 400;   // number of tasks to use when computing the HellaSwag score

    bool   winogrande       = false; // compute Winogrande score over random tasks from datafile supplied in prompt
    size_t winogrande_tasks = 0;     // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

    bool   multiple_choice  = false;  // compute TruthfulQA score over random tasks from datafile supplied in prompt
    size_t multiple_choice_tasks = 0; // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

    bool   kl_divergence    = false; // compute KL divergence

    bool usage             = false; // print usage
    bool use_color         = false; // use color to distinguish generations and inputs
    bool special           = false; // enable special token output
    bool interactive       = false; // interactive mode
    bool interactive_first = false; // wait for user input immediately
    bool conversation      = false; // conversation mode (does not print special tokens and suffix/prefix)
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool escape            = true;  // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles
    bool cont_batching     = true;  // insert new sequences for decoding on-the-fly
    bool flash_attn        = false; // flash attention

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool ignore_eos        = false; // ignore generated EOS tokens
    bool logits_all        = false; // return logits for all tokens in the batch
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool verbose_prompt    = false; // print prompt tokens before generation
    bool display_prompt    = true;  // print prompt before generation
    bool infill            = false; // use infill mode
    bool dump_kv_cache     = false; // dump the KV cache contents for debugging purposes
    bool no_kv_offload     = false; // disable KV offloading
    bool warmup            = true;  // warmup run
    bool check_tensors     = false; // validate tensor data

    std::string cache_type_k = X86_HAVE(AVX512_BF16) ? "bf16" : "f16"; // KV cache data type for the K [jart]
    std::string cache_type_v = X86_HAVE(AVX512_BF16) ? "bf16" : "f16"; // KV cache data type for the V [jart]

    // multimodal models (see examples/llava)
    std::string mmproj = "";        // path to multimodal projector
    std::vector<std::string> image; // path to image file(s)

    // embedding
    bool embedding         = false; // get only sentence embedding
    int32_t embd_normalize = 2;     // normalisation for embendings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
    std::string embd_out   = "";    // empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
    std::string embd_sep   = "\n";  // separator of embendings

    // server params
    int32_t port           = 8080;         // server listens on this network port
    int32_t timeout_read   = 600;          // http read timeout in seconds
    int32_t timeout_write  = timeout_read; // http write timeout in seconds
    int32_t n_threads_http = -1;           // number of threads to process HTTP requests

    std::string hostname      = "127.0.0.1";
    std::string public_path   = "";
    std::string url_prefix    = "";
    std::string chat_template = "";
    std::string system_prompt = "";
    bool enable_chat_template = true;

    std::vector<std::string> api_keys;

    std::string ssl_file_key  = "";
    std::string ssl_file_cert = "";

    bool endpoint_slots   = true;
    bool endpoint_metrics = false;

    bool log_json = false;

    std::string slot_save_path;

    float slot_prompt_similarity = 0.5f;

    // batched-bench params
    bool is_pp_shared = false;

    std::vector<int32_t> n_pp;
    std::vector<int32_t> n_tg;
    std::vector<int32_t> n_pl;

    // retrieval params
    std::vector<std::string> context_files; // context files to embed

    int32_t chunk_size = 64; // chunk size for context embedding

    std::string chunk_separator = "\n"; // chunk separator for context embedding

    // passkey params
    int32_t n_junk = 250; // number of times to repeat the junk text
    int32_t i_pos  = -1;  // position of the passkey in the junk text

    // imatrix params
    std::string out_file = "imatrix.dat"; // save the resulting imatrix to this file

    int32_t n_out_freq  = 10; // output the imatrix every n_out_freq iterations
    int32_t n_save_freq =  0; // save the imatrix every n_save_freq iterations
    int32_t i_chunk     =  0; // start processing from this chunk

    bool process_output = false; // collect data for the output tensor
    bool compute_ppl    = true;  // whether to compute perplexity

    // cvector-generator params
    int n_pca_batch = 100;
    int n_pca_iterations = 1000;
    dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    std::string cvector_outfile       = "control_vector.gguf";
    std::string cvector_positive_file = "examples/cvector-generator/positive.txt";
    std::string cvector_negative_file = "examples/cvector-generator/negative.txt";

    bool spm_infill = false; // suffix/prefix/middle pattern for infill

    std::string lora_outfile = "ggml-lora-merged-f16.gguf";
};

void gpt_params_handle_hf_token(gpt_params & params);
void gpt_params_handle_model_default(gpt_params & params);

bool gpt_params_parse_ex   (int argc, char ** argv, gpt_params & params);
bool gpt_params_parse      (int argc, char ** argv, gpt_params & params);
bool gpt_params_find_arg   (int argc, char ** argv, const std::string & arg, gpt_params & params, int & i, bool & invalid_param);
void gpt_params_print_usage(int argc, char ** argv, const gpt_params & params);

std::string gpt_params_get_system_info(const gpt_params & params);

//
// String utils
//

std::vector<std::string> string_split(std::string input, char separator);

template<class T>
static std::vector<T> string_split(const std::string & str, char delim) {
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        token_stream >> value;
        values.push_back(value);
    }
    return values;
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides);

//
// Model utils
//

struct llama_init_result {
    struct llama_model   * model   = nullptr;
    struct llama_context * context = nullptr;
    std::vector<llama_lora_adapter_container> lora_adapters;
};

struct llama_init_result    llama_init_from_gpt_params(gpt_params & params);

struct llama_model_params   llama_model_params_from_gpt_params  (const gpt_params & params);
struct llama_context_params llama_context_params_from_gpt_params(const gpt_params & params);

struct llama_model * llama_load_model_from_url(const char * model_url, const char * path_model, const char * hf_token, const struct llama_model_params & params);
struct llama_model * llama_load_model_from_hf(const char * repo, const char * file, const char * path_model, const char * hf_token, const struct llama_model_params & params);

// clear LoRA adapters from context, then apply new list of adapters
void llama_lora_adapters_apply(struct llama_context * ctx, std::vector<llama_lora_adapter_container> & lora_adapters);

// Batch utils

void llama_batch_clear(struct llama_batch & batch);

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
std::string llama_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token,
                       bool          special = true);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// optionally renders special/control tokens
std::string llama_detokenize(
                         llama_context * ctx,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

// Uses the value from the model metadata if possible, otherwise
// defaults to true when model type is SPM, otherwise false.
bool llama_should_add_bos_token(const llama_model * model);

//
// Chat template utils
//

// same with llama_chat_message, but uses std::string
struct llama_chat_msg {
    std::string role;
    std::string content;
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool llama_chat_verify_template(const std::string & tmpl);

// CPP wrapper for llama_chat_apply_template
// If the built-in template is not supported, we default to chatml
// If the custom "tmpl" is not supported, we throw an error
std::string llama_chat_apply_template(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & chat,
        bool add_ass);

// Format single message, while taking into account the position of that message in chat history
std::string llama_chat_format_single(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & past_msg,
        const llama_chat_msg & new_msg,
        bool add_ass);

// Returns an example of formatted chat
std::string llama_chat_format_example(const struct llama_model * model,
        const std::string & tmpl);

//
// KV cache utils
//

// Dump the KV cache view with the number of sequences per cell.
void llama_kv_cache_dump_view(const llama_kv_cache_view & view, int row_size = 80);

// Dump the KV cache view showing individual sequences in each cell (long output).
void llama_kv_cache_dump_view_seqs(const llama_kv_cache_view & view, int row_size = 40);

//
// Embedding utils
//

void llama_embd_normalize(const float * inp, float * out, int n, int embd_norm = 2);

float llama_embd_similarity_cos(const float * embd1, const float * embd2, int n);

//
// Control vector utils
//

struct llama_control_vector_data {
    int n_embd;

    // stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
    std::vector<float> data;
};

struct llama_control_vector_load_info {
    float strength;

    std::string fname;
};

// Load control vectors, scale each by strength, and add them together.
// On error, returns {-1, empty}
llama_control_vector_data llama_control_vector_load(const std::vector<llama_control_vector_load_info> & load_infos);

//
// Split utils
//

static const char * const LLM_KV_SPLIT_NO            = "split.no";
static const char * const LLM_KV_SPLIT_COUNT         = "split.count";
static const char * const LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count";

//
// YAML utils
//

void yaml_dump_vector_float    (FILE * stream, const char * prop_name, const std::vector<float> & data);
void yaml_dump_vector_int      (FILE * stream, const char * prop_name, const std::vector<int> & data);
void yaml_dump_string_multiline(FILE * stream, const char * prop_name, const char * data);

void yaml_dump_non_result_info(
    FILE * stream, const gpt_params & params, const llama_context * lctx,
    const std::string & timestamp, const std::vector<int> & prompt_tokens, const char * model_desc);
