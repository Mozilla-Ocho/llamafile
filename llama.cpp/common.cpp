// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#if defined(_MSC_VER)
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "common.h"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "llamafile/debug.h"
#include "string.h"

#include <cosmo.h>
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(LLAMA_USE_CURL)
#include <curl/curl.h>
#include <curl/easy.h>
#include <thread>
#include <future>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if (defined(GGML_USE_CUDA) || defined(GGML_USE_SYCL))
#define GGML_USE_CUDA_SYCL
#endif

#if (defined(GGML_USE_CUDA) || defined(GGML_USE_SYCL)) || defined(GGML_USE_VULKAN)
#define GGML_USE_CUDA_SYCL_VULKAN
#endif

#if defined(LLAMA_USE_CURL)
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#define PATH_MAX MAX_PATH
#else
#include <sys/syslimits.h>
#endif
#define LLAMA_CURL_MAX_URL_LENGTH 2084 // Maximum URL Length in Chrome: 2083
#endif // LLAMA_USE_CURL

// [jart] cuda must not self-init until after flags are parsed
#define llama_supports_gpu_offload() 1

using json = nlohmann::ordered_json;

//
// CPU utils
//

int32_t cpu_get_num_physical_cores();

//
// CLI argument parsing
//

void gpt_params_handle_hf_token(gpt_params & params) {
    if (params.hf_token.empty() && std::getenv("HF_TOKEN")) {
        params.hf_token = std::getenv("HF_TOKEN");
    }
}

void gpt_params_handle_model_default(gpt_params & params) {
    if (!params.hf_repo.empty()) {
        // short-hand to avoid specifying --hf-file -> default it to --model
        if (params.hf_file.empty()) {
            if (params.model.empty()) {
                throw std::invalid_argument("error: --hf-repo requires either --hf-file or --model\n");
            }
            params.hf_file = params.model;
        } else if (params.model.empty()) {
            params.model = fs_get_cache_file(string_split(params.hf_file, '/').back());
        }
    } else if (!params.model_url.empty()) {
        if (params.model.empty()) {
            auto f = string_split(params.model_url, '#').front();
            f = string_split(f, '?').front();
            params.model = fs_get_cache_file(string_split(f, '/').back());
        }
    } else if (params.model.empty()) {
        params.model = DEFAULT_MODEL_PATH;
    }
}

bool gpt_params_parse_ex(int argc, char ** argv, gpt_params & params) {
    bool invalid_param = false;
    std::string arg;
    const std::string arg_prefix = "--";
    llama_sampling_params & sparams = params.sparams;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (!gpt_params_find_arg(argc, argv, arg, params, i, invalid_param)) {
            throw std::invalid_argument("error: unknown argument: " + arg);
        }
        if (invalid_param) {
            throw std::invalid_argument("error: invalid parameter for argument: " + arg);
        }
    }

    if (params.prompt_cache_all && (params.interactive || params.interactive_first)) {
        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    gpt_params_handle_model_default(params);

    gpt_params_handle_hf_token(params);

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        string_process_escapes(sparams.cfg_negative_prompt);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    FLAGS_READY = true;
    params.n_gpu_layers = llamafile_gpu_layers(params.n_gpu_layers);

    return true;
}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    const auto params_org = params; // the example can modify the default params

    try {
        if (!gpt_params_parse_ex(argc, argv, params) || params.usage) {
            params = params_org;
            params.usage = true;
            return false;
        }
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        params = params_org;
        return false;
    }

    return true;
}

#define CHECK_ARG if (++i >= argc) { invalid_param = true; return true; }

bool gpt_params_find_arg(int argc, char ** argv, const std::string & arg, gpt_params & params, int & i, bool & invalid_param) {
    const char split_delim = ',';

    llama_sampling_params & sparams = params.sparams;

    if (arg == "--cli") {
        return true;
    }
    if (arg == "--chat") {
        return true;
    }
    if (arg == "--server") {
        return true;
    }
    if (arg == "--trace") {
        FLAG_trace = true;
        FLAG_unsecure = true;
        return true;
    }
    if (arg == "--fast") {
        FLAG_fast = true;
        return true;
    }
    if (arg == "--iq") {
        FLAG_iq = true;
        return true;
    }
    if (arg == "--ascii") {
        FLAG_ascii = true;
        return true;
    }
    if (arg == "--precise") {
        FLAG_precise = true;
        return true;
    }
    if (arg == "--trap") {
        FLAG_trap = true;
        FLAG_unsecure = true; // for better backtraces
        llamafile_trapping_enabled(+1);
        return true;
    }
    if (arg == "--unsecure") {
        FLAG_unsecure = true;
        return true;
    }
    if (arg == "--nocompile") {
        FLAG_nocompile = true;
        return true;
    }
    if (arg == "--recompile") {
        FLAG_recompile = true;
        return true;
    }
    if (arg == "--tinyblas") {
        FLAG_tinyblas = true;  // undocumented
        return true;
    }
    if (arg == "--gpu") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        FLAG_gpu = llamafile_gpu_parse(argv[i]);
        if (FLAG_gpu == LLAMAFILE_GPU_ERROR) {
            fprintf(stderr, "error: invalid --gpu flag value: %s\n", argv[i]);
            exit(1);
        }
        return true;
    }

    if (arg == "-s" || arg == "--seed") {
        CHECK_ARG
        // TODO: this is temporary, in the future the sampling state will be moved fully to llama_sampling_context.
        params.seed = std::stoul(argv[i]);
        sparams.seed = std::stoul(argv[i]);
        return true;
    }
    if (arg == "-t" || arg == "--threads") {
        CHECK_ARG
        params.n_threads = std::stoi(argv[i]);
        if (params.n_threads <= 0) {
            params.n_threads = std::thread::hardware_concurrency();
        }
        FLAG_threads = params.n_threads; // [jart]
        return true;
    }
    if (arg == "-tb" || arg == "--threads-batch") {
        CHECK_ARG
        params.n_threads_batch = std::stoi(argv[i]);
        if (params.n_threads_batch <= 0) {
            params.n_threads_batch = std::thread::hardware_concurrency();
        }
        return true;
    }
    if (arg == "-td" || arg == "--threads-draft") {
        CHECK_ARG
        params.n_threads_draft = std::stoi(argv[i]);
        if (params.n_threads_draft <= 0) {
            params.n_threads_draft = std::thread::hardware_concurrency();
        }
        return true;
    }
    if (arg == "-tbd" || arg == "--threads-batch-draft") {
        CHECK_ARG
        params.n_threads_batch_draft = std::stoi(argv[i]);
        if (params.n_threads_batch_draft <= 0) {
            params.n_threads_batch_draft = std::thread::hardware_concurrency();
        }
        return true;
    }
    if (arg == "-p" || arg == "--prompt") {
        CHECK_ARG
        params.prompt = argv[i];
        return true;
    }
    if (arg == "-e" || arg == "--escape") {
        params.escape = true;
        return true;
    }
    if (arg == "--no-escape") {
        params.escape = false;
        return true;
    }
    if (arg == "--prompt-cache") {
        CHECK_ARG
        params.path_prompt_cache = argv[i];
        return true;
    }
    if (arg == "--prompt-cache-all") {
        params.prompt_cache_all = true;
        return true;
    }
    if (arg == "--prompt-cache-ro") {
        params.prompt_cache_ro = true;
        return true;
    }
    if (arg == "-bf" || arg == "--binary-file") {
        CHECK_ARG
        std::ifstream file(argv[i], std::ios::binary);
        if (!file) {
            fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
            invalid_param = true;
            return true;
        }
        // store the external file name in params
        params.prompt_file = argv[i];
        std::ostringstream ss;
        ss << file.rdbuf();
        params.prompt = ss.str();
        fprintf(stderr, "Read %zu bytes from binary file %s\n", params.prompt.size(), argv[i]);
        return true;
    }
    if (arg == "-f" || arg == "--file") {
        CHECK_ARG
        std::ifstream file(argv[i]);
        if (!file) {
            fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
            invalid_param = true;
            return true;
        }
        // store the external file name in params
        params.prompt_file = argv[i];
        std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
        if (!params.prompt.empty() && params.prompt.back() == '\n') {
            params.prompt.pop_back();
        }
        return true;
    }
    if (arg == "--in-file") {
        CHECK_ARG
        std::ifstream file(argv[i]);
        if (!file) {
            fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
            invalid_param = true;
            return true;
        }
        params.in_files.push_back(argv[i]);
        return true;
    }
    if (arg == "-n" || arg == "--predict" || arg == "--n-predict") {
        CHECK_ARG
        params.n_predict = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--top-k") {
        CHECK_ARG
        sparams.top_k = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-c" || arg == "--ctx-size") {
        CHECK_ARG
        params.n_ctx = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--grp-attn-n" || arg == "-gan") {
        CHECK_ARG
        params.grp_attn_n = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--grp-attn-w" || arg == "-gaw") {
        CHECK_ARG
        params.grp_attn_w = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--rope-freq-base") {
        CHECK_ARG
        params.rope_freq_base = std::stof(argv[i]);
        return true;
    }
    if (arg == "--rope-freq-scale") {
        CHECK_ARG
        params.rope_freq_scale = std::stof(argv[i]);
        return true;
    }
    if (arg == "--rope-scaling") {
        CHECK_ARG
        std::string value(argv[i]);
        /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
        else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
        else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
        else { invalid_param = true; }
        return true;
    }
    if (arg == "--rope-scale") {
        CHECK_ARG
        params.rope_freq_scale = 1.0f / std::stof(argv[i]);
        return true;
    }
    if (arg == "--yarn-orig-ctx") {
        CHECK_ARG
        params.yarn_orig_ctx = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--yarn-ext-factor") {
        CHECK_ARG
        params.yarn_ext_factor = std::stof(argv[i]);
        return true;
    }
    if (arg == "--yarn-attn-factor") {
        CHECK_ARG
        params.yarn_attn_factor = std::stof(argv[i]);
        return true;
    }
    if (arg == "--yarn-beta-fast") {
        CHECK_ARG
        params.yarn_beta_fast = std::stof(argv[i]);
        return true;
    }
    if (arg == "--yarn-beta-slow") {
        CHECK_ARG
        params.yarn_beta_slow = std::stof(argv[i]);
        return true;
    }
    if (arg == "--pooling") {
        CHECK_ARG
        std::string value(argv[i]);
        /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
        else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
        else if (value == "cls") { params.pooling_type = LLAMA_POOLING_TYPE_CLS; }
        else if (value == "last") { params.pooling_type = LLAMA_POOLING_TYPE_LAST; }
        else { invalid_param = true; }
        return true;
    }
    if (arg == "--attention") {
        CHECK_ARG
        std::string value(argv[i]);
        /**/ if (value == "causal") { params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; }
        else if (value == "non-causal") { params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; }
        else { invalid_param = true; }
        return true;
    }
    if (arg == "--defrag-thold" || arg == "-dt") {
        CHECK_ARG
        params.defrag_thold = std::stof(argv[i]);
        return true;
    }
    if (arg == "--samplers") {
        CHECK_ARG
        const auto sampler_names = string_split(argv[i], ';');
        sparams.samplers_sequence = llama_sampling_types_from_names(sampler_names, true);
        return true;
    }
    if (arg == "--sampling-seq") {
        CHECK_ARG
        sparams.samplers_sequence = llama_sampling_types_from_chars(argv[i]);
        return true;
    }
    if (arg == "--top-p") {
        CHECK_ARG
        sparams.top_p = std::stof(argv[i]);
        return true;
    }
    if (arg == "--min-p") {
        CHECK_ARG
        sparams.min_p = std::stof(argv[i]);
        return true;
    }
    if (arg == "--temp") {
        CHECK_ARG
        sparams.temp = std::stof(argv[i]);
        sparams.temp = std::max(sparams.temp, 0.0f);
        return true;
    }
    if (arg == "--tfs") {
        CHECK_ARG
        sparams.tfs_z = std::stof(argv[i]);
        return true;
    }
    if (arg == "--typical") {
        CHECK_ARG
        sparams.typical_p = std::stof(argv[i]);
        return true;
    }
    if (arg == "--repeat-last-n") {
        CHECK_ARG
        sparams.penalty_last_n = std::stoi(argv[i]);
        sparams.n_prev = std::max(sparams.n_prev, sparams.penalty_last_n);
        return true;
    }
    if (arg == "--repeat-penalty") {
        CHECK_ARG
        sparams.penalty_repeat = std::stof(argv[i]);
        return true;
    }
    if (arg == "--frequency-penalty") {
        CHECK_ARG
        sparams.penalty_freq = std::stof(argv[i]);
        return true;
    }
    if (arg == "--presence-penalty") {
        CHECK_ARG
        sparams.penalty_present = std::stof(argv[i]);
        return true;
    }
    if (arg == "--dynatemp-range") {
        CHECK_ARG
        sparams.dynatemp_range = std::stof(argv[i]);
        return true;
    }
    if (arg == "--dynatemp-exp") {
        CHECK_ARG
        sparams.dynatemp_exponent = std::stof(argv[i]);
        return true;
    }
    if (arg == "--mirostat") {
        CHECK_ARG
        sparams.mirostat = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--mirostat-lr") {
        CHECK_ARG
        sparams.mirostat_eta = std::stof(argv[i]);
        return true;
    }
    if (arg == "--mirostat-ent") {
        CHECK_ARG
        sparams.mirostat_tau = std::stof(argv[i]);
        return true;
    }
    if (arg == "--cfg-negative-prompt") {
        CHECK_ARG
        sparams.cfg_negative_prompt = argv[i];
        return true;
    }
    if (arg == "--cfg-negative-prompt-file") {
        CHECK_ARG
        std::ifstream file(argv[i]);
        if (!file) {
            fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
            invalid_param = true;
            return true;
        }
        std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(sparams.cfg_negative_prompt));
        if (!sparams.cfg_negative_prompt.empty() && sparams.cfg_negative_prompt.back() == '\n') {
            sparams.cfg_negative_prompt.pop_back();
        }
        return true;
    }
    if (arg == "--cfg-scale") {
        CHECK_ARG
        sparams.cfg_scale = std::stof(argv[i]);
        return true;
    }
    if (arg == "-b" || arg == "--batch-size") {
        CHECK_ARG
        params.n_batch = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-ub" || arg == "--ubatch-size") {
        CHECK_ARG
        params.n_ubatch = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--keep") {
        CHECK_ARG
        params.n_keep = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--draft") {
        CHECK_ARG
        params.n_draft = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--chunks") {
        CHECK_ARG
        params.n_chunks = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-np" || arg == "--parallel") {
        CHECK_ARG
        params.n_parallel = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-ns" || arg == "--sequences") {
        CHECK_ARG
        params.n_sequences = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--p-split" || arg == "-ps") {
        CHECK_ARG
        params.p_split = std::stof(argv[i]);
        return true;
    }
    if (arg == "-m" || arg == "--model") {
        CHECK_ARG
        params.model = argv[i];
        return true;
    }
    if (arg == "-md" || arg == "--model-draft") {
        CHECK_ARG
        params.model_draft = argv[i];
        return true;
    }
    if (arg == "-a" || arg == "--alias") {
        CHECK_ARG
        params.model_alias = argv[i];
        return true;
    }
    if (arg == "-mu" || arg == "--model-url") {
        CHECK_ARG
        params.model_url = argv[i];
        return true;
    }
    if (arg == "-hft" || arg == "--hf-token") {
        if (++i >= argc) {
          invalid_param = true;
          return true;
        }
        params.hf_token = argv[i];
        return true;
    }
    if (arg == "-hfr" || arg == "--hf-repo") {
        CHECK_ARG
        params.hf_repo = argv[i];
        return true;
    }
    if (arg == "-hff" || arg == "--hf-file") {
        CHECK_ARG
        params.hf_file = argv[i];
        return true;
    }
    if (arg == "--lora") {
        CHECK_ARG
        params.lora_adapters.push_back({
            std::string(argv[i]),
            1.0,
        });
        return true;
    }
    if (arg == "--lora-scaled") {
        CHECK_ARG
        const char* lora_adapter = argv[i];
        CHECK_ARG
        params.lora_adapters.push_back({
            lora_adapter,
            std::stof(argv[i]),
        });
        return true;
    }
    if (arg == "--control-vector") {
        CHECK_ARG
        params.control_vectors.push_back({ 1.0f, argv[i], });
        return true;
    }
    if (arg == "--control-vector-scaled") {
        CHECK_ARG
        const char* fname = argv[i];
        CHECK_ARG
        params.control_vectors.push_back({ std::stof(argv[i]), fname, });
        return true;
    }
    if (arg == "--control-vector-layer-range") {
        CHECK_ARG
        params.control_vector_layer_start = std::stoi(argv[i]);
        CHECK_ARG
        params.control_vector_layer_end = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--mmproj") {
        CHECK_ARG
        params.mmproj = argv[i];
        return true;
    }
    if (arg == "--image") {
        CHECK_ARG
        params.image.emplace_back(argv[i]);
        return true;
    }
    if (arg == "-i" || arg == "--interactive") {
        params.interactive = true;
        return true;
    }
    if (arg == "-sp" || arg == "--special") {
        params.special = true;
        return true;
    }
    if (arg == "--embedding" || arg == "--embeddings") {
        params.embedding = true;
        return true;
    }
    if (arg == "--embd-normalize") {
        CHECK_ARG
        params.embd_normalize = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--embd-output-format") {
        CHECK_ARG
        params.embd_out = argv[i];
        return true;
    }
    if (arg == "--embd-separator") {
        CHECK_ARG
        params.embd_sep = argv[i];
        return true;
    }
    if (arg == "-if" || arg == "--interactive-first") {
        params.interactive_first = true;
        return true;
    }
    if (arg == "-cnv" || arg == "--conversation") {
        params.conversation = true;
        return true;
    }
    if (arg == "--infill") {
        params.infill = true;
        return true;
    }
    if (arg == "-dkvc" || arg == "--dump-kv-cache") {
        params.dump_kv_cache = true;
        return true;
    }
    if (arg == "-nkvo" || arg == "--no-kv-offload") {
        params.no_kv_offload = true;
        return true;
    }
    if (arg == "-ctk" || arg == "--cache-type-k") {
        params.cache_type_k = argv[++i];
        return true;
    }
    if (arg == "-ctv" || arg == "--cache-type-v") {
        params.cache_type_v = argv[++i];
        return true;
    }
    if (arg == "-mli" || arg == "--multiline-input") {
        params.multiline_input = true;
        return true;
    }
    if (arg == "--simple-io") {
        params.simple_io = true;
        return true;
    }
    if (arg == "-cb" || arg == "--cont-batching") {
        params.cont_batching = true;
        return true;
    }
    if (arg == "-nocb" || arg == "--no-cont-batching") {
        params.cont_batching = false;
        return true;
    }
    if (arg == "-fa" || arg == "--flash-attn") {
        params.flash_attn = true;
        FLAG_flash_attn = true; // [jart]
        return true;
    }
    if (arg == "-co" || arg == "--color") {
        params.use_color = true;
        return true;
    }
    if (arg == "--mlock") {
        params.use_mlock = true;
        return true;
    }
    if (arg == "-ngl" || arg == "--gpu-layers" || arg == "--n-gpu-layers") {
        CHECK_ARG
        params.n_gpu_layers = std::stoi(argv[i]);
        if (params.n_gpu_layers <= 0)
            FLAG_gpu = LLAMAFILE_GPU_DISABLE;
        if (!llama_supports_gpu_offload()) {
            fprintf(stderr, "warning: not compiled with GPU offload support, --gpu-layers option will be ignored\n");
            fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
        }
        return true;
    }
    if (arg == "-ngld" || arg == "--gpu-layers-draft" || arg == "--gpu-layers-draft") {
        CHECK_ARG
        params.n_gpu_layers_draft = std::stoi(argv[i]);
        if (!llama_supports_gpu_offload()) {
            fprintf(stderr, "warning: not compiled with GPU offload support, --gpu-layers-draft option will be ignored\n");
            fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
        }
        return true;
    }
    if (arg == "--main-gpu" || arg == "-mg") {
        CHECK_ARG
        params.main_gpu = std::stoi(argv[i]);
// #ifndef GGML_USE_CUDA_SYCL_VULKAN // [jart]
//         fprintf(stderr, "warning: llama.cpp was compiled without CUDA/SYCL/Vulkan. Setting the main GPU has no effect.\n");
// #endif // GGML_USE_CUDA_SYCL_VULKAN
        return true;
    }
    if (arg == "--split-mode" || arg == "-sm") {
        CHECK_ARG
        std::string arg_next = argv[i];
        if (arg_next == "none") {
            params.split_mode = LLAMA_SPLIT_MODE_NONE;
        }
        else if (arg_next == "layer") {
            params.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }
        else if (arg_next == "row") {
#ifdef GGML_USE_SYCL
            fprintf(stderr, "warning: The split mode value:[row] is not supported by llama.cpp with SYCL. It's developing.\nExit!\n");
            exit(1);
#endif // GGML_USE_SYCL
            params.split_mode = LLAMA_SPLIT_MODE_ROW;
        }
        else {
            invalid_param = true;
            return true;
        }
// #ifndef GGML_USE_CUDA_SYCL_VULKAN // [jart]
//         fprintf(stderr, "warning: llama.cpp was compiled without CUDA/SYCL/Vulkan. Setting the split mode has no effect.\n");
// #endif // GGML_USE_CUDA_SYCL_VULKAN
        return true;
    }
    if (arg == "--tensor-split" || arg == "-ts") {
        CHECK_ARG
        std::string arg_next = argv[i];

        // split string by , and /
        const std::regex regex{ R"([,/]+)" };
        std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
        std::vector<std::string> split_arg{ it, {} };
        if (split_arg.size() >= llama_max_devices()) {
            invalid_param = true;
            return true;
        }
        for (size_t i = 0; i < llama_max_devices(); ++i) {
            if (i < split_arg.size()) {
                params.tensor_split[i] = std::stof(split_arg[i]);
            }
            else {
                params.tensor_split[i] = 0.0f;
            }
        }
// #ifndef GGML_USE_CUDA_SYCL_VULKAN // [jart]
//         fprintf(stderr, "warning: llama.cpp was compiled without CUDA/SYCL/Vulkan. Setting a tensor split has no effect.\n");
// #endif // GGML_USE_CUDA_SYCL_VULKAN
        return true;
    }
    if (arg == "--rpc") {
        CHECK_ARG
        params.rpc_servers = argv[i];
        return true;
    }
    if (arg == "--no-mmap") {
        params.use_mmap = false;
        return true;
    }
    if (arg == "--numa") {
        CHECK_ARG
        std::string value(argv[i]);
        /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
        else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
        else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
        else { invalid_param = true; }
        return true;
    }
    if (arg == "-v" || arg == "--verbose") {
        params.verbosity = 1;
        return true;
    }
    if (arg == "--verbosity") {
        CHECK_ARG
        params.verbosity = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--verbose-prompt") {
        params.verbose_prompt = true;
        return true;
    }
    if (arg == "--no-display-prompt" || arg == "--silent-prompt") {
        params.display_prompt = false;
        return true;
    }
    if (arg == "-r" || arg == "--reverse-prompt") {
        CHECK_ARG
        params.antiprompt.emplace_back(argv[i]);
        return true;
    }
    if (arg == "-ld" || arg == "--logdir") {
        CHECK_ARG
        params.logdir = argv[i];

        if (params.logdir.back() != DIRECTORY_SEPARATOR) {
            params.logdir += DIRECTORY_SEPARATOR;
        }
        return true;
    }
    if (arg == "-lcs" || arg == "--lookup-cache-static") {
        CHECK_ARG
        params.lookup_cache_static = argv[i];
        return true;
    }
    if (arg == "-lcd" || arg == "--lookup-cache-dynamic") {
        CHECK_ARG
        params.lookup_cache_dynamic = argv[i];
        return true;
    }
    if (arg == "--save-all-logits" || arg == "--kl-divergence-base") {
        CHECK_ARG
        params.logits_file = argv[i];
        return true;
    }
    if (arg == "--perplexity" || arg == "--all-logits") {
        params.logits_all = true;
        return true;
    }
    if (arg == "--ppl-stride") {
        CHECK_ARG
        params.ppl_stride = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--ppl-output-type") {
        CHECK_ARG
        params.ppl_output_type = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-ptc" || arg == "--print-token-count") {
        CHECK_ARG
        params.n_print = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--check-tensors") {
        params.check_tensors = true;
        return true;
    }
    if (arg == "--hellaswag") {
        params.hellaswag = true;
        return true;
    }
    if (arg == "--hellaswag-tasks") {
        CHECK_ARG
        params.hellaswag_tasks = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--winogrande") {
        params.winogrande = true;
        return true;
    }
    if (arg == "--winogrande-tasks") {
        CHECK_ARG
        params.winogrande_tasks = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--multiple-choice") {
        params.multiple_choice = true;
        return true;
    }
    if (arg == "--multiple-choice-tasks") {
        CHECK_ARG
        params.multiple_choice_tasks = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--kl-divergence") {
        params.kl_divergence = true;
        return true;
    }
    if (arg == "--ignore-eos") {
        params.ignore_eos = true;
        return true;
    }
    if (arg == "--penalize-nl") {
        sparams.penalize_nl = true;
        return true;
    }
    if (arg == "-l" || arg == "--logit-bias") {
        CHECK_ARG
        std::stringstream ss(argv[i]);
        llama_token key;
        char sign;
        std::string value_str;
        try {
            if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                sparams.logit_bias[key] = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
            }
            else {
                throw std::exception();
            }
        }
        catch (const std::exception&) {
            invalid_param = true;
            return true;
        }
        return true;
    }
    if (arg == "-h" || arg == "--help" || arg == "--usage"  ) {
        params.usage = true;
        return true;
    }
    if (arg == "--version") {
        fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
        fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
        exit(0);
    }
    if (arg == "--in-prefix-bos") {
        params.input_prefix_bos = true;
        params.enable_chat_template = false;
        return true;
    }
    if (arg == "--in-prefix") {
        CHECK_ARG
        params.input_prefix = argv[i];
        params.enable_chat_template = false;
        return true;
    }
    if (arg == "--in-suffix") {
        CHECK_ARG
        params.input_suffix = argv[i];
        params.enable_chat_template = false;
        return true;
    }
    if (arg == "--spm-infill") {
        params.spm_infill = true;
        return true;
    }
    if (arg == "--grammar") {
        CHECK_ARG
        sparams.grammar = argv[i];
        return true;
    }
    if (arg == "--grammar-file") {
        CHECK_ARG
        std::ifstream file(argv[i]);
        if (!file) {
            fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
            invalid_param = true;
            return true;
        }
        std::copy(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>(),
            std::back_inserter(sparams.grammar)
        );
        return true;
    }
    if (arg == "-j" || arg == "--json-schema") {
        CHECK_ARG
        sparams.grammar = json_schema_to_grammar(json::parse(argv[i]));
        return true;
    }
    if (arg == "--override-kv") {
        CHECK_ARG
        if (!string_parse_kv_override(argv[i], params.kv_overrides)) {
            fprintf(stderr, "error: Invalid type for KV override: %s\n", argv[i]);
            invalid_param = true;
            return true;
        }
        return true;
    }
    if (arg == "--host") {
        CHECK_ARG
        params.hostname = argv[i];
        return true;
    }
    if (arg == "--port") {
        CHECK_ARG
        params.port = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--path") {
        CHECK_ARG
        params.public_path = argv[i];
        return true;
    }
    if (arg == "--url-prefix") {
        CHECK_ARG
        params.url_prefix = argv[i];
        return true;
    }
    if (arg == "--api-key") {
        CHECK_ARG
        params.api_keys.push_back(argv[i]);
        return true;
    }
    if (arg == "--api-key-file") {
        CHECK_ARG
        std::ifstream key_file(argv[i]);
        if (!key_file) {
            fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
            invalid_param = true;
            return true;
        }
        std::string key;
        while (std::getline(key_file, key)) {
            if (!key.empty()) {
                params.api_keys.push_back(key);
            }
        }
        key_file.close();
        return true;
    }
    if (arg == "--ssl-key-file") {
        CHECK_ARG
        params.ssl_file_key = argv[i];
        return true;
    }
    if (arg == "--ssl-cert-file") {
        CHECK_ARG
        params.ssl_file_cert = argv[i];
        return true;
    }
    if (arg == "--timeout" || arg == "-to") {
        CHECK_ARG
        params.timeout_read  = std::stoi(argv[i]);
        params.timeout_write = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--threads-http") {
        CHECK_ARG
        params.n_threads_http = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-spf" || arg == "--system-prompt-file") {
        CHECK_ARG
        std::ifstream file(argv[i]);
        if (!file) {
            fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
            invalid_param = true;
            return true;
        }
        std::string system_prompt;
        std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(system_prompt)
                );
        params.system_prompt = system_prompt;
        return true;
    }
    if (arg == "--log-format") {
        CHECK_ARG
        if (std::strcmp(argv[i], "json") == 0) {
            params.log_json = true;
        } else if (std::strcmp(argv[i], "text") == 0) {
            params.log_json = false;
        } else {
            invalid_param = true;
            return true;
        }
        return true;
    }
    if (arg == "--no-slots") {
        params.endpoint_slots = false;
        return true;
    }
    if (arg == "--metrics") {
        params.endpoint_metrics = true;
        return true;
    }
    if (arg == "--slot-save-path") {
        CHECK_ARG
        params.slot_save_path = argv[i];
        // if doesn't end with DIRECTORY_SEPARATOR, add it
        if (!params.slot_save_path.empty() && params.slot_save_path[params.slot_save_path.size() - 1] != DIRECTORY_SEPARATOR) {
            params.slot_save_path += DIRECTORY_SEPARATOR;
        }
        return true;
    }
    if (arg == "--chat-template") {
        CHECK_ARG
        if (!llama_chat_verify_template(argv[i])) {
            fprintf(stderr, "error: the supplied chat template is not supported: %s\n", argv[i]);
            fprintf(stderr, "note: llama.cpp does not use jinja parser, we only support commonly used templates\n");
            invalid_param = true;
            return true;
        }
        params.chat_template = argv[i];
        return true;
    }
    if (arg == "--slot-prompt-similarity" || arg == "-sps") {
        CHECK_ARG
        params.slot_prompt_similarity = std::stof(argv[i]);
        return true;
    }
    if (arg == "-pps") {
        params.is_pp_shared = true;
        return true;
    }
    if (arg == "-npp") {
        CHECK_ARG
        auto p = string_split<int>(argv[i], split_delim);
        params.n_pp.insert(params.n_pp.end(), p.begin(), p.end());
        return true;
    }
    if (arg == "-ntg") {
        CHECK_ARG
        auto p = string_split<int>(argv[i], split_delim);
        params.n_tg.insert(params.n_tg.end(), p.begin(), p.end());
        return true;
    }
    if (arg == "-npl") {
        CHECK_ARG
        auto p = string_split<int>(argv[i], split_delim);
        params.n_pl.insert(params.n_pl.end(), p.begin(), p.end());
        return true;
    }
    if (arg == "--context-file") {
        CHECK_ARG
        std::ifstream file(argv[i], std::ios::binary);
        if (!file) {
            fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
            invalid_param = true;
            return true;
        }
        params.context_files.push_back(argv[i]);
        return true;
    }
    if (arg == "--chunk-size") {
        CHECK_ARG
        params.chunk_size = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--chunk-separator") {
        CHECK_ARG
        params.chunk_separator = argv[i];
        return true;
    }
    if (arg == "--junk") {
        CHECK_ARG
        params.n_junk = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--pos") {
        CHECK_ARG
        params.i_pos = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-o" || arg == "--output" || arg == "--output-file") {
        CHECK_ARG
        params.out_file = argv[i];
        params.cvector_outfile = argv[i];
        params.lora_outfile = argv[i];
        return true;
    }
    if (arg == "-ofreq" || arg == "--output-frequency") {
        CHECK_ARG
        params.n_out_freq = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--save-frequency") {
        CHECK_ARG
        params.n_save_freq = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--process-output") {
        params.process_output = true;
        return true;
    }
    if (arg == "--no-ppl") {
        params.compute_ppl = false;
        return true;
    }
    if (arg == "--chunk" || arg == "--from-chunk") {
        CHECK_ARG
        params.i_chunk = std::stoi(argv[i]);
        return true;
    }
    // cvector params
    if (arg == "--positive-file") {
        CHECK_ARG
        params.cvector_positive_file = argv[i];
        return true;
    }
    if (arg == "--negative-file") {
        CHECK_ARG
        params.cvector_negative_file = argv[i];
        return true;
    }
    if (arg == "--pca-batch") {
        CHECK_ARG
        params.n_pca_batch = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--pca-iter") {
        CHECK_ARG
        params.n_pca_iterations = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--method") {
        CHECK_ARG
        std::string value(argv[i]);
        /**/ if (value == "pca") { params.cvector_dimre_method = DIMRE_METHOD_PCA; }
        else if (value == "mean") { params.cvector_dimre_method = DIMRE_METHOD_MEAN; }
        else { invalid_param = true; }
        return true;
    }
    if (arg == "--no-warmup") {
        params.warmup = false;
        return true;
    }
#ifndef LOG_DISABLE_LOGS
    // Parse args for logging parameters
    if (log_param_single_parse(argv[i])) {
        // Do nothing, log_param_single_parse automatically does it's thing
        //  and returns if a match was found and parsed.
        return true;
    }
    if (log_param_pair_parse( /*check_but_dont_parse*/ true, argv[i])) {
        // We have a matching known parameter requiring an argument,
        //  now we need to check if there is anything after this argv
        //  and flag invalid_param or parse it.
        CHECK_ARG
        if (!log_param_pair_parse( /*check_but_dont_parse*/ false, argv[i - 1], argv[i])) {
            invalid_param = true;
            return true;
        }
        return true;
    }
    // End of Parse args for logging parameters
#endif // LOG_DISABLE_LOGS

    return false;
}

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

void gpt_params_print_usage(int /*argc*/, char ** argv, const gpt_params & params) {
    const llama_sampling_params & sparams = params.sparams;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto sampler_type : sparams.samplers_sequence) {
        sampler_type_chars += static_cast<char>(sampler_type);
        sampler_type_names += llama_sampling_type_to_str(sampler_type) + ";";
    }
    sampler_type_names.pop_back();

    struct option_info {
        LLAMA_COMMON_ATTRIBUTE_FORMAT(4, 5)
        option_info(const std::string & tags, const char * args, const char * desc, ...) : tags(tags), args(args), desc(desc) {
            va_list args_list;
            va_start(args_list, desc);
            char buffer[1024];
            vsnprintf(buffer, sizeof(buffer), desc, args_list);
            va_end(args_list);
            this->desc = buffer;
        }

        option_info(const std::string & grp) : grp(grp) {}

        std::string tags;
        std::string args;
        std::string desc;
        std::string grp;
    };

    std::vector<option_info> options;

    // TODO: filter by tags

    options.push_back({ "general" });
    options.push_back({ "*",           "-h,    --help, --usage",        "print usage and exit" });
    options.push_back({ "*",           "       --version",              "show version and build info" });
    options.push_back({ "*",           "-v,    --verbose",              "print verbose information" });
    options.push_back({ "*",           "       --verbosity N",          "set specific verbosity level (default: %d)", params.verbosity });
    options.push_back({ "*",           "       --verbose-prompt",       "print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false" });
    options.push_back({ "*",           "       --no-display-prompt",    "don't print prompt at generation (default: %s)", !params.display_prompt ? "true" : "false" });
    options.push_back({ "*",           "-co,   --color",                "colorise output to distinguish prompt and user input from generations (default: %s)", params.use_color ? "true" : "false" });
    options.push_back({ "*",           "-s,    --seed SEED",            "RNG seed (default: %d, use random seed for < 0)", params.seed });
    options.push_back({ "*",           "-t,    --threads N",            "number of threads to use during generation (default: %d)", params.n_threads });
    options.push_back({ "*",           "-tb,   --threads-batch N",      "number of threads to use during batch and prompt processing (default: same as --threads)" });
    options.push_back({ "speculative", "-td,   --threads-draft N",      "number of threads to use during generation (default: same as --threads)" });
    options.push_back({ "speculative", "-tbd,  --threads-batch-draft N",
                                                                        "number of threads to use during batch and prompt processing (default: same as --threads-draft)" });
    options.push_back({ "speculative", "       --draft N",              "number of tokens to draft for speculative decoding (default: %d)", params.n_draft });
    options.push_back({ "speculative", "-ps,   --p-split N",            "speculative decoding split probability (default: %.1f)", (double)params.p_split });
    options.push_back({ "*",           "-lcs,  --lookup-cache-static FNAME",
                                                                        "path to static lookup cache to use for lookup decoding (not updated by generation)" });
    options.push_back({ "*",           "-lcd,  --lookup-cache-dynamic FNAME",
                                                                        "path to dynamic lookup cache to use for lookup decoding (updated by generation)" });

    options.push_back({ "*",           "-c,    --ctx-size N",           "size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx });
    options.push_back({ "*",           "-n,    --predict N",            "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)", params.n_predict });
    options.push_back({ "*",           "-b,    --batch-size N",         "logical maximum batch size (default: %d)", params.n_batch });
    options.push_back({ "*",           "-ub,   --ubatch-size N",        "physical maximum batch size (default: %d)", params.n_ubatch });
    options.push_back({ "*",           "       --keep N",               "number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep });
    options.push_back({ "*",           "       --chunks N",             "max number of chunks to process (default: %d, -1 = all)", params.n_chunks });
    options.push_back({ "*",           "-fa,   --flash-attn",           "enable Flash Attention (default: %s)", params.flash_attn ? "enabled" : "disabled" });
    options.push_back({ "*",           "-p,    --prompt PROMPT",        "prompt to start generation with\n"
                                                                        "in conversation mode, this will be used as system prompt\n"
                                                                        "(default: '%s')", params.prompt.c_str() });
    options.push_back({ "*",           "-f,    --file FNAME",           "a file containing the prompt (default: none)" });
    options.push_back({ "*",           "       --in-file FNAME",        "an input file (repeat to specify multiple files)" });
    options.push_back({ "*",           "-bf,   --binary-file FNAME",    "binary file containing the prompt (default: none)" });
    options.push_back({ "*",           "-e,    --escape",               "process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", params.escape ? "true" : "false" });
    options.push_back({ "*",           "       --no-escape",            "do not process escape sequences" });
    options.push_back({ "main",        "-ptc,  --print-token-count N",  "print token count every N tokens (default: %d)", params.n_print });
    options.push_back({ "main",        "       --prompt-cache FNAME",   "file to cache prompt state for faster startup (default: none)" });
    options.push_back({ "main",        "       --prompt-cache-all",     "if specified, saves user input and generations to cache as well\n"
                                                                        "not supported with --interactive or other interactive options" });
    options.push_back({ "main",        "       --prompt-cache-ro",      "if specified, uses the prompt cache but does not update it" });
    options.push_back({ "main",        "-r,    --reverse-prompt PROMPT",
                                                                        "halt generation at PROMPT, return control in interactive mode\n"
                                                                        "can be specified more than once for multiple prompts" });
    options.push_back({ "main",        "-sp,   --special",              "special tokens output enabled (default: %s)", params.special ? "true" : "false" });
    options.push_back({ "main",        "-cnv,  --conversation",         "run in conversation mode, does not print special tokens and suffix/prefix\n"
                                                                        "if suffix/prefix are not specified, default chat template will be used\n"
                                                                        "(default: %s)", params.conversation ? "true" : "false" });
    options.push_back({ "main infill", "-i,    --interactive",          "run in interactive mode (default: %s)", params.interactive ? "true" : "false" });
    options.push_back({ "main infill", "-if,   --interactive-first",    "run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false" });
    options.push_back({ "main infill", "-mli,  --multiline-input",      "allows you to write or paste multiple lines without ending each in '\\'" });
    options.push_back({ "main infill", "       --in-prefix-bos",        "prefix BOS to user inputs, preceding the `--in-prefix` string" });
    options.push_back({ "main infill", "       --in-prefix STRING",     "string to prefix user inputs with (default: empty)" });
    options.push_back({ "main infill", "       --in-suffix STRING",     "string to suffix after user inputs with (default: empty)" });
    options.push_back({ "main",        "       --no-warmup",            "skip warming up the model with an empty run" });
    options.push_back({ "server infill",
                                       "       --spm-infill",           "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)", params.spm_infill ? "enabled" : "disabled" });

    options.push_back({ "sampling" });
    options.push_back({ "*",           "       --samplers SAMPLERS",    "samplers that will be used for generation in the order, separated by \';\'\n"
                                                                        "(default: %s)", sampler_type_names.c_str() });
    options.push_back({ "*",           "       --sampling-seq SEQUENCE",
                                                                        "simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str() });
    options.push_back({ "*",           "       --ignore-eos",           "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)" });
    options.push_back({ "*",           "       --penalize-nl",          "penalize newline tokens (default: %s)", sparams.penalize_nl ? "true" : "false" });
    options.push_back({ "*",           "       --temp N",               "temperature (default: %.1f)", (double)sparams.temp });
    options.push_back({ "*",           "       --top-k N",              "top-k sampling (default: %d, 0 = disabled)", sparams.top_k });
    options.push_back({ "*",           "       --top-p N",              "top-p sampling (default: %.1f, 1.0 = disabled)", (double)sparams.top_p });
    options.push_back({ "*",           "       --min-p N",              "min-p sampling (default: %.1f, 0.0 = disabled)", (double)sparams.min_p });
    options.push_back({ "*",           "       --tfs N",                "tail free sampling, parameter z (default: %.1f, 1.0 = disabled)", (double)sparams.tfs_z });
    options.push_back({ "*",           "       --typical N",            "locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)", (double)sparams.typical_p });
    options.push_back({ "*",           "       --repeat-last-n N",      "last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", sparams.penalty_last_n });
    options.push_back({ "*",           "       --repeat-penalty N",     "penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)", (double)sparams.penalty_repeat });
    options.push_back({ "*",           "       --presence-penalty N",   "repeat alpha presence penalty (default: %.1f, 0.0 = disabled)", (double)sparams.penalty_present });
    options.push_back({ "*",           "       --frequency-penalty N",  "repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)", (double)sparams.penalty_freq });
    options.push_back({ "*",           "       --dynatemp-range N",     "dynamic temperature range (default: %.1f, 0.0 = disabled)", (double)sparams.dynatemp_range });
    options.push_back({ "*",           "       --dynatemp-exp N",       "dynamic temperature exponent (default: %.1f)", (double)sparams.dynatemp_exponent });
    options.push_back({ "*",           "       --mirostat N",           "use Mirostat sampling.\n"
                                                                        "Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n"
                                                                        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", sparams.mirostat });
    options.push_back({ "*",           "       --mirostat-lr N",        "Mirostat learning rate, parameter eta (default: %.1f)", (double)sparams.mirostat_eta });
    options.push_back({ "*",           "       --mirostat-ent N",       "Mirostat target entropy, parameter tau (default: %.1f)", (double)sparams.mirostat_tau });
    options.push_back({ "*",           "       -l TOKEN_ID(+/-)BIAS",   "modifies the likelihood of token appearing in the completion,\n"
                                                                        "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
                                                                        "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'" });
    options.push_back({ "main",        "       --cfg-negative-prompt PROMPT",
                                                                        "negative prompt to use for guidance (default: '%s')", sparams.cfg_negative_prompt.c_str() });
    options.push_back({ "main",        "       --cfg-negative-prompt-file FNAME",
                                                                        "negative prompt file to use for guidance" });
    options.push_back({ "main",        "       --cfg-scale N",          "strength of guidance (default: %.1f, 1.0 = disable)", (double)sparams.cfg_scale });
    options.push_back({ "main",        "       --chat-template JINJA_TEMPLATE",
                                                                        "set custom jinja chat template (default: template taken from model's metadata)\n"
                                                                        "if suffix/prefix are specified, template will be disabled\n"
                                                                        "only commonly used templates are accepted:\n"
                                                                        "https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template" });
    options.push_back({ "grammar" });
    options.push_back({ "*",           "       --grammar GRAMMAR",      "BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '%s')", sparams.grammar.c_str() });
    options.push_back({ "*",           "       --grammar-file FNAME",   "file to read grammar from" });
    options.push_back({ "*",           "-j,    --json-schema SCHEMA",
                                                                        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\n"
                                                                        "For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead" });

    options.push_back({ "embedding" });
    options.push_back({ "embedding",   "       --pooling {none,mean,cls,last}",
                                                                        "pooling type for embeddings, use model default if unspecified" });
    options.push_back({ "embedding",   "       --attention {causal,non-causal}",
                                                                        "attention type for embeddings, use model default if unspecified" });

    options.push_back({ "context hacking" });
    options.push_back({ "*",           "       --rope-scaling {none,linear,yarn}",
                                                                        "RoPE frequency scaling method, defaults to linear unless specified by the model" });
    options.push_back({ "*",           "       --rope-scale N",         "RoPE context scaling factor, expands context by a factor of N" });
    options.push_back({ "*",           "       --rope-freq-base N",     "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)" });
    options.push_back({ "*",           "       --rope-freq-scale N",    "RoPE frequency scaling factor, expands context by a factor of 1/N" });
    options.push_back({ "*",           "       --yarn-orig-ctx N",      "YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx });
    options.push_back({ "*",           "       --yarn-ext-factor N",    "YaRN: extrapolation mix factor (default: %.1f, 0.0 = full interpolation)", (double)params.yarn_ext_factor });
    options.push_back({ "*",           "       --yarn-attn-factor N",   "YaRN: scale sqrt(t) or attention magnitude (default: %.1f)", (double)params.yarn_attn_factor });
    options.push_back({ "*",           "       --yarn-beta-slow N",     "YaRN: high correction dim or alpha (default: %.1f)", (double)params.yarn_beta_slow });
    options.push_back({ "*",           "       --yarn-beta-fast N",     "YaRN: low correction dim or beta (default: %.1f)", (double)params.yarn_beta_fast });
    options.push_back({ "*",           "-gan,  --grp-attn-n N",         "group-attention factor (default: %d)", params.grp_attn_n });
    options.push_back({ "*",           "-gaw,  --grp-attn-w N",         "group-attention width (default: %.1f)", (double)params.grp_attn_w });
    options.push_back({ "*",           "-dkvc, --dump-kv-cache",        "verbose print of the KV cache" });
    options.push_back({ "*",           "-nkvo, --no-kv-offload",        "disable KV offload" });
    options.push_back({ "*",           "-ctk,  --cache-type-k TYPE",    "KV cache data type for K (default: %s)", params.cache_type_k.c_str() });
    options.push_back({ "*",           "-ctv,  --cache-type-v TYPE",    "KV cache data type for V (default: %s)", params.cache_type_v.c_str() });

    options.push_back({ "perplexity" });
    options.push_back({ "perplexity",  "       --all-logits",           "return logits for all tokens in the batch (default: %s)", params.logits_all ? "true" : "false" });
    options.push_back({ "perplexity",  "       --hellaswag",            "compute HellaSwag score over random tasks from datafile supplied with -f" });
    options.push_back({ "perplexity",  "       --hellaswag-tasks N",    "number of tasks to use when computing the HellaSwag score (default: %zu)", params.hellaswag_tasks });
    options.push_back({ "perplexity",  "       --winogrande",           "compute Winogrande score over random tasks from datafile supplied with -f" });
    options.push_back({ "perplexity",  "       --winogrande-tasks N",   "number of tasks to use when computing the Winogrande score (default: %zu)", params.winogrande_tasks });
    options.push_back({ "perplexity",  "       --multiple-choice",      "compute multiple choice score over random tasks from datafile supplied with -f" });
    options.push_back({ "perplexity",  "       --multiple-choice-tasks N",
                                                                        "number of tasks to use when computing the multiple choice score (default: %zu)", params.multiple_choice_tasks });
    options.push_back({ "perplexity",  "       --kl-divergence",        "computes KL-divergence to logits provided via --kl-divergence-base" });
    options.push_back({ "perplexity",  "       --ppl-stride N",         "stride for perplexity calculation (default: %d)", params.ppl_stride });
    options.push_back({ "perplexity",  "       --ppl-output-type {0,1}",
                                                                        "output type for perplexity calculation (default: %d)", params.ppl_output_type });

    options.push_back({ "parallel" });
    options.push_back({ "*",           "-dt,   --defrag-thold N",       "KV cache defragmentation threshold (default: %.1f, < 0 - disabled)", (double)params.defrag_thold });
    options.push_back({ "*",           "-np,   --parallel N",           "number of parallel sequences to decode (default: %d)", params.n_parallel });
    options.push_back({ "*",           "-ns,   --sequences N",          "number of sequences to decode (default: %d)", params.n_sequences });
    options.push_back({ "*",           "-cb,   --cont-batching",        "enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled" });
    options.push_back({ "*",           "-nocb, --no-cont-batching",     "disable continuous batching" });

    options.push_back({ "multi-modality" });
    options.push_back({ "*",           "       --mmproj FILE",          "path to a multimodal projector file for LLaVA. see examples/llava/README.md" });
    options.push_back({ "*",           "       --image FILE",           "path to an image file. use with multimodal models. Specify multiple times for batching" });

    options.push_back({ "backend" });
    options.push_back({ "*",           "       --rpc SERVERS",          "comma separated list of RPC servers" });

    if (llama_supports_mlock()) {
        options.push_back({ "*",           "       --mlock",                "force system to keep model in RAM rather than swapping or compressing" });
    }
    if (llama_supports_mmap()) {
        options.push_back({ "*",           "       --no-mmap",              "do not memory-map model (slower load but may reduce pageouts if not using mlock)" });
    }
    options.push_back({ "*",           "       --numa TYPE",            "attempt optimizations that help on some NUMA systems\n"
                                                                        "  - distribute: spread execution evenly over all nodes\n"
                                                                        "  - isolate: only spawn threads on CPUs on the node that execution started on\n"
                                                                        "  - numactl: use the CPU map provided by numactl\n"
                                                                        "if run without this previously, it is recommended to drop the system page cache before using this\n"
                                                                        "see https://github.com/ggerganov/llama.cpp/issues/1437" });

    if (llama_supports_gpu_offload()) {
        options.push_back({ "*",           "-ngl,  --gpu-layers N",
                                                                        "number of layers to store in VRAM" });
        options.push_back({ "*",           "-ngld, --gpu-layers-draft N",
                                                                        "number of layers to store in VRAM for the draft model" });
        options.push_back({ "*",           "-sm,   --split-mode SPLIT_MODE",
                                                                        "how to split the model across multiple GPUs, one of:\n"
                                                                        "  - none: use one GPU only\n"
                                                                        "  - layer (default): split layers and KV across GPUs\n"
                                                                        "  - row: split rows across GPUs" });
        options.push_back({ "*",           "-ts,   --tensor-split SPLIT",
                                                                        "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1" });
        options.push_back({ "*",           "-mg,   --main-gpu i",       "the GPU to use for the model (with split-mode = none),\n"
                                                                        "or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu });
    }

    options.push_back({ "model" });
    options.push_back({ "*",           "       --check-tensors",        "check model tensor data for invalid values (default: %s)", params.check_tensors ? "true" : "false" });
    options.push_back({ "*",           "       --override-kv KEY=TYPE:VALUE",
                                                                        "advanced option to override model metadata by key. may be specified multiple times.\n"
                                                                        "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false" });
    options.push_back({ "*",           "       --lora FNAME",           "apply LoRA adapter (can be repeated to use multiple adapters)" });
    options.push_back({ "*",           "       --lora-scaled FNAME S",  "apply LoRA adapter with user defined scaling S (can be repeated to use multiple adapters)" });
    options.push_back({ "*",           "       --control-vector FNAME", "add a control vector\n"
                                                                        "note: this argument can be repeated to add multiple control vectors" });
    options.push_back({ "*",           "       --control-vector-scaled FNAME SCALE",
                                                                        "add a control vector with user defined scaling SCALE\n"
                                                                        "note: this argument can be repeated to add multiple scaled control vectors" });
    options.push_back({ "*",           "       --control-vector-layer-range START END",
                                                                        "layer range to apply the control vector(s) to, start and end inclusive" });
    options.push_back({ "*",           "-m,    --model FNAME",          "model path (default: models/$filename with filename from --hf-file\n"
                                                                        "or --model-url if set, otherwise %s)", DEFAULT_MODEL_PATH });
    options.push_back({ "*",           "-md,   --model-draft FNAME",    "draft model for speculative decoding (default: unused)" });
    options.push_back({ "*",           "-mu,   --model-url MODEL_URL",  "model download url (default: unused)" });
    options.push_back({ "*",           "-hfr,  --hf-repo REPO",         "Hugging Face model repository (default: unused)" });
    options.push_back({ "*",           "-hff,  --hf-file FILE",         "Hugging Face model file (default: unused)" });
    options.push_back({ "*",           "-hft,  --hf-token TOKEN",       "Hugging Face access token (default: value from HF_TOKEN environment variable)" });

    options.push_back({ "retrieval" });
    options.push_back({ "retrieval",   "       --context-file FNAME",   "file to load context from (repeat to specify multiple files)" });
    options.push_back({ "retrieval",   "       --chunk-size N",         "minimum length of embedded text chunks (default: %d)", params.chunk_size });
    options.push_back({ "retrieval",   "       --chunk-separator STRING",
                                                                        "separator between chunks (default: '%s')", params.chunk_separator.c_str() });

    options.push_back({ "passkey" });
    options.push_back({ "passkey",     "       --junk N",               "number of times to repeat the junk text (default: %d)", params.n_junk });
    options.push_back({ "passkey",     "       --pos N",                "position of the passkey in the junk text (default: %d)", params.i_pos });

    options.push_back({ "imatrix" });
    options.push_back({ "imatrix",     "-o,    --output FNAME",         "output file (default: '%s')", params.out_file.c_str() });
    options.push_back({ "imatrix",     "       --output-frequency N",   "output the imatrix every N iterations (default: %d)", params.n_out_freq });
    options.push_back({ "imatrix",     "       --save-frequency N",     "save an imatrix copy every N iterations (default: %d)", params.n_save_freq });
    options.push_back({ "imatrix",     "       --process-output",       "collect data for the output tensor (default: %s)", params.process_output ? "true" : "false" });
    options.push_back({ "imatrix",     "       --no-ppl",               "do not compute perplexity (default: %s)", params.compute_ppl ? "true" : "false" });
    options.push_back({ "imatrix",     "       --chunk N",              "start processing the input from chunk N (default: %d)", params.i_chunk });

    options.push_back({ "bench" });
    options.push_back({ "bench",       "-pps",                          "is the prompt shared across parallel sequences (default: %s)", params.is_pp_shared ? "true" : "false" });
    options.push_back({ "bench",       "-npp n0,n1,...",                "number of prompt tokens" });
    options.push_back({ "bench",       "-ntg n0,n1,...",                "number of text generation tokens" });
    options.push_back({ "bench",       "-npl n0,n1,...",                "number of parallel prompts" });

    options.push_back({ "embedding" });
    options.push_back({ "embedding",   "       --embd-normalize",       "normalisation for embendings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", params.embd_normalize });
    options.push_back({ "embedding",   "       --embd-output-format",   "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix" });
    options.push_back({ "embedding",   "       --embd-separator",       "separator of embendings (default \\n) for example \"<#sep#>\"" });

    options.push_back({ "server" });
    options.push_back({ "server",      "       --host HOST",            "ip address to listen (default: %s)", params.hostname.c_str() });
    options.push_back({ "server",      "       --port PORT",            "port to listen (default: %d)", params.port });
    options.push_back({ "server",      "       --path PATH",            "path to serve static files from (default: %s)", params.public_path.c_str() });
    options.push_back({ "server",      "       --url-prefix PREFIX",    "Specify a URL prefix (subdirectory) under which the API will be served, e.g. /llamafile (default: %s)", params.url_prefix.c_str() });
    options.push_back({ "server",      "       --embedding(s)",         "enable embedding endpoint (default: %s)", params.embedding ? "enabled" : "disabled" });
    options.push_back({ "server",      "       --api-key KEY",          "API key to use for authentication (default: none)" });
    options.push_back({ "server",      "       --api-key-file FNAME",   "path to file containing API keys (default: none)" });
    options.push_back({ "server",      "       --ssl-key-file FNAME",   "path to file a PEM-encoded SSL private key" });
    options.push_back({ "server",      "       --ssl-cert-file FNAME",  "path to file a PEM-encoded SSL certificate" });
    options.push_back({ "server",      "       --timeout N",            "server read/write timeout in seconds (default: %d)", params.timeout_read });
    options.push_back({ "server",      "       --threads-http N",       "number of threads used to process HTTP requests (default: %d)", params.n_threads_http });
    options.push_back({ "server",      "       --system-prompt-file FNAME",
                                                                        "set a file to load a system prompt (initial prompt of all slots), this is useful for chat applications" });
    options.push_back({ "server",      "       --log-format {text,json}",
                                                                        "log output format: json or text (default: json)" });
    options.push_back({ "server",      "       --metrics",              "enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled" });
    options.push_back({ "server",      "       --no-slots",             "disables slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled" });
    options.push_back({ "server",      "       --slot-save-path PATH",  "path to save slot kv cache (default: disabled)" });
    options.push_back({ "server",      "       --chat-template JINJA_TEMPLATE",
                                                                        "set custom jinja chat template (default: template taken from model's metadata)\n"
                                                                        "only commonly used templates are accepted:\n"
                                                                        "https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template" });
    options.push_back({ "server",      "-sps,  --slot-prompt-similarity SIMILARITY",
                                                                        "how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity });

#ifndef LOG_DISABLE_LOGS
    options.push_back({ "logging" });
    options.push_back({ "*",           "       --simple-io",            "use basic IO for better compatibility in subprocesses and limited consoles" });
    options.push_back({ "*",           "-ld,   --logdir LOGDIR",        "path under which to save YAML logs (no logging if unset)" });
    options.push_back({ "logging",     "       --log-test",             "Run simple logging test" });
    options.push_back({ "logging",     "       --log-disable",          "Disable trace logs" });
    options.push_back({ "logging",     "       --log-enable",           "Enable trace logs" });
    options.push_back({ "logging",     "       --log-file FNAME",       "Specify a log filename (without extension)" });
    options.push_back({ "logging",     "       --log-new",              "Create a separate new log file on start. "
                                                                        "Each log file will have unique name: \"<name>.<ID>.log\"" });
    options.push_back({ "logging",     "       --log-append",           "Don't truncate the old log file." });
#endif // LOG_DISABLE_LOGS

    options.push_back({ "cvector" });
    options.push_back({ "cvector",     "-o,    --output FNAME",         "output file (default: '%s')", params.cvector_outfile.c_str() });
    options.push_back({ "cvector",     "       --positive-file FNAME",  "positive prompts file, one prompt per line (default: '%s')", params.cvector_positive_file.c_str() });
    options.push_back({ "cvector",     "       --negative-file FNAME",  "negative prompts file, one prompt per line (default: '%s')", params.cvector_negative_file.c_str() });
    options.push_back({ "cvector",     "       --pca-batch N",          "batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", params.n_pca_batch });
    options.push_back({ "cvector",     "       --pca-iter N",           "number of iterations used for PCA (default: %d)", params.n_pca_iterations });
    options.push_back({ "cvector",     "       --method {pca,mean}",    "dimensionality reduction method to be used (default: pca)" });

    options.push_back({ "export-lora" });
    options.push_back({ "export-lora", "-m,    --model",                "model path from which to load base model (default '%s')", params.model.c_str() });
    options.push_back({ "export-lora", "       --lora FNAME",           "path to LoRA adapter  (can be repeated to use multiple adapters)" });
    options.push_back({ "export-lora", "       --lora-scaled FNAME S",  "path to LoRA adapter with user defined scaling S  (can be repeated to use multiple adapters)" });
    options.push_back({ "*",           "-t,    --threads N",            "number of threads to use during computation (default: %d)", params.n_threads });
    options.push_back({ "export-lora", "-o,    --output FNAME",         "output file (default: '%s')", params.lora_outfile.c_str() });

    printf("usage: %s [options]\n", argv[0]);

    for (const auto & o : options) {
        if (!o.grp.empty()) {
            printf("\n%s:\n\n", o.grp.c_str());
            continue;
        }
        printf("  %-32s", o.args.c_str());
        if (o.args.length() > 30) {
            printf("\n%34s", "");
        }

        const auto desc = o.desc;
        size_t start = 0;
        size_t end = desc.find('\n');
        while (end != std::string::npos) {
            printf("%s\n%34s", desc.substr(start, end - start).c_str(), "");
            start = end + 1;
            end = desc.find('\n', start);
        }

        printf("%s\n", desc.substr(start).c_str());
    }
    printf("\n");
}

std::string gpt_params_get_system_info(const gpt_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.n_threads;
    if (params.n_threads_batch != -1) {
        os << " (n_threads_batch = " << params.n_threads_batch << ")";
    }
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();

    return os.str();
}

//
// String utils
//

std::vector<std::string> string_split(std::string input, char separator) {
    std::vector<std::string> parts;
    size_t separator_pos = input.find(separator);
    while (separator_pos != std::string::npos) {
        std::string part = input.substr(0, separator_pos);
        parts.emplace_back(part);
        input = input.substr(separator_pos + 1);
        separator_pos = input.find(separator);
    }
    parts.emplace_back(input);
    return parts;
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

std::string replace_all(const std::string& s, const std::string& search, const std::string& replace) {
    if (search.empty())
        return s;
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    return builder;
}

void string_process_escapes(std::string & input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        fprintf(stderr, "%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            fprintf(stderr, "%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            fprintf(stderr, "%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        fprintf(stderr, "%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    std::u32string filename_utf32;
    try {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        filename_utf32 = converter.from_bytes(filename);

        // If the reverse conversion mismatches, it means overlong UTF-8 sequences were used,
        // or invalid encodings were encountered. Reject such attempts
        std::string filename_reencoded = converter.to_bytes(filename_utf32);
        if (filename_reencoded != filename) {
            return false;
        }
    } catch (const std::exception &) {
        return false;
    }

    // Check for forbidden codepoints:
    // - Control characters
    // - Unicode equivalents of illegal characters
    // - UTF-16 surrogate pairs
    // - UTF-8 replacement character
    // - Byte order mark (BOM)
    // - Illegal characters: / \ : * ? " < > |
    for (char32_t c : filename_utf32) {
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == '/' || c == '\\' || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    return true;
}

// returns true if successful, false otherwise
bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wpath = converter.from_bytes(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);
        const wchar_t * test = subpath.c_str();

        const bool success = CreateDirectoryW(test, NULL);
        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
        if (IsLinux()) { // [jart]
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        }
        } else if (IsXnu()) {
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
        } else if (IsWindows()) {
        cache_directory = std::getenv("LOCALAPPDATA");
        }
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

std::string fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    std::string cache_directory = fs_get_cache_directory();
    const bool success = fs_create_directory_with_parents(cache_directory);
    if (!success) {
        throw std::runtime_error("failed to create cache directory: " + cache_directory);
    }
    return cache_directory + filename;
}


//
// Model utils
//
struct llama_init_result llama_init_from_gpt_params(gpt_params & params) {
    llama_init_result iparams;
    auto mparams = llama_model_params_from_gpt_params(params);

    llama_model * model = nullptr;

    if (!params.hf_repo.empty() && !params.hf_file.empty()) {
        model = llama_load_model_from_hf(params.hf_repo.c_str(), params.hf_file.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else if (!params.model_url.empty()) {
        model = llama_load_model_from_url(params.model_url.c_str(), params.model.c_str(), params.hf_token.c_str(), mparams);
    } else {
        model = llama_load_model_from_file(params.model.c_str(), mparams);
    }

    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return iparams;
    }

    auto cparams = llama_context_params_from_gpt_params(params);

    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return iparams;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_n_layer(model);

        const auto cvec = llama_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }

        int err = llama_control_vector_apply(lctx,
                                             cvec.data.data(),
                                             cvec.data.size(),
                                             cvec.n_embd,
                                             params.control_vector_layer_start,
                                             params.control_vector_layer_end);
        if (err) {
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }
    }

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_lora_adapter_container loaded_la;
        loaded_la.path = la.path;
        loaded_la.scale = la.scale;
        loaded_la.adapter = llama_lora_adapter_init(model, la.path.c_str());
        if (loaded_la.adapter == nullptr) {
            fprintf(stderr, "%s: error: failed to apply lora adapter '%s'\n", __func__, la.path.c_str());
            llama_free(lctx);
            llama_free_model(model);
            return iparams;
        }
        iparams.lora_adapters.push_back(loaded_la); // copy to list of loaded adapters
    }
    if (!params.lora_init_without_apply) {
        llama_lora_adapters_apply(lctx, iparams.lora_adapters);
    }

    if (params.ignore_eos) {
        params.sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
    }

    if (params.warmup) {
        LOG("warming up the model with an empty run\n");

        std::vector<llama_token> tmp;
        llama_token bos = llama_token_bos(model);
        llama_token eos = llama_token_eos(model);
        // some models (e.g. T5) don't have a BOS token
        if (bos != -1) {
            tmp.push_back(bos);
        }
        tmp.push_back(eos);

        if (llama_model_has_encoder(model)) {
            llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size(), 0, 0));
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == -1) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch), 0, 0));
        }
        llama_kv_cache_clear(lctx);
        llama_synchronize(lctx);
        llama_reset_timings(lctx);
    }

    iparams.model   = model;
    iparams.context = lctx;
    return iparams;
}

void llama_lora_adapters_apply(struct llama_context * ctx, std::vector<llama_lora_adapter_container> & lora_adapters) {
    llama_lora_adapter_clear(ctx);
    for (auto & la : lora_adapters) {
        if (la.scale != 0.0f) {
            llama_lora_adapter_set(ctx, la.adapter, la.scale);
        }
    }
}

struct llama_model_params llama_model_params_from_gpt_params(const gpt_params & params) {
    auto mparams = llama_model_default_params();

    if (params.n_gpu_layers != -1) {
        mparams.n_gpu_layers = params.n_gpu_layers;
    }
    mparams.rpc_servers     = params.rpc_servers.c_str();
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.use_mmap        = params.use_mmap;
    mparams.use_mlock       = params.use_mlock;
    mparams.check_tensors   = params.check_tensors;
    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    return mparams;
}

static ggml_type kv_cache_type_from_str(const std::string & s) {
    if (s == "f32") {
        return GGML_TYPE_F32;
    }
    if (s == "bf16") {
        return GGML_TYPE_BF16;
    }
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    if (s == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (s == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (s == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (s == "iq4_nl") {
        return GGML_TYPE_IQ4_NL;
    }
    if (s == "q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (s == "q5_1") {
        return GGML_TYPE_Q5_1;
    }

    throw std::runtime_error("Invalid cache type: " + s);
}

struct llama_context_params llama_context_params_from_gpt_params(const gpt_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.n_threads;
    cparams.n_threads_batch   = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
    cparams.seed              = params.seed;
    cparams.logits_all        = params.logits_all;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.defrag_thold      = params.defrag_thold;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.flash_attn        = params.flash_attn;

    cparams.type_k = kv_cache_type_from_str(params.cache_type_k);
    cparams.type_v = kv_cache_type_from_str(params.cache_type_v);

    return cparams;
}

#ifdef LLAMA_USE_CURL

static bool starts_with(const std::string & str, const std::string & prefix) {
    // While we wait for C++20's std::string::starts_with...
    return str.rfind(prefix, 0) == 0;
}

static bool llama_download_file(const std::string & url, const std::string & path, const std::string & hf_token) {

    // Initialize libcurl
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), &curl_easy_cleanup);
    if (!curl) {
        fprintf(stderr, "%s: error initializing libcurl\n", __func__);
        return false;
    }

    bool force_download = false;

    // Set the URL, allow to follow http redirection
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);

    // Check if hf-token or bearer-token was specified
    if (!hf_token.empty()) {
      std::string auth_header = "Authorization: Bearer ";
      auth_header += hf_token.c_str();
      struct curl_slist *http_headers = NULL;
      http_headers = curl_slist_append(http_headers, auth_header.c_str());
      curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, http_headers);
    }

#if defined(_WIN32)
    // CURLSSLOPT_NATIVE_CA tells libcurl to use standard certificate store of
    //   operating system. Currently implemented under MS-Windows.
    curl_easy_setopt(curl.get(), CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
#endif

    // Check if the file already exists locally
    struct stat model_file_info;
    auto file_exists = (stat(path.c_str(), &model_file_info) == 0);

    // If the file exists, check its JSON metadata companion file.
    std::string metadata_path = path + ".json";
    nlohmann::json metadata;
    std::string etag;
    std::string last_modified;

    if (file_exists) {
        // Try and read the JSON metadata file (note: stream autoclosed upon exiting this block).
        std::ifstream metadata_in(metadata_path);
        if (metadata_in.good()) {
            try {
                metadata_in >> metadata;
                fprintf(stderr, "%s: previous metadata file found %s: %s\n", __func__, metadata_path.c_str(), metadata.dump().c_str());
                if (metadata.contains("url") && metadata.at("url").is_string()) {
                    auto previous_url = metadata.at("url").get<std::string>();
                    if (previous_url != url) {
                        fprintf(stderr, "%s: Model URL mismatch: %s != %s\n", __func__, url.c_str(), previous_url.c_str());
                        return false;
                    }
                }
                if (metadata.contains("etag") && metadata.at("etag").is_string()) {
                    etag = metadata.at("etag");
                }
                if (metadata.contains("lastModified") && metadata.at("lastModified").is_string()) {
                    last_modified = metadata.at("lastModified");
                }
            } catch (const nlohmann::json::exception & e) {
                fprintf(stderr, "%s: error reading metadata file %s: %s\n", __func__, metadata_path.c_str(), e.what());
                return false;
            }
        }
    } else {
        fprintf(stderr, "%s: no previous model file found %s\n", __func__, path.c_str());
    }

    // Send a HEAD request to retrieve the etag and last-modified headers
    struct llama_load_model_from_url_headers {
        std::string etag;
        std::string last_modified;
    };
    llama_load_model_from_url_headers headers;
    {
        typedef size_t(*CURLOPT_HEADERFUNCTION_PTR)(char *, size_t, size_t, void *);
        auto header_callback = [](char * buffer, size_t /*size*/, size_t n_items, void * userdata) -> size_t {
            llama_load_model_from_url_headers *headers = (llama_load_model_from_url_headers *) userdata;

            static std::regex header_regex("([^:]+): (.*)\r\n");
            static std::regex etag_regex("ETag", std::regex_constants::icase);
            static std::regex last_modified_regex("Last-Modified", std::regex_constants::icase);

            std::string header(buffer, n_items);
            std::smatch match;
            if (std::regex_match(header, match, header_regex)) {
                const std::string & key = match[1];
                const std::string & value = match[2];
                if (std::regex_match(key, match, etag_regex)) {
                    headers->etag = value;
                } else if (std::regex_match(key, match, last_modified_regex)) {
                    headers->last_modified = value;
                }
            }
            return n_items;
        };

        curl_easy_setopt(curl.get(), CURLOPT_NOBODY, 1L); // will trigger the HEAD verb
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 1L); // hide head request progress
        curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, static_cast<CURLOPT_HEADERFUNCTION_PTR>(header_callback));
        curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &headers);

        CURLcode res = curl_easy_perform(curl.get());
        if (res != CURLE_OK) {
            fprintf(stderr, "%s: curl_easy_perform() failed: %s\n", __func__, curl_easy_strerror(res));
            return false;
        }

        long http_code = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code != 200) {
            // HEAD not supported, we don't know if the file has changed
            // force trigger downloading
            force_download = true;
            fprintf(stderr, "%s: HEAD invalid http status code received: %ld\n", __func__, http_code);
        }
    }

    bool should_download = !file_exists || force_download;
    if (!should_download) {
        if (!etag.empty() && etag != headers.etag) {
            fprintf(stderr, "%s: ETag header is different (%s != %s): triggering a new download\n", __func__, etag.c_str(), headers.etag.c_str());
            should_download = true;
        } else if (!last_modified.empty() && last_modified != headers.last_modified) {
            fprintf(stderr, "%s: Last-Modified header is different (%s != %s): triggering a new download\n", __func__, last_modified.c_str(), headers.last_modified.c_str());
            should_download = true;
        }
    }
    if (should_download) {
        std::string path_temporary = path + ".downloadInProgress";
        if (file_exists) {
            fprintf(stderr, "%s: deleting previous downloaded file: %s\n", __func__, path.c_str());
            if (remove(path.c_str()) != 0) {
                fprintf(stderr, "%s: unable to delete file: %s\n", __func__, path.c_str());
                return false;
            }
        }

        // Set the output file

        struct FILE_deleter {
            void operator()(FILE * f) const {
                fclose(f);
            }
        };

        std::unique_ptr<FILE, FILE_deleter> outfile(fopen(path_temporary.c_str(), "wb"));
        if (!outfile) {
            fprintf(stderr, "%s: error opening local file for writing: %s\n", __func__, path.c_str());
            return false;
        }

        typedef size_t(*CURLOPT_WRITEFUNCTION_PTR)(void * data, size_t size, size_t nmemb, void * fd);
        auto write_callback = [](void * data, size_t size, size_t nmemb, void * fd) -> size_t {
            return fwrite(data, size, nmemb, (FILE *)fd);
        };
        curl_easy_setopt(curl.get(), CURLOPT_NOBODY, 0L);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, static_cast<CURLOPT_WRITEFUNCTION_PTR>(write_callback));
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, outfile.get());

        //  display download progress
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);

        // helper function to hide password in URL
        auto llama_download_hide_password_in_url = [](const std::string & url) -> std::string {
            std::size_t protocol_pos = url.find("://");
            if (protocol_pos == std::string::npos) {
                return url;  // Malformed URL
            }

            std::size_t at_pos = url.find('@', protocol_pos + 3);
            if (at_pos == std::string::npos) {
                return url;  // No password in URL
            }

            return url.substr(0, protocol_pos + 3) + "********" + url.substr(at_pos);
        };

        // start the download
        fprintf(stderr, "%s: downloading from %s to %s (server_etag:%s, server_last_modified:%s)...\n", __func__,
                llama_download_hide_password_in_url(url).c_str(), path.c_str(), headers.etag.c_str(), headers.last_modified.c_str());
        auto res = curl_easy_perform(curl.get());
        if (res != CURLE_OK) {
            fprintf(stderr, "%s: curl_easy_perform() failed: %s\n", __func__, curl_easy_strerror(res));
            return false;
        }

        long http_code = 0;
        curl_easy_getinfo (curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code < 200 || http_code >= 400) {
            fprintf(stderr, "%s: invalid http status code received: %ld\n", __func__, http_code);
            return false;
        }

        // Causes file to be closed explicitly here before we rename it.
        outfile.reset();

        // Write the updated JSON metadata file.
        metadata.update({
            {"url", url},
            {"etag", headers.etag},
            {"lastModified", headers.last_modified}
        });
        std::ofstream(metadata_path) << metadata.dump(4);
        fprintf(stderr, "%s: file metadata saved: %s\n", __func__, metadata_path.c_str());

        if (rename(path_temporary.c_str(), path.c_str()) != 0) {
            fprintf(stderr, "%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
            return false;
        }
    }

    return true;
}

struct llama_model * llama_load_model_from_url(
        const char * model_url,
        const char * path_model,
        const char * hf_token,
        const struct llama_model_params & params) {
    // Basic validation of the model_url
    if (!model_url || strlen(model_url) == 0) {
        fprintf(stderr, "%s: invalid model_url\n", __func__);
        return NULL;
    }

    if (!llama_download_file(model_url, path_model, hf_token)) {
        return NULL;
    }

    // check for additional GGUFs split to download
    int n_split = 0;
    {
        struct gguf_init_params gguf_params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ NULL,
        };
        auto * ctx_gguf = gguf_init_from_file(path_model, gguf_params);
        if (!ctx_gguf) {
            fprintf(stderr, "\n%s:  failed to load input GGUF from %s\n", __func__, path_model);
            return NULL;
        }

        auto key_n_split = gguf_find_key(ctx_gguf, LLM_KV_SPLIT_COUNT);
        if (key_n_split >= 0) {
            n_split = gguf_get_val_u16(ctx_gguf, key_n_split);
        }

        gguf_free(ctx_gguf);
    }

    if (n_split > 1) {
        char split_prefix[PATH_MAX] = {0};
        char split_url_prefix[LLAMA_CURL_MAX_URL_LENGTH] = {0};

        // Verify the first split file format
        // and extract split URL and PATH prefixes
        {
            if (!llama_split_prefix(split_prefix, sizeof(split_prefix), path_model, 0, n_split)) {
                fprintf(stderr, "\n%s: unexpected model file name: %s"
                                " n_split=%d\n", __func__, path_model, n_split);
                return NULL;
            }

            if (!llama_split_prefix(split_url_prefix, sizeof(split_url_prefix), model_url, 0, n_split)) {
                fprintf(stderr, "\n%s: unexpected model url: %s"
                                " n_split=%d\n", __func__, model_url, n_split);
                return NULL;
            }
        }

        // Prepare download in parallel
        std::vector<std::future<bool>> futures_download;
        for (int idx = 1; idx < n_split; idx++) {
            futures_download.push_back(std::async(std::launch::async, [&split_prefix, &split_url_prefix, &n_split, hf_token](int download_idx) -> bool {
                char split_path[PATH_MAX] = {0};
                llama_split_path(split_path, sizeof(split_path), split_prefix, download_idx, n_split);

                char split_url[LLAMA_CURL_MAX_URL_LENGTH] = {0};
                llama_split_path(split_url, sizeof(split_url), split_url_prefix, download_idx, n_split);

                return llama_download_file(split_url, split_path, hf_token);
            }, idx));
        }

        // Wait for all downloads to complete
        for (auto & f : futures_download) {
            if (!f.get()) {
                return NULL;
            }
        }
    }

    return llama_load_model_from_file(path_model, params);
}

struct llama_model * llama_load_model_from_hf(
        const char * repo,
        const char * model,
        const char * path_model,
        const char * hf_token,
        const struct llama_model_params & params) {
    // construct hugging face model url:
    //
    //  --repo ggml-org/models --file tinyllama-1.1b/ggml-model-f16.gguf
    //    https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf
    //
    //  --repo TheBloke/Mixtral-8x7B-v0.1-GGUF --file mixtral-8x7b-v0.1.Q4_K_M.gguf
    //    https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b-v0.1.Q4_K_M.gguf
    //

    std::string model_url = "https://huggingface.co/";
    model_url += repo;
    model_url += "/resolve/main/";
    model_url += model;

    return llama_load_model_from_url(model_url.c_str(), path_model, hf_token, params);
}

#else

struct llama_model * llama_load_model_from_url(
        const char * /*model_url*/,
        const char * /*path_model*/,
        const char * /*hf_token*/,
        const struct llama_model_params & /*params*/) {
    fprintf(stderr, "%s: llama.cpp built without libcurl, downloading from an url not supported.\n", __func__);
    return nullptr;
}

struct llama_model * llama_load_model_from_hf(
        const char * /*repo*/,
        const char * /*model*/,
        const char * /*path_model*/,
        const char * /*hf_token*/,
        const struct llama_model_params & /*params*/) {
    fprintf(stderr, "%s: llama.cpp built without libcurl, downloading from Hugging Face not supported.\n", __func__);
    return nullptr;
}

#endif // LLAMA_USE_CURL

//
// Batch utils
//

void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    return llama_tokenize(llama_get_model(ctx), text, add_special, parse_special);
}

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string llama_detokenize(llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

bool llama_should_add_bos_token(const llama_model * model) {
    const int add_bos = llama_add_bos_token(model);

    return add_bos != -1 ? bool(add_bos) : (llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM);
}

//
// Chat template utils
//

bool llama_chat_verify_template(const std::string & tmpl) {
    llama_chat_message chat[] = {{"user", "test"}};
    int res = llama_chat_apply_template(nullptr, tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string llama_chat_apply_template(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & msgs,
        bool add_ass) {
    int alloc_size = 0;
    bool fallback = false; // indicate if we must fallback to default chatml
    std::vector<llama_chat_message> chat;
    for (auto & msg : msgs) {
        chat.push_back({msg.role.c_str(), msg.content.c_str()});
        alloc_size += (msg.role.size() + msg.content.size()) * 1.25;
    }

    const char * ptr_tmpl = tmpl.empty() ? nullptr : tmpl.c_str();
    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    int32_t res = llama_chat_apply_template(model, ptr_tmpl, chat.data(), chat.size(), add_ass, buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        if (ptr_tmpl != nullptr) {
            // if the custom "tmpl" is not supported, we throw an error
            // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
            throw std::runtime_error("this custom template is not supported");
        } else {
            // If the built-in template is not supported, we default to chatml
            res = llama_chat_apply_template(nullptr, "chatml", chat.data(), chat.size(), add_ass, buf.data(), buf.size());
            fallback = true;
        }
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(
            fallback ? nullptr : model,
            fallback ? "chatml" : ptr_tmpl,
            chat.data(), chat.size(), add_ass, buf.data(), buf.size());
    }

    std::string formatted_chat(buf.data(), res);
    return formatted_chat;
}

std::string llama_chat_format_single(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & past_msg,
        const llama_chat_msg & new_msg,
        bool add_ass) {
    std::ostringstream ss;
    auto fmt_past_msg = past_msg.empty() ? "" : llama_chat_apply_template(model, tmpl, past_msg, false);
    std::vector<llama_chat_msg> chat_new(past_msg);
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    chat_new.push_back(new_msg);
    auto fmt_new_msg = llama_chat_apply_template(model, tmpl, chat_new, add_ass);
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string llama_chat_format_example(const struct llama_model * model,
        const std::string & tmpl) {
    std::vector<llama_chat_msg> msgs = {
        {"system",    "You are a helpful assistant"},
        {"user",      "Hello"},
        {"assistant", "Hi there"},
        {"user",      "How are you?"},
    };
    return llama_chat_apply_template(model, tmpl, msgs, true);
}

//
// KV cache utils
//

void llama_kv_cache_dump_view(const llama_kv_cache_view & view, int row_size) {
    static const char slot_chars[] = ".123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+";

    printf("=== Dumping KV cache. total cells %d, max sequences per cell %d, populated cells %d, total tokens in cache %d, largest empty slot=%d @ %d",
        view.n_cells, view.n_seq_max, view.used_cells, view.token_count, view.max_contiguous, view.max_contiguous_idx);

    llama_kv_cache_view_cell * c_curr = view.cells;
    llama_seq_id * cs_curr = view.cells_sequences;

    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        if (i % row_size == 0) {
            printf("\n%5d: ", i);
        }
        int seq_count = 0;
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] >= 0) { seq_count++; }
        }
        putchar(slot_chars[std::min(sizeof(slot_chars) - 2, size_t(seq_count))]);
    }

    printf("\n=== Done dumping\n");
}

void llama_kv_cache_dump_view_seqs(const llama_kv_cache_view & view, int row_size) {
    static const char slot_chars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    printf("=== Dumping KV cache. total cells %d, max sequences per cell %d, populated cells %d, total tokens in cache %d, largest empty slot=%d @ %d\n",
        view.n_cells, view.n_seq_max, view.used_cells, view.token_count, view.max_contiguous, view.max_contiguous_idx);

    std::unordered_map<llama_seq_id, size_t> seqs;
    llama_kv_cache_view_cell * c_curr = view.cells;
    llama_seq_id * cs_curr = view.cells_sequences;

    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] < 0) { continue; }
            if (seqs.find(cs_curr[j]) == seqs.end()) {
                if (seqs.size() + 1 >= sizeof(slot_chars)) { break; }
                const size_t sz = seqs.size();
                seqs[cs_curr[j]] = sz;
            }
        }
        if (seqs.size() + 1 >= sizeof(slot_chars)) { break; }
    }

    printf("=== Sequence legend: ");
    for (const auto & it : seqs) {
        printf("%zu=%d, ", it.second, it.first);
    }
    printf("'+'=other sequence ids");

    c_curr = view.cells;
    cs_curr = view.cells_sequences;
    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        if (i % row_size == 0) {
            printf("\n%5d: ", i);
        }
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] >= 0) {
                const auto & it = seqs.find(cs_curr[j]);
                putchar(it != seqs.end() ? int(slot_chars[it->second]) : '+');
            } else {
                putchar('.');
            }
        }
        putchar(' ');
    }

    printf("\n=== Done dumping\n");
}

//
// Embedding utils
//

void llama_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) sum = std::abs(inp[i]);
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float llama_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static llama_control_vector_data llama_control_vector_load_one(const llama_control_vector_load_info & load_info) {
    llama_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct llamafile * file = llamafile_open_gguf(load_info.fname.c_str(), "rb");
    if (!file) {
        perror(load_info.fname.c_str());
        return result;
    }
    struct gguf_context * ctx_gguf = gguf_init_from_file(file, meta_gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: failed to load control vector file from %s\n", __func__, load_info.fname.c_str());
        llamafile_close(file);
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        fprintf(stderr, "%s: no direction tensors found in %s\n", __func__, load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            fprintf(stderr, "%s: invalid/unparsable direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            fprintf(stderr, "%s: invalid (zero) direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            fprintf(stderr, "%s: invalid (non-F32) direction tensor type in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            fprintf(stderr, "%s: invalid (non-1D) direction tensor shape in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            fprintf(stderr, "%s: direction tensor in %s does not match previous dimensions\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        fprintf(stderr, "%s: skipping %s due to invalid direction tensors\n", __func__, load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);
    llamafile_close(file);

    return result;
}

llama_control_vector_data llama_control_vector_load(const std::vector<llama_control_vector_load_info> & load_infos) {
    llama_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = llama_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            fprintf(stderr, "%s: control vectors in %s does not match previous dimensions\n", __func__, info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        fprintf(stderr, "%s: no valid control vector files passed\n", __func__);
        result.data.clear();
    }

    return result;
}

//
// YAML utils
//

void yaml_dump_vector_float(FILE * stream, const char * prop_name, const std::vector<float> & data) {
    if (data.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    fprintf(stream, "%s: [", prop_name);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        fprintf(stream, "%e, ", data[i]);
    }
    fprintf(stream, "%e]\n", data.back());
}

void yaml_dump_vector_int(FILE * stream, const char * prop_name, const std::vector<int> & data) {
    if (data.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    fprintf(stream, "%s: [", prop_name);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        fprintf(stream, "%d, ", data[i]);
    }
    fprintf(stream, "%d]\n", data.back());
}

void yaml_dump_string_multiline(FILE * stream, const char * prop_name, const char * data) {
    std::string data_str(data == NULL ? "" : data);

    if (data_str.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    size_t pos_start = 0;
    size_t pos_found = 0;

    if (std::isspace(data_str[0]) || std::isspace(data_str.back())) {
        data_str = std::regex_replace(data_str, std::regex("\n"), "\\n");
        data_str = std::regex_replace(data_str, std::regex("\""), "\\\"");
        data_str = std::regex_replace(data_str, std::regex(R"(\\[^n"])"), R"(\$&)");
        data_str = "\"" + data_str + "\"";
        fprintf(stream, "%s: %s\n", prop_name, data_str.c_str());
        return;
    }

    if (data_str.find('\n') == std::string::npos) {
        fprintf(stream, "%s: %s\n", prop_name, data_str.c_str());
        return;
    }

    fprintf(stream, "%s: |\n", prop_name);
    while ((pos_found = data_str.find('\n', pos_start)) != std::string::npos) {
        fprintf(stream, "  %s\n", data_str.substr(pos_start, pos_found-pos_start).c_str());
        pos_start = pos_found + 1;
    }
}

void yaml_dump_non_result_info(FILE * stream, const gpt_params & params, const llama_context * lctx,
                               const std::string & timestamp, const std::vector<int> & prompt_tokens, const char * model_desc) {
    const llama_sampling_params & sparams = params.sparams;

    fprintf(stream, "build_commit: %s\n",        LLAMA_COMMIT);
    fprintf(stream, "build_number: %d\n",        LLAMA_BUILD_NUMBER);
    fprintf(stream, "cpu_has_arm_fma: %s\n",     ggml_cpu_has_arm_fma()     ? "true" : "false");
    fprintf(stream, "cpu_has_avx: %s\n",         ggml_cpu_has_avx()         ? "true" : "false");
    fprintf(stream, "cpu_has_avx_vnni: %s\n",    ggml_cpu_has_avx_vnni()    ? "true" : "false");
    fprintf(stream, "cpu_has_avx2: %s\n",        ggml_cpu_has_avx2()        ? "true" : "false");
    fprintf(stream, "cpu_has_avx512: %s\n",      ggml_cpu_has_avx512()      ? "true" : "false");
    fprintf(stream, "cpu_has_avx512_vbmi: %s\n", ggml_cpu_has_avx512_vbmi() ? "true" : "false");
    fprintf(stream, "cpu_has_avx512_vnni: %s\n", ggml_cpu_has_avx512_vnni() ? "true" : "false");
    fprintf(stream, "cpu_has_cuda: %s\n",        ggml_cpu_has_cuda()        ? "true" : "false");
    fprintf(stream, "cpu_has_vulkan: %s\n",      ggml_cpu_has_vulkan()      ? "true" : "false");
    fprintf(stream, "cpu_has_kompute: %s\n",     ggml_cpu_has_kompute()     ? "true" : "false");
    fprintf(stream, "cpu_has_fma: %s\n",         ggml_cpu_has_fma()         ? "true" : "false");
    fprintf(stream, "cpu_has_gpublas: %s\n",     ggml_cpu_has_gpublas()     ? "true" : "false");
    fprintf(stream, "cpu_has_neon: %s\n",        ggml_cpu_has_neon()        ? "true" : "false");
    fprintf(stream, "cpu_has_sve: %s\n",         ggml_cpu_has_sve()         ? "true" : "false");
    fprintf(stream, "cpu_has_f16c: %s\n",        ggml_cpu_has_f16c()        ? "true" : "false");
    fprintf(stream, "cpu_has_fp16_va: %s\n",     ggml_cpu_has_fp16_va()     ? "true" : "false");
    fprintf(stream, "cpu_has_wasm_simd: %s\n",   ggml_cpu_has_wasm_simd()   ? "true" : "false");
    fprintf(stream, "cpu_has_blas: %s\n",        ggml_cpu_has_blas()        ? "true" : "false");
    fprintf(stream, "cpu_has_sse3: %s\n",        ggml_cpu_has_sse3()        ? "true" : "false");
    fprintf(stream, "cpu_has_vsx: %s\n",         ggml_cpu_has_vsx()         ? "true" : "false");
    fprintf(stream, "cpu_has_matmul_int8: %s\n", ggml_cpu_has_matmul_int8() ? "true" : "false");

#ifdef NDEBUG
    fprintf(stream, "debug: false\n");
#else
    fprintf(stream, "debug: true\n");
#endif // NDEBUG

    fprintf(stream, "model_desc: %s\n", model_desc);
    fprintf(stream, "n_vocab: %d  # output size of the final layer, 32001 for some models\n", llama_n_vocab(llama_get_model(lctx)));

#ifdef __OPTIMIZE__
    fprintf(stream, "optimize: true\n");
#else
    fprintf(stream, "optimize: false\n");
#endif // __OPTIMIZE__

    fprintf(stream, "time: %s\n", timestamp.c_str());

    fprintf(stream, "\n");
    fprintf(stream, "###############\n");
    fprintf(stream, "# User Inputs #\n");
    fprintf(stream, "###############\n");
    fprintf(stream, "\n");

    fprintf(stream, "alias: %s # default: unknown\n", params.model_alias.c_str());
    fprintf(stream, "batch_size: %d # default: 512\n", params.n_batch);
    yaml_dump_string_multiline(stream, "cfg_negative_prompt", sparams.cfg_negative_prompt.c_str());
    fprintf(stream, "cfg_scale: %f # default: 1.0\n", sparams.cfg_scale);
    fprintf(stream, "chunks: %d # default: -1 (unlimited)\n", params.n_chunks);
    fprintf(stream, "color: %s # default: false\n", params.use_color ? "true" : "false");
    fprintf(stream, "ctx_size: %d # default: 512\n", params.n_ctx);
    fprintf(stream, "escape: %s # default: false\n", params.escape ? "true" : "false");
    fprintf(stream, "file: # never logged, see prompt instead. Can still be specified for input.\n");
    fprintf(stream, "frequency_penalty: %f # default: 0.0 \n", sparams.penalty_freq);
    yaml_dump_string_multiline(stream, "grammar", sparams.grammar.c_str());
    fprintf(stream, "grammar-file: # never logged, see grammar instead. Can still be specified for input.\n");
    fprintf(stream, "hellaswag: %s # default: false\n", params.hellaswag ? "true" : "false");
    fprintf(stream, "hellaswag_tasks: %zu # default: 400\n", params.hellaswag_tasks);

    const auto logit_bias_eos = sparams.logit_bias.find(llama_token_eos(llama_get_model(lctx)));
    const bool ignore_eos = logit_bias_eos != sparams.logit_bias.end() && logit_bias_eos->second == -INFINITY;
    fprintf(stream, "ignore_eos: %s # default: false\n", ignore_eos ? "true" : "false");

    yaml_dump_string_multiline(stream, "in_prefix", params.input_prefix.c_str());
    fprintf(stream, "in_prefix_bos: %s # default: false\n", params.input_prefix_bos ? "true" : "false");
    yaml_dump_string_multiline(stream, "in_suffix", params.input_prefix.c_str());
    fprintf(stream, "interactive: %s # default: false\n", params.interactive ? "true" : "false");
    fprintf(stream, "interactive_first: %s # default: false\n", params.interactive_first ? "true" : "false");
    fprintf(stream, "keep: %d # default: 0\n", params.n_keep);
    fprintf(stream, "logdir: %s # default: unset (no logging)\n", params.logdir.c_str());

    fprintf(stream, "logit_bias:\n");
    for (std::pair<llama_token, float> lb : sparams.logit_bias) {
        if (ignore_eos && lb.first == logit_bias_eos->first) {
            continue;
        }
        fprintf(stream, "  %d: %f", lb.first, lb.second);
    }

    fprintf(stream, "lora:\n");
    for (auto & la : params.lora_adapters) {
        if (la.scale == 1.0f) {
            fprintf(stream, "  - %s\n", la.path.c_str());
        }
    }
    fprintf(stream, "lora_scaled:\n");
    for (auto & la : params.lora_adapters) {
        if (la.scale != 1.0f) {
            fprintf(stream, "  - %s: %f\n", la.path.c_str(), la.scale);
        }
    }
    fprintf(stream, "main_gpu: %d # default: 0\n", params.main_gpu);
    fprintf(stream, "min_keep: %d # default: 0 (disabled)\n", sparams.min_keep);
    fprintf(stream, "mirostat: %d # default: 0 (disabled)\n", sparams.mirostat);
    fprintf(stream, "mirostat_ent: %f # default: 5.0\n", sparams.mirostat_tau);
    fprintf(stream, "mirostat_lr: %f # default: 0.1\n", sparams.mirostat_eta);
    fprintf(stream, "mlock: %s # default: false\n", params.use_mlock ? "true" : "false");
    fprintf(stream, "model: %s # default: %s\n", params.model.c_str(), DEFAULT_MODEL_PATH);
    fprintf(stream, "model_draft: %s # default:\n", params.model_draft.c_str());
    fprintf(stream, "multiline_input: %s # default: false\n", params.multiline_input ? "true" : "false");
    fprintf(stream, "n_gpu_layers: %d # default: -1\n", params.n_gpu_layers);
    fprintf(stream, "n_predict: %d # default: -1 (unlimited)\n", params.n_predict);
    fprintf(stream, "n_probs: %d # only used by server binary, default: 0\n", sparams.n_probs);
    fprintf(stream, "no_mmap: %s # default: false\n", !params.use_mmap ? "true" : "false");
    fprintf(stream, "penalize_nl: %s # default: false\n", sparams.penalize_nl ? "true" : "false");
    fprintf(stream, "ppl_output_type: %d # default: 0\n", params.ppl_output_type);
    fprintf(stream, "ppl_stride: %d # default: 0\n", params.ppl_stride);
    fprintf(stream, "presence_penalty: %f # default: 0.0\n", sparams.penalty_present);
    yaml_dump_string_multiline(stream, "prompt", params.prompt.c_str());
    fprintf(stream, "prompt_cache: %s\n", params.path_prompt_cache.c_str());
    fprintf(stream, "prompt_cache_all: %s # default: false\n", params.prompt_cache_all ? "true" : "false");
    fprintf(stream, "prompt_cache_ro: %s # default: false\n", params.prompt_cache_ro ? "true" : "false");
    yaml_dump_vector_int(stream, "prompt_tokens", prompt_tokens);
    fprintf(stream, "repeat_penalty: %f # default: 1.1\n", sparams.penalty_repeat);

    fprintf(stream, "reverse_prompt:\n");
    for (std::string ap : params.antiprompt) {
        size_t pos = 0;
        while ((pos = ap.find('\n', pos)) != std::string::npos) {
            ap.replace(pos, 1, "\\n");
            pos += 1;
        }

        fprintf(stream, "  - %s\n", ap.c_str());
    }

    fprintf(stream, "rope_freq_base: %f # default: 10000.0\n", params.rope_freq_base);
    fprintf(stream, "rope_freq_scale: %f # default: 1.0\n", params.rope_freq_scale);
    fprintf(stream, "seed: %u # default: -1 (random seed)\n", params.seed);
    fprintf(stream, "simple_io: %s # default: false\n", params.simple_io ? "true" : "false");
    fprintf(stream, "cont_batching: %s # default: false\n", params.cont_batching ? "true" : "false");
    fprintf(stream, "flash_attn: %s # default: false\n", params.flash_attn ? "true" : "false");
    fprintf(stream, "temp: %f # default: 0.8\n", sparams.temp);

    const std::vector<float> tensor_split_vector(params.tensor_split, params.tensor_split + llama_max_devices());
    yaml_dump_vector_float(stream, "tensor_split", tensor_split_vector);

    fprintf(stream, "tfs: %f # default: 1.0\n", sparams.tfs_z);
    fprintf(stream, "threads: %d # default: %u\n", params.n_threads, std::thread::hardware_concurrency());
    fprintf(stream, "top_k: %d # default: 40\n", sparams.top_k);
    fprintf(stream, "top_p: %f # default: 0.95\n", sparams.top_p);
    fprintf(stream, "min_p: %f # default: 0.0\n", sparams.min_p);
    fprintf(stream, "typical_p: %f # default: 1.0\n", sparams.typical_p);
    fprintf(stream, "verbose_prompt: %s # default: false\n", params.verbose_prompt ? "true" : "false");
    fprintf(stream, "display_prompt: %s # default: true\n", params.display_prompt ? "true" : "false");
}
