// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#include "common.h"
#include "json.h"
#include "json-schema-to-grammar.h"
#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cinttypes>
#include <codecvt>
#include <cosmo.h>

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
#define LLAMA_CURL_MAX_HEADER_LENGTH 256
#endif // LLAMA_USE_CURL

using json = nlohmann::ordered_json;

int32_t get_num_physical_cores() {
    if (IsLinux()) {
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
    }
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

#if defined(__x86_64__) && (defined(__linux__) || defined(__COSMOPOLITAN__)) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int count_math_cpus(int cpu_count) {
    int result = 0;
    for (int cpu = 0; cpu < cpu_count; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int get_math_cpu_count() {
#if defined(__x86_64__) && (defined(__linux__) || defined(__COSMOPOLITAN__)) && !defined(__ANDROID__)
    int cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_count < 1) {
        return get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = count_math_cpus(cpu_count);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#endif
    return get_num_physical_cores();
}

void process_escapes(std::string & input) {
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

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    bool result = true;
    try {
        if (!gpt_params_parse_ex(argc, argv, params)) {
            // [jart] don't show help if user didn't ask for help
            // gpt_print_usage(argc, argv, gpt_params());
            exit(0);
        }
    }
    catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        // [jart] don't show help if user didn't ask for help
        // gpt_print_usage(argc, argv, gpt_params());
        exit(1);
    }
    return result;
}

bool parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
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

bool gpt_params_find_arg(int argc, char ** argv, const std::string & arg, gpt_params & params, int & i, bool & invalid_param) {
    llama_sampling_params & sparams = params.sparams;

    if (arg == "--cli") {
        return true;
    }
    if (arg == "--fast") {
        FLAG_precise = false;
        FLAG_precision_specified = true;
        return true;
    }
    if (arg == "--precise") {
        FLAG_precise = true;
        FLAG_precision_specified = true;
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
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        // This is temporary, in the future the samplign state will be moved fully to llama_sampling_context.
        params.seed = std::stoul(argv[i]);
        sparams.seed = std::stoul(argv[i]);
        return true;
    }
    if (arg == "-t" || arg == "--threads") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_threads = std::stoi(argv[i]);
        if (params.n_threads <= 0) {
            params.n_threads = std::thread::hardware_concurrency();
        }
        return true;
    }
    if (arg == "-tb" || arg == "--threads-batch") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_threads_batch = std::stoi(argv[i]);
        if (params.n_threads_batch <= 0) {
            params.n_threads_batch = std::thread::hardware_concurrency();
        }
        return true;
    }
    if (arg == "-td" || arg == "--threads-draft") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_threads_draft = std::stoi(argv[i]);
        if (params.n_threads_draft <= 0) {
            params.n_threads_draft = std::thread::hardware_concurrency();
        }
        return true;
    }
    if (arg == "-tbd" || arg == "--threads-batch-draft") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_threads_batch_draft = std::stoi(argv[i]);
        if (params.n_threads_batch_draft <= 0) {
            params.n_threads_batch_draft = std::thread::hardware_concurrency();
        }
        return true;
    }
    if (arg == "-p" || arg == "--prompt") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.prompt = argv[i];
        return true;
    }
    if (arg == "-e" || arg == "--escape") {
        params.escape = true;
        return true;
    }
    if (arg == "--prompt-cache") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
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
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
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
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
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
    if (arg == "-n" || arg == "--n-predict") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_predict = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--top-k") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.top_k = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-c" || arg == "--ctx-size") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_ctx = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--grp-attn-n" || arg == "-gan") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.grp_attn_n = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--grp-attn-w" || arg == "-gaw") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.grp_attn_w = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--rope-freq-base") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.rope_freq_base = std::stof(argv[i]);
        return true;
    }
    if (arg == "--rope-freq-scale") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.rope_freq_scale = std::stof(argv[i]);
        return true;
    }
    if (arg == "--rope-scaling") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        std::string value(argv[i]);
        /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
        else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
        else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
        else { invalid_param = true; }
        return true;
    }
    if (arg == "--rope-scale") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.rope_freq_scale = 1.0f / std::stof(argv[i]);
        return true;
    }
    if (arg == "--yarn-orig-ctx") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.yarn_orig_ctx = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--yarn-ext-factor") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.yarn_ext_factor = std::stof(argv[i]);
        return true;
    }
    if (arg == "--yarn-attn-factor") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.yarn_attn_factor = std::stof(argv[i]);
        return true;
    }
    if (arg == "--yarn-beta-fast") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.yarn_beta_fast = std::stof(argv[i]);
        return true;
    }
    if (arg == "--yarn-beta-slow") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.yarn_beta_slow = std::stof(argv[i]);
        return true;
    }
    if (arg == "--pooling") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        std::string value(argv[i]);
        /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
        else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
        else if (value == "cls") { params.pooling_type = LLAMA_POOLING_TYPE_CLS; }
        else { invalid_param = true; }
        return true;
    }
    if (arg == "--defrag-thold" || arg == "-dt") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.defrag_thold = std::stof(argv[i]);
        return true;
    }
    if (arg == "--samplers") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        const auto sampler_names = string_split(argv[i], ';');
        sparams.samplers_sequence = sampler_types_from_names(sampler_names, true);
        return true;
    }
    if (arg == "--sampling-seq") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.samplers_sequence = sampler_types_from_chars(argv[i]);
        return true;
    }
    if (arg == "--top-p") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.top_p = std::stof(argv[i]);
        return true;
    }
    if (arg == "--min-p") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.min_p = std::stof(argv[i]);
        return true;
    }
    if (arg == "--temp") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.temp = std::stof(argv[i]);
        sparams.temp = std::max(sparams.temp, 0.0f);
        return true;
    }
    if (arg == "--tfs") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.tfs_z = std::stof(argv[i]);
        return true;
    }
    if (arg == "--typical") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.typical_p = std::stof(argv[i]);
        return true;
    }
    if (arg == "--repeat-last-n") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.penalty_last_n = std::stoi(argv[i]);
        sparams.n_prev = std::max(sparams.n_prev, sparams.penalty_last_n);
        return true;
    }
    if (arg == "--repeat-penalty") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.penalty_repeat = std::stof(argv[i]);
        return true;
    }
    if (arg == "--frequency-penalty") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.penalty_freq = std::stof(argv[i]);
        return true;
    }
    if (arg == "--presence-penalty") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.penalty_present = std::stof(argv[i]);
        return true;
    }
    if (arg == "--dynatemp-range") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.dynatemp_range = std::stof(argv[i]);
        return true;
    }
    if (arg == "--dynatemp-exp") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.dynatemp_exponent = std::stof(argv[i]);
        return true;
    }
    if (arg == "--mirostat") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.mirostat = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--mirostat-lr") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.mirostat_eta = std::stof(argv[i]);
        return true;
    }
    if (arg == "--mirostat-ent") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.mirostat_tau = std::stof(argv[i]);
        return true;
    }
    if (arg == "--cfg-negative-prompt") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.cfg_negative_prompt = argv[i];
        return true;
    }
    if (arg == "--cfg-negative-prompt-file") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
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
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.cfg_scale = std::stof(argv[i]);
        return true;
    }
    if (arg == "-b" || arg == "--batch-size") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_batch = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-ub" || arg == "--ubatch-size") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_ubatch = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--keep") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_keep = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--draft") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_draft = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--chunks") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_chunks = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-np" || arg == "--parallel") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_parallel = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-ns" || arg == "--sequences") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_sequences = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--p-split" || arg == "-ps") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.p_split = std::stof(argv[i]);
        return true;
    }
    if (arg == "-m" || arg == "--model") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.model = argv[i];
        return true;
    }
    if (arg == "-md" || arg == "--model-draft") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.model_draft = argv[i];
        return true;
    }
    if (arg == "-a" || arg == "--alias") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.model_alias = argv[i];
        return true;
    }
    if (arg == "-mu" || arg == "--model-url") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.model_url = argv[i];
        return true;
    }
    if (arg == "-hfr" || arg == "--hf-repo") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.hf_repo = argv[i];
        return true;
    }
    if (arg == "-hff" || arg == "--hf-file") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.hf_file = argv[i];
        return true;
    }
    if (arg == "--lora") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.lora_adapter.emplace_back(argv[i], 1.0f);
        params.use_mmap = false;
        return true;
    }
    if (arg == "--lora-scaled") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        const char* lora_adapter = argv[i];
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.lora_adapter.emplace_back(lora_adapter, std::stof(argv[i]));
        params.use_mmap = false;
        return true;
    }
    if (arg == "--lora-base") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.lora_base = argv[i];
        return true;
    }
    if (arg == "--control-vector") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.control_vectors.push_back({ 1.0f, argv[i], });
        return true;
    }
    if (arg == "--control-vector-scaled") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        const char* fname = argv[i];
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.control_vectors.push_back({ std::stof(argv[i]), fname, });
        return true;
    }
    if (arg == "--control-vector-layer-range") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.control_vector_layer_start = std::stoi(argv[i]);
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.control_vector_layer_end = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--mmproj") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.mmproj = argv[i];
        return true;
    }
    if (arg == "--image") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.image.emplace_back(argv[i]);
        return true;
    }
    if (arg == "-i" || arg == "--interactive") {
        params.interactive = true;
        return true;
    }
    if (arg == "--embedding") {
        params.embedding = true;
        return true;
    }
    if (arg == "--interactive-first") {
        params.interactive_first = true;
        return true;
    }
    if (arg == "-ins" || arg == "--instruct") {
        params.instruct = true;
        return true;
    }
    if (arg == "-cml" || arg == "--chatml") {
        params.chatml = true;
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
    if (arg == "--multiline-input") {
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
    if (arg == "-fa" || arg == "--flash-attn") {
        params.flash_attn = true;
        return true;
    }
    if (arg == "--color") {
        params.use_color = true;
        return true;
    }
    if (arg == "--mlock") {
        params.use_mlock = true;
        return true;
    }
    if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_gpu_layers = std::stoi(argv[i]);
        if (params.n_gpu_layers <= 0)
            FLAG_gpu = LLAMAFILE_GPU_DISABLE;
        return true;
    }
    if (arg == "--gpu-layers-draft" || arg == "-ngld" || arg == "--n-gpu-layers-draft") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_gpu_layers_draft = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--main-gpu" || arg == "-mg") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.main_gpu = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--split-mode" || arg == "-sm") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        std::string arg_next = argv[i];
        if (arg_next == "none") {
            params.split_mode = LLAMA_SPLIT_MODE_NONE;
        }
        else if (arg_next == "layer") {
            params.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }
        else if (arg_next == "row") {
            params.split_mode = LLAMA_SPLIT_MODE_ROW;
        }
        else {
            invalid_param = true;
            return true;
        }
        return true;
    }
    if (arg == "--tensor-split" || arg == "-ts") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
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
        return true;
    }
    if (arg == "--no-mmap") {
        params.use_mmap = false;
        return true;
    }
    if (arg == "--numa") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        std::string value(argv[i]);
        /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
        else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
        else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
        else { invalid_param = true; }
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
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.antiprompt.emplace_back(argv[i]);
        return true;
    }
    if (arg == "-ld" || arg == "--logdir") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.logdir = argv[i];

        if (params.logdir.back() != DIRECTORY_SEPARATOR) {
            params.logdir += DIRECTORY_SEPARATOR;
        }
        return true;
    }
    if (arg == "-lcs" || arg == "--lookup-cache-static") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.lookup_cache_static = argv[i];
        return true;
    }
    if (arg == "-lcd" || arg == "--lookup-cache-dynamic") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.lookup_cache_dynamic = argv[i];
        return true;
    }
    if (arg == "--save-all-logits" || arg == "--kl-divergence-base") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.logits_file = argv[i];
        return true;
    }
    if (arg == "--perplexity" || arg == "--all-logits") {
        params.logits_all = true;
        return true;
    }
    if (arg == "--ppl-stride") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.ppl_stride = std::stoi(argv[i]);
        return true;
    }
    if (arg == "-ptc" || arg == "--print-token-count") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.n_print = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--check-tensors") {
        params.check_tensors = true;
        return true;
    }
    if (arg == "--ppl-output-type") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.ppl_output_type = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--hellaswag") {
        params.hellaswag = true;
        return true;
    }
    if (arg == "--hellaswag-tasks") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.hellaswag_tasks = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--winogrande") {
        params.winogrande = true;
        return true;
    }
    if (arg == "--winogrande-tasks") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.winogrande_tasks = std::stoi(argv[i]);
        return true;
    }
    if (arg == "--multiple-choice") {
        params.multiple_choice = true;
        return true;
    }
    if (arg == "--multiple-choice-tasks") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
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
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
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
    if (arg == "-h" || arg == "--help") {
        gpt_print_usage(argc, argv, gpt_params());
        exit(0);
    }
    if (arg == "--version") {
        fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
        fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
        exit(0);
    }
    if (arg == "--random-prompt") {
        params.random_prompt = true;
        return true;
    }
    if (arg == "--in-prefix-bos") {
        params.input_prefix_bos = true;
        return true;
    }
    if (arg == "--in-prefix") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.input_prefix = argv[i];
        return true;
    }
    if (arg == "--in-suffix") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        params.input_suffix = argv[i];
        return true;
    }
    if (arg == "--grammar") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.grammar = argv[i];
        return true;
    }
    if (arg == "--grammar-file") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
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
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        sparams.grammar = json_schema_to_grammar(json::parse(argv[i]));
        return true;
    }
    if (arg == "--override-kv") {
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
        if (!parse_kv_override(argv[i], params.kv_overrides)) {
            fprintf(stderr, "error: Invalid type for KV override: %s\n", argv[i]);
            invalid_param = true;
            return true;
        }
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
        if (++i >= argc) {
            invalid_param = true;
            return true;
        }
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

void gpt_params_handle_model_default(gpt_params & params) {
    if (!params.hf_repo.empty()) {
        // short-hand to avoid specifying --hf-file -> default it to --model
        if (params.hf_file.empty()) {
            if (params.model.empty()) {
                throw std::invalid_argument("error: --hf-repo requires either --hf-file or --model\n");
            }
            params.hf_file = params.model;
        } else if (params.model.empty()) {
            params.model = "models/" + string_split(params.hf_file, '/').back();
        }
    } else if (!params.model_url.empty()) {
        if (params.model.empty()) {
            auto f = string_split(params.model_url, '#').front();
            f = string_split(f, '?').front();
            f = string_split(f, '/').back();
            params.model =  "models/" + f;
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
    }

    if (invalid_param) {
        throw std::invalid_argument("error: invalid parameter for argument: " + arg);
    }

    if (params.prompt_cache_all &&
            (params.interactive || params.interactive_first ||
             params.instruct)) {

        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    gpt_params_handle_model_default(params);

    if (params.escape) {
        process_escapes(params.prompt);
        process_escapes(params.input_prefix);
        process_escapes(params.input_suffix);
        process_escapes(sparams.cfg_negative_prompt);
        for (auto & antiprompt : params.antiprompt) {
            process_escapes(antiprompt);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    params.n_gpu_layers = llamafile_gpu_layers(params.n_gpu_layers);

    return true;
}

void gpt_print_usage(int /*argc*/, char ** argv, const gpt_params & params) {
    const llama_sampling_params & sparams = params.sparams;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto sampler_type : sparams.samplers_sequence) {
        sampler_type_chars += static_cast<char>(sampler_type);
        sampler_type_names += sampler_type_to_name_string(sampler_type) + ";";
    }
    sampler_type_names.pop_back();

    printf("\n");
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  --version             show version and build info\n");
    printf("  -i, --interactive     run in interactive mode\n");
    printf("  --interactive-first   run in interactive mode and wait for input right away\n");
    printf("  -ins, --instruct      run in instruction mode (use with Alpaca models)\n");
    printf("  -cml, --chatml        run in chatml mode (use with ChatML-compatible models)\n");
    printf("  --multiline-input     allows you to write or paste multiple lines without ending each in '\\'\n");
    printf("  -r PROMPT, --reverse-prompt PROMPT\n");
    printf("                        halt generation at PROMPT, return control in interactive mode\n");
    printf("                        (can be specified more than once for multiple prompts).\n");
    printf("  --color               colorise output to distinguish prompt and user input from generations\n");
    printf("  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)\n");
    printf("  -t N, --threads N     number of threads to use during generation (default: %d)\n", params.n_threads);
    printf("  -tb N, --threads-batch N\n");
    printf("                        number of threads to use during batch and prompt processing (default: same as --threads)\n");
    printf("  -td N, --threads-draft N");
    printf("                        number of threads to use during generation (default: same as --threads)\n");
    printf("  -tbd N, --threads-batch-draft N\n");
    printf("                        number of threads to use during batch and prompt processing (default: same as --threads-draft)\n");
    printf("  -p PROMPT, --prompt PROMPT\n");
    printf("                        prompt to start generation with (default: empty)\n");
    printf("  -e, --escape          process prompt escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)\n");
    printf("  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)\n");
    printf("  --prompt-cache-all    if specified, saves user input and generations to cache as well.\n");
    printf("                        not supported with --interactive or other interactive options\n");
    printf("  --prompt-cache-ro     if specified, uses the prompt cache but does not update it.\n");
    printf("  --random-prompt       start with a randomized prompt.\n");
    printf("  --in-prefix-bos       prefix BOS to user inputs, preceding the `--in-prefix` string\n");
    printf("  --in-prefix STRING    string to prefix user inputs with (default: empty)\n");
    printf("  --in-suffix STRING    string to suffix after user inputs with (default: empty)\n");
    printf("  -f FNAME, --file FNAME\n");
    printf("                        prompt file to start generation.\n");
    printf("  -bf FNAME, --binary-file FNAME\n");
    printf("                        binary file containing multiple choice tasks.\n");
    printf("  -n N, --n-predict N   number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)\n", params.n_predict);
    printf("  -c N, --ctx-size N    size of the prompt context (default: %d, 0 = loaded from model)\n", params.n_ctx);
    printf("  -b N, --batch-size N  logical maximum batch size (default: %d)\n", params.n_batch);
    printf("  -ub N, --ubatch-size N\n");
    printf("                        physical maximum batch size (default: %d)\n", params.n_ubatch);
    printf("  --samplers            samplers that will be used for generation in the order, separated by \';\'\n");
    printf("                        (default: %s)\n", sampler_type_names.c_str());
    printf("  --sampling-seq        simplified sequence for samplers that will be used (default: %s)\n", sampler_type_chars.c_str());
    printf("  --top-k N             top-k sampling (default: %d, 0 = disabled)\n", sparams.top_k);
    printf("  --top-p N             top-p sampling (default: %.1f, 1.0 = disabled)\n", (double)sparams.top_p);
    printf("  --min-p N             min-p sampling (default: %.1f, 0.0 = disabled)\n", (double)sparams.min_p);
    printf("  --tfs N               tail free sampling, parameter z (default: %.1f, 1.0 = disabled)\n", (double)sparams.tfs_z);
    printf("  --typical N           locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)\n", (double)sparams.typical_p);
    printf("  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)\n", sparams.penalty_last_n);
    printf("  --repeat-penalty N    penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)\n", (double)sparams.penalty_repeat);
    printf("  --presence-penalty N  repeat alpha presence penalty (default: %.1f, 0.0 = disabled)\n", (double)sparams.penalty_present);
    printf("  --frequency-penalty N repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)\n", (double)sparams.penalty_freq);
    printf("  --dynatemp-range N    dynamic temperature range (default: %.1f, 0.0 = disabled)\n", (double)sparams.dynatemp_range);
    printf("  --dynatemp-exp N      dynamic temperature exponent (default: %.1f)\n", (double)sparams.dynatemp_exponent);
    printf("  --mirostat N          use Mirostat sampling.\n");
    printf("                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n");
    printf("                        (default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)\n", sparams.mirostat);
    printf("  --mirostat-lr N       Mirostat learning rate, parameter eta (default: %.1f)\n", (double)sparams.mirostat_eta);
    printf("  --mirostat-ent N      Mirostat target entropy, parameter tau (default: %.1f)\n", (double)sparams.mirostat_tau);
    printf("  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS\n");
    printf("                        modifies the likelihood of token appearing in the completion,\n");
    printf("                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n");
    printf("                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'\n");
    printf("  --grammar GRAMMAR     BNF-like grammar to constrain generations (see samples in grammars/ dir)\n");
    printf("  --grammar-file FNAME  file to read grammar from\n");
    printf("  -j SCHEMA, --json-schema SCHEMA\n");
    printf("                        JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object.\n");
    printf("                        For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead\n");
    printf("  --cfg-negative-prompt PROMPT\n");
    printf("                        negative prompt to use for guidance. (default: empty)\n");
    printf("  --cfg-negative-prompt-file FNAME\n");
    printf("                        negative prompt file to use for guidance. (default: empty)\n");
    printf("  --cfg-scale N         strength of guidance (default: %f, 1.0 = disable)\n", sparams.cfg_scale);
    printf("  --rope-scaling {none,linear,yarn}\n");
    printf("                        RoPE frequency scaling method, defaults to linear unless specified by the model\n");
    printf("  --rope-scale N        RoPE context scaling factor, expands context by a factor of N\n");
    printf("  --rope-freq-base N    RoPE base frequency, used by NTK-aware scaling (default: loaded from model)\n");
    printf("  --rope-freq-scale N   RoPE frequency scaling factor, expands context by a factor of 1/N\n");
    printf("  --yarn-orig-ctx N     YaRN: original context size of model (default: 0 = model training context size)\n");
    printf("  --yarn-ext-factor N   YaRN: extrapolation mix factor (default: 1.0, 0.0 = full interpolation)\n");
    printf("  --yarn-attn-factor N  YaRN: scale sqrt(t) or attention magnitude (default: 1.0)\n");
    printf("  --yarn-beta-slow N    YaRN: high correction dim or alpha (default: %.1f)\n", params.yarn_beta_slow);
    printf("  --yarn-beta-fast N    YaRN: low correction dim or beta (default: %.1f)\n", params.yarn_beta_fast);
    printf("  --pooling {none,mean,cls}\n");
    printf("                        pooling type for embeddings, use model default if unspecified\n");
    printf("  -dt N, --defrag-thold N\n");
    printf("                        KV cache defragmentation threshold (default: %.1f, < 0 - disabled)\n", params.defrag_thold);
    printf("  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)\n");
    printf("  --penalize-nl         penalize newline tokens\n");
    printf("  --temp N              temperature (default: %.1f)\n", (double)sparams.temp);
    printf("  --all-logits          return logits for all tokens in the batch (default: disabled)\n");
    printf("  --hellaswag           compute HellaSwag score over random tasks from datafile supplied with -f\n");
    printf("  --hellaswag-tasks N   number of tasks to use when computing the HellaSwag score (default: %zu)\n", params.hellaswag_tasks);
    printf("  --winogrande          compute Winogrande score over random tasks from datafile supplied with -f\n");
    printf("  --winogrande-tasks N  number of tasks to use when computing the Winogrande score (default: %zu)\n", params.winogrande_tasks);
    printf("  --multiple-choice     compute multiple choice score over random tasks from datafile supplied with -f\n");
    printf("  --multiple-choice-tasks N number of tasks to use when computing the multiple choice score (default: %zu)\n", params.winogrande_tasks);
    printf("  --kl-divergence       computes KL-divergence to logits provided via --kl-divergence-base\n");
    printf("  --keep N              number of tokens to keep from the initial prompt (default: %d, -1 = all)\n", params.n_keep);
    printf("  --draft N             number of tokens to draft for speculative decoding (default: %d)\n", params.n_draft);
    printf("  --chunks N            max number of chunks to process (default: %d, -1 = all)\n", params.n_chunks);
    printf("  -np N, --parallel N   number of parallel sequences to decode (default: %d)\n", params.n_parallel);
    printf("  -ns N, --sequences N  number of sequences to decode (default: %d)\n", params.n_sequences);
    printf("  -ps N, --p-split N    speculative decoding split probability (default: %.1f)\n", (double)params.p_split);
    printf("  -cb, --cont-batching  enable continuous batching (a.k.a dynamic batching) (default: disabled)\n");
    printf("  -fa, --flash-attn     enable Flash Attention (default: %s)\n", params.flash_attn ? "enabled" : "disabled");
    printf("  --mmproj MMPROJ_FILE  path to a multimodal projector file for LLaVA. see examples/llava/README.md\n");
    printf("  --image IMAGE_FILE    path to an image file. use with multimodal models. Specify multiple times for batching\n");
    if (llama_supports_mlock()) {
        printf("  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_supports_mmap()) {
        printf("  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
    printf("  --numa TYPE           attempt optimizations that help on some NUMA systems\n");
    printf("                          - distribute: spread execution evenly over all nodes\n");
    printf("                          - isolate: only spawn threads on CPUs on the node that execution started on\n");
    printf("                          - numactl: use the CPU map provided by numactl\n");
    printf("                        if run without this previously, it is recommended to drop the system page cache before using this\n");
    printf("                        see https://github.com/ggerganov/llama.cpp/issues/1437\n");
    // if (llama_supports_gpu_offload()) {
        printf("  -ngl N, --n-gpu-layers N\n");
        printf("                        number of layers to store in VRAM\n");
        printf("  -ngld N, --n-gpu-layers-draft N\n");
        printf("                        number of layers to store in VRAM for the draft model\n");
        printf("  -sm SPLIT_MODE, --split-mode SPLIT_MODE\n");
        printf("                        how to split the model across multiple GPUs, one of:\n");
        printf("                          - none: use one GPU only\n");
        printf("                          - layer (default): split layers and KV across GPUs\n");
        printf("                          - row: split rows across GPUs\n");
        printf("  -ts SPLIT, --tensor-split SPLIT\n");
        printf("                        fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1\n");
        printf("  -mg i, --main-gpu i   the GPU to use for the model (with split-mode = none),\n");
        printf("                        or for intermediate results and KV (with split-mode = row) (default: %d)\n", params.main_gpu);
    // }
    printf("  --verbose-prompt      print a verbose prompt before generation (default: %s)\n", params.verbose_prompt ? "true" : "false");
    printf("  --no-display-prompt   don't print prompt at generation (default: %s)\n", !params.display_prompt ? "true" : "false");
    printf("  -gan N, --grp-attn-n N\n");
    printf("                        group-attention factor (default: %d)\n", params.grp_attn_n);
    printf("  -gaw N, --grp-attn-w N\n");
    printf("                        group-attention width (default: %.1f)\n", (double)params.grp_attn_w);
    printf("  -dkvc, --dump-kv-cache\n");
    printf("                        verbose print of the KV cache\n");
    printf("  -nkvo, --no-kv-offload\n");
    printf("                        disable KV offload\n");
    printf("  -ctk TYPE, --cache-type-k TYPE\n");
    printf("                        KV cache data type for K (default: %s)\n", params.cache_type_k.c_str());
    printf("  -ctv TYPE, --cache-type-v TYPE\n");
    printf("                        KV cache data type for V (default: %s)\n", params.cache_type_v.c_str());
    printf("  --simple-io           use basic IO for better compatibility in subprocesses and limited consoles\n");
    printf("  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
    printf("  --lora-scaled FNAME S apply LoRA adapter with user defined scaling S (implies --no-mmap)\n");
    printf("  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
    printf("  --control-vector FNAME\n");
    printf("                        add a control vector\n");
    printf("  --control-vector-scaled FNAME S\n");
    printf("                        add a control vector with user defined scaling S\n");
    printf("  --control-vector-layer-range START END\n");
    printf("                        layer range to apply the control vector(s) to, start and end inclusive\n");
    printf("  -m FNAME, --model FNAME\n");
    printf("                        model path (default: models/$filename with filename from --hf-file or --model-url if set, otherwise %s)\n", DEFAULT_MODEL_PATH);
    printf("  -md FNAME, --model-draft FNAME\n");
    printf("                        draft model for speculative decoding (default: unused)\n");
    printf("  -mu MODEL_URL, --model-url MODEL_URL\n");
    printf("                        model download url (default: unused)\n");
    printf("  -hfr REPO, --hf-repo REPO\n");
    printf("                        Hugging Face model repository (default: unused)\n");
    printf("  -hff FILE, --hf-file FILE\n");
    printf("                        Hugging Face model file (default: unused)\n");
    printf("  -ld LOGDIR, --logdir LOGDIR\n");
    printf("                        path under which to save YAML logs (no logging if unset)\n");
    printf("  -lcs FNAME, --lookup-cache-static FNAME\n");
    printf("                        path to static lookup cache to use for lookup decoding (not updated by generation)\n");
    printf("  -lcd FNAME, --lookup-cache-dynamic FNAME\n");
    printf("                        path to dynamic lookup cache to use for lookup decoding (updated by generation)\n");
    printf("  --override-kv KEY=TYPE:VALUE\n");
    printf("                        advanced option to override model metadata by key. may be specified multiple times.\n");
    printf("                        types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false\n");
    printf("  -ptc N, --print-token-count N\n");
    printf("                        print token count every N tokens (default: %d)\n", params.n_print);
    printf("  --check-tensors       check model tensor data for invalid values\n");
    printf("\n");
#ifndef LOG_DISABLE_LOGS
    log_print_usage();
#endif // LOG_DISABLE_LOGS
}

std::string get_system_info(const gpt_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.n_threads;
    if (params.n_threads_batch != -1) {
        os << " (n_threads_batch = " << params.n_threads_batch << ")";
    }
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();

    return os.str();
}

std::string gpt_random_prompt(std::mt19937 & rng) {
    const int r = rng() % 10;
    switch (r) {
        case 0: return "So";
        case 1: return "Once upon a time";
        case 2: return "When";
        case 3: return "The";
        case 4: return "After";
        case 5: return "If";
        case 6: return "import";
        case 7: return "He";
        case 8: return "She";
        case 9: return "They";
    }

    GGML_UNREACHABLE();
}

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool validate_file_name(const std::string & filename) {
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

std::vector<llama_sampler_type> sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, llama_sampler_type> sampler_canonical_name_map {
        {"top_k",       llama_sampler_type::TOP_K},
        {"top_p",       llama_sampler_type::TOP_P},
        {"typical_p",   llama_sampler_type::TYPICAL_P},
        {"min_p",       llama_sampler_type::MIN_P},
        {"tfs_z",       llama_sampler_type::TFS_Z},
        {"temperature", llama_sampler_type::TEMPERATURE}
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, llama_sampler_type> sampler_alt_name_map {
        {"top-k",       llama_sampler_type::TOP_K},
        {"top-p",       llama_sampler_type::TOP_P},
        {"nucleus",     llama_sampler_type::TOP_P},
        {"typical-p",   llama_sampler_type::TYPICAL_P},
        {"typical",     llama_sampler_type::TYPICAL_P},
        {"min-p",       llama_sampler_type::MIN_P},
        {"tfs-z",       llama_sampler_type::TFS_Z},
        {"tfs",         llama_sampler_type::TFS_Z},
        {"temp",        llama_sampler_type::TEMPERATURE}
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names.size());
    for (const auto & name : names)
    {
        auto sampler_item = sampler_canonical_name_map.find(name);
        if (sampler_item != sampler_canonical_name_map.end())
        {
            sampler_types.push_back(sampler_item->second);
        }
        else
        {
            if (allow_alt_names)
            {
                sampler_item = sampler_alt_name_map.find(name);
                if (sampler_item != sampler_alt_name_map.end())
                {
                    sampler_types.push_back(sampler_item->second);
                }
            }
        }
    }
    return sampler_types;
}

std::vector<llama_sampler_type> sampler_types_from_chars(const std::string & names_string) {
    std::unordered_map<char, llama_sampler_type> sampler_name_map {
        {'k', llama_sampler_type::TOP_K},
        {'p', llama_sampler_type::TOP_P},
        {'y', llama_sampler_type::TYPICAL_P},
        {'m', llama_sampler_type::MIN_P},
        {'f', llama_sampler_type::TFS_Z},
        {'t', llama_sampler_type::TEMPERATURE}
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names_string.size());
    for (const auto & c : names_string) {
        const auto sampler_item = sampler_name_map.find(c);
        if (sampler_item != sampler_name_map.end()) {
            sampler_types.push_back(sampler_item->second);
        }
    }
    return sampler_types;
}

std::string sampler_type_to_name_string(llama_sampler_type sampler_type) {
    switch (sampler_type) {
        case llama_sampler_type::TOP_K:       return "top_k";
        case llama_sampler_type::TFS_Z:       return "tfs_z";
        case llama_sampler_type::TYPICAL_P:   return "typical_p";
        case llama_sampler_type::TOP_P:       return "top_p";
        case llama_sampler_type::MIN_P:       return "min_p";
        case llama_sampler_type::TEMPERATURE: return "temperature";
        default : return "";
    }
}

//
// Model utils
//

struct llama_model_params llama_model_params_from_gpt_params(const gpt_params & params) {
    auto mparams = llama_model_default_params();

    if (params.n_gpu_layers != -1) {
        mparams.n_gpu_layers = params.n_gpu_layers;
    }
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
    cparams.defrag_thold      = params.defrag_thold;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.flash_attn        = params.flash_attn;

    cparams.type_k = kv_cache_type_from_str(params.cache_type_k);
    cparams.type_v = kv_cache_type_from_str(params.cache_type_v);

    return cparams;
}

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

#ifdef LLAMA_USE_CURL

static bool starts_with(const std::string & str, const std::string & prefix) {
    // While we wait for C++20's std::string::starts_with...
    return str.rfind(prefix, 0) == 0;
}

static bool llama_download_file(const std::string & url, const std::string & path) {

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
                if (metadata.contains("url") && metadata["url"].is_string()) {
                    auto previous_url = metadata["url"].get<std::string>();
                    if (previous_url != url) {
                        fprintf(stderr, "%s: Model URL mismatch: %s != %s\n", __func__, url.c_str(), previous_url.c_str());
                        return false;
                    }
                }
                if (metadata.contains("etag") && metadata["etag"].is_string()) {
                    etag = metadata["etag"];
                }
                if (metadata.contains("lastModified") && metadata["lastModified"].is_string()) {
                    last_modified = metadata["lastModified"];
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
        std::unique_ptr<FILE, decltype(&fclose)> outfile(fopen(path_temporary.c_str(), "wb"), fclose);
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
        const struct llama_model_params & params) {
    // Basic validation of the model_url
    if (!model_url || strlen(model_url) == 0) {
        fprintf(stderr, "%s: invalid model_url\n", __func__);
        return NULL;
    }

    if (!llama_download_file(model_url, path_model)) {
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
            futures_download.push_back(std::async(std::launch::async, [&split_prefix, &split_url_prefix, &n_split](int download_idx) -> bool {
                char split_path[PATH_MAX] = {0};
                llama_split_path(split_path, sizeof(split_path), split_prefix, download_idx, n_split);

                char split_url[LLAMA_CURL_MAX_URL_LENGTH] = {0};
                llama_split_path(split_url, sizeof(split_url), split_url_prefix, download_idx, n_split);

                return llama_download_file(split_url, split_path);
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

    return llama_load_model_from_url(model_url.c_str(), path_model, params);
}

#else

struct llama_model * llama_load_model_from_url(
        const char * /*model_url*/,
        const char * /*path_model*/,
        const struct llama_model_params & /*params*/) {
    fprintf(stderr, "%s: llama.cpp built without libcurl, downloading from an url not supported.\n", __func__);
    return nullptr;
}

struct llama_model * llama_load_model_from_hf(
        const char * /*repo*/,
        const char * /*model*/,
        const char * /*path_model*/,
        const struct llama_model_params & /*params*/) {
    fprintf(stderr, "%s: llama.cpp built without libcurl, downloading from Hugging Face not supported.\n", __func__);
    return nullptr;
}

#endif // LLAMA_USE_CURL

std::tuple<struct llama_model *, struct llama_context *> llama_init_from_gpt_params(gpt_params & params) {
    auto mparams = llama_model_params_from_gpt_params(params);

    llama_model * model = nullptr;

    if (!params.hf_repo.empty() && !params.hf_file.empty()) {
        model = llama_load_model_from_hf(params.hf_repo.c_str(), params.hf_file.c_str(), params.model.c_str(), mparams);
    } else if (!params.model_url.empty()) {
        model = llama_load_model_from_url(params.model_url.c_str(), params.model.c_str(), mparams);
    } else {
        model = llama_load_model_from_file(params.model.c_str(), mparams);
    }

    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return std::make_tuple(nullptr, nullptr);
    }

    auto cparams = llama_context_params_from_gpt_params(params);

    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return std::make_tuple(nullptr, nullptr);
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_n_layer(model);

        const auto cvec = llama_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            llama_free(lctx);
            llama_free_model(model);
            return std::make_tuple(nullptr, nullptr);
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
            return std::make_tuple(nullptr, nullptr);
        }
    }

    for (unsigned int i = 0; i < params.lora_adapter.size(); ++i) {
        const std::string & lora_adapter = std::get<0>(params.lora_adapter[i]);
        float lora_scale = std::get<1>(params.lora_adapter[i]);
        int err = llama_model_apply_lora_from_file(model,
                                             lora_adapter.c_str(),
                                             lora_scale,
                                             ((i > 0) || params.lora_base.empty())
                                                ? NULL
                                                : params.lora_base.c_str(),
                                             params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            llama_free(lctx);
            llama_free_model(model);
            return std::make_tuple(nullptr, nullptr);
        }
    }

    if (params.ignore_eos) {
        params.sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
    }

    if (params.warmup) {
        LOG("warming up the model with an empty run\n");

        std::vector<llama_token> tmp = { llama_token_bos(model), llama_token_eos(model), };
        llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch), 0, 0));
        llama_kv_cache_clear(lctx);
        llama_synchronize(lctx);
        llama_reset_timings(lctx);
    }

    return std::make_tuple(model, lctx);
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
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

std::string llama_detokenize_spm(llama_context * ctx, const std::vector<llama_token> & tokens) {
    const llama_token bos_id = llama_token_bos(llama_get_model(ctx));

    std::string piece;
    std::string result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        piece = llama_token_to_piece(ctx, tokens[i]);

        // remove the leading space of the first non-BOS token
        if (((tokens[0] == bos_id && i == 1) || (tokens[0] != bos_id && i == 0)) && piece[0] == ' ') {
            piece = piece.substr(1);
        }

        result += piece;
    }

    return result;
}

std::string llama_detokenize_bpe(llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::string piece;
    std::string result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        piece = llama_token_to_piece(ctx, tokens[i]);

        result += piece;
    }

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return result;
}

bool llama_should_add_bos_token(const llama_model * model) {
    const int add_bos = llama_add_bos_token(model);

    return add_bos != -1 ? bool(add_bos) : (llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM);
}

//
// YAML utils
//

// returns true if successful, false otherwise
bool create_directory_with_parents(const std::string & path) {
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

void dump_vector_float_yaml(FILE * stream, const char * prop_name, const std::vector<float> & data) {
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

void dump_vector_int_yaml(FILE * stream, const char * prop_name, const std::vector<int> & data) {
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

void dump_string_yaml_multiline(FILE * stream, const char * prop_name, const char * data) {
    std::string data_str(data == NULL ? "" : data);

    if (data_str.empty()) {
        fprintf(stream, "%s:\n", prop_name);
        return;
    }

    size_t pos_start = 0;
    size_t pos_found = 0;

    if (!data_str.empty() && (std::isspace(data_str[0]) || std::isspace(data_str.back()))) {
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

std::string get_sortable_timestamp() {
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

void dump_non_result_info_yaml(FILE * stream, const gpt_params & params, const llama_context * lctx,
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
    fprintf(stream, "cpu_has_clblast: %s\n",     ggml_cpu_has_clblast()     ? "true" : "false");
    fprintf(stream, "cpu_has_kompute: %s\n",     ggml_cpu_has_kompute()     ? "true" : "false");
    fprintf(stream, "cpu_has_fma: %s\n",         ggml_cpu_has_fma()         ? "true" : "false");
    fprintf(stream, "cpu_has_gpublas: %s\n",     ggml_cpu_has_gpublas()     ? "true" : "false");
    fprintf(stream, "cpu_has_neon: %s\n",        ggml_cpu_has_neon()        ? "true" : "false");
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
    dump_string_yaml_multiline(stream, "cfg_negative_prompt", sparams.cfg_negative_prompt.c_str());
    fprintf(stream, "cfg_scale: %f # default: 1.0\n", sparams.cfg_scale);
    fprintf(stream, "chunks: %d # default: -1 (unlimited)\n", params.n_chunks);
    fprintf(stream, "color: %s # default: false\n", params.use_color ? "true" : "false");
    fprintf(stream, "ctx_size: %d # default: 512\n", params.n_ctx);
    fprintf(stream, "escape: %s # default: false\n", params.escape ? "true" : "false");
    fprintf(stream, "file: # never logged, see prompt instead. Can still be specified for input.\n");
    fprintf(stream, "frequency_penalty: %f # default: 0.0 \n", sparams.penalty_freq);
    dump_string_yaml_multiline(stream, "grammar", sparams.grammar.c_str());
    fprintf(stream, "grammar-file: # never logged, see grammar instead. Can still be specified for input.\n");
    fprintf(stream, "hellaswag: %s # default: false\n", params.hellaswag ? "true" : "false");
    fprintf(stream, "hellaswag_tasks: %zu # default: 400\n", params.hellaswag_tasks);

    const auto logit_bias_eos = sparams.logit_bias.find(llama_token_eos(llama_get_model(lctx)));
    const bool ignore_eos = logit_bias_eos != sparams.logit_bias.end() && logit_bias_eos->second == -INFINITY;
    fprintf(stream, "ignore_eos: %s # default: false\n", ignore_eos ? "true" : "false");

    dump_string_yaml_multiline(stream, "in_prefix", params.input_prefix.c_str());
    fprintf(stream, "in_prefix_bos: %s # default: false\n", params.input_prefix_bos ? "true" : "false");
    dump_string_yaml_multiline(stream, "in_suffix", params.input_prefix.c_str());
    fprintf(stream, "instruct: %s # default: false\n", params.instruct ? "true" : "false");
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
    for (std::tuple<std::string, float> la : params.lora_adapter) {
        if (std::get<1>(la) != 1.0f) {
            continue;
        }
        fprintf(stream, "  - %s\n", std::get<0>(la).c_str());
    }
    fprintf(stream, "lora_scaled:\n");
    for (std::tuple<std::string, float> la : params.lora_adapter) {
        if (std::get<1>(la) == 1.0f) {
            continue;
        }
        fprintf(stream, "  - %s: %f\n", std::get<0>(la).c_str(), std::get<1>(la));
    }
    fprintf(stream, "lora_base: %s\n", params.lora_base.c_str());
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
    dump_string_yaml_multiline(stream, "prompt", params.prompt.c_str());
    fprintf(stream, "prompt_cache: %s\n", params.path_prompt_cache.c_str());
    fprintf(stream, "prompt_cache_all: %s # default: false\n", params.prompt_cache_all ? "true" : "false");
    fprintf(stream, "prompt_cache_ro: %s # default: false\n", params.prompt_cache_ro ? "true" : "false");
    dump_vector_int_yaml(stream, "prompt_tokens", prompt_tokens);
    fprintf(stream, "random_prompt: %s # default: false\n", params.random_prompt ? "true" : "false");
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
    dump_vector_float_yaml(stream, "tensor_split", tensor_split_vector);

    fprintf(stream, "tfs: %f # default: 1.0\n", sparams.tfs_z);
    fprintf(stream, "threads: %d # default: %u\n", params.n_threads, std::thread::hardware_concurrency());
    fprintf(stream, "top_k: %d # default: 40\n", sparams.top_k);
    fprintf(stream, "top_p: %f # default: 0.95\n", sparams.top_p);
    fprintf(stream, "min_p: %f # default: 0.0\n", sparams.min_p);
    fprintf(stream, "typical_p: %f # default: 1.0\n", sparams.typical_p);
    fprintf(stream, "verbose_prompt: %s # default: false\n", params.verbose_prompt ? "true" : "false");
    fprintf(stream, "display_prompt: %s # default: true\n", params.display_prompt ? "true" : "false");
}

//
// KV cache utils
//

void dump_kv_cache_view(const llama_kv_cache_view & view, int row_size) {
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

void dump_kv_cache_view_seqs(const llama_kv_cache_view & view, int row_size) {
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

void llama_embd_normalize(const float * inp, float * out, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += inp[i] * inp[i];
    }
    sum = sqrt(sum);

    const float norm = sum > 0.0 ? 1.0f / sum : 0.0f;

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

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static llama_control_vector_data llama_control_vector_load_one(const llama_control_vector_load_info & load_info) {
    int32_t n_tensors;

    size_t n_bytes = 0;

    uint32_t max_direction_layer = 0;

    llama_control_vector_data result = { -1, {} };

    // calculate size of ctx needed for tensors, ensure tensors are f32, and find max layer
    {
        struct ggml_init_params meta_params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * meta_ctx = ggml_init(meta_params);
        struct gguf_init_params meta_gguf_params = {
            /* .no_alloc = */ true,
            /* .ctx      = */ &meta_ctx,
        };
        struct llamafile * file = llamafile_open_gguf(load_info.fname.c_str(), "rb");
        if (!file) {
            perror(load_info.fname.c_str());
            ggml_free(meta_ctx);
            return result;
        }
        struct gguf_context * meta_ctx_gguf = gguf_init_from_file(file, meta_gguf_params);
        if (!meta_ctx_gguf) {
            fprintf(stderr, "%s: failed to load control vector from %s\n", __func__, load_info.fname.c_str());
            llamafile_close(file);
            ggml_free(meta_ctx);
            return result;
        }

        n_tensors = gguf_get_n_tensors(meta_ctx_gguf);
        for (int i = 0; i < n_tensors; i++) {
            std::string name = gguf_get_tensor_name(meta_ctx_gguf, i);

            // split on '.'
            size_t dotpos = name.find('.');
            if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
                try {
                    uint32_t layer = std::stoi(name.substr(dotpos + 1));
                    if (layer == 0) {
                        fprintf(stderr, "%s: direction tensor invalid in %s\n", __func__, load_info.fname.c_str());
                        ggml_free(meta_ctx);
                        gguf_free(meta_ctx_gguf);
                        llamafile_close(file);
                        return result;
                    }
                    if (layer > max_direction_layer) {
                        max_direction_layer = layer;
                    }
                } catch (...) {
                    fprintf(stderr, "%s: direction tensor invalid in %s\n", __func__, load_info.fname.c_str());
                    ggml_free(meta_ctx);
                    gguf_free(meta_ctx_gguf);
                    llamafile_close(file);
                    return result;
                }
            }

            struct ggml_tensor * tensor_meta = ggml_get_tensor(meta_ctx, name.c_str());
            if (tensor_meta->type != GGML_TYPE_F32 || ggml_n_dims(tensor_meta) != 1) {
                fprintf(stderr, "%s: direction tensor invalid in %s\n", __func__, load_info.fname.c_str());
                ggml_free(meta_ctx);
                gguf_free(meta_ctx_gguf);
                llamafile_close(file);
                return result;
            }
            if (result.n_embd == -1) {
                result.n_embd = ggml_nelements(tensor_meta);
            } else if (ggml_nelements(tensor_meta) != result.n_embd) {
                fprintf(stderr, "%s: direction tensor sizes mismatched in %s\n", __func__, load_info.fname.c_str());
                ggml_free(meta_ctx);
                gguf_free(meta_ctx_gguf);
                llamafile_close(file);
                return result;
            }
            n_bytes += ggml_nbytes(tensor_meta);
        }
        ggml_free(meta_ctx);
        gguf_free(meta_ctx_gguf);
        llamafile_close(file);
    }

    if (n_tensors == 0) {
        fprintf(stderr, "%s: no direction tensors found in %s\n", __func__, load_info.fname.c_str());
        return result;
    }

    // load and scale tensors into final control vector context
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ ggml_tensor_overhead() * n_tensors + n_bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx,
    };
    struct llamafile * file = llamafile_open_gguf(load_info.fname.c_str(), "rb");
    if (!file) {
        perror(load_info.fname.c_str());
        ggml_free(ctx);
        return result;
    }
    struct gguf_context * ctx_gguf = gguf_init_from_file(file, params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: failed to load control vector from %s\n", __func__, load_info.fname.c_str());
        ggml_free(ctx);
        llamafile_close(file);
        return result;
    }

    // do not store data for layer 0 (it's not used)
    result.data.resize(result.n_embd * max_direction_layer);

    for (uint32_t il = 1; il <= max_direction_layer; il++) {
        const std::string name = "direction." + std::to_string(il);
        const ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());

        float * dst = result.data.data() + result.n_embd * (il - 1);

        if (tensor) {
            const float * src = (const float *) tensor->data;
            for (int j = 0; j < result.n_embd; j++) {
                dst[j] = src[j] * load_info.strength;
            }
        } else {
            for (int j = 0; j < result.n_embd; j++) {
                dst[j] = 0.0f;
            }
        }
    }

    return result;
}

llama_control_vector_data llama_control_vector_load(const std::vector<llama_control_vector_load_info> & load_infos) {
    llama_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = llama_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            return result;
        }
        if (result.n_embd != -1 && (result.n_embd != cur.n_embd || result.data.size() != cur.data.size())) {
            fprintf(stderr, "%s: control vector in %s does not match previous vector dimensions\n", __func__, info.fname.c_str());
            return result;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        fprintf(stderr, "%s: no vectors passed\n", __func__);
    }

    return result;
}
