#include <algorithm>

#include "cmd.h"
#include "llama.cpp/cores.h"
#include "system.h"
#include <cosmo.h>

static const cmd_params cmd_params_defaults = {
    /* model         */ "", // [jart] no default guessing
    /* n_prompt      */ 0,
    /* n_gen         */ 0,
    /* n_batch       */ 2048,
    /* n_ubatch      */ 512,
    /* type_k        */ X86_HAVE(AVX512_BF16) ? GGML_TYPE_BF16 : GGML_TYPE_F16,
    /* type_v        */ X86_HAVE(AVX512_BF16) ? GGML_TYPE_BF16 : GGML_TYPE_F16,
    /* n_threads     */ cpu_get_num_math(),
    /* gpu           */ LLAMAFILE_GPU_AUTO,
    /* n_gpu_layers  */ 9999,
    /* split_mode    */ LLAMA_SPLIT_MODE_NONE,
    /* main_gpu      */ UINT_MAX,
    /* no_kv_offload */ false,
    /* flash_attn    */ false,
    /* tensor_split  */ std::vector<float>(llama_max_devices(), 0.0f),
    /* use_mmap      */ true,
    /* embeddings    */ false,
    /* numa          */ GGML_NUMA_STRATEGY_DISABLED,
    /* reps          */ 1,
    /* verbose       */ false,
    /* plaintext     */ false,
    /* send_results  */ SEND_ASK,
    /* output_format */ CONSOLE,
};

llama_model_params cmd_params::to_llama_mparams() const {
    llama_model_params mparams = llama_model_default_params();

    mparams.n_gpu_layers = n_gpu_layers;
    mparams.split_mode = split_mode;
    mparams.main_gpu = main_gpu;
    mparams.tensor_split = tensor_split.data();
    mparams.use_mmap = use_mmap;

    return mparams;
}

bool cmd_params::equal_mparams(const cmd_params & other) const {
    return model == other.model &&
           n_gpu_layers == other.n_gpu_layers &&
           split_mode == other.split_mode &&
           main_gpu == other.main_gpu && 
           use_mmap == other.use_mmap &&
           tensor_split == other.tensor_split;
}

llama_context_params cmd_params::to_llama_cparams() const {
    llama_context_params cparams = llama_context_default_params();

    cparams.n_ctx = n_prompt + n_gen;
    cparams.n_batch = n_batch;
    cparams.n_ubatch = n_ubatch;
    cparams.type_k = type_k;
    cparams.type_v = type_v;
    cparams.offload_kqv = !no_kv_offload;
    cparams.flash_attn = flash_attn;
    cparams.embeddings = embeddings;

    return cparams;
}

cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params params = cmd_params_defaults;
    std::string arg;
    bool invalid_param = false;
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
            }
            else params.model = argv[i];
        } else if (arg == "-i" || arg == "--gpu-index") {
            if (++i >= argc) {
                invalid_param = true;
            }
            else params.main_gpu = std::stoi(argv[i]);
        } else if (arg == "--list-gpus") {
            FLAG_gpu = LLAMAFILE_GPU_AUTO;
            FLAGS_READY = true;
            list_available_accelerators();  // You'll need to implement this
            exit(0);
        } else if (arg == "-c" || arg == "--cpu") {
            FLAG_gpu = LLAMAFILE_GPU_DISABLE;
            params.n_gpu_layers = 0;
        } else if (arg == "-g" || arg == "--gpu") {
            if (++i >= argc) {
                invalid_param = true;
            }
            else {
                FLAG_gpu = llamafile_gpu_parse(argv[i]);
                if (FLAG_gpu == LLAMAFILE_GPU_ERROR) {
                    fprintf(stderr, "error: invalid --gpu value: %s\n", argv[i]);
                    exit(1);
                }
                params.n_gpu_layers = (FLAG_gpu >= 0) ? 9999 : 0;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_param = true;
            }
            else {
                std::string fmt = argv[i];
                if (fmt == "csv") params.output_format = CSV;
                else if (fmt == "json") params.output_format = JSON;
                else if (fmt == "console") params.output_format = CONSOLE;
                else {
                    invalid_param = true;
                }
            }
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--plaintext") {
            params.plaintext = true;
        } else if (arg == "-y" || arg == "--send-results") {
            params.send_results = SEND_YES;
        } else if (arg == "-n" || arg == "--no-send-results") {
            params.send_results = SEND_NO;
        } else if (arg == "--extended") {
            params.reps = 4;
        } else if (arg == "--long") {
            params.reps = 16;
        } else if (arg == "--reps") {
            if (++i >= argc) {
                invalid_param = true;
            }
            else params.reps = std::max(1, std::stoi(argv[i]));
        } else if (arg == "--recompile") {
            FLAG_recompile = true;
        } else if (arg == "--localscore") {
        } else if (arg[0] == '-') {
            invalid_param = true;
        } else {
            params.model = argv[i];
        }

        if (invalid_param) break;
    }

    if (invalid_param) {
        fprintf(stderr, "%s: invalid parameter for: %s\n", 
                program_invocation_name, arg.c_str());
        exit(1);
    }

    if (params.model.empty()) {
        fprintf(stderr, "%s: missing model file\n", program_invocation_name);
        exit(1);
    }

    // Validate mutually exclusive flags
    if (params.send_results == SEND_YES && params.send_results == SEND_NO) {
        fprintf(stderr, "%s: cannot use both --send-results and --no-send-results\n",
                program_invocation_name);
        exit(1);
    }

    return params;
}

static const char * output_format_str(output_formats format) {
    switch (format) {
        case CSV:      return "csv";
        case JSON:     return "json";
        case CONSOLE:  return "console";
        default: GGML_ASSERT(!"invalid output format");
    }
}

void print_usage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  -m, --model <filename>\n");
    printf("  -c, --cpu                                  disable GPU acceleration (alias for --gpu=disabled)\n");
    printf("  -g, --gpu <auto|amd|apple|nvidia|disabled> (default: \"auto\")\n");
    printf("  -i, --gpu-index <i>                        select GPU by index (default: %d)\n", cmd_params_defaults.main_gpu);
    printf("  --list-gpus                                list available GPUs and exit\n");
    printf("  -o, --output <csv|json|console>            (default: %s)\n", output_format_str(cmd_params_defaults.output_format));
    printf("  -v, --verbose                              verbose output (default: %s)\n", cmd_params_defaults.verbose ? "on" : "off");
    printf("  --plaintext                                plaintext output (default: %s)\n", cmd_params_defaults.plaintext ? "on" : "off");
    printf("  -y, --send-results                         send results without confirmation\n");
    printf("  -n, --no-send-results                      disable sending results\n");
    printf("  -e, --extended                             run 4 reps (shortcut for --reps=4)\n");
    printf("  --long                                     run 16 reps (shortcut for --reps=16)\n");
    printf("  --reps <N>                                 set custom number of repetitions\n");
}