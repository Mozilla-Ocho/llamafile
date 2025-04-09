// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#include <algorithm>
#include <cassert>
// #include <chrono> [jart]
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <map>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <cosmo.h>
#include <dlfcn.h>
#include <libgen.h>
#include <pthread.h>
#include <sys/stat.h>
#include <libc/intrin/x86.h>
#include <libc/sysv/consts/hwcap.h>

#include <sys/utsname.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include "http.h"
#include "powersampler.h"
#include "ascii_digits.h"
#include "system.h"
#include "cmd.h"
#include "benchmark.h"
#include "printer.h"
#include "utils.h"

#include "localscore.h"

#include "llama.cpp/cores.h"
#include "llama.cpp/ggml.h"
#include "llama.cpp/ggml-metal.h"
#include "llama.cpp/llama.h"
#include "llama.cpp/string.h"
#include "llama.cpp/common.h"
#include "llama.cpp/ggml-cuda.h"

#include "llamafile/llamafile.h"
#include "llamafile/compute.h"
#include "llamafile/json.h"

using jt::Json;

const std::string BASE_LOCALSCORE_URL = "https://www.localscore.ai";

const std::string test::build_commit = LLAMA_COMMIT;
const int         test::build_number = LLAMA_BUILD_NUMBER;
const bool        test::cuda         = false; // !!ggml_cpu_has_cuda(); // [jart]
const bool        test::opencl       = false; // !!ggml_cpu_has_clblast(); // [jart]
const bool        test::vulkan       = false; // !!ggml_cpu_has_vulkan(); // [jart]
const bool        test::kompute      = false; // !!ggml_cpu_has_kompute(); // [jart]
const bool        test::metal        = false; // !!ggml_cpu_has_metal(); // [jart]
const bool        test::gpu_blas     = false; // !!ggml_cpu_has_gpublas(); // [jart]
const bool        test::blas         = false; // !!ggml_cpu_has_blas(); // [jart]
const bool        test::sycl         = false; // !!ggml_cpu_has_sycl(); // [jart]
const std::string test::cpu_info     = llamafile_describe_cpu();
const std::string test::gpu_info     = ""; //get_gpu_info(); // [jart]

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

struct update_t_gen_column_args {
    const test & t;
    printer* p;
};

void* update_t_gen_column(void* args) {
    update_t_gen_column_args* argv = static_cast<update_t_gen_column_args*>(args);
    const test & t = argv->t;
    printer* p = argv->p;

    // Check if printer is markdown printer
    console_printer* md_printer = dynamic_cast<console_printer*>(p);
    if (!md_printer) {
        // For non-markdown printers, wait until test is completed
        while (!t.test_completed) {
            usleep(100000);
        }
        p->print_test(t);
        return nullptr;
    }

    // For markdown printer, update continuously
    p->print_test(t);
    int last_t_gen = 0;
    while (!t.test_completed) {
        last_t_gen = t.t_gen;
        // Move up to the previous line and clear it
        printf("\033[A"); // Move up
        printf("\033[2K"); // Clear the entire line

        // Re-print the entire row with the updated t_gen value
        p->print_test(t);

        fflush(stdout);

        usleep(100000); // sleep for 100ms (100,000 microseconds)
    }
    
    printf("\033[A"); // Move up
    printf("\033[2K"); // Clear the entire line
    p->print_test(t);
    return nullptr;
}

std::string getUserConfirmation() {
    std::string user_input;
    printf("\nDo you want to submit your results to https://localscore.ai? The results will be public (y/n): ");
    std::getline(std::cin, user_input);
    
    // Convert to lowercase for case-insensitive comparison
    std::transform(user_input.begin(), user_input.end(), user_input.begin(), ::tolower);
    return user_input;
}

__attribute__((__constructor__(101))) static void init(void) {
    FLAG_gpu = LLAMAFILE_GPU_AUTO;
}

static void warmup_run(llama_model *model, llama_context *ctx, cmd_params inst) {
    printf("Warming up...... ");
    int n_batch = inst.n_batch;
    int n_processed = 0;
    int n_prompt = inst.n_prompt;
    int n_gen = inst.n_gen;

    const int32_t n_vocab = llama_n_vocab(model);
    std::vector<llama_token> tokens(n_batch);

    llama_kv_cache_clear(ctx);

    // warmup prompt
    while (n_processed < n_prompt) {
        int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0] = n_processed == 0 && llama_add_bos_token(model)
                        ? llama_token_bos(model)
                        : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
        }
        llama_decode(
            ctx, llama_batch_get_one(tokens.data(), n_tokens, n_processed, 0));
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);

    // warmup gen
    llama_token token = llama_add_bos_token(model) ? llama_token_bos(model)
                                                   : std::rand() % n_vocab;
    for (int i = 0; i < n_gen; i++) {
        llama_decode(ctx, llama_batch_get_one(&token, 1, n_prompt + i, 0));
        llama_synchronize(ctx);
        token = std::rand() % n_vocab;
    }

    llama_free(ctx);

    printf("Warmup complete.\n\n");
}

static bool submitBenchmarkResults(const std::string& req_payload, const cmd_params& params, int max_retries = 3) {
    // Ask user for confirmation before sending the data
    std::string user_cnf;
    if (params.send_results == SEND_ASK) {
        user_cnf = getUserConfirmation();
    }

    if (!(user_cnf == "yes" || user_cnf == "y" || params.send_results == SEND_YES)) {
        printf("\nResults Not Submitted.\n");
        return false;
    }

    if (params.verbose) {
        printf("Submitting results...\n Payload: %s\n", req_payload.c_str());
    } else {
        printf("\nSubmitting results...\n");
    }
    
    // Implement retry with exponential backoff
    for (int attempt = 0; attempt < max_retries; attempt++) {
        if (attempt > 0) {
            // Exponential backoff: wait 2^attempt seconds before retrying
            int wait_time = (1 << attempt);
            printf("Retry attempt %d of %d after %d seconds...\n", attempt + 1, max_retries, wait_time);
            usleep(wait_time * 1000000);
        }

        try { 
            Response response = POST(BASE_LOCALSCORE_URL + "/api/results", req_payload, {
                {"Content-Type", "application/json"}
            });

            if (response.status == 200) {
                std::pair<Json::Status, Json> json = Json::parse(response.body);

                if (json.first != Json::success) {
                    printf("Error parsing response json\n");
                    continue;
                }
                if (!json.second.isObject()) {
                    printf("Response json is not an object\n");
                    continue;
                }

                if (json.second["id"].isNumber()) {
                    printf("Result Link: %s/result/%d\n",
                        BASE_LOCALSCORE_URL.c_str(),
                        (int)json.second["id"].getNumber());
                    return true;
                }
            } else {
                printf("Error submitting results to the public database. Status: %d\n", response.status);
                if (attempt < max_retries - 1) {
                    continue;
                }
            }
        } catch (const std::exception& e) {
            printf("Error submitting results: %s\n", e.what());
            if (attempt < max_retries - 1) {
                continue;
            }
        }
    }

    printf("Failed to submit results after %d attempts\n", max_retries);
    return false;
}

static void acceleratorSelector(cmd_params* params) {
    if (FLAG_gpu >= 0 && llamafile_has_gpu()) {
        if (llamafile_has_cuda()) {
            int count = ggml_backend_cuda_get_device_count();
            if (params->main_gpu == UINT_MAX) {
                if (count == 1) {
                    params->main_gpu = 0;
                } else {
                    fprintf(stderr, "\n\033[0;33mMultiple GPUs detected. Please select the main GPU to use:\n");
                    list_available_accelerators();
                    fprintf(stderr, "\n\033[0m");
                    unsigned int main_gpu;
                    while (true) {
                        fprintf(stderr, "Enter the number of the main GPU: ");
                        std::string input;
                        std::getline(std::cin, input);
                        std::istringstream iss(input);
                        if (iss >> main_gpu && main_gpu >= 0 && main_gpu < count) {
                            break;
                        }
                        fprintf(stderr, "Invalid GPU number. Please try again.\n");
                    }
                    params->main_gpu = main_gpu;
                }
            }
        }
    }
}

struct LocalScoreResultsSummary {
    double avg_prompt_tps;
    double avg_gen_tps;
    double avg_ttft_ms;
    double performance_score;
};

static LocalScoreResultsSummary getResultsSummary(Json data) {
    LocalScoreResultsSummary rs = {
        0.0,
        0.0,
        0.0,
        0.0
    };

    if (data["results"].isArray()) {
        std::vector<Json> results = data["results"].getArray();
        
        double total_prompt_tps = 0.0;
        double total_gen_tps = 0.0;
        double total_ttft_ms = 0.0;
        int valid_count = 0;

        for (const auto & result : results) {
            if (result.isObject()) {
                bool valid_entry = true;
                
                // Check if all required fields exist and are numbers
                if (!result.contains("prompt_tps") || 
                    !result.contains("gen_tps") || 
                    !result.contains("ttft_ms")) {
                    valid_entry = false;
                } else {
                    // Get a non-const reference to access the values
                    auto& obj = const_cast<Json&>(result);
                    if (!obj["prompt_tps"].isNumber() ||
                        !obj["gen_tps"].isNumber() ||
                        !obj["ttft_ms"].isNumber()) {
                        valid_entry = false;
                    }
                }

                if (valid_entry) {
                    auto& obj = const_cast<Json&>(result);
                    total_prompt_tps += obj["prompt_tps"].getNumber();
                    total_gen_tps += obj["gen_tps"].getNumber();
                    total_ttft_ms += obj["ttft_ms"].getNumber();
                    valid_count++;
                }
            }
        }

        if (valid_count > 0) {
            rs.avg_prompt_tps = total_prompt_tps / valid_count;
            rs.avg_gen_tps = total_gen_tps / valid_count;
            rs.avg_ttft_ms = total_ttft_ms / valid_count;

            // calculate the geometric mean of the performance values for a score
            double score = pow(rs.avg_prompt_tps * rs.avg_gen_tps * (1000 / rs.avg_ttft_ms), 1.0 / 3.0) * 10;

            rs.performance_score = score;
        }
    }

    return rs;
}

static void displayResults(LocalScoreResultsSummary results_summary, bool plaintext) {

    printf("\n%s", utils::color_str("\033[1;35m"));
    if (!plaintext) {
        ascii_display::print_logo();
        printf("\n");
        ascii_display::printLargeNumber((int)results_summary.performance_score);
    } else {
        printf("LocalScore: \t\t %d", (int)results_summary.performance_score);
    }
    printf("%s\n", utils::color_str("\033[0m"));

    printf("%sToken Generation: \t %s%.2f%s %stok/s%s\n", 
           utils::color_str("\033[32m"), 
           utils::color_str("\033[1;32m"), 
           results_summary.avg_gen_tps,
           utils::color_str("\033[0m"), 
           utils::color_str("\033[3;32m"), 
           utils::color_str("\033[0m"));
           
    printf("%sPrompt Processing: \t %s%.2f%s %stok/s%s\n", 
           utils::color_str("\033[36m"), 
           utils::color_str("\033[1;36m"), 
           results_summary.avg_prompt_tps,
           utils::color_str("\033[0m"), 
           utils::color_str("\033[3;36m"), 
           utils::color_str("\033[0m"));
           
    printf("%sTime to First Token:\t %s%.2f%s %sms%s\n", 
           utils::color_str("\033[33m"), 
           utils::color_str("\033[1;33m"), 
           results_summary.avg_ttft_ms,
           utils::color_str("\033[0m"), 
           utils::color_str("\033[3;33m"), 
           utils::color_str("\033[0m"));
           
    printf("%s", utils::color_str("\033[0m"));
}

struct SystemData {
    RuntimeInfo runtime;
    SystemInfo sys;
    AcceleratorInfo accelerator;
};


// Helper function implementations
std::vector<test_config> get_baseline_test_configs() {
    return {
        {1024, 16},     // 64:1 title generation
        {4096, 256},    // 16:1 content summarization
        {2048, 256},    // 8:1  lots of code to fix
        {2048, 768},    // 3:1  standard code chat
        {1024, 1024},   // 1:1  code back and forth
        {1280, 3072},   // 1:3  reasoning over code
        {384, 1152},    // 1:3  code gen with back and forth
        {64, 1024},     // 1:16 code gen/ideation
        {16, 1536}      // 1:96 QA, Storytelling, Reasoning
    };
}

void setup_initial_environment(int* argc, char*** argv, cmd_params* params, SystemData* sys_data) {
    LoadZipArgs(argc, argv);
    setlocale(LC_CTYPE, "C.UTF-8");
    *params = parse_cmd_params(*argc, *argv);
    FLAGS_READY = true;
    acceleratorSelector(params);
    get_runtime_info(&sys_data->runtime);
    get_sys_info(&sys_data->sys);
    get_accelerator_info(&sys_data->accelerator, params);
}

void initialize_llama_backend(const cmd_params& params) {
    if (!params.verbose) {
        llama_log_set(llama_null_log_callback, NULL);
        ggml_backend_metal_log_set_callback(llama_null_log_callback, NULL);
    }
    llama_backend_init();
    llama_numa_init(params.numa);
}

llama_model* load_model(const cmd_params& params) {
    printf("Loading model... ");
    cmd_params inst = params;
    inst.n_prompt = 1024;
    inst.n_gen = 16;
    llama_model* model = llama_load_model_from_file(inst.model.c_str(), inst.to_llama_mparams());
    if (!model) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
    }
    printf("Model loaded.\n");
    return model;
}

std::unique_ptr<printer> create_printer(const cmd_params& params) {
    std::unique_ptr<printer> p;
    switch (params.output_format) {
        case CSV: p.reset(new csv_printer()); break;
        case JSON: p.reset(new json_printer()); break;
        case CONSOLE: p.reset(new console_printer()); break;
        default: assert(false); exit(1);
    }
    p->set_file_output(stdout);
    return p;
}

void perform_warmup(llama_model* model, const cmd_params& params) {
    cmd_params inst = params;
    inst.n_prompt = 1024;
    inst.n_gen = 16;
    
    llama_context_params cparams = inst.to_llama_cparams();
    cparams.n_ctx = inst.n_prompt + inst.n_gen;
    
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "%s: error: failed to create warmup context\n", __func__);
        exit(1);
    }
    // ctx free happens in warmup
    warmup_run(model, ctx, inst);
}

bool run_baseline_tests(const std::vector<test_config>& tests, llama_model* model, 
                        const cmd_params& params, printer* p, PowerSampler* sampler, 
                        json_printer* req_printer) {
    for (const auto& test_cfg : tests) {
        cmd_params inst = params;
        inst.n_prompt = test_cfg.n_prompt;
        inst.n_gen = test_cfg.n_gen;

        llama_context_params cparams = inst.to_llama_cparams();
        cparams.n_ctx = test_cfg.n_prompt + test_cfg.n_gen;

        llama_context* ctx = llama_new_context_with_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "%s: error: failed to create context\n", __func__);
            return false;
        }

        test t(inst, model, ctx, params.reps, sampler);
        
        update_t_gen_column_args args = {t, p};
        pthread_t update_thread;
        if (int rc = pthread_create(&update_thread, NULL, update_t_gen_column, &args)) {
            fprintf(stderr, "Error creating pthread: %d\n", rc);
            return false;
        }
        
        t.run();
        pthread_join(update_thread, NULL);
        req_printer->print_test(t);
        
        llama_free(ctx);
    }
    return true;
}

void process_and_submit_results(const std::string& req_payload, const cmd_params& params) {
    auto [status, data] = Json::parse(req_payload);
    if (status != Json::success || !data.isObject()) {
        printf("Invalid JSON results\n");
        exit(1);
    }
    LocalScoreResultsSummary rs = getResultsSummary(data);
    displayResults(rs, params.plaintext);

    Json results_summary;
    results_summary.setObject();
    results_summary["avg_prompt_tps"] = rs.avg_prompt_tps;
    results_summary["avg_gen_tps"] = rs.avg_gen_tps;
    results_summary["avg_ttft_ms"] = rs.avg_ttft_ms;
    results_summary["performance_score"] = rs.performance_score;

    data["results_summary"] = results_summary;

    const std::string payload = data.toString();

    submitBenchmarkResults(payload, params);
}

int localscore_cli(int argc, char** argv) {
    ShowCrashReports();
    
    auto baseline_tests = get_baseline_test_configs();
    
    cmd_params params;
    SystemData sys_data;
    setup_initial_environment(&argc, &argv, &params, &sys_data);
    
    initialize_llama_backend(params);
    
    llama_model* lmodel = load_model(params);
    if (!lmodel) return 1;
    
    ModelInfo model_info;
    get_model_info(&model_info, lmodel);
    
    std::string req_payload;
    json_printer* req_printer = new json_printer();
    req_printer->set_string_output(req_payload);
    req_printer->print_header(params, sys_data.accelerator, sys_data.runtime, sys_data.sys, model_info);
    
    auto p = create_printer(params);
    PowerSampler* sampler = getPowerSampler(100, params.main_gpu);
    
    perform_warmup(lmodel, params);
    
    p->print_header(params, sys_data.accelerator, sys_data.runtime, sys_data.sys, model_info);
    
    if (!run_baseline_tests(baseline_tests, lmodel, params, p.get(), sampler, req_printer)) {
        llama_free_model(lmodel);
        llama_backend_free();
        return 1;
    }
    
    llama_free_model(lmodel);
    p->print_footer();
    req_printer->print_footer();
    
    llama_backend_free();
    
    process_and_submit_results(req_payload, params);
    delete req_printer;
    
    return 0;
}