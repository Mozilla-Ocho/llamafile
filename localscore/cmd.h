#pragma once

#include <string>
#include <vector>
#include "llama.cpp/llama.h"

enum output_formats {CSV, JSON, CONSOLE};
enum send_results_mode {SEND_ASK, SEND_YES, SEND_NO};

struct cmd_params {
    std::string model;
    int n_prompt;
    int n_gen;
    int n_batch;
    int n_ubatch;
    ggml_type type_k;
    ggml_type type_v;
    int n_threads;
    int gpu;
    int n_gpu_layers;
    llama_split_mode split_mode;
    unsigned int main_gpu;
    bool no_kv_offload;
    bool flash_attn;
    std::vector<float> tensor_split;
    bool use_mmap;
    bool embeddings;
    ggml_numa_strategy numa;
    int reps;
    bool verbose;
    bool plaintext;
    send_results_mode send_results;
    output_formats output_format;

    llama_model_params to_llama_mparams() const;
    bool equal_mparams(const cmd_params & other) const;
    llama_context_params to_llama_cparams() const;
};

cmd_params parse_cmd_params(int argc, char ** argv);
void print_usage(int argc, char ** argv);