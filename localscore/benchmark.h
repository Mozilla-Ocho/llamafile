#pragma once

#include "llama.cpp/llama.h"
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "cmd.h"
#include "powersampler.h"

struct time_interval {
    uint64_t start;
    uint64_t end;
};

struct test_config {
    int n_prompt;
    int n_gen;
};

enum token_metric { TOTAL_TPS, PROMPT_TPS, GEN_TPS };

struct test {
    static const std::string build_commit;
    static const int build_number;
    static const bool cuda;
    static const bool opencl;
    static const bool vulkan;
    static const bool kompute;
    static const bool metal;
    static const bool sycl;
    static const bool gpu_blas;
    static const bool blas;
    static const std::string cpu_info;
    static const std::string gpu_info;
    std::string name;
    std::string model_name;
    std::string model_filename;
    std::string model_type;
    std::string model_quant_str;
    std::string model_params_str;
    uint64_t model_size;
    uint64_t model_n_params;
    int n_batch;
    int n_ubatch;
    int n_threads;
    ggml_type type_k;
    ggml_type type_v;
    int n_gpu_layers;
    llama_split_mode split_mode;
    int main_gpu;
    bool no_kv_offload;
    bool flash_attn;
    std::vector<float> tensor_split;
    bool use_mmap;
    bool embeddings;
    int n_prompt;
    int n_gen;
    int reps;
    mutable std::mutex t_gen_mutex;
    std::atomic_bool test_completed{false};
    volatile int curr_run;
    volatile int t_gen;       // this is the total number of tokens generated
    volatile int t_processed; // this is the total number of tokens processed
    power_sample_t monitor_result;
    std::string test_time;
    std::vector<time_interval> test_intervals;
    std::vector<time_interval> prompt_intervals;
    std::vector<time_interval> gen_intervals;
    std::vector<uint64_t> time_to_first_token;
    llama_context *ctx;
    PowerSampler *pwr_sampler;

    enum field_type { STRING, BOOL, INT, FLOAT };

    test(const cmd_params &inst, const llama_model *lmodel,
         llama_context *context, int repetitions, PowerSampler *sampler);
    void run();
    void test_prompt();
    void test_gen();
    std::vector<uint64_t> get_samples_ns(token_metric metric = TOTAL_TPS) const;
    uint64_t avg_ns(token_metric metric = TOTAL_TPS) const;
    uint64_t stdev_ns(token_metric metric = TOTAL_TPS) const;
    float get_power() const;
    std::vector<double> get_ts(token_metric metric = TOTAL_TPS) const;
    double avg_ts(token_metric metric = TOTAL_TPS) const;
    double stdev_ts(token_metric metric = TOTAL_TPS) const;
    double get_tps_watt(token_metric metric = TOTAL_TPS) const;
    double ttft() const;

    std::vector<std::string> get_values() const;
    std::map<std::string, std::string> get_map() const;

    static std::string get_backend();
    static const std::vector<std::string> get_fields();
    static field_type get_field_type(const std::string &field);
};
