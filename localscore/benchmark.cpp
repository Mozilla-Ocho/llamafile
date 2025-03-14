#include "benchmark.h"

#include <iterator>

#include "llama.cpp/ggml-cuda.h"
#include "llama.cpp/string.h"
#include "llamafile/string.h"
#include "utils.h"

test::test(const cmd_params &inst, const llama_model *lmodel,
           llama_context *context, int repetitions, PowerSampler *sampler) {
    model_filename = lf::basename(strdup(inst.model.c_str())); // [jart]
    char buf[128];
    llama_model_desc(lmodel, buf, sizeof(buf));
    model_type = buf;
    llama_model_meta_val_str(lmodel, "general.name", buf, sizeof(buf));
    model_name = buf;
    llama_model_quant_str(lmodel, buf, sizeof(buf));
    model_quant_str = buf;
    model_size = llama_model_size(lmodel);
    model_n_params = llama_model_n_params(lmodel);
    llama_model_meta_val_str(lmodel, "general.size_label", buf, sizeof(buf));
    model_params_str = buf;
    n_batch = inst.n_batch;
    n_ubatch = inst.n_ubatch;
    n_threads = inst.n_threads;
    type_k = inst.type_k;
    type_v = inst.type_v;
    n_gpu_layers = inst.n_gpu_layers;
    split_mode = inst.split_mode;
    main_gpu = inst.main_gpu;
    no_kv_offload = inst.no_kv_offload;
    flash_attn = inst.flash_attn;
    tensor_split = inst.tensor_split;
    use_mmap = inst.use_mmap;
    embeddings = inst.embeddings;
    n_prompt = inst.n_prompt;
    n_gen = inst.n_gen;
    reps = repetitions;
    test_completed = false;
    curr_run = 0;
    t_gen = 0;
    t_processed = 0;
    monitor_result = {0.0};
    pwr_sampler = sampler;

    if (n_prompt > 0 && n_gen == 0) {
        snprintf(buf, sizeof(buf), "pp%d", n_prompt);
    } else if (n_gen > 0 && n_prompt == 0) {
        snprintf(buf, sizeof(buf), "tg%d", n_gen);
    } else {
        snprintf(buf, sizeof(buf), "pp%d+tg%d", n_prompt, n_gen);
    }
    name = buf;

    // RFC 3339 date-time format
    time_t t = time(NULL);
    std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
    test_time = buf;

    ctx = context;
}

void test::run() {

    // run the test for however many repetitions specified
    pwr_sampler->start();
    for (int i = 0; i < reps; i++) {
        curr_run = i;
        t_processed = 0;
        t_gen = 0;
        llama_kv_cache_clear(ctx);

        time_interval interval;
        interval.start = utils::get_time_ns();
        interval.end = 0;
        test_intervals.push_back(interval);

        if (n_prompt > 0) {
            test_prompt();
        }
        if (n_gen > 0) {
            test_gen();
        }

        test_intervals.back().end = utils::get_time_ns();
    }
    monitor_result = pwr_sampler->stop();

    test_completed = true;
}

void test::test_prompt() {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model *model = llama_get_model(ctx);
    const int32_t n_vocab = llama_n_vocab(model);

    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;

    time_interval interval;
    interval.start = utils::get_time_ns();
    interval.end = 0;
    prompt_intervals.push_back(interval);

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
        t_processed = n_processed;
    }

    llama_synchronize(ctx);

    prompt_intervals.back().end = utils::get_time_ns();
}

void test::test_gen() {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model *model = llama_get_model(ctx);
    const int32_t n_vocab = llama_n_vocab(model);

    llama_token token = llama_add_bos_token(model) ? llama_token_bos(model)
                                                   : std::rand() % n_vocab;

    time_interval interval;
    interval.start = utils::get_time_ns();
    interval.end = 0;
    gen_intervals.push_back(interval);

    for (int i = 0; i < n_gen; i++) {
        llama_decode(ctx, llama_batch_get_one(&token, 1, n_prompt + i, 0));
        llama_synchronize(ctx);
        if (i == 0) {
            uint64_t ttft = utils::get_time_ns() - test_intervals.back().start;
            time_to_first_token.push_back(ttft);
        }
        token = std::rand() % n_vocab;
        t_gen = i + 1;
    }

    gen_intervals.back().end = utils::get_time_ns();
}

std::vector<uint64_t> test::get_samples_ns(token_metric metric) const {
    const std::vector<time_interval> &intervals =
        metric == PROMPT_TPS ? prompt_intervals
        : metric == GEN_TPS  ? gen_intervals
                             : test_intervals;

    std::vector<uint64_t> samples_ns;
    for (const auto &interval : intervals) {
        if (interval.end == 0) {
            continue;
        }
        samples_ns.push_back(interval.end - interval.start);
    }
    return samples_ns;
}

uint64_t test::avg_ns(token_metric metric) const {
    std::vector<uint64_t> samples_ns = get_samples_ns(metric);
    return utils::avg(samples_ns);
}

uint64_t test::stdev_ns(token_metric metric) const {
    std::vector<uint64_t> samples_ns = get_samples_ns(metric);
    return utils::stdev(samples_ns);
}

float test::get_power() const {
    if (monitor_result.power > 0) {
        return monitor_result.power;
    } else {
        // the sample is in mw, convert to w
        return pwr_sampler->getLatestSample().power / 1000.0f;
    }
}

std::vector<double> test::get_ts(token_metric metric) const {
    int n_tokens = 0;
    switch (metric) {
    case TOTAL_TPS:
        n_tokens = n_prompt + n_gen;
        break;
    case PROMPT_TPS:
        n_tokens = n_prompt;
        break;
    case GEN_TPS:
        n_tokens = n_gen;
        break;
    }

    std::vector<double> ts;
    std::vector<uint64_t> samples_ns = get_samples_ns(metric);
    std::transform(samples_ns.begin(), samples_ns.end(), std::back_inserter(ts),
                   [n_tokens](uint64_t t) { return 1e9 * n_tokens / t; });
    return ts;
}

double test::avg_ts(token_metric metric) const {
    return utils::avg(get_ts(metric));
}

double test::stdev_ts(token_metric metric) const {
    return utils::stdev(get_ts(metric));
}

double test::get_tps_watt(token_metric metric) const {
    double power = get_power();
    double ts = avg_ts(metric);

    if (ts == 0.0 || power == 0.0) {
        return 0.0;
    }

    return avg_ts(metric) / get_power();
}

double test::ttft() const {
    if (time_to_first_token.empty()) {
        return 0.0;
    }
    return utils::avg(time_to_first_token);
}

std::vector<std::string> test::get_values() const {
    std::string tensor_split_str;
    int max_nonzero = 0;
    for (size_t i = 0; i < llama_max_devices(); i++) {
        if (tensor_split[i] > 0) {
            max_nonzero = i;
        }
    }
    for (int i = 0; i <= max_nonzero; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2f", tensor_split[i]);
        tensor_split_str += buf;
        if (i < max_nonzero) {
            tensor_split_str += "/";
        }
    }
    double power = get_power();

    std::vector<std::string> values = {
        build_commit, std::to_string(build_number), model_name, model_quant_str,
        model_params_str,
        // std::to_string(cuda), std::to_string(opencl), std::to_string(vulkan),
        // std::to_string(vulkan), std::to_string(metal), std::to_string(sycl),
        // std::to_string(gpu_blas), std::to_string(blas), cpu_info, gpu_info,
        model_filename, model_type, std::to_string(model_size),
        std::to_string(model_n_params),
        // std::to_string(n_batch), std::to_string(n_ubatch),
        // std::to_string(n_threads), ggml_type_name(type_k),
        // ggml_type_name(type_v), std::to_string(n_gpu_layers),
        // split_mode_str(split_mode), std::to_string(main_gpu),
        // std::to_string(no_kv_offload), std::to_string(flash_attn),
        // tensor_split_str, std::to_string(use_mmap),
        // std::to_string(embeddings),
        std::to_string(n_prompt), std::to_string(n_gen), test_time,
        std::to_string(avg_ns() / 1e6), std::to_string(stdev_ns() / 1e6),
        std::to_string(avg_ts(PROMPT_TPS)),
        std::to_string(get_tps_watt(PROMPT_TPS)),
        std::to_string(stdev_ts(PROMPT_TPS)), std::to_string(avg_ts(GEN_TPS)),
        std::to_string(get_tps_watt(GEN_TPS)),
        std::to_string(stdev_ts(GEN_TPS)),
        // name, std::to_string(power), std::to_string(monitor_result.vram),
        // std::to_string(ttft() / 1e6)
        name, std::to_string(power), std::to_string(ttft() / 1e6),
        std::to_string(main_gpu)};
    return values;
}

std::map<std::string, std::string> test::get_map() const {
    std::map<std::string, std::string> map;
    auto fields = get_fields();
    auto values = get_values();
    std::transform(fields.begin(), fields.end(), values.begin(),
                   std::inserter(map, map.end()),
                   std::make_pair<const std::string &, const std::string &>);
    return map;
}

const std::vector<std::string> test::get_fields() {
    static const std::vector<std::string> fields = {
        "build_commit", "build_number", "model_name", "model_quant_str",
        "model_params_str",
        // "cuda", "opencl", "vulkan", "kompute", "metal", "sycl", "gpu_blas",
        // "blas", "cpu_info", "gpu_info",
        "model_filename", "model_type", "model_size", "model_n_params",
        // "n_batch", "n_ubatch",
        // "n_threads", "type_k", "type_v",
        // "n_gpu_layers", "split_mode",
        // "main_gpu", "no_kv_offload", "flash_attn",
        // "tensor_split", "use_mmap", "embeddings",
        "n_prompt", "n_gen", "test_time", "avg_time_ms", "stddev_time_ms",
        "prompt_tps", "prompt_tps_watt", "prompt_tps_stddev", "gen_tps",
        "gen_tps_watt", "gen_tps_stddev",
        // "name", "power_watts", "vram_used_mb", "ttft_ms"
        "name", "power_watts", "ttft_ms", "main_gpu"};
    return fields;
}

std::string test::get_backend() {
    if (cuda) {
        return GGML_CUDA_NAME;
    }
    if (opencl) {
        return "OpenCL";
    }
    if (vulkan) {
        return "Vulkan";
    }
    if (kompute) {
        return "Kompute";
    }
    if (metal) {
        return "Metal";
    }
    if (gpu_blas) {
        return "GPU BLAS";
    }
    if (blas) {
        return "BLAS";
    }

    return "CPU";
}

test::field_type test::get_field_type(const std::string &field) {
    if (field == "build_number" || field == "n_batch" || field == "n_ubatch" ||
        field == "n_threads" || field == "model_size" ||
        field == "model_n_params" || field == "n_gpu_layers" ||
        field == "main_gpu" || field == "n_prompt" || field == "n_gen" ||
        field == "avg_time_ms" || field == "stddev_time_ms" ||
        field == "ttft_ms") {
        return INT;
    }
    if (field == "cuda" || field == "opencl" || field == "vulkan" ||
        field == "kompute" || field == "metal" || field == "gpu_blas" ||
        field == "blas" || field == "sycl" || field == "f16_kv" ||
        field == "no_kv_offload" || field == "flash_attn" ||
        field == "use_mmap" || field == "embeddings") {
        return BOOL;
    }
    if (field == "prompt_tps" || field == "prompt_tps_watt" ||
        field == "prompt_tps_stddev" || field == "gen_tps" ||
        field == "gen_tps_watt" || field == "gen_tps_stddev" ||
        field == "power_watts" || field == "vram_used_mb") {
        return FLOAT;
    }
    return STRING;
}