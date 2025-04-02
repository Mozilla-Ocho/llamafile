#include <cassert>

#include "printer.h"
#include "utils.h"

#include "llama.cpp/string.h"

FileWriter::FileWriter(FILE* f): fout(f) {}

void FileWriter::write(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(fout, format, args);
    va_end(args);
}

void FileWriter::flush() {
    fflush(fout);
}

StringWriter::StringWriter(std::string& str) : output(str) {}

void StringWriter::write(const char* format, ...) {
    va_list args;
    va_start(args, format);
    char tmp[1024];
    vsnprintf(tmp, sizeof(tmp), format, args);
    output += tmp;
    va_end(args);
}

void StringWriter::flush() {}

std::string csv_printer::escape_csv(const std::string& field) {
    std::string escaped = "\"";
    for (auto c : field) {
        if (c == '"') {
            escaped += "\"";
        }
        escaped += c;
    }
    escaped += "\"";
    return escaped;
}

void csv_printer::print_header(const cmd_params& params, 
                             AcceleratorInfo accelerator_info, 
                             RuntimeInfo runtime_info, 
                             SystemInfo sys_info, 
                             ModelInfo model_info) {
    std::vector<std::string> fields = test::get_fields();
    writer->write("%s\n", utils::join(fields, ",").c_str());
    (void) params;
}

void csv_printer::print_test(const test& t) {
    std::vector<std::string> values = t.get_values();
    std::transform(values.begin(), values.end(), values.begin(), escape_csv);
    writer->write("%s\n", utils::join(values, ",").c_str());
}

std::string json_printer::escape_json(const std::string & value) {
    std::string escaped;
    for (auto c : value) {
        if (c == '"') {
            escaped += "\\\"";
        } else if (c == '\\') {
            escaped += "\\\\";
        } else  if (c <= 0x1f) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            escaped += buf;
        } else {
            escaped += c;
        }
    }
    return escaped;
}

std::string json_printer::format_value(const std::string & field, const std::string & value) {
    switch (test::get_field_type(field)) {
        case test::STRING:
            return "\"" + escape_json(value) + "\"";
        case test::BOOL:
            return value == "0" ? "false" : "true";
        default:
            return value;
    }
}

void json_printer::print_header(const cmd_params & params, AcceleratorInfo accelerator_info, RuntimeInfo runtime_info, SystemInfo sys_info, ModelInfo model_info) {
    // TODO we really should use jart's JSON.
    writer->write("{\n");
    
    // Print RuntimeInfo object
    writer->write("  \"runtime_info\": {\n");
    writer->write("    \"name\": \"%s\",\n", "llamafile");
    writer->write("    \"version\": \"%s\",\n", runtime_info.llamafile_version);
    writer->write("    \"commit\": \"%s\"\n", runtime_info.llama_commit);
    writer->write("  },\n");

    // Print SystemInfo object
    writer->write("  \"system_info\": {\n");
    writer->write("    \"cpu_name\": \"%s\",\n", sys_info.cpu);
    writer->write("    \"cpu_arch\": \"%s\",\n", sys_info.system_architecture);
    writer->write("    \"ram_gb\": %.2f,\n", sys_info.ram_gb);
    writer->write("    \"kernel_type\": \"%s\",\n", sys_info.kernel_type);
    writer->write("    \"kernel_release\": \"%s\",\n", sys_info.kernel_release);
    writer->write("    \"version\": \"%s\"\n", sys_info.version);
    writer->write("  },\n");

    // Print GPUInfo object
    writer->write("  \"accelerator_info\": {\n");
    writer->write("    \"name\": \"%s\",\n", accelerator_info.name);
    writer->write("    \"manufacturer\": \"%s\",\n", accelerator_info.manufacturer);
    writer->write("    \"memory_gb\": %.2f,\n", accelerator_info.total_memory_gb);
    writer->write("    \"type\": \"%s\"\n", (FLAG_gpu >= 0 && llamafile_has_gpu()) ? "GPU" : "CPU");
    writer->write("  },\n");

    // Start the results array
    writer->write("  \"results\": [\n");
    
    (void) params;
}

void json_printer::print_fields(const std::vector<std::string> & fields, const std::vector<std::string> & values) {
    assert(fields.size() == values.size());
    for (size_t i = 0; i < fields.size(); i++) {
        writer->write("      \"%s\": %s,\n", fields.at(i).c_str(), format_value(fields.at(i), values.at(i)).c_str());
    }
}

void json_printer::print_test(const test & t) {
    if (first) {
        first = false;
    } else {
        writer->write(",\n");
    }
    writer->write("    {\n");
    print_fields(test::get_fields(), t.get_values());
    writer->write("      \"samples_ns\": [ %s ]\n", utils::join(t.get_samples_ns(), ", ").c_str());
    writer->write("    }");
    writer->flush();
}

void json_printer::print_footer() {
    writer->write("\n  ]\n");
    writer->write("}");
}

int console_printer::get_field_width(const std::string & field) {
    if (field == "model") {
        return -30;
    }
    if (field == "t/s") {
        return 15;
    }
    if (field == "cpu_info") {
        return test::cpu_info.size();
    }
    if (field == "model_filename") {
        return 40;
    }
    if (field == "size" || field == "params") {
        return 10;
    }
    if (field == "n_gpu_layers") {
        return 3;
    }
    if (field == "test") {
        return 13;
    }

    int width = std::max((int)field.length(), 10);

    if (test::get_field_type(field) == test::STRING) {
        return -width;
    }
    return width;
}

std::string console_printer::get_field_display_name(const std::string & field) {
    if (field == "n_gpu_layers") {
        return "ngl";
    }
    if (field == "split_mode") {
        return "sm";
    }
    if (field == "n_threads") {
        return "threads";
    }
    if (field == "no_kv_offload") {
        return "nkvo";
    }
    if (field == "flash_attn") {
        return "fa";
    }
    if (field == "use_mmap") {
        return "mmap";
    }
    if (field == "embeddings") {
        return "embd";
    }
    if (field == "tensor_split") {
        return "ts";
    }
    return field;
}

int console_printer::calculate_total_width() const {
    int total_width = 0;
    for (const auto & field : fields) {
        int width = get_field_width(field);
        if (width < 0) {
            width = std::abs(width);
        }
        total_width += width;
    }
    total_width += fields.size() * 3 + 1;
    return total_width;
}    

void console_printer::print_header(const cmd_params & params, AcceleratorInfo accelerator_info, RuntimeInfo runtime_info, SystemInfo sys_info, ModelInfo model_info) {
    fields.emplace_back("test");
    fields.emplace_back("run number");
    fields.emplace_back("avg time");
    // fields.emplace_back("power");
    fields.emplace_back("tokens processed");
    fields.emplace_back("pp t/s");
    fields.emplace_back("tg t/s");
    fields.emplace_back("ttft");

    int total_width = calculate_total_width();

    std::string border(total_width, '-');
    border[0] = '+';
    border[total_width-1] = '+';
    writer->write("%s\n", border.c_str());

    // Create the GPU info string
    char gpu_info_str[256];
    int content_length = snprintf(gpu_info_str, sizeof(gpu_info_str), 
                                "%s - %.1f GiB", 
                                accelerator_info.name, 
                                accelerator_info.total_memory_gb);
    
    // Calculate left and right padding to properly center the text
    int left_padding = (total_width - 2 - content_length) / 2;
    int right_padding = total_width - 2 - content_length - left_padding;
    
    // Print the GPU info with correct padding
    writer->write("|%*s%s%*s|\n",
        left_padding, "", 
        gpu_info_str,
        right_padding, "");

    // Create the model info string
    char model_info_str[256];
    content_length = snprintf(model_info_str, sizeof(model_info_str), 
                                "%s - %s", 
                                model_info.name, 
                                model_info.quant);
    
    // Calculate left and right padding separately
    left_padding = (total_width - 2 - content_length) / 2;
    right_padding = total_width - 2 - content_length - left_padding;
    
    // Print the model info with correct padding
    writer->write("|%*s%s%*s|\n",
        left_padding, "", 
        model_info_str,
        right_padding, "");

    writer->write("%s\n", border.c_str());

    writer->write("|");
    for (const auto & field : fields) {
        writer->write(" %*s |", get_field_width(field), get_field_display_name(field).c_str());
    }
    writer->write("\n");
    writer->write("|");
    for (const auto & field : fields) {
        int width = get_field_width(field);
        writer->write(" %s |", std::string(std::abs(width), '-').c_str());
    }
    writer->write("\n");
}

void console_printer::print_test(const test & t) {
    std::map<std::string, std::string> vmap = t.get_map();

    // float power = t.get_power();

    writer->write("|");
    for (const auto & field : fields) {
        std::string value;
        char buf[128];
        if (field == "model") {
            value = t.model_type;
        } else if (field == "size") {
            if (t.model_size < 1024*1024*1024) {
                snprintf(buf, sizeof(buf), "%.2f MiB", t.model_size / 1024.0 / 1024.0);
            } else {
                snprintf(buf, sizeof(buf), "%.2f GiB", t.model_size / 1024.0 / 1024.0 / 1024.0);
            }
            value = buf;
        } else if (field == "params") {
            snprintf(buf, sizeof(buf), "%ld", t.model_n_params);
            // if (t.model_n_params < 1000*1000*1000) {
            //     snprintf(buf, sizeof(buf), "%.2f M", t.model_n_params / 1e6);
            // } else {
            //     snprintf(buf, sizeof(buf), "%.2f B", t.model_n_params / 1e9);
            // }
            value = buf;
        } else if (field == "backend") {
            value = test::get_backend();
        } else if (field == "run number") {
            snprintf(buf, sizeof(buf), "%d/%d", t.curr_run + 1, t.reps);
            value = buf;
        } else if (field == "test") {
            value = t.name;
        } else if (field == "pp t/s") {
            snprintf(buf, sizeof(buf), "%.2f", t.avg_ts(PROMPT_TPS));

            value = buf;
        } else if (field == "tg t/s") {
            snprintf(buf, sizeof(buf), "%.2f", t.avg_ts(GEN_TPS));
            if (!t.gen_intervals.empty() && t.curr_run < t.gen_intervals.size()) {
                time_interval curr_interval = t.gen_intervals[t.curr_run];
            
                if (curr_interval.end == 0) {
                    // get the live tps instead of avg
                    uint64_t elapsed_ns = utils::get_time_ns() - curr_interval.start;
                    float elapsed_s = elapsed_ns / 1e9;
                    float tps = t.t_gen / elapsed_s;
                    snprintf(buf, sizeof(buf), "%.2f", tps);
                }
            }
        
            value = buf;
        } else if (field == "tokens processed") {
            int num_generated = t.t_gen + (t.curr_run * t.n_gen);
            int num_processed = t.t_processed + (t.curr_run * t.n_prompt);

            snprintf(buf, sizeof(buf), "%d / %d", num_generated + num_processed,  (t.n_gen * t.reps) + (t.n_prompt * t.reps));

            value = buf;
        } else if (field == "pp t/s/watt") {
            snprintf(buf, sizeof(buf), "%.4f", t.get_tps_watt(PROMPT_TPS));

            value = buf;
        } else if (field == "tg t/s/watt") {
            snprintf(buf, sizeof(buf), "%.4f", t.get_tps_watt(GEN_TPS));

            value = buf;
        } else if (field == "ttft") {
            float ttft = t.ttft() / 1e6;

            if (ttft < 1000) {
                snprintf(buf, sizeof(buf), "%.2f ms", ttft);
            } else {
                snprintf(buf, sizeof(buf), "%.2f s", ttft / 1e3);
            }

            value = buf;
        } else if (field == "power") {
            if (t.monitor_result.power > 0) {
                snprintf(buf, sizeof(buf), "%.2f W", t.monitor_result.power);
                value = buf;
            } else {
                // read instant power
                power_sample_t sample = t.pwr_sampler->getLatestSample();
                snprintf(buf, sizeof(buf), "%.2f W", sample.power / 1e3);
            }

            value = buf;
        } else if (field == "avg time") {
            float avg_ms = t.avg_ns() / 1e6;

            if (avg_ms < 1000) {
                snprintf(buf, sizeof(buf), "%.2f ms", avg_ms);
            } else {
                snprintf(buf, sizeof(buf), "%.2f s", avg_ms / 1e3);
            }

            value = buf;
        } else if (vmap.find(field) != vmap.end()) {
            value = replace_all(replace_all(vmap.at(field), ".gguf", ""), ".llamafile", ""); // [jart]
        } else {
            assert(false);
            exit(1);
        }

        int width = get_field_width(field);
        // if (field == "t/s") { // [jart]
        //     // HACK: the utf-8 character is 2 bytes
        //     width += 1;
        // }
        writer->write(" %*s |", width, value.c_str());
    }
    writer->write("\n");
}

void console_printer::print_footer() {
    int total_width = calculate_total_width();
    std::string border(total_width, '-');
    border[0] = '+';
    border[total_width-1] = '+';
    writer->write("%s\n", border.c_str());
}