#pragma once

#include <memory>
#include <string>
#include <cstdio>

#include "system.h"
#include "cmd.h"
#include "benchmark.h"

struct OutputWriter {
    virtual ~OutputWriter() {};
    virtual void write(const char* buf, ...) = 0;
    virtual void flush() = 0;
};

struct FileWriter : public OutputWriter {
    FILE* fout;
    
    FileWriter(FILE* f);
    void write(const char* format, ...) override;
    void flush() override;
};

struct StringWriter : public OutputWriter {
    std::string& output;
    
    StringWriter(std::string& str);
    void write(const char* format, ...) override;
    void flush() override;
};

struct printer {
    virtual ~printer() {}

    std::unique_ptr<OutputWriter> writer;
    
    void set_file_output(FILE* fout) {
        writer = std::make_unique<FileWriter>(fout);
    }
    
    void set_string_output(std::string& output) {
        writer = std::make_unique<StringWriter>(output);
    }

    virtual void print_header(const cmd_params & params, AcceleratorInfo accelerator_info, RuntimeInfo runtime_info, SystemInfo sys_info, ModelInfo model_info) { (void) params; }
    virtual void print_test(const test & t) = 0;
    virtual void print_footer() { }
};

struct csv_printer : public printer {
    static std::string escape_csv(const std::string& field);

    void print_header(const cmd_params& params, 
                     AcceleratorInfo accelerator_info, 
                     RuntimeInfo runtime_info, 
                     SystemInfo sys_info, 
                     ModelInfo model_info) override;

    void print_test(const test& t) override;
};

struct json_printer : public printer {
    bool first = true;

    static std::string escape_json(const std::string& value);
    static std::string format_value(const std::string& field, const std::string& value);

    void print_header(const cmd_params& params, 
                     AcceleratorInfo accelerator_info, 
                     RuntimeInfo runtime_info, 
                     SystemInfo sys_info, 
                     ModelInfo model_info) override;

    void print_test(const test& t) override;
    void print_footer() override;

private:
    void print_fields(const std::vector<std::string>& fields, 
                     const std::vector<std::string>& values);
};

struct console_printer : public printer {
    std::vector<std::string> fields;

    static int get_field_width(const std::string& field);
    static std::string get_field_display_name(const std::string& field);

    int calculate_total_width() const;

    void print_header(const cmd_params& params, 
                     AcceleratorInfo accelerator_info, 
                     RuntimeInfo runtime_info, 
                     SystemInfo sys_info, 
                     ModelInfo model_info) override;
    void print_test(const test& t) override;
    void print_footer() override;
};

