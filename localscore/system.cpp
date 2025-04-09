#include "system.h"
#include <cstdio>
#include <sys/utsname.h>
#include <sys/sysinfo.h>
#include <cosmo.h>
#include <iostream>
#include <string>
#include <sstream>
#include "llama.cpp/string.h"

#include "cmd.h"
#include "utils.h"

#include "llama.cpp/ggml-metal.h"
#include "llama.cpp/ggml-cuda.h"
#include "llama.cpp/common.h"

#include <libc/intrin/x86.h>

#ifdef __x86_64__
void cpuid(unsigned leaf, unsigned subleaf, unsigned *info) {
    asm("movq\t%%rbx,%%rsi\n\t"
        "cpuid\n\t"
        "xchgq\t%%rbx,%%rsi"
        : "=a"(info[0]), "=S"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "0"(leaf), "2"(subleaf));
}

// TODO implement an arm version as well
char* get_cpu_manufacturer(void) {
    union {
        char str[13];      // 12 chars + null terminator
        unsigned reg[4];   // For the 4 registers (EAX, EBX, ECX, EDX)
    } u = {0};            // Initialize to zero

    // Get manufacturer ID with leaf 0
    cpuid(0, 0, u.reg);

    // Rearrange the registers to get the correct string
    // The manufacturer string is in EBX,EDX,ECX order
    unsigned temp = u.reg[1];        // Save EBX
    u.reg[0] = temp;                 // Move EBX to first position
    u.reg[1] = u.reg[3];             // Move EDX to second position
    u.reg[2] = u.reg[2];             // ECX stays in third position
    u.reg[3] = 0;                     // Ensure null termination

    const char* manufacturer = u.str;
    if (strcmp(manufacturer, "AuthenticAMD") == 0) {
        return strdup("AMD");
    } else if (strcmp(manufacturer, "GenuineIntel") == 0) {
        return strdup("Intel");
    }

    return strdup(manufacturer);  // Return the original string if unknown
}
#endif // __x86_64__

std::string get_cpu_info() { // [jart]
    std::string id;

#ifdef __x86_64__
    union { // [jart]
        char str[64];
        unsigned reg[16];
    } u = {0};
    cpuid(0x80000002, 0, u.reg + 0*4);
    cpuid(0x80000003, 0, u.reg + 1*4);
    cpuid(0x80000004, 0, u.reg + 2*4);
    int len = strlen(u.str);
    while (len > 0 && u.str[len - 1] == ' ')
        u.str[--len] = 0;
    id = u.str;
#else
    if (IsLinux()) {
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (!strncmp(buf, "model name", 10) ||
                    startswith(buf, "Model\t\t:")) { // e.g. raspi
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        id = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
    }
    if (IsXnu()) {
        // TODO we can also do something similar to https://github.com/vladkens/macmon/blob/main/src/sources.rs#L424
        char cpu_name[128] = {0};
        size_t size = sizeof(cpu_name);
        if (sysctlbyname("machdep.cpu.brand_string", cpu_name, &size, NULL, 0) != -1) {
            id = cpu_name;
        }

        // Get number of performance cores on macos
        int num_perf0_cpu;
        size = sizeof(num_perf0_cpu);
        if (sysctlbyname("hw.perflevel0.logicalcpu", &num_perf0_cpu, &size, NULL, 0) != -1) {
            id += " ";
            id += std::to_string(num_perf0_cpu);
            id += "P";
        }

        // Get number of efficiency cores on macos
        int num_perf1_cpu;
        size = sizeof(num_perf1_cpu);
        if (sysctlbyname("hw.perflevel1.logicalcpu", &num_perf1_cpu, &size, NULL, 0) != -1) {
            id += "+";
            id += std::to_string(num_perf1_cpu);
            id += "E";
        }

    }
#endif
    id = replace_all(id, " 96-Cores", "");
    id = replace_all(id, "(TM)", "");
    id = replace_all(id, "(R)", "");

    std::string march;
#ifdef __x86_64__
    if (__cpu_march(__cpu_model.__cpu_subtype))
        march = __cpu_march(__cpu_model.__cpu_subtype);
#else
    // TODO. We can do this separately as part of 'features' or something
    // long hwcap = getauxval(AT_HWCAP);
    // if (hwcap & HWCAP_ASIMDHP)
    //     march += "+fp16";
    // if (hwcap & HWCAP_ASIMDDP)
    //     march += "+dotprod";
#endif

    if (!march.empty()) {
        bool empty = id.empty();
        if (!empty)
            id += " (";
        id += march;
        if (!empty)
            id += ")";
    }

    return id;
}

void get_runtime_info(RuntimeInfo* info) {
    if (info == NULL) return;

    strncpy(info->llamafile_version, LLAMAFILE_VERSION_STRING, MAX_STRING_LENGTH - 1);
    strncpy(info->llama_commit, LLAMA_COMMIT, MAX_STRING_LENGTH - 1);

    fprintf(stderr, "%s\n", utils::color_str("\033[0;35m"));  // Sets purple color
    utils::print_centered(stderr, 70, '=', "%sLocalScore Runtime Information%s", utils::color_str("\033[1m"), utils::color_str("\033[0;35m"));
    fprintf(stderr, "\n");
    fprintf(stderr, "%-20s %s%s%s\n", "llamafile version:", utils::color_str("\033[1m"), info->llamafile_version, utils::color_str("\033[22m"));
    fprintf(stderr, "%-20s %s\n", "llama.cpp commit:", info->llama_commit);
    fprintf(stderr, "\n======================================================================\n\n%s", utils::color_str("\033[0m"));
}

double get_mem_gb() {
    struct sysinfo si;
    if (sysinfo(&si)) {
        return 0.0;
    }

    return utils::round_to_decimal(si.totalram * si.mem_unit / 1073741824.0, 1);
}

void get_sys_info(SystemInfo* info) {
    if (info == NULL) return;

    struct utsname names;
    if (uname(&names)) {
        return;
    }

    utils::sanitize_string(info->kernel_type, names.sysname, MAX_STRING_LENGTH);
    utils::sanitize_string(info->kernel_release, names.release, MAX_STRING_LENGTH);
    // TODO on darwin we might want to get from systemprofiler SPSoftwareDataType os_version
    utils::sanitize_string(info->version, names.version, MAX_STRING_LENGTH);
    utils::sanitize_string(info->system_architecture, names.machine, MAX_STRING_LENGTH);

    std::string cpu_info = get_cpu_info();
    strncpy(info->cpu, cpu_info.c_str(), MAX_STRING_LENGTH - 1);

    info->ram_gb = get_mem_gb();

    utils::print_centered(stderr, 70, '=', "%sSystem Information%s", utils::color_str("\033[1m"), utils::color_str("\033[0m"));
    fprintf(stderr, "\n");
    fprintf(stderr, "%-20s %s\n", "Kernel Type:", info->kernel_type);
    fprintf(stderr, "%-20s %s\n", "Kernel Release:", info->kernel_release);
    fprintf(stderr, "%-20s %s\n", "Version:", info->version);
    fprintf(stderr, "%-20s %s\n", "System Architecture:", info->system_architecture);
    fprintf(stderr, "%-20s %s\n", "CPU:", info->cpu);
    fprintf(stderr, "%-20s %.1f GiB\n", "RAM:", info->ram_gb);
    fprintf(stderr, "\n======================================================================\n\n");
}

void get_accelerator_info(AcceleratorInfo* info, cmd_params * params) {
    if (info == NULL) return;

    if (FLAG_gpu >= 0 && llamafile_has_gpu()) {
        if (llamafile_has_cuda()) {
            int count = ggml_backend_cuda_get_device_count();
            for (int i = 0; i < count; i++) {
                struct ggml_cuda_device_properties props;
                ggml_backend_cuda_get_device_properties(i, &props);

                if (params->verbose) {
                    printf("Raw GPU %d Memory %lld Bytes, %.2f GiB\n", i, props.totalGlobalMem, props.totalGlobalMem / 1073741824.0);
                }

                // TODO it would be much better to query NVML directly instead and similar for rocm
                double rounded_memory_gb = utils::round_to_decimal(props.totalGlobalMem / 1073741824.0, 0);

                if (i == params->main_gpu) {
                    strncpy(info->name, props.name, MAX_STRING_LENGTH - 1);

                    info->total_memory_gb = rounded_memory_gb;
                    info->core_count = props.multiProcessorCount;
                    info->capability = atof(props.compute);
                    strncpy(info->manufacturer, llamafile_has_amd_gpu() ? "AMD" : "NVIDIA", MAX_STRING_LENGTH - 1);
                }

                if (i == params->main_gpu) {
                    fprintf(stderr, "%s", utils::color_str("\033[0;32m"));  // Sets green color
                    utils::print_centered(stderr, 70, '=', "%sActive GPU (GPU %d) Information%s", utils::color_str("\033[1m"), i, utils::color_str("\033[0;32m"));
                    fprintf(stderr, "\n");
                } else {
                    fprintf(stderr, "%s", utils::color_str("\033[0;90m"));  // Sets gray color
                    utils::print_centered(stderr, 70, '=', "GPU %d Information", i);
                    fprintf(stderr, "\n");
                }

                fprintf(stderr, "%-26s %s\n", "GPU Name:", props.name);
                fprintf(stderr, "%-26s %.1f GiB\n", "VRAM:", rounded_memory_gb);
                fprintf(stderr, "%-26s %d\n", "Streaming Multiprocessors:", props.multiProcessorCount);
                fprintf(stderr, "%-26s %.1f\n", "CUDA Capability:", atof(props.compute));
                fprintf(stderr, "\n======================================================================\n\n%s", utils::color_str("\033[0m"));
            }
        }

        if (llamafile_has_metal()) {
            // TODO there is probably a cleaner way of doing this. we should only need to init once.
            // this is probably the same issue why the other thing is init multiple time too
            struct ggml_metal_device_properties props;

            std::string command = "system_profiler SPDisplaysDataType | grep \"Total Number of Cores:\" | awk '{print $5}'";
            std::string num_cores = utils::exec(command.c_str());
            props.core_count = std::stoi(num_cores);

            // Remove any trailing newline
            if (!num_cores.empty() && num_cores[num_cores.length()-1] == '\n') {
                num_cores.erase(num_cores.length()-1);
            }

            ggml_backend_t result = ggml_backend_metal_init();

            ggml_backend_metal_get_device_properties(result, &props);

            std::string cpu_info = get_cpu_info();
            cpu_info += "+" + num_cores + "GPU";
            strncpy(info->name, cpu_info.c_str(), MAX_STRING_LENGTH - 1);
            info->total_memory_gb = get_mem_gb();
            info->core_count = props.core_count;
            info->capability = props.metal_version;
            strncpy(info->manufacturer, "Apple", MAX_STRING_LENGTH - 1);


            fprintf(stderr, "%s===== GPU information =====\n\n", utils::color_str("\033[0;32m"));
            fprintf(stderr, "%-26s %s\n", "GPU Name:", props.name);
            fprintf(stderr, "%-26s %.1f GiB\n", "VRAM:", info->total_memory_gb);
            fprintf(stderr, "%-26s %d\n", "Core Count:", props.core_count);
            fprintf(stderr, "%-26s %d\n", "Metal Version:", props.metal_version);
            fprintf(stderr, "%-26s %d\n", "GPU Family:", props.gpu_family);
            fprintf(stderr, "%-26s %d\n", "Common GPU Family:", props.gpu_family_common);
            fprintf(stderr, "\n======================================================================\n\n%s", utils::color_str("\033[0m"));
        }
    } else {
        #ifdef __x86_64__
            strncpy(info->manufacturer, get_cpu_manufacturer(), MAX_STRING_LENGTH - 1);
        #else
            if IsXnu() {
                strncpy(info->manufacturer, "Apple", MAX_STRING_LENGTH - 1);
            } else {
                strncpy(info->manufacturer, "Unknown", MAX_STRING_LENGTH - 1);
            }
        #endif
        strncpy(info->name, get_cpu_info().c_str(), MAX_STRING_LENGTH - 1);
        info->total_memory_gb = get_mem_gb(); 
    }
}

void list_available_accelerators() {
    if (llamafile_has_gpu()) {
        if (llamafile_has_metal()) {
            fprintf(stderr, "Apple Metal\n");
        } else if (llamafile_has_cuda()) {
            int count = ggml_backend_cuda_get_device_count();
            fprintf(stderr, "\n%s==================== Available GPUs ====================\n\n", utils::color_str("\033[0;32m"));
            for (int i = 0; i < count; i++) {
                struct ggml_cuda_device_properties props;
                ggml_backend_cuda_get_device_properties(i, &props);
                fprintf(stderr, "%d: %s - %.2f GiB\n", i, props.name, props.totalGlobalMem / 1073741824.0);
            }
        } else {
            fprintf(stderr, "No Accelerator support available\n");
        }
    } else {
        fprintf(stderr, "No Accelerator support available\n");
    }
    fprintf(stderr, "\n======================================================================\n%s", utils::color_str("\033[0m"));
}

void get_model_info(ModelInfo *info, llama_model *model) {
    char buf[MAX_STRING_LENGTH];
    llama_model_desc(model, buf, sizeof(buf));
    strncpy(info->type, buf, sizeof(buf));
    llama_model_meta_val_str(model, "general.name", buf, sizeof(buf));
    strncpy(info->name, buf, sizeof(buf));

    // check if the model name is empty, if it is then exit the program
    if (strcmp(info->name, "") == 0) {
        fprintf(stderr, "Error: Model name is empty. Please use a valid .gguf.\n");
        exit(1);
    }

    llama_model_quant_str(model, buf, sizeof(buf));
    strncpy(info->quant, buf, sizeof(buf));
    llama_model_meta_val_str(model, "general.size_label", buf, sizeof(buf));
    strncpy(info->size_label, buf, sizeof(buf));
    info->size = llama_model_size(model);
    info->params = llama_model_n_params(model);
}