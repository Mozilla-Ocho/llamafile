#include <cosmo.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include "nvml.h"
#include "llama.cpp/common.h"

static void *imp(void *lib, const char *sym) {
    void *fun = cosmo_dlsym(lib, sym);
    if (!fun)
        tinylog(__func__, ": error: failed to import symbol: ", sym, "\n", NULL);
    return fun;
}

static struct Nvml {
    union {
        int (*default_abi)(void);
        int (__attribute__((__ms_abi__)) *windows_abi)(void);
    } nvmlInit_v2;

    union {
        int (*default_abi)(unsigned int *deviceCount);
        int (__attribute__((__ms_abi__)) *windows_abi)(unsigned int *deviceCount);
    } nvmlDeviceGetCount_v2;

    union {
        int (*default_abi)(unsigned int index, void **device);
        int (__attribute__((__ms_abi__)) *windows_abi)(unsigned int index, void **device);
    } nvmlDeviceGetHandleByIndex_v2;

    union {
        int (*default_abi)(void *device, unsigned long long *energy);
        int (__attribute__((__ms_abi__)) *windows_abi)(void *device, unsigned long long *energy);
    } nvmlDeviceGetTotalEnergyConsumption;

    union {
        int (*default_abi)(void *device, unsigned int *power);
        int (__attribute__((__ms_abi__)) *windows_abi)(void *device, unsigned int *power);
    } nvmlDeviceGetPowerUsage;

    union {
        int (*default_abi)(void);
        int (__attribute__((__ms_abi__)) *windows_abi)(void);
    } nvmlShutdown;
} nvml;

template <typename UnionType>
static void import_nvml_function(void *lib, const char *func_name, UnionType *member, bool *ok) {
    using DefaultFuncType = decltype(UnionType::default_abi);
    using WindowsFuncType = decltype(UnionType::windows_abi);

    void *func_ptr = imp(lib, func_name);
    if (!func_ptr) {
        *ok = false;
        return;
    }

    if (IsWindows()) {
        member->windows_abi = reinterpret_cast<WindowsFuncType>(func_ptr);
    } else {
        member->default_abi = reinterpret_cast<DefaultFuncType>(func_ptr);
    }

    *ok &= true;
}

template <typename UnionType, typename... Args>
auto nvml_function_call(UnionType &func_union, Args &&... args) -> decltype(func_union.default_abi(std::forward<Args>(args)...)) {
    if (IsWindows()) {
        return func_union.windows_abi(std::forward<Args>(args)...);
    } else {
        return func_union.default_abi(std::forward<Args>(args)...);
    }
}

static bool FileExists(const char *path) {
    struct stat st;
    return !stat(path, &st);
}

static bool get_nvml_bin_path(char path[PATH_MAX]) {
    char name[NAME_MAX];
    if (IsWindows())
        strlcpy(name, "nvml.dll", PATH_MAX);
    else
        strlcpy(name, "libnvidia-ml.so", PATH_MAX);

    if (commandv(name, path, PATH_MAX)) {
        printf("Found nvml on path: %s\n", path);
        return true;
    }

    if (IsWindows()) {
        const char *program_files = getenv("ProgramW6432");
        if (!program_files) {
            tinylog(__func__, ": note: $ProgramW6432 not set\n", NULL);
            return false;
        }
        snprintf(path, PATH_MAX, "%s\\NVIDIA Corporation\\NVSMI\\%s", program_files, name);
        printf("Attempting to load %s\n", path);
        if (FileExists(path)) {
            return true;
        } else {
            tinylog(__func__, ": note: %s does not exist\n", path);
            snprintf(path, PATH_MAX, "C:\\Windows\\System32\\%s", name);
            if (FileExists(path)) {
                return true;
            } else {
                tinylog(__func__, ": note: %s does not exist\n", path);
                return false;
            }
        }
    } else {
        strlcpy(path, name, PATH_MAX);
        return true;
    }
}

bool nvml_init() {
    char dso[PATH_MAX];
    if (!get_nvml_bin_path(dso)) {
        tinylog(__func__, ": error: failed to find nvml library\n", NULL);
        return false;
    }

    void *lib = cosmo_dlopen(dso, RTLD_LAZY);
    bool ok = true;

    // TODO we need a more robust way to import symbols and versions.
    // this may end in a segfault currently.
    import_nvml_function(lib, "nvmlInit_v2", &nvml.nvmlInit_v2, &ok);
    import_nvml_function(lib, "nvmlDeviceGetCount_v2", &nvml.nvmlDeviceGetCount_v2, &ok);
    import_nvml_function(lib, "nvmlDeviceGetHandleByIndex_v2", &nvml.nvmlDeviceGetHandleByIndex_v2, &ok);
    import_nvml_function(lib, "nvmlDeviceGetTotalEnergyConsumption", &nvml.nvmlDeviceGetTotalEnergyConsumption, &ok);
    import_nvml_function(lib, "nvmlDeviceGetPowerUsage", &nvml.nvmlDeviceGetPowerUsage, &ok);
    import_nvml_function(lib, "nvmlShutdown", &nvml.nvmlShutdown, &ok);

    if (!ok) {
        tinylog(__func__, ": error: not all nvml symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }

    nvml_function_call(nvml.nvmlInit_v2);
    unsigned int deviceCount;
    nvml_function_call(nvml.nvmlDeviceGetCount_v2, &deviceCount);

    return true;
}

bool nvml_get_device(nvmlDevice_t *device, unsigned int index) {
    nvml_function_call(nvml.nvmlDeviceGetHandleByIndex_v2, index, device);
    return true;
}

bool nvml_get_power_usage(nvmlDevice_t device, unsigned int *power) {
    nvml_function_call(nvml.nvmlDeviceGetPowerUsage, device, power);
    return true;
}

bool nvml_get_energy_consumption(nvmlDevice_t device, unsigned long long *energy) {
    nvml_function_call(nvml.nvmlDeviceGetTotalEnergyConsumption, device, energy);
    return true;
}

bool nvml_shutdown() {
    nvml_function_call(nvml.nvmlShutdown);
    return true;
}