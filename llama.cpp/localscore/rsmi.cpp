#include <cosmo.h>
#include <dlfcn.h>

#include "rsmi.h"
#include "llama.cpp/common.h"

#define IMPORT_RSMI_FUNCTION(func_name, func_type) \
    ok &= !!(rsmi.func_name = (func_type)(imp(lib, #func_name)))

#define RSMI_FUNCTION_CALL(func_name, error_msg, ...) \
    do { \
        if (!rsmi.func_name) { \
            tinylog(__func__, ": error: " #func_name " not imported\n", NULL); \
            return false; \
        } \
        int status = rsmi.func_name(__VA_ARGS__); \
        if (status != 0) { \
            tinylog(__func__, ": error: " error_msg "\n", NULL); \
            return false; \
        } \
    } while(0)

static void *imp(void *lib, const char *sym) {
    void *fun = cosmo_dlsym(lib, sym);
    if (!fun)
        tinylog(__func__, ": error: failed to import symbol: ", sym, "\n", NULL);
    return fun;
}

typedef enum {
  RSMI_AVERAGE_POWER = 0,            //!< Average Power
  RSMI_CURRENT_POWER,                //!< Current / Instant Power
  RSMI_INVALID_POWER = 0xFFFFFFFF    //!< Invalid / Undetected Power
} RSMI_POWER_TYPE;

static struct Rsmi {
    int (*rsmi_init)(uint64_t init_flags);
    int (*rsmi_num_monitor_devices)(uint32_t *num_devices);
    int (*rsmi_dev_id_get)(uint32_t dv_ind, uint16_t *id);
    int (*rsmi_dev_power_get)(uint32_t dv_ind, uint64_t *power, RSMI_POWER_TYPE *type);
    int (*rsmi_dev_current_socket_power_get)(uint32_t dv_ind, uint64_t *power); // in uW
    int (*rsmi_dev_power_ave_get)(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power);
    int (*rsmi_dev_energy_count_get)(uint32_t dv_ind, uint64_t *power, float *counter_resolution, uint64_t *timestamp);
    int (*rsmi_dev_memory_usage_get)(uint32_t dv_ind, int mem_type, uint64_t *used);
    int (*rsmi_shut_down)(void);
} rsmi;

bool rsmi_init() {
    void *lib = cosmo_dlopen("/opt/rocm/lib/librocm_smi64.so", RTLD_NOW);
    bool ok = true;

    IMPORT_RSMI_FUNCTION(rsmi_init, int (*)(uint64_t));
    IMPORT_RSMI_FUNCTION(rsmi_num_monitor_devices, int (*)(uint32_t*));
    IMPORT_RSMI_FUNCTION(rsmi_dev_id_get, int (*)(uint32_t, uint16_t*));
    IMPORT_RSMI_FUNCTION(rsmi_dev_power_get, int (*)(uint32_t, uint64_t*, RSMI_POWER_TYPE*));
    IMPORT_RSMI_FUNCTION(rsmi_dev_current_socket_power_get, int (*)(uint32_t, uint64_t*));
    IMPORT_RSMI_FUNCTION(rsmi_dev_power_ave_get, int (*)(uint32_t, uint32_t, uint64_t*));
    IMPORT_RSMI_FUNCTION(rsmi_dev_energy_count_get, int (*)(uint32_t, uint64_t*, float*, uint64_t*));
    IMPORT_RSMI_FUNCTION(rsmi_dev_memory_usage_get, int (*)(uint32_t, int, uint64_t*));
    IMPORT_RSMI_FUNCTION(rsmi_shut_down, int (*)(void));

    if (!ok) {
        tinylog(__func__, ": error: not all rocm smi symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }

    RSMI_FUNCTION_CALL(rsmi_init, "failed to initialize ROCm SMI", 0);
    return true;
}

bool rsmi_get_avg_power(double *power) {
    uint64_t power_val;
    RSMI_FUNCTION_CALL(rsmi_dev_power_ave_get, "failed to get average power", 0, 0, &power_val);
    *power = (double)power_val;
    return true;
}

bool rsmi_get_power(double *power) {
    uint64_t power_val;
    RSMI_POWER_TYPE type;

    RSMI_FUNCTION_CALL(rsmi_dev_power_get, "failed to get power", 0, &power_val, &type);
    // Convert microwatts to milliwatts
    *power = (double)power_val / 1000.0;
    return true;
}

bool rsmi_get_energy_count(double *energy) {
    uint64_t power;
    float counter_resolution;
    uint64_t timestamp;
    RSMI_FUNCTION_CALL(rsmi_dev_energy_count_get, "failed to get energy count", 0, &power, &counter_resolution, &timestamp);
    // Convert microjoules to millijoules
    *energy = (double)(power * counter_resolution) / 1000.0;
    return true;
}

bool rsmi_get_power_instant(double *power) {
    uint64_t power_val;
    RSMI_FUNCTION_CALL(rsmi_dev_current_socket_power_get, "failed to get instant power", 0, &power_val);
    *power = (double)power_val;
    return true;
}

bool rsmi_get_memory_usage(float *memory) {
    uint64_t used;
    // this is device 0 and memory type 0 (RSMI_MEM_TYPE_VRAM)
    RSMI_FUNCTION_CALL(rsmi_dev_memory_usage_get, "failed to get memory usage", 0, 0, &used);
    *memory = (float)used / 1024.0 / 1024.0;
    return true;
}

bool rsmi_shutdown() {
    RSMI_FUNCTION_CALL(rsmi_shut_down, "failed to shutdown ROCm SMI");
    return true;
}