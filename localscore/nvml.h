#pragma once

typedef void* nvmlDevice_t;

bool nvml_init();

bool nvml_get_device(nvmlDevice_t *device, unsigned int index);

bool nvml_get_energy_consumption(nvmlDevice_t device, unsigned long long *energy);

bool nvml_get_power_usage(nvmlDevice_t device, unsigned int *power);

bool nvml_get_memory_usage(nvmlDevice_t device, float *memory);

bool nvml_shutdown();