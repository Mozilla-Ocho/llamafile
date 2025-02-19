bool rsmi_init();

bool rsmi_get_power(double *power);

bool rsmi_get_power_instant(double *power);

bool rsmi_get_avg_power(double *power);

bool rsmi_get_energy_count(double *energy);

bool rsmi_get_memory_usage(float *memory);

bool rsmi_shutdown();