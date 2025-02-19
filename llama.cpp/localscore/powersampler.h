#pragma once

#include <pthread.h>
#include <time.h>
#include <vector>
#include "nvml.h"
#include "rsmi.h"
#include "apple.h"
#include "llama.cpp/ggml-backend-impl.h"

typedef struct {
    double  power;
} power_sample_t;

struct PowerSampler {
    // vars
    long sample_length_ms_;

    timespec sampling_start_time_;
    timespec sampling_end_time_;
    double energy_consumed_start_;

    std::vector<power_sample_t> samples_;

    bool is_sampling_;
    pthread_t sampling_thread_;
    mutable pthread_mutex_t samples_mutex_;

    // funcs
    PowerSampler(long sample_length_ms);
    virtual ~PowerSampler();

    void start();
    power_sample_t stop();

    power_sample_t getLatestSample();

    // this returns the instantaneous power in microwatts
    virtual power_sample_t sample() = 0;
    
    // this returns the energy consumed in millijoules
    virtual double getEnergyConsumed() = 0;

private:
    static void* sampling_thread_func(void* arg);
};

struct NvidiaPowerSampler : public PowerSampler {
    nvmlDevice_t device_;
    unsigned long long start_joules_;
    unsigned long long end_joules_;

    NvidiaPowerSampler(long sample_length_ms, unsigned int index);
    ~NvidiaPowerSampler() override;

protected:
    power_sample_t sample() override;
    double getEnergyConsumed() override;
};

struct AMDPowerSampler : public PowerSampler {
    AMDPowerSampler(long sample_length_ms);
    ~AMDPowerSampler() override;

protected:
    power_sample_t sample() override;
    double getEnergyConsumed() override;
};

struct ApplePowerSampler : public PowerSampler {
    ggml_backend_t metal_backend_;
    CFMutableDictionaryRef power_channel_;
    IOReportSubscriptionRef sub_;
    long long last_sample_time_;
    double last_sample_mj_;

    ApplePowerSampler(long sample_length_ms);
    ~ApplePowerSampler() override;

protected:
    power_sample_t sample() override;
    double getEnergyConsumed() override;
};

struct DummyPowerSampler : public PowerSampler {
    DummyPowerSampler(long sample_length_ms);
    ~DummyPowerSampler() override {}

protected:
    power_sample_t sample() override;
    double getEnergyConsumed() override;
};

PowerSampler* getPowerSampler(long sample_length_ms, unsigned int index);
