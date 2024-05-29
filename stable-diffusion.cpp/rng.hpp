#ifndef __RNG_H__
#define __RNG_H__

#include <random>
#include <vector>

class RNG {
public:
    virtual void manual_seed(uint64_t seed)      = 0;
    virtual std::vector<float> randn(uint32_t n) = 0;
};

class STDDefaultRNG : public RNG {
private:
    std::default_random_engine generator;

public:
    void manual_seed(uint64_t seed) {
        generator.seed((unsigned int)seed);
    }

    std::vector<float> randn(uint32_t n) {
        std::vector<float> result;
        float mean   = 0.0;
        float stddev = 1.0;
        std::normal_distribution<float> distribution(mean, stddev);
        for (uint32_t i = 0; i < n; i++) {
            float random_number = distribution(generator);
            result.push_back(random_number);
        }
        return result;
    }
};

#endif  // __RNG_H__