#ifndef __RNG_PHILOX_H__
#define __RNG_PHILOX_H__

#include <cmath>
#include <vector>

#include "rng.hpp"

// RNG imitiating torch cuda randn on CPU.
// Port from: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/5ef669de080814067961f28357256e8fe27544f4/modules/rng_philox.py
class PhiloxRNG : public RNG {
private:
    uint64_t seed;
    uint32_t offset;

private:
    std::vector<uint32_t> philox_m = {0xD2511F53, 0xCD9E8D57};
    std::vector<uint32_t> philox_w = {0x9E3779B9, 0xBB67AE85};
    float two_pow32_inv            = 2.3283064e-10f;
    float two_pow32_inv_2pi        = 2.3283064e-10f * 6.2831855f;

    std::vector<uint32_t> uint32(uint64_t x) {
        std::vector<uint32_t> result(2);
        result[0] = static_cast<uint32_t>(x & 0xFFFFFFFF);
        result[1] = static_cast<uint32_t>(x >> 32);
        return result;
    }

    std::vector<std::vector<uint32_t>> uint32(const std::vector<uint64_t>& x) {
        uint32_t N = (uint32_t)x.size();
        std::vector<std::vector<uint32_t>> result(2, std::vector<uint32_t>(N));

        for (uint32_t i = 0; i < N; ++i) {
            result[0][i] = static_cast<uint32_t>(x[i] & 0xFFFFFFFF);
            result[1][i] = static_cast<uint32_t>(x[i] >> 32);
        }

        return result;
    }

    //  A single round of the Philox 4x32 random number generator.
    void philox4_round(std::vector<std::vector<uint32_t>>& counter,
                       const std::vector<std::vector<uint32_t>>& key) {
        uint32_t N = (uint32_t)counter[0].size();
        for (uint32_t i = 0; i < N; i++) {
            std::vector<uint32_t> v1 = uint32(static_cast<uint64_t>(counter[0][i]) * static_cast<uint64_t>(philox_m[0]));
            std::vector<uint32_t> v2 = uint32(static_cast<uint64_t>(counter[2][i]) * static_cast<uint64_t>(philox_m[1]));

            counter[0][i] = v2[1] ^ counter[1][i] ^ key[0][i];
            counter[1][i] = v2[0];
            counter[2][i] = v1[1] ^ counter[3][i] ^ key[1][i];
            counter[3][i] = v1[0];
        }
    }

    // Generates 32-bit random numbers using the Philox 4x32 random number generator.
    // Parameters:
    //     counter : A 4xN array of 32-bit integers representing the counter values (offset into generation).
    //     key : A 2xN array of 32-bit integers representing the key values (seed).
    //     rounds : The number of rounds to perform.
    // Returns:
    //     std::vector<std::vector<uint32_t>>: A 4xN array of 32-bit integers containing the generated random numbers.
    std::vector<std::vector<uint32_t>> philox4_32(std::vector<std::vector<uint32_t>>& counter,
                                                  std::vector<std::vector<uint32_t>>& key,
                                                  int rounds = 10) {
        uint32_t N = (uint32_t)counter[0].size();
        for (int i = 0; i < rounds - 1; ++i) {
            philox4_round(counter, key);

            for (uint32_t j = 0; j < N; ++j) {
                key[0][j] += philox_w[0];
                key[1][j] += philox_w[1];
            }
        }

        philox4_round(counter, key);
        return counter;
    }

    float box_muller(float x, float y) {
        float u = x * two_pow32_inv + two_pow32_inv / 2;
        float v = y * two_pow32_inv_2pi + two_pow32_inv_2pi / 2;

        float s = sqrt(-2.0f * log(u));

        float r1 = s * sin(v);
        return r1;
    }

public:
    PhiloxRNG(uint64_t seed = 0) {
        this->seed   = seed;
        this->offset = 0;
    }

    void manual_seed(uint64_t seed) {
        this->seed   = seed;
        this->offset = 0;
    }

    std::vector<float> randn(uint32_t n) {
        std::vector<std::vector<uint32_t>> counter(4, std::vector<uint32_t>(n, 0));
        for (uint32_t i = 0; i < n; i++) {
            counter[0][i] = this->offset;
        }

        for (uint32_t i = 0; i < n; i++) {
            counter[2][i] = i;
        }
        this->offset += 1;

        std::vector<uint64_t> key(n, this->seed);
        std::vector<std::vector<uint32_t>> key_uint32 = uint32(key);

        std::vector<std::vector<uint32_t>> g = philox4_32(counter, key_uint32);

        std::vector<float> result;
        for (uint32_t i = 0; i < n; ++i) {
            result.push_back(box_muller((float)g[0][i], (float)g[1][i]));
        }
        return result;
    }
};

#endif  // __RNG_PHILOX_H__