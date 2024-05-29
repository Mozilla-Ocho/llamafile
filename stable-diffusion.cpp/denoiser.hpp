#ifndef __DENOISER_HPP__
#define __DENOISER_HPP__

#include "ggml_extend.hpp"

/*================================================= CompVisDenoiser ==================================================*/

// Ref: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

#define TIMESTEPS 1000

struct SigmaSchedule {
    float alphas_cumprod[TIMESTEPS];
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];
    int version = 0;

    virtual std::vector<float> get_sigmas(uint32_t n) = 0;

    float sigma_to_t(float sigma) {
        float log_sigma = std::log(sigma);
        std::vector<float> dists;
        dists.reserve(TIMESTEPS);
        for (float log_sigma_val : log_sigmas) {
            dists.push_back(log_sigma - log_sigma_val);
        }

        int low_idx = 0;
        for (size_t i = 0; i < TIMESTEPS; i++) {
            if (dists[i] >= 0) {
                low_idx++;
            }
        }
        low_idx      = std::min(std::max(low_idx - 1, 0), TIMESTEPS - 2);
        int high_idx = low_idx + 1;

        float low  = log_sigmas[low_idx];
        float high = log_sigmas[high_idx];
        float w    = (low - log_sigma) / (low - high);
        w          = std::max(0.f, std::min(1.f, w));
        float t    = (1.0f - w) * low_idx + w * high_idx;

        return t;
    }

    float t_to_sigma(float t) {
        int low_idx     = static_cast<int>(std::floor(t));
        int high_idx    = static_cast<int>(std::ceil(t));
        float w         = t - static_cast<float>(low_idx);
        float log_sigma = (1.0f - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx];
        return std::exp(log_sigma);
    }
};

struct DiscreteSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n) {
        std::vector<float> result;

        int t_max = TIMESTEPS - 1;

        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0);
            return result;
        }

        float step = static_cast<float>(t_max) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            float t = t_max - step * i;
            result.push_back(t_to_sigma(t));
        }
        result.push_back(0);
        return result;
    }
};

/*
https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
*/
struct AYSSchedule : SigmaSchedule {
    /* interp and linear_interp adapted from dpilger26's NumCpp library:
     * https://github.com/dpilger26/NumCpp/tree/5e40aab74d14e257d65d3dc385c9ff9e2120c60e */
    constexpr double interp(double left, double right, double perc) noexcept {
        return (left * (1. - perc)) + (right * perc);
    }

    /* This will make the assumption that the reference x and y values are
     * already sorted in ascending order because they are being generated as
     * such in the calling function */
    std::vector<double> linear_interp(std::vector<float> new_x,
                                      const std::vector<float> ref_x,
                                      const std::vector<float> ref_y) {
        const size_t len_x = new_x.size();
        size_t i           = 0;
        size_t j           = 0;
        std::vector<double> new_y(len_x);

        if (ref_x.size() != ref_y.size()) {
            LOG_ERROR("Linear Interoplation Failed: length mismatch");
            return new_y;
        }

        /* serves as the bounds checking for the below while loop */
        if ((new_x[0] < ref_x[0]) || (new_x[new_x.size() - 1] > ref_x[ref_x.size() - 1])) {
            LOG_ERROR("Linear Interpolation Failed: bad bounds");
            return new_y;
        }

        while (i < len_x) {
            if ((ref_x[j] > new_x[i]) || (new_x[i] > ref_x[j + 1])) {
                j++;
                continue;
            }

            const double perc = static_cast<double>(new_x[i] - ref_x[j]) / static_cast<double>(ref_x[j + 1] - ref_x[j]);

            new_y[i] = interp(ref_y[j], ref_y[j + 1], perc);
            i++;
        }

        return new_y;
    }

    std::vector<float> linear_space(const float start, const float end, const size_t num_points) {
        std::vector<float> result(num_points);
        const float inc = (end - start) / (static_cast<float>(num_points - 1));

        if (num_points > 0) {
            result[0] = start;

            for (size_t i = 1; i < num_points; i++) {
                result[i] = result[i - 1] + inc;
            }
        }

        return result;
    }

    std::vector<float> log_linear_interpolation(std::vector<float> sigma_in,
                                                const size_t new_len) {
        const size_t s_len        = sigma_in.size();
        std::vector<float> x_vals = linear_space(0.f, 1.f, s_len);
        std::vector<float> y_vals(s_len);

        /* Reverses the input array to be ascending instead of descending,
         * also hits it with a log, it is log-linear interpolation after all */
        for (size_t i = 0; i < s_len; i++) {
            y_vals[i] = std::log(sigma_in[s_len - i - 1]);
        }

        std::vector<float> new_x_vals  = linear_space(0.f, 1.f, new_len);
        std::vector<double> new_y_vals = linear_interp(new_x_vals, x_vals, y_vals);
        std::vector<float> results(new_len);

        for (size_t i = 0; i < new_len; i++) {
            results[i] = static_cast<float>(std::exp(new_y_vals[new_len - i - 1]));
        }

        return results;
    }

    std::vector<float> get_sigmas(uint32_t len) {
        const std::vector<float> noise_levels[] = {
            /* SD1.5 */
            {14.6146412293f, 6.4745760956f, 3.8636745985f, 2.6946151520f,
             1.8841921177f, 1.3943805092f, 0.9642583904f, 0.6523686016f,
             0.3977456272f, 0.1515232662f, 0.0291671582f},
            /* SDXL */
            {14.6146412293f, 6.3184485287f, 3.7681790315f, 2.1811480769f,
             1.3405244945f, 0.8620721141f, 0.5550693289f, 0.3798540708f,
             0.2332364134f, 0.1114188177f, 0.0291671582f},
            /* SVD */
            {700.00f, 54.5f, 15.886f, 7.977f, 4.248f, 1.789f, 0.981f, 0.403f,
             0.173f, 0.034f, 0.002f},
        };

        std::vector<float> inputs;
        std::vector<float> results(len + 1);

        switch (version) {
            case VERSION_2_x: /* fallthrough */
                LOG_WARN("AYS not designed for SD2.X models");
            case VERSION_1_x:
                LOG_INFO("AYS using SD1.5 noise levels");
                inputs = noise_levels[0];
                break;
            case VERSION_XL:
                LOG_INFO("AYS using SDXL noise levels");
                inputs = noise_levels[1];
                break;
            case VERSION_SVD:
                LOG_INFO("AYS using SVD noise levels");
                inputs = noise_levels[2];
                break;
            default:
                LOG_ERROR("Version not compatable with AYS scheduler");
                return results;
        }

        /* Stretches those pre-calculated reference levels out to the desired
         * size using log-linear interpolation */
        if ((len + 1) != inputs.size()) {
            results = log_linear_interpolation(inputs, len + 1);
        } else {
            results = inputs;
        }

        /* Not sure if this is strictly neccessary */
        results[len] = 0.0f;

        return results;
    }
};

struct KarrasSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n) {
        // These *COULD* be function arguments here,
        // but does anybody ever bother to touch them?
        float sigma_min = 0.1f;
        float sigma_max = 10.f;
        float rho       = 7.f;

        std::vector<float> result(n + 1);

        float min_inv_rho = pow(sigma_min, (1.f / rho));
        float max_inv_rho = pow(sigma_max, (1.f / rho));
        for (uint32_t i = 0; i < n; i++) {
            // Eq. (5) from Karras et al 2022
            result[i] = pow(max_inv_rho + (float)i / ((float)n - 1.f) * (min_inv_rho - max_inv_rho), rho);
        }
        result[n] = 0.;
        return result;
    }
};

struct Denoiser {
    std::shared_ptr<SigmaSchedule> schedule              = std::make_shared<DiscreteSchedule>();
    virtual std::vector<float> get_scalings(float sigma) = 0;
};

struct CompVisDenoiser : public Denoiser {
    float sigma_data = 1.0f;

    std::vector<float> get_scalings(float sigma) {
        float c_out = -sigma;
        float c_in  = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_out, c_in};
    }
};

struct CompVisVDenoiser : public Denoiser {
    float sigma_data = 1.0f;

    std::vector<float> get_scalings(float sigma) {
        float c_skip = sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data);
        float c_out  = -sigma * sigma_data / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }
};

#endif  // __DENOISER_HPP__
