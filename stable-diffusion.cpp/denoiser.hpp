#ifndef __DENOISER_HPP__
#define __DENOISER_HPP__

#include "ggml_extend.hpp"

/*================================================= CompVisDenoiser ==================================================*/

// Ref: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

#define TIMESTEPS 1000

struct SigmaSchedule {
    int version = 0;
    typedef std::function<float(float)> t_to_sigma_t;

    virtual std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) = 0;
};

struct DiscreteSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) {
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

    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) {
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
        std::vector<float> results(n + 1);

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
        if ((n + 1) != inputs.size()) {
            results = log_linear_interpolation(inputs, n + 1);
        } else {
            results = inputs;
        }

        /* Not sure if this is strictly neccessary */
        results[n] = 0.0f;

        return results;
    }
};

struct KarrasSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) {
        // These *COULD* be function arguments here,
        // but does anybody ever bother to touch them?
        float rho = 7.f;

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
    std::shared_ptr<SigmaSchedule> schedule                                                  = std::make_shared<DiscreteSchedule>();
    virtual float sigma_min()                                                                = 0;
    virtual float sigma_max()                                                                = 0;
    virtual float sigma_to_t(float sigma)                                                    = 0;
    virtual float t_to_sigma(float t)                                                        = 0;
    virtual std::vector<float> get_scalings(float sigma)                                     = 0;
    virtual ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) = 0;
    virtual ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent)             = 0;

    virtual std::vector<float> get_sigmas(uint32_t n) {
        auto bound_t_to_sigma = std::bind(&Denoiser::t_to_sigma, this, std::placeholders::_1);
        return schedule->get_sigmas(n, sigma_min(), sigma_max(), bound_t_to_sigma);
    }
};

struct CompVisDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

    float sigma_data = 1.0f;

    float sigma_min() {
        return sigmas[0];
    }

    float sigma_max() {
        return sigmas[TIMESTEPS - 1];
    }

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

    std::vector<float> get_scalings(float sigma) {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }

    // this function will modify noise/latent
    ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) {
        ggml_tensor_scale(noise, sigma);
        ggml_tensor_add(latent, noise);
        return latent;
    }

    ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent) {
        return latent;
    }
};

struct CompVisVDenoiser : public CompVisDenoiser {
    std::vector<float> get_scalings(float sigma) {
        float c_skip = sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data);
        float c_out  = -sigma * sigma_data / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }
};

float time_snr_shift(float alpha, float t) {
    if (alpha == 1.0f) {
        return t;
    }
    return alpha * t / (1 + (alpha - 1) * t);
}

struct DiscreteFlowDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float shift = 3.0f;

    float sigma_data = 1.0f;

    DiscreteFlowDenoiser() {
        set_parameters();
    }

    void set_parameters() {
        for (int i = 1; i < TIMESTEPS + 1; i++) {
            sigmas[i - 1] = t_to_sigma(i);
        }
    }

    float sigma_min() {
        return sigmas[0];
    }

    float sigma_max() {
        return sigmas[TIMESTEPS - 1];
    }

    float sigma_to_t(float sigma) {
        return sigma * 1000.f;
    }

    float t_to_sigma(float t) {
        t = t + 1;
        return time_snr_shift(shift, t / 1000.f);
    }

    std::vector<float> get_scalings(float sigma) {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f;
        return {c_skip, c_out, c_in};
    }

    // this function will modify noise/latent
    ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) {
        ggml_tensor_scale(noise, sigma);
        ggml_tensor_scale(latent, 1.0f - sigma);
        ggml_tensor_add(latent, noise);
        return latent;
    }

    ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent) {
        ggml_tensor_scale(latent, 1.0f / (1.0f - sigma));
        return latent;
    }
};

typedef std::function<ggml_tensor*(ggml_tensor*, float, int)> denoise_cb_t;

// k diffusion reverse ODE: dx = (x - D(x;\sigma)) / \sigma dt; \sigma(t) = t
static void sample_k_diffusion(sample_method_t method,
                               denoise_cb_t model,
                               ggml_context* work_ctx,
                               ggml_tensor* x,
                               std::vector<float> sigmas,
                               std::shared_ptr<RNG> rng) {
    size_t steps = sigmas.size() - 1;
    // sample_euler_ancestral
    switch (method) {
        case EULER_A: {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int i = 0; i < ggml_nelements(d); i++) {
                        vec_d[i] = (vec_x[i] - vec_denoised[i]) / sigma;
                    }
                }

                // get_ancestral_step
                float sigma_up   = std::min(sigmas[i + 1],
                                            std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);

                // Euler method
                float dt = sigma_down - sigmas[i];
                // x = x + d * dt
                {
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int i = 0; i < ggml_nelements(x); i++) {
                        vec_x[i] = vec_x[i] + vec_d[i] * dt;
                    }
                }

                if (sigmas[i + 1] > 0) {
                    // x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                    ggml_tensor_set_f32_randn(noise, rng);
                    // noise = load_tensor_from_file(work_ctx, "./rand" + std::to_string(i+1) + ".bin");
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int i = 0; i < ggml_nelements(x); i++) {
                            vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                        }
                    }
                }
            }
        } break;
        case EULER:  // Implemented without any sigma churn
        {
            struct ggml_tensor* d = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(d); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigma;
                    }
                }

                float dt = sigmas[i + 1] - sigma;
                // x = x + d * dt
                {
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                }
            }
        } break;
        case HEUN: {
            struct ggml_tensor* d  = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x2 = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], -(i + 1));

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                    }
                }

                float dt = sigmas[i + 1] - sigmas[i];
                if (sigmas[i + 1] == 0) {
                    // Euler step
                    // x = x + d * dt
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                } else {
                    // Heun step
                    float* vec_d  = (float*)d->data;
                    float* vec_d2 = (float*)d->data;
                    float* vec_x  = (float*)x->data;
                    float* vec_x2 = (float*)x2->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x2[j] = vec_x[j] + vec_d[j] * dt;
                    }

                    ggml_tensor* denoised = model(x2, sigmas[i + 1], i + 1);
                    float* vec_denoised   = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float d2 = (vec_x2[j] - vec_denoised[j]) / sigmas[i + 1];
                        vec_d[j] = (vec_d[j] + d2) / 2;
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                }
            }
        } break;
        case DPM2: {
            struct ggml_tensor* d  = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x2 = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                    }
                }

                if (sigmas[i + 1] == 0) {
                    // Euler step
                    // x = x + d * dt
                    float dt     = sigmas[i + 1] - sigmas[i];
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                } else {
                    // DPM-Solver-2
                    float sigma_mid = exp(0.5f * (log(sigmas[i]) + log(sigmas[i + 1])));
                    float dt_1      = sigma_mid - sigmas[i];
                    float dt_2      = sigmas[i + 1] - sigmas[i];

                    float* vec_d  = (float*)d->data;
                    float* vec_x  = (float*)x->data;
                    float* vec_x2 = (float*)x2->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x2[j] = vec_x[j] + vec_d[j] * dt_1;
                    }

                    ggml_tensor* denoised = model(x2, sigma_mid, i + 1);
                    float* vec_denoised   = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float d2 = (vec_x2[j] - vec_denoised[j]) / sigma_mid;
                        vec_x[j] = vec_x[j] + d2 * dt_2;
                    }
                }
            }

        } break;
        case DPMPP2S_A: {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x2    = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                // get_ancestral_step
                float sigma_up   = std::min(sigmas[i + 1],
                                            std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);
                auto t_fn        = [](float sigma) -> float { return -log(sigma); };
                auto sigma_fn    = [](float t) -> float { return exp(-t); };

                if (sigma_down == 0) {
                    // Euler step
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(d); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                    }

                    // TODO: If sigma_down == 0, isn't this wrong?
                    // But
                    // https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py#L525
                    // has this exactly the same way.
                    float dt = sigma_down - sigmas[i];
                    for (int j = 0; j < ggml_nelements(d); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                } else {
                    // DPM-Solver++(2S)
                    float t      = t_fn(sigmas[i]);
                    float t_next = t_fn(sigma_down);
                    float h      = t_next - t;
                    float s      = t + 0.5f * h;

                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_x2       = (float*)x2->data;
                    float* vec_denoised = (float*)denoised->data;

                    // First half-step
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x2[j] = (sigma_fn(s) / sigma_fn(t)) * vec_x[j] - (exp(-h * 0.5f) - 1) * vec_denoised[j];
                    }

                    ggml_tensor* denoised = model(x2, sigmas[i + 1], i + 1);

                    // Second half-step
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = (sigma_fn(t_next) / sigma_fn(t)) * vec_x[j] - (exp(-h) - 1) * vec_denoised[j];
                    }
                }

                // Noise addition
                if (sigmas[i + 1] > 0) {
                    ggml_tensor_set_f32_randn(noise, rng);
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int i = 0; i < ggml_nelements(x); i++) {
                            vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                        }
                    }
                }
            }
        } break;
        case DPMPP2M:  // DPM++ (2M) from Karras et al (2022)
        {
            struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

            auto t_fn = [](float sigma) -> float { return -log(sigma); };

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                float t                 = t_fn(sigmas[i]);
                float t_next            = t_fn(sigmas[i + 1]);
                float h                 = t_next - t;
                float a                 = sigmas[i + 1] / sigmas[i];
                float b                 = exp(-h) - 1.f;
                float* vec_x            = (float*)x->data;
                float* vec_denoised     = (float*)denoised->data;
                float* vec_old_denoised = (float*)old_denoised->data;

                if (i == 0 || sigmas[i + 1] == 0) {
                    // Simpler step for the edge cases
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                    }
                } else {
                    float h_last = t - t_fn(sigmas[i - 1]);
                    float r      = h_last / h;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                        vec_x[j]         = a * vec_x[j] - b * denoised_d;
                    }
                }

                // old_denoised = denoised
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_old_denoised[j] = vec_denoised[j];
                }
            }
        } break;
        case DPMPP2Mv2:  // Modified DPM++ (2M) from https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457
        {
            struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

            auto t_fn = [](float sigma) -> float { return -log(sigma); };

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                float t                 = t_fn(sigmas[i]);
                float t_next            = t_fn(sigmas[i + 1]);
                float h                 = t_next - t;
                float a                 = sigmas[i + 1] / sigmas[i];
                float* vec_x            = (float*)x->data;
                float* vec_denoised     = (float*)denoised->data;
                float* vec_old_denoised = (float*)old_denoised->data;

                if (i == 0 || sigmas[i + 1] == 0) {
                    // Simpler step for the edge cases
                    float b = exp(-h) - 1.f;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                    }
                } else {
                    float h_last = t - t_fn(sigmas[i - 1]);
                    float h_min  = std::min(h_last, h);
                    float h_max  = std::max(h_last, h);
                    float r      = h_max / h_min;
                    float h_d    = (h_max + h_min) / 2.f;
                    float b      = exp(-h_d) - 1.f;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                        vec_x[j]         = a * vec_x[j] - b * denoised_d;
                    }
                }

                // old_denoised = denoised
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_old_denoised[j] = vec_denoised[j];
                }
            }
        } break;
        case LCM:  // Latent Consistency Models
        {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);

                // x = denoised
                {
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_denoised[j];
                    }
                }

                if (sigmas[i + 1] > 0) {
                    // x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
                    ggml_tensor_set_f32_randn(noise, rng);
                    // noise = load_tensor_from_file(res_ctx, "./rand" + std::to_string(i+1) + ".bin");
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + sigmas[i + 1] * vec_noise[j];
                        }
                    }
                }
            }
        } break;

        default:
            LOG_ERROR("Attempting to sample with nonexisting sample method %i", method);
            abort();
    }
}

#endif  // __DENOISER_HPP__
