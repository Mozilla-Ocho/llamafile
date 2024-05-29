#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "clip.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "tae.hpp"
#include "unet.hpp"
#include "vae.hpp"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

const char* model_version_to_str[] = {
    "1.x",
    "2.x",
    "XL",
    "SVD",
};

const char* sampling_methods_str[] = {
    "Euler A",
    "Euler",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "LCM",
};

/*================================================== Helper Functions ================================================*/

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120,
                              int timesteps      = TIMESTEPS) {
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_backend_t backend             = NULL;  // general backend
    ggml_backend_t clip_backend        = NULL;
    ggml_backend_t control_net_backend = NULL;
    ggml_backend_t vae_backend         = NULL;
    ggml_type model_data_type          = GGML_TYPE_COUNT;

    SDVersion version;
    bool vae_decode_only         = false;
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    int n_threads            = -1;
    float scale_factor       = 0.18215f;

    std::shared_ptr<FrozenCLIPEmbedderWithCustomWords> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd
    std::shared_ptr<UNetModel> diffusion_model;
    std::shared_ptr<AutoEncoderKL> first_stage_model;
    std::shared_ptr<TinyAutoEncoder> tae_first_stage;
    std::shared_ptr<ControlNet> control_net;
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;
    std::shared_ptr<LoraModel> pmid_lora;

    std::string taesd_path;
    bool use_tiny_autoencoder = false;
    bool vae_tiling           = false;
    bool stacked_id           = false;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::string lora_model_dir;
    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    std::string trigger_word = "img";  // should be user settable

    StableDiffusionGGML() = default;

    StableDiffusionGGML(int n_threads,
                        bool vae_decode_only,
                        bool free_params_immediately,
                        std::string lora_model_dir,
                        rng_type_t rng_type)
        : n_threads(n_threads),
          vae_decode_only(vae_decode_only),
          free_params_immediately(free_params_immediately),
          lora_model_dir(lora_model_dir) {
        if (rng_type == STD_DEFAULT_RNG) {
            rng = std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CUDA_RNG) {
            rng = std::make_shared<PhiloxRNG>();
        }
    }

    ~StableDiffusionGGML() {
        if (clip_backend != backend) {
            ggml_backend_free(clip_backend);
        }
        if (control_net_backend != backend) {
            ggml_backend_free(control_net_backend);
        }
        if (vae_backend != backend) {
            ggml_backend_free(vae_backend);
        }
        ggml_backend_free(backend);
    }

    bool load_from_file(const std::string& model_path,
                        const std::string& vae_path,
                        const std::string control_net_path,
                        const std::string embeddings_path,
                        const std::string id_embeddings_path,
                        const std::string& taesd_path,
                        bool vae_tiling_,
                        ggml_type wtype,
                        schedule_t schedule,
                        bool clip_on_cpu,
                        bool control_net_cpu,
                        bool vae_on_cpu) {
        use_tiny_autoencoder = taesd_path.size() > 0;
#ifdef SD_USE_CUBLAS
        LOG_DEBUG("Using CUDA backend");
        backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
        LOG_DEBUG("Using Metal backend");
        ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
        backend = ggml_backend_metal_init();
#endif

        if (!backend) {
            LOG_DEBUG("Using CPU backend");
            backend = ggml_backend_cpu_init();
        }
#ifdef SD_USE_FLASH_ATTENTION
#if defined(SD_USE_CUBLAS) || defined(SD_USE_METAL)
        LOG_WARN("Flash Attention not supported with GPU Backend");
#else
        LOG_INFO("Flash Attention enabled");
#endif
#endif
        LOG_INFO("loading model from '%s'", model_path.c_str());
        ModelLoader model_loader;

        vae_tiling = vae_tiling_;

        if (!model_loader.init_from_file(model_path)) {
            LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
            return false;
        }

        if (vae_path.size() > 0) {
            LOG_INFO("loading vae from '%s'", vae_path.c_str());
            if (!model_loader.init_from_file(vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", vae_path.c_str());
            }
        }

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", model_path.c_str());
            return false;
        }

        LOG_INFO("Stable Diffusion %s ", model_version_to_str[version]);
        if (wtype == GGML_TYPE_COUNT) {
            model_data_type = model_loader.get_sd_wtype();
        } else {
            model_data_type = wtype;
        }
        LOG_INFO("Stable Diffusion weight type: %s", ggml_type_name(model_data_type));
        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (version == VERSION_XL) {
            scale_factor = 0.13025f;
            if (vae_path.size() == 0 && taesd_path.size() == 0) {
                LOG_WARN(
                    "!!!It looks like you are using SDXL model. "
                    "If you find that the generated images are completely black, "
                    "try specifying SDXL VAE FP16 Fix with the --vae parameter. "
                    "You can find it here: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors");
            }
        }

        if (version == VERSION_SVD) {
            clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend, model_data_type);
            clip_vision->alloc_params_buffer();
            clip_vision->get_param_tensors(tensors, "cond_stage_model.");

            diffusion_model = std::make_shared<UNetModel>(backend, model_data_type, version);
            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors, "model.diffusion_model");

            first_stage_model = std::make_shared<AutoEncoderKL>(backend, model_data_type, vae_decode_only, true);
            LOG_DEBUG("vae_decode_only %d", vae_decode_only);
            first_stage_model->alloc_params_buffer();
            first_stage_model->get_param_tensors(tensors, "first_stage_model");
        } else {
            clip_backend = backend;
            if (clip_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("CLIP: Using CPU backend");
                clip_backend = ggml_backend_cpu_init();
            }
            cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend, model_data_type, version);
            cond_stage_model->alloc_params_buffer();
            cond_stage_model->get_param_tensors(tensors, "cond_stage_model.");

            cond_stage_model->embd_dir = embeddings_path;

            diffusion_model = std::make_shared<UNetModel>(backend, model_data_type, version);
            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors, "model.diffusion_model");

            ggml_type vae_type = model_data_type;
            if (version == VERSION_XL) {
                vae_type = GGML_TYPE_F32;  // avoid nan, not work...
            }

            if (!use_tiny_autoencoder) {
                if (vae_on_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_INFO("VAE Autoencoder: Using CPU backend");
                    vae_backend = ggml_backend_cpu_init();
                } else {
                    vae_backend = backend;
                }
                first_stage_model = std::make_shared<AutoEncoderKL>(vae_backend, vae_type, vae_decode_only);
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "first_stage_model");
            } else {
                tae_first_stage = std::make_shared<TinyAutoEncoder>(backend, model_data_type, vae_decode_only);
            }
            // first_stage_model->get_param_tensors(tensors, "first_stage_model.");

            if (control_net_path.size() > 0) {
                ggml_backend_t controlnet_backend = NULL;
                if (control_net_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_DEBUG("ControlNet: Using CPU backend");
                    controlnet_backend = ggml_backend_cpu_init();
                } else {
                    controlnet_backend = backend;
                }
                control_net = std::make_shared<ControlNet>(controlnet_backend, model_data_type, version);
            }

            pmid_model = std::make_shared<PhotoMakerIDEncoder>(clip_backend, model_data_type, version);
            if (id_embeddings_path.size() > 0) {
                pmid_lora = std::make_shared<LoraModel>(backend, model_data_type, id_embeddings_path, "");
                if (!pmid_lora->load_from_file(true)) {
                    LOG_WARN("load photomaker lora tensors from %s failed", id_embeddings_path.c_str());
                    return false;
                }
                LOG_INFO("loading stacked ID embedding (PHOTOMAKER) model file from '%s'", id_embeddings_path.c_str());
                if (!model_loader.init_from_file(id_embeddings_path, "pmid.")) {
                    LOG_WARN("loading stacked ID embedding from '%s' failed", id_embeddings_path.c_str());
                } else {
                    stacked_id = true;
                }
            }
            if (stacked_id) {
                if (!pmid_model->alloc_params_buffer()) {
                    LOG_ERROR(" pmid model params buffer allocation failed");
                    return false;
                }
                // LOG_INFO("pmid param memory buffer size = %.2fMB ",
                //     pmid_model->params_buffer_size / 1024.0 / 1024.0);
                pmid_model->get_param_tensors(tensors, "pmid");
            }
            // if(stacked_id){
            //    pmid_model.init_params(GGML_TYPE_F32);
            //    pmid_model.map_by_name(tensors, "pmid.");
            // }

            LOG_DEBUG("loading vocab");
            std::string merges_utf8_str = model_loader.load_merges();
            if (merges_utf8_str.size() == 0) {
                LOG_ERROR("get merges failed: '%s'", model_path.c_str());
                return false;
            }
            cond_stage_model->tokenizer.load_from_merges(merges_utf8_str);
        }

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        struct ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != NULL);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        int64_t t0 = ggml_time_ms();

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tiny_autoencoder) {
            ignore_tensors.insert("first_stage_model.");
        }
        if (stacked_id) {
            ignore_tensors.insert("lora.");
        }

        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.quant");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        bool success = model_loader.load_tensors(tensors, backend, ignore_tensors);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        // LOG_DEBUG("model size = %.2fMB", total_size / 1024.0 / 1024.0);

        if (version == VERSION_SVD) {
            // diffusion_model->test();
            // first_stage_model->test();
            // return false;
        } else {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
            size_t vae_params_mem_size  = 0;
            if (!use_tiny_autoencoder) {
                vae_params_mem_size = first_stage_model->get_params_buffer_size();
            } else {
                if (!tae_first_stage->load_from_file(taesd_path)) {
                    return false;
                }
                vae_params_mem_size = tae_first_stage->get_params_buffer_size();
            }
            size_t control_net_params_mem_size = 0;
            if (control_net) {
                if (!control_net->load_from_file(control_net_path)) {
                    return false;
                }
                control_net_params_mem_size = control_net->get_params_buffer_size();
            }
            size_t pmid_params_mem_size = 0;
            if (stacked_id) {
                pmid_params_mem_size = pmid_model->get_params_buffer_size();
            }

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            if (ggml_backend_is_cpu(clip_backend)) {
                total_params_ram_size += clip_params_mem_size + pmid_params_mem_size;
            } else {
                total_params_vram_size += clip_params_mem_size + pmid_params_mem_size;
            }

            if (ggml_backend_is_cpu(backend)) {
                total_params_ram_size += unet_params_mem_size;
            } else {
                total_params_vram_size += unet_params_mem_size;
            }

            if (ggml_backend_is_cpu(vae_backend)) {
                total_params_ram_size += vae_params_mem_size;
            } else {
                total_params_vram_size += vae_params_mem_size;
            }

            if (ggml_backend_is_cpu(control_net_backend)) {
                total_params_ram_size += control_net_params_mem_size;
            } else {
                total_params_vram_size += control_net_params_mem_size;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "clip %.2fMB(%s), unet %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), pmid %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM",
                unet_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                vae_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM",
                control_net_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(control_net_backend) ? "RAM" : "VRAM",
                pmid_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM");
        }

        int64_t t1 = ggml_time_ms();
        LOG_INFO("loading model from '%s' completed, taking %.2fs", model_path.c_str(), (t1 - t0) * 1.0f / 1000);

        // check is_using_v_parameterization_for_sd2
        bool is_using_v_parameterization = false;
        if (version == VERSION_2_x) {
            if (is_using_v_parameterization_for_sd2(ctx)) {
                is_using_v_parameterization = true;
            }
        } else if (version == VERSION_SVD) {
            // TODO: V_PREDICTION_EDM
            is_using_v_parameterization = true;
        }

        if (is_using_v_parameterization) {
            denoiser = std::make_shared<CompVisVDenoiser>();
            LOG_INFO("running in v-prediction mode");
        } else {
            LOG_INFO("running in eps-prediction mode");
        }

        if (schedule != DEFAULT) {
            switch (schedule) {
                case DISCRETE:
                    LOG_INFO("running with discrete schedule");
                    denoiser->schedule = std::make_shared<DiscreteSchedule>();
                    break;
                case KARRAS:
                    LOG_INFO("running with Karras schedule");
                    denoiser->schedule = std::make_shared<KarrasSchedule>();
                    break;
                case AYS:
                    LOG_INFO("Running with Align-Your-Steps schedule");
                    denoiser->schedule          = std::make_shared<AYSSchedule>();
                    denoiser->schedule->version = version;
                    break;
                case DEFAULT:
                    // Don't touch anything.
                    break;
                default:
                    LOG_ERROR("Unknown schedule %i", schedule);
                    abort();
            }
        }

        for (int i = 0; i < TIMESTEPS; i++) {
            denoiser->schedule->alphas_cumprod[i] = ((float*)alphas_cumprod_tensor->data)[i];
            denoiser->schedule->sigmas[i]         = std::sqrt((1 - denoiser->schedule->alphas_cumprod[i]) / denoiser->schedule->alphas_cumprod[i]);
            denoiser->schedule->log_sigmas[i]     = std::log(denoiser->schedule->sigmas[i]);
        }

        LOG_DEBUG("finished loaded file");
        ggml_free(ctx);
        return true;
    }

    bool is_using_v_parameterization_for_sd2(ggml_context* work_ctx) {
        struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 4, 1);
        ggml_set_f32(x_t, 0.5);
        struct ggml_tensor* c = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 1024, 2, 1, 1);
        ggml_set_f32(c, 0.5);

        struct ggml_tensor* timesteps = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
        ggml_set_f32(timesteps, 999);
        int64_t t0              = ggml_time_ms();
        struct ggml_tensor* out = ggml_dup_tensor(work_ctx, x_t);
        diffusion_model->compute(n_threads, x_t, timesteps, c, NULL, NULL, -1, {}, 0.f, &out);
        diffusion_model->free_compute_buffer();

        double result = 0.f;
        {
            float* vec_x   = (float*)x_t->data;
            float* vec_out = (float*)out->data;

            int64_t n = ggml_nelements(out);

            for (int i = 0; i < n; i++) {
                result += ((double)vec_out[i] - (double)vec_x[i]);
            }
            result /= n;
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    void apply_lora(const std::string& lora_name, float multiplier) {
        int64_t t0                 = ggml_time_ms();
        std::string st_file_path   = path_join(lora_model_dir, lora_name + ".safetensors");
        std::string ckpt_file_path = path_join(lora_model_dir, lora_name + ".ckpt");
        std::string file_path;
        if (file_exists(st_file_path)) {
            file_path = st_file_path;
        } else if (file_exists(ckpt_file_path)) {
            file_path = ckpt_file_path;
        } else {
            LOG_WARN("can not find %s or %s for lora %s", st_file_path.c_str(), ckpt_file_path.c_str(), lora_name.c_str());
            return;
        }
        LoraModel lora(backend, model_data_type, file_path);
        if (!lora.load_from_file()) {
            LOG_WARN("load lora tensors from %s failed", file_path.c_str());
            return;
        }

        lora.multiplier = multiplier;
        lora.apply(tensors, n_threads);
        lora.free_params_buffer();

        int64_t t1 = ggml_time_ms();

        LOG_INFO("lora '%s' applied, taking %.2fs", lora_name.c_str(), (t1 - t0) * 1.0f / 1000);
    }

    void apply_loras(const std::unordered_map<std::string, float>& lora_state) {
        if (lora_state.size() > 0 && model_data_type != GGML_TYPE_F16 && model_data_type != GGML_TYPE_F32) {
            LOG_WARN("In quantized models when applying LoRA, the images have poor quality.");
        }
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;

            if (curr_lora_state.find(lora_name) != curr_lora_state.end()) {
                float curr_multiplier = curr_lora_state[lora_name];
                float multiplier_diff = multiplier - curr_multiplier;
                if (multiplier_diff != 0.f) {
                    lora_state_diff[lora_name] = multiplier_diff;
                }
            } else {
                lora_state_diff[lora_name] = multiplier;
            }
        }

        LOG_INFO("Attempting to apply %lu LoRAs", lora_state.size());

        for (auto& kv : lora_state_diff) {
            apply_lora(kv.first, kv.second);
        }

        curr_lora_state = lora_state;
    }

    std::string remove_trigger_from_prompt(ggml_context* work_ctx,
                                           const std::string& prompt) {
        auto image_tokens = cond_stage_model->convert_token_to_id(trigger_word);
        GGML_ASSERT(image_tokens.size() == 1);
        auto tokens_and_weights  = cond_stage_model->tokenize(prompt, false);
        std::vector<int>& tokens = tokens_and_weights.first;
        auto it                  = std::find(tokens.begin(), tokens.end(), image_tokens[0]);
        GGML_ASSERT(it != tokens.end());  // prompt must have trigger word
        tokens.erase(it);
        return cond_stage_model->decode(tokens);
    }

    std::tuple<ggml_tensor*, ggml_tensor*, std::vector<bool>>
    get_learned_condition_with_trigger(ggml_context* work_ctx,
                                       const std::string& text,
                                       int clip_skip,
                                       int width,
                                       int height,
                                       int num_input_imgs,
                                       bool force_zero_embeddings = false) {
        auto image_tokens = cond_stage_model->convert_token_to_id(trigger_word);
        // if(image_tokens.size() == 1){
        //     printf(" image token id is: %d \n", image_tokens[0]);
        // }
        GGML_ASSERT(image_tokens.size() == 1);
        auto tokens_and_weights     = cond_stage_model->tokenize_with_trigger_token(text,
                                                                                    num_input_imgs,
                                                                                    image_tokens[0],
                                                                                    true);
        std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
        std::vector<float>& weights = std::get<1>(tokens_and_weights);
        std::vector<bool>& clsm     = std::get<2>(tokens_and_weights);
        // printf("tokens: \n");
        // for(int i = 0; i < tokens.size(); ++i)
        //    printf("%d ", tokens[i]);
        // printf("\n");
        // printf("clsm: \n");
        // for(int i = 0; i < clsm.size(); ++i)
        //    printf("%d ", clsm[i]?1:0);
        // printf("\n");
        auto cond = get_learned_condition_common(work_ctx, tokens, weights, clip_skip, width, height, force_zero_embeddings);
        return std::make_tuple(cond.first, cond.second, clsm);
    }

    ggml_tensor* id_encoder(ggml_context* work_ctx,
                            ggml_tensor* init_img,
                            ggml_tensor* prompts_embeds,
                            std::vector<bool>& class_tokens_mask) {
        ggml_tensor* res = NULL;
        pmid_model->compute(n_threads, init_img, prompts_embeds, class_tokens_mask, &res, work_ctx);

        return res;
    }

    std::pair<ggml_tensor*, ggml_tensor*> get_learned_condition(ggml_context* work_ctx,
                                                                const std::string& text,
                                                                int clip_skip,
                                                                int width,
                                                                int height,
                                                                bool force_zero_embeddings = false) {
        auto tokens_and_weights     = cond_stage_model->tokenize(text, true);
        std::vector<int>& tokens    = tokens_and_weights.first;
        std::vector<float>& weights = tokens_and_weights.second;
        return get_learned_condition_common(work_ctx, tokens, weights, clip_skip, width, height, force_zero_embeddings);
    }

    std::pair<ggml_tensor*, ggml_tensor*> get_learned_condition_common(ggml_context* work_ctx,
                                                                       std::vector<int>& tokens,
                                                                       std::vector<float>& weights,
                                                                       int clip_skip,
                                                                       int width,
                                                                       int height,
                                                                       bool force_zero_embeddings = false) {
        cond_stage_model->set_clip_skip(clip_skip);
        int64_t t0                              = ggml_time_ms();
        struct ggml_tensor* hidden_states       = NULL;  // [N, n_token, hidden_size]
        struct ggml_tensor* chunk_hidden_states = NULL;  // [n_token, hidden_size]
        struct ggml_tensor* pooled              = NULL;
        std::vector<float> hidden_states_vec;

        size_t chunk_len   = 77;
        size_t chunk_count = tokens.size() / chunk_len;
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            std::vector<int> chunk_tokens(tokens.begin() + chunk_idx * chunk_len,
                                          tokens.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_weights(weights.begin() + chunk_idx * chunk_len,
                                             weights.begin() + (chunk_idx + 1) * chunk_len);

            auto input_ids                 = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
            struct ggml_tensor* input_ids2 = NULL;
            size_t max_token_idx           = 0;
            if (version == VERSION_XL) {
                auto it = std::find(chunk_tokens.begin(), chunk_tokens.end(), EOS_TOKEN_ID);
                if (it != chunk_tokens.end()) {
                    std::fill(std::next(it), chunk_tokens.end(), 0);
                }

                max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

                input_ids2 = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);

                // for (int i = 0; i < chunk_tokens.size(); i++) {
                //     printf("%d ", chunk_tokens[i]);
                // }
                // printf("\n");
            }

            cond_stage_model->compute(n_threads, input_ids, input_ids2, max_token_idx, false, &chunk_hidden_states, work_ctx);
            if (version == VERSION_XL && chunk_idx == 0) {
                cond_stage_model->compute(n_threads, input_ids, input_ids2, max_token_idx, true, &pooled, work_ctx);
            }
            // if (pooled != NULL) {
            //     print_ggml_tensor(chunk_hidden_states);
            //     print_ggml_tensor(pooled);
            // }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            ggml_tensor* result = ggml_dup_tensor(work_ctx, chunk_hidden_states);
            {
                float original_mean = ggml_tensor_mean(chunk_hidden_states);
                for (int i2 = 0; i2 < chunk_hidden_states->ne[2]; i2++) {
                    for (int i1 = 0; i1 < chunk_hidden_states->ne[1]; i1++) {
                        for (int i0 = 0; i0 < chunk_hidden_states->ne[0]; i0++) {
                            float value = ggml_tensor_get_f32(chunk_hidden_states, i0, i1, i2);
                            value *= chunk_weights[i1];
                            ggml_tensor_set_f32(result, value, i0, i1, i2);
                        }
                    }
                }
                float new_mean = ggml_tensor_mean(result);
                ggml_tensor_scale(result, (original_mean / new_mean));
            }
            if (force_zero_embeddings) {
                float* vec = (float*)result->data;
                for (int i = 0; i < ggml_nelements(result); i++) {
                    vec[i] = 0;
                }
            }
            hidden_states_vec.insert(hidden_states_vec.end(), (float*)result->data, ((float*)result->data) + ggml_nelements(result));
        }

        hidden_states = vector_to_ggml_tensor(work_ctx, hidden_states_vec);
        hidden_states = ggml_reshape_2d(work_ctx,
                                        hidden_states,
                                        chunk_hidden_states->ne[0],
                                        ggml_nelements(hidden_states) / chunk_hidden_states->ne[0]);

        ggml_tensor* vec = NULL;
        if (version == VERSION_XL) {
            int out_dim = 256;
            vec         = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, diffusion_model->unet.adm_in_channels);
            // [0:1280]
            size_t offset = 0;
            memcpy(vec->data, pooled->data, ggml_nbytes(pooled));
            offset += ggml_nbytes(pooled);

            // original_size_as_tuple
            float orig_width             = (float)width;
            float orig_height            = (float)height;
            std::vector<float> timesteps = {orig_height, orig_width};

            ggml_tensor* embed_view = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // crop_coords_top_left
            float crop_coord_top  = 0.f;
            float crop_coord_left = 0.f;
            timesteps             = {crop_coord_top, crop_coord_left};
            embed_view            = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // target_size_as_tuple
            float target_width  = (float)width;
            float target_height = (float)height;
            timesteps           = {target_height, target_width};
            embed_view          = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            GGML_ASSERT(offset == ggml_nbytes(vec));
        }
        // print_ggml_tensor(result);
        return {hidden_states, vec};
    }

    std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> get_svd_condition(ggml_context* work_ctx,
                                                                           sd_image_t init_image,
                                                                           int width,
                                                                           int height,
                                                                           int fps                    = 6,
                                                                           int motion_bucket_id       = 127,
                                                                           float augmentation_level   = 0.f,
                                                                           bool force_zero_embeddings = false) {
        // c_crossattn
        int64_t t0                      = ggml_time_ms();
        struct ggml_tensor* c_crossattn = NULL;
        {
            if (force_zero_embeddings) {
                c_crossattn = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, clip_vision->vision_model.projection_dim);
                ggml_set_f32(c_crossattn, 0.f);
            } else {
                sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                sd_image_f32_t resized_image = clip_preprocess(image, clip_vision->vision_model.image_size);
                free(image.data);
                image.data = NULL;

                ggml_tensor* pixel_values = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, resized_image.width, resized_image.height, 3, 1);
                sd_image_f32_to_tensor(resized_image.data, pixel_values, false);
                free(resized_image.data);
                resized_image.data = NULL;

                // print_ggml_tensor(pixel_values);
                clip_vision->compute(n_threads, pixel_values, &c_crossattn, work_ctx);
                // print_ggml_tensor(c_crossattn);
            }
        }

        // c_concat
        struct ggml_tensor* c_concat = NULL;
        {
            if (force_zero_embeddings) {
                c_concat = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / 8, height / 8, 4, 1);
                ggml_set_f32(c_concat, 0.f);
            } else {
                ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);

                if (width != init_image.width || height != init_image.height) {
                    sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                    sd_image_f32_t resized_image = resize_sd_image_f32_t(image, width, height);
                    free(image.data);
                    image.data = NULL;
                    sd_image_f32_to_tensor(resized_image.data, init_img, false);
                    free(resized_image.data);
                    resized_image.data = NULL;
                } else {
                    sd_image_to_tensor(init_image.data, init_img);
                }
                if (augmentation_level > 0.f) {
                    struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, init_img);
                    ggml_tensor_set_f32_randn(noise, rng);
                    // encode_pixels += torch.randn_like(pixels) * augmentation_level
                    ggml_tensor_scale(noise, augmentation_level);
                    ggml_tensor_add(init_img, noise);
                }
                print_ggml_tensor(init_img);
                ggml_tensor* moments = encode_first_stage(work_ctx, init_img);
                print_ggml_tensor(moments);
                c_concat = get_first_stage_encoding(work_ctx, moments);
            }
            print_ggml_tensor(c_concat);
        }

        // y
        struct ggml_tensor* y = NULL;
        {
            y                            = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, diffusion_model->unet.adm_in_channels);
            int out_dim                  = 256;
            int fps_id                   = fps - 1;
            std::vector<float> timesteps = {(float)fps_id, (float)motion_bucket_id, augmentation_level};
            set_timestep_embedding(timesteps, y, out_dim);
            print_ggml_tensor(y);
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing svd condition graph completed, taking %" PRId64 " ms", t1 - t0);
        return {c_crossattn, c_concat, y};
    }

    ggml_tensor* sample(ggml_context* work_ctx,
                        ggml_tensor* x_t,
                        ggml_tensor* noise,
                        ggml_tensor* c,
                        ggml_tensor* c_concat,
                        ggml_tensor* c_vector,
                        ggml_tensor* uc,
                        ggml_tensor* uc_concat,
                        ggml_tensor* uc_vector,
                        ggml_tensor* control_hint,
                        float control_strength,
                        float min_cfg,
                        float cfg_scale,
                        sample_method_t method,
                        const std::vector<float>& sigmas,
                        int start_merge_step,
                        ggml_tensor* c_id,
                        ggml_tensor* c_vec_id) {
        size_t steps = sigmas.size() - 1;
        // x_t = load_tensor_from_file(work_ctx, "./rand0.bin");
        // print_ggml_tensor(x_t);
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, x_t);
        copy_ggml_tensor(x, x_t);

        struct ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, x_t);

        bool has_unconditioned = cfg_scale != 1.0 && uc != NULL;

        if (noise == NULL) {
            // x = x * sigmas[0]
            ggml_tensor_scale(x, sigmas[0]);
        } else {
            // xi = x + noise * sigma_sched[0]
            ggml_tensor_scale(noise, sigmas[0]);
            ggml_tensor_add(x, noise);
        }

        // denoise wrapper
        struct ggml_tensor* out_cond   = ggml_dup_tensor(work_ctx, x);
        struct ggml_tensor* out_uncond = NULL;
        if (has_unconditioned) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }
        struct ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

        auto denoise = [&](ggml_tensor* input, float sigma, int step) {
            if (step == 1) {
                pretty_progress(0, (int)steps, 0);
            }
            int64_t t0 = ggml_time_us();

            float c_skip               = 1.0f;
            float c_out                = 1.0f;
            float c_in                 = 1.0f;
            std::vector<float> scaling = denoiser->get_scalings(sigma);

            if (scaling.size() == 3) {  // CompVisVDenoiser
                c_skip = scaling[0];
                c_out  = scaling[1];
                c_in   = scaling[2];
            } else {  // CompVisDenoiser
                c_out = scaling[0];
                c_in  = scaling[1];
            }

            float t = denoiser->schedule->sigma_to_t(sigma);
            std::vector<float> timesteps_vec(x->ne[3], t);  // [N, ]
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            ggml_tensor_scale(noised_input, c_in);

            std::vector<struct ggml_tensor*> controls;

            if (control_hint != NULL) {
                control_net->compute(n_threads, noised_input, control_hint, timesteps, c, c_vector);
                controls = control_net->controls;
                // print_ggml_tensor(controls[12]);
                // GGML_ASSERT(0);
            }

            if (start_merge_step == -1 || step <= start_merge_step) {
                // cond
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         c,
                                         c_concat,
                                         c_vector,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_cond);
            } else {
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         c_id,
                                         c_concat,
                                         c_vec_id,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_cond);
            }

            float* negative_data = NULL;
            if (has_unconditioned) {
                // uncond
                if (control_hint != NULL) {
                    control_net->compute(n_threads, noised_input, control_hint, timesteps, uc, uc_vector);
                    controls = control_net->controls;
                }
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         uc,
                                         uc_concat,
                                         uc_vector,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_uncond);
                negative_data = (float*)out_uncond->data;
            }
            float* vec_denoised  = (float*)denoised->data;
            float* vec_input     = (float*)input->data;
            float* positive_data = (float*)out_cond->data;
            int ne_elements      = (int)ggml_nelements(denoised);
            for (int i = 0; i < ne_elements; i++) {
                float latent_result = positive_data[i];
                if (has_unconditioned) {
                    // out_uncond + cfg_scale * (out_cond - out_uncond)
                    int64_t ne3 = out_cond->ne[3];
                    if (min_cfg != cfg_scale && ne3 != 1) {
                        int64_t i3  = i / out_cond->ne[0] * out_cond->ne[1] * out_cond->ne[2];
                        float scale = min_cfg + (cfg_scale - min_cfg) * (i3 * 1.0f / ne3);
                    } else {
                        latent_result = negative_data[i] + cfg_scale * (positive_data[i] - negative_data[i]);
                    }
                }
                // v = latent_result, eps = latent_result
                // denoised = (v * c_out + input * c_skip) or (input + eps * c_out)
                vec_denoised[i] = latent_result * c_out + vec_input[i] * c_skip;
            }
            int64_t t1 = ggml_time_us();
            if (step > 0) {
                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);
                // LOG_INFO("step %d sampling completed taking %.2fs", step, (t1 - t0) * 1.0f / 1000000);
            }
        };

        // sample_euler_ancestral
        switch (method) {
            case EULER_A: {
                struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
                struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

                for (int i = 0; i < steps; i++) {
                    float sigma = sigmas[i];

                    // denoise
                    denoise(x, sigma, i + 1);

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
                    denoise(x, sigma, i + 1);

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
                    denoise(x, sigmas[i], -(i + 1));

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

                        denoise(x2, sigmas[i + 1], i + 1);
                        float* vec_denoised = (float*)denoised->data;
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
                    denoise(x, sigmas[i], i + 1);

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

                        denoise(x2, sigma_mid, i + 1);
                        float* vec_denoised = (float*)denoised->data;
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
                    denoise(x, sigmas[i], i + 1);

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

                        denoise(x2, sigmas[i + 1], i + 1);

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
                    denoise(x, sigmas[i], i + 1);

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
                    denoise(x, sigmas[i], i + 1);

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
                    denoise(x, sigma, i + 1);

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
        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        diffusion_model->free_compute_buffer();
        return x;
    }

    // ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding
    ggml_tensor* get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* moments) {
        // ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
        ggml_tensor* latent       = ggml_new_tensor_4d(work_ctx, moments->type, moments->ne[0], moments->ne[1], moments->ne[2] / 2, moments->ne[3]);
        struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, latent);
        ggml_tensor_set_f32_randn(noise, rng);
        // noise = load_tensor_from_file(work_ctx, "noise.bin");
        {
            float mean   = 0;
            float logvar = 0;
            float value  = 0;
            float std_   = 0;
            for (int i = 0; i < latent->ne[3]; i++) {
                for (int j = 0; j < latent->ne[2]; j++) {
                    for (int k = 0; k < latent->ne[1]; k++) {
                        for (int l = 0; l < latent->ne[0]; l++) {
                            mean   = ggml_tensor_get_f32(moments, l, k, j, i);
                            logvar = ggml_tensor_get_f32(moments, l, k, j + (int)latent->ne[2], i);
                            logvar = std::max(-30.0f, std::min(logvar, 20.0f));
                            std_   = std::exp(0.5f * logvar);
                            value  = mean + std_ * ggml_tensor_get_f32(noise, l, k, j, i);
                            value  = value * scale_factor;
                            // printf("%d %d %d %d -> %f\n", i, j, k, l, value);
                            ggml_tensor_set_f32(latent, value, l, k, j, i);
                        }
                    }
                }
            }
        }
        return latent;
    }

    ggml_tensor* compute_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode) {
        int64_t W           = x->ne[0];
        int64_t H           = x->ne[1];
        ggml_tensor* result = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                 decode ? (W * 8) : (W / 8),  // width
                                                 decode ? (H * 8) : (H / 8),  // height
                                                 decode ? 3 : (use_tiny_autoencoder ? 4 : 8),
                                                 x->ne[3]);  // channels
        int64_t t0          = ggml_time_ms();
        if (!use_tiny_autoencoder) {
            if (decode) {
                ggml_tensor_scale(x, 1.0f / scale_factor);
            } else {
                ggml_tensor_scale_input(x);
            }
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 32x32 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    first_stage_model->compute(n_threads, in, decode, &out);
                };
                sd_tiling(x, result, 8, 32, 0.5f, on_tiling);
            } else {
                first_stage_model->compute(n_threads, x, decode, &result);
            }
            first_stage_model->free_compute_buffer();
            if (decode) {
                ggml_tensor_scale_output(result);
            }
        } else {
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 64x64 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    tae_first_stage->compute(n_threads, in, decode, &out);
                };
                sd_tiling(x, result, 8, 64, 0.5f, on_tiling);
            } else {
                tae_first_stage->compute(n_threads, x, decode, &result);
            }
            tae_first_stage->free_compute_buffer();
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae [mode: %s] graph completed, taking %.2fs", decode ? "DECODE" : "ENCODE", (t1 - t0) * 1.0f / 1000);
        if (decode) {
            ggml_tensor_clamp(result, 0.0f, 1.0f);
        }
        return result;
    }

    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, false);
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, true);
    }
};

/*================================================= SD API ==================================================*/

struct sd_ctx_t {
    StableDiffusionGGML* sd = NULL;
};

sd_ctx_t* new_sd_ctx(const char* model_path_c_str,
                     const char* vae_path_c_str,
                     const char* taesd_path_c_str,
                     const char* control_net_path_c_str,
                     const char* lora_model_dir_c_str,
                     const char* embed_dir_c_str,
                     const char* id_embed_dir_c_str,
                     bool vae_decode_only,
                     bool vae_tiling,
                     bool free_params_immediately,
                     int n_threads,
                     enum ggml_type wtype,
                     enum rng_type_t rng_type,
                     enum schedule_t s,
                     bool keep_clip_on_cpu,
                     bool keep_control_net_cpu,
                     bool keep_vae_on_cpu) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == NULL) {
        return NULL;
    }
    std::string model_path(model_path_c_str);
    std::string vae_path(vae_path_c_str);
    std::string taesd_path(taesd_path_c_str);
    std::string control_net_path(control_net_path_c_str);
    std::string embd_path(embed_dir_c_str);
    std::string id_embd_path(id_embed_dir_c_str);
    std::string lora_model_dir(lora_model_dir_c_str);

    sd_ctx->sd = new StableDiffusionGGML(n_threads,
                                         vae_decode_only,
                                         free_params_immediately,
                                         lora_model_dir,
                                         rng_type);
    if (sd_ctx->sd == NULL) {
        return NULL;
    }

    if (!sd_ctx->sd->load_from_file(model_path,
                                    vae_path,
                                    control_net_path,
                                    embd_path,
                                    id_embd_path,
                                    taesd_path,
                                    vae_tiling,
                                    (ggml_type)wtype,
                                    s,
                                    keep_clip_on_cpu,
                                    keep_control_net_cpu,
                                    keep_vae_on_cpu)) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
        free(sd_ctx);
        return NULL;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != NULL) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
    }
    free(sd_ctx);
}

sd_image_t* generate_image(sd_ctx_t* sd_ctx,
                           struct ggml_context* work_ctx,
                           ggml_tensor* init_latent,
                           std::string prompt,
                           std::string negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           const std::vector<float>& sigmas,
                           int64_t seed,
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_ratio,
                           bool normalize_input,
                           std::string input_id_images_path) {
    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(NULL));
        seed = rand();
    }

    int sample_steps = sigmas.size() - 1;

    // Apply lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier

    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }

    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    int64_t t0 = ggml_time_ms();
    sd_ctx->sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    // Photo Maker
    std::string prompt_text_only;
    ggml_tensor* init_img              = NULL;
    ggml_tensor* prompts_embeds        = NULL;
    ggml_tensor* pooled_prompts_embeds = NULL;
    std::vector<bool> class_tokens_mask;
    if (sd_ctx->sd->stacked_id) {
        if (!sd_ctx->sd->pmid_lora->applied) {
            t0 = ggml_time_ms();
            sd_ctx->sd->pmid_lora->apply(sd_ctx->sd->tensors, sd_ctx->sd->n_threads);
            t1                             = ggml_time_ms();
            sd_ctx->sd->pmid_lora->applied = true;
            LOG_INFO("pmid_lora apply completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_lora->free_params_buffer();
            }
        }
        // preprocess input id images
        std::vector<sd_image_t*> input_id_images;
        if (sd_ctx->sd->pmid_model && input_id_images_path.size() > 0) {
            std::vector<std::string> img_files = get_files_from_dir(input_id_images_path);
            for (std::string img_file : img_files) {
                int c = 0;
                int width, height;
                uint8_t* input_image_buffer = stbi_load(img_file.c_str(), &width, &height, &c, 3);
                if (input_image_buffer == NULL) {
                    LOG_ERROR("PhotoMaker load image from '%s' failed", img_file.c_str());
                    continue;
                } else {
                    LOG_INFO("PhotoMaker loaded image from '%s'", img_file.c_str());
                }
                sd_image_t* input_image = NULL;
                input_image             = new sd_image_t{(uint32_t)width,
                                             (uint32_t)height,
                                             3,
                                             input_image_buffer};
                input_image             = preprocess_id_image(input_image);
                if (input_image == NULL) {
                    LOG_ERROR("preprocess input id image from '%s' failed", img_file.c_str());
                    continue;
                }
                input_id_images.push_back(input_image);
            }
        }
        if (input_id_images.size() > 0) {
            sd_ctx->sd->pmid_model->style_strength = style_ratio;
            int32_t w                              = input_id_images[0]->width;
            int32_t h                              = input_id_images[0]->height;
            int32_t channels                       = input_id_images[0]->channel;
            int32_t num_input_images               = (int32_t)input_id_images.size();
            init_img                               = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, w, h, channels, num_input_images);
            // TODO: move these to somewhere else and be user settable
            float mean[] = {0.48145466f, 0.4578275f, 0.40821073f};
            float std[]  = {0.26862954f, 0.26130258f, 0.27577711f};
            for (int i = 0; i < num_input_images; i++) {
                sd_image_t* init_image = input_id_images[i];
                if (normalize_input)
                    sd_mul_images_to_tensor(init_image->data, init_img, i, mean, std);
                else
                    sd_mul_images_to_tensor(init_image->data, init_img, i, NULL, NULL);
            }
            t0                    = ggml_time_ms();
            auto cond_tup         = sd_ctx->sd->get_learned_condition_with_trigger(work_ctx, prompt,
                                                                                   clip_skip, width, height, num_input_images);
            prompts_embeds        = std::get<0>(cond_tup);
            pooled_prompts_embeds = std::get<1>(cond_tup);  // [adm_in_channels, ]
            class_tokens_mask     = std::get<2>(cond_tup);  //

            prompts_embeds = sd_ctx->sd->id_encoder(work_ctx, init_img, prompts_embeds, class_tokens_mask);
            t1             = ggml_time_ms();
            LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1 - t0);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_model->free_params_buffer();
            }
            // Encode input prompt without the trigger word for delayed conditioning
            prompt_text_only = sd_ctx->sd->remove_trigger_from_prompt(work_ctx, prompt);
            // printf("%s || %s \n", prompt.c_str(), prompt_text_only.c_str());
            prompt = prompt_text_only;  //
            // if (sample_steps < 50) {
            //     LOG_INFO("sampling steps increases from %d to 50 for PHOTOMAKER", sample_steps);
            //     sample_steps = 50;
            // }
        } else {
            LOG_WARN("Provided PhotoMaker model file, but NO input ID images");
            LOG_WARN("Turn off PhotoMaker");
            sd_ctx->sd->stacked_id = false;
        }
        for (sd_image_t* img : input_id_images) {
            free(img->data);
        }
        input_id_images.clear();
    }

    // Get learned condition
    t0                    = ggml_time_ms();
    auto cond_pair        = sd_ctx->sd->get_learned_condition(work_ctx, prompt, clip_skip, width, height);
    ggml_tensor* c        = cond_pair.first;
    ggml_tensor* c_vector = cond_pair.second;  // [adm_in_channels, ]

    struct ggml_tensor* uc        = NULL;
    struct ggml_tensor* uc_vector = NULL;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd_ctx->sd->version == VERSION_XL && negative_prompt.size() == 0) {
            force_zero_embeddings = true;
        }
        auto uncond_pair = sd_ctx->sd->get_learned_condition(work_ctx, negative_prompt, clip_skip, width, height, force_zero_embeddings);
        uc               = uncond_pair.first;
        uc_vector        = uncond_pair.second;  // [adm_in_channels, ]
    }
    t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    // Control net hint
    struct ggml_tensor* image_hint = NULL;
    if (control_cond != NULL) {
        image_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_tensor(control_cond->data, image_hint);
    }

    // Sample
    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = 4;
    int W = width / 8;
    int H = height / 8;
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %i", b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t   = NULL;
        struct ggml_tensor* noise = NULL;
        if (init_latent == NULL) {
            x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
            ggml_tensor_set_f32_randn(x_t, sd_ctx->sd->rng);
        } else {
            x_t   = init_latent;
            noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
            ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);
        }

        int start_merge_step = -1;
        if (sd_ctx->sd->stacked_id) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * sample_steps);
            // if (start_merge_step > 30)
            //     start_merge_step = 30;
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }

        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                     x_t,
                                                     noise,
                                                     c,
                                                     NULL,
                                                     c_vector,
                                                     uc,
                                                     NULL,
                                                     uc_vector,
                                                     image_hint,
                                                     control_strength,
                                                     cfg_scale,
                                                     cfg_scale,
                                                     sample_method,
                                                     sigmas,
                                                     start_merge_step,
                                                     prompts_embeds,
                                                     pooled_prompts_embeds);
        // struct ggml_tensor* x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
        // print_ggml_tensor(x_0);
        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        final_latents.push_back(x_0);
    }

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3 - t1) * 1.0f / 1000);

    // Decode to image
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        t1                      = ggml_time_ms();
        struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */);
        // print_ggml_tensor(img);
        if (img != NULL) {
            decoded_images.push_back(img);
        }
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->use_tiny_autoencoder) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    sd_image_t* result_images = (sd_image_t*)calloc(batch_count, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(decoded_images[i]);
    }
    ggml_free(work_ctx);

    return result_images;
}

sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    int width,
                    int height,
                    enum sample_method_t sample_method,
                    int sample_steps,
                    int64_t seed,
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength,
                    float style_ratio,
                    bool normalize_input,
                    const char* input_id_images_path_c_str) {
    LOG_DEBUG("txt2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    if (sd_ctx->sd->stacked_id) {
        params.mem_size += static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    }
    params.mem_size += width * height * 3 * sizeof(float);
    params.mem_size *= batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    size_t t0 = ggml_time_ms();

    std::vector<float> sigmas = sd_ctx->sd->denoiser->schedule->get_sigmas(sample_steps);

    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx,
                                               NULL,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               width,
                                               height,
                                               sample_method,
                                               sigmas,
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str);

    size_t t1 = ggml_time_ms();

    LOG_INFO("txt2img completed in %.2fs", (t1 - t0) * 1.0f / 1000);

    return result_images;
}

sd_image_t* img2img(sd_ctx_t* sd_ctx,
                    sd_image_t init_image,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    int width,
                    int height,
                    sample_method_t sample_method,
                    int sample_steps,
                    float strength,
                    int64_t seed,
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength,
                    float style_ratio,
                    bool normalize_input,
                    const char* input_id_images_path_c_str) {
    LOG_DEBUG("img2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    if (sd_ctx->sd->stacked_id) {
        params.mem_size += static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    }
    params.mem_size += width * height * 3 * sizeof(float) * 2;
    params.mem_size *= batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    size_t t0 = ggml_time_ms();

    if (seed < 0) {
        srand((int)time(NULL));
        seed = rand();
    }
    sd_ctx->sd->rng->manual_seed(seed);

    ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
    sd_image_to_tensor(init_image.data, init_img);
    ggml_tensor* init_latent = NULL;
    if (!sd_ctx->sd->use_tiny_autoencoder) {
        ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
        init_latent          = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
    }
    // print_ggml_tensor(init_latent);
    size_t t1 = ggml_time_ms();
    LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->schedule->get_sigmas(sample_steps);
    size_t t_enc              = static_cast<size_t>(sample_steps * strength);
    LOG_INFO("target t_enc is %zu steps", t_enc);
    std::vector<float> sigma_sched;
    sigma_sched.assign(sigmas.begin() + sample_steps - t_enc - 1, sigmas.end());

    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx,
                                               init_latent,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               width,
                                               height,
                                               sample_method,
                                               sigma_sched,
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str);

    size_t t2 = ggml_time_ms();

    LOG_INFO("img2img completed in %.2fs", (t1 - t0) * 1.0f / 1000);

    return result_images;
}

SD_API sd_image_t* img2vid(sd_ctx_t* sd_ctx,
                           sd_image_t init_image,
                           int width,
                           int height,
                           int video_frames,
                           int motion_bucket_id,
                           int fps,
                           float augmentation_level,
                           float min_cfg,
                           float cfg_scale,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           float strength,
                           int64_t seed) {
    if (sd_ctx == NULL) {
        return NULL;
    }

    LOG_INFO("img2vid %dx%d", width, height);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->schedule->get_sigmas(sample_steps);

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024) * 1024;  // 10 MB
    params.mem_size += width * height * 3 * sizeof(float) * video_frames;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    // draft context
    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    if (seed < 0) {
        seed = (int)time(NULL);
    }

    sd_ctx->sd->rng->manual_seed(seed);

    int64_t t0 = ggml_time_ms();

    ggml_tensor* c_crossattn = NULL;
    ggml_tensor* c_concat    = NULL;
    ggml_tensor* c_vector    = NULL;

    ggml_tensor* uc_crossattn = NULL;
    ggml_tensor* uc_concat    = NULL;
    ggml_tensor* uc_vector    = NULL;

    std::tie(c_crossattn, c_concat, c_vector) = sd_ctx->sd->get_svd_condition(work_ctx,
                                                                              init_image,
                                                                              width,
                                                                              height,
                                                                              fps,
                                                                              motion_bucket_id,
                                                                              augmentation_level);

    uc_crossattn = ggml_dup_tensor(work_ctx, c_crossattn);
    ggml_set_f32(uc_crossattn, 0.f);

    uc_concat = ggml_dup_tensor(work_ctx, c_concat);
    ggml_set_f32(uc_concat, 0.f);

    uc_vector = ggml_dup_tensor(work_ctx, c_vector);

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->clip_vision->free_params_buffer();
    }

    sd_ctx->sd->rng->manual_seed(seed);
    int C                   = 4;
    int W                   = width / 8;
    int H                   = height / 8;
    struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, video_frames);
    ggml_tensor_set_f32_randn(x_t, sd_ctx->sd->rng);

    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                 x_t,
                                                 NULL,
                                                 c_crossattn,
                                                 c_concat,
                                                 c_vector,
                                                 uc_crossattn,
                                                 uc_concat,
                                                 uc_vector,
                                                 {},
                                                 0.f,
                                                 min_cfg,
                                                 cfg_scale,
                                                 sample_method,
                                                 sigmas,
                                                 -1,
                                                 NULL,
                                                 NULL);

    int64_t t2 = ggml_time_ms();
    LOG_INFO("sampling completed, taking %.2fs", (t2 - t1) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }

    struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, x_0);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (img == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(video_frames, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < video_frames; i++) {
        auto img_i = ggml_view_3d(work_ctx, img, img->ne[0], img->ne[1], img->ne[2], img->nb[1], img->nb[2], img->nb[3] * i);

        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(img_i);
    }
    ggml_free(work_ctx);

    int64_t t3 = ggml_time_ms();

    LOG_INFO("img2vid completed in %.2fs", (t3 - t0) * 1.0f / 1000);

    return result_images;
}
