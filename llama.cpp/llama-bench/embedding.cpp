// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#include "llama.cpp/common.h"
#include "llama.cpp/llama.h"
#include "llamafile/llamafile.h"

#include <ctime>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// [jart] dont do the multiline demo unless it's requested
static std::vector<std::string> split_lines(const std::string & s, bool multiline) {
    std::vector<std::string> lines;
    if (multiline) {
        std::string line;
        std::stringstream ss(s);
        while (std::getline(ss, line)) {
            lines.push_back(line);
        }
    } else {
        lines.push_back(s);
    }
    return lines;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, int seq_id) {
    for (size_t i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, { seq_id }, i == tokens.size() - 1);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd) {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    LOG("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        exit(1);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == NULL) {
                LOG("%s: failed to get embeddings for token %d\n", __func__, i);
                continue;
            }
        }

        float * out = output + batch.seq_id[i][0] * n_embd;
        llama_embd_normalize(embd, out, n_embd);
    }
}

static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

int embedding_cli(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

#ifndef LOG_DISABLE_LOGS
    log_set_target(stderr);
    log_dump_cmdline(argc, argv);
    llama_log_set(llama_log_callback_logTee, nullptr);
#endif // LOG_DISABLE_LOGS

    params.embedding = true;
    // For non-causal models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;

    print_build_info();

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = _rand64();
    }

    LOG("%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG("\n");
        LOG("%s\n", get_system_info(params).c_str());
    }

    // split the prompt into lines
    std::vector<std::string> prompts = split_lines(params.prompt, params.multiline_input);

    // max batch size
    const uint64_t n_batch = params.n_batch;
    GGML_ASSERT(params.n_batch >= params.n_ctx);

    // tokenize the prompts and trim
    std::vector<std::vector<int32_t>> inputs;
    for (const auto & prompt : prompts) {
        auto inp = ::llama_tokenize(ctx, prompt, true, false);
        if (inp.size() > n_batch) {
            fprintf(stderr, "%s: error: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        inputs.push_back(inp);
    }

    // add SEP if not present
    for (auto & inp : inputs) {
        if (inp.empty() || inp.back() != llama_token_sep(model)) {
            inp.push_back(llama_token_sep(model));
        }
    }

    // tokenization stats
    if (params.verbose_prompt) {
        for (int i = 0; i < (int) inputs.size(); i++) {
            LOG("%s: prompt %d: '%s'\n", __func__, i, prompts[i].c_str());
            LOG("%s: number of tokens in prompt = %zu\n", __func__, inputs[i].size());
            for (int j = 0; j < (int) inputs[i].size(); j++) {
                LOG("%6d -> '%s'\n", inputs[i][j], llama_token_to_piece(ctx, inputs[i][j]).c_str());
            }
            LOG("\n\n");
        }
    }

    // initialize batch
    const int n_prompts = prompts.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // allocate output
    const int n_embd = llama_n_embd(model);
    std::vector<float> embeddings(n_prompts * n_embd, 0);
    float * emb = embeddings.data();

    // break into batches
    int p = 0; // number of prompts processed already
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        auto & inp = inputs[k];

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float * out = emb + p * n_embd;
            batch_decode(ctx, batch, out, s, n_embd);
            llama_batch_clear(batch);
            p += s;
            s = 0;
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float * out = emb + p * n_embd;
    batch_decode(ctx, batch, out, s, n_embd);

    LOG("\n");

    // [jart] don't truncate data unless we're in interactive mode
    // [jart] print very carefully so this tool can be shell scriptable
    // print the first part of the embeddings or for a single prompt, the full embedding
    bool demo_mode = n_prompts > 1 && params.interactive;
    for (int j = 0; j < n_prompts; j++) {
        LOG("embedding %d: ", j);
        int display_count = n_embd;
        if (demo_mode) {
            display_count = std::min(16, display_count);
        }
        for (int i = 0; i < (n_prompts > 1 ? std::min(16, n_embd) : n_embd); i++) {
            if (i) {
                fprintf(stdout, " ");
            }
            if (demo_mode) {
                fprintf(stdout, "%9.6f", emb[j * n_embd + i]);
            } else {
                fprintf(stdout, "%g", emb[j * n_embd + i]);
            }
        }
        fprintf(stdout, "\n");
        fflush(stdout);
    }

    // [jart] print very carefully so this tool can be shell scriptable
    // print cosine similarity matrix
    if (n_prompts > 1) {
        LOG("\n");
        LOG("cosine similarity matrix:\n\n");
        for (int i = 0; i < n_prompts; i++) {
            for (int j = 0; j < n_prompts; j++) {
                float sim = llama_embd_similarity_cos(emb + i * n_embd, emb + j * n_embd, n_embd);
                LOG("%6.2f ", sim);
            }
            LOG("\n");
        }
    }

    // clean up
    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
