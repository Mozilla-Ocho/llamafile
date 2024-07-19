// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "client.h"

#include <ctl/vector.h>
#include <math.h>
#include <string.h>
#include <sys/resource.h>

#include "llama.cpp/llama.h"
#include "llama.cpp/sampling.h"

#include "cleanup.h"
#include "fastjson.h"
#include "json.h"
#include "log.h"
#include "utils.h"

struct CompletionParams
{
    bool add_special;
    bool parse_special;
    bool display_special;
    ctl::string_view prompt;
    ctl::string content;
};

struct CompletionResponse
{
    ctl::string prompt;
    ctl::string content;
    bool stop = false;
};

extern llama_model* g_model;

static void
add_token_to_batch(llama_batch& batch,
                   llama_token id,
                   llama_pos pos,
                   const ctl::vector<llama_seq_id>& seq_ids,
                   bool logits)
{
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i)
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

static ctl::string
token_to_piece(const struct llama_context* ctx, llama_token token, bool special)
{
    ctl::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(
      llama_get_model(ctx), token, result.data(), result.size(), special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(
          llama_get_model(ctx), token, result.data(), result.size(), special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return ctl::string(result.data(), result.size());
}

void
cleanup_completion_params(void* arg)
{
    delete (CompletionParams*)arg;
}

void
cleanup_completion_response(void* arg)
{
    delete (CompletionResponse*)arg;
}

void
cleanup_llama_sampling_context(void* arg)
{
    llama_sampling_free((llama_sampling_context*)arg);
}

bool
Client::get_completion_params(CompletionParams* params)
{
    params->add_special = atob(or_empty(param("add_special")), true);
    params->parse_special = atob(or_empty(param("parse_special")), false);
    params->display_special = atob(or_empty(param("display_special")), true);
    ctl::optional<ctl::string_view> prompt = param("prompt");
    if (prompt.has_value()) {
        params->prompt = prompt.value();
    } else if (HasHeader(kHttpContentType)) {
        if (IsMimeType(HeaderData(kHttpContentType),
                       HeaderLength(kHttpContentType),
                       "text/plain")) {
            params->prompt = payload;
        } else if (IsMimeType(HeaderData(kHttpContentType),
                              HeaderLength(kHttpContentType),
                              "application/json")) {
            ctl::pair<Json::Status, Json> json = Json::parse(payload);
            if (json.first != Json::success)
                return send_error(400, Json::StatusToString(json.first));
            if (!json.second["prompt"].isString())
                return send_error(400, "JSON missing \"prompt\" key");
            params->content = ctl::move(json.second["prompt"].getString());
            params->prompt = params->content;
        } else {
            return send_error(501, "Content Type Not Implemented");
        }
    } else {
        params->prompt = payload;
    }
    return true;
}

bool
Client::completion()
{
    if (msg.method != kHttpGet && msg.method != kHttpPost)
        return send_error(405);

    if (!read_payload())
        return false;

    // get parameters
    auto params = new CompletionParams;
    defer_cleanup(cleanup_completion_params, params);
    if (!get_completion_params(params))
        return false;

    // create response object
    auto response = new CompletionResponse;
    defer_cleanup(cleanup_completion_response, response);

    // setup statistics
    rusage rustart = {};
    getrusage(RUSAGE_THREAD, &rustart);
    timespec started = timespec_real();

    // turn text into tokens
    auto toks = new ctl::vector<llama_token>(params->prompt.size() + 16);
    defer_cleanup(cleanup_token_vector, toks);
    int count = llama_tokenize(g_model,
                               params->prompt.data(),
                               params->prompt.size(),
                               &(*toks)[0],
                               toks->size(),
                               params->add_special,
                               params->parse_special);
    if (count < 0) {
        SLOG("llama_tokenize failed");
        return send_error(405);
    }
    toks->resize(count);

    if (toks->empty())
        return send_error(400, "completely empty prompt disallowed");

    // fail if exceeds model context size
    const int n_ctx_train = llama_n_ctx_train(g_model);
    if (count + 1 > n_ctx_train)
        return send_error(400, "prompt too big for model context size");

    // initialize context
    llama_context_params cparams = {};
    cparams.embeddings = true;
    cparams.embeddings_only = false;
    cparams.logits_all = true;
    cparams.seed = _rand64();
    cparams.n_ctx = FLAG_ctx;
    if (cparams.n_ctx < count + 512)
        cparams.n_ctx = count + 512;
    if (cparams.n_ctx > n_ctx_train)
        cparams.n_ctx = n_ctx_train;
    cparams.n_batch = count;
    cparams.n_ubatch = count;
    cparams.n_seq_max = 1;
    cparams.n_threads = FLAG_threads;
    cparams.n_threads_batch = FLAG_threads;
    cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.rope_freq_base = 0;
    cparams.yarn_ext_factor = -1;
    cparams.yarn_attn_factor = 1;
    cparams.yarn_beta_fast = 32;
    cparams.yarn_beta_slow = 1;
    cparams.yarn_orig_ctx = 0;
    cparams.defrag_thold = -1;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    cparams.flash_attn = FLAG_flash_attn;

    llama_context* ctx = llama_new_context_with_model(g_model, cparams);
    if (!ctx) {
        SLOG("llama_new_context_with_model failed");
        return send_error(500);
    }
    defer_cleanup(cleanup_llama_context, ctx);

    // turn prompt back into string
    for (auto tok : *toks)
        response->prompt += token_to_piece(ctx, tok, params->display_special);

    // prefill time
    if (llama_decode(ctx, llama_batch_get_one(&(*toks)[0], count, 0, 0))) {
        SLOG("llama_decode prefill failed");
        return send_error(500, "llama_decode prefill failed");
    }

    // init sampling
    llama_sampling_params sparams;
    sparams.temp = FLAG_temp;
    sparams.seed = cparams.seed;
    llama_sampling_context* ctx_sampling = llama_sampling_init(sparams);
    if (!ctx_sampling) {
        SLOG("llama_sampling_init() failed");
        return send_error(500, "llama_sampling_init() failed");
    }
    defer_cleanup(cleanup_llama_sampling_context, ctx_sampling);

    // prediction time
    for (;;) {
        llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL);
        llama_sampling_accept(ctx_sampling, ctx, id, true);
        if (llama_token_is_eog(g_model, id)) {
            response->stop = true;
            break;
        }
        response->content += token_to_piece(ctx, id, params->display_special);
        if (llama_decode(ctx, llama_batch_get_one(&id, 1, count, 0)))
            break;
        ++count;
    }

    // serialize tokens to json
    char* p = obuf.p;
    p = stpcpy(p, "{\n");
    p = stpcpy(p, "  \"add_special\": ");
    p = encode_bool(p, params->add_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"parse_special\": ");
    p = encode_bool(p, params->parse_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"display_special\": ");
    p = encode_bool(p, params->display_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"tokens_provided\": ");
    p = encode_json(p, toks->size());
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"tokens_used\": ");
    p = encode_json(p, count);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"prompt\": ");
    p = encode_json(p, response->prompt);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"content\": ");
    p = encode_json(p, response->content);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"stop\": ");
    p = encode_bool(p, response->stop);
    p = stpcpy(p, "\n");
    p = stpcpy(p, "}\n");
    ctl::string_view content(obuf.p, p - obuf.p);

    // collect statistics
    rusage ruend = {};
    getrusage(RUSAGE_THREAD, &ruend);
    timeval user = timeval_sub(ruend.ru_utime, rustart.ru_utime);
    timeval system = timeval_sub(ruend.ru_stime, rustart.ru_stime);
    timespec ended = timespec_real();
    timespec wall = timespec_sub(ended, started);
    long wall_us = timespec_tomicros(wall);
    long user_us = timeval_tomicros(user);
    long system_us = timeval_tomicros(system);

    // send response
    char* headers = p;
    p = start_response(p, 200);
    p = stpcpy(p, "Content-Type: application/json\r\n");
    p = stpcpy(p, "X-Wall-Micros: ");
    p = FormatInt64(p, wall_us);
    p = stpcpy(p, "\r\nX-User-Micros: ");
    p = FormatInt64(p, user_us);
    p = stpcpy(p, "\r\nX-System-Micros: ");
    p = FormatInt64(p, system_us);
    p = stpcpy(p, "\r\n");
    return send_response(headers, p, content);
}
