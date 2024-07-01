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

#include "json.h"
#include "log.h"
#include "utils.h"

extern llama_model* g_model;

void
normalize_embeddings(const float* inp, float* out, int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += inp[i] * inp[i];
    sum = sqrt(sum);
    const float norm = sum > 0 ? 1.f / sum : 0.f;
    for (int i = 0; i < n; i++)
        out[i] = inp[i] * norm;
}

void
add_token_to_batch(struct llama_batch& batch,
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

void
cleanup_llama_batch(void* arg)
{
    llama_batch* batch = (llama_batch*)arg;
    llama_batch_free(*batch);
    delete batch;
}

void
cleanup_token_vector(void* arg)
{
    delete (ctl::vector<llama_token>*)arg;
}

void
cleanup_llama_context(void* arg)
{
    llama_free((llama_context*)arg);
}

bool
Client::embedding()
{
    if (msg.method != kHttpGet && msg.method != kHttpPost)
        return send_error(405);

    if (!read_payload())
        return false;

    // get prompt
    //
    //   1. Allow GET "/tokenize?prompt=foo"
    //   2. Allow POST "prompt=foo" (application/x-www-form-urlencoded)
    //   3. Allow POST "foo" (text/plain)
    //
    ctl::string_view input;
    ctl::optional<ctl::string_view> prompt = param("prompt");
    if (prompt.has_value()) {
        input = prompt.value();
    } else if (HasHeader(kHttpContentType)) {
        if (IsMimeType(HeaderData(kHttpContentType),
                       HeaderLength(kHttpContentType),
                       "text/plain")) {
            input = payload;
        } else {
            return send_error(501, "Content Type Not Implemented");
        }
    } else {
        input = payload;
    }

    // get optional parameters
    bool add_special = atob(or_empty(param("add_special")), true);
    bool parse_special = atob(or_empty(param("parse_special")), false);

    // setup statistics
    rusage rustart = {};
    getrusage(RUSAGE_THREAD, &rustart);
    timespec started = timespec_real();

    // turn text into tokens
    auto toks = new ctl::vector<llama_token>(input.size() + 16);
    defer_cleanup(cleanup_token_vector, toks);
    int count = llama_tokenize(g_model,
                               input.data(),
                               input.size(),
                               &(*toks)[0],
                               toks->size(),
                               add_special,
                               parse_special);
    if (count < 0) {
        LOG("llama_tokenize failed");
        return send_error(405);
    }
    toks->resize(count);

    // truncate if exceeds model context size
    const int n_ctx_train = llama_n_ctx_train(g_model);
    if (count > n_ctx_train)
        count = n_ctx_train;

    // initialize context
    llama_context_params cparams = {};
    cparams.embeddings = true;
    cparams.embeddings_only = true;
    cparams.logits_all = true;
    cparams.seed = _rand64();
    cparams.n_ctx = count;
    cparams.n_batch = count;
    cparams.n_ubatch = count;
    cparams.n_seq_max = 1;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    cparams.flash_attn = FLAG_flash_attn;
    llama_context* ctx = llama_new_context_with_model(g_model, cparams);
    if (!ctx) {
        LOG("llama_new_context_with_model failed");
        return send_error(500);
    }
    defer_cleanup(cleanup_llama_context, ctx);

    // initialize batch
    const int n_embd = llama_n_embd(g_model);
    llama_batch* batch = new llama_batch;
    *batch = llama_batch_init(count, 0, 1);
    defer_cleanup(cleanup_llama_batch, batch);
    for (size_t i = 0; i < count; ++i)
        add_token_to_batch(*batch, (*toks)[i], i, { 0 }, i == count - 1);

    // inference time
    if (llama_decode(ctx, *batch) < 0) {
        LOG("llama_decode failed");
        return send_error(500);
    }
    ctl::vector<float> embeddings(n_embd, 0);
    for (int i = 0; i < batch->n_tokens; i++) {
        if (!batch->logits[i])
            continue;
        const float* embd = llama_get_embeddings_ith(ctx, i);
        if (!embd) {
            LOG("llama_get_embeddings_ith failed");
            return send_error(500);
        }
        normalize_embeddings(
          embd, &embeddings[0] + batch->seq_id[i][0] * n_embd, n_embd);
    }

    // serialize tokens to json
    char* p = obuf.p;
    p = stpcpy(p, "{\r\n");
    p = stpcpy(p, "  \"add_special\": ");
    p = encode_bool(p, add_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"parse_special\": ");
    p = encode_bool(p, parse_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"tokens_provided\": ");
    p = encode_json(p, toks->size());
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"tokens_used\": ");
    p = encode_json(p, count);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"embedding\": [");
    for (size_t i = 0; i < embeddings.size(); ++i) {
        if (i) {
            *p++ = ',';
            *p++ = ' ';
        }
        p = encode_json(p, embeddings[i]);
    }
    p = stpcpy(p, "]\r\n");
    p = stpcpy(p, "}\r\n");
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
