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
#include "llama.cpp/llama.h"
#include "llamafile/json.h"
#include "llamafile/server/cleanup.h"
#include "llamafile/server/fastjson.h"
#include "llamafile/server/log.h"
#include "llamafile/server/utils.h"
#include <cmath>
#include <cstring>
#include <sys/resource.h>
#include <vector>

using jt::Json;

namespace lf {
namespace server {

struct EmbeddingParams
{
    bool add_special;
    bool parse_special;
    std::string_view prompt;
    std::string content;
    std::string model;
};

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

static void
add_token_to_batch(struct llama_batch& batch,
                   llama_token id,
                   llama_pos pos,
                   const std::vector<llama_seq_id>& seq_ids,
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
cleanup_embedding_params(void* arg)
{
    delete (EmbeddingParams*)arg;
}

bool
Client::get_embedding_params(EmbeddingParams* params)
{
    params->add_special = atob(or_empty(param("add_special")), true);
    params->parse_special = atob(or_empty(param("parse_special")), false);

    // try obtaining prompt (or its aliases) from request-uri
    std::optional<std::string_view> prompt = param("content");
    if (!prompt.has_value()) {
        std::optional<std::string_view> prompt2 = param("prompt");
        if (prompt2.has_value()) {
            prompt = std::move(prompt2);
        } else {
            std::optional<std::string_view> prompt3 = param("input");
            if (prompt3.has_value()) {
                prompt = std::move(prompt3);
            }
        }
    }

    if (prompt.has_value()) {
        // [simple mode] if the prompt was supplied in the request-uri
        //               then we don't bother looking for a json body.
        params->prompt = prompt.value();
    } else if (HasHeader(kHttpContentType)) {
        // [standard mode] if the prompt wasn't specified as a
        //                 request-uri parameter, then it must be in the
        //                 http message body.
        if (IsMimeType(HeaderData(kHttpContentType),
                       HeaderLength(kHttpContentType),
                       "text/plain")) {
            params->prompt = payload_;
        } else if (IsMimeType(HeaderData(kHttpContentType),
                              HeaderLength(kHttpContentType),
                              "application/json")) {
            std::pair<Json::Status, Json> json =
              Json::parse(std::string(payload_));
            if (json.first != Json::success)
                return send_error(400, Json::StatusToString(json.first));
            if (!json.second.isObject())
                return send_error(400, "JSON body must be an object");
            if (json.second["content"].isString())
                params->content = json.second["content"].getString();
            else if (json.second["prompt"].isString())
                params->content = json.second["prompt"].getString();
            else if (json.second["input"].isString())
                params->content = json.second["input"].getString();
            else
                return send_error(400, "JSON missing content/prompt/input key");
            params->prompt = params->content;
            if (json.second["add_special"].isBool())
                params->add_special = json.second["add_special"].getBool();
            if (json.second["parse_special"].isBool())
                params->parse_special = json.second["parse_special"].getBool();
            if (json.second["model"].isString())
                params->model = json.second["model"].getString();
        } else {
            return send_error(501, "Content Type Not Implemented");
        }
    } else {
        params->prompt = payload_;
    }
    return true;
}

bool
Client::embedding()
{
    if (msg_.method != kHttpGet && msg_.method != kHttpPost)
        return send_error(405);

    if (!read_payload())
        return false;

    // get parameters
    auto params = new EmbeddingParams;
    defer_cleanup(cleanup_embedding_params, params);
    if (!get_embedding_params(params))
        return false;

    // setup statistics
    rusage rustart = {};
    getrusage(RUSAGE_THREAD, &rustart);
    timespec started = timespec_real();

    // turn text into tokens
    auto toks = new std::vector<llama_token>(params->prompt.size() + 16);
    defer_cleanup(cleanup_token_vector, toks);
    int count = llama_tokenize(model_,
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

    // truncate if exceeds model context size
    const int n_ctx_train = llama_n_ctx_train(model_);
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
    cparams.attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
    cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    cparams.flash_attn = FLAG_flash_attn;
    llama_context* ctx = llama_new_context_with_model(model_, cparams);
    if (!ctx) {
        SLOG("llama_new_context_with_model failed");
        return send_error(500);
    }
    defer_cleanup(cleanup_llama_context, ctx);

    // initialize batch
    const int n_embd = llama_n_embd(model_);
    llama_batch* batch = new llama_batch;
    *batch = llama_batch_init(count, 0, 1);
    defer_cleanup(cleanup_llama_batch, batch);
    for (size_t i = 0; i < count; ++i)
        add_token_to_batch(*batch, (*toks)[i], i, { 0 }, i == count - 1);

    // inference time
    if (llama_decode(ctx, *batch) < 0) {
        SLOG("llama_decode failed");
        return send_error(500);
    }
    auto embeddings = new std::vector<float>(n_embd, 0);
    defer_cleanup(cleanup_float_vector, embeddings);
    for (int i = 0; i < batch->n_tokens; i++) {
        if (!batch->logits[i])
            continue;
        const float* embd = llama_get_embeddings_ith(ctx, i);
        if (!embd) {
            SLOG("llama_get_embeddings_ith failed");
            return send_error(500);
        }
        normalize_embeddings(
          embd, embeddings->data() + batch->seq_id[i][0] * n_embd, n_embd);
    }

    // determine how output json should look
    bool in_openai_mode = path() == "/v1/embeddings";

    // serialize tokens to json
    char* p = obuf_.p;
    p = stpcpy(p, "{\n");

    // Here's what an OpenAI /v1/embedding response looks like:
    //
    //     {
    //       "object": "list",
    //       "data": [
    //         {
    //           "object": "embedding",
    //           "index": 0,
    //           "embedding": [
    //             -0.006929283495992422,
    //             -0.005336422007530928,
    //             ... (omitted for spacing)
    //             -4.547132266452536e-05,
    //             -0.024047505110502243
    //           ],
    //         }
    //       ],
    //       "model": "text-embedding-3-small",
    //       "usage": {
    //         "prompt_tokens": 5,
    //         "total_tokens": 5
    //       }
    //     }
    //

    if (in_openai_mode) {
        p = stpcpy(p, "  \"object\": \"list\",\n");
        p = stpcpy(p, "  \"model\": ");
        p = encode_json(p, params->model);
        p = stpcpy(p, ",\n");
        p = stpcpy(p, "  \"usage\": {\n");
        p = stpcpy(p, "    \"prompt_tokens\": ");
        p = encode_json(p, count);
        p = stpcpy(p, ",\n");
        p = stpcpy(p, "    \"total_tokens\": ");
        p = encode_json(p, toks->size());
        p = stpcpy(p, "\n  },\n");
        p = stpcpy(p, "  \"data\": [{\n");
        p = stpcpy(p, "  \"object\": \"embedding\",\n");
        p = stpcpy(p, "  \"index\": 0,\n");
    } else {
        p = stpcpy(p, "  \"add_special\": ");
        p = encode_bool(p, params->add_special);
        p = stpcpy(p, ",\n");
        p = stpcpy(p, "  \"parse_special\": ");
        p = encode_bool(p, params->parse_special);
        p = stpcpy(p, ",\n");
        p = stpcpy(p, "  \"tokens_provided\": ");
        p = encode_json(p, toks->size());
        p = stpcpy(p, ",\n");
        p = stpcpy(p, "  \"tokens_used\": ");
        p = encode_json(p, count);
        p = stpcpy(p, ",\n");
    }

    p = stpcpy(p, "  \"embedding\": [");
    for (size_t i = 0; i < embeddings->size(); ++i) {
        if (i) {
            *p++ = ',';
            *p++ = ' ';
        }
        p = encode_json(p, (*embeddings)[i]);
    }
    p = stpcpy(p, "]\n");
    if (in_openai_mode)
        p = stpcpy(p, "  }]\n");
    p = stpcpy(p, "}\n");
    std::string_view content(obuf_.p, p - obuf_.p);

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
    p = append_http_response_message(p, 200);
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

} // namespace server
} // namespace lf
