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
#include "llamafile/server/signals.h"
#include "llamafile/server/utils.h"
#include <cstring>
#include <sys/resource.h>
#include <utility>
#include <vector>

using jt::Json;

namespace lf {
namespace server {

struct TokenizeParams
{
    bool add_special;
    bool parse_special;
    std::string_view prompt;
    std::string content;
};

void
cleanup_tokenize_params(void* arg)
{
    delete (TokenizeParams*)arg;
}

bool
Client::get_tokenize_params(TokenizeParams* params)
{
    params->add_special = atob(or_empty(param("add_special")), true);
    params->parse_special = atob(or_empty(param("parse_special")), false);
    std::optional<std::string_view> prompt = param("prompt");
    if (prompt.has_value()) {
        params->prompt = prompt.value();
    } else if (HasHeader(kHttpContentType)) {
        if (IsMimeType(HeaderData(kHttpContentType),
                       HeaderLength(kHttpContentType),
                       "text/plain")) {
            params->prompt = payload_;
        } else if (IsMimeType(HeaderData(kHttpContentType),
                              HeaderLength(kHttpContentType),
                              "application/json")) {
            auto [status, json] = Json::parse(std::string(payload_));
            if (status != Json::success)
                return send_error(400, Json::StatusToString(status));
            if (!json.isObject())
                return send_error(400, "JSON body must be an object");
            if (!json["prompt"].isString())
                return send_error(400, "JSON missing \"prompt\" key");
            params->content = std::move(json["prompt"].getString());
            params->prompt = params->content;
            if (json["add_special"].isBool())
                params->add_special = json["add_special"].getBool();
            if (json["parse_special"].isBool())
                params->parse_special = json["parse_special"].getBool();
        } else {
            return send_error(501, "Content Type Not Implemented");
        }
    } else {
        params->prompt = payload_;
    }
    return true;
}

bool
Client::tokenize()
{
    if (msg_.method != kHttpGet && msg_.method != kHttpPost)
        return send_error(405);

    if (!read_payload())
        return false;

    // get parameters
    auto params = new TokenizeParams;
    defer_cleanup(cleanup_tokenize_params, params);
    if (!get_tokenize_params(params))
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

    // serialize tokens to json
    char* p = obuf_.p;
    p = stpcpy(p, "{\n");
    p = stpcpy(p, "  \"add_special\": ");
    p = encode_bool(p, params->add_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"parse_special\": ");
    p = encode_bool(p, params->parse_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"tokens\": [");
    for (int i = 0; i < count; ++i) {
        if (i)
            *p++ = ',';
        p = stpcpy(p, "\n    ");
        char s[32];
        int n =
          llama_token_to_piece(model_, (*toks)[i], s, sizeof(s), false, true);
        if (n < 0) {
            SLOG("failed to turn token into string");
            return send_error(405);
        }
        p = encode_json(p, std::string_view(s, n));
    }
    p = stpcpy(p, "\n  ]\n");
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
