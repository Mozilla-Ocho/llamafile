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
#include <string.h>
#include <sys/resource.h>

#include "llama.cpp/llama.h"

#include "cleanup.h"
#include "fastjson.h"
#include "log.h"
#include "utils.h"

bool
Client::tokenize()
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
    extern llama_model* g_model;
    int maxcount = input.size() + 16;
    auto toks = new ctl::vector<llama_token>(maxcount);
    defer_cleanup(cleanup_token_vector, toks);
    int count = llama_tokenize(g_model,
                               input.data(),
                               input.size(),
                               &(*toks)[0],
                               maxcount,
                               add_special,
                               parse_special);
    if (count < 0) {
        SLOG("failed to tokenize");
        return send_error(405);
    }
    toks->resize(count);

    // serialize tokens to json
    char* p = obuf.p;
    p = stpcpy(p, "{\n");
    p = stpcpy(p, "  \"add_special\": ");
    p = encode_bool(p, add_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"parse_special\": ");
    p = encode_bool(p, parse_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"tokens\": [");
    for (int i = 0; i < count; ++i) {
        if (i)
            *p++ = ',';
        p = stpcpy(p, "\n    ");
        char s[32];
        int n = llama_token_to_piece(g_model, (*toks)[i], s, sizeof(s), true);
        if (n < 0) {
            SLOG("failed to turn token into string");
            return send_error(405);
        }
        p = encode_json(p, ctl::string_view(s, n));
    }
    p = stpcpy(p, "\n  ]\n");
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
