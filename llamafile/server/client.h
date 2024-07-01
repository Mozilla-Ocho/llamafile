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

#pragma once

#include "buffer.h"

#include <ctl/optional.h>
#include <ctl/string.h>
#include <libc/fmt/itoa.h>
#include <libc/str/slice.h>
#include <net/http/http.h>
#include <net/http/url.h>
#include <sys/resource.h>
#include <time.h>

#define HasHeader(H) (!!msg.headers[H].a)
#define HeaderData(H) (ibuf.p + msg.headers[H].a)
#define HeaderLength(H) (msg.headers[H].b - msg.headers[H].a)
#define HeaderEqual(H, S) \
    SlicesEqual(S, strlen(S), HeaderData(H), HeaderLength(H))
#define HeaderEqualCase(H, S) \
    SlicesEqualCase(S, strlen(S), HeaderData(H), HeaderLength(H))

struct Cleanup
{
    Cleanup* next;
    void (*func)(void*);
    void* arg;
};

struct Client
{
    int fd = -1;
    bool close_connection = false;
    size_t unread = 0;
    timespec message_started;
    HttpMessage msg;
    Url url = {};
    char* url_memory = nullptr;
    char* params_memory = nullptr;
    ctl::string_view payload;
    Cleanup* cleanups;
    Buffer ibuf;
    Buffer obuf;

    Client();

    void run();
    int close();
    void clear();
    void cleanup();
    bool transport() __wur;
    bool synchronize() __wur;
    bool read_payload() __wur;
    bool read_request() __wur;
    bool read_content() __wur;
    bool send_continue() __wur;
    bool send(const ctl::string_view) __wur;
    void defer_cleanup(void (*)(void*), void*);
    char* start_response(char*, int, const char* = nullptr);
    bool send_error(int, const char* = nullptr) __wur;
    bool send_response(char*, char*, const ctl::string_view) __wur;
    bool send2(const ctl::string_view, const ctl::string_view) __wur;
    char* append_header(const ctl::string_view, const ctl::string_view);
    bool has_at_most_this_element(int, const ctl::string_view);

    ctl::string_view path();
    ctl::optional<ctl::string_view> param(ctl::string_view);

    bool dispatch() __wur;
    bool tokenize() __wur;
    bool embedding() __wur;
};
