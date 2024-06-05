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
#include <ctl/string_view.h>
#include <net/http/http.h>
#include <net/http/url.h>
#include <sys/resource.h>
#include <time.h>

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
    Buffer ibuf;
    Buffer obuf;

    Client();

    void run();
    int close();
    void clear();
    bool transport() __wur;
    bool synchronize() __wur;
    bool read_payload() __wur;
    bool read_request() __wur;
    bool read_content() __wur;
    bool send_continue() __wur;
    bool send(const ctl::string_view) __wur;
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
};
