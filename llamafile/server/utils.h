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
#include <ctl/optional.h>
#include <ctl/string.h>
#include <ctl/string_view.h>

extern const signed char kHexToInt[256];

bool
atob(ctl::string_view, bool);

static inline ctl::string_view
or_empty(ctl::optional<ctl::string_view> x)
{
    if (x.has_value())
        return x.value();
    return {};
}

static inline ctl::string normalize_url_prefix(ctl::string url_prefix) {
    // Rule 1: Replace multiple slashes with single slash
    while (url_prefix.find("//") != ctl::string::npos) {
        url_prefix.replace(url_prefix.find("//"), 2, "/");
    }

    // Rule 2: Remove trailing slash
    if (!url_prefix.empty() && url_prefix.back() == '/') {
        url_prefix.pop_back();
    }

    // Rule 3: Convert single slash to empty string
    if (url_prefix == "/") {
        url_prefix.clear();
    }
    return url_prefix;
}