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

#include "trust.h"

#include <cctype>
#include <cstring>
#include <stdckdint.h>
#include <string>

long parse_ip(const std::string_view &str) noexcept {
    int c, j;
    size_t i, n;
    unsigned b, x;
    const char *s;
    bool dotted = false;
    s = str.data();
    n = str.size();

    // trim leading spaces
    while (n && isspace(*s)) {
        ++s;
        --n;
    }

    // trim trailing spaces
    while (n && isspace(s[n - 1]))
        --n;

    // forbid empty string
    if (!n)
        return -1;

    // parse ipv4 address
    for (b = x = j = i = 0; i < n; ++i) {
        c = s[i] & 255;
        if (isdigit(c)) {
            if (ckd_mul(&b, b, 10) || //
                ckd_add(&b, b, c - '0') || //
                (b > 255 && dotted)) {
                return -1;
            }
        } else if (c == '.') {
            if (b > 255)
                return -1;
            dotted = true;
            x <<= 8;
            x |= b;
            b = 0;
            ++j;
        } else {
            return -1;
        }
    }
    x <<= 8;
    x |= b;
    return x;
}
