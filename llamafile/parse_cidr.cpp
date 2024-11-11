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

#include <cstring>
#include <ctype.h>
#include <string>

bool parse_cidr(const std::string_view &str, cidr *out_cidr) noexcept {
    long ip;
    int bits;
    size_t i, n;
    const char *s, *p;
    s = str.data();
    n = str.size();
    if ((p = (const char *)memchr(s, '/', n))) {
        if ((ip = parse_ip(std::string_view(s, (i = p - s)))) == -1)
            return false;
        bits = 0;
        for (++i; i < n; ++i) {
            if (!isdigit(s[i]))
                return false;
            bits *= 10;
            bits += s[i] - '0';
            if (bits > 32)
                return false;
        }
        if (bits <= 0)
            return false;
    } else {
        if ((ip = parse_ip(str)) == -1)
            return false;
        bits = 32;
    }
    if (out_cidr) {
        out_cidr->ip = ip;
        out_cidr->bits = bits;
    }
    return true;
}
