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

#include "string.h"

#include <cosmo.h>

wchar_t read_wchar(const std::string_view &s, size_t *i) {
    wint_t c = s[(*i)++] & 255;
    if (c >= 0300) {
        wint_t a = ThomPikeByte(c);
        size_t m = ThomPikeLen(c) - 1;
        if (*i + m <= s.size()) {
            for (int j = 0;;) {
                wint_t b = s[*i + j] & 255;
                if (!ThomPikeCont(b))
                    break;
                a = ThomPikeMerge(a, b);
                if (++j == m) {
                    c = a;
                    *i += j;
                    break;
                }
            }
        }
    }
    return c;
}

void append_wchar(std::string *r, wchar_t c) {
    if (isascii(c)) {
        *r += c;
    } else {
        char s[8];
        uint64_t w = tpenc(c);
        WRITE64LE(s, w);
        *r += s;
    }
}
