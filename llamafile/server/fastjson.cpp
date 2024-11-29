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

#include "fastjson.h"
#include "third_party/double-conversion/double-to-string.h"
#include "third_party/double-conversion/utils.h"
#include <cosmo.h>
#include <net/http/escape.h>
#include <string>

namespace lf {
namespace server {

static const char kEscapeLiteral[128] = {
    9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 2, 9, 4, 3, 9, 9, // 0x00
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 0x10
    0, 0, 7, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 6, // 0x20
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, // 0x30
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x40
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, // 0x50
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x60
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, // 0x70
};

static const double_conversion::DoubleToStringConverter kDoubleToJson(
  double_conversion::DoubleToStringConverter::UNIQUE_ZERO |
    double_conversion::DoubleToStringConverter::EMIT_POSITIVE_EXPONENT_SIGN,
  "1e5000",
  "null",
  'e',
  -6,
  21,
  6,
  0);

char*
encode_bool(char* p, bool x) noexcept
{
    return stpcpy(p, x ? "true" : "false");
}

char*
encode_json(char* p, int x) noexcept
{
    return FormatInt32(p, x);
}

char*
encode_json(char* p, long x) noexcept
{
    return FormatInt64(p, x);
}

char*
encode_json(char* p, unsigned x) noexcept
{
    return FormatUint32(p, x);
}

char*
encode_json(char* p, unsigned long x) noexcept
{
    return FormatUint64(p, x);
}

char*
encode_json(char* p, float x) noexcept
{
    double_conversion::StringBuilder b(p, 256);
    kDoubleToJson.ToShortestSingle(x, &b);
    b.Finalize();
    return p + strlen(p);
}

char*
encode_json(char* p, double x) noexcept
{
    double_conversion::StringBuilder b(p, 256);
    kDoubleToJson.ToShortest(x, &b);
    b.Finalize();
    return p + strlen(p);
}

char*
encode_json(char* p, const std::string_view s) noexcept
{
    *p++ = '"';
    p = encode_js_string_literal(p, s);
    *p++ = '"';
    *p = 0;
    return p;
}

char*
encode_js_string_literal(char* p, const std::string_view s) noexcept
{
    uint64_t w;
    size_t i, j, m;
    wint_t x, a, b;
    for (size_t i = 0; i < s.size();) {
        x = s[i++] & 255;
        if (x >= 0300) {
            a = ThomPikeByte(x);
            m = ThomPikeLen(x) - 1;
            if (i + m <= s.size()) {
                for (j = 0;;) {
                    b = s[i + j] & 0xff;
                    if (!ThomPikeCont(b))
                        break;
                    a = ThomPikeMerge(a, b);
                    if (++j == m) {
                        x = a;
                        i += j;
                        break;
                    }
                }
            }
        }
        switch (0 <= x && x <= 127 ? kEscapeLiteral[x] : 9) {
            case 0:
                *p++ = x;
                break;
            case 1:
                *p++ = '\\';
                *p++ = 't';
                break;
            case 2:
                *p++ = '\\';
                *p++ = 'n';
                break;
            case 3:
                *p++ = '\\';
                *p++ = 'r';
                break;
            case 4:
                *p++ = '\\';
                *p++ = 'f';
                break;
            case 5:
                *p++ = '\\';
                *p++ = '\\';
                break;
            case 6:
                *p++ = '\\';
                *p++ = '/';
                break;
            case 7:
                *p++ = '\\';
                *p++ = '"';
                break;
            case 9:
                w = EncodeUtf16(x);
                do {
                    *p++ = '\\';
                    *p++ = 'u';
                    *p++ = "0123456789abcdef"[(w & 0xF000) >> 014];
                    *p++ = "0123456789abcdef"[(w & 0x0F00) >> 010];
                    *p++ = "0123456789abcdef"[(w & 0x00F0) >> 004];
                    *p++ = "0123456789abcdef"[(w & 0x000F) >> 000];
                } while ((w >>= 16));
                break;
            default:
                __builtin_unreachable();
        }
    }
    *p = 0;
    return p;
}

} // namespace server
} // namespace lf
