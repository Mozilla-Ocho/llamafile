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

#include <climits>
#include <cmath>
#include <cstdio>
#include <string>

namespace lf {
namespace server {

std::string
encode_json(int x) noexcept
{
    char buf[12];
    return { buf, (size_t)(encode_json(buf, x) - buf) };
}

std::string
encode_json(long x) noexcept
{
    char buf[21];
    return { buf, (size_t)(encode_json(buf, x) - buf) };
}

std::string
encode_json(double x) noexcept
{
    char buf[128];
    return { buf, (size_t)(encode_json(buf, x) - buf) };
}

std::string
encode_json(unsigned x) noexcept
{
    char buf[128];
    return { buf, (size_t)(encode_json(buf, x) - buf) };
}

std::string
encode_json(unsigned long x) noexcept
{
    char buf[128];
    return { buf, (size_t)(encode_json(buf, x) - buf) };
}

std::string
encode_json(const std::string_view x) noexcept
{
    char buf[128];
    return { buf, (size_t)(encode_json(buf, x) - buf) };
}

std::string
encode_js_string_literal(const std::string_view x) noexcept
{
    char buf[256]; // this isn't secure (no guard page)
    return { buf, (size_t)(encode_json(buf, x) - buf) };
}

int
fastjson_test()
{
    if (encode_json(0) != "0")
        return 1;
    if (encode_json(INT_MAX) != "2147483647")
        return 2;
    if (encode_json(INT_MIN) != "-2147483648")
        return 3;
    if (encode_json(UINT_MAX) != "4294967295")
        return 4;
    if (encode_json(LONG_MAX) != "9223372036854775807")
        return 5;
    if (encode_json(LONG_MIN) != "-9223372036854775808")
        return 6;
    if (encode_json(ULONG_MAX) != "18446744073709551615")
        return 7;

    if (encode_json("") != "\"\"")
        return 8;
    if (encode_json(std::string_view("\0\1", 2)) != "\"\\u0000\\u0001\"")
        return 9;
    if (encode_json("\n\"\\\t") != "\"\\n\\\"\\\\\\t\"")
        return 10;
    if (encode_json("'") != "\"\\u0027\"")
        return 11;
    if (encode_json("µ") != "\"\\u00b5\"")
        return 12;
    if (encode_json("𐌰") != "\"\\ud800\\udf30\"")
        return 13;

    if (encode_json(3.) != "3")
        return 14;
    if (encode_json(3.14) != "3.14")
        return 15;
    if (encode_json(1e+100) != "1e+100")
        return 16;
    if (encode_json(1e-100) != "1e-100")
        return 17;
    if (encode_json(+INFINITY) != "1e5000")
        return 18;
    if (encode_json(-INFINITY) != "-1e5000")
        return 19;
    if (encode_json(+NAN) != "null")
        return 20;
    if (encode_json(-NAN) != "null")
        return 21;
    if (encode_json(1e-300) != "1e-300")
        return 21;

    return 0;
}

} // namespace server
} // namespace lf

int
main()
{
    return lf::server::fastjson_test();
}
