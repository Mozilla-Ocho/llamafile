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
#include <ctype.h>

namespace lf {

std::string tolower(const std::string_view &s) {
    std::string b;
    for (char c : s)
        b += std::tolower(c);
    return b;
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

std::string format(const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = 512;
    std::string res(size, '\0');
    int need = vsnprintf(res.data(), size, fmt, ap);
    res.resize(need + 1, '\0');
    if (need + 1 > size)
        vsnprintf(res.data(), need + 1, fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return res;
}

std::string basename(const std::string_view path) {
    size_t i, e;
    if ((e = path.size())) {
        while (e > 1 && path[e - 1] == '/')
            --e;
        i = e - 1;
        while (i && path[i - 1] != '/')
            --i;
        return std::string(path.substr(i, e - i));
    } else {
        return ".";
    }
}

std::string extname(const std::string_view path) {
    std::string name = basename(path);
    size_t dot_pos = name.find_last_of('.');
    if (dot_pos == std::string::npos || dot_pos == name.length() - 1)
        return name;
    return std::string(name.substr(dot_pos + 1));
}

std::string dirname(const std::string_view path) {
    size_t e = path.size();
    if (e--) {
        for (; path[e] == '/'; e--)
            if (!e)
                return "/";
        for (; path[e] != '/'; e--)
            if (!e)
                return ".";
        for (; path[e] == '/'; e--)
            if (!e)
                return "/";
        return std::string(path.substr(0, e + 1));
    }
    return ".";
}

std::string resolve(const std::string_view lhs, const std::string_view rhs) {
    if (lhs.empty())
        return std::string(rhs);
    if (!rhs.empty() && rhs[0] == '/')
        return std::string(rhs);
    if (!lhs.empty() && lhs[lhs.size() - 1] == '/') {
        std::string res;
        res += lhs;
        res += rhs;
        return res;
    }
    std::string res;
    res += lhs;
    res += '/';
    res += rhs;
    return res;
}

} // namespace lf
