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
#include <cctype>
#include <cosmo.h>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace lf {

std::string tolower(const std::string_view &s) {
    std::string b;
    for (char c : s)
        b += std::tolower(c);
    return b;
}

int strcasecmp(const std::string_view &a, const std::string_view &b) {
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        unsigned char al = std::tolower(a[i] & 255);
        unsigned char bl = std::tolower(b[i] & 255);
        if (al != bl)
            return (al > bl) - (al < bl);
    }
    return (a.size() > b.size()) - (a.size() < b.size());
}

bool startscasewith(const std::string_view &str, const std::string_view &prefix) {
    if (prefix.size() > str.size())
        return false;
    for (size_t i = 0; i < prefix.size(); ++i)
        if (std::tolower(str[i] & 255) != std::tolower(prefix[i] & 255))
            return false;
    return true;
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

std::string join(const std::vector<std::string> &vec, const std::string_view &delim) {
    std::string result;
    for (size_t i = 0; i < vec.size(); i++) {
        result += vec[i];
        if (i < vec.size() - 1)
            result += delim;
    }
    return result;
}

std::string basename(const std::string_view &path) {
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

std::string stripext(const std::string &path) {
    size_t i = path.size();
    while (i--)
        if (path[i] == '.')
            return path.substr(0, i);
    return path;
}

std::string_view extname(const std::string_view &path) {
    size_t i = path.size();
    while (i--)
        if (path[i] == '.' || path[i] == '/')
            return path.substr(i + 1);
    return path;
}

std::string dirname(const std::string_view &path) {
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

std::string resolve(const std::string_view &lhs, const std::string_view &rhs) {
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

// replaces multiple isspace() with one space and trims result
std::string collapse(const std::string_view &input) {
    size_t start = 0;
    while (start < input.length() && std::isspace(input[start]))
        ++start;
    if (start == input.length())
        return "";
    size_t end = input.length() - 1;
    while (end > start && std::isspace(input[end]))
        --end;
    std::string result;
    result.reserve(end - start + 1);
    bool lastWasSpace = false;
    for (size_t i = start; i <= end; ++i) {
        if (std::isspace(input[i])) {
            if (!lastWasSpace) {
                result += ' ';
                lastWasSpace = true;
            }
        } else {
            result += input[i];
            lastWasSpace = false;
        }
    }
    return result;
}

} // namespace lf
