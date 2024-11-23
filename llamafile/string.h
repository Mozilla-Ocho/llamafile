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
#include <__fwd/string.h>
#include <__fwd/string_view.h>
#include <__fwd/vector.h>
#include <ctime>

namespace lf {

bool startscasewith(const std::string_view &, const std::string_view &);
int strcasecmp(const std::string_view &, const std::string_view &);
ssize_t slurp(std::string *, const char *);
std::string basename(const std::string_view &);
std::string collapse(const std::string_view &);
std::string dirname(const std::string_view &);
std::string format(const char *, ...) __attribute__((format(printf, 1, 2)));
std::string iso8601(struct timespec);
std::string join(const std::vector<std::string> &, const std::string_view &);
std::string resolve(const std::string_view &, const std::string_view &);
std::string stripext(const std::string &);
std::string tolower(const std::string_view &);
std::string_view extname(const std::string_view &);
void append_wchar(std::string *, wchar_t);

} // namespace lf
