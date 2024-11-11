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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

namespace lf {

/**
 * Turns timestamp into string.
 */
std::string iso8601(struct timespec ts) {
    struct tm tm;
    if (!localtime_r(&ts.tv_sec, &tm))
        if (!gmtime_r(&ts.tv_sec, &tm))
            abort();
    char res[256];
    char *ptr = res;
    char *end = res + sizeof(res);
    ptr += strftime(ptr, end - ptr, "%Y-%m-%d %H:%M:%S", &tm);
    ptr += snprintf(ptr, end - ptr, ".%09ld", ts.tv_nsec);
    ptr += strftime(ptr, end - ptr, "%z %Z", &tm);
    npassert(ptr + 1 <= end);
    return res;
}

} // namespace lf
