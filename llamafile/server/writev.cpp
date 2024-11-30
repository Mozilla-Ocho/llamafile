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

#include "llamafile/server/log.h"
#include "utils.h"
#include <cerrno>
#include <string_view>

namespace lf {
namespace server {

ssize_t
safe_writev(int fd, const iovec* iov, int iovcnt)
{
    for (int i = 0; i < iovcnt; ++i) {
        bool has_binary = false;
        size_t n = iov[i].iov_len;
        unsigned char* p = (unsigned char*)iov[i].iov_base;
        for (size_t j = 0; j < n; ++j) {
            has_binary |= p[j] < 7;
        }
        if (has_binary) {
            SLOG("safe_writev() detected binary server is compromised");
            errno = EINVAL;
            return -1;
        }
    }
    return writev(fd, iov, iovcnt);
}

} // namespace server
} // namespace lf
