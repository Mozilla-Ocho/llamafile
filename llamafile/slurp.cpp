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

#include <fcntl.h>
#include <string>
#include <unistd.h>

namespace lf {

/**
 * Reads entire file into memory.
 */
ssize_t slurp(std::string *r, const char *path) {
    int fd;
    if ((fd = open(path, O_RDONLY)) == -1)
        return -1;
    size_t toto = 0;
    size_t orig = r->size();
    for (;;) {
        size_t want = 16384;
        size_t size = r->size();
        r->resize(size + want);
        ssize_t rc;
        if ((rc = read(fd, r->data() + size, want)) == -1) {
            r->resize(orig);
            close(fd);
            return -1;
        }
        size_t got = rc;
        r->resize(size + got);
        toto += got;
        if (!got)
            break;
    }
    if (close(fd)) {
        r->resize(orig);
        return -1;
    }
    return toto;
}

} // namespace lf
