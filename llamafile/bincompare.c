// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#include "llama.cpp/ggml-impl.h"
#include <cosmo.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

// tool for checking if the binary payload of a llamafile matches a
// given executable. for example:
//
//     make -j32 o//llamafile/bincompare
//     o//llamafile/bincompare \
//         /mnt/videos/llamafile-0.7.3/bin/llamafile-0.7.3 \
//         phi-2.Q5_K_M.llamafile
//
// if the first MIN(size1, size2) bytes aren't identical, an error
// message is printed, and your process exit code will be nonzero.

wontreturn void Die(const char *thing, const char *reason) {
    tinyprint(2, thing, ": fatal error: ", reason, "\n", NULL);
    exit(1);
}

wontreturn void DieSys(const char *thing) {
    tinyprint(2, thing, ": fatal error: ", strerror(errno), "\n", NULL);
    exit(1);
}

int main(int argc, char *argv[]) {

    if (argc != 3)
        Die(argv[0], "missing operand");

    const char *path1 = argv[1];
    int fd1 = open(path1, O_RDONLY);
    if (fd1 == -1)
        DieSys(path1);
    ssize_t size1 = lseek(fd1, 0, SEEK_END);
    if (size1 == -1)
        DieSys(path1);

    const char *path2 = argv[2];
    int fd2 = open(path2, O_RDONLY);
    if (fd2 == -1)
        DieSys(path2);
    ssize_t size2 = lseek(fd2, 0, SEEK_END);
    if (size2 == -1)
        DieSys(path2);

    size_t chunk;
    size_t size = MIN(size1, size2);
    for (ssize_t i = 0; i < size; i += chunk) {
        size_t want = MIN(512, size - i);

        char buf1[512];
        ssize_t got1 = pread(fd1, buf1, want, i);
        if (got1 == -1)
            DieSys(path1);

        char buf2[512];
        ssize_t got2 = pread(fd2, buf2, want, i);
        if (got2 == -1)
            DieSys(path2);

        if (!(chunk = MIN(got1, got2)))
            break;

        for (size_t j = 0; j < chunk; ++j) {
            if (buf1[j] != buf2[j])
                Die(path2, "files differed!");
        }
    }
}
