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

#include <cosmo.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#define MIN(X, Y) ((Y) > (X) ? (X) : (Y))

#define MAX_SIZE (50L * 1000 * 1000 * 1000)

static wontreturn void Die(const char *thing, const char *reason) {
    tinyprint(2, thing, ": ", reason, "\n", NULL);
    exit(1);
}

static wontreturn void DieSys(const char *thing) {
    perror(thing);
    exit(1);
}

void slicehf(const char *in_path) {
    int in_fd = open(in_path, O_RDONLY);
    if (in_fd == -1)
        DieSys(in_path);

    struct stat st;
    if (fstat(in_fd, &st) == -1)
        DieSys(in_path);

    if (st.st_size <= MAX_SIZE)
        return;

    int item = 0;
    long total = st.st_size;
    long n = (st.st_size + MAX_SIZE - 1) / MAX_SIZE;
    while (total) {
        long amount = MIN(total, MAX_SIZE);
        total -= amount;
        char out_path[PATH_MAX];
        snprintf(out_path, PATH_MAX, "%s.cat%d", in_path, item++);
        int out_fd = open(out_path, O_WRONLY | O_TRUNC | O_CREAT, 0644);
        if (out_fd == -1)
            DieSys(out_path);
        while (amount) {
            long chunk = MIN(amount, 65536);
            long copied = copy_file_range(in_fd, 0, out_fd, 0, chunk, 0);
            if (copied == -1)
                DieSys(out_path);
            if (!copied)
                Die(in_path, "unexpected eof");
            amount -= copied;
        }
        close(out_fd);
    }

    close(in_fd);
}

int main(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        slicehf(argv[i]);
    }
}
