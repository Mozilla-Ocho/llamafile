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

#include "llamafile.h"
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

/**
 * Displays man page.
 *
 * This function uses the system pagination program (less or more) when
 * it's available. That'll be the case for all OSes, including Windows.
 * When the standard output file descriptor isn't a teletypewriter, the
 * man page will be formatted as plain text utf8 without any paginator.
 *
 * @param path should be a .txt or .asc file, which is allowed to use
 *     ascii combining characters (for bold and underscore text)
 * @noreturn
 */
void llamafile_help(const char *path) {

    // slurp rendered `man llamafile` page
    int fd;
    fd = open(path, O_RDONLY);
    if (fd == -1) {
        perror(path);
        exit(1);
    }
    int size;
    size = lseek(fd, 0, SEEK_END);
    if (size == -1) {
        perror(path);
        exit(1);
    }
    char *text = calloc(1, size + 1);
    if (pread(fd, text, size, 0) != size) {
        perror(path);
        exit(1);
    }
    close(fd);

    // strip ascii combining sequences for bold and underscore text
    if (!isatty(1)) {
        int j = 0;
        for (int i = 0; i < size; ++i) {
            switch (text[i]) {
                case '\b':
                    --j;
                    break;
                default:
                    text[j++] = text[i];
                    break;
            }
        }
        text[j] = 0;
    }

    // diplay manual page in less or more if available
    __paginate(1, text);
    free(text);
    exit(0);
}
