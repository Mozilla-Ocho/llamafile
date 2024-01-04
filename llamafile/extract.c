// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2023 Mozilla Foundation
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
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "llamafile.h"
#include "llamafile/log.h"

/**
 * Returns true if `zip` was successfully copied to `to`.
 *
 * Copying happens atomically. The `zip` argument is a file system path,
 * which may reside under `/zip/...` to relocate a compressed executable
 * asset to the local filesystem.
 */
bool llamafile_extract(const char *zip, const char *to) {
    int fdin, fdout;
    char stage[PATH_MAX];
    tinylog("extracting ", zip, " to ", to, "\n", NULL);
    strlcpy(stage, to, sizeof(stage));
    if (strlcat(stage, ".XXXXXX", sizeof(stage)) >= sizeof(stage)) {
        errno = ENAMETOOLONG;
        perror(to);
        return false;
    }
    if ((fdout = mkostemp(stage, O_CLOEXEC)) == -1) {
        perror(stage);
        return false;
    }
    if ((fdin = open(zip, O_RDONLY | O_CLOEXEC)) == -1) {
        perror(zip);
        close(fdout);
        unlink(stage);
        return false;
    }
    if (copyfd(fdin, fdout, -1) == -1) {
        perror(zip);
        close(fdin);
        close(fdout);
        unlink(stage);
        return false;
    }
    if (close(fdout)) {
        perror(to);
        close(fdin);
        unlink(stage);
        return false;
    }
    if (close(fdin)) {
        perror(zip);
        unlink(stage);
        return false;
    }
    if (rename(stage, to)) {
        perror(to);
        unlink(stage);
        return false;
    }
    return true;
}
