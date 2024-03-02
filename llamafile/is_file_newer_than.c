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

#include "llamafile.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int is_file_newer_than_time(const char *path, const char *other) {
    struct stat st1, st2;
    if (stat(path, &st1)) {
        // PATH should always exist when calling this function
        perror(path);
        return -1;
    }
    if (stat(other, &st2)) {
        if (errno == ENOENT) {
            // PATH should replace OTHER because OTHER doesn't exist yet
            return true;
        } else {
            // some other error happened, so we can't do anything
            perror(other);
            return -1;
        }
    }
    // PATH should replace OTHER if PATH was modified more recently
    return timespec_cmp(st1.st_mtim, st2.st_mtim) > 0;
}

static int is_file_newer_than_bytes(const char *path, const char *other) {
    int other_fd;
    if ((other_fd = open(other, O_RDONLY | O_CLOEXEC)) == -1) {
        if (errno == ENOENT) {
            return true;
        } else {
            perror(other);
            return -1;
        }
    }
    int path_fd;
    if ((path_fd = open(path, O_RDONLY | O_CLOEXEC)) == -1) {
        perror(path);
        close(other_fd);
        return -1;
    }
    int res;
    off_t i = 0;
    for (;;) {
        char path_buf[512];
        ssize_t path_rc = pread(path_fd, path_buf, sizeof(path_buf), i);
        if (path_rc == -1) {
            perror(path);
            res = -1;
            break;
        }
        char other_buf[512];
        ssize_t other_rc = pread(other_fd, other_buf, sizeof(other_buf), i);
        if (other_rc == -1) {
            perror(other);
            res = -1;
            break;
        }
        if (!path_rc || !other_rc) {
            if (!path_rc && !other_rc)
                res = false;
            else
                res = true;
            break;
        }
        size_t size = path_rc;
        if (other_rc < path_rc)
            size = other_rc;
        if (memcmp(path_buf, other_buf, size)) {
            res = true;
            break;
        }
        i += size;
    }
    if (close(path_fd)) {
        perror(path);
        res = -1;
    }
    if (close(other_fd)) {
        perror(other);
        res = -1;
    }
    return res;
}

/**
 * Returns true if `path` should replace `other`.
 *
 * The use case for this function is determining if a source code file
 * needs to be rebuilt. In that case, `path` can be thought of as the
 * source code file (which may reside under `/zip/...`, and `other`
 * would be the generated artifact that's dependent on `path`.
 */
int llamafile_is_file_newer_than(const char *path, const char *other) {
    if (startswith(path, "/zip/"))
        // to keep builds deterministic, embedded zip files always have
        // the same timestamp from back in 2022 when it was implemented
        return is_file_newer_than_bytes(path, other);
    else
        return is_file_newer_than_time(path, other);
}
