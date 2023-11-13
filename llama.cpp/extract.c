// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set net ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
#define _COSMO_SOURCE
#include <cosmo.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "ggml.h"

static const char *GetTmpDir(void) {
    const char *tmpdir;
    if (!(tmpdir = getenv("TMPDIR")) || !*tmpdir) {
        if (!(tmpdir = getenv("HOME")) || !*tmpdir) {
            tmpdir = ".";
        }
    }
    return tmpdir;
}

/**
 * Returns path of directory for app-specific files.
 */
void ggml_get_app_dir(char *path, size_t size) {
    strlcpy(path, GetTmpDir(), size);
    strlcat(path, "/.llamafile/", size);
}

/**
 * Returns true if `path` should replace `other`.
 *
 * The use case for this function is determining if a source code file
 * needs to be rebuilt. In that case, `path` can be thought of as the
 * source code file (which may reside under `/zip/...`, and `other`
 * would be the generated artifact that's dependent on `path`.
 */
bool ggml_is_newer_than(const char *path, const char *other) {
    struct stat st1, st2;
    if (stat(path, &st1) == -1) {
        // PATH should always exist when calling this function
        perror(path);
        return false;
    }
    if (stat(other, &st2) == -1) {
        if (errno == ENOENT) {
            // PATH should replace OTHER because OTHER doesn't exist yet
            return true;
        } else {
            // some other error happened, so we can't do anything
            perror(path);
            return false;
        }
    }
    // PATH should replace OTHER if PATH was modified more recently
    return timespec_cmp(st1.st_mtim, st2.st_mtim) > 0;
}

/**
 * Returns true if `zip` was successfully copied to `to`.
 *
 * Copying happens atomically. The `zip` argument is a file system path,
 * which may reside under `/zip/...` to relocate a compressed executable
 * asset to the local filesystem.
 */
bool ggml_extract(const char *zip, const char *to) {
    int fdin, fdout;
    char stage[PATH_MAX];
    tinyprint(2, "extracting ", zip, " to ", to, "\n", NULL);
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
