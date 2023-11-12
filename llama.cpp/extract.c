// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set net ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
#define _COSMO_SOURCE
#include <cosmo.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include "ggml.h"

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
