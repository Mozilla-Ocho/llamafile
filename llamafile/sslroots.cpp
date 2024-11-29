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

#include "net.h"
#include <cosmo.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>

__static_yoink("ssl_root_support");

#define SSL_ROOT_DIR "/zip/third_party/mbedtls/sslroot"

namespace lf {
namespace {

struct {
    unsigned once;
    mbedtls_x509_crt chain;
} g_ssl_roots;

void sslroots_free(void) {
    mbedtls_x509_crt_free(&g_ssl_roots.chain);
}

void sslroots_init(void) {
    DIR *dir;
    if (!(dir = opendir(SSL_ROOT_DIR))) {
        perror(SSL_ROOT_DIR);
        return;
    }
    struct dirent *ent;
    while ((ent = readdir(dir))) {
        if (ent->d_type != DT_REG && //
            ent->d_type != DT_UNKNOWN) {
            continue;
        }
        char path[PATH_MAX];
        strlcpy(path, SSL_ROOT_DIR "/", sizeof(path));
        strlcat(path, ent->d_name, sizeof(path));
        uint8_t *data;
        int fd = open(path, O_RDONLY); // punt error to lseek
        size_t size = lseek(fd, 0, SEEK_END); // punt error to calloc
        if ((data = (uint8_t *)calloc(1, size + 1)) && pread(fd, data, size, 0) == size) {
            if (mbedtls_x509_crt_parse(&g_ssl_roots.chain, data, size + 1)) {
                tinyprint(2, path, ": error loading ssl root\n", NULL);
            }
        } else {
            perror(path);
        }
        free(data);
        close(fd);
    }
    closedir(dir);
    atexit(sslroots_free);
}

} // namespace

/**
 * Returns singleton of SSL roots stored in /zip/usr/share/ssl/root/...
 */
mbedtls_x509_crt *sslroots(void) {
    cosmo_once(&g_ssl_roots.once, sslroots_init);
    return &g_ssl_roots.chain;
}

} // namespace lf
