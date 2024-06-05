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

#include "log.h"
#include "llamafile.h"

#include <pthread.h>
#include <string.h>
#include <unistd.h>

void(tinylog)(const char *s, ...) {
    size_t n;
    int c, cs;
    va_list va;
    char buf[512];
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    va_start(va, s);
    for (n = 0; s; s = va_arg(va, const char *)) {
        while ((c = *s++)) {
            buf[n++] = c;
            if (n == sizeof(buf)) {
                write(2, buf, n);
                n = 0;
            }
        }
    }
    va_end(va);
    write(2, buf, n);
    pthread_setcancelstate(cs, 0);
}

void llamafile_log_command(char *args[]) {
    tinylog(__func__, ": ", NULL);
    for (int i = 0; args[i]; ++i) {
        if (i)
            tinylog(" ", NULL);
        // this quoting should be close enough to correct to be
        // copy/pastable on both unix and windows command terms
        bool need_quotes = !!strchr(args[i], ' ');
        if (need_quotes)
            tinylog("\"", NULL);
        tinylog(args[i], NULL);
        if (need_quotes)
            tinylog("\"", NULL);
    }
    tinylog("\n", NULL);
}
