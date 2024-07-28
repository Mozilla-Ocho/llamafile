// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
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

#include "llama.cpp/ggml.h"
#include "llamafile/log.h"
#include <cosmo.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>

#define FPS 24
#define THREADS 4

struct PageFaulter {
    long left;
    long right;
    long pagesz;
    pthread_t th;
    const char *data;
    atomic_long *faults;
};

static char Peek(volatile const char *ptr) {
    return *ptr;
}

char (*pPeek)(volatile const char *) = Peek;

static void *PageFaulter(void *arg) {
    struct PageFaulter *pf = arg;
    for (long i = pf->left; i < pf->right; i += pf->pagesz) {
        pPeek(pf->data + i);
        atomic_fetch_add_explicit(pf->faults, 1, memory_order_release);
    }
    return 0;
}

static void FormatPercent(char sbuf[static 8], double x) {
    char *p = sbuf;
    int n = x * 100000.5;
    if (n >= 100000)
        *p++ = '0' + n / 100000 % 10;
    if (n >= 10000)
        *p++ = '0' + n / 10000 % 10;
    *p++ = '0' + n / 1000 % 10;
    *p++ = '.';
    *p++ = '0' + n / 100 % 10;
    *p++ = '0' + n / 10 % 10;
    *p++ = '0' + n % 10;
    *p = 0;
}

/**
 * Loads memory off disk while reporting progress.
 */
void llamafile_schlep(const void *data, size_t size) {

    // avoid warmup
    if (!FLAG_warmup)
        return;

    // don't bother if logging is disabled
    if (FLAG_log_disable)
        return;

    // don't bother if memory is small
    if (size < 128 * 1024 * 1024)
        return;

    // don't bother if stderr isn't a terminal
    if (!isatty(2))
        return;

    // launch threads
    errno_t err;
    atomic_long faults = 0;
    long pagesz = getpagesize();
    long stride = size / THREADS;
    long pages = (stride + pagesz - 1) / pagesz * THREADS;
    struct PageFaulter pf[THREADS];
    for (int i = 0; i < THREADS; ++i) {
        pf[i].data = data;
        pf[i].pagesz = pagesz;
        pf[i].faults = &faults;
        pf[i].left = stride * i;
        pf[i].right = stride * (i + 1);
        err = pthread_create(&pf[i].th, 0, PageFaulter, pf + i);
        if (err) {
            errno = err;
            perror("pthread_create");
            exit(1);
        }
    }

    // report progress
    for (;;) {
        char percent[8];
        long count = atomic_load_explicit(&faults, memory_order_acquire);
        if (count == pages)
            break;
        FormatPercent(percent, (double)count / pages);
        tinyprint(2, "\rmemory map ", percent, "% loaded...\033[K", NULL);
        usleep(1. / FPS * 1e6);
    }
    tinyprint(2, "\r\033[K", NULL);

    // wait for workers
    for (int i = 0; i < THREADS; ++i)
        pthread_join(pf[i].th, 0);
}
