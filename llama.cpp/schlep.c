// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set net ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#define _COSMO_SOURCE
#include <cosmo.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/auxv.h>
#include <stdatomic.h>
#include "ggml.h"

#define FPS 24

struct Schlep {
    long n;
    atomic_long i;
};

static char Peek(volatile const char *ptr) {
    return *ptr;
}

char (*pPeek)(volatile const char *) = Peek;

static void FormatPercent(char sbuf[static 8], double x) {
    char *p = sbuf;
    int n = x * 100000.5;
    if (n >= 100000) *p++ = '0' + n / 100000 % 10;
    if (n >= 10000) *p++ = '0' + n / 10000 % 10;
    *p++ = '0' + n / 1000 % 10;
    *p++ = '.';
    *p++ = '0' + n / 100 % 10;
    *p++ = '0' + n / 10 % 10;
    *p++ = '0' + n % 10;
    *p = 0;
}

static void *ProgressReporter(void *arg) {
    struct Schlep *s = arg;
    pthread_setcancelstate(PTHREAD_CANCEL_MASKED, 0);
    while (!usleep(1. / FPS * 1e6)) {
        long i = atomic_load_explicit(&s->i, memory_order_acquire);
        char percent[8];
        FormatPercent(percent, (double)i / s->n);
        tinyprint(2, "\rmemory map ", percent, "% loaded...\033[K", NULL);
    }
    tinyprint(2, "\r\033[K", NULL);
    return 0;
}

/**
 * Loads memory off disk while reporting progress.
 */
void ggml_schlep(const void *data, size_t size) {

    // don't bother if memory is small
    if (size < 128 * 1024 * 1024) {
        return;
    }

    // don't bother if stderr isn't a terminal
    if (!isatty(2)) {
        return;
    }

    // setup shared memory
    struct Schlep s = {size};

    // get microprocessor page size
    long pagesz = getauxval(AT_PAGESZ);

    // create worker thread
    errno_t err;
    pthread_t th;
    sigset_t blockall;
    pthread_attr_t attr;
    sigfillset(&blockall);
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 65536);
    pthread_attr_setguardsize(&attr, pagesz);
    pthread_attr_setsigmask_np(&attr, &blockall);
    err = pthread_create(&th, &attr, ProgressReporter, &s);
    pthread_attr_destroy(&attr);
    if (err) {
        // don't bother without thread
        errno = err;
        perror("pthread_create");
        return;
    }

    // fault each page in memory region
    long i = 0;
    volatile const char *p = data;
    while (i < s.n) {
        pPeek(p + i);
        i += pagesz;
        atomic_store_explicit(&s.i, i, memory_order_release);
    }

    // terminate worker thread
    pthread_cancel(th);
    pthread_join(th, 0);
}
