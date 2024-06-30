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

#include "thread.h"

#include <cosmo.h>
#include <stdatomic.h>
#include <stdlib.h>

#define WORKERS 10
#define MATHS 10

atomic_int barrier;

void *start_math(void *arg) {
    ++barrier;
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, 0);
    pthread_testcancel();
    for (;;) {
    }
}

void *start_worker(void *arg) {
    ++barrier;
    pthread_t th[MATHS];
    for (int i = 0; i < MATHS; ++i)
        if (llamafile_thread_create(&th[i], 0, start_math, 0))
            _Exit(3);
    for (int i = 0; i < MATHS; ++i)
        if (pthread_join(th[i], 0))
            _Exit(4);
    return 0;
}

int main() {
    pthread_t th[WORKERS];
    for (int i = 0; i < WORKERS; ++i)
        if (llamafile_thread_create(&th[i], 0, start_worker, 0))
            _Exit(1);
    while (barrier < WORKERS * MATHS) {
    }
    for (int i = 0; i < WORKERS; ++i)
        pthread_cancel(th[i]);
    for (int i = 0; i < WORKERS; ++i)
        if (pthread_join(th[i], 0))
            _Exit(2);
    CheckForMemoryLeaks();
}
