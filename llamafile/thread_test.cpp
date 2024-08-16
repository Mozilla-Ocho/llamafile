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

#include <cosmo.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

#define WORKERS 20
#define MATHS 20

atomic_int barrier;

void *start_math(void *arg) {
    long id = (long)arg;
    ++barrier;
    if ((long)arg % 2 == 0)
        return 0;
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, 0);
    pthread_testcancel();
    for (;;) {
    }
}

void *start_worker(void *arg) {
    _Atomic(pthread_t) th[MATHS];
    for (long i = 0; i < MATHS; ++i)
        if (pthread_create((pthread_t *)&th[i], 0, start_math, (void *)i))
            _Exit(3);
    pthread_cleanup_push(
        [](void *arg) {
            _Atomic(pthread_t) *th = (_Atomic(pthread_t) *)arg;
            for (long i = 0; i < MATHS; ++i) {
                pthread_t t;
                if ((t = atomic_exchange(&th[i], 0))) {
                    pthread_cancel(t);
                    if (pthread_join(t, 0))
                        _Exit(4);
                }
            }
        },
        th);
    start_math(arg);
    pthread_setcancelstate(PTHREAD_CANCEL_MASKED, 0);
    for (long i = 0; i < MATHS; ++i) {
        pthread_t t;
        if ((t = atomic_exchange(&th[i], 0))) {
            int err = pthread_join(t, 0);
            if (err == ECANCELED) {
                th[i] = t;
                pthread_exit(PTHREAD_CANCELED);
            }
            if (err)
                _Exit(4);
        }
    }
    pthread_cleanup_pop(false);
    return 0;
}

int main() {
    pthread_t th[WORKERS];
    for (long i = 0; i < WORKERS; ++i)
        if (pthread_create(&th[i], 0, start_worker, (void *)i))
            _Exit(1);
    while (barrier < WORKERS * MATHS) {
    }
    for (int i = 0; i < WORKERS; ++i)
        pthread_cancel(th[i]);
    for (int i = 0; i < WORKERS; ++i)
        if (pthread_join(th[i], 0))
            _Exit(2);
    while (!pthread_orphan_np())
        pthread_decimate_np();
    CheckForMemoryLeaks();
}
