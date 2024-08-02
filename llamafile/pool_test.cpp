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

#include "pool.h"

#include <assert.h>
#include <cosmo.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>

#define BENCHMARK(ITERATIONS, WORK_PER_RUN, CODE) \
    do { \
        struct timespec start = timespec_real(); \
        for (int __i = 0; __i < ITERATIONS; ++__i) { \
            asm volatile("" ::: "memory"); \
            CODE; \
        } \
        long long work = ((WORK_PER_RUN) ? (WORK_PER_RUN) : 1) * (ITERATIONS); \
        double nanos = \
            (timespec_tonanos(timespec_sub(timespec_real(), start)) + work - 1) / (double)work; \
        if (nanos < 1000) { \
            printf("%10g ns %2dx %s\n", nanos, (ITERATIONS), #CODE); \
        } else { \
            printf("%10lld ns %2dx %s\n", (long long)nanos, (ITERATIONS), #CODE); \
        } \
    } while (0)

void *noop(void *arg) {
    return arg;
}

void run_task() {
    llamafile_task_t task;
    npassert(!llamafile_task_create(&task, noop, 0));
    npassert(!llamafile_task_join(task, 0));
}

void run_thread() {
    pthread_t task;
    npassert(!pthread_create(&task, 0, noop, 0));
    npassert(!pthread_join(task, 0));
}

#define N 20

void run_many_tasks() {
    llamafile_task_t task[N];
    for (int i = 0; i < N; ++i)
        npassert(!llamafile_task_create(&task[i], noop, 0));
    for (int i = 0; i < N; ++i)
        npassert(!llamafile_task_join(task[i], 0));
}

void run_many_threads() {
    pthread_t task[N];
    for (int i = 0; i < N; ++i)
        npassert(!pthread_create(&task[i], 0, noop, 0));
    for (int i = 0; i < N; ++i)
        npassert(!pthread_join(task[i], 0));
}

int main(int argc, char *argv[]) {
    ShowCrashReports();
    run_many_tasks();
    BENCHMARK(10, 1, run_task());
    BENCHMARK(10, 1, run_thread());
    BENCHMARK(10, N, run_many_tasks());
    BENCHMARK(10, N, run_many_threads());
    llamafile_task_shutdown();
    while (!pthread_orphan_np())
        pthread_decimate_np();
    CheckForMemoryLeaks();
}
