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

#include <cosmo.h>
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>

#ifdef __x86_64__

static errno_t pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    int abcd[4];
    __cpuidex(abcd, 7, 0);
    return !!(abcd[3] & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    int abcd[4];
    __cpuidex(abcd, 0x1a, 0);
    int intel_atom = 0x20;
    int core_type = (abcd[0] & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int count_math_cpus(int cpu_count) {
    int result = 0;
    for (int cpu = 0; cpu < cpu_count; ++cpu) {
        if (pin_cpu(cpu))
            return -1; // xnu and openbsd don't support affinity
        if (is_running_on_efficiency_core())
            continue; // efficiency cores harm lockstep threading
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

static void *count_math_cpus_worker(void *arg) {
    return (void *)(intptr_t)count_math_cpus((intptr_t)arg);
}

#endif // __x86_64__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int llamafile_get_math_cpu_count(void) {
    int cpu_count = __get_cpu_count();
    if (cpu_count < 1)
        return 4;
#ifdef __x86_64__
    if (is_hybrid_cpu()) {
        pthread_t th; // some OSes don't support getaffinity
        if (!pthread_create(&th, 0, count_math_cpus_worker, (void *)(intptr_t)cpu_count)) {
            void *result;
            if (!pthread_join(th, &result) && (intptr_t)result > 0)
                return (intptr_t)result;
        }
    }
#endif
    if (cpu_count <= 4)
        return cpu_count;
    else
        return cpu_count / 2;
}
