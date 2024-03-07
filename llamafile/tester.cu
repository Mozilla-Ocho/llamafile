// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#include "tester.h"

#include <algorithm>
#include <atomic>
#include <ctime>

#ifndef _WIN32
#include <unistd.h>
#else
#include "windows.h"
#endif

std::recursive_mutex g_log_lock;
thread_local const char *is_self_testing;

const int kPageSize =
#ifndef _WIN32
    std::max(sysconf(_SC_PAGESIZE), 4096l);
#else
    4096;
#endif

int rand32(void) {
    static unsigned long long lcg = 1;
    lcg *= 6364136223846793005;
    lcg += 1442695040888963407;
    return lcg >> 32;
}

float float01(unsigned x) { // (0,1)
    return 1.f / 8388608 * ((x >> 9) + .5f);
}

float numba(void) { // (-1,1)
    return float01(rand32()) * 2 - 1;
}

int hamming(int x, int y) {
    return popcount(x ^ y);
}

int popcount(unsigned x) {
    x = x - ((x >> 1) & 0x55555555);
    x = ((x >> 2) & 0x33333333) + (x & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = (x + (x >> 16));
    return (x + (x >> 8)) & 0x0000003F;
}

void cudaFreeOrDie(void *p) {
    CUDA_OR_DIE(cudaFree(p));
}

void *cudaMallocManagedOrDie(size_t n) {
    void *p;
    CUDA_OR_DIE(cudaMallocManaged(&p, n));
    return p;
}

#ifdef _WIN32
static long long GetQueryPerformanceFrequency() {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    return t.QuadPart;
}
static long long GetQueryPerformanceCounter() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}
#endif

long long micros(void) {
#ifndef _WIN32
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000 + (ts.tv_nsec + 999) / 1000;
#else
    static long long timer_freq = GetQueryPerformanceFrequency();
    static long long timer_start = GetQueryPerformanceCounter();
    return ((GetQueryPerformanceCounter() - timer_start) * 1000000) / timer_freq;
#endif
}

[[noreturn]] void cuda_die(const char *stmt, const char *func, const char *file, int line,
                           const char *msg) {
    int id = -1;
    (void)cudaGetDevice(&id);
    fprintf(stderr, "CUDA error: %s\n", msg);
    fprintf(stderr, "  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    fprintf(stderr, "  %s\n", stmt);
    (void)cudaDeviceReset();
    _Exit(1);
}

void test_matmul(std::function<void(int m, int n, int k, int l, float alpha, float beta)> f) {
    static const int kDims[] = {1, 2, 23, 77, 15, 2048, 512, 127, 129, 128, 16};
    static const float kAlphas[] = {1, .1};
    static const float kBetas[] = {0, .1};
    static const int kLeads[] = {0, 1};
    std::atomic_llong t = ATOMIC_VAR_INIT(0);
    for (int mi = 0; mi < ARRAYLEN(kDims); ++mi)
        for (int ni = 0; ni < ARRAYLEN(kDims); ++ni)
            for (int li = 0; li < ARRAYLEN(kLeads); ++li)
                for (int ki = 0; ki < ARRAYLEN(kDims); ++ki) {
                    int m = kDims[mi];
                    int n = kDims[ARRAYLEN(kDims) - 1 - ni];
                    int k = kDims[ki];
                    int l = kLeads[li];
                    char name[256];
                    sprintf(name, "testing %4d %4d %4d ld+%d", m, n, k, l);
                    if (t++ % 7 == 0)
                        fprintf(stderr, "%s\r", name);
                    is_self_testing = name;
                    f(m, n, k, l, 1, 0);
                    is_self_testing = 0;
                }
    for (int ai = 0; ai < ARRAYLEN(kAlphas); ++ai)
        for (int bi = 0; bi < ARRAYLEN(kBetas); ++bi) {
            float alpha = kAlphas[ai];
            float beta = kBetas[bi];
            int m = 128;
            int n = 128;
            int k = 128;
            char name[256];
            sprintf(name, "testing %4d %4d %4d alpha=%g beta=%g", m, n, k, alpha, beta);
            if (t++ % 7 == 0)
                fprintf(stderr, "%s\r", name);
            is_self_testing = name;
            f(m, n, k, 0, alpha, beta);
            is_self_testing = 0;
        }
}

static int cuda_tester_init() {
    CUDA_OR_DIE(cudaSetDevice(0));
#ifdef __HIP__
    rocblas_initialize();
#endif
    CUDA_OR_DIE(cudaDeviceSynchronize());
    return 0;
}

const int cuda_tester_ = cuda_tester_init();
