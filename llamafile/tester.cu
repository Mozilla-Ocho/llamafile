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

#include "tester.h"

#include <algorithm>
#include <atomic>
#include <iostream>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __HIP__
#define VENDOR "AMD"
#else
#define VENDOR "NVIDIA"
#endif

std::recursive_mutex g_log_lock;
thread_local const char *is_self_testing;

const int kPageSize =
#ifndef _WIN32
    std::max(sysconf(_SC_PAGESIZE), 4096l);
#else
    4096;
#endif

void cudaFreeOrDie(void *p) {
    CUDA_OR_DIE(cudaFree(p));
}

void *cudaMallocManagedOrDie(size_t n) {
    void *p;
    CUDA_OR_DIE(cudaMallocManaged(&p, n));
    return p;
}

[[noreturn]] void cuda_die(const char *stmt, const char *func, const char *file, int line,
                           const char *msg) {
    int id = -1;
    (void)cudaGetDevice(&id);
    fprintf(stderr, "CUDA error: %s%s%s%s\n", msg, is_self_testing ? " (" : "",
            is_self_testing ? is_self_testing : "", is_self_testing ? ")" : "");
    fprintf(stderr, "  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    fprintf(stderr, "  %s\n", stmt);
    (void)cudaDeviceReset();
    _Exit(1);
}

void test_matmul(std::function<void(int m, int n, int k, int l, float alpha, float beta)> f) {
    static const int kDims[] = {1, 2, 23, 65, 63, 64, 1024, 512, 127, 129, 128, 16};
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

void show_cuda_device(int idev) {
    cudaDeviceProp dp;
    CUDA_OR_DIE(cudaSetDevice(idev));
    CUDA_OR_DIE(cudaGetDeviceProperties(&dp, idev));
    fprintf(stderr,
            "\n" VENDOR " %s\n"
            "\t%.2f Ghz\n"
            "\t%g GB VRAM\n"
            "\t%d CPUs\n"
            "\tcompute_%d\n",
            dp.name, dp.clockRate * 1e-6, dp.totalGlobalMem / 1073741824., dp.multiProcessorCount,
            dp.major * 10 + dp.minor);
#define SHOW(s) std::cout << #s << " = " << dp.s << std::endl;
    SHOW(warpSize);
    SHOW(regsPerBlock);
    SHOW(regsPerMultiprocessor);
    SHOW(multiProcessorCount);
    SHOW(memoryBusWidth);
    SHOW(memoryClockRate);
    SHOW(sharedMemPerBlock);
    SHOW(sharedMemPerMultiprocessor);
    SHOW(totalConstMem);
    SHOW(maxThreadsPerBlock);
    SHOW(maxThreadsPerMultiProcessor);
    SHOW(maxThreadsDim[0]);
    SHOW(maxThreadsDim[1]);
    SHOW(maxThreadsDim[2]);
    SHOW(asyncEngineCount);
    SHOW(managedMemory);
    SHOW(concurrentManagedAccess);
    SHOW(directManagedMemAccessFromHost);
    SHOW(globalL1CacheSupported);
    SHOW(hostNativeAtomicSupported);
    // SHOW(hostRegisterReadOnlySupported);
    // SHOW(hostRegisterSupported);
    SHOW(integrated);
    SHOW(l2CacheSize);
}

void show_cuda_devices() {
    int device_count;
    CUDA_OR_DIE(cudaGetDeviceCount(&device_count));
    for (int idev = 0; idev < device_count; ++idev) {
        show_cuda_device(idev);
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
