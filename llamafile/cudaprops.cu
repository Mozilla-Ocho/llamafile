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

#include "cuda.h"
#include "tester.h"
#include <iostream>

#ifdef __HIP__
#define VENDOR "AMD"
#else
#define VENDOR "NVIDIA"
#endif

namespace {

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

} // namespace

int main() {
    int device_count;
    CUDA_OR_DIE(cudaGetDeviceCount(&device_count));
    for (int idev = 0; idev < device_count; ++idev) {
        show_cuda_device(idev);
    }
}
