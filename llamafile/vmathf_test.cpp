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

#include "float.h"
#include "numba.h"

#include <assert.h>
#include <cosmo.h>
#include <math.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <unistd.h>

#include "llama.cpp/ggml-vector.h"

extern "C" void ggml_vec_gelu_f32_amd_avx512bf16(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_amd_avx512vl(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_amd_avx512(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_amd_avx2(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_amd_f16c(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_amd_fma(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_amd_avx(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_amd_ssse3(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_amd_k8(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_arm82(const int n, float *y, const float *x);
extern "C" void ggml_vec_gelu_f32_arm80(const int n, float *y, const float *x);

extern "C" void ggml_vec_silu_f32_amd_avx512bf16(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_amd_avx512vl(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_amd_avx512(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_amd_avx2(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_amd_f16c(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_amd_fma(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_amd_avx(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_amd_ssse3(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_amd_k8(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_arm82(const int n, float *y, const float *x);
extern "C" void ggml_vec_silu_f32_arm80(const int n, float *y, const float *x);

#define N 256

float mathf(void vmathf(int, float *, const float *), float x) {
    float A[1] = {x};
    float B[1];
    vmathf(1, B, A);
    return B[0];
}

void test_vmathf(void vmathf(int, float *, const float *)) {

    // create page protected mappings that let us verify no memory
    // access overlaps the end of an array.

    int pagesz = sysconf(_SC_PAGESIZE);
    int need = N + sizeof(float);
    int greed = (need + pagesz - 1) & -pagesz;

    char *map1 =
        (char *)mmap(0, greed + pagesz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    npassert(map1 != MAP_FAILED);
    npassert(!mprotect(map1 + greed, pagesz, PROT_NONE));

    char *map2 =
        (char *)mmap(0, greed + pagesz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    npassert(map2 != MAP_FAILED);
    npassert(!mprotect(map2 + greed, pagesz, PROT_NONE));

    // verify the vectorized operation is internally consistent, i.e.
    // checking the single-element version of itself yields identical
    // values as the N-element version of itself.

    float *AE = (float *)(map1 + greed);
    float *BE = (float *)(map2 + greed);
    for (int n = 0; n < N; ++n) {
        float *A = AE - n;
        float *B = BE - n;
        for (int i = 0; i < n; ++i) {
            if (rand32() % 2 == 0) {
                A[i] = numba();
            } else {
                A[i] = flt::tofloat(rand32());
            }
        }
        vmathf(n, B, A);
        for (int i = 0; i < n; ++i) {
            float x = B[i];
            float y = mathf(vmathf, A[i]);
            npassert(flt::toint(x) == flt::toint(y));
        }
    }

    npassert(!munmap(map2, greed + pagesz));
    npassert(!munmap(map1, greed + pagesz));
}

int main(int argc, char *argv[]) {
    ShowCrashReports();

    test_vmathf(ggml_vec_gelu_f32);
    test_vmathf(ggml_vec_silu_f32);

#ifdef __x86_64__

    if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) &&
        X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL) && X86_HAVE(AVX512_BF16)) {
        test_vmathf(ggml_vec_gelu_f32_amd_avx512bf16);
        test_vmathf(ggml_vec_silu_f32_amd_avx512bf16);
    }

    if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) &&
        X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL)) {
        test_vmathf(ggml_vec_gelu_f32_amd_avx512vl);
        test_vmathf(ggml_vec_silu_f32_amd_avx512vl);
    }

    if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F)) {
        test_vmathf(ggml_vec_gelu_f32_amd_avx512);
        test_vmathf(ggml_vec_silu_f32_amd_avx512);
    }

    if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2)) {
        test_vmathf(ggml_vec_gelu_f32_amd_avx2);
        test_vmathf(ggml_vec_silu_f32_amd_avx2);
    }

    if (X86_HAVE(AVX) && X86_HAVE(F16C)) {
        test_vmathf(ggml_vec_gelu_f32_amd_f16c);
        test_vmathf(ggml_vec_silu_f32_amd_f16c);
    }

    if (X86_HAVE(AVX) && X86_HAVE(FMA)) {
        test_vmathf(ggml_vec_gelu_f32_amd_fma);
        test_vmathf(ggml_vec_silu_f32_amd_fma);
    }

    if (X86_HAVE(AVX)) {
        test_vmathf(ggml_vec_gelu_f32_amd_avx);
        test_vmathf(ggml_vec_silu_f32_amd_avx);
    }

    if (X86_HAVE(SSSE3)) {
        test_vmathf(ggml_vec_gelu_f32_amd_ssse3);
        test_vmathf(ggml_vec_silu_f32_amd_ssse3);
    }

    test_vmathf(ggml_vec_gelu_f32_amd_k8);
    test_vmathf(ggml_vec_silu_f32_amd_k8);

#elif defined(__aarch64__)

    if ((getauxval(AT_HWCAP) & HWCAP_FPHP) && (getauxval(AT_HWCAP) & HWCAP_ASIMDHP)) {
        test_vmathf(ggml_vec_gelu_f32_arm82);
        test_vmathf(ggml_vec_silu_f32_arm82);
    }

    test_vmathf(ggml_vec_gelu_f32_arm80);
    test_vmathf(ggml_vec_silu_f32_arm80);

#endif

    CheckForMemoryLeaks();
    return 0;
}
