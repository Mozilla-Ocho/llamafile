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

#ifdef __x86_64__

#include "sgemm.h"
#include <immintrin.h>

#define KN 8

typedef __m256 V;
typedef __m256 D;
typedef ggml_bf16_t TA;
typedef float TB;
typedef float TC;

static inline __m256 zero() {
    return _mm256_setzero_ps();
}

static inline __m256 load(const float *p) {
    return _mm256_loadu_ps(p);
}

static inline __m256 load(const ggml_bf16_t *p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}

#include "sgemmer.inc"

bool llamafile_sgemm_bss_avx2(int m, int n, int k, const TA *A, int lda, const TB *B, int ldb,
                              TC *C, int ldc, int ith, int nth, int task) {
    if (task != GGML_TASK_TYPE_COMPUTE)
        return true;
    SGEMMER tb{k, A, lda, B, ldb, C, ldc, ith, nth};
    tb.matmul(m, n);
    return true;
}

#endif // __x86_64__
