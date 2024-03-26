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

#include "sgemm.h"
#include "llamafile.h"
#include <cassert>
#include <cosmo.h>
#include <libc/sysv/consts/hwcap.h>
#include <sys/auxv.h>

static const long hwcap = getauxval(AT_HWCAP);

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param task is GGML task type
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(int m, int n, int k, const void *A, int lda, const void *B, int ldb, void *C,
                     int ldc, int ith, int nth, int task, int Atype, int Btype, int Ctype) {

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(nth > 0);
    assert(ith < nth);

    if (Ctype != GGML_TYPE_F32)
        return false;

    switch (Atype) {

    case GGML_TYPE_F32:
        if (Btype != GGML_TYPE_F32)
            return false;
#ifdef __x86_64__
        if (!X86_HAVE(AVX))
            return false;
        if (X86_HAVE(AVX512F) && !(k % 16))
            return llamafile_sgemm_sss_avx512f(m, n, k, (const float *)A, lda, (const float *)B,
                                               ldb, (float *)C, ldc, ith, nth, task);
        if (X86_HAVE(FMA) && !(k % 8))
            return llamafile_sgemm_sss_fma(m, n, k, (const float *)A, lda, (const float *)B, ldb,
                                           (float *)C, ldc, ith, nth, task);
        if (!(k % 8))
            return llamafile_sgemm_sss_avx(m, n, k, (const float *)A, lda, (const float *)B, ldb,
                                           (float *)C, ldc, ith, nth, task);
#elif defined(__aarch64__)
        if (n > 1 && !(k % 4))
            return llamafile_sgemm_sss_neon(m, n, k, (const float *)A, lda, (const float *)B, ldb,
                                            (float *)C, ldc, ith, nth, task);
#endif
        return false;

    case GGML_TYPE_F16:
#ifdef __x86_64__
        if (!X86_HAVE(AVX))
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        if (X86_HAVE(AVX512F) && !(k % 16))
            return llamafile_sgemm_hss_avx512f(m, n, k, (const unsigned short *)A, lda,
                                               (const float *)B, ldb, (float *)C, ldc, ith, nth,
                                               task);
        if (X86_HAVE(FMA) && X86_HAVE(F16C) && !(k % 8))
            return llamafile_sgemm_hss_f16c(m, n, k, (const unsigned short *)A, lda,
                                            (const float *)B, ldb, (float *)C, ldc, ith, nth, task);
#elif defined(__aarch64__)
        if (n > 1 && !(k % 8) && (hwcap & HWCAP_FPHP) && Btype == GGML_TYPE_F16)
            return llamafile_sgemm_hhs_neon(m, n, k, (const unsigned short *)A, lda,
                                            (const unsigned short *)B, ldb, (float *)C, ldc, ith,
                                            nth, task);
        if (n > 1 && !(k % 4) && !(hwcap & HWCAP_FPHP) && Btype == GGML_TYPE_F32)
            return llamafile_sgemm_hss_neon(m, n, k, (const unsigned short *)A, lda,
                                            (const float *)B, ldb, (float *)C, ldc, ith, nth, task);
#endif
        return false;

    case GGML_TYPE_Q8_0:
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#ifdef __x86_64__
        if (!X86_HAVE(AVX))
            return false;
        if (X86_HAVE(AVX512VL) && X86_HAVE(AVX512_VNNI))
            return llamafile_sgemm_q0q0s_avx512vnni(m, n, k, (const block_q8_0 *)A, lda,
                                                    (const block_q8_0 *)B, ldb, (float *)C, ldc,
                                                    ith, nth, task);
        if (X86_HAVE(FMA) && X86_HAVE(AVXVNNI))
            return llamafile_sgemm_q0q0s_avxvnni(m, n, k, (const block_q8_0 *)A, lda,
                                                 (const block_q8_0 *)B, ldb, (float *)C, ldc, ith,
                                                 nth, task);
        if (X86_HAVE(FMA) && X86_HAVE(AVX2))
            return llamafile_sgemm_q0q0s_fma(m, n, k, (const block_q8_0 *)A, lda,
                                             (const block_q8_0 *)B, ldb, (float *)C, ldc, ith, nth,
                                             task);
#elif defined(__aarch64__)
        return llamafile_sgemm_q0q0s_dotprod(m, n, k, (const block_q8_0 *)A, lda,
                                             (const block_q8_0 *)B, ldb, (float *)C, ldc, ith, nth,
                                             task);
#endif
        return false;

    case GGML_TYPE_Q4_0:
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#ifdef __x86_64__
        if (!X86_HAVE(AVX))
            return false;
        if (X86_HAVE(AVX512VL) && X86_HAVE(AVX512_VNNI))
            return llamafile_sgemm_e0q0s_avx512vnni(m, n, k, (const block_q4_0 *)A, lda,
                                                    (const block_q8_0 *)B, ldb, (float *)C, ldc,
                                                    ith, nth, task);
        if (X86_HAVE(FMA) && X86_HAVE(AVXVNNI))
            return llamafile_sgemm_e0q0s_avxvnni(m, n, k, (const block_q4_0 *)A, lda,
                                                 (const block_q8_0 *)B, ldb, (float *)C, ldc, ith,
                                                 nth, task);
        if (X86_HAVE(FMA) && X86_HAVE(AVX2))
            return llamafile_sgemm_e0q0s_fma(m, n, k, (const block_q4_0 *)A, lda,
                                             (const block_q8_0 *)B, ldb, (float *)C, ldc, ith, nth,
                                             task);
#endif
        return false;

    case GGML_TYPE_Q4_1:
        if (Btype != GGML_TYPE_Q8_1)
            return false;
#ifdef __x86_64__
        if (!X86_HAVE(AVX))
            return false;
        if (X86_HAVE(AVX512VL) && X86_HAVE(AVX512_VNNI))
            return llamafile_sgemm_e1q1s_avx512vnni(m, n, k, (const block_q4_1 *)A, lda,
                                                    (const block_q8_1 *)B, ldb, (float *)C, ldc,
                                                    ith, nth, task);
        if (X86_HAVE(FMA) && X86_HAVE(AVXVNNI))
            return llamafile_sgemm_e1q1s_avxvnni(m, n, k, (const block_q4_1 *)A, lda,
                                                 (const block_q8_1 *)B, ldb, (float *)C, ldc, ith,
                                                 nth, task);
        if (X86_HAVE(FMA) && X86_HAVE(AVX2))
            return llamafile_sgemm_e1q1s_fma(m, n, k, (const block_q4_1 *)A, lda,
                                             (const block_q8_1 *)B, ldb, (float *)C, ldc, ith, nth,
                                             task);
#endif
        return false;

    default:
        return false;
    }
}
