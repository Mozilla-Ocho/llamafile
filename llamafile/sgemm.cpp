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
#include <cpuid.h>
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
#ifdef __x86_64__
    if (X86_HAVE(FMA)) {
        if (X86_HAVE(AVX2)) {
            if (X86_HAVE(AVX512F)) {
                if (X86_HAVE(AVX512VL) && X86_HAVE(AVX512_VNNI) && X86_HAVE(AVX512_BF16)) {
                    return llamafile_sgemm_amd_zen4(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task,
                                                    Atype, Btype, Ctype);
                } else {
                    return llamafile_sgemm_amd_avx512f(m, n, k, A, lda, B, ldb, C, ldc, ith, nth,
                                                       task, Atype, Btype, Ctype);
                }
            } else if (X86_HAVE(AVXVNNI)) {
                return llamafile_sgemm_amd_avxvnni(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task,
                                                   Atype, Btype, Ctype);
            } else {
                return llamafile_sgemm_amd_avx2(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task,
                                                Atype, Btype, Ctype);
            }
        } else {
            return llamafile_sgemm_amd_fma(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task, Atype,
                                           Btype, Ctype);
        }
    } else {
        return llamafile_sgemm_amd_avx(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task, Atype,
                                       Btype, Ctype);
    }
#elif defined(__aarch64__)
    if (hwcap & HWCAP_FPHP) {
        return llamafile_sgemm_arm82(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task, Atype, Btype,
                                     Ctype);
    } else {
        return llamafile_sgemm_arm80(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task, Atype, Btype,
                                     Ctype);
    }
#endif
}

/**
 * Performs "mixture of experts" tensor multiplication on CPU.
 */
bool llamafile_mixmul(const ggml_compute_params *params, const ggml_tensor *weights,
                      const ggml_tensor *thought, const ggml_tensor *plan, ggml_tensor *result) {
#ifdef __x86_64__
    if (X86_HAVE(FMA)) {
        if (X86_HAVE(AVX2)) {
            if (X86_HAVE(AVX512F)) {
                if (X86_HAVE(AVX512VL) && X86_HAVE(AVX512_VNNI) && X86_HAVE(AVX512_BF16)) {
                    return llamafile_mixmul_amd_zen4(params, weights, thought, plan, result);
                } else {
                    return llamafile_mixmul_amd_avx512f(params, weights, thought, plan, result);
                }
            } else if (X86_HAVE(AVXVNNI)) {
                return llamafile_mixmul_amd_avxvnni(params, weights, thought, plan, result);
            } else {
                return llamafile_mixmul_amd_avx2(params, weights, thought, plan, result);
            }
        } else {
            return llamafile_mixmul_amd_fma(params, weights, thought, plan, result);
        }
    } else {
        return llamafile_mixmul_amd_avx(params, weights, thought, plan, result);
    }
#elif defined(__aarch64__)
    if (hwcap & HWCAP_FPHP) {
        return llamafile_mixmul_arm82(params, weights, thought, plan, result);
    } else {
        return llamafile_mixmul_arm80(params, weights, thought, plan, result);
    }
#endif
}
