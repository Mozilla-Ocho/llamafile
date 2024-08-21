// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#ifdef __x86_64__

#include "llama_matmul.h"

#include <immintrin.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>

#define MR 64
#define NR 6

#define MX 2
#define NX 4

// Consider fine-tuning the following parameters for your CPU
#define MC (MX * MR * NTHREADS)
#define NC (NX * NR * NTHREADS)
#define KC 1000

#define ALIGNED __attribute__((__aligned__(64)))

#define min(x, y) ((x) < (y) ? (x) : (y))

#define _mm512_loadu_hs(u16ptr) \
    _mm512_castsi512_ps(_mm512_slli_epi32( \
        _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(u16ptr))), 16))

static void syncthreads(int ith) {
    static atomic_uint count;
    static struct {
        alignas(64) atomic_uint i;
    } ph[NTHREADS];
    int phase = atomic_load_explicit(&ph[ith].i, memory_order_relaxed);
    if (atomic_fetch_add_explicit(&count, 1, memory_order_acq_rel) + 1 == NTHREADS) {
        atomic_store_explicit(&count, 0, memory_order_relaxed);
        for (int i = 0; i < NTHREADS; ++i)
            atomic_store_explicit(&ph[i].i, phase + 1, memory_order_relaxed);
        atomic_thread_fence(memory_order_release);
    } else {
        for (;;)
            if (atomic_load_explicit(&ph[ith].i, memory_order_relaxed) != phase)
                break;
        atomic_thread_fence(memory_order_acquire);
    }
}

static float from_brain(uint16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h << 16;
    return u.f;
}

static void pack_panelB(const uint16_t *B, float *blockB_packed, const int nr, const int kc,
                        const long ldb) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = from_brain(B[j * ldb + p]);
        }
        for (int j = nr; j < NR; j++) {
            *blockB_packed++ = 0;
        }
    }
}

static void pack_blockB(const uint16_t *B, float *blockB_packed, const int nc, const int kc,
                        const long ldb, const int ith) {
    for (int j = ith * NR; j < nc; j += NR * NTHREADS) {
        const int nr = min(NR, nc - j);
        pack_panelB(&B[j * ldb], &blockB_packed[j * kc], nr, kc, ldb);
    }
}

static void pack_panelA(const uint16_t *A, uint16_t *blockA_packed, const int mr, const int kc,
                        const long lda) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[i * lda + p];
        }
        for (int i = mr; i < MR; i++) {
            *blockA_packed++ = 0;
        }
    }
}

static void pack_blockA(const uint16_t *A, uint16_t *blockA_packed, const int mc, const int kc,
                        const long lda, const int ith) {
    for (int i = ith * MR; i < mc; i += MR * NTHREADS) {
        const int mr = min(MR, mc - i);
        pack_panelA(&A[i * lda], &blockA_packed[i * kc], mr, kc, lda);
    }
}

static void kernel_64x6(uint16_t *blockA_packed, float *blockB_packed, float *C, const int m,
                        const int n, const int k, const int ldc) {
    __m512 C_buffer[4][6];
    __m512 a_packFloat16[4];
    __m512 b_packFloat16;
    __mmask16 mask[4] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
    if (m != 64) {
        for (int i = 0; i < 4; i++) {
            mask[i] = (m > i * 16)
                          ? (__mmask16)((1ULL << ((m - i * 16) > 16 ? 16 : (m - i * 16))) - 1)
                          : 0x0000;
        }
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 4; i++) {
                C_buffer[i][j] = _mm512_maskz_loadu_ps(mask[i], &C[j * ldc + i * 16]);
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 4; i++) {
                C_buffer[i][j] = _mm512_loadu_ps(&C[j * ldc + i * 16]);
            }
        }
    }
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < 4; i++) {
            a_packFloat16[i] = _mm512_loadu_hs(blockA_packed + i * 16);
        }
        for (int j = 0; j < 6; j++) {
            b_packFloat16 = _mm512_set1_ps(blockB_packed[j]);
            for (int i = 0; i < 4; i++) {
                C_buffer[i][j] = _mm512_fmadd_ps(a_packFloat16[i], b_packFloat16, C_buffer[i][j]);
            }
        }
        blockA_packed += 64;
        blockB_packed += 6;
    }
    if (m != 64) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 4; i++) {
                _mm512_mask_storeu_ps(&C[j * ldc + i * 16], mask[i], C_buffer[i][j]);
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps(&C[j * ldc + i * 16], C_buffer[i][j]);
            }
        }
    }
}

static void clear_tile(float *C, const int m, const int n, const long ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C[j * ldc + i] = 0;
        }
    }
}

static float blockB_packed[NC * KC] ALIGNED;
static uint16_t blockA_packed[MC * KC] ALIGNED;

void llama_matmul_avx512_bf16(const uint16_t *A, const uint16_t *B, float *C, const int M,
                              const int N, const int K, const long lda, const long ldb,
                              const long ldc, const int ith) {

    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int i = 0; i < M; i += MC) {
            const int mc = min(MC, M - i);
            for (int jr = ith * NR; jr < nc; jr += NR * NTHREADS) {
                const int nr = min(NR, nc - jr);
                for (int ir = 0; ir < mc; ir += MR) {
                    const int mr = min(MR, mc - ir);
                    clear_tile(&C[(j + jr) * ldc + (i + ir)], mr, nr, ldc);
                }
            }
        }
    }

    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            const int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, ldb, ith);
            for (int i = 0; i < M; i += MC) {
                const int mc = min(MC, M - i);

                float blockC[(NX * NR) * (MX * MR)] ALIGNED = {0};
                for (int jr = ith * NR; jr < nc; jr += NR * NTHREADS) {
                    const int nr = min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = min(MR, mc - ir);
                        for (int jb = 0; jb < nr; ++jb) {
                            for (int ib = 0; ib < mr; ++ib) {
                                blockC[(jr / NTHREADS + jb) * (MX * MR) + (ir / NTHREADS + ib)] =
                                    C[(j + jr + jb) * ldc + (i + ir + ib)];
                            }
                        }
                    }
                }

                pack_blockA(&A[i * K + p], blockA_packed, mc, kc, lda, ith);
                syncthreads(ith);
                for (int jr = ith * NR; jr < nc; jr += NR * NTHREADS) {
                    const int nr = min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = min(MR, mc - ir);
                        kernel_64x6(&blockA_packed[ir * kc], &blockB_packed[jr * kc],
                                    &blockC[jr / NTHREADS * (MX * MR) + ir / NTHREADS], MR, NR, kc,
                                    MX * MR);
                    }
                }

                for (int jr = ith * NR; jr < nc; jr += NR * NTHREADS) {
                    const int nr = min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = min(MR, mc - ir);
                        for (int jb = 0; jb < nr; ++jb) {
                            for (int ib = 0; ib < mr; ++ib) {
                                C[(j + jr + jb) * ldc + (i + ir + ib)] =
                                    blockC[(jr / NTHREADS + jb) * (MX * MR) + (ir / NTHREADS + ib)];
                            }
                        }
                    }
                }

                syncthreads(ith);
            }
        }
    }
}

#endif // __x86_64__
