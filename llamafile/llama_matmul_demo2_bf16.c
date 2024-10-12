// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi

#ifdef __x86_64__

// run me:
//
//     make -j o//llamafile/llama_matmul_demo2_bf16
//     o//llamafile/llama_matmul_demo2_bf16
//
// - demo  => GFLOPS = 528.307
// - demo2 => GFLOPS = 934.148

#include <immintrin.h>
#include <math.h>
#include <sched.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define TEST 1

#ifndef MDIM
#define MDIM 6000
#endif

#ifndef NDIM
#define NDIM 6000
#endif

#ifndef KDIM
#define KDIM 6000
#endif

#ifndef NITER
#define NITER 10
#endif

////////////////////////////////////////////////////////////////////////////////
// BEGIN LLAMA MATMUL LIBRARY
////////////////////////////////////////////////////////////////////////////////

#define MEM_ALIGN 64

#define MR 32
#define NR 12

// Consider fine-tuning the following parameters for your CPU
// NB_BLOC is not implemented, do not change it for now.
#define NB_CORE 8
#define NB_BLOC 1
#define NTHREADS (NB_CORE*NB_BLOC)

#define MC MR*4
#define NC NR*32
#define KC 1000

#define MT MC*NTHREADS // TODO: NB_CORE_PER_L3
#define NT NC*NB_BLOC  // TODO: NB_L3


#define ALIGNED __attribute__((__aligned__(MEM_ALIGN)))

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

// TODO: use bf16 for blockB too!!!
static float    blockB_packed[NT*KC] ALIGNED;
static uint16_t blockA_packed[NTHREADS][MC*KC] ALIGNED;

static float from_brain(uint16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h << 16;
    return u.f;
}

static void pack_panelB(const uint16_t *B, float *blockB_packed, const int nr, const int kc,
                        const int K) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = from_brain(B[j * K + p]);
        }
        for (int j = nr; j < NR; j++) {
            *blockB_packed++ = 0;
        }
    }
}

static void pack_blockB(const uint16_t *B, float *blockB_packed, const int nc, const int kc,
                        const int K, const int ith) {
    for (int j = ith * NR; j < nc; j += NR*NTHREADS) {
        const int nr = min(NR, nc - j);
        pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
    }
}

static void pack_panelA(const uint16_t *A, uint16_t *blockA_packed, const int mr, const int kc,
                        const int K) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[i * K + p];
        }
        for (int i = mr; i < MR; i++) {
            *blockA_packed++ = 0;
        }
    }
}

// 1 blockA per THREAD: blockA_packed[NTHREADS][MC/MR][MR]
static void pack_blockA(const uint16_t *A, uint16_t *blockA_packed, const int mc, const int kc, const int K) {
    for (int i = 0; i < MC; i += MR) {
        const int mr = min(MR, mc - i);
        pack_panelA(&A[i * K], &blockA_packed[i * kc], mr, kc, K);
    }
}

static void kernel_32x12(uint16_t *blockA_packed, float *blockB_packed, float *C, const int m,
                        const int n, const int k, const int M) {
    __m512 C_buffer[2][12];
    __m512 a_packFloat16[2];
    __m512 b_packFloat16;
    __mmask16 mask[4] = {0xFFFF, 0xFFFF};
    if (m != 32) {
        for (int i = 0; i < 2; i++) {
            mask[i] = (m > i * 16)
                          ? (__mmask16)((1ULL << ((m - i * 16) > 16 ? 16 : (m - i * 16))) - 1)
                          : 0x0000;
        }
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 2; i++) {
                C_buffer[i][j] = _mm512_maskz_loadu_ps(mask[i], &C[j * M + i * 16]);
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 2; i++) {
                C_buffer[i][j] = _mm512_loadu_ps(&C[j * M + i * 16]);
            }
        }
    }
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < 2; i++) {
            a_packFloat16[i] = _mm512_loadu_hs(blockA_packed + i * 16);
        }
        for (int j = 0; j < 12; j++) {
            b_packFloat16 = _mm512_set1_ps(blockB_packed[j]);
            for (int i = 0; i < 2; i++) {
                C_buffer[i][j] = _mm512_fmadd_ps(a_packFloat16[i], b_packFloat16, C_buffer[i][j]);
            }
        }
        blockA_packed += 32;
        blockB_packed += 12;
    }
    if (m != 32) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 2; i++) {
                _mm512_mask_storeu_ps(&C[j * M + i * 16], mask[i], C_buffer[i][j]);
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < 2; i++) {
                _mm512_storeu_ps(&C[j * M + i * 16], C_buffer[i][j]);
            }
        }
    }
}

static void matmul_llama(uint16_t *A, uint16_t *B, float *C, const int M, const int N, const int K, const int ith) {
    for (int j = 0; j < N; j += NT) {  // TODO: add // level per_l3_cache
        const int nt = min(NT, N - j);
        for (int p = 0; p < K; p += KC) {
            const int kc = min(KC, K - p);
            
            pack_blockB(&B[j * K + p], blockB_packed, nt, kc, K, ith);
            
            syncthreads(ith);

            for (int i = ith * MC; i < M; i += MT) {  // this is the on that is //
                const int mc = min(MC, M - i);
                pack_blockA(&A[i*K + p], blockA_packed[ith], mc, kc, K);
                for (int jr = 0; jr < nt; jr += NR) {
                    const int nr = min(NR, nt - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = min(MR, mc - ir);
                        kernel_32x12(&blockA_packed[ith][ir * kc], &blockB_packed[jr * kc],
                                     &C[(j + jr) * M + (i + ir)], mr, nr, kc, M);
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// END LLAMA MATMUL LIBRARY
////////////////////////////////////////////////////////////////////////////////

static void matmul_llama_mt(uint16_t *A, uint16_t *B, float *C, const int M, const int N,
                            const int K) {
#pragma omp parallel for num_threads(NTHREADS)
    for (int ith = 0; ith < NTHREADS; ++ith)
        matmul_llama(A, B, C, M, N, K, ith);
}

static uint16_t to_brain(float s) {
    uint16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h = (u.i >> 16) | 64; /* force to quiet */
        return h;
    }
    return (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
}

static void matmul_naive(uint16_t *A, uint16_t *B, float *C, const int M, const int N,
                         const int K) {
#pragma omp parallel for num_threads(NTHREADS) collapse(2) schedule(static)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int p = 0; p < K; p++)
                C[j * M + i] += from_brain(A[i * K + p]) * from_brain(B[j * K + p]);
}

static void print_mat(float *mat, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%f ", mat[i * N + j]);
        printf("\n");
    }
    printf("\n");
}

static void init_rand(uint16_t *mat, const int M, const int N) {
    for (int i = 0; i < M * N; i++)
        *mat++ = to_brain(rand() / (float)INT_MAX);
}

static void init_const(float *mat, const float value, const int M, const int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            *mat++ = value;
}

static void compare_mats(float *mat1, float *mat2, const int M, const int N) {
    int ouch = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabsf(mat1[j * M + i] - mat2[j * M + i]) > 1e-4) {
                printf("MISMATCH! Element[%d][%d] %f != %f\n", i, j, mat1[j * M + i],
                       mat2[j * M + i]);
                if (ouch++ == 15)
                    return;
            }
        }
    }
    printf("MATCH!\n");
    return;
}

static uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main() {
    const int M = MDIM;
    const int N = NDIM;
    const int K = KDIM;
    uint16_t *A = (uint16_t *)_mm_malloc(M * K * sizeof(uint16_t), MEM_ALIGN);
    uint16_t *B = (uint16_t *)_mm_malloc(K * N * sizeof(uint16_t), MEM_ALIGN);
    float *C = (float *)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float *C_ref = (float *)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    init_rand(A, M, K);
    init_rand(B, K, N);

#ifdef TEST
    matmul_naive(A, B, C_ref, M, N, K);
#endif
    double FLOP = 2 * (double)M * N * K;

    for (int i = 0; i < NITER; i++) {
        init_const(C, 0, M, N);
        uint64_t start = timer();
        matmul_llama_mt(A, B, C, M, N, K);
        uint64_t end = timer();

        double exec_time = (end - start) * 1e-9;
        double FLOPS = FLOP / exec_time;

        printf("Exec. time = %.3fms\n", exec_time * 1000);
        printf("GFLOPS = %.3f\n", FLOPS / 1e9);
#ifdef TEST
        compare_mats(C, C_ref, M, N);
#endif
        printf("\n");
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_ref);

    return 0;
}

#else
int main(int argc, char *argv[]) {
}
#endif /* __x86_64__ */
