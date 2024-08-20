// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi

#ifdef __x86_64__

// run me:
//
//     make -j o//llamafile/llama_matmul_demo
//     o//llamafile/llama_matmul_demo
//

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
#define MDIM 4000
#endif

#ifndef NDIM
#define NDIM 4000
#endif

#ifndef KDIM
#define KDIM 4000
#endif

#ifndef NITER
#define NITER 10
#endif

////////////////////////////////////////////////////////////////////////////////
// BEGIN LLAMA MATMUL LIBRARY
////////////////////////////////////////////////////////////////////////////////

#define MEM_ALIGN 64

#define MR 16
#define NR 6

// Consider fine-tuning the following parameters for your CPU
#define NTHREADS 96
#define MC MR *NTHREADS * 4
#define NC NR *NTHREADS * 32
#define KC 1000

#define ALIGNED __attribute__((__aligned__(MEM_ALIGN)))

#define min(x, y) ((x) < (y) ? (x) : (y))

#define _mm256_loadu_hs(u16ptr) \
    _mm256_castsi256_ps( \
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(u16ptr))), 16))

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

static float blockB_packed[NC * KC] ALIGNED;
static uint16_t blockA_packed[MC * KC] ALIGNED;

static int8_t mask_32[32] ALIGNED = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

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
    for (int j = ith * NR; j < nc; j += NR * NTHREADS) {
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

static void pack_blockA(const uint16_t *A, uint16_t *blockA_packed, const int mc, const int kc,
                        const int K, const int ith) {
    for (int i = ith * MR; i < mc; i += MR * NTHREADS) {
        const int mr = min(MR, mc - i);
        pack_panelA(&A[i * K], &blockA_packed[i * kc], mr, kc, K);
    }
}

static void kernel_16x6(uint16_t *blockA_packed, float *blockB_packed, float *C, const int m,
                        const int n, const int k, const int M) {
    __m256 C_buffer[2][6];
    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;
    __m256i packed_masks[2] = {};
    if (m != 16) {
        packed_masks[0] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_32[16 - m]));
        packed_masks[1] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_32[16 - m + 8]));
        for (int j = 0; j < n; j++) {
            C_buffer[0][j] = _mm256_maskload_ps(&C[j * M], packed_masks[0]);
            C_buffer[1][j] = _mm256_maskload_ps(&C[j * M + 8], packed_masks[1]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            C_buffer[0][j] = _mm256_loadu_ps(&C[j * M]);
            C_buffer[1][j] = _mm256_loadu_ps(&C[j * M + 8]);
        }
    }
    for (int p = 0; p < k; p++) {
        a0_packFloat8 = _mm256_loadu_hs(blockA_packed);
        a1_packFloat8 = _mm256_loadu_hs(blockA_packed + 8);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C_buffer[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][0]);
        C_buffer[1][0] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][0]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C_buffer[0][1] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][1]);
        C_buffer[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C_buffer[0][2] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][2]);
        C_buffer[1][2] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][2]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C_buffer[0][3] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][3]);
        C_buffer[1][3] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][3]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C_buffer[0][4] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][4]);
        C_buffer[1][4] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][4]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C_buffer[0][5] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][5]);
        C_buffer[1][5] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][5]);

        blockA_packed += 16;
        blockB_packed += 6;
    }
    if (m != 16) {
        for (int j = 0; j < n; j++) {
            _mm256_maskstore_ps(&C[j * M], packed_masks[0], C_buffer[0][j]);
            _mm256_maskstore_ps(&C[j * M + 8], packed_masks[1], C_buffer[1][j]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            _mm256_storeu_ps(&C[j * M], C_buffer[0][j]);
            _mm256_storeu_ps(&C[j * M + 8], C_buffer[1][j]);
        }
    }
}

static void matmul_llama(uint16_t *A, uint16_t *B, float *C, const int M, const int N, const int K,
                         const int ith) {
    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            const int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K, ith);
            for (int i = 0; i < M; i += MC) {
                const int mc = min(MC, M - i);
                pack_blockA(&A[i * K + p], blockA_packed, mc, kc, K, ith);
                syncthreads(ith);
                for (int jr = ith * NR; jr < nc; jr += NR * NTHREADS) {
                    const int nr = min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = min(MR, mc - ir);
                        kernel_16x6(&blockA_packed[ir * kc], &blockB_packed[jr * kc],
                                    &C[(j + jr) * M + (i + ir)], mr, nr, kc, M);
                    }
                }
                syncthreads(ith);
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
