#pragma once

#ifdef __AVX512F__
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

#define BEGIN_KERNEL(RM, RN) \
    int ytiles = (m - m0) / RM; \
    int xtiles = (n - n0) / RN; \
    int tiles = ytiles * xtiles; \
    int duty = (tiles + nth - 1) / nth; \
    if (duty < 1) \
        duty = 1; \
    int start = duty * ith; \
    int end = start + duty; \
    if (end > tiles) \
        end = tiles; \
    for (int job = start; job < end; ++job) { \
        int i = m0 + job / xtiles * RM; \
        int j = n0 + job % xtiles * RN;

#define END_KERNEL() }

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
