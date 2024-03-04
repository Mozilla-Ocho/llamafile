// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi

#pragma once

#ifdef __NVCC__
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#endif

#define ITERATIONS 30

extern const char *is_self_testing;

float float01(unsigned);
float numba(void);
int rand32(void);
long long micros(void);
void bench(long long, const char *);
void broadcast(int, int, float *, int, float);
void broadcast(int, int, half *, int, half);
void check(double, int, int, const float *, int, const float *, int, const char *, int);
void check(double, int, int, const half *, int, const half *, int, const char *, int);
void fill(int, int, float *, int);
void fill(int, int, half *, int);
void passert(const char *, int, const char *);
void run(const char *);

void dgemm(bool, bool, int, int, int, float, const half *, int, const half *, int, float, half *,
           int);
void dgemm(bool, bool, int, int, int, float, const half *, int, const half *, int, float, float *,
           int);
void dgemm(bool, bool, int, int, int, float, const float *, int, const float *, int, float, float *,
           int);

inline int toint(half f) {
    union {
        half f;
        unsigned short i;
    } u = {f};
    return u.i;
}

inline int toint(float f) {
    union {
        float f;
        unsigned i;
    } u = {f};
    return u.i;
}

inline int IsNan(half f) {
    return (toint(f) & 0x7fff) > 0x7c00;
}

inline int IsNan(float f) {
    return (toint(f) & 0x7fffffff) > 0x7f800000;
}

inline int IsNegativeZero(half f) {
    return toint(f) == 0x8000;
}

inline int IsNegativeZero(float f) {
    return toint(f) == -2147483647 - 1;
}

#define ASSERT(x) \
    do { \
        if (!(x)) { \
            passert(__FILE__, __LINE__, #x); \
            __builtin_trap(); \
        } \
    } while (0)

#define CHECK(tol, m, n, A, lda, B, ldb) check(tol, m, n, A, lda, B, ldb, __FILE__, __LINE__)

#define BENCH(x) \
    do { \
        x; \
        __asm__ volatile("" ::: "memory"); \
        long long start = micros(); \
        for (int i = 0; i < ITERATIONS; ++i) { \
            __asm__ volatile("" ::: "memory"); \
            x; \
            __asm__ volatile("" ::: "memory"); \
        } \
        bench(start, #x); \
    } while (0)

#define RUN(x) \
    run(#x); \
    x
