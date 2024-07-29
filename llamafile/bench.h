// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#include <stdio.h>

#include "micros.h"

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
        printf("%12lld us %s\n", (micros() - start + ITERATIONS - 1) / ITERATIONS, #x); \
    } while (0)
