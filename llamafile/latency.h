// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once
#include <cosmo.h>
#include <time.h>

#define LATENCY1() \
    do { \
    struct timespec started = timespec_real()

#define LATENCY2(NAME) \
    struct timespec ended = timespec_real(); \
    struct timespec elapsed = timespec_sub(ended, started); \
    long micros = timespec_tomicros(elapsed); \
    kprintf("%10ld us %s\n", micros, NAME); \
    } \
    while (0)

#define LATENCV(x) \
    LATENCY1(); \
    (x); \
    LATENCY2(#x)

#define LATENCY(x) \
    ({ \
        typeof(x) res; \
        LATENCY1(); \
        res = (x); \
        LATENCY2(#x); \
        res; \
    })
