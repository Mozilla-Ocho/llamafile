// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#include <ctime>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif

#ifdef _WIN32
static long long GetQueryPerformanceFrequency() {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    return t.QuadPart;
}
static long long GetQueryPerformanceCounter() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}
#endif

static long long micros(void) {
#ifndef _WIN32
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000 + (ts.tv_nsec + 999) / 1000;
#else
    static long long timer_freq = GetQueryPerformanceFrequency();
    static long long timer_start = GetQueryPerformanceCounter();
    return ((GetQueryPerformanceCounter() - timer_start) * 1000000) / timer_freq;
#endif
}
