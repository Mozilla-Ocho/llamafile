// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

// floating point utilities
//
// these functions let you fiddle with float bits. some of these
// functions e.g. flt::isnan(), won't be optimized away when the
// compiler is placed in -ffast-math mode.
//
// @see IEEE 754-2008

#if defined(__NVCC__) || defined(__CUDA_ARCH__)
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#endif

namespace flt {

inline int toint(float f) {
    union {
        float f;
        unsigned i;
    } u = {f};
    return u.i;
}

inline int toint(half f) {
    union {
        half f;
        unsigned short i;
    } u = {f};
    return u.i;
}

inline int signbit(float f) {
    return toint(f) & (-2147483647 - 1);
}

inline int signbit(half f) {
    return toint(f) & 0x8000;
}

inline int isnan(float f) {
    return (toint(f) & 0x7fffffff) > 0x7f800000;
}

inline int isnan(half f) {
    return (toint(f) & 0x7fff) > 0x7c00;
}

inline int isinf(float f) {
    return (toint(f) & 0x7fffffff) == 0x7f800000;
}

inline int isinf(half f) {
    return (toint(f) & 0x7fff) == 0x7c00;
}

inline int isnormal(float f) {
    int expo = toint(f) & 0x7f800000;
    return expo && expo != 0x7f800000;
}

inline int isnormal(half f) {
    int expo = toint(f) & 0x7c00;
    return expo && expo != 0x7c00;
}

template <typename T> inline int isdenormal(T f) {
    return f && !isnormal(f) && !isnan(f) && !isinf(f);
}

} // namespace flt
