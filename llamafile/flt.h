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

#ifndef __HIP__
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#endif

namespace flt {

inline int toint(half f) {
    union {
        half f;
        unsigned short i;
    } u = {f};
    return u.i;
}

inline unsigned toint(float f) {
    union {
        float f;
        unsigned i;
    } u = {f};
    return u.i;
}

inline bool isnan(half f) {
    return (toint(f) & 0x7fff) > 0x7c00;
}

inline bool isnan(float f) {
    return (toint(f) & 0x7fffffff) > 0x7f800000;
}

inline bool isinf(half f) {
    return (toint(f) & 0x7fff) == 0x7c00;
}

inline bool isinf(float f) {
    return (toint(f) & 0x7fffffff) == 0x7f800000;
}

inline int signbit(half f) {
    return toint(f) & 0x8000;
}

inline unsigned signbit(float f) {
    return toint(f) & 0x80000000u;
}

inline bool isnormal(half f) {
    int expo = toint(f) & 0x7c00;
    return expo && expo != 0x7c00;
}

inline bool isnormal(float f) {
    int expo = toint(f) & 0x7f800000;
    return expo && expo != 0x7f800000;
}

template <typename T> inline bool isdenormal(T f) {
    return f && !isnormal(f) && !isnan(f) && !isinf(f);
}

} // namespace flt
