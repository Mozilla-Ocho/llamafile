// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

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

inline bool isnan(half f) {
    return (toint(f) & 0x7fff) > 0x7c00;
}

inline bool isinf(half f) {
    return (toint(f) & 0x7fff) == 0x7c00;
}

inline int sign(half f) {
    return toint(f) & 0x8000;
}

inline bool isnormal(half f) {
    int expo = toint(f) & 0x7c00;
    return expo && expo != 0x7c00;
}

inline bool isdenormal(half f) {
    return f && !isnormal(f) && !isnan(f) && !isinf(f);
}

} // namespace flt
