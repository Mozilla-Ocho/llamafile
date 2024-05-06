// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

namespace flt {

inline unsigned toint(float f) {
    union {
        float f;
        unsigned i;
    } u = {f};
    return u.i;
}

inline float tofloat(unsigned i) {
    union {
        unsigned i;
        float f;
    } u = {i};
    return u.f;
}

inline bool isnan(float f) {
    return (toint(f) & 0x7fffffff) > 0x7f800000;
}

inline bool isinf(float f) {
    return (toint(f) & 0x7fffffff) == 0x7f800000;
}

inline unsigned sign(float f) {
    return toint(f) & 0x80000000u;
}

inline bool isnormal(float f) {
    int expo = toint(f) & 0x7f800000;
    return expo && expo != 0x7f800000;
}

inline bool isdenormal(float f) {
    return f && !isnormal(f) && !isnan(f) && !isinf(f);
}

} // namespace flt
