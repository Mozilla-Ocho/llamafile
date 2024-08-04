// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2023 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

/* Helper routine for calculating exp(x) - 1.
   Copied from expm1f_1u6.c, with several simplifications:
   - No special-case handling for tiny or special values, instead return early
   from the main routine.
   - No special handling for large values:
   - No early return for infinity.
   - Simpler combination of p and t in final stage of algorithm.
   - |i| < 27, so can calculate t by simpler shift-and-add, instead of ldexpf.
   From Optimized Routines by Arm Limited.  */
static inline float Expm1f(float x) {
    /* Reduce argument: f in [-ln2/2, ln2/2], i is exact.  */
    float Shift = 0x1.8p23f;
    float j = fmaf(0x1.715476p+0f, x, Shift) - Shift;
    int i = j;
    float f = fmaf(j, -0x1.62e4p-1f, x);
    f = fmaf(j, -0x1.7f7d1cp-20f, f);

    /* Approximate expm1(f) with polynomial P, expm1(f) ~= f + f^2 * P(f).
       Uses Estrin scheme, where the main expm1f routine uses Horner.  */
    float f2 = f * f;
    float p_01 = fmaf(f, 0x1.5554aep-3, 0x1.fffffep-2);
    float p_23 = fmaf(f, 0x1.12287cp-7, 0x1.555736p-5);
    float p = fmaf(f2, p_23, p_01);
    p = fmaf(f2 * f2, 0x1.6b55a2p-10, p);
    p = fmaf(f2, p, f);

    /* t = 2^i.  */
    union {
        unsigned i;
        float f;
    } u = {(i + 127) << 23};
    float t = u.f;

    /* expm1(x) ~= p * t + (t - 1).  */
    return fmaf(p, t, t - 1);
}

/* Single-precision tanh(x) approximation.
   The maximum error is 2.58 ULP.
   Designed by Arm Limited.  */
static inline float Tanhf(float x) {
    union {
        float f;
        unsigned i;
    } u = {x};
    unsigned iax = u.i & 0x7fffffff;
    unsigned sign = u.i & ~0x7fffffff;

    /* Above 0x1.205966p+3 tanhf rounds to 1 (or -1 for negative).  */
    if (iax > 0x41102cb3) {
        if (iax > 0x7f800000)
            return (x - x) / (x - x);
        u.i = 0x3f800000 | sign;
        return u.f;
    }
    if (iax < 0x34000000)
        return x;

    /* tanh(x) = (e^2x - 1) / (e^2x + 1).  */
    float q = Expm1f(2 * x);
    return q / (q + 2);
}
