/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;tab-width:4;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright The Mbed TLS Contributors                                          │
│                                                                              │
│ Licensed under the Apache License, Version 2.0 (the "License");              │
│ you may not use this file except in compliance with the License.             │
│ You may obtain a copy of the License at                                      │
│                                                                              │
│     http://www.apache.org/licenses/LICENSE-2.0                               │
│                                                                              │
│ Unless required by applicable law or agreed to in writing, software          │
│ distributed under the License is distributed on an "AS IS" BASIS,            │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
╚─────────────────────────────────────────────────────────────────────────────*/
#include <libc/nexgen32e/x86feature.h>
#include "third_party/mbedtls/bignum_internal.h"
#include "third_party/mbedtls/math.h"

/**
 * Computes 512-bit product of 256-bit and 256-bit numbers.
 *
 * @param C receives 8 quadword result
 * @param A is left hand side which must have 4 quadwords
 * @param B is right hand side which must have 4 quadwords
 * @note words are host endian while array is little endian
 * @mayalias
 */
void (*Mul4x4)(uint64_t C[16], const uint64_t A[8], const uint64_t B[8]);

__attribute__((__constructor__)) static textstartup void Mul4x4Init()
{
    Mul4x4 = X86_HAVE(ADX) && X86_HAVE(BMI2) ? Mul4x4Adx : Mul4x4Pure;
}

void Mul4x4Pure(uint64_t C[16], const uint64_t A[8], const uint64_t B[8])
{
    uint128_t t;
    uint64_t h, c1, c2, c3;
    uint64_t r0, r1, r2, r3;
    c1 = c2 = c3 = 0;
    MADD(A[0], B[0], c1, c2, c3);
    r0 = c1, c1 = 0;
    MADD(A[0], B[1], c2, c3, c1);
    MADD(A[1], B[0], c2, c3, c1);
    r1 = c2, c2 = 0;
    MADD(A[2], B[0], c3, c1, c2);
    MADD(A[1], B[1], c3, c1, c2);
    MADD(A[0], B[2], c3, c1, c2);
    r2 = c3, c3 = 0;
    MADD(A[0], B[3], c1, c2, c3);
    MADD(A[1], B[2], c1, c2, c3);
    MADD(A[2], B[1], c1, c2, c3);
    MADD(A[3], B[0], c1, c2, c3);
    C[0] = r0;
    r3 = c1, c1 = 0;
    MADD(A[3], B[1], c2, c3, c1);
    MADD(A[2], B[2], c2, c3, c1);
    MADD(A[1], B[3], c2, c3, c1);
    C[1] = r1;
    C[4] = c2, c2 = 0;
    MADD(A[2], B[3], c3, c1, c2);
    MADD(A[3], B[2], c3, c1, c2);
    C[2] = r2;
    C[5] = c3, c3 = 0;
    MADD(A[3], B[3], c1, c2, c3);
    C[3] = r3;
    C[6] = c1;
    C[7] = c2;
}
