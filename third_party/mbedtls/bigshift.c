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
#include <libc/macros.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/bignum.h"
#include "third_party/mbedtls/bignum_internal.h"
#include "third_party/mbedtls/platform.h"

typedef long long xmm_t __attribute__((__vector_size__(16), __aligned__(1)));

static inline void shrd(mbedtls_mpi_uint *p, size_t n, size_t j, size_t m,
                        char k)
{
    mbedtls_mpi_uint x, y, *e, *f;
    f = p + m;
    if (n)
    {
        y = 0;
        x = p[j];
        e = p + n;
        for (; ++p < e; x = y)
        {
            y = p[j];
            p[-1] = x >> k | y << (biL - k);
        }
        p[-1] = x >> k;
    }
    while (p < f)
        *p++ = 0;
}

static inline void shld(mbedtls_mpi_uint *p, size_t n, size_t m, char k)
{
    size_t i;
    mbedtls_mpi_uint x, y;
    MBEDTLS_ASSERT(n > m);
    i = n - 1;
    y = p[i - m];
    for (; i - m > 0; --i, y = x)
    {
        x = p[i - m - 1];
        p[i] = y << k | x >> (64 - k);
    }
    p[i] = y << k;
    while (i)
    {
        p[--i] = 0;
    }
}

/**
 * Performs left shift on big number: X <<= k
 */
int mbedtls_mpi_shift_l(mbedtls_mpi *X, size_t k)
{
    int r;
    size_t b, n, m, l;
    MPI_VALIDATE_RET(X);
    l = mbedtls_mpi_bitlen(X);
    b = l + k;
    n = BITS_TO_LIMBS(b);
    m = k / biL;
    k = k % biL;
    if (n > X->n && (r = mbedtls_mpi_grow(X, n))) 
        return r;
    if (k)
    {
        shld(X->p, X->n, m, k);
    }
    else if (m)
    {
        memmove(X->p + m, X->p, (X->n - m) * ciL);
        explicit_bzero(X->p, m * ciL);
    }
    return 0;
}

void ShiftRightPure(mbedtls_mpi_uint *p, size_t n, unsigned char k) {
    shrd(p, n, 0, n, k);
}

/**
 * Performs right arithmetic shift on big number: X >>= k
 */
int mbedtls_mpi_shift_r(mbedtls_mpi *X, size_t k)
{
    size_t n;
    MPI_VALIDATE_RET(X);
    k = MIN(k, X->n * biL);
    n = k / biL;
    k = k % biL;
    if (k)
    {
        if (!n)
            ShiftRight(X->p, X->n, k);
        else
            shrd(X->p, X->n - n, n, X->n, k);
    }
    else if (n)
    {
        memmove(X->p, X->p + n, (X->n - n) * ciL);
        explicit_bzero(X->p + X->n - n, n * ciL);
    }
    return 0;
}
