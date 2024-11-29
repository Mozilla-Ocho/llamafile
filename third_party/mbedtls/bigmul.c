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
#include <libc/log/backtrace.internal.h>
#include <libc/log/check.h>
#include <libc/macros.h>
#include <libc/mem/mem.h>
#include <libc/nexgen32e/x86feature.h>
#include "third_party/mbedtls/bignum.h"
#include "third_party/mbedtls/bignum_internal.h"
#include "third_party/mbedtls/profile.h"

void Mul(uint64_t *c, const uint64_t *A, unsigned n, const uint64_t *B, unsigned m)
{
    if (!m--) return;
    mbedtls_platform_zeroize(c, m * ciL);
    mbedtls_mpi_mul_hlp1(n, A, c + m, B[m]);
    for (; m > 0; m--)
        mbedtls_mpi_mul_hlp(n, A, c + m - 1, B[m - 1]);
}

/**
 * Computes inner loop of multiplication algorithm.
 */
void mbedtls_mpi_mul_hlp1(size_t n, const uint64_t *s, uint64_t *d, uint64_t b)
{
    size_t i;
    uint64_t c;
    uint128_t x;
    i = c = 0;
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    if( X86_HAVE(BMI2) )
    {
        for( ; i + 8 <= n; i += 8 )
        {
            asm volatile("mulx\t(%2),%%rax,%%rbx\n\t"
                         "add\t%0,%%rax\n\t"
                         "mov\t%%rax,(%1)\n\t"
                         "mulx\t8(%2),%%rax,%0\n\t"
                         "adc\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,8(%1)\n\t"
                         "mulx\t16(%2),%%rax,%%rbx\n\t"
                         "adc\t%0,%%rax\n\t"
                         "mov\t%%rax,16(%1)\n\t"
                         "mulx\t24(%2),%%rax,%0\n\t"
                         "adc\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,24(%1)\n\t"
                         "mulx\t32(%2),%%rax,%%rbx\n\t"
                         "adc\t%0,%%rax\n\t"
                         "mov\t%%rax,32(%1)\n\t"
                         "mulx\t40(%2),%%rax,%0\n\t"
                         "adc\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,40(%1)\n\t"
                         "mulx\t48(%2),%%rax,%%rbx\n\t"
                         "adc\t%0,%%rax\n\t"
                         "mov\t%%rax,48(%1)\n\t"
                         "mulx\t56(%2),%%rax,%0\n\t"
                         "adc\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,56(%1)\n\t"
                         "adc\t$0,%0"
                         : "+r"(c)
                         : "r"(d + i), "r"(s + i), "d"(b)
                         : "rax", "rbx", "memory", "cc");
        }
        for( ; i + 4 <= n; i += 4 )
        {
            asm volatile("mulx\t(%2),%%rax,%%rbx\n\t"
                         "add\t%0,%%rax\n\t"
                         "mov\t%%rax,(%1)\n\t"
                         "mulx\t8(%2),%%rax,%0\n\t"
                         "adc\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,8(%1)\n\t"
                         "mulx\t16(%2),%%rax,%%rbx\n\t"
                         "adc\t%0,%%rax\n\t"
                         "mov\t%%rax,16(%1)\n\t"
                         "mulx\t24(%2),%%rax,%0\n\t"
                         "adc\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,24(%1)\n\t"
                         "adc\t$0,%0"
                         : "+r"(c)
                         : "r"(d + i), "r"(s + i), "d"(b)
                         : "rax", "rbx", "memory", "cc");
        }
    }
#endif
    for( ; i < n; ++i )
    {
        x = s[i];
        x *= b;
        x += c;
        c = x >> 64;
        d[i] = x;
    }
    d[i] = c;
}

/**
 * Computes inner loop of multiplication algorithm.
 */
void mbedtls_mpi_mul_hlp(size_t n, const uint64_t *s, uint64_t *d, uint64_t b)
{
    size_t i;
    uint128_t x;
    uint64_t c, l, h, t;
    i = c = 0;
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    if (X86_HAVE(BMI2) && X86_HAVE(ADX))
    {
        for( ; i + 8 <= n; i += 8 )
        {
            asm volatile("xor\t%%r8d,%%r8d\n\t"
                         "mulx\t(%2),%%rax,%%rbx\n\t"
                         "adcx\t(%1),%%rax\n\t"
                         "adox\t%0,%%rax\n\t"
                         "mov\t%%rax,(%1)\n\t"
                         "mulx\t8(%2),%%rax,%0\n\t"
                         "adcx\t8(%1),%%rax\n\t"
                         "adox\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,8(%1)\n\t"
                         "mulx\t16(%2),%%rax,%%rbx\n\t"
                         "adcx\t16(%1),%%rax\n\t"
                         "adox\t%0,%%rax\n\t"
                         "mov\t%%rax,16(%1)\n\t"
                         "mulx\t24(%2),%%rax,%0\n\t"
                         "adcx\t24(%1),%%rax\n\t"
                         "adox\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,24(%1)\n\t"
                         "mulx\t32(%2),%%rax,%%rbx\n\t"
                         "adcx\t32(%1),%%rax\n\t"
                         "adox\t%0,%%rax\n\t"
                         "mov\t%%rax,32(%1)\n\t"
                         "mulx\t40(%2),%%rax,%0\n\t"
                         "adcx\t40(%1),%%rax\n\t"
                         "adox\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,40(%1)\n\t"
                         "mulx\t48(%2),%%rax,%%rbx\n\t"
                         "adcx\t48(%1),%%rax\n\t"
                         "adox\t%0,%%rax\n\t"
                         "mov\t%%rax,48(%1)\n\t"
                         "mulx\t56(%2),%%rax,%0\n\t"
                         "adcx\t56(%1),%%rax\n\t"
                         "adox\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,56(%1)\n\t"
                         "adcx\t%%r8,%0\n\t"
                         "adox\t%%r8,%0"
                         : "+r"(c)
                         : "r"(d + i), "r"(s + i), "d"(b)
                         : "rax", "rbx", "r8", "memory", "cc");
        }
        for( ; i + 4 <= n; i += 4 )
        {
            asm volatile("xor\t%%r8d,%%r8d\n\t"
                         "mulx\t(%2),%%rax,%%rbx\n\t"
                         "adcx\t(%1),%%rax\n\t"
                         "adox\t%0,%%rax\n\t"
                         "mov\t%%rax,(%1)\n\t"
                         "mulx\t8(%2),%%rax,%0\n\t"
                         "adcx\t8(%1),%%rax\n\t"
                         "adox\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,8(%1)\n\t"
                         "mulx\t16(%2),%%rax,%%rbx\n\t"
                         "adcx\t16(%1),%%rax\n\t"
                         "adox\t%0,%%rax\n\t"
                         "mov\t%%rax,16(%1)\n\t"
                         "mulx\t24(%2),%%rax,%0\n\t"
                         "adcx\t24(%1),%%rax\n\t"
                         "adox\t%%rbx,%%rax\n\t"
                         "mov\t%%rax,24(%1)\n\t"
                         "adcx\t%%r8,%0\n\t"
                         "adox\t%%r8,%0"
                         : "+r"(c)
                         : "r"(d + i), "r"(s + i), "d"(b)
                         : "rax", "rbx", "r8", "memory", "cc");
        }
    }
#endif
    for( ; i < n; ++i )
    {
        x = s[i];
        x *= b;
        x += c;
        l = x;
        h = x >> 64;
        t = d[i];
        d[i] = t + l;
        c = (t + l < t) + h;
    }
    do
    {
        d[i] += c;
    } while ((c = d[i++] < c));
}

/**
 * Multiplies big number with unsigned scalar: X = A × b
 *
 * @param X receives result w/ aliasing permitted
 * @param A is left-hand side big number
 * @param B is left-hand side unsigned scalar
 * @return 0 on success or negative on error
 */
int mbedtls_mpi_mul_int(mbedtls_mpi *X, const mbedtls_mpi *A,
                        mbedtls_mpi_uint b)
{
    int r;
    size_t n;
    MPI_VALIDATE_RET(X);
    MPI_VALIDATE_RET(A);
    n = mbedtls_mpi_limbs(A);
    if ((r = mbedtls_mpi_grow(X, n + 1))) return r;
    mbedtls_mpi_mul_hlp1(n, A->p, X->p, b);
    X->s = A->s;
    X->n = n + 1;
    return 0;
}

/**
 * Multiplies big numbers: X = A * B
 *
 * @param X is destination mpi
 * @param A is first factor
 * @param B is second factor
 * @return 0 on success or <0 on error
 */
int mbedtls_mpi_mul_mpi(mbedtls_mpi *X, const mbedtls_mpi *A,
                        const mbedtls_mpi *B)
{
    int i, j, t, ret;
    mbedtls_mpi TA, TB;
    mbedtls_mpi_uint *K;
    const mbedtls_mpi *T;
    MPI_VALIDATE_RET(X);
    MPI_VALIDATE_RET(A);
    MPI_VALIDATE_RET(B);

    i = mbedtls_mpi_limbs(A);
    j = mbedtls_mpi_limbs(B);

    if (!i || !j)
        return mbedtls_mpi_lset(X, 0);

    if( j > i )
        T = A,
        A = B,
        B = T,
        t = i,
        i = j,
        j = t;

    if (!IsTiny() && j == 1) {
        if (X->n < i + 1) {
            if ((ret = mbedtls_mpi_grow(X, i + 1))) return ret;
        } else if (X->n > i + 1) {
            mbedtls_platform_zeroize(X->p + i + 1, (X->n - (i + 1)) * ciL);
        }
        mbedtls_mpi_mul_hlp1(i, A->p, X->p, B->p[0]);
        X->s = A->s * B->s;
        return 0;
    }

#ifdef __x86_64__
    if (!IsTiny() && i == j) {
        if (X->n < i * 2) {
            if ((ret = mbedtls_mpi_grow(X, i * 2))) return ret;
        } else if (X->n > i * 2) {
            mbedtls_platform_zeroize(X->p + i * 2, (X->n - (i * 2)) * ciL);
        }
        if (i == 4) {
            Mul4x4(X->p, A->p, B->p);
            X->s = A->s * B->s;
            return 0;
        } else if (i == 6 && X86_HAVE(BMI2) && X86_HAVE(ADX)) {
            Mul6x6Adx(X->p, A->p, B->p);
            X->s = A->s * B->s;
            return 0;
        } else if (i == 8 && X86_HAVE(BMI2) && X86_HAVE(ADX)) {
            Mul8x8Adx(X->p, A->p, B->p);
            X->s = A->s * B->s;
            return 0;
        }
    }
#endif /* __x86_64__ */

    mbedtls_mpi_init( &TA );
    mbedtls_mpi_init( &TB );

    if (X->n < i + j)
        MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, i + j ) );
    else if (X->n > i + j)
        mbedtls_platform_zeroize( X->p + i + j, (X->n - (i + j)) * ciL );
    if (X == A) {
        MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &TA, A ) );
        A = &TA;
    }
    if (X == B) {
        MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &TB, B ) );
        B = &TB;
    }
    if (!IsTiny() &&
          i >= 16 && i == j && !(i & (i - 1)) &&
          (K = malloc(i * 4 * sizeof(*K)))) {
        Karatsuba(X->p, A->p, B->p, i, K);
        free(K);
    } else {
        Mul(X->p, A->p, i, B->p, j);
    }

    X->s = A->s * B->s;
    ret = 0;
cleanup:
    mbedtls_mpi_free(&TB);
    mbedtls_mpi_free(&TA);
    return ret;
}
