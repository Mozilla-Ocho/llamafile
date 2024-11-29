/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;tab-width:4;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2021 Justine Alexandra Roberts Tunney                              │
│                                                                              │
│ Permission to use, copy, modify, and/or distribute this software for         │
│ any purpose with or without fee is hereby granted, provided that the         │
│ above copyright notice and this permission notice appear in all copies.      │
│                                                                              │
│ THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL                │
│ WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED                │
│ WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE             │
│ AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL         │
│ DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR        │
│ PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER               │
│ TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR             │
│ PERFORMANCE OF THIS SOFTWARE.                                                │
╚─────────────────────────────────────────────────────────────────────────────*/
#include <libc/nexgen32e/x86feature.h>
#include <libc/runtime/runtime.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/bignum_internal.h"
#include "third_party/mbedtls/ecp.h"
#include "third_party/mbedtls/ecp_internal.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/math.h"
#include "third_party/mbedtls/profile.h"
#include "third_party/mbedtls/select.h"

// todo(jart): investigate gnu assembler warning
#ifndef __STRICT_ANSI__
#define __STRICT_ANSI__
#endif

static bool
mbedtls_p384_isz( uint64_t p[6] )
{
    return( !p[0] & !p[1] & !p[2] & !p[3] & !p[4] & !p[5] );
}

static bool
mbedtls_p384_gte( uint64_t p[7] )
{
    return( (((int64_t)p[6] > 0) |
             ((!p[6]) &
              ((p[5] > 0xffffffffffffffff) |
               ((p[5] == 0xffffffffffffffff) &
                ((p[4] > 0xffffffffffffffff) |
                 ((p[4] == 0xffffffffffffffff) &
                  ((p[3] > 0xffffffffffffffff) |
                   ((p[3] == 0xffffffffffffffff) &
                    ((p[2] > 0xfffffffffffffffe) |
                     ((p[2] == 0xfffffffffffffffe) &
                      ((p[1] > 0xffffffff00000000) |
                       ((p[1] == 0xffffffff00000000) &
                        ((p[0] > 0x00000000ffffffff) |
                         (p[0] == 0x00000000ffffffff)))))))))))))) );
}

static int
mbedtls_p384_cmp( const uint64_t a[7],
                  const uint64_t b[7] )
{
    int i, x, y, done = 0;
    // return -1 if a[6] < b[6]
    x = -((int64_t)a[6] < (int64_t)b[6]);
    done = x;
    // return +1 if a[6] > b[6]
    y = (int64_t)a[6] > (int64_t)b[6];
    x = Select(x, y, done);
    done |= -y;
    for (i = 6; i--;) {
        y = -(a[i] < b[i]);
        x = Select(x, y, done);
        done |= y;
        y = a[i] > b[i];
        x = Select(x, y, done);
        done |= -y;
    }
    return x;
}

static inline void
mbedtls_p384_red( uint64_t p[7] )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    asm("subq\t%1,%0\n\t"
        "sbbq\t%2,8+%0\n\t"
        "sbbq\t%3,16+%0\n\t"
        "sbbq\t%4,24+%0\n\t"
        "sbbq\t%4,32+%0\n\t"
        "sbbq\t%4,40+%0\n\t"
        "sbbq\t$0,48+%0"
        : "+o"(*p)
        : "r"(0x00000000ffffffffl), "r"(0xffffffff00000000),
          "i"(0xfffffffffffffffel), "i"(0xffffffffffffffff)
        : "memory", "cc");
#else
    uint64_t c;
    SBB( p[0], p[0], 0x00000000ffffffff, 0, c );
    SBB( p[1], p[1], 0xffffffff00000000, c, c );
    SBB( p[2], p[2], 0xfffffffffffffffe, c, c );
    SBB( p[3], p[3], 0xffffffffffffffff, c, c );
    SBB( p[4], p[4], 0xffffffffffffffff, c, c );
    SBB( p[5], p[5], 0xffffffffffffffff, c, c );
    SBB( p[6], p[6], 0,                  c, c );
#endif
}

static inline void
mbedtls_p384_gro( uint64_t p[7] )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    asm("addq\t%1,%0\n\t"
        "adcq\t%2,8+%0\n\t"
        "adcq\t%3,16+%0\n\t"
        "adcq\t%4,24+%0\n\t"
        "adcq\t%4,32+%0\n\t"
        "adcq\t%4,40+%0\n\t"
        "adcq\t$0,48+%0"
        : "+o"(*p)
        : "r"(0x00000000ffffffffl), "r"(0xffffffff00000000),
          "i"(0xfffffffffffffffel), "i"(0xffffffffffffffff)
        : "memory", "cc");
#else
    uint64_t c;
    ADC( p[0], p[0], 0x00000000ffffffff, 0, c );
    ADC( p[1], p[1], 0xffffffff00000000, c, c );
    ADC( p[2], p[2], 0xfffffffffffffffe, c, c );
    ADC( p[3], p[3], 0xffffffffffffffff, c, c );
    ADC( p[4], p[4], 0xffffffffffffffff, c, c );
    ADC( p[5], p[5], 0xffffffffffffffff, c, c );
    ADC( p[6], p[6], 0,                  c, c );
#endif
}

static inline void
mbedtls_p384_rum( uint64_t p[7] )
{
    while( mbedtls_p384_gte( p ) )
        mbedtls_p384_red( p );
}

void mbedtls_p384_mod( uint64_t X[12] )
{
    secp384r1(X);
    if( (int64_t)X[6] < 0 ){
        do {
            mbedtls_p384_gro(X);
        } while( (int64_t)X[6] < 0 );
    } else {
        mbedtls_p384_rum(X);
    }
}

static inline void
mbedtls_p384_sar( uint64_t p[7] )
{
    p[0] = p[0] >> 1 | p[1] << 63;
    p[1] = p[1] >> 1 | p[2] << 63;
    p[2] = p[2] >> 1 | p[3] << 63;
    p[3] = p[3] >> 1 | p[4] << 63;
    p[4] = p[4] >> 1 | p[5] << 63;
    p[5] = p[5] >> 1 | p[6] << 63;
    p[6] = (int64_t)p[6] >> 1;
}

static inline void
mbedtls_p384_shl( uint64_t p[7] )
{
    p[6] =             p[5] >> 63;
    p[5] = p[5] << 1 | p[4] >> 63;
    p[4] = p[4] << 1 | p[3] >> 63;
    p[3] = p[3] << 1 | p[2] >> 63;
    p[2] = p[2] << 1 | p[1] >> 63;
    p[1] = p[1] << 1 | p[0] >> 63;
    p[0] = p[0] << 1;
    mbedtls_p384_rum( p );
}

static void
mbedtls_p384_mul( uint64_t X[12],
                  const uint64_t A[6], size_t n,
                  const uint64_t B[6], size_t m )
{
    if( n == 6 && m == 6 && X86_HAVE(ADX) && X86_HAVE(BMI2) )
    {
        Mul6x6Adx( X, A, B );
    }
    else
    {
        void *f = 0;
        if( A == X )
        {
            A = f = memcpy( malloc( 6 * 8 ), A, 6 * 8 );
        }
        else if( B == X )
        {
            B = f = memcpy( malloc( 6 * 8 ), B, 6 * 8 );
        }
        Mul( X, A, n, B, m );
        mbedtls_platform_zeroize( X + n + m, (12 - n - m) * 8 );
        free( f );
    }
    mbedtls_p384_mod( X );
}

static void
mbedtls_p384_plu( uint64_t A[7],
                  const uint64_t B[7] )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    asm("mov\t%1,%%rax\n\t"
        "add\t%%rax,%0\n\t"
        "mov\t8+%1,%%rax\n\t"
        "adc\t%%rax,8+%0\n\t"
        "mov\t16+%1,%%rax\n\t"
        "adc\t%%rax,16+%0\n\t"
        "mov\t24+%1,%%rax\n\t"
        "adc\t%%rax,24+%0\n\t"
        "mov\t32+%1,%%rax\n\t"
        "adc\t%%rax,32+%0\n\t"
        "mov\t40+%1,%%rax\n\t"
        "adc\t%%rax,40+%0\n\t"
        "mov\t48+%1,%%rax\n\t"
        "adc\t%%rax,48+%0"
        : /* no outputs */
        : "o"(*A), "o"(*B)
        : "rax", "memory", "cc");
#else
    uint64_t c;
    ADC( A[0], A[0], B[0], 0, c );
    ADC( A[1], A[1], B[1], c, c );
    ADC( A[2], A[2], B[2], c, c );
    ADC( A[3], A[3], B[3], c, c );
    ADC( A[4], A[4], B[4], c, c );
    ADC( A[5], A[5], B[5], c, c );
    ADC( A[6], A[6], B[6], c, c );
#endif
}

static void
mbedtls_p384_slu( uint64_t A[7],
                  const uint64_t B[7] )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    asm("mov\t%1,%%rax\n\t"
        "sub\t%%rax,%0\n\t"
        "mov\t8+%1,%%rax\n\t"
        "sbb\t%%rax,8+%0\n\t"
        "mov\t16+%1,%%rax\n\t"
        "sbb\t%%rax,16+%0\n\t"
        "mov\t24+%1,%%rax\n\t"
        "sbb\t%%rax,24+%0\n\t"
        "mov\t32+%1,%%rax\n\t"
        "sbb\t%%rax,32+%0\n\t"
        "mov\t40+%1,%%rax\n\t"
        "sbb\t%%rax,40+%0\n\t"
        "mov\t48+%1,%%rax\n\t"
        "sbb\t%%rax,48+%0"
        : /* no outputs */
        : "o"(*A), "o"(*B)
        : "rax", "memory", "cc");
#else
    uint64_t c;
    SBB( A[0], A[0], B[0], 0, c );
    SBB( A[1], A[1], B[1], c, c );
    SBB( A[2], A[2], B[2], c, c );
    SBB( A[3], A[3], B[3], c, c );
    SBB( A[4], A[4], B[4], c, c );
    SBB( A[5], A[5], B[5], c, c );
    SBB( A[6], A[6], B[6], c, c );
#endif
}

static void
mbedtls_p384_add( uint64_t X[7],
                  const uint64_t A[6],
                  const uint64_t B[6] )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    asm("xor\t%%rcx,%%rcx\n\t"
        "mov\t%1,%%rax\n\t"
        "add\t%2,%%rax\n\t"
        "mov\t%%rax,%0\n\t"
        "mov\t8+%1,%%rax\n\t"
        "adc\t8+%2,%%rax\n\t"
        "mov\t%%rax,8+%0\n\t"
        "mov\t16+%1,%%rax\n\t"
        "adc\t16+%2,%%rax\n\t"
        "mov\t%%rax,16+%0\n\t"
        "mov\t24+%1,%%rax\n\t"
        "adc\t24+%2,%%rax\n\t"
        "mov\t%%rax,24+%0\n\t"
        "mov\t32+%1,%%rax\n\t"
        "adc\t32+%2,%%rax\n\t"
        "mov\t%%rax,32+%0\n\t"
        "mov\t40+%1,%%rax\n\t"
        "adc\t40+%2,%%rax\n\t"
        "mov\t%%rax,40+%0\n\t"
        "adc\t$0,%%rcx\n\t"
        "mov\t%%rcx,48+%0"
        : "+o"(*X)
        : "o"(*A), "o"(*B)
        : "rax", "rcx", "memory", "cc");
#else
    uint64_t c;
    ADC( X[0], A[0], B[0], 0, c    );
    ADC( X[1], A[1], B[1], c, c    );
    ADC( X[2], A[2], B[2], c, c    );
    ADC( X[3], A[3], B[3], c, c    );
    ADC( X[4], A[4], B[4], c, c    );
    ADC( X[5], A[5], B[5], c, X[6] );
#endif
    mbedtls_p384_rum( X );
    MBEDTLS_ASSERT(0 == X[6]);
}

static void
mbedtls_p384_sub( uint64_t X[7],
                  const uint64_t A[6],
                  const uint64_t B[6] )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    asm("xor\t%%rcx,%%rcx\n\t"
        "mov\t%1,%%rax\n\t"
        "sub\t%2,%%rax\n\t"
        "mov\t%%rax,%0\n\t"
        "mov\t8+%1,%%rax\n\t"
        "sbb\t8+%2,%%rax\n\t"
        "mov\t%%rax,8+%0\n\t"
        "mov\t16+%1,%%rax\n\t"
        "sbb\t16+%2,%%rax\n\t"
        "mov\t%%rax,16+%0\n\t"
        "mov\t24+%1,%%rax\n\t"
        "sbb\t24+%2,%%rax\n\t"
        "mov\t%%rax,24+%0\n\t"
        "mov\t32+%1,%%rax\n\t"
        "sbb\t32+%2,%%rax\n\t"
        "mov\t%%rax,32+%0\n\t"
        "mov\t40+%1,%%rax\n\t"
        "sbb\t40+%2,%%rax\n\t"
        "mov\t%%rax,40+%0\n\t"
        "sbb\t$0,%%rcx\n\t"
        "mov\t%%rcx,48+%0"
        : "+o"(*X)
        : "o"(*A), "o"(*B)
        : "rax", "rcx", "memory", "cc");
#else
    uint64_t c;
    SBB( X[0], A[0], B[0], 0, c );
    SBB( X[1], A[1], B[1], c, c );
    SBB( X[2], A[2], B[2], c, c );
    SBB( X[3], A[3], B[3], c, c );
    SBB( X[4], A[4], B[4], c, c );
    SBB( X[5], A[5], B[5], c, c );
    X[6] = -c;
#endif
    while( (int64_t)X[6] < 0 )
        mbedtls_p384_gro( X );
    MBEDTLS_ASSERT(0 == X[6]);
}

static void
mbedtls_p384_hub( uint64_t A[7],
                  const uint64_t B[6] )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    asm("xor\t%%rcx,%%rcx\n\t"
        "mov\t%1,%%rax\n\t"
        "sub\t%%rax,%0\n\t"
        "mov\t8+%1,%%rax\n\t"
        "sbb\t%%rax,8+%0\n\t"
        "mov\t16+%1,%%rax\n\t"
        "sbb\t%%rax,16+%0\n\t"
        "mov\t24+%1,%%rax\n\t"
        "sbb\t%%rax,24+%0\n\t"
        "mov\t32+%1,%%rax\n\t"
        "sbb\t%%rax,32+%0\n\t"
        "mov\t40+%1,%%rax\n\t"
        "sbb\t%%rax,40+%0\n\t"
        "sbb\t$0,%%rcx\n\t"
        "mov\t%%rcx,48+%0"
        : "+o"(*A)
        : "o"(*B)
        : "rax", "rcx", "memory", "cc");
    while( (int64_t)A[6] < 0 )
        mbedtls_p384_gro( A );
    MBEDTLS_ASSERT(0 == A[6]);
#else
    mbedtls_p384_sub(A, A, B);
#endif
}

static inline void
mbedtls_p384_cop( uint64_t X[6],
                  const uint64_t Y[6] )
{
    memcpy( X, Y, 6*8 );
}

static int
mbedtls_p384_dim( mbedtls_ecp_point *R )
{
    int ret;
    if( R->X.n < 6 && ( ret = mbedtls_mpi_grow( &R->X, 6 ) ) ) return( ret );
    if( R->Y.n < 6 && ( ret = mbedtls_mpi_grow( &R->Y, 6 ) ) ) return( ret );
    if( R->Z.n < 6 && ( ret = mbedtls_mpi_grow( &R->Z, 6 ) ) ) return( ret );
    return( 0 );
}

int mbedtls_p384_double_jac( const mbedtls_ecp_group *G,
                             const mbedtls_ecp_point *P,
                             mbedtls_ecp_point *R )
{
    int ret;
    uint64_t T[4][12];
    if( ( ret = mbedtls_p384_dim( R ) ) ) return( ret );
    if( ( ret = mbedtls_p384_dim( (void *)P ) ) ) return( ret );
    mbedtls_platform_zeroize( T, sizeof( T ) );
    mbedtls_p384_mul( T[1], P->Z.p, 6, P->Z.p, 6 );
    mbedtls_p384_add( T[2], P->X.p, T[1] );
    mbedtls_p384_sub( T[3], P->X.p, T[1] );
    mbedtls_p384_mul( T[1], T[2], 6, T[3], 6 );
    mbedtls_mpi_mul_hlp1( 6, T[1], T[0], 3 );
    mbedtls_p384_rum( T[0] );
    mbedtls_p384_mul( T[2], P->Y.p, 6, P->Y.p, 6 );
    mbedtls_p384_shl( T[2] );
    mbedtls_p384_mul( T[1], P->X.p, 6, T[2], 6 );
    mbedtls_p384_shl( T[1] );
    mbedtls_p384_mul( T[3], T[2], 6, T[2], 6 );
    mbedtls_p384_shl( T[3] );
    mbedtls_p384_mul( T[2], T[0], 6, T[0], 6 );
    mbedtls_p384_hub( T[2], T[1] );
    mbedtls_p384_hub( T[2], T[1] );
    mbedtls_p384_hub( T[1], T[2] );
    mbedtls_p384_mul( T[1], T[1], 6, T[0], 6 );
    mbedtls_p384_hub( T[1], T[3] );
    mbedtls_p384_mul( T[3], P->Y.p, 6, P->Z.p, 6 );
    mbedtls_p384_shl( T[3] );
    mbedtls_p384_cop( R->X.p, T[2] );
    mbedtls_p384_cop( R->Y.p, T[1] );
    mbedtls_p384_cop( R->Z.p, T[3] );
    return( 0 );
}

int mbedtls_p384_add_mixed( const mbedtls_ecp_group *G,
                            const mbedtls_ecp_point *P,
                            const mbedtls_ecp_point *Q,
                            mbedtls_ecp_point *R )
{
    int ret;
    struct {
        uint64_t X[12], Y[12], Z[12];
        uint64_t T1[12], T2[12], T3[12], T4[12];
        size_t Xn, Yn, Zn, QXn, QYn;
    } s;
    if( ( ret = mbedtls_p384_dim( R ) ) ) return( ret );
    mbedtls_platform_zeroize( &s, sizeof( s ) );
    s.Xn  = mbedtls_mpi_limbs( &P->X );
    s.Yn  = mbedtls_mpi_limbs( &P->Y );
    s.Zn  = mbedtls_mpi_limbs( &P->Z );
    s.QXn = mbedtls_mpi_limbs( &Q->X );
    s.QYn = mbedtls_mpi_limbs( &Q->Y );
    MBEDTLS_ASSERT( s.Xn  <= 6 );
    MBEDTLS_ASSERT( s.Yn  <= 6 );
    MBEDTLS_ASSERT( s.Zn  <= 6 );
    MBEDTLS_ASSERT( s.QXn <= 6 );
    MBEDTLS_ASSERT( s.QYn <= 6 );
    memcpy( s.X, P->X.p, s.Xn * 8 );
    memcpy( s.Y, P->Y.p, s.Yn * 8 );
    memcpy( s.Z, P->Z.p, s.Zn * 8 );
    mbedtls_p384_mul( s.T1,  s.Z,  s.Zn, s.Z,    s.Zn  );
    mbedtls_p384_mul( s.T2,  s.T1, 6,    s.Z,    s.Zn  );
    mbedtls_p384_mul( s.T1,  s.T1, 6,    Q->X.p, s.QXn );
    mbedtls_p384_mul( s.T2,  s.T2, 6,    Q->Y.p, s.QYn );
    mbedtls_p384_hub(        s.T1,       s.X           );
    mbedtls_p384_hub(        s.T2,       s.Y           );
    if( mbedtls_p384_isz( s.T1 ) )
    {
        if( mbedtls_p384_isz( s.T2 ) )
            return( mbedtls_p384_double_jac( G, P, R ) );
        else
            return( mbedtls_ecp_set_zero( R ) );
    }
    mbedtls_p384_mul( s.Z,    s.Z,   s.Zn, s.T1, 6    );
    mbedtls_p384_mul( s.T3,   s.T1,  6,    s.T1, 6    );
    mbedtls_p384_mul( s.T4,   s.T3,  6,    s.T1, 6    );
    mbedtls_p384_mul( s.T3,   s.T3,  6,    s.X,  s.Xn );
    mbedtls_p384_cop( s.T1,   s.T3                    );
    mbedtls_p384_shl( s.T1                            );
    mbedtls_p384_mul( s.X,    s.T2,  6,    s.T2, 6    );
    mbedtls_p384_hub(         s.X,         s.T1       );
    mbedtls_p384_hub(         s.X,         s.T4       );
    mbedtls_p384_hub(         s.T3,        s.X        );
    mbedtls_p384_mul( s.T3,   s.T3,  6,    s.T2, 6    );
    mbedtls_p384_mul( s.T4,   s.T4,  6,    s.Y,  s.Yn );
    mbedtls_p384_sub( s.Y,    s.T3,        s.T4       );
    mbedtls_p384_cop( R->X.p, s.X                     );
    mbedtls_p384_cop( R->Y.p, s.Y                     );
    mbedtls_p384_cop( R->Z.p, s.Z                     );
    mbedtls_platform_zeroize( &s, sizeof( s ) );
    return( 0 );
}

static void
mbedtls_p384_inv( uint64_t X[6],
                  const uint64_t A[6],
                  const uint64_t N[6] )
{
    uint64_t TA[7], TU[7], TV[7], UV[4][7];
    mbedtls_platform_zeroize( UV, sizeof( UV ) );
    *(uint64_t *)mempcpy( TA, A, 6*8 ) = 0;
    *(uint64_t *)mempcpy( TU, A, 6*8 ) = 0;
    *(uint64_t *)mempcpy( TV, N, 6*8 ) = 0;
    UV[0][0] = 1;
    UV[3][0] = 1;
    do {
        while( ~TU[0] & 1 ){
            mbedtls_p384_sar( TU );
            if( ( UV[0][0] | UV[1][0] ) & 1 ){
                mbedtls_p384_gro( UV[0] );
                mbedtls_p384_slu( UV[1], TA );
            }
            mbedtls_p384_sar( UV[0] );
            mbedtls_p384_sar( UV[1] );
        }
        while( ~TV[0] & 1 ){
            mbedtls_p384_sar( TV );
            if( ( UV[2][0] | UV[3][0] ) & 1 ){
                mbedtls_p384_gro( UV[2] );
                mbedtls_p384_slu( UV[3], TA );
            }
            mbedtls_p384_sar( UV[2] );
            mbedtls_p384_sar( UV[3] );
        }
        if( mbedtls_p384_cmp( TU, TV ) >= 0 ){
            mbedtls_p384_slu( TU, TV );
            mbedtls_p384_slu( UV[0], UV[2] );
            mbedtls_p384_slu( UV[1], UV[3] );
        } else {
            mbedtls_p384_slu( TV, TU );
            mbedtls_p384_slu( UV[2], UV[0] );
            mbedtls_p384_slu( UV[3], UV[1] );
        }
    } while( TU[0] | TU[1] | TU[2] | TU[3] | TU[4] | TU[5] | TU[6] );
    while( (int64_t)UV[2][6] < 0 )
        mbedtls_p384_gro( UV[2] );
    while( mbedtls_p384_gte( UV[2] ) )
        mbedtls_p384_red( UV[2] );
    mbedtls_p384_cop( X, UV[2] );
}

int mbedtls_p384_normalize_jac( const mbedtls_ecp_group *grp,
                                mbedtls_ecp_point *pt )
{
    int ret;
    uint64_t t[12], Zi[12], ZZi[12];
    if(( ret = mbedtls_p384_dim(pt)) ) return( ret );
    mbedtls_p384_inv( Zi, pt->Z.p, grp->P.p );
    mbedtls_p384_mul( ZZi, Zi, 6, Zi, 6 );
    mbedtls_p384_mul( t, pt->X.p, 6, ZZi, 6 );
    mbedtls_p384_cop( pt->X.p, t );
    mbedtls_p384_mul( t, pt->Y.p, 6, ZZi, 6 );
    mbedtls_p384_mul( t, t, 6, Zi, 6 );
    mbedtls_p384_cop( pt->Y.p, t );
    mbedtls_mpi_lset( &pt->Z, 1 );
    return( 0 );
}

int mbedtls_p384_normalize_jac_many( const mbedtls_ecp_group *grp,
                                     mbedtls_ecp_point *T[], size_t n )
{
    size_t i;
    uint64_t *c, u[12], ta[12], Zi[12], ZZi[12];
    if( !( c = mbedtls_calloc( n, 12*8 ) ) )
        return( MBEDTLS_ERR_ECP_ALLOC_FAILED );
    memcpy( c, T[0]->Z.p, T[0]->Z.n*8 );
    for( i = 1; i < n; i++ )
        mbedtls_p384_mul( c+i*12, c+(i-1)*12, 6, T[i]->Z.p, 6 );
    mbedtls_p384_inv( u, c+(n-1)*12, grp->P.p );
    for( i = n - 1; ; i-- ){
        if( !i ){
            mbedtls_p384_cop( Zi, u );
        } else {
            mbedtls_p384_mul( Zi, u, 6, c+(i-1)*12, 6 );
            mbedtls_p384_mul( u, u, 6, T[i]->Z.p, 6 );
        }
        mbedtls_p384_mul( ZZi, Zi, 6, Zi, 6 );
        mbedtls_p384_mul( ta, T[i]->X.p, 6, ZZi, 6 );
        memcpy( T[i]->X.p, ta, 6 * 8 );
        mbedtls_p384_mul( ta, T[i]->Y.p, 6, ZZi, 6 );
        mbedtls_p384_mul( ta, ta, 6, Zi, 6 );
        memcpy( T[i]->Y.p, ta, 6 * 8 );
        mbedtls_mpi_free( &T[i]->Z );
        if( !i ) break;
    }
    mbedtls_platform_zeroize( ta, sizeof( ta ) );
    mbedtls_platform_zeroize( c, n*12*8 );
    mbedtls_free( c );
    return( 0 );
}
