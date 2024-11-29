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
mbedtls_p256_isz( uint64_t p[4] )
{
    return( !p[0] & !p[1] & !p[2] & !p[3] );
}

static bool
mbedtls_p256_gte( uint64_t p[5] )
{
    return( (((int64_t)p[4] > 0) |
             ((!p[4]) &
              ((p[3] > 0xffffffff00000001) |
               ((p[3] == 0xffffffff00000001) &
                ((p[2] > 0x0000000000000000) |
                 ((p[2] == 0x0000000000000000) &
                  ((p[1] > 0x00000000ffffffff) |
                   ((p[1] == 0x00000000ffffffff) &
                    ((p[0] > 0xffffffffffffffff) |
                     (p[0] == 0xffffffffffffffff)))))))))) );
}

static int
mbedtls_p256_cmp( const uint64_t a[5],
                  const uint64_t b[5] )
{
    int i, x, y, done = 0;
    // return -1 if a[4] < b[4]
    x = -((int64_t)a[4] < (int64_t)b[4]);
    done = x;
    // return +1 if a[4] > b[4]
    y = (int64_t)a[4] > (int64_t)b[4];
    x = Select(x, y, done);
    done |= -y;
    for (i = 4; i--;) {
        y = -(a[i] < b[i]);
        x = Select(x, y, done);
        done |= y;
        y = a[i] > b[i];
        x = Select(x, y, done);
        done |= -y;
    }
    return x;
}

static void
mbedtls_p256_red( uint64_t p[5] )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    asm("subq\t%1,%0\n\t"
        "sbbq\t%2,8+%0\n\t"
        "sbbq\t%3,16+%0\n\t"
        "sbbq\t%4,24+%0\n\t"
        "sbbq\t$0,32+%0"
        : "+o"(*p)
        : "i"(0xffffffffffffffffl), "r"(0x00000000ffffffffl),
          "i"(0x0000000000000000l), "r"(0xffffffff00000001l)
        : "memory", "cc");
#else
    uint64_t c;
    SBB( p[0], p[0], 0xffffffffffffffff, 0, c );
    SBB( p[1], p[1], 0x00000000ffffffff, c, c );
    SBB( p[2], p[2], 0x0000000000000000, c, c );
    SBB( p[3], p[3], 0xffffffff00000001, c, c );
    SBB( p[4], p[4], 0,                  c, c );
#endif
}

static void
mbedtls_p256_gro( uint64_t p[5] )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    asm("addq\t%1,%0\n\t"
        "adcq\t%2,8+%0\n\t"
        "adcq\t%3,16+%0\n\t"
        "adcq\t%4,24+%0\n\t"
        "adcq\t$0,32+%0"
        : "+o"(*p)
        : "i"(0xffffffffffffffffl), "r"(0x00000000ffffffffl),
          "i"(0x0000000000000000l), "r"(0xffffffff00000001l)
        : "memory", "cc");
#else
    uint64_t c;
    ADC( p[0], p[0], 0xffffffffffffffff, 0, c );
    ADC( p[1], p[1], 0x00000000ffffffff, c, c );
    ADC( p[2], p[2], 0x0000000000000000, c, c );
    ADC( p[3], p[3], 0xffffffff00000001, c, c );
    ADC( p[4], p[4], 0,                  c, c );
#endif
}

static void
mbedtls_p256_rum( uint64_t p[5] )
{
    while( mbedtls_p256_gte( p ) )
        mbedtls_p256_red( p );
}

static void
mbedtls_p256_mod(uint64_t X[8])
{
    secp256r1(X);
    if ((int64_t)X[4] < 0) {
        do {
            mbedtls_p256_gro(X);
        } while ((int64_t)X[4] < 0);
    } else {
        while (mbedtls_p256_gte(X)) {
            mbedtls_p256_red(X);
        }
    }
}

static void
mbedtls_p256_sar( uint64_t p[5] )
{
    p[0] = p[0] >> 1 | p[1] << 63;
    p[1] = p[1] >> 1 | p[2] << 63;
    p[2] = p[2] >> 1 | p[3] << 63;
    p[3] = p[3] >> 1 | p[4] << 63;
    p[4] = (int64_t)p[4] >> 1;
}

static void
mbedtls_p256_shl( uint64_t p[5] )
{
    p[4] =             p[3] >> 63;
    p[3] = p[3] << 1 | p[2] >> 63;
    p[2] = p[2] << 1 | p[1] >> 63;
    p[1] = p[1] << 1 | p[0] >> 63;
    p[0] = p[0] << 1;
    mbedtls_p256_rum( p );
}

static inline void
mbedtls_p256_mul( uint64_t X[8],
                  const uint64_t A[4], size_t n,
                  const uint64_t B[4], size_t m )
{
    Mul4x4( X, A, B );
    mbedtls_p256_mod( X );
}

static void
mbedtls_p256_plu( uint64_t A[5],
                  const uint64_t B[5] )
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
        "adc\t%%rax,32+%0"
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
#endif
}

static void
mbedtls_p256_slu( uint64_t A[5],
                  const uint64_t B[5] )
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
        "sbb\t%%rax,32+%0"
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
#endif
}

static void
mbedtls_p256_add( uint64_t X[5],
                  const uint64_t A[4],
                  const uint64_t B[4] )
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
        "adc\t$0,%%rcx\n\t"
        "mov\t%%rcx,32+%0"
        : "+o"(*X)
        : "o"(*A), "o"(*B)
        : "rax", "rcx", "memory", "cc");
#else
    uint64_t c;
    ADC( X[0], A[0], B[0], 0, c    );
    ADC( X[1], A[1], B[1], c, c    );
    ADC( X[2], A[2], B[2], c, c    );
    ADC( X[3], A[3], B[3], c, X[4] );
#endif
    mbedtls_p256_rum( X );
    MBEDTLS_ASSERT( 0 == X[4] );
}

static void
mbedtls_p256_sub( uint64_t X[5],
                  const uint64_t A[4],
                  const uint64_t B[4] )
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
        "sbb\t$0,%%rcx\n\t"
        "mov\t%%rcx,32+%0"
        : "+o"(*X)
        : "o"(*A), "o"(*B)
        : "rax", "rcx", "memory", "cc");
#else
    uint64_t c;
    SBB( X[0], A[0], B[0], 0, c );
    SBB( X[1], A[1], B[1], c, c );
    SBB( X[2], A[2], B[2], c, c );
    SBB( X[3], A[3], B[3], c, c );
    X[4] = -c;
#endif
    while( (int64_t)X[4] < 0 )
        mbedtls_p256_gro( X );
    MBEDTLS_ASSERT( 0 == X[4] );
}

static void
mbedtls_p256_hub( uint64_t A[5],
                  const uint64_t B[4] )
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
        "sbb\t$0,%%rcx\n\t"
        "mov\t%%rcx,32+%0"
        : "+o"(*A)
        : "o"(*B)
        : "rax", "rcx", "memory", "cc");
    while( (int64_t)A[4] < 0 )
        mbedtls_p256_gro( A );
    MBEDTLS_ASSERT( 0 == A[4] );
#else
    mbedtls_p256_sub( A, A, B );
#endif
}

static inline void
mbedtls_p256_cop( uint64_t X[4],
                  const uint64_t Y[4] )
{
    memcpy( X, Y, 4 * 8 );
}

static int
mbedtls_p256_dim( mbedtls_ecp_point *R )
{
    int ret;
    if( R->X.n < 4 && ( ret = mbedtls_mpi_grow( &R->X, 4 ) ) ) return ret;
    if( R->Y.n < 4 && ( ret = mbedtls_mpi_grow( &R->Y, 4 ) ) ) return ret;
    if( R->Z.n < 4 && ( ret = mbedtls_mpi_grow( &R->Z, 4 ) ) ) return ret;
    return 0;
}

int mbedtls_p256_double_jac( const mbedtls_ecp_group *G,
                             const mbedtls_ecp_point *P,
                             mbedtls_ecp_point *R )
{
    int ret;
    struct {
        uint64_t X[4], Y[4], Z[4];
        uint64_t M[8], S[8], T[8], U[8];
        size_t Xn, Yn, Zn;
    } s;
    MBEDTLS_ASSERT( G->A.p == 0 );
    MBEDTLS_ASSERT( P->X.s == 1 );
    MBEDTLS_ASSERT( P->Y.s == 1 );
    MBEDTLS_ASSERT( P->Z.s == 1 );
    MBEDTLS_ASSERT( G->P.p[0] == 0xffffffffffffffff );
    MBEDTLS_ASSERT( G->P.p[1] == 0x00000000ffffffff );
    MBEDTLS_ASSERT( G->P.p[2] == 0x0000000000000000 );
    MBEDTLS_ASSERT( G->P.p[3] == 0xffffffff00000001 );
    if ( ( ret = mbedtls_p256_dim( R ) ) ) return ret;
    mbedtls_platform_zeroize(&s, sizeof(s));
    s.Xn = mbedtls_mpi_limbs( &P->X );
    s.Yn = mbedtls_mpi_limbs( &P->Y );
    s.Zn = mbedtls_mpi_limbs( &P->Z );
    MBEDTLS_ASSERT( s.Xn <= 4 );
    MBEDTLS_ASSERT( s.Yn <= 4 );
    MBEDTLS_ASSERT( s.Zn <= 4 );
    memcpy( s.X, P->X.p, s.Xn * 8 );
    memcpy( s.Y, P->Y.p, s.Yn * 8 );
    memcpy( s.Z, P->Z.p, s.Zn * 8 );
    mbedtls_p256_mul( s.S, s.Z, s.Zn, s.Z, s.Zn );
    mbedtls_p256_add( s.T, s.X, s.S );
    mbedtls_p256_sub( s.U, s.X, s.S );
    mbedtls_p256_mul( s.S, s.T, 4, s.U, 4 );
    mbedtls_mpi_mul_hlp1( 4, s.S, s.M, 3 );
    mbedtls_p256_rum( s.M );
    mbedtls_p256_mul( s.T, s.Y, s.Yn, s.Y, s.Yn );
    mbedtls_p256_shl( s.T );
    mbedtls_p256_mul( s.S, s.X, s.Xn, s.T, 4 );
    mbedtls_p256_shl( s.S );
    mbedtls_p256_mul( s.U, s.T, 4, s.T, 4 );
    mbedtls_p256_shl( s.U );
    mbedtls_p256_mul( s.T, s.M, 4, s.M, 4 );
    mbedtls_p256_hub( s.T, s.S );
    mbedtls_p256_hub( s.T, s.S );
    mbedtls_p256_hub( s.S, s.T );
    mbedtls_p256_mul( s.S, s.S, 4, s.M, 4 );
    mbedtls_p256_hub( s.S, s.U );
    mbedtls_p256_mul( s.U, s.Y, s.Yn, s.Z, s.Zn );
    mbedtls_p256_shl( s.U );
    mbedtls_p256_cop( R->X.p, s.T );
    mbedtls_p256_cop( R->Y.p, s.S );
    mbedtls_p256_cop( R->Z.p, s.U );
    mbedtls_platform_zeroize( &s, sizeof(s) );
    return 0;
}

int mbedtls_p256_add_mixed( const mbedtls_ecp_group *G,
                            const mbedtls_ecp_point *P,
                            const mbedtls_ecp_point *Q,
                            mbedtls_ecp_point *R )
{
    int ret;
    struct {
        uint64_t X[8], Y[8], Z[8];
        uint64_t T1[8], T2[8], T3[8], T4[8];
        size_t Xn, Yn, Zn, QXn, QYn;
    } s;
    MBEDTLS_ASSERT( P->X.s == 1 );
    MBEDTLS_ASSERT( P->Y.s == 1 );
    MBEDTLS_ASSERT( P->Z.s == 1 );
    MBEDTLS_ASSERT( Q->X.s == 1 );
    MBEDTLS_ASSERT( Q->Y.s == 1 );
    if ( ( ret = mbedtls_p256_dim( R ) ) ) return ret;
    mbedtls_platform_zeroize(&s, sizeof(s));
    s.Xn  = mbedtls_mpi_limbs( &P->X );
    s.Yn  = mbedtls_mpi_limbs( &P->Y );
    s.Zn  = mbedtls_mpi_limbs( &P->Z );
    s.QXn = mbedtls_mpi_limbs( &Q->X );
    s.QYn = mbedtls_mpi_limbs( &Q->Y );
    MBEDTLS_ASSERT( s.Xn  <= 4 );
    MBEDTLS_ASSERT( s.Yn  <= 4 );
    MBEDTLS_ASSERT( s.Zn  <= 4 );
    MBEDTLS_ASSERT( s.QXn <= 4 );
    MBEDTLS_ASSERT( s.QYn <= 4 );
    memcpy( s.X, P->X.p, s.Xn * 8 );
    memcpy( s.Y, P->Y.p, s.Yn * 8 );
    memcpy( s.Z, P->Z.p, s.Zn * 8 );
    mbedtls_p256_mul( s.T1,  s.Z,  s.Zn, s.Z,    s.Zn  );
    mbedtls_p256_mul( s.T2,  s.T1, 4,    s.Z,    s.Zn  );
    mbedtls_p256_mul( s.T1,  s.T1, 4,    Q->X.p, s.QXn );
    mbedtls_p256_mul( s.T2,  s.T2, 4,    Q->Y.p, s.QYn );
    mbedtls_p256_hub(        s.T1,       s.X           );
    mbedtls_p256_hub(        s.T2,       s.Y           );
    if( mbedtls_p256_isz( s.T1 ) )
    {
        if( mbedtls_p256_isz( s.T2 ) )
            return mbedtls_p256_double_jac( G, P, R );
        else
            return mbedtls_ecp_set_zero( R );
    }
    mbedtls_p256_mul( s.Z,    s.Z,   s.Zn, s.T1, 4    );
    mbedtls_p256_mul( s.T3,   s.T1,  4,    s.T1, 4    );
    mbedtls_p256_mul( s.T4,   s.T3,  4,    s.T1, 4    );
    mbedtls_p256_mul( s.T3,   s.T3,  4,    s.X,  s.Xn );
    mbedtls_p256_cop( s.T1,   s.T3                    );
    mbedtls_p256_shl( s.T1                            );
    mbedtls_p256_mul( s.X,    s.T2,  4,    s.T2, 4    );
    mbedtls_p256_hub(         s.X,         s.T1       );
    mbedtls_p256_hub(         s.X,         s.T4       );
    mbedtls_p256_hub(         s.T3,        s.X        );
    mbedtls_p256_mul( s.T3,   s.T3,  4,    s.T2, 4    );
    mbedtls_p256_mul( s.T4,   s.T4,  4,    s.Y,  s.Yn );
    mbedtls_p256_sub( s.Y,    s.T3,        s.T4       );
    mbedtls_p256_cop( R->X.p, s.X                     );
    mbedtls_p256_cop( R->Y.p, s.Y                     );
    mbedtls_p256_cop( R->Z.p, s.Z                     );
    mbedtls_platform_zeroize(&s, sizeof(s));
    return 0;
}

static void
mbedtls_p256_inv( uint64_t X[4],
                  const uint64_t A[4],
                  const uint64_t N[4] )
{
    uint64_t TA[5], TU[5], TV[5], UV[4][5];
    mbedtls_platform_zeroize( UV, sizeof( UV ) );
    *(uint64_t *)mempcpy( TA, A, 4*8 ) = 0;
    *(uint64_t *)mempcpy( TU, A, 4*8 ) = 0;
    *(uint64_t *)mempcpy( TV, N, 4*8 ) = 0;
    UV[0][0] = 1;
    UV[3][0] = 1;
    do {
        while( ~TU[0] & 1 ){
            mbedtls_p256_sar( TU );
            if( ( UV[0][0] | UV[1][0] ) & 1 ){
                mbedtls_p256_gro( UV[0] );
                mbedtls_p256_slu( UV[1], TA );
            }
            mbedtls_p256_sar( UV[0] );
            mbedtls_p256_sar( UV[1] );
        }
        while( ~TV[0] & 1 ){
            mbedtls_p256_sar( TV );
            if( ( UV[2][0] | UV[3][0] ) & 1 ){
                mbedtls_p256_gro( UV[2] );
                mbedtls_p256_slu( UV[3], TA );
            }
            mbedtls_p256_sar( UV[2] );
            mbedtls_p256_sar( UV[3] );
        }
        if( mbedtls_p256_cmp( TU, TV ) >= 0 ){
            mbedtls_p256_slu( TU, TV );
            mbedtls_p256_slu( UV[0], UV[2] );
            mbedtls_p256_slu( UV[1], UV[3] );
        } else {
            mbedtls_p256_slu( TV, TU );
            mbedtls_p256_slu( UV[2], UV[0] );
            mbedtls_p256_slu( UV[3], UV[1] );
        }
    } while( TU[0] | TU[1] | TU[2] | TU[3] | TU[4] );
    while( (int64_t)UV[2][4] < 0 )
        mbedtls_p256_gro( UV[2] );
    while( mbedtls_p256_gte( UV[2] ) )
        mbedtls_p256_red( UV[2] );
    mbedtls_p256_cop( X, UV[2] );
}

int mbedtls_p256_normalize_jac_many( const mbedtls_ecp_group *grp,
                                     mbedtls_ecp_point *T[], size_t n )
{
    size_t i;
    uint64_t *c, u[8], ta[8], Zi[8], ZZi[8];
    if( !( c = mbedtls_calloc( n, 8*8 ) ) )
        return( MBEDTLS_ERR_ECP_ALLOC_FAILED );
    memcpy( c, T[0]->Z.p, T[0]->Z.n*8 );
    for( i = 1; i < n; i++ )
        mbedtls_p256_mul( c+i*8, c+(i-1)*8, 4, T[i]->Z.p, 4 );
    mbedtls_p256_inv( u, c+(n-1)*8, grp->P.p );
    for( i = n - 1; ; i-- ){
        if( !i ){
            mbedtls_p256_cop( Zi, u );
        } else {
            mbedtls_p256_mul( Zi, u, 4, c+(i-1)*8, 4 );
            mbedtls_p256_mul( u, u, 4, T[i]->Z.p, 4 );
        }
        mbedtls_p256_mul( ZZi, Zi, 4, Zi, 4 );
        mbedtls_p256_mul( ta, T[i]->X.p, 4, ZZi, 4 );
        mbedtls_p256_cop( T[i]->X.p, ta );
        mbedtls_p256_mul( ta, T[i]->Y.p, 4, ZZi, 4 );
        mbedtls_p256_mul( ta, ta, 4, Zi, 4 );
        mbedtls_p256_cop( T[i]->Y.p, ta );
        mbedtls_mpi_free( &T[i]->Z );
        if( !i ) break;
    }
    mbedtls_platform_zeroize( ta, sizeof( ta ) );
    mbedtls_platform_zeroize( c, n*8*8 );
    mbedtls_free( c );
    return( 0 );
}

int mbedtls_p256_normalize_jac( const mbedtls_ecp_group *grp,
                                mbedtls_ecp_point *pt )
{
    int ret;
    uint64_t t[8], Zi[8], ZZi[8];
    if ((ret = mbedtls_p256_dim(pt))) return ret;
    mbedtls_p256_inv( Zi, pt->Z.p, grp->P.p );
    mbedtls_p256_mul( ZZi, Zi, 4, Zi, 4 );
    mbedtls_p256_mul( t, pt->X.p, 4, ZZi, 4 );
    mbedtls_p256_cop( pt->X.p, t );
    mbedtls_p256_mul( t, pt->Y.p, 4, ZZi, 4 );
    mbedtls_p256_mul( t, t, 4, Zi, 4 );
    mbedtls_p256_cop( pt->Y.p, t );
    mbedtls_mpi_lset( &pt->Z, 1 );
    return( 0 );
}
