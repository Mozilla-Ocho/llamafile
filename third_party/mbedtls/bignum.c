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
#include "third_party/mbedtls/bignum.h"
#include <libc/serialize.h>
#include <libc/intrin/bsf.h>
#include <libc/intrin/bswap.h>
#include <libc/macros.h>
#include <libc/nexgen32e/nexgen32e.h>
#include <libc/nexgen32e/x86feature.h>
#include <libc/runtime/runtime.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/bignum_internal.h"
#include "third_party/mbedtls/chk.h"
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/fastdiv.h"
#include "third_party/mbedtls/math.h"
#include "third_party/mbedtls/platform.h"
#include "third_party/mbedtls/profile.h"
#include "third_party/mbedtls/select.h"
__static_yoink("mbedtls_notice");

/**
 * @fileoverview Big Numbers.
 *
 * The following sources were referenced in the design of this
 * Multi-precision Integer library:
 *
 * [1] Handbook of Applied Cryptography - 1997
 *     Menezes, van Oorschot and Vanstone
 *
 * [2] Multi-Precision Math
 *     Tom St Denis
 *     https://github.com/libtom/libtommath/blob/develop/tommath.pdf
 *
 * [3] GNU Multi-Precision Arithmetic Library
 *     https://gmplib.org/manual/index.html
 */

#if defined(MBEDTLS_BIGNUM_C)

/* Implementation that should never be optimized out by the compiler */
static inline void mbedtls_mpi_zeroize( mbedtls_mpi_uint *v, size_t n )
{
    mbedtls_platform_zeroize( v, ciL * n );
}

/**
 * \brief          This function frees the components of an MPI context.
 *
 * \param X        The MPI context to be cleared. This may be \c NULL,
 *                 in which case this function is a no-op. If it is
 *                 not \c NULL, it must point to an initialized MPI.
 */
void mbedtls_mpi_free( mbedtls_mpi *X )
{
    if( !X )
        return;
    if( X->p )
    {
        mbedtls_mpi_zeroize( X->p, X->n );
        mbedtls_free( X->p );
    }
    X->s = 1;
    X->n = 0;
    X->p = NULL;
}

/**
 * \brief          Enlarge an MPI to the specified number of limbs.
 *
 * \note           This function does nothing if the MPI is
 *                 already large enough.
 *
 * \param X        The MPI to grow. It must be initialized.
 * \param nblimbs  The target number of limbs.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         Another negative error code on other kinds of failure.
 */
int mbedtls_mpi_grow(mbedtls_mpi *X, size_t nblimbs)
{
    mbedtls_mpi_uint *p;
    MPI_VALIDATE_RET(X);
    if (nblimbs > MBEDTLS_MPI_MAX_LIMBS)
        return MBEDTLS_ERR_MPI_ALLOC_FAILED;
    if (nblimbs > X->n)
    {
        if (X->p && (p = realloc_in_place(X->p, nblimbs * ciL)))
        {
            mbedtls_mpi_zeroize(p + X->n, nblimbs - X->n);
        }
        else
        {
            if (!(p = malloc(nblimbs * ciL)))
                return MBEDTLS_ERR_MPI_ALLOC_FAILED;
            if (X->p)
            {
                memcpy(p, X->p, X->n * ciL);
                mbedtls_mpi_zeroize(p + X->n, nblimbs - X->n);
                mbedtls_mpi_zeroize(X->p, X->n);
                free(X->p);
            }
            else
            {
                mbedtls_mpi_zeroize(p, nblimbs);
            }
        }
        X->n = nblimbs;
        X->p = p;
    }
    return 0;
}

/**
 * \brief          This function resizes an MPI to a number of limbs.
 *
 * \param X        The MPI to resize. This must point to an initialized MPI.
 * \param n        The minimum number of limbs to keep.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed
 *                 which can only happen when resizing up
 * \return         Another negative error code on other kinds of failure.
 */
int mbedtls_mpi_resize(mbedtls_mpi *X, size_t n)
{
    mbedtls_mpi_uint *p;
    MPI_VALIDATE_RET(X);
    if (X->n == n)
        return 0;
    if (X->n <= n)
        return mbedtls_mpi_grow(X, n);
    if (n > MBEDTLS_MPI_MAX_LIMBS)
        return MBEDTLS_ERR_MPI_ALLOC_FAILED;
    mbedtls_mpi_zeroize(X->p + n, X->n - n);
    if (!realloc_in_place(X->p, n * ciL))
    {
        if (!(p = malloc(n * ciL)))
            return MBEDTLS_ERR_MPI_ALLOC_FAILED;
        memcpy(p, X->p, n * ciL);
        mbedtls_mpi_zeroize(X->p, n);
        free(X->p);
        X->p = p;
    }
    X->n = n;
    return 0;
}

/**
 * \brief          This function resizes an MPI downwards, keeping at
 *                 least the specified number of limbs.
 *
 *                 If \c X is smaller than \c nblimbs, it is resized up
 *                 instead.
 *
 * \param X        The MPI to shrink. This must point to an initialized MPI.
 * \param nblimbs  The minimum number of limbs to keep.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed
 *                 which can only happen when resizing up
 * \return         Another negative error code on other kinds of failure.
 */
int mbedtls_mpi_shrink(mbedtls_mpi *X, size_t nblimbs)
{
    MPI_VALIDATE_RET(X);
    if (X->n <= nblimbs) return mbedtls_mpi_grow(X, nblimbs);
    return mbedtls_mpi_resize(X, MAX(MAX(1, nblimbs), mbedtls_mpi_limbs(X)));
}

/**
 * \brief          Make a copy of an MPI.
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param Y        The source MPI. This must point to an initialized MPI.
 *
 * \note           The limb-buffer in the destination MPI is enlarged
 *                 if necessary to hold the value in the source MPI.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         Another negative error code on other kinds of failure.
 */
int mbedtls_mpi_copy( mbedtls_mpi *X, const mbedtls_mpi *Y )
{
    int ret = 0;
    size_t i;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( Y );
    if( X == Y )
        return( 0 );
    if( Y->n == 0 )
    {
        mbedtls_mpi_free( X );
        return( 0 );
    }
    for( i = Y->n - 1; i > 0; i-- )
        if( Y->p[i] != 0 )
            break;
    i++;
    X->s = Y->s;
    if( X->n < i )
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, i ) );
    }
    else
    {
        mbedtls_platform_zeroize( X->p + i, ( X->n - i ) * ciL );
    }
    memcpy( X->p, Y->p, i * ciL );
cleanup:
    return( ret );
}

/**
 * \brief          Swap the contents of two MPIs.
 *
 * \param X        The first MPI. It must be initialized.
 * \param Y        The second MPI. It must be initialized.
 */
void mbedtls_mpi_swap( mbedtls_mpi *X, mbedtls_mpi *Y )
{
    mbedtls_mpi T;
    MPI_VALIDATE( X );
    MPI_VALIDATE( Y );
    memcpy( &T,  X, sizeof( mbedtls_mpi ) );
    memcpy(  X,  Y, sizeof( mbedtls_mpi ) );
    memcpy(  Y, &T, sizeof( mbedtls_mpi ) );
}

/**
 * \brief          Perform a safe conditional copy of MPI which doesn't
 *                 reveal whether the condition was true or not.
 *
 * \param X        The MPI to conditionally assign to. This must point
 *                 to an initialized MPI.
 * \param Y        The MPI to be assigned from. This must point to an
 *                 initialized MPI.
 * \param assign   The condition deciding whether to perform the
 *                 assignment or not. Possible values:
 *                 * \c 1: Perform the assignment `X = Y`.
 *                 * \c 0: Keep the original value of \p X.
 *
 * \note           This function is equivalent to
 *                      `if( assign ) mbedtls_mpi_copy( X, Y );`
 *                 except that it avoids leaking any information about whether
 *                 the assignment was done or not (the above code may leak
 *                 information through branch prediction and/or memory access
 *                 patterns analysis).
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         Another negative error code on other kinds of failure.
 */
int mbedtls_mpi_safe_cond_assign(mbedtls_mpi *X,
                                 const mbedtls_mpi *Y,
                                 unsigned char assign)
{
    int ret = 0;
    size_t i;
    MPI_VALIDATE_RET(X);
    MPI_VALIDATE_RET(Y);
    /* make sure assign is 0 or 1 in a time-constant manner */
    if (Y->n > X->n)
        MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, Y->n ) );
    assign = (assign | (unsigned char)-assign) >> 7;
    X->s = Select(Y->s, X->s, -assign);
    for (i = 0; i < Y->n; i++)
        X->p[i] = Select(Y->p[i], X->p[i], -assign);
    for (i = Y->n; i < X->n; i++)
        X->p[i] &= __conceal("r", assign - 1);
cleanup:
    return( ret );
}

/**
 * \brief          Perform a safe conditional swap which doesn't
 *                 reveal whether the condition was true or not.
 *
 * \param X        The first MPI. This must be initialized.
 * \param Y        The second MPI. This must be initialized.
 * \param assign   The condition deciding whether to perform
 *                 the swap or not. Possible values:
 *                 * \c 1: Swap the values of \p X and \p Y.
 *                 * \c 0: Keep the original values of \p X and \p Y.
 *
 * \note           This function is equivalent to
 *                      if( assign ) mbedtls_mpi_swap( X, Y );
 *                 except that it avoids leaking any information about whether
 *                 the assignment was done or not (the above code may leak
 *                 information through branch prediction and/or memory access
 *                 patterns analysis).
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         Another negative error code on other kinds of failure.
 *
 */
int mbedtls_mpi_safe_cond_swap( mbedtls_mpi *X, mbedtls_mpi *Y, unsigned char swap )
{
    int ret, s;
    size_t i;
    mbedtls_mpi_uint tmp;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( Y );
    if( X == Y )
        return( 0 );
    /* make sure swap is 0 or 1 in a time-constant manner */
    swap = (swap | (unsigned char)-swap) >> 7;
    MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, Y->n ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_grow( Y, X->n ) );
    s = X->s;
    X->s = X->s * ( 1 - swap ) + Y->s * swap;
    Y->s = Y->s * ( 1 - swap ) +    s * swap;
    for( i = 0; i < X->n; i++ )
    {
        tmp = X->p[i];
        X->p[i] = X->p[i] * ( 1 - swap ) + Y->p[i] * swap;
        Y->p[i] = Y->p[i] * ( 1 - swap ) +     tmp * swap;
    }
cleanup:
    return( ret );
}

/**
 * \brief          Store integer value in MPI.
 *
 * \param X        The MPI to set. This must be initialized.
 * \param z        The value to use.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         Another negative error code on other kinds of failure.
 */
int mbedtls_mpi_lset( mbedtls_mpi *X, mbedtls_mpi_sint z )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    MPI_VALIDATE_RET( X );
    MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, 1 ) );
    mbedtls_platform_zeroize( X->p, X->n * ciL );
    X->p[0] = ( z < 0 ) ? -z : z;
    X->s    = ( z < 0 ) ? -1 : 1;
cleanup:
    return( ret );
}

/**
 * \brief          Get a specific bit from an MPI.
 *
 * \param X        The MPI to query. This must be initialized.
 * \param pos      Zero-based index of the bit to query.
 *
 * \return         \c 0 or \c 1 on success, depending on whether bit \c pos
 *                 of \c X is unset or set.
 * \return         A negative error code on failure.
 */
int mbedtls_mpi_get_bit( const mbedtls_mpi *X, size_t pos )
{
    MPI_VALIDATE_RET( X );
    if( X->n * biL <= pos )
        return( 0 );
    return( ( X->p[pos / biL] >> ( pos % biL ) ) & 0x01 );
}

/* Get a specific byte, without range checks. */
#define GET_BYTE( X, i )                                \
    ( ( ( X )->p[( i ) / ciL] >> ( ( ( i ) % ciL ) * 8 ) ) & 0xff )

/**
 * \brief          Modify a specific bit in an MPI.
 *
 * \note           This function will grow the target MPI if necessary to set a
 *                 bit to \c 1 in a not yet existing limb. It will not grow if
 *                 the bit should be set to \c 0.
 *
 * \param X        The MPI to modify. This must be initialized.
 * \param pos      Zero-based index of the bit to modify.
 * \param val      The desired value of bit \c pos: \c 0 or \c 1.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         Another negative error code on other kinds of failure.
 */
int mbedtls_mpi_set_bit( mbedtls_mpi *X, size_t pos, unsigned char val )
{
    int ret = 0;
    size_t off = pos / biL;
    size_t idx = pos % biL;
    MPI_VALIDATE_RET( X );
    if( val != 0 && val != 1 )
        return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    if( X->n * biL <= pos )
    {
        if( !val )
            return( 0 );
        MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, off + 1 ) );
    }
    X->p[off] &= ~( (mbedtls_mpi_uint) 0x01 << idx );
    X->p[off] |= (mbedtls_mpi_uint) val << idx;
cleanup:
    return( ret );
}

/**
 * \brief          Return the number of bits of value \c 0 before the
 *                 least significant bit of value \c 1.
 *
 * \note           This is the same as the zero-based index of
 *                 the least significant bit of value \c 1.
 *
 * \param X        The MPI to query.
 *
 * \return         The number of bits of value \c 0 before the least significant
 *                 bit of value \c 1 in \p X.
 */
size_t mbedtls_mpi_lsb( const mbedtls_mpi *X )
{
    size_t i, count = 0;
    MBEDTLS_INTERNAL_VALIDATE_RET(X, 0);
    for( i = 0; i < X->n; i++ )
    {
        if ( X->p[i] )
            return count + __builtin_ctzll(X->p[i]);
        else
            count += biL;
    }
    return 0;
}

/*
 * Count leading zero bits in a given integer
 */
static inline size_t mbedtls_clz( const mbedtls_mpi_uint x )
{
    return x ? __builtin_clzll(x) : biL;
}

/**
 * \brief          Return the number of bits up to and including the most
 *                 significant bit of value \c 1.
 *
 * \note           This is same as the one-based index of the most
 *                 significant bit of value \c 1.
 *
 * \param X        The MPI to query. This must point to an initialized MPI.
 *
 * \return         The number of bits up to and including the most
 *                 significant bit of value \c 1.
 */
size_t mbedtls_mpi_bitlen(const mbedtls_mpi *X)
{
    size_t n;
    n = mbedtls_mpi_limbs(X);
    if (!n) return 0;
    return biL - __builtin_clzll(X->p[n - 1]) + (n - 1) * biL;
}

/**
 * \brief          Return the total size of an MPI value in bytes.
 *
 * \param X        The MPI to use. This must point to an initialized MPI.
 *
 * \note           The value returned by this function may be less than
 *                 the number of bytes used to store \p X internally.
 *                 This happens if and only if there are trailing bytes
 *                 of value zero.
 *
 * \return         The least number of bytes capable of storing
 *                 the absolute value of \p X.
 */
size_t mbedtls_mpi_size( const mbedtls_mpi *X )
{
    return( ( mbedtls_mpi_bitlen( X ) + 7 ) >> 3 );
}

/*
 * Convert an ASCII character to digit value
 */
static int mpi_get_digit( mbedtls_mpi_uint *d, int radix, char c )
{
    *d = 255;
    if( c >= 0x30 && c <= 0x39 ) *d = c - 0x30;
    if( c >= 0x41 && c <= 0x46 ) *d = c - 0x37;
    if( c >= 0x61 && c <= 0x66 ) *d = c - 0x57;
    if( *d >= (mbedtls_mpi_uint) radix )
        return( MBEDTLS_ERR_MPI_INVALID_CHARACTER );
    return( 0 );
}

/**
 * \brief          Import an MPI from an ASCII string.
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param radix    The numeric base of the input string.
 * \param s        Null-terminated string buffer.
 *
 * \return         \c 0 if successful.
 * \return         A negative error code on failure.
 */
int mbedtls_mpi_read_string( mbedtls_mpi *X, int radix, const char *s )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t i, j, slen, n;
    mbedtls_mpi_uint d;
    mbedtls_mpi T;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( s );
    if( radix < 2 || radix > 16 )
        return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    mbedtls_mpi_init( &T );
    slen = strlen( s );
    if( radix == 16 )
    {
        if( slen > MPI_SIZE_T_MAX >> 2 )
            return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
        n = BITS_TO_LIMBS( slen << 2 );
        MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, n ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_lset( X, 0 ) );
        for( i = slen, j = 0; i > 0; i--, j++ )
        {
            if( i == 1 && s[i - 1] == '-' )
            {
                X->s = -1;
                break;
            }
            MBEDTLS_MPI_CHK( mpi_get_digit( &d, radix, s[i - 1] ) );
            X->p[j / ( 2 * ciL )] |= d << ( ( j % ( 2 * ciL ) ) << 2 );
        }
    }
    else
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_lset( X, 0 ) );
        for( i = 0; i < slen; i++ )
        {
            if( i == 0 && s[i] == '-' )
            {
                X->s = -1;
                continue;
            }
            MBEDTLS_MPI_CHK( mpi_get_digit( &d, radix, s[i] ) );
            MBEDTLS_MPI_CHK( mbedtls_mpi_mul_int( &T, X, radix ) );
            if( X->s == 1 )
            {
                MBEDTLS_MPI_CHK( mbedtls_mpi_add_int( X, &T, d ) );
            }
            else
            {
                MBEDTLS_MPI_CHK( mbedtls_mpi_sub_int( X, &T, d ) );
            }
        }
    }
cleanup:
    mbedtls_mpi_free( &T );
    return( ret );
}

/*
 * Helper to write the digits high-order first.
 */
static int mpi_write_hlp( mbedtls_mpi *X, int radix,
                          char **p, const size_t buflen )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_mpi_uint r;
    size_t length = 0;
    char *p_end = *p + buflen;
    do
    {
        if( length >= buflen )
        {
            return( MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL );
        }
        MBEDTLS_MPI_CHK( mbedtls_mpi_mod_int( &r, X, radix ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_div_int( X, NULL, X, radix ) );
        /*
         * Write the residue in the current position, as an ASCII character.
         */
        if( r < 0xA )
            *(--p_end) = (char)( '0' + r );
        else
            *(--p_end) = (char)( 'A' + ( r - 0xA ) );
        length++;
    } while( mbedtls_mpi_cmp_int( X, 0 ) != 0 );
    memmove( *p, p_end, length );
    *p += length;
cleanup:
    return( ret );
}

/**
 * \brief          Export an MPI to an ASCII string.
 *
 * \param X        The source MPI. This must point to an initialized MPI.
 * \param radix    The numeric base of the output string.
 * \param buf      The buffer to write the string to. This must be writable
 *                 buffer of length \p buflen Bytes.
 * \param buflen   The available size in Bytes of \p buf.
 * \param olen     The address at which to store the length of the string
 *                 written, including the  final \c NULL byte. This must
 *                 not be \c NULL.
 *
 * \note           You can call this function with `buflen == 0` to obtain the
 *                 minimum required buffer size in `*olen`.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL if the target buffer \p buf
 *                 is too small to hold the value of \p X in the desired base.
 *                 In this case, `*olen` is nonetheless updated to contain the
 *                 size of \p buf required for a successful call.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_write_string( const mbedtls_mpi *X, int radix,
                              char *buf, size_t buflen, size_t *olen )
{
    int ret = 0;
    size_t n;
    char *p;
    mbedtls_mpi T;
    MPI_VALIDATE_RET( X    );
    MPI_VALIDATE_RET( olen );
    MPI_VALIDATE_RET( buflen == 0 || buf );
    if( radix < 2 || radix > 16 )
        return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    n = mbedtls_mpi_bitlen( X ); /* Number of bits necessary to present `n`. */
    if( radix >=  4 ) n >>= 1;   /* Number of 4-adic digits necessary to present
                                  * `n`. If radix > 4, this might be a strict
                                  * overapproximation of the number of
                                  * radix-adic digits needed to present `n`. */
    if( radix >= 16 ) n >>= 1;   /* Number of hexadecimal digits necessary to
                                  * present `n`. */
    n += 1; /* Terminating null byte */
    n += 1; /* Compensate for the divisions above, which round down `n`
             * in case it's not even. */
    n += 1; /* Potential '-'-sign. */
    n += ( n & 1 ); /* Make n even to have enough space for hexadecimal writing,
                     * which always uses an even number of hex-digits. */
    if( buflen < n )
    {
        *olen = n;
        return( MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL );
    }
    p = buf;
    mbedtls_mpi_init( &T );
    if( X->s == -1 )
    {
        *p++ = '-';
        buflen--;
    }
    if( radix == 16 )
    {
        int c;
        size_t i, j, k;
        for( i = X->n, k = 0; i > 0; i-- )
        {
            for( j = ciL; j > 0; j-- )
            {
                c = ( X->p[i - 1] >> ( ( j - 1 ) << 3) ) & 0xFF;
                if( c == 0 && k == 0 && ( i + j ) != 2 )
                    continue;
                *(p++) = "0123456789ABCDEF" [c / 16];
                *(p++) = "0123456789ABCDEF" [c % 16];
                k = 1;
            }
        }
    }
    else
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &T, X ) );
        if( T.s == -1 )
            T.s = 1;
        MBEDTLS_MPI_CHK( mpi_write_hlp( &T, radix, &p, buflen ) );
    }
    *p++ = '\0';
    *olen = p - buf;
cleanup:
    mbedtls_mpi_free( &T );
    return( ret );
}

#if defined(MBEDTLS_FS_IO)
/**
 * \brief          Read an MPI from a line in an opened file.
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param radix    The numeric base of the string representation used
 *                 in the source line.
 * \param fin      The input file handle to use. This must not be \c NULL.
 *
 * \note           On success, this function advances the file stream
 *                 to the end of the current line or to EOF.
 *
 *                 The function returns \c 0 on an empty line.
 *
 *                 Leading whitespaces are ignored, as is a
 *                 '0x' prefix for radix \c 16.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL if the file read buffer
 *                 is too small.
 * \return         Another negative error code on failure.
 */
int mbedtls_mpi_read_file( mbedtls_mpi *X, int radix, FILE *fin )
{
    mbedtls_mpi_uint d;
    size_t slen;
    char *p;
    /*
     * Buffer should have space for (short) label and decimal formatted MPI,
     * newline characters and '\0'
     */
    char s[ MBEDTLS_MPI_RW_BUFFER_SIZE ];
    MPI_VALIDATE_RET( X   );
    MPI_VALIDATE_RET( fin );
    if( radix < 2 || radix > 16 )
        return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    mbedtls_platform_zeroize( s, sizeof( s ) );
    if( fgets( s, sizeof( s ) - 1, fin ) == NULL )
        return( MBEDTLS_ERR_MPI_FILE_IO_ERROR );
    slen = strlen( s );
    if( slen == sizeof( s ) - 2 )
        return( MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL );
    if( slen > 0 && s[slen - 1] == '\n' ) { slen--; s[slen] = '\0'; }
    if( slen > 0 && s[slen - 1] == '\r' ) { slen--; s[slen] = '\0'; }
    p = s + slen;
    while( p-- > s )
        if( mpi_get_digit( &d, radix, *p ) != 0 )
            break;
    return( mbedtls_mpi_read_string( X, radix, p + 1 ) );
}

/**
 * \brief          Export an MPI into an opened file.
 *
 * \param p        A string prefix to emit prior to the MPI data.
 *                 For example, this might be a label, or "0x" when
 *                 printing in base \c 16. This may be \c NULL if no prefix
 *                 is needed.
 * \param X        The source MPI. This must point to an initialized MPI.
 * \param radix    The numeric base to be used in the emitted string.
 * \param fout     The output file handle. This may be \c NULL, in which case
 *                 the output is written to \c stdout.
 *
 * \return         \c 0 if successful.
 * \return         A negative error code on failure.
 */
int mbedtls_mpi_write_file( const char *p, const mbedtls_mpi *X, int radix, FILE *fout )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t n, slen, plen;
    /*
     * Buffer should have space for (short) label and decimal formatted MPI,
     * newline characters and '\0'
     */
    char s[ MBEDTLS_MPI_RW_BUFFER_SIZE ];
    MPI_VALIDATE_RET( X );
    if( radix < 2 || radix > 16 )
        return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    mbedtls_platform_zeroize( s, sizeof( s ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_write_string( X, radix, s, sizeof( s ) - 2, &n ) );
    if( p == NULL ) p = "";
    plen = strlen( p );
    slen = strlen( s );
    s[slen++] = '\r';
    s[slen++] = '\n';
    if( fout )
    {
        if( fwrite( p, 1, plen, fout ) != plen ||
            fwrite( s, 1, slen, fout ) != slen )
            return( MBEDTLS_ERR_MPI_FILE_IO_ERROR );
    }
    else
        mbedtls_printf( "%s%s", p, s );
cleanup:
    return( ret );
}
#endif /* MBEDTLS_FS_IO */

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define mpi_uint_bigendian_to_host(x) (x)
#elif __SIZEOF_LONG__ == 8
#define mpi_uint_bigendian_to_host(x) __builtin_bswap64(x)
#elif __SIZEOF_LONG__ == 4
#define mpi_uint_bigendian_to_host(x) __builtin_bswap32(x)
#endif

static void mpi_bigendian_to_host( mbedtls_mpi_uint * const p, size_t limbs )
{
    mbedtls_mpi_uint *cur_limb_left;
    mbedtls_mpi_uint *cur_limb_right;
    if( !limbs )
        return;
    /*
     * Traverse limbs and
     * - adapt byte-order in each limb
     * - swap the limbs themselves.
     * For that, simultaneously traverse the limbs from left to right
     * and from right to left, as long as the left index is not bigger
     * than the right index (it's not a problem if limbs is odd and the
     * indices coincide in the last iteration).
     */
    for( cur_limb_left = p, cur_limb_right = p + ( limbs - 1 );
         cur_limb_left <= cur_limb_right;
         cur_limb_left++, cur_limb_right-- )
    {
        mbedtls_mpi_uint tmp;
        /* Note that if cur_limb_left == cur_limb_right,
         * this code effectively swaps the bytes only once. */
        tmp             = mpi_uint_bigendian_to_host( *cur_limb_left  );
        *cur_limb_left  = mpi_uint_bigendian_to_host( *cur_limb_right );
        *cur_limb_right = tmp;
    }
}

/**
 * \brief          Import X from unsigned binary data, little endian
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param p        The input buffer with \p n bytes.
 * \param n        The length of the input buffer \p p in Bytes.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_read_binary_le(mbedtls_mpi *X, const unsigned char *p, size_t n)
{
    int ret;
    size_t i;
    mbedtls_mpi_uint w;
    MPI_VALIDATE_RET(X);
    MPI_VALIDATE_RET(!n || p);
    if ((ret = mbedtls_mpi_resize(X, MAX(1, CHARS_TO_LIMBS(n))))) return ret;
    if (n) {
        for (i = 0; i + 8 <= n; i += 8)
            X->p[i / ciL] = READ64LE(p + i);
        if (i < n) {
            w = 0;
            do {
                w <<= 8;
                w |= p[i];
            } while (++i < n);
            X->p[i / ciL] = w;
        }
    } else {
        X->p[0] = 0;
    }
    X->s = 1;
    return 0;
}

/**
 * \brief          Import an MPI from unsigned big endian binary data.
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param p        The input buffer. This must be a readable buffer of length
 *                 \p n Bytes.
 * \param n        The length of the input buffer \p p in Bytes.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_read_binary(mbedtls_mpi *X, const unsigned char *p, size_t n)
{
    int ret;
    size_t i, j, k;
    mbedtls_mpi_uint w;
    MPI_VALIDATE_RET(X);
    MPI_VALIDATE_RET(!n || p);
    if ((ret = mbedtls_mpi_resize(X, MAX(1, CHARS_TO_LIMBS(n)))))
        return ret;
    if (n)
    {
        for (j = 0, i = n; i >= 8; i -= 8)
            X->p[j++] = READ64BE(p + i - ciL);
        if (i)
        {
            k = 0;
            w = 0;
            do
            {
                --i;
                w <<= 8;
                w |= p[k++];
            } while (i);
            X->p[j] = w;
        }
    }
    else
    {
        X->p[0] = 0;
    }
    X->s = 1;
    return 0;
}

/**
 * \brief          Export X into unsigned binary data, little endian.
 *                 Always fills the whole buffer, which will end with zeros
 *                 if the number is smaller.
 *
 * \param X        The source MPI. This must point to an initialized MPI.
 * \param buf      The output buffer. This must be a writable buffer of length
 *                 \p buflen Bytes.
 * \param buflen   The size of the output buffer \p buf in Bytes.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL if \p buf isn't
 *                 large enough to hold the value of \p X.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_write_binary_le( const mbedtls_mpi *X,
                                 unsigned char *buf, size_t buflen )
{
    size_t stored_bytes = X->n * ciL;
    size_t bytes_to_copy;
    size_t i;
    if( stored_bytes < buflen )
    {
        bytes_to_copy = stored_bytes;
    }
    else
    {
        bytes_to_copy = buflen;
        /* The output buffer is smaller than the allocated size of X.
         * However X may fit if its leading bytes are zero. */
        for( i = bytes_to_copy; i < stored_bytes; i++ )
        {
            if( GET_BYTE( X, i ) != 0 )
                return( MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL );
        }
    }
    for( i = 0; i < bytes_to_copy; i++ )
        buf[i] = GET_BYTE( X, i );
    if( stored_bytes < buflen )
    {
        /* Write trailing 0 bytes */
        mbedtls_platform_zeroize( buf + stored_bytes, buflen - stored_bytes );
    }
    return( 0 );
}

/**
 * \brief          Export X into unsigned binary data, big endian.
 *                 Always fills the whole buffer, which will start with zeros
 *                 if the number is smaller.
 *
 * \param X        The source MPI. This must point to an initialized MPI.
 * \param buf      The output buffer. This must be a writable buffer of length
 *                 \p buflen Bytes.
 * \param buflen   The size of the output buffer \p buf in Bytes.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL if \p buf isn't
 *                 large enough to hold the value of \p X.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_write_binary( const mbedtls_mpi *X,
                              unsigned char *buf, size_t buflen )
{
    size_t stored_bytes;
    size_t bytes_to_copy;
    unsigned char *p;
    size_t i;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( buflen == 0 || buf );
    stored_bytes = X->n * ciL;
    if( stored_bytes < buflen )
    {
        /* There is enough space in the output buffer. Write initial
         * null bytes and record the position at which to start
         * writing the significant bytes. In this case, the execution
         * trace of this function does not depend on the value of the
         * number. */
        bytes_to_copy = stored_bytes;
        p = buf + buflen - stored_bytes;
        mbedtls_platform_zeroize( buf, buflen - stored_bytes );
    }
    else
    {
        /* The output buffer is smaller than the allocated size of X.
         * However X may fit if its leading bytes are zero. */
        bytes_to_copy = buflen;
        p = buf;
        for( i = bytes_to_copy; i < stored_bytes; i++ )
        {
            if( GET_BYTE( X, i ) != 0 )
                return( MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL );
        }
    }
    for( i = 0; i < bytes_to_copy; i++ )
        p[bytes_to_copy - i - 1] = GET_BYTE( X, i );
    return( 0 );
}

/**
 * \brief          Compare the absolute values of two MPIs.
 *
 * \param X        The left-hand MPI. This must point to an initialized MPI.
 * \param Y        The right-hand MPI. This must point to an initialized MPI.
 *
 * \return         \c 1 if `|X|` is greater than `|Y|`.
 * \return         \c -1 if `|X|` is lesser than `|Y|`.
 * \return         \c 0 if `|X|` is equal to `|Y|`.
 */
int mbedtls_mpi_cmp_abs( const mbedtls_mpi *X, const mbedtls_mpi *Y )
{
    size_t i, j;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( Y );
    i = mbedtls_mpi_limbs(X);
    j = mbedtls_mpi_limbs(Y);
    if( !i && !j )
        return( 0 );
    if( i > j ) return(  1 );
    if( j > i ) return( -1 );
    for( ; i > 0; i-- )
    {
        if( X->p[i - 1] > Y->p[i - 1] ) return(  1 );
        if( X->p[i - 1] < Y->p[i - 1] ) return( -1 );
    }
    return( 0 );
}

/**
 * \brief          Compare two MPIs.
 *
 * \param X        The left-hand MPI. This must point to an initialized MPI.
 * \param Y        The right-hand MPI. This must point to an initialized MPI.
 *
 * \return         \c 1 if \p X is greater than \p Y.
 * \return         \c -1 if \p X is lesser than \p Y.
 * \return         \c 0 if \p X is equal to \p Y.
 */
int mbedtls_mpi_cmp_mpi( const mbedtls_mpi *X, const mbedtls_mpi *Y )
{
    size_t i, j;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( Y );
    i = mbedtls_mpi_limbs(X);
    j = mbedtls_mpi_limbs(Y);
    if( !i && !j )
        return( 0 );
    if( i > j ) return(  X->s );
    if( j > i ) return( -Y->s );
    if( X->s > 0 && Y->s < 0 ) return(  1 );
    if( Y->s > 0 && X->s < 0 ) return( -1 );
    for( ; i > 0; i-- )
    {
        if( X->p[i - 1] > Y->p[i - 1] ) return(  X->s );
        if( X->p[i - 1] < Y->p[i - 1] ) return( -X->s );
    }
    return( 0 );
}

/**
 * Decide if an integer is less than the other, without branches.
 *
 * \param x         First integer.
 * \param y         Second integer.
 *
 * \return          1 if \p x is less than \p y, 0 otherwise
 */
static unsigned ct_lt_mpi_uint( const mbedtls_mpi_uint x,
                                const mbedtls_mpi_uint y )
{
    mbedtls_mpi_uint ret;
    mbedtls_mpi_uint cond;
    /*
     * Check if the most significant bits (MSB) of the operands are different.
     */
    cond = ( x ^ y );
    /*
     * If the MSB are the same then the difference x-y will be negative (and
     * have its MSB set to 1 during conversion to unsigned) if and only if x<y.
     */
    ret = ( x - y ) & ~cond;
    /*
     * If the MSB are different, then the operand with the MSB of 1 is the
     * bigger. (That is if y has MSB of 1, then x<y is true and it is false if
     * the MSB of y is 0.)
     */
    ret |= y & cond;

    ret = ret >> ( biL - 1 );
    return (unsigned) ret;
}

/**
 * \brief          Check if an MPI is less than the other in constant time.
 *
 * \param X        The left-hand MPI. This must point to an initialized MPI
 *                 with the same allocated length as Y.
 * \param Y        The right-hand MPI. This must point to an initialized MPI
 *                 with the same allocated length as X.
 * \param ret      The result of the comparison:
 *                 \c 1 if \p X is less than \p Y.
 *                 \c 0 if \p X is greater than or equal to \p Y.
 *
 * \return         0 on success.
 * \return         MBEDTLS_ERR_MPI_BAD_INPUT_DATA if the allocated length of
 *                 the two input MPIs is not the same.
 */
int mbedtls_mpi_lt_mpi_ct( const mbedtls_mpi *X, const mbedtls_mpi *Y,
                           unsigned *ret )
{
    size_t i;
    /* The value of any of these variables is either 0 or 1 at all times. */
    unsigned cond, done, X_is_negative, Y_is_negative;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( Y );
    MPI_VALIDATE_RET( ret );
    if( X->n != Y->n )
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    /*
     * Set sign_N to 1 if N >= 0, 0 if N < 0.
     * We know that N->s == 1 if N >= 0 and N->s == -1 if N < 0.
     */
    X_is_negative = ( X->s & 2 ) >> 1;
    Y_is_negative = ( Y->s & 2 ) >> 1;
    /*
     * If the signs are different, then the positive operand is the bigger.
     * That is if X is negative (X_is_negative == 1), then X < Y is true and it
     * is false if X is positive (X_is_negative == 0).
     */
    cond = ( X_is_negative ^ Y_is_negative );
    *ret = cond & X_is_negative;
    /*
     * This is a constant-time function. We might have the result, but we still
     * need to go through the loop. Record if we have the result already.
     */
    done = cond;
    for( i = X->n; i > 0; i-- )
    {
        /*
         * If Y->p[i - 1] < X->p[i - 1] then X < Y is true if and only if both
         * X and Y are negative.
         *
         * Again even if we can make a decision, we just mark the result and
         * the fact that we are done and continue looping.
         */
        cond = ct_lt_mpi_uint( Y->p[i - 1], X->p[i - 1] );
        *ret |= cond & ( 1 - done ) & X_is_negative;
        done |= cond;
        /*
         * If X->p[i - 1] < Y->p[i - 1] then X < Y is true if and only if both
         * X and Y are positive.
         *
         * Again even if we can make a decision, we just mark the result and
         * the fact that we are done and continue looping.
         */
        cond = ct_lt_mpi_uint( X->p[i - 1], Y->p[i - 1] );
        *ret |= cond & ( 1 - done ) & ( 1 - X_is_negative );
        done |= cond;
    }
    return( 0 );
}

/**
 * \brief          Compare an MPI with an integer.
 *
 * \param X        The left-hand MPI. This must point to an initialized MPI.
 * \param z        The integer value to compare \p X to.
 *
 * \return         \c 1 if \p X is greater than \p z.
 * \return         \c -1 if \p X is lesser than \p z.
 * \return         \c 0 if \p X is equal to \p z.
 */
int mbedtls_mpi_cmp_int( const mbedtls_mpi *X, mbedtls_mpi_sint z )
{
    mbedtls_mpi Y;
    mbedtls_mpi_uint p[1];
    MPI_VALIDATE_RET( X );
    *p  = ( z < 0 ) ? -z : z;
    Y.s = ( z < 0 ) ? -1 : 1;
    Y.n = 1;
    Y.p = p;
    return( mbedtls_mpi_cmp_mpi( X, &Y ) );
}

/**
 * \brief          Perform an unsigned addition of MPIs: X = |A| + |B|
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param A        The first summand. This must point to an initialized MPI.
 * \param B        The second summand. This must point to an initialized MPI.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_add_abs( mbedtls_mpi *X, const mbedtls_mpi *A, const mbedtls_mpi *B )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t i, j;
    mbedtls_mpi_uint *o, *p, c, tmp;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( A );
    MPI_VALIDATE_RET( B );
    if( X == B )
    {
        const mbedtls_mpi *T = A; A = X; B = T;
    }
    if( X != A )
        MBEDTLS_MPI_CHK( mbedtls_mpi_copy( X, A ) );
    /*
     * X should always be positive as a result of unsigned additions.
     */
    X->s = 1;
    for( j = B->n; j > 0; j-- )
        if( B->p[j - 1] != 0 )
            break;
    MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, j ) );
    o = B->p; p = X->p; c = 0;
    /*
     * tmp is used because it might happen that p == o
     */
    for( i = 0; i < j; i++, o++, p++ )
    {
        tmp= *o;
        *p +=  c; c  = ( *p <  c );
        *p += tmp; c += ( *p < tmp );
    }
    while( c != 0 )
    {
        if( i >= X->n )
        {
            MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, i + 1 ) );
            p = X->p + i;
        }
        *p += c; c = ( *p < c ); i++; p++;
    }
cleanup:
    return( ret );
}

/**
 * Helper for mbedtls_mpi subtraction.
 *
 * Calculate d = a - b where d, a, and b have the same size.
 * This function operates modulo (2^ciL)^n and returns the carry
 * (1 if there was a wraparound, i.e. if `a < b`, and 0 otherwise).
 *
 * \param[out] d        Result of subtraction.
 * \param[in] a         Left operand.
 * \param[in] b         Right operand.
 * \param n             Number of limbs of \p a and \p b.
 * \return              1 if `d < s`.
 *                      0 if `d >= s`.
 */
forceinline mbedtls_mpi_uint mpi_sub_hlp(mbedtls_mpi_uint *d,
                                         const mbedtls_mpi_uint *a,
                                         const mbedtls_mpi_uint *b,
                                         size_t n)
{
    size_t i;
    unsigned char cf;
    mbedtls_mpi_uint c, x;
    (void)x;
    (void)cf;
    cf = c = i = 0;
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    if (!n) return 0;
    asm volatile("xor\t%1,%1\n\t"
                 ".align\t16\n1:\t"
                 "mov\t(%5,%3,8),%1\n\t"
                 "sbb\t(%6,%3,8),%1\n\t"
                 "mov\t%1,(%4,%3,8)\n\t"
                 "lea\t1(%3),%3\n\t"
                 "dec\t%2\n\t"
                 "jnz\t1b"
                 : "=@ccb"(cf), "=&r"(x), "+&c"(n), "=&r"(i)
                 : "r"(d), "r"(a), "r"(b), "3"(0)
                 : "cc", "memory");
    return cf;
#else
    for (; i < n; ++i)
        SBB(d[i], a[i], b[i], c, c);
    return c;
#endif
}

/**
 * \brief          Perform an unsigned subtraction of MPIs: X = |A| - |B|
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param A        The minuend. This must point to an initialized MPI.
 * \param B        The subtrahend. This must point to an initialized MPI.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_NEGATIVE_VALUE if \p B is greater than \p A.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_sub_abs( mbedtls_mpi *X, const mbedtls_mpi *A, const mbedtls_mpi *B )
{
    size_t n, m, r;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( A );
    MPI_VALIDATE_RET( B );
    if( X != A && !B->n )
        return mbedtls_mpi_copy( X, A ); /* wut */
    for( n = B->n; n > 0; n-- )
        if( B->p[n - 1] != 0 )
            break;
    if( n > A->n )
        return MBEDTLS_ERR_MPI_NEGATIVE_VALUE; /* B >= (2^ciL)^n > A */
    if (X != A)
    {
        if (X->n < A->n) {
            if ((r = mbedtls_mpi_grow(X, A->n))) return r;
        } else if (X->n > A->n) {
            mbedtls_mpi_zeroize(X->p + A->n, X->n - A->n);
        }
        if ((m = A->n - n))
            memcpy(X->p + n, A->p + n, m * ciL);
    }
    /*
     * X should always be positive as a result of unsigned subtractions.
     */
    X->s = 1;
    if( mpi_sub_hlp( X->p, A->p, B->p, n ) ){
        /* Propagate the carry to the first nonzero limb of X. */
        for( ; n < A->n && A->p[n] == 0; n++ )
            /* --X->p[n]; */
            X->p[n] = A->p[n] - 1;
        /* If we ran out of space for the carry, it means that the result
         * is negative. */
        if( n == X->n )
            return MBEDTLS_ERR_MPI_NEGATIVE_VALUE;
        --X->p[n];
    }
    return( 0 );
}

static int mpi_cmp_abs(const mbedtls_mpi *X,
                       const mbedtls_mpi *Y,
                       size_t *Xn,
                       size_t *Yn)
{
    size_t i, j;
    i = mbedtls_mpi_limbs(X);
    j = mbedtls_mpi_limbs(Y);
    *Xn = i;
    *Yn = j;
    if (!i && !j) return 0;
    if (i > j) return 1;
    if (j > i) return -1;
    for (; i > 0; i--)
    {
        if (X->p[i - 1] > Y->p[i - 1]) return 1;
        if (X->p[i - 1] < Y->p[i - 1]) return -1;
    }
    return 0;
}

static int mpi_sub_abs( mbedtls_mpi *X, const mbedtls_mpi *A, const mbedtls_mpi *B, size_t n )
{
    size_t m, r;
    if( X != A && !B->n )
        return mbedtls_mpi_copy( X, A ); /* wut */
    if( n > A->n )
        return MBEDTLS_ERR_MPI_NEGATIVE_VALUE; /* B >= (2^ciL)^n > A */
    if (X != A)
    {
        if (X->n < A->n) {
            if ((r = mbedtls_mpi_grow(X, A->n))) return r;
        } else if (X->n > A->n) {
            mbedtls_mpi_zeroize(X->p + A->n, X->n - A->n);
        }
        if ((m = A->n - n))
            memcpy(X->p + n, A->p + n, m * ciL);
    }
    /*
     * X should always be positive as a result of unsigned subtractions.
     */
    X->s = 1;
    if( mpi_sub_hlp( X->p, A->p, B->p, n ) ){
        /* Propagate the carry to the first nonzero limb of X. */
        for( ; n < A->n && A->p[n] == 0; n++ )
            /* --X->p[n]; */
            X->p[n] = A->p[n] - 1;
        /* If we ran out of space for the carry, it means that the result
         * is negative. */
        if( n == X->n )
            return MBEDTLS_ERR_MPI_NEGATIVE_VALUE;
        --X->p[n];
    }
    return( 0 );
}

/**
 * \brief          Perform a signed addition of MPIs: X = A + B
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param A        The first summand. This must point to an initialized MPI.
 * \param B        The second summand. This must point to an initialized MPI.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_add_mpi( mbedtls_mpi *X, const mbedtls_mpi *A, const mbedtls_mpi *B )
{
    int ret, s;
    size_t i, j;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( A );
    MPI_VALIDATE_RET( B );
    s = A->s;
    if( A->s * B->s < 0 )
    {
        if( mpi_cmp_abs( A, B, &i, &j ) >= 0 )
        {
            MBEDTLS_MPI_CHK( mpi_sub_abs( X, A, B, j ) );
            X->s =  s;
        }
        else
        {
            MBEDTLS_MPI_CHK( mpi_sub_abs( X, B, A, i ) );
            X->s = -s;
        }
    }
    else
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_add_abs( X, A, B ) );
        X->s = s;
    }
cleanup:
    return( ret );
}

/**
 * \brief          Perform a signed subtraction of MPIs: X = A - B
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param A        The minuend. This must point to an initialized MPI.
 * \param B        The subtrahend. This must point to an initialized MPI.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_sub_mpi( mbedtls_mpi *X, const mbedtls_mpi *A, const mbedtls_mpi *B )
{
    int ret, s;
    size_t i, j;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( A );
    MPI_VALIDATE_RET( B );
    s = A->s;
    if( A->s * B->s > 0 )
    {
        if( mpi_cmp_abs( A, B, &i, &j ) >= 0 )
        {
            MBEDTLS_MPI_CHK( mpi_sub_abs( X, A, B, j ) );
            X->s =  s;
        }
        else
        {
            MBEDTLS_MPI_CHK( mpi_sub_abs( X, B, A, i ) );
            X->s = -s;
        }
    }
    else
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_add_abs( X, A, B ) );
        X->s = s;
    }
cleanup:
    return( ret );
}

/**
 * \brief          Perform a signed addition of an MPI and an integer: X = A + b
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param A        The first summand. This must point to an initialized MPI.
 * \param b        The second summand.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_add_int( mbedtls_mpi *X, const mbedtls_mpi *A, mbedtls_mpi_sint b )
{
    mbedtls_mpi _B;
    mbedtls_mpi_uint p[1];
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( A );
    p[0] = ( b < 0 ) ? -b : b;
    _B.s = ( b < 0 ) ? -1 : 1;
    _B.n = 1;
    _B.p = p;
    return( mbedtls_mpi_add_mpi( X, A, &_B ) );
}

/**
 * \brief          Perform a signed subtraction of an MPI and an integer:
 *                 X = A - b
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param A        The minuend. This must point to an initialized MPI.
 * \param b        The subtrahend.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_sub_int( mbedtls_mpi *X, const mbedtls_mpi *A, mbedtls_mpi_sint b )
{
    mbedtls_mpi _B;
    mbedtls_mpi_uint p[1];
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( A );
    p[0] = ( b < 0 ) ? -b : b;
    _B.s = ( b < 0 ) ? -1 : 1;
    _B.n = 1;
    _B.p = p;
    return( mbedtls_mpi_sub_mpi( X, A, &_B ) );
}

/*
 * Unsigned integer divide - double mbedtls_mpi_uint dividend, u1/u0, and
 * mbedtls_mpi_uint divisor, d
 */
static mbedtls_mpi_uint mbedtls_int_div_int( mbedtls_mpi_uint u1,
                                             mbedtls_mpi_uint u0,
                                             mbedtls_mpi_uint d,
                                             mbedtls_mpi_uint *r )
{
#if defined(__x86_64__) && !defined(__STRICT_ANSI__)
    if (d && u1 < d)
    {
        mbedtls_mpi_uint quo, rem;
        asm("div\t%2" : "=a"(quo), "=d"(rem) : "r"(d), "0"(u0), "1"(u1) : "cc");
        if (r) *r = rem;
        return quo;
    }
    else
    {
        if (r) *r = ~0;
        return ~0;
    }
#else
#if defined(MBEDTLS_HAVE_UDBL)
    mbedtls_t_udbl dividend, quotient;
#else
    const mbedtls_mpi_uint radix = (mbedtls_mpi_uint) 1 << biH;
    const mbedtls_mpi_uint uint_halfword_mask = ( (mbedtls_mpi_uint) 1 << biH ) - 1;
    mbedtls_mpi_uint d0, d1, q0, q1, rAX, r0, quotient;
    mbedtls_mpi_uint u0_msw, u0_lsw;
    size_t s;
#endif
    /*
     * Check for overflow
     */
    if( 0 == d || u1 >= d )
    {
        if (r) *r = ~0;
        return ( ~0 );
    }
#if defined(MBEDTLS_HAVE_UDBL)
    dividend  = (mbedtls_t_udbl) u1 << biL;
    dividend |= (mbedtls_t_udbl) u0;
    quotient = dividend / d;
    if( quotient > ( (mbedtls_t_udbl) 1 << biL ) - 1 )
        quotient = ( (mbedtls_t_udbl) 1 << biL ) - 1;
    if( r )
        *r = (mbedtls_mpi_uint)( dividend - (quotient * d ) );
    return (mbedtls_mpi_uint) quotient;
#else
    /*
     * Algorithm D, Section 4.3.1 - The Art of Computer Programming
     *   Vol. 2 - Seminumerical Algorithms, Knuth
     */
    /*
     * Normalize the divisor, d, and dividend, u0, u1
     */
    s = mbedtls_clz( d );
    d = d << s;
    u1 = u1 << s;
    u1 |= ( u0 >> ( biL - s ) ) & ( -(mbedtls_mpi_sint)s >> ( biL - 1 ) );
    u0 =  u0 << s;
    d1 = d >> biH;
    d0 = d & uint_halfword_mask;
    u0_msw = u0 >> biH;
    u0_lsw = u0 & uint_halfword_mask;
    /*
     * Find the first quotient and remainder
     */
    q1 = u1 / d1;
    r0 = u1 - d1 * q1;
    while( q1 >= radix || ( q1 * d0 > radix * r0 + u0_msw ) )
    {
        q1 -= 1;
        r0 += d1;
        if ( r0 >= radix ) break;
    }
    rAX = ( u1 * radix ) + ( u0_msw - q1 * d );
    q0 = rAX / d1;
    r0 = rAX - q0 * d1;
    while( q0 >= radix || ( q0 * d0 > radix * r0 + u0_lsw ) )
    {
        q0 -= 1;
        r0 += d1;
        if ( r0 >= radix ) break;
    }
    if (r)
        *r = ( rAX * radix + u0_lsw - q0 * d ) >> s;
    quotient = q1 * radix + q0;
    return quotient;
#endif
#endif
}

static inline void Multiply2x1(uint64_t a[3], uint64_t b) {
    uint128_t x;
    uint64_t l, h;
    x = a[0];
    x *= b;
    l = x;
    h = x >> 64;
    x = a[1];
    x *= b;
    x += h + ((a[0] = l) < 0);
    l = x;
    h = x >> 64;
    a[2] = h + ((a[1] = l) < 0);
}

static inline bool GreaterThan3x3(uint64_t a[3], uint64_t b[3]) {
    if (a[2] > b[2]) return true;
    if (a[2] < b[2]) return false;
    if (a[1] > b[1]) return true;
    if (a[1] < b[1]) return false;
    return a[0] > b[0];
}

/**
 * \brief          Perform a division with remainder of two MPIs:
 *                 A = Q * B + R
 *
 * \param Q        The destination MPI for the quotient.
 *                 This may be \c NULL if the value of the
 *                 quotient is not needed.
 * \param R        The destination MPI for the remainder value.
 *                 This may be \c NULL if the value of the
 *                 remainder is not needed.
 * \param A        The dividend. This must point to an initialized MPi.
 * \param B        The divisor. This must point to an initialized MPI.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         #MBEDTLS_ERR_MPI_DIVISION_BY_ZERO if \p B equals zero.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_div_mpi(mbedtls_mpi *Q, mbedtls_mpi *R, const mbedtls_mpi *A,
                        const mbedtls_mpi *B)
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t i, n, t, k;
    mbedtls_mpi X, Y, Z, T1, T2;
    mbedtls_mpi_uint TP2[3];
    MPI_VALIDATE_RET(A);
    MPI_VALIDATE_RET(B);
    if (mbedtls_mpi_is_zero(B))
        return MBEDTLS_ERR_MPI_DIVISION_BY_ZERO;
    mbedtls_mpi_init(&X);
    mbedtls_mpi_init(&Y);
    mbedtls_mpi_init(&Z);
    mbedtls_mpi_init(&T1);
    /*
     * Avoid dynamic memory allocations for constant-size T2.
     *
     * T2 is used for comparison only and the 3 limbs are assigned explicitly,
     * so nobody increase the size of the MPI and we're safe to use an on-stack
     * buffer.
     */
    T2.s = 1;
    T2.n = sizeof(TP2) / sizeof(*TP2);
    T2.p = TP2;
    if (mbedtls_mpi_cmp_abs(A, B) < 0)
    {
        if (Q) MBEDTLS_MPI_CHK(mbedtls_mpi_lset(Q, 0));
        if (R) MBEDTLS_MPI_CHK(mbedtls_mpi_copy(R, A));
        return 0;
    }
    MBEDTLS_MPI_CHK(mbedtls_mpi_copy(&X, A));
    MBEDTLS_MPI_CHK(mbedtls_mpi_copy(&Y, B));
    X.s = Y.s = 1;
    MBEDTLS_MPI_CHK(mbedtls_mpi_grow(&Z, A->n + 2));
    MBEDTLS_MPI_CHK(mbedtls_mpi_lset(&Z, 0));
    MBEDTLS_MPI_CHK(mbedtls_mpi_grow(&T1, 80)); /* we need left pad hard below */
    k = mbedtls_mpi_bitlen(&Y) % biL;
    if (k < biL - 1)
    {
        k = biL - 1 - k;
        MBEDTLS_MPI_CHK(mbedtls_mpi_shift_l(&X, k));
        MBEDTLS_MPI_CHK(mbedtls_mpi_shift_l(&Y, k));
    }
    else
    {
        k = 0;
    }
    n = X.n - 1;
    t = Y.n - 1;
    MBEDTLS_MPI_CHK(mbedtls_mpi_shift_l(&Y, biL * (n - t)));
    while (mbedtls_mpi_cmp_abs(&X, &Y) >= 0)
    {
        Z.p[n - t]++;
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_abs(&X, &X, &Y));
    }
    mbedtls_mpi_shift_r(&Y, biL * (n - t));
    for (i = n; i > t; i--)
    {
        if (X.p[i] >= Y.p[t])
            Z.p[i - t - 1] = ~0;
        else
            Z.p[i - t - 1] = mbedtls_int_div_int(X.p[i], X.p[i - 1], Y.p[t], NULL);
        T2.p[0] = (i < 2) ? 0 : X.p[i - 2];
        T2.p[1] = (i < 1) ? 0 : X.p[i - 1];
        T2.p[2] = X.p[i];
        Z.p[i - t - 1]++;
        do {
            Z.p[i - t - 1]--;
            T1.p[0] = (t < 1) ? 0 : Y.p[t - 1];
            T1.p[1] = Y.p[t];
            Multiply2x1(T1.p, Z.p[i - t - 1]);
        } while (GreaterThan3x3(T1.p, T2.p));
        MBEDTLS_MPI_CHK(mbedtls_mpi_mul_int(&T1, &Y, Z.p[i - t - 1]));
        MBEDTLS_MPI_CHK(mbedtls_mpi_shift_l(&T1, biL * (i - t - 1)));
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_mpi(&X, &X, &T1));
        if (X.s < 0)
        {
            MBEDTLS_MPI_CHK(mbedtls_mpi_copy(&T1, &Y));
            MBEDTLS_MPI_CHK(mbedtls_mpi_shift_l(&T1, biL * (i - t - 1)));
            MBEDTLS_MPI_CHK(mbedtls_mpi_add_mpi(&X, &X, &T1));
            Z.p[i - t - 1]--;
        }
    }
    if (Q)
    {
        MBEDTLS_MPI_CHK(mbedtls_mpi_copy(Q, &Z));
        Q->s = A->s * B->s;
    }
    if (R)
    {
        mbedtls_mpi_shift_r(&X, k);
        X.s = A->s;
        MBEDTLS_MPI_CHK(mbedtls_mpi_copy(R, &X));
        if (mbedtls_mpi_is_zero(R)) R->s = 1;
    }
cleanup:
    mbedtls_mpi_free(&X);
    mbedtls_mpi_free(&Y);
    mbedtls_mpi_free(&Z);
    mbedtls_mpi_free(&T1);
    mbedtls_platform_zeroize(TP2, sizeof(TP2));
    return ret;
}

/**
 * \brief          Perform a division with remainder of an MPI by an integer:
 *                 A = Q * b + R
 *
 * \param Q        The destination MPI for the quotient.
 *                 This may be \c NULL if the value of the
 *                 quotient is not needed.
 * \param R        The destination MPI for the remainder value.
 *                 This may be \c NULL if the value of the
 *                 remainder is not needed.
 * \param A        The dividend. This must point to an initialized MPi.
 * \param b        The divisor.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if memory allocation failed.
 * \return         #MBEDTLS_ERR_MPI_DIVISION_BY_ZERO if \p b equals zero.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_div_int( mbedtls_mpi *Q, mbedtls_mpi *R,
                         const mbedtls_mpi *A,
                         mbedtls_mpi_sint b )
{
    mbedtls_mpi _B;
    mbedtls_mpi_uint p[1];
    MPI_VALIDATE_RET( A );
    p[0] = ( b < 0 ) ? -b : b;
    _B.s = ( b < 0 ) ? -1 : 1;
    _B.n = 1;
    _B.p = p;
    return( mbedtls_mpi_div_mpi( Q, R, A, &_B ) );
}

/**
 * \brief          Perform a modular reduction. R = A mod B
 *
 * \param R        The destination MPI for the residue value.
 *                 This must point to an initialized MPI.
 * \param A        The MPI to compute the residue of.
 *                 This must point to an initialized MPI.
 * \param B        The base of the modular reduction.
 *                 This must point to an initialized MPI.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         #MBEDTLS_ERR_MPI_DIVISION_BY_ZERO if \p B equals zero.
 * \return         #MBEDTLS_ERR_MPI_NEGATIVE_VALUE if \p B is negative.
 * \return         Another negative error code on different kinds of failure.
 *
 */
int mbedtls_mpi_mod_mpi( mbedtls_mpi *R, const mbedtls_mpi *A, const mbedtls_mpi *B )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    MPI_VALIDATE_RET( R );
    MPI_VALIDATE_RET( A );
    MPI_VALIDATE_RET( B );
    if( mbedtls_mpi_cmp_int( B, 0 ) < 0 )
        return( MBEDTLS_ERR_MPI_NEGATIVE_VALUE );
    MBEDTLS_MPI_CHK( mbedtls_mpi_div_mpi( NULL, R, A, B ) );
    while( mbedtls_mpi_cmp_int( R, 0 ) < 0 )
      MBEDTLS_MPI_CHK( mbedtls_mpi_add_mpi( R, R, B ) );
    while( mbedtls_mpi_cmp_mpi( R, B ) >= 0 )
      MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( R, R, B ) );
cleanup:
    return( ret );
}

/**
 * \brief          Perform a modular reduction with respect to an integer.
 *                 r = A mod b
 *
 * \param r        The address at which to store the residue.
 *                 This must not be \c NULL.
 * \param A        The MPI to compute the residue of.
 *                 This must point to an initialized MPi.
 * \param b        The integer base of the modular reduction.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         #MBEDTLS_ERR_MPI_DIVISION_BY_ZERO if \p b equals zero.
 * \return         #MBEDTLS_ERR_MPI_NEGATIVE_VALUE if \p b is negative.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_mod_int( mbedtls_mpi_uint *r, const mbedtls_mpi *A, mbedtls_mpi_sint b )
{
    size_t i;
    mbedtls_mpi_uint x, y, z;
    MPI_VALIDATE_RET( r );
    MPI_VALIDATE_RET( A );
    if( b == 0 )
        return( MBEDTLS_ERR_MPI_DIVISION_BY_ZERO );
    if( b < 0 )
        return( MBEDTLS_ERR_MPI_NEGATIVE_VALUE );
    /*
     * handle trivial cases
     */
    if( b == 1 )
    {
        *r = 0;
        return( 0 );
    }
    if( b == 2 )
    {
        *r = A->p[0] & 1;
        return( 0 );
    }
    /*
     * general case
     */
    for( i = A->n, y = 0; i > 0; i-- )
    {
        x  = A->p[i - 1];
        y  = ( y << biH ) | ( x >> biH );
        z  = y / b;
        y -= z * b;
        x <<= biH;
        y  = ( y << biH ) | ( x >> biH );
        z  = y / b;
        y -= z * b;
    }
    /*
     * If A is negative, then the current y represents a negative value.
     * Flipping it to the positive side.
     */
    if( A->s < 0 && y != 0 )
        y = b - y;
    *r = y;
    return( 0 );
}

/*
 * Fast Montgomery initialization (thanks to Tom St Denis)
 */
static void mpi_montg_init( mbedtls_mpi_uint *mm, const mbedtls_mpi *N )
{
    mbedtls_mpi_uint x, m0 = N->p[0];
    unsigned int i;
    x  = m0;
    x += ( ( m0 + 2 ) & 4 ) << 1;
    for( i = biL; i >= 8; i /= 2 )
        x *= ( 2 - ( m0 * x ) );
    *mm = -x;
}

/**
 * Montgomery multiplication: A = A * B * R^-1 mod N  (HAC 14.36)
 *
 * \param[in,out]   A   One of the numbers to multiply.
 *                      It must have at least as many limbs as N
 *                      (A->n >= N->n), and any limbs beyond n are ignored.
 *                      On successful completion, A contains the result of
 *                      the multiplication A * B * R^-1 mod N where
 *                      R = (2^ciL)^n.
 * \param[in]       B   One of the numbers to multiply.
 *                      It must be nonzero and must not have more limbs than N
 *                      (B->n <= N->n).
 * \param[in]       N   The modulo. N must be odd.
 * \param           mm  The value calculated by `mpi_montg_init(&mm, N)`.
 *                      This is -N^-1 mod 2^ciL.
 * \param[in,out]   T   A bignum for temporary storage.
 *                      It must be at least twice the limb size of N plus 2
 *                      (T->n >= 2 * (N->n + 1)).
 *                      Its initial content is unused and
 *                      its final content is indeterminate.
 *                      Note that unlike the usual convention in the library
 *                      for `const mbedtls_mpi*`, the content of T can change.
 */
static void mpi_montmul( mbedtls_mpi *A, const mbedtls_mpi *B, const mbedtls_mpi *N,
                         mbedtls_mpi_uint mm, const mbedtls_mpi *T )
{
    size_t i, n, m;
    mbedtls_mpi_uint u0, u1, *d, *Ap, *Bp, *Np;
    mbedtls_platform_zeroize( T->p, T->n * ciL );
    d = T->p;
    n = N->n;
    m = ( B->n < n ) ? B->n : n;
    Ap = A->p;
    Bp = B->p;
    Np = N->p;
    for( i = 0; i < n; i++ )
    {
        /*
         * T = (T + u0*B + u1*N) / 2^biL
         */
        u0 = Ap[i];
        u1 = ( d[0] + u0 * Bp[0] ) * mm;
        mbedtls_mpi_mul_hlp( m, Bp, d, u0 );
        mbedtls_mpi_mul_hlp( n, Np, d, u1 );
        *d++ = u0; d[n + 1] = 0;
    }
    /* At this point, d is either the desired result or the desired result
     * plus N. We now potentially subtract N, avoiding leaking whether the
     * subtraction is performed through side channels. */
    /* Copy the n least significant limbs of d to A, so that
     * A = d if d < N (recall that N has n limbs). */
    memcpy( Ap, d, n * ciL );
    /* If d >= N then we want to set A to d - N. To prevent timing attacks,
     * do the calculation without using conditional tests. */
    /* Set d to d0 + (2^biL)^n - N where d0 is the current value of d. */
    d[n] += 1;
    d[n] -= mpi_sub_hlp( d, d, Np, n );
    /* If d0 < N then d < (2^biL)^n
     * so d[n] == 0 and we want to keep A as it is.
     * If d0 >= N then d >= (2^biL)^n, and d <= (2^biL)^n + N < 2 * (2^biL)^n
     * so d[n] == 1 and we want to set A to the result of the subtraction
     * which is d - (2^biL)^n, i.e. the n least significant limbs of d.
     * This exactly corresponds to a conditional assignment. */
    for (i = 0; i < n; ++i) {
        Ap[i] = Select(d[i], Ap[i], -d[n]);
    }
}

/*
 * Montgomery reduction: A = A * R^-1 mod N
 *
 * See mpi_montmul() regarding constraints and guarantees on the parameters.
 */
static void mpi_montred( mbedtls_mpi *A, const mbedtls_mpi *N,
                         mbedtls_mpi_uint mm, const mbedtls_mpi *T )
{
    mbedtls_mpi_uint z = 1;
    mbedtls_mpi U;
    U.n = U.s = (int) z;
    U.p = &z;
    mpi_montmul( A, &U, N, mm, T );
}

/**
 * \brief          Perform a sliding-window exponentiation: X = A^E mod N
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param A        The base of the exponentiation.
 *                 This must point to an initialized MPI.
 * \param E        The exponent MPI. This must point to an initialized MPI.
 * \param N        The base for the modular reduction. This must point to an
 *                 initialized MPI.
 * \param _RR      A helper MPI depending solely on \p N which can be used to
 *                 speed-up multiple modular exponentiations for the same value
 *                 of \p N. This may be \c NULL. If it is not \c NULL, it must
 *                 point to an initialized MPI. If it hasn't been used after
 *                 the call to mbedtls_mpi_init(), this function will compute
 *                 the helper value and store it in \p _RR for reuse on
 *                 subsequent calls to this function. Otherwise, the function
 *                 will assume that \p _RR holds the helper value set by a
 *                 previous call to mbedtls_mpi_exp_mod(), and reuse it.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         #MBEDTLS_ERR_MPI_BAD_INPUT_DATA if \c N is negative or
 *                 even, or if \c E is negative.
 * \return         Another negative error code on different kinds of failures.
 *
 */
int mbedtls_mpi_exp_mod( mbedtls_mpi *X, const mbedtls_mpi *A,
                         const mbedtls_mpi *E, const mbedtls_mpi *N,
                         mbedtls_mpi *_RR )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t wbits, wsize, one = 1;
    size_t i, j, nblimbs;
    size_t bufsize, nbits;
    mbedtls_mpi_uint ei, mm, state;
    mbedtls_mpi RR, T, W[ 1 << MBEDTLS_MPI_WINDOW_SIZE ], Apos;
    int neg;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( A );
    MPI_VALIDATE_RET( E );
    MPI_VALIDATE_RET( N );
    if( mbedtls_mpi_cmp_int( N, 0 ) <= 0 || ( N->p[0] & 1 ) == 0 )
        return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    if( mbedtls_mpi_cmp_int( E, 0 ) < 0 )
        return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    if( mbedtls_mpi_bitlen( E ) > MBEDTLS_MPI_MAX_BITS ||
        mbedtls_mpi_bitlen( N ) > MBEDTLS_MPI_MAX_BITS )
        return ( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    /*
     * Init temps and window size
     */
    mpi_montg_init( &mm, N );
    mbedtls_mpi_init( &RR ); mbedtls_mpi_init( &T );
    mbedtls_mpi_init( &Apos );
    mbedtls_platform_zeroize( W, sizeof( W ) );
    i = mbedtls_mpi_bitlen( E );
    wsize = ( i > 671 ) ? 6 : ( i > 239 ) ? 5 :
            ( i >  79 ) ? 4 : ( i >  23 ) ? 3 : 1;
#if( MBEDTLS_MPI_WINDOW_SIZE < 6 )
    if( wsize > MBEDTLS_MPI_WINDOW_SIZE )
        wsize = MBEDTLS_MPI_WINDOW_SIZE;
#endif
    j = N->n + 1;
    MBEDTLS_MPI_CHK( mbedtls_mpi_grow( X, j ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_grow( &W[1],  j ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_grow( &T, j * 2 ) );
    /*
     * Compensate for negative A (and correct at the end)
     */
    neg = ( A->s == -1 );
    if( neg )
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &Apos, A ) );
        Apos.s = 1;
        A = &Apos;
    }
    /*
     * If 1st call, pre-compute R^2 mod N
     */
    if( _RR == NULL || _RR->p == NULL )
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &RR, 1 ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_shift_l( &RR, N->n * 2 * biL ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_mod_mpi( &RR, &RR, N ) );
        if( _RR )
            memcpy( _RR, &RR, sizeof( mbedtls_mpi ) );
    }
    else
        memcpy( &RR, _RR, sizeof( mbedtls_mpi ) );
    /*
     * W[1] = A * R^2 * R^-1 mod N = A * R mod N
     */
    if( mbedtls_mpi_cmp_mpi( A, N ) >= 0 )
        MBEDTLS_MPI_CHK( mbedtls_mpi_mod_mpi( &W[1], A, N ) );
    else
        MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &W[1], A ) );
    mpi_montmul( &W[1], &RR, N, mm, &T );
    /*
     * X = R^2 * R^-1 mod N = R mod N
     */
    MBEDTLS_MPI_CHK( mbedtls_mpi_copy( X, &RR ) );
    mpi_montred( X, N, mm, &T );
    if( wsize > 1 )
    {
        /*
         * W[1 << (wsize - 1)] = W[1] ^ (wsize - 1)
         */
        j =  one << ( wsize - 1 );
        MBEDTLS_MPI_CHK( mbedtls_mpi_grow( &W[j], N->n + 1 ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &W[j], &W[1]    ) );
        for( i = 0; i < wsize - 1; i++ )
            mpi_montmul( &W[j], &W[j], N, mm, &T );
        /*
         * W[i] = W[i - 1] * W[1]
         */
        for( i = j + 1; i < ( one << wsize ); i++ )
        {
            MBEDTLS_MPI_CHK( mbedtls_mpi_grow( &W[i], N->n + 1 ) );
            MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &W[i], &W[i - 1] ) );
            mpi_montmul( &W[i], &W[1], N, mm, &T );
        }
    }
    nblimbs = E->n;
    bufsize = 0;
    nbits   = 0;
    wbits   = 0;
    state   = 0;
    while( 1 )
    {
        if( bufsize == 0 )
        {
            if( nblimbs == 0 )
                break;
            nblimbs--;
            bufsize = sizeof( mbedtls_mpi_uint ) << 3;
        }
        bufsize--;
        ei = (E->p[nblimbs] >> bufsize) & 1;
        /*
         * skip leading 0s
         */
        if( ei == 0 && state == 0 )
            continue;
        if( ei == 0 && state == 1 )
        {
            /*
             * out of window, square X
             */
            mpi_montmul( X, X, N, mm, &T );
            continue;
        }
        /*
         * add ei to current window
         */
        state = 2;
        nbits++;
        wbits |= ( ei << ( wsize - nbits ) );
        if( nbits == wsize )
        {
            /*
             * X = X^wsize R^-1 mod N
             */
            for( i = 0; i < wsize; i++ )
                mpi_montmul( X, X, N, mm, &T );
            /*
             * X = X * W[wbits] R^-1 mod N
             */
            mpi_montmul( X, &W[wbits], N, mm, &T );
            state--;
            nbits = 0;
            wbits = 0;
        }
    }
    /*
     * process the remaining bits
     */
    for( i = 0; i < nbits; i++ )
    {
        mpi_montmul( X, X, N, mm, &T );
        wbits <<= 1;
        if( ( wbits & ( one << wsize ) ) != 0 )
            mpi_montmul( X, &W[1], N, mm, &T );
    }
    /*
     * X = A^E * R * R^-1 mod N = A^E mod N
     */
    mpi_montred( X, N, mm, &T );
    if( neg && E->n != 0 && ( E->p[0] & 1 ) != 0 )
    {
        X->s = -1;
        MBEDTLS_MPI_CHK( mbedtls_mpi_add_mpi( X, N, X ) );
    }
cleanup:
    for( i = ( one << ( wsize - 1 ) ); i < ( one << wsize ); i++ )
        mbedtls_mpi_free( &W[i] );
    mbedtls_mpi_free( &W[1] ); mbedtls_mpi_free( &T ); mbedtls_mpi_free( &Apos );
    if( _RR == NULL || _RR->p == NULL )
        mbedtls_mpi_free( &RR );
    return( ret );
}

/**
 * \brief          Compute the greatest common divisor: G = gcd(A, B)
 *
 * \param G        The destination MPI. This must point to an initialized MPI.
 * \param A        The first operand. This must point to an initialized MPI.
 * \param B        The second operand. This must point to an initialized MPI.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         Another negative error code on different kinds of failure.
 */
int mbedtls_mpi_gcd( mbedtls_mpi *G, const mbedtls_mpi *A, const mbedtls_mpi *B )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_mpi TA, TB;
    size_t lz, lzt, i, j;
    MPI_VALIDATE_RET( G );
    MPI_VALIDATE_RET( A );
    MPI_VALIDATE_RET( B );
    mbedtls_mpi_init( &TA ); mbedtls_mpi_init( &TB );
    MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &TA, A ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &TB, B ) );
    lz = mbedtls_mpi_lsb( &TA );
    lzt = mbedtls_mpi_lsb( &TB );
    if( lzt < lz )
        lz = lzt;
    MBEDTLS_MPI_CHK( mbedtls_mpi_shift_r( &TA, lz ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_shift_r( &TB, lz ) );
    TA.s = TB.s = 1;
    while( !mbedtls_mpi_is_zero( &TA ) )
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_shift_r( &TA, mbedtls_mpi_lsb( &TA ) ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_shift_r( &TB, mbedtls_mpi_lsb( &TB ) ) );
        if( mpi_cmp_abs( &TA, &TB, &i, &j ) >= 0 )
        {
            MBEDTLS_MPI_CHK( mpi_sub_abs( &TA, &TA, &TB, j ) );
            ShiftRight( TA.p, TA.n, 1 );
        }
        else
        {
            MBEDTLS_MPI_CHK( mpi_sub_abs( &TB, &TB, &TA, i ) );
            ShiftRight( TB.p, TB.n, 1 );
        }
    }
    MBEDTLS_MPI_CHK( mbedtls_mpi_shift_l( &TB, lz ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_copy( G, &TB ) );
cleanup:
    mbedtls_mpi_free( &TA ); mbedtls_mpi_free( &TB );
    return( ret );
}

/**
 * \brief          Fill an MPI with a number of random bytes.
 *
 * Use a temporary bytes representation to make sure the result is the
 * same regardless of the platform endianness (useful when f_rng is
 * actually deterministic, eg for tests).
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param size     The number of random bytes to generate.
 * \param f_rng    The RNG function to use. This must not be \c NULL.
 * \param p_rng    The RNG parameter to be passed to \p f_rng. This may be
 *                 \c NULL if \p f_rng doesn't need a context argument.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         Another negative error code on failure.
 *
 * \note           The bytes obtained from the RNG are interpreted
 *                 as a big-endian representation of an MPI; this can
 *                 be relevant in applications like deterministic ECDSA.
 */
int mbedtls_mpi_fill_random( mbedtls_mpi *X, size_t size,
                             int (*f_rng)(void *, unsigned char *, size_t),
                             void *p_rng )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t const limbs = CHARS_TO_LIMBS( size );
    size_t const overhead = ( limbs * ciL ) - size;
    unsigned char *Xp;
    MPI_VALIDATE_RET( X     );
    MPI_VALIDATE_RET( f_rng );
    MBEDTLS_MPI_CHK(mbedtls_mpi_resize( X, limbs ));
    MBEDTLS_MPI_CHK( mbedtls_mpi_lset( X, 0 ) );
    Xp = (unsigned char*) X->p;
    MBEDTLS_MPI_CHK( f_rng( p_rng, Xp + overhead, size ) );
    mpi_bigendian_to_host( X->p, limbs );
cleanup:
    return( ret );
}

/**
 * \brief          Compute the modular inverse: X = A^-1 mod N
 *
 * \param X        The destination MPI. This must point to an initialized MPI.
 * \param A        The MPI to calculate the modular inverse of. This must point
 *                 to an initialized MPI.
 * \param N        The base of the modular inversion. This must point to an
 *                 initialized MPI.
 *
 * \return         \c 0 if successful.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         #MBEDTLS_ERR_MPI_BAD_INPUT_DATA if \p N is less than
 *                 or equal to one.
 * \return         #MBEDTLS_ERR_MPI_NOT_ACCEPTABLE if \p has no modular inverse
 *                 with respect to \p N.
 */
int mbedtls_mpi_inv_mod( mbedtls_mpi *X, const mbedtls_mpi *A, const mbedtls_mpi *N )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_mpi G, TA, TU, U1, U2, TB, TV, V1, V2;
    MPI_VALIDATE_RET( X );
    MPI_VALIDATE_RET( A );
    MPI_VALIDATE_RET( N );
    if( mbedtls_mpi_cmp_int( N, 1 ) <= 0 )
        return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    mbedtls_mpi_init( &TA ); mbedtls_mpi_init( &TU ); mbedtls_mpi_init( &U1 ); mbedtls_mpi_init( &U2 );
    mbedtls_mpi_init( &G ); mbedtls_mpi_init( &TB ); mbedtls_mpi_init( &TV );
    mbedtls_mpi_init( &V1 ); mbedtls_mpi_init( &V2 );
    MBEDTLS_MPI_CHK( mbedtls_mpi_gcd( &G, A, N ) );
    if( mbedtls_mpi_cmp_int( &G, 1 ) != 0 )
    {
        ret = MBEDTLS_ERR_MPI_NOT_ACCEPTABLE;
        goto cleanup;
    }
    MBEDTLS_MPI_CHK( mbedtls_mpi_mod_mpi( &TA, A, N ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &TU, &TA ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &TB, N ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &TV, N ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &U1, 1 ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &U2, 0 ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &V1, 0 ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &V2, 1 ) );
    do
    {
        while( ( TU.p[0] & 1 ) == 0 )
        {
            ShiftRight( TU.p, TU.n, 1 );
            if( ( U1.p[0] & 1 ) != 0 || ( U2.p[0] & 1 ) != 0 )
            {
                MBEDTLS_MPI_CHK( mbedtls_mpi_add_mpi( &U1, &U1, &TB ) );
                MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( &U2, &U2, &TA ) );
            }
            ShiftRight( U1.p, U1.n, 1 );
            ShiftRight( U2.p, U2.n, 1 );
        }
        while( ( TV.p[0] & 1 ) == 0 )
        {
            ShiftRight( TV.p, TV.n, 1 );
            if( ( V1.p[0] & 1 ) != 0 || ( V2.p[0] & 1 ) != 0 )
            {
                MBEDTLS_MPI_CHK( mbedtls_mpi_add_mpi( &V1, &V1, &TB ) );
                MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( &V2, &V2, &TA ) );
            }
            ShiftRight( V1.p, V1.n, 1 );
            ShiftRight( V2.p, V2.n, 1 );
        }
        if( mbedtls_mpi_cmp_mpi( &TU, &TV ) >= 0 )
        {
            MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( &TU, &TU, &TV ) );
            MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( &U1, &U1, &V1 ) );
            MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( &U2, &U2, &V2 ) );
        }
        else
        {
            MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( &TV, &TV, &TU ) );
            MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( &V1, &V1, &U1 ) );
            MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( &V2, &V2, &U2 ) );
        }
    }
    while( !mbedtls_mpi_is_zero(&TU) );
    while( mbedtls_mpi_cmp_int( &V1, 0 ) < 0 )
        MBEDTLS_MPI_CHK( mbedtls_mpi_add_mpi( &V1, &V1, N ) );
    while( mbedtls_mpi_cmp_mpi( &V1, N ) >= 0 )
        MBEDTLS_MPI_CHK( mbedtls_mpi_sub_mpi( &V1, &V1, N ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_copy( X, &V1 ) );
cleanup:
    mbedtls_mpi_free( &TA ); mbedtls_mpi_free( &TU ); mbedtls_mpi_free( &U1 ); mbedtls_mpi_free( &U2 );
    mbedtls_mpi_free( &G ); mbedtls_mpi_free( &TB ); mbedtls_mpi_free( &TV );
    mbedtls_mpi_free( &V1 ); mbedtls_mpi_free( &V2 );
    return( ret );
}

#if defined(MBEDTLS_GENPRIME)

static const short small_prime[] =
{
        3,    5,    7,   11,   13,   17,   19,   23,
       29,   31,   37,   41,   43,   47,   53,   59,
       61,   67,   71,   73,   79,   83,   89,   97,
      101,  103,  107,  109,  113,  127,  131,  137,
      139,  149,  151,  157,  163,  167,  173,  179,
      181,  191,  193,  197,  199,  211,  223,  227,
      229,  233,  239,  241,  251,  257,  263,  269,
      271,  277,  281,  283,  293,  307,  311,  313,
      317,  331,  337,  347,  349,  353,  359,  367,
      373,  379,  383,  389,  397,  401,  409,  419,
      421,  431,  433,  439,  443,  449,  457,  461,
      463,  467,  479,  487,  491,  499,  503,  509,
      521,  523,  541,  547,  557,  563,  569,  571,
      577,  587,  593,  599,  601,  607,  613,  617,
      619,  631,  641,  643,  647,  653,  659,  661,
      673,  677,  683,  691,  701,  709,  719,  727,
      733,  739,  743,  751,  757,  761,  769,  773,
      787,  797,  809,  811,  821,  823,  827,  829,
      839,  853,  857,  859,  863,  877,  881,  883,
      887,  907,  911,  919,  929,  937,  941,  947,
      953,  967,  971,  977,  983,  991,  997, -103
};

/*
 * Small divisors test (X must be positive)
 *
 * Return values:
 * 0: no small factor (possible prime, more tests needed)
 * 1: certain prime
 * MBEDTLS_ERR_MPI_NOT_ACCEPTABLE: certain non-prime
 * other negative: error
 */
static int mpi_check_small_factors( const mbedtls_mpi *X )
{
    int ret = 0;
    size_t i;
    mbedtls_mpi_uint r;
    if( ( X->p[0] & 1 ) == 0 )
        return( MBEDTLS_ERR_MPI_NOT_ACCEPTABLE );
    for( i = 0; small_prime[i] > 0; i++ )
    {
        if( mbedtls_mpi_cmp_int( X, small_prime[i] ) <= 0 )
            return( 1 );
        MBEDTLS_MPI_CHK( mbedtls_mpi_mod_int( &r, X, small_prime[i] ) );
        if( r == 0 )
            return( MBEDTLS_ERR_MPI_NOT_ACCEPTABLE );
    }
cleanup:
    return( ret );
}

/*
 * Miller-Rabin pseudo-primality test  (HAC 4.24)
 */
static int mpi_miller_rabin( const mbedtls_mpi *X, size_t rounds,
                             int (*f_rng)(void *, unsigned char *, size_t),
                             void *p_rng )
{
    int ret, count;
    size_t i, j, k, s;
    mbedtls_mpi W, R, T, A, RR;
    MPI_VALIDATE_RET( X     );
    MPI_VALIDATE_RET( f_rng );
    mbedtls_mpi_init( &W ); mbedtls_mpi_init( &R );
    mbedtls_mpi_init( &T ); mbedtls_mpi_init( &A );
    mbedtls_mpi_init( &RR );
    /*
     * W = |X| - 1
     * R = W >> lsb( W )
     */
    MBEDTLS_MPI_CHK( mbedtls_mpi_sub_int( &W, X, 1 ) );
    s = mbedtls_mpi_lsb( &W );
    MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &R, &W ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_shift_r( &R, s ) );
    for( i = 0; i < rounds; i++ )
    {
        /*
         * pick a random A, 1 < A < |X| - 1
         */
        count = 0;
        do {
            MBEDTLS_MPI_CHK( mbedtls_mpi_fill_random( &A, X->n * ciL, f_rng, p_rng ) );
            j = mbedtls_mpi_bitlen( &A );
            k = mbedtls_mpi_bitlen( &W );
            if (j > k) {
                A.p[A.n - 1] &= ( (mbedtls_mpi_uint) 1 << ( k - ( A.n - 1 ) * biL - 1 ) ) - 1;
            }
            if (count++ > 30) {
                ret = MBEDTLS_ERR_MPI_NOT_ACCEPTABLE;
                goto cleanup;
            }
        } while ( mbedtls_mpi_cmp_mpi( &A, &W ) >= 0 ||
                  mbedtls_mpi_cmp_int( &A, 1 )  <= 0    );
        /*
         * A = A^R mod |X|
         */
        MBEDTLS_MPI_CHK( mbedtls_mpi_exp_mod( &A, &A, &R, X, &RR ) );
        if( mbedtls_mpi_cmp_mpi( &A, &W ) == 0 ||
            mbedtls_mpi_cmp_int( &A,  1 ) == 0 )
            continue;
        j = 1;
        while( j < s && mbedtls_mpi_cmp_mpi( &A, &W ) != 0 )
        {
            /*
             * A = A * A mod |X|
             */
            MBEDTLS_MPI_CHK( mbedtls_mpi_mul_mpi( &T, &A, &A ) );
            MBEDTLS_MPI_CHK( mbedtls_mpi_mod_mpi( &A, &T, X  ) );
            if( mbedtls_mpi_cmp_int( &A, 1 ) == 0 )
                break;
            j++;
        }
        /*
         * not prime if A != |X| - 1 or A == 1
         */
        if( mbedtls_mpi_cmp_mpi( &A, &W ) != 0 ||
            mbedtls_mpi_cmp_int( &A,  1 ) == 0 )
        {
            ret = MBEDTLS_ERR_MPI_NOT_ACCEPTABLE;
            break;
        }
    }
cleanup:
    mbedtls_mpi_free( &W ); mbedtls_mpi_free( &R );
    mbedtls_mpi_free( &T ); mbedtls_mpi_free( &A );
    mbedtls_mpi_free( &RR );
    return( ret );
}

/**
 * \brief          Miller-Rabin primality test.
 *
 * \warning        If \p X is potentially generated by an adversary, for example
 *                 when validating cryptographic parameters that you didn't
 *                 generate yourself and that are supposed to be prime, then
 *                 \p rounds should be at least the half of the security
 *                 strength of the cryptographic algorithm. On the other hand,
 *                 if \p X is chosen uniformly or non-adversially (as is the
 *                 case when mbedtls_mpi_gen_prime calls this function), then
 *                 \p rounds can be much lower.
 *
 * \param X        The MPI to check for primality.
 *                 This must point to an initialized MPI.
 * \param rounds   The number of bases to perform the Miller-Rabin primality
 *                 test for. The probability of returning 0 on a composite is
 *                 at most 2<sup>-2*\p rounds</sup>.
 * \param f_rng    The RNG function to use. This must not be \c NULL.
 * \param p_rng    The RNG parameter to be passed to \p f_rng.
 *                 This may be \c NULL if \p f_rng doesn't use
 *                 a context parameter.
 *
 * \return         \c 0 if successful, i.e. \p X is probably prime.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         #MBEDTLS_ERR_MPI_NOT_ACCEPTABLE if \p X is not prime.
 * \return         Another negative error code on other kinds of failure.
 */
int mbedtls_mpi_is_prime_ext( const mbedtls_mpi *X, int rounds,
                              int (*f_rng)(void *, unsigned char *, size_t),
                              void *p_rng )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_mpi XX;
    MPI_VALIDATE_RET( X     );
    MPI_VALIDATE_RET( f_rng );
    XX.s = 1;
    XX.n = X->n;
    XX.p = X->p;
    if( mbedtls_mpi_cmp_int( &XX, 0 ) == 0 ||
        mbedtls_mpi_cmp_int( &XX, 1 ) == 0 )
        return( MBEDTLS_ERR_MPI_NOT_ACCEPTABLE );
    if( mbedtls_mpi_cmp_int( &XX, 2 ) == 0 )
        return( 0 );
    if( ( ret = mpi_check_small_factors( &XX ) ) != 0 )
    {
        if( ret == 1 )
            return( 0 );
        return( ret );
    }
    return( mpi_miller_rabin( &XX, rounds, f_rng, p_rng ) );
}

/**
 * \brief          Generate a prime number.
 *
 *                 To generate an RSA key in a way recommended by FIPS
 *                 186-4, both primes must be either 1024 bits or 1536
 *                 bits long, and flags must contain
 *                 MBEDTLS_MPI_GEN_PRIME_FLAG_LOW_ERR.
 *
 * \param X        The destination MPI to store the generated prime in.
 *                 This must point to an initialized MPi.
 * \param nbits    The required size of the destination MPI in bits.
 *                 This must be between \c 3 and #MBEDTLS_MPI_MAX_BITS.
 * \param flags    A mask of flags of type #mbedtls_mpi_gen_prime_flag_t.
 * \param f_rng    The RNG function to use. This must not be \c NULL.
 * \param p_rng    The RNG parameter to be passed to \p f_rng.
 *                 This may be \c NULL if \p f_rng doesn't use
 *                 a context parameter.
 *
 * \return         \c 0 if successful, in which case \p X holds a
 *                 probably prime number.
 * \return         #MBEDTLS_ERR_MPI_ALLOC_FAILED if a memory allocation failed.
 * \return         #MBEDTLS_ERR_MPI_BAD_INPUT_DATA if `nbits` is not between
 *                 \c 3 and #MBEDTLS_MPI_MAX_BITS.
 */
int mbedtls_mpi_gen_prime( mbedtls_mpi *X, size_t nbits, int flags,
                           int (*f_rng)(void *, unsigned char *, size_t),
                           void *p_rng )
{
    int ret = MBEDTLS_ERR_MPI_NOT_ACCEPTABLE;
    size_t k, n;
    int rounds;
    mbedtls_mpi_uint r;
    mbedtls_mpi Y;
    MPI_VALIDATE_RET( X     );
    MPI_VALIDATE_RET( f_rng );
    if( nbits < 3 || nbits > MBEDTLS_MPI_MAX_BITS )
        return( MBEDTLS_ERR_MPI_BAD_INPUT_DATA );
    mbedtls_mpi_init( &Y );
    n = BITS_TO_LIMBS( nbits );
    if( ( flags & MBEDTLS_MPI_GEN_PRIME_FLAG_LOW_ERR ) == 0 )
    {
        /*
         * 2^-80 error probability, number of rounds chosen per HAC, table 4.4
         */
        rounds = ( ( nbits >= 1300 ) ?  2 : ( nbits >=  850 ) ?  3 :
                   ( nbits >=  650 ) ?  4 : ( nbits >=  350 ) ?  8 :
                   ( nbits >=  250 ) ? 12 : ( nbits >=  150 ) ? 18 : 27 );
    }
    else
    {
        /*
         * 2^-100 error probability, number of rounds computed based on HAC,
         * fact 4.48
         */
        rounds = ( ( nbits >= 1450 ) ?  4 : ( nbits >=  1150 ) ?  5 :
                   ( nbits >= 1000 ) ?  6 : ( nbits >=   850 ) ?  7 :
                   ( nbits >=  750 ) ?  8 : ( nbits >=   500 ) ? 13 :
                   ( nbits >=  250 ) ? 28 : ( nbits >=   150 ) ? 40 : 51 );
    }
    while( 1 )
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_fill_random( X, n * ciL, f_rng, p_rng ) );
        /* make sure generated number is at least (nbits-1)+0.5 bits (FIPS 186-4 §B.3.3 steps 4.4, 5.5) */
        if( X->p[n-1] < 0xb504f333f9de6485ULL  /* ceil(2^63.5) */ ) continue;
        k = n * biL;
        if( k > nbits ) MBEDTLS_MPI_CHK( mbedtls_mpi_shift_r( X, k - nbits ) );
        X->p[0] |= 1;
        if( ( flags & MBEDTLS_MPI_GEN_PRIME_FLAG_DH ) == 0 )
        {
            ret = mbedtls_mpi_is_prime_ext( X, rounds, f_rng, p_rng );
            if( ret != MBEDTLS_ERR_MPI_NOT_ACCEPTABLE )
                goto cleanup;
        }
        else
        {
            /*
             * An necessary condition for Y and X = 2Y + 1 to be prime
             * is X = 2 mod 3 (which is equivalent to Y = 2 mod 3).
             * Make sure it is satisfied, while keeping X = 3 mod 4
             */
            X->p[0] |= 2;
            MBEDTLS_MPI_CHK( mbedtls_mpi_mod_int( &r, X, 3 ) );
            if( r == 0 )
                MBEDTLS_MPI_CHK( mbedtls_mpi_add_int( X, X, 8 ) );
            else if( r == 1 )
                MBEDTLS_MPI_CHK( mbedtls_mpi_add_int( X, X, 4 ) );
            /* Set Y = (X-1) / 2, which is X / 2 because X is odd */
            MBEDTLS_MPI_CHK( mbedtls_mpi_copy( &Y, X ) );
            ShiftRight( Y.p, Y.n, 1 );
            while( 1 )
            {
                /*
                 * First, check small factors for X and Y
                 * before doing Miller-Rabin on any of them
                 */
                if( ( ret = mpi_check_small_factors(  X         ) ) == 0 &&
                    ( ret = mpi_check_small_factors( &Y         ) ) == 0 &&
                    ( ret = mpi_miller_rabin(  X, rounds, f_rng, p_rng  ) )
                                                                    == 0 &&
                    ( ret = mpi_miller_rabin( &Y, rounds, f_rng, p_rng  ) )
                                                                    == 0 )
                    goto cleanup;
                if( ret != MBEDTLS_ERR_MPI_NOT_ACCEPTABLE )
                    goto cleanup;
                /*
                 * Next candidates. We want to preserve Y = (X-1) / 2 and
                 * Y = 1 mod 2 and Y = 2 mod 3 (eq X = 3 mod 4 and X = 2 mod 3)
                 * so up Y by 6 and X by 12.
                 */
                MBEDTLS_MPI_CHK( mbedtls_mpi_add_int(  X,  X, 12 ) );
                MBEDTLS_MPI_CHK( mbedtls_mpi_add_int( &Y, &Y, 6  ) );
            }
        }
    }
cleanup:
    mbedtls_mpi_free( &Y );
    return( ret );
}

#endif /* MBEDTLS_GENPRIME */

#if defined(MBEDTLS_SELF_TEST)

#define GCD_PAIR_COUNT  3

static const int gcd_pairs[GCD_PAIR_COUNT][3] =
{
    { 693, 609, 21 },
    { 1764, 868, 28 },
    { 768454923, 542167814, 1 }
};

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 */
int mbedtls_mpi_self_test( int verbose )
{
    int ret, i;
    mbedtls_mpi A, E, N, X, Y, U, V;
    mbedtls_mpi_init( &A ); mbedtls_mpi_init( &E ); mbedtls_mpi_init( &N ); mbedtls_mpi_init( &X );
    mbedtls_mpi_init( &Y ); mbedtls_mpi_init( &U ); mbedtls_mpi_init( &V );
    MBEDTLS_MPI_CHK( mbedtls_mpi_read_string( &A, 16,
        "EFE021C2645FD1DC586E69184AF4A31E" \
        "D5F53E93B5F123FA41680867BA110131" \
        "944FE7952E2517337780CB0DB80E61AA" \
        "E7C8DDC6C5C6AADEB34EB38A2F40D5E6" ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_read_string( &E, 16,
        "B2E7EFD37075B9F03FF989C7C5051C20" \
        "34D2A323810251127E7BF8625A4F49A5" \
        "F3E27F4DA8BD59C47D6DAABA4C8127BD" \
        "5B5C25763222FEFCCFC38B832366C29E" ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_read_string( &N, 16,
        "0066A198186C18C10B2F5ED9B522752A" \
        "9830B69916E535C8F047518A889A43A5" \
        "94B6BED27A168D31D4A52F88925AA8F5" ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_mul_mpi( &X, &A, &N ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_read_string( &U, 16,
        "602AB7ECA597A3D6B56FF9829A5E8B85" \
        "9E857EA95A03512E2BAE7391688D264A" \
        "A5663B0341DB9CCFD2C4C5F421FEC814" \
        "8001B72E848A38CAE1C65F78E56ABDEF" \
        "E12D3C039B8A02D6BE593F0BBBDA56F1" \
        "ECF677152EF804370C1A305CAF3B5BF1" \
        "30879B56C61DE584A0F53A2447A51E" ) );
    if( verbose != 0 )
        mbedtls_printf( "  MPI test #1 (mul_mpi): " );
    if( mbedtls_mpi_cmp_mpi( &X, &U ) != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "failed\n" );
        ret = 1;
        goto cleanup;
    }
    if( verbose != 0 )
        mbedtls_printf( "passed\n" );
    MBEDTLS_MPI_CHK( mbedtls_mpi_div_mpi( &X, &Y, &A, &N ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_read_string( &U, 16,
        "256567336059E52CAE22925474705F39A94" ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_read_string( &V, 16,
        "6613F26162223DF488E9CD48CC132C7A" \
        "0AC93C701B001B092E4E5B9F73BCD27B" \
        "9EE50D0657C77F374E903CDFA4C642" ) );
    if( verbose != 0 )
        mbedtls_printf( "  MPI test #2 (div_mpi): " );
    if( mbedtls_mpi_cmp_mpi( &X, &U ) != 0 ||
        mbedtls_mpi_cmp_mpi( &Y, &V ) != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "failed\n" );
        ret = 1;
        goto cleanup;
    }
    if( verbose != 0 )
        mbedtls_printf( "passed\n" );
    MBEDTLS_MPI_CHK( mbedtls_mpi_exp_mod( &X, &A, &E, &N, NULL ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_read_string( &U, 16,
        "36E139AEA55215609D2816998ED020BB" \
        "BD96C37890F65171D948E9BC7CBAA4D9" \
        "325D24D6A3C12710F10A09FA08AB87" ) );
    if( verbose != 0 )
        mbedtls_printf( "  MPI test #3 (exp_mod): " );
    if( mbedtls_mpi_cmp_mpi( &X, &U ) != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "failed\n" );
        ret = 1;
        goto cleanup;
    }
    if( verbose != 0 )
        mbedtls_printf( "passed\n" );
    MBEDTLS_MPI_CHK( mbedtls_mpi_inv_mod( &X, &A, &N ) );
    MBEDTLS_MPI_CHK( mbedtls_mpi_read_string( &U, 16,
        "003A0AAEDD7E784FC07D8F9EC6E3BFD5" \
        "C3DBA76456363A10869622EAC2DD84EC" \
        "C5B8A74DAC4D09E03B5E0BE779F2DF61" ) );
    if( verbose != 0 )
        mbedtls_printf( "  MPI test #4 (inv_mod): " );
    if( mbedtls_mpi_cmp_mpi( &X, &U ) != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "failed\n" );
        ret = 1;
        goto cleanup;
    }
    if( verbose != 0 )
        mbedtls_printf( "passed\n" );
    if( verbose != 0 )
        mbedtls_printf( "  MPI test #5 (simple gcd): " );
    for( i = 0; i < GCD_PAIR_COUNT; i++ )
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &X, gcd_pairs[i][0] ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_lset( &Y, gcd_pairs[i][1] ) );
        MBEDTLS_MPI_CHK( mbedtls_mpi_gcd( &A, &X, &Y ) );
        if( mbedtls_mpi_cmp_int( &A, gcd_pairs[i][2] ) != 0 )
        {
            if( verbose != 0 )
                mbedtls_printf( "failed at %d\n", i );
            ret = 1;
            goto cleanup;
        }
    }
    if( verbose != 0 )
        mbedtls_printf( "passed\n" );
cleanup:
    if( ret != 0 && verbose != 0 )
        mbedtls_printf( "Unexpected error, return code = %08X\n", (unsigned int) ret );
    mbedtls_mpi_free( &A ); mbedtls_mpi_free( &E ); mbedtls_mpi_free( &N ); mbedtls_mpi_free( &X );
    mbedtls_mpi_free( &Y ); mbedtls_mpi_free( &U ); mbedtls_mpi_free( &V );
    if( verbose != 0 )
        mbedtls_printf( "\n" );
    return( ret );
}

#endif /* MBEDTLS_SELF_TEST */

#endif /* MBEDTLS_BIGNUM_C */
