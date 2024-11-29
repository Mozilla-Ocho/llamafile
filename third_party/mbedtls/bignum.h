#ifndef MBEDTLS_BIGNUM_H_
#define MBEDTLS_BIGNUM_H_
#include <libc/stdio/stdio.h>
#include "third_party/mbedtls/bignum_internal.h"
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/platform.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_MPI_FILE_IO_ERROR                     -0x0002  /*< An error occurred while reading from or writing to a file. */
#define MBEDTLS_ERR_MPI_BAD_INPUT_DATA                    -0x0004  /*< Bad input parameters to function. */
#define MBEDTLS_ERR_MPI_INVALID_CHARACTER                 -0x0006  /*< There is an invalid character in the digit string. */
#define MBEDTLS_ERR_MPI_BUFFER_TOO_SMALL                  -0x0008  /*< The buffer is too small to write to. */
#define MBEDTLS_ERR_MPI_NEGATIVE_VALUE                    -0x000A  /*< The input arguments are negative or result in illegal output. */
#define MBEDTLS_ERR_MPI_DIVISION_BY_ZERO                  -0x000C  /*< The input argument for division is zero, which is not allowed. */
#define MBEDTLS_ERR_MPI_NOT_ACCEPTABLE                    -0x000E  /*< The input arguments are not acceptable. */
#define MBEDTLS_ERR_MPI_ALLOC_FAILED                      -0x0010  /*< Memory allocation failed. */
#define MBEDTLS_MPI_CHK(f)       \
    do                           \
    {                            \
        if( ( ret = (f) ) )      \
            goto cleanup;        \
    } while( 0 )

/*
 * Maximum size MPIs are allowed to grow to in number of limbs.
 */
#define MBEDTLS_MPI_MAX_LIMBS                             10000

#if !defined(MBEDTLS_MPI_WINDOW_SIZE)
/*
 * Maximum window size used for modular exponentiation. Default: 6
 * Minimum value: 1. Maximum value: 6.
 *
 * Result is an array of ( 2 ** MBEDTLS_MPI_WINDOW_SIZE ) MPIs used
 * for the sliding window calculation. (So 64 by default)
 *
 * Reduction in size, reduces speed.
 */
#define MBEDTLS_MPI_WINDOW_SIZE                           6        /*< Maximum window size used. */
#endif /* !MBEDTLS_MPI_WINDOW_SIZE */

#if !defined(MBEDTLS_MPI_MAX_SIZE)
/*
 * Maximum size of MPIs allowed in bits and bytes for user-MPIs.
 * ( Default: 512 bytes => 4096 bits, Maximum tested: 2048 bytes => 16384 bits )
 *
 * Note: Calculations can temporarily result in larger MPIs. So the number
 * of limbs required (MBEDTLS_MPI_MAX_LIMBS) is higher.
 */
#define MBEDTLS_MPI_MAX_SIZE                              1024     /*< Maximum number of bytes for usable MPIs. */
#endif /* !MBEDTLS_MPI_MAX_SIZE */

#define MBEDTLS_MPI_MAX_BITS                              ( 8 * MBEDTLS_MPI_MAX_SIZE )    /*< Maximum number of bits for usable MPIs. */

/*
 * When reading from files with mbedtls_mpi_read_file() and writing to files with
 * mbedtls_mpi_write_file() the buffer should have space
 * for a (short) label, the MPI (in the provided radix), the newline
 * characters and the '\0'.
 *
 * By default we assume at least a 10 char label, a minimum radix of 10
 * (decimal) and a maximum of 4096 bit numbers (1234 decimal chars).
 * Autosized at compile time for at least a 10 char label, a minimum radix
 * of 10 (decimal) for a number of MBEDTLS_MPI_MAX_BITS size.
 *
 * This used to be statically sized to 1250 for a maximum of 4096 bit
 * numbers (1234 decimal chars).
 *
 * Calculate using the formula:
 *  MBEDTLS_MPI_RW_BUFFER_SIZE = ceil(MBEDTLS_MPI_MAX_BITS / ln(10) * ln(2)) +
 *                                LabelSize + 6
 */
#define MBEDTLS_MPI_MAX_BITS_SCALE100          ( 100 * MBEDTLS_MPI_MAX_BITS )
#define MBEDTLS_LN_2_DIV_LN_10_SCALE100                 332
#define MBEDTLS_MPI_RW_BUFFER_SIZE             ( ((MBEDTLS_MPI_MAX_BITS_SCALE100 + MBEDTLS_LN_2_DIV_LN_10_SCALE100 - 1) / MBEDTLS_LN_2_DIV_LN_10_SCALE100) + 10 + 6 )

typedef int64_t mbedtls_mpi_sint;
typedef uint64_t mbedtls_mpi_uint;

/**
 * \brief          MPI structure
 */
typedef struct mbedtls_mpi
{
    int s;                        /*!<  Sign: -1 if the mpi is negative, 1 otherwise */
    unsigned n;                   /*!<  total # of limbs  */
    mbedtls_mpi_uint *p;          /*!<  pointer to limbs  */
}
mbedtls_mpi forcealign(16);

/**
 * \brief Flags for mbedtls_mpi_gen_prime()
 *
 * Each of these flags is a constraint on the result X returned by
 * mbedtls_mpi_gen_prime().
 */
typedef enum {
    MBEDTLS_MPI_GEN_PRIME_FLAG_DH =      0x0001, /*< (X-1)/2 is prime too */
    MBEDTLS_MPI_GEN_PRIME_FLAG_LOW_ERR = 0x0002, /*< lower error rate from 2<sup>-80</sup> to 2<sup>-128</sup> */
} mbedtls_mpi_gen_prime_flag_t;

int mbedtls_mpi_add_abs( mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_add_int( mbedtls_mpi *, const mbedtls_mpi *, mbedtls_mpi_sint );
int mbedtls_mpi_add_mpi( mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_cmp_abs( const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_cmp_int( const mbedtls_mpi *, mbedtls_mpi_sint );
int mbedtls_mpi_cmp_mpi( const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_copy( mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_div_int( mbedtls_mpi *, mbedtls_mpi *, const mbedtls_mpi *, mbedtls_mpi_sint );
int mbedtls_mpi_div_mpi( mbedtls_mpi *, mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_exp_mod( mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi *, mbedtls_mpi * );
int mbedtls_mpi_fill_random( mbedtls_mpi *, size_t, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_mpi_gcd( mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_gen_prime( mbedtls_mpi *, size_t, int, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_mpi_get_bit( const mbedtls_mpi *, size_t );
int mbedtls_mpi_grow( mbedtls_mpi *, size_t );
int mbedtls_mpi_inv_mod( mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_is_prime_ext( const mbedtls_mpi *, int, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_mpi_lset( mbedtls_mpi *, mbedtls_mpi_sint );
int mbedtls_mpi_lt_mpi_ct( const mbedtls_mpi *, const mbedtls_mpi *, unsigned * );
int mbedtls_mpi_mod_int( mbedtls_mpi_uint *, const mbedtls_mpi *, mbedtls_mpi_sint );
int mbedtls_mpi_mod_mpi( mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_mul_int( mbedtls_mpi *, const mbedtls_mpi *, mbedtls_mpi_uint );
int mbedtls_mpi_mul_mpi( mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_read_binary( mbedtls_mpi *, const unsigned char *, size_t );
int mbedtls_mpi_read_binary_le( mbedtls_mpi *, const unsigned char *, size_t );
int mbedtls_mpi_read_file( mbedtls_mpi *, int, FILE * );
int mbedtls_mpi_read_string( mbedtls_mpi *, int, const char * );
int mbedtls_mpi_resize( mbedtls_mpi *, size_t );
int mbedtls_mpi_safe_cond_assign( mbedtls_mpi *, const mbedtls_mpi *, unsigned char );
int mbedtls_mpi_safe_cond_swap( mbedtls_mpi *, mbedtls_mpi *, unsigned char );
int mbedtls_mpi_self_test( int );
int mbedtls_mpi_set_bit( mbedtls_mpi *, size_t, unsigned char );
int mbedtls_mpi_shift_l( mbedtls_mpi *, size_t );
int mbedtls_mpi_shift_r( mbedtls_mpi *, size_t );
int mbedtls_mpi_shrink( mbedtls_mpi *, size_t );
int mbedtls_mpi_sub_abs( mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_sub_int( mbedtls_mpi *, const mbedtls_mpi *, mbedtls_mpi_sint );
int mbedtls_mpi_sub_mpi( mbedtls_mpi *, const mbedtls_mpi *, const mbedtls_mpi * );
int mbedtls_mpi_write_binary( const mbedtls_mpi *, unsigned char *, size_t );
int mbedtls_mpi_write_binary_le( const mbedtls_mpi *, unsigned char *, size_t );
int mbedtls_mpi_write_file( const char *, const mbedtls_mpi *, int, FILE * );
int mbedtls_mpi_write_string( const mbedtls_mpi *, int, char *, size_t, size_t * );
size_t mbedtls_mpi_bitlen( const mbedtls_mpi * );
size_t mbedtls_mpi_lsb( const mbedtls_mpi * );
size_t mbedtls_mpi_size( const mbedtls_mpi * );
void mbedtls_mpi_free( mbedtls_mpi * );
void mbedtls_mpi_swap( mbedtls_mpi *, mbedtls_mpi * );

/**
 * \brief           Initialize an MPI context.
 *
 *                  This makes the MPI ready to be set or freed,
 *                  but does not define a value for the MPI.
 *
 * \param X         The MPI context to initialize. This must not be \c NULL.
 */
forceinline void mbedtls_mpi_init(mbedtls_mpi *X)
{
    MBEDTLS_INTERNAL_VALIDATE(X);
    typedef int mbedtls_mpi_lol
            __attribute__((__vector_size__(16), __aligned__(16)));
    *(mbedtls_mpi_lol *)X = (mbedtls_mpi_lol){1};
}

forceinline size_t mbedtls_mpi_limbs(const mbedtls_mpi *X) {
  size_t i;
  for (i = X->n; i; i--) {
    if (X->p[i - 1]) {
      break;
    }
  }
  return i;
}

static inline bool mbedtls_mpi_is_zero(const mbedtls_mpi *X)
{
    if (X->n && *X->p) return false;
    if (!mbedtls_mpi_limbs(X)) return true;
    return false;
}

static inline bool mbedtls_mpi_is_one(const mbedtls_mpi *X)
{
    if (!X->n || *X->p != 1 || X->s != 1) return false;
    return mbedtls_mpi_limbs(X) == 1;
}

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_BIGNUM_H_ */
