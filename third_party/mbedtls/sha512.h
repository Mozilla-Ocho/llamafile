#ifndef MBEDTLS_SHA512_H_
#define MBEDTLS_SHA512_H_
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/platform.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_SHA512_HW_ACCEL_FAILED -0x0039  /*< SHA-512 hardware accelerator failed */
#define MBEDTLS_ERR_SHA512_BAD_INPUT_DATA  -0x0075  /*< SHA-512 input data was malformed. */

/**
 * \brief          The SHA-512 context structure.
 *
 *                 The structure is used both for SHA-384 and for SHA-512
 *                 checksum calculations. The choice between these two is
 *                 made in the call to mbedtls_sha512_starts_ret().
 */
typedef struct mbedtls_sha512_context
{
    uint64_t state[8];          /*!< The intermediate digest state. */
    uint64_t total[2];          /*!< The number of Bytes processed. */
    unsigned char buffer[128];  /*!< The data block being processed. */
#if !defined(MBEDTLS_SHA512_NO_SHA384)
    int is384;                  /*!< Determines which function to use:
                                     0: Use SHA-512, or 1: Use SHA-384. */
#endif
}
mbedtls_sha512_context;

void mbedtls_sha512_clone( mbedtls_sha512_context *, const mbedtls_sha512_context * );
int mbedtls_sha512_starts_ret( mbedtls_sha512_context *, int );
int mbedtls_sha512_update_ret( mbedtls_sha512_context *, const unsigned char *, size_t );
int mbedtls_sha512_finish_ret( mbedtls_sha512_context *, unsigned char[64] );
int mbedtls_internal_sha512_process( mbedtls_sha512_context *, const unsigned char[128] );
int mbedtls_sha512_ret( const void *, size_t, unsigned char[64], int );
int mbedtls_sha512_ret_384( const void *, size_t, unsigned char * );
int mbedtls_sha512_ret_512( const void *, size_t, unsigned char * );
int mbedtls_sha512_self_test( int );

/**
 * \brief          This function initializes a SHA-512 context.
 *
 * \param ctx      The SHA-512 context to initialize. This must
 *                 not be \c NULL.
 */
static inline void mbedtls_sha512_init( mbedtls_sha512_context *ctx )
{
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_sha512_context ) );
}

/**
 * \brief          This function clears a SHA-512 context.
 *
 * \param ctx      The SHA-512 context to clear. This may be \c NULL,
 *                 in which case this function does nothing. If it
 *                 is not \c NULL, it must point to an initialized
 *                 SHA-512 context.
 */
static inline void mbedtls_sha512_free( mbedtls_sha512_context *ctx )
{
    if( !ctx ) return;
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_sha512_context ) );
}

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_SHA512_H_ */
