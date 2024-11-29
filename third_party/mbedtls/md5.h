#ifndef MBEDTLS_MD5_H_
#define MBEDTLS_MD5_H_
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/platform.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_MD5_HW_ACCEL_FAILED -0x002F  /*< MD5 hardware accelerator failed */

/**
 * \brief          MD5 context structure
 *
 * \warning        MD5 is considered a weak message digest and its use
 *                 constitutes a security risk. We recommend considering
 *                 stronger message digests instead.
 *
 */
typedef struct mbedtls_md5_context
{
    uint32_t total[2];          /*!< number of bytes processed  */
    uint32_t state[4];          /*!< intermediate digest state  */
    unsigned char buffer[64];   /*!< data block being processed */
}
mbedtls_md5_context;

void mbedtls_md5_clone( mbedtls_md5_context *, const mbedtls_md5_context * );
int mbedtls_md5_starts_ret( mbedtls_md5_context * );
int mbedtls_md5_update_ret( mbedtls_md5_context *, const unsigned char *, size_t );
int mbedtls_md5_finish_ret( mbedtls_md5_context *, unsigned char[16] );
int mbedtls_internal_md5_process( mbedtls_md5_context *, const unsigned char[64] );
int mbedtls_md5_ret( const void *, size_t, unsigned char[16] );
int mbedtls_md5_self_test( int );

/**
 * \brief          Initialize MD5 context
 *
 * \param ctx      MD5 context to be initialized
 *
 * \warning        MD5 is considered a weak message digest and its use
 *                 constitutes a security risk. We recommend considering
 *                 stronger message digests instead.
 */
static inline void mbedtls_md5_init( mbedtls_md5_context *ctx )
{
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_md5_context ) );
}

/**
 * \brief          Clear MD5 context
 *
 * \param ctx      MD5 context to be cleared
 *
 * \warning        MD5 is considered a weak message digest and its use
 *                 constitutes a security risk. We recommend considering
 *                 stronger message digests instead.
 */
static inline void mbedtls_md5_free( mbedtls_md5_context *ctx )
{
    if( !ctx ) return;
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_md5_context ) );
}

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_MD5_H_ */
