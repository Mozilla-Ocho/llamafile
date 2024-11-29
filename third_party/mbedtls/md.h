#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_MD_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_MD_H_
#include "third_party/mbedtls/config.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_MD_FEATURE_UNAVAILABLE                -0x5080  /*< The selected feature is not available. */
#define MBEDTLS_ERR_MD_BAD_INPUT_DATA                     -0x5100  /*< Bad input parameters to function. */
#define MBEDTLS_ERR_MD_ALLOC_FAILED                       -0x5180  /*< Failed to allocate memory. */
#define MBEDTLS_ERR_MD_FILE_IO_ERROR                      -0x5200  /*< Opening or reading of file failed. */

/* MBEDTLS_ERR_MD_HW_ACCEL_FAILED is deprecated and should not be used. */
#define MBEDTLS_ERR_MD_HW_ACCEL_FAILED                    -0x5280  /*< MD hardware accelerator failed. */

/**
 * \brief     Supported message digests.
 *
 * \warning   MD2, MD4, MD5 and SHA-1 are considered weak message digests and
 *            their use constitutes a security risk. We recommend considering
 *            stronger message digests instead.
 */
typedef enum {
    MBEDTLS_MD_NONE=0,     /*< None. */
    MBEDTLS_MD_SHA1,       /*< The SHA-1 message digest. */
    MBEDTLS_MD_SHA224,     /*< The SHA-224 message digest. */
    MBEDTLS_MD_SHA256,     /*< The SHA-256 message digest. */
    MBEDTLS_MD_SHA384,     /*< The SHA-384 message digest. */
    MBEDTLS_MD_SHA512,     /*< The SHA-512 message digest. */
    MBEDTLS_MD_BLAKE2B256, /*< The BLAKE2B256 message digest. */
    MBEDTLS_MD_RIPEMD160,  /*< The RIPEMD-160 message digest. */
    MBEDTLS_MD_MD2,        /*< The MD2 message digest. */
    MBEDTLS_MD_MD4,        /*< The MD4 message digest. */
    MBEDTLS_MD_MD5,        /*< The MD5 message digest. */
} mbedtls_md_type_t;

#if defined(MBEDTLS_SHA512_C)
#define MBEDTLS_MD_MAX_SIZE         64  /* longest known is SHA512 */
#else
#define MBEDTLS_MD_MAX_SIZE         32  /* longest known is SHA256 or less */
#endif

#if defined(MBEDTLS_SHA512_C)
#define MBEDTLS_MD_MAX_BLOCK_SIZE         128
#else
#define MBEDTLS_MD_MAX_BLOCK_SIZE         64
#endif

/**
 * Message digest information.
 * Allows message digest functions to be called in a generic way.
 */
typedef struct mbedtls_md_info_t {
    const char *name;         /** Name of the message digest */
    mbedtls_md_type_t type;   /** Digest identifier */
    unsigned char size;       /** Output length of the digest function in bytes */
    unsigned char block_size; /** Block length of the digest function in bytes */
    int (*f_starts)(void *);
    int (*f_update)(void *, const void *, size_t);
    int (*f_process)(void *, const void *);
    int (*f_finish)(void *, void *);
    int (*f_md)(const void *, size_t, unsigned char *);
} mbedtls_md_info_t;

/**
 * The generic message-digest context.
 */
typedef struct mbedtls_md_context_t {
    const mbedtls_md_info_t *md_info; /** Information about the associated message digest. */
    void *md_ctx;                     /** The digest-specific context. */
    void *hmac_ctx;                   /** The HMAC part of the context. */
} mbedtls_md_context_t;

const uint8_t *mbedtls_md_list( void );
const mbedtls_md_info_t *mbedtls_md_info_from_string( const char * );
const mbedtls_md_info_t *mbedtls_md_info_from_type( mbedtls_md_type_t );
int mbedtls_md_clone( mbedtls_md_context_t *, const mbedtls_md_context_t * );
int mbedtls_md_setup( mbedtls_md_context_t *, const mbedtls_md_info_t *, int );
void mbedtls_md_free( mbedtls_md_context_t * );
void mbedtls_md_init( mbedtls_md_context_t * );

/**
 * \brief           This function extracts the message-digest size from the
 *                  message-digest information structure.
 *
 * \param md_info   The information structure of the message-digest algorithm
 *                  to use.
 *
 * \return          The size of the message-digest output in Bytes.
 */
forceinline unsigned char mbedtls_md_get_size( const mbedtls_md_info_t *md_info )
{
    if( !md_info )
        return( 0 );
    return md_info->size;
}

/**
 * \brief           This function extracts the message-digest size from the
 *                  message-digest information structure.
 *
 * \param md_info   The information structure of the message-digest algorithm
 *                  to use.
 *
 * \return          The size of the message-digest output in Bytes.
 */
forceinline unsigned char mbedtls_md_get_block_size( const mbedtls_md_info_t *md_info )
{
    if( !md_info )
        return( 0 );
    return md_info->block_size;
}

/**
 * \brief           This function extracts the message-digest type from the
 *                  message-digest information structure.
 *
 * \param md_info   The information structure of the message-digest algorithm
 *                  to use.
 *
 * \return          The type of the message digest.
 */
forceinline mbedtls_md_type_t mbedtls_md_get_type( const mbedtls_md_info_t *md_info )
{
    if( !md_info )
        return( MBEDTLS_MD_NONE );
    return md_info->type;
}

/**
 * \brief           This function extracts the message-digest name from the
 *                  message-digest information structure.
 *
 * \param md_info   The information structure of the message-digest algorithm
 *                  to use.
 *
 * \return          The name of the message digest.
 */
forceinline const char *mbedtls_md_get_name( const mbedtls_md_info_t *md_info )
{
    if( !md_info )
        return( NULL );
    return md_info->name;
}

/**
 * \brief           This function starts a message-digest computation.
 *
 *                  You must call this function after setting up the context
 *                  with mbedtls_md_setup(), and before passing data with
 *                  mbedtls_md_update().
 *
 * \param ctx       The generic message-digest context.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                  failure.
 */
forceinline int mbedtls_md_starts( mbedtls_md_context_t *ctx )
{
    if( !ctx || !ctx->md_info )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    return ctx->md_info->f_starts( ctx->md_ctx );
}

/**
 * \brief           This function feeds an input buffer into an ongoing
 *                  message-digest computation.
 *
 *                  You must call mbedtls_md_starts() before calling this
 *                  function. You may call this function multiple times.
 *                  Afterwards, call mbedtls_md_finish().
 *
 * \param ctx       The generic message-digest context.
 * \param input     The buffer holding the input data.
 * \param ilen      The length of the input data.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                  failure.
 */
forceinline int mbedtls_md_update( mbedtls_md_context_t *ctx, 
                                   const unsigned char *input, size_t ilen )
{
    if( !ctx || !ctx->md_info )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    return ctx->md_info->f_update( ctx->md_ctx, input, ilen );
}

/**
 * \brief           This function finishes the digest operation,
 *                  and writes the result to the output buffer.
 *
 *                  Call this function after a call to mbedtls_md_starts(),
 *                  followed by any number of calls to mbedtls_md_update().
 *                  Afterwards, you may either clear the context with
 *                  mbedtls_md_free(), or call mbedtls_md_starts() to reuse
 *                  the context for another digest operation with the same
 *                  algorithm.
 *
 * \param ctx       The generic message-digest context.
 * \param output    The buffer for the generic message-digest checksum result.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                  failure.
 */
forceinline int mbedtls_md_finish( mbedtls_md_context_t *ctx, 
                                   unsigned char *output )
{
    if( !ctx || !ctx->md_info )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    return ctx->md_info->f_finish( ctx->md_ctx, output );
}

/**
 * \brief          This function calculates the message-digest of a buffer,
 *                 with respect to a configurable message-digest algorithm
 *                 in a single call.
 *
 *                 The result is calculated as
 *                 Output = message_digest(input buffer).
 *
 * \param md_info  The information structure of the message-digest algorithm
 *                 to use.
 * \param input    The buffer holding the data.
 * \param ilen     The length of the input data.
 * \param output   The generic message-digest checksum result.
 *
 * \return         \c 0 on success.
 * \return         #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                 failure.
 */
forceinline int mbedtls_md( const mbedtls_md_info_t *md_info, 
                            const unsigned char *input, size_t ilen,
                            unsigned char *output )
{
    if( !md_info )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    return md_info->f_md(input, ilen, output );
}

int mbedtls_md_file( const mbedtls_md_info_t *md_info, const char *path,
                     unsigned char *output );

int mbedtls_md_hmac_starts( mbedtls_md_context_t *ctx, const unsigned char *key,
                            size_t keylen );

/**
 * \brief           This function feeds an input buffer into an ongoing HMAC
 *                  computation.
 *
 *                  Call mbedtls_md_hmac_starts() or mbedtls_md_hmac_reset()
 *                  before calling this function.
 *                  You may call this function multiple times to pass the
 *                  input piecewise.
 *                  Afterwards, call mbedtls_md_hmac_finish().
 *
 * \param ctx       The message digest context containing an embedded HMAC
 *                  context.
 * \param input     The buffer holding the input data.
 * \param ilen      The length of the input data.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                  failure.
 */
forceinline int mbedtls_md_hmac_update( mbedtls_md_context_t *ctx, 
                                        const unsigned char *input, 
                                        size_t ilen )
{
    if( ctx == NULL || ctx->md_info == NULL || ctx->hmac_ctx == NULL )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    return( mbedtls_md_update( ctx, input, ilen ) );
}

int mbedtls_md_hmac_finish( mbedtls_md_context_t *, unsigned char *);
int mbedtls_md_hmac_reset( mbedtls_md_context_t * );
int mbedtls_md_hmac( const mbedtls_md_info_t *, const unsigned char *, size_t, const unsigned char *, size_t, unsigned char * );

forceinline int mbedtls_md_process( mbedtls_md_context_t *ctx, const unsigned char *data )
{
    if( !ctx || !ctx->md_info )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    return ctx->md_info->f_process( ctx->md_ctx, data );
}

const char *mbedtls_md_type_name(mbedtls_md_type_t);

extern const mbedtls_md_info_t mbedtls_md2_info;
extern const mbedtls_md_info_t mbedtls_md4_info;
extern const mbedtls_md_info_t mbedtls_md5_info;
extern const mbedtls_md_info_t mbedtls_sha1_info;
extern const mbedtls_md_info_t mbedtls_sha224_info;
extern const mbedtls_md_info_t mbedtls_sha256_info;
extern const mbedtls_md_info_t mbedtls_sha384_info;
extern const mbedtls_md_info_t mbedtls_sha512_info;
extern const mbedtls_md_info_t mbedtls_blake2b256_info;

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_MD_H_ */
