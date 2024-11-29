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
#include <libc/mem/mem.h>
#include <libc/stdio/stdio.h>
#include <libc/str/blake2.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/md.h"
#include "third_party/mbedtls/md5.h"
#include "third_party/mbedtls/platform.h"
#include "third_party/mbedtls/sha1.h"
#include "third_party/mbedtls/sha256.h"
#include "third_party/mbedtls/sha512.h"
__static_yoink("mbedtls_notice");

/**
 * \file md.c
 *
 * \brief Generic message digest wrapper for mbed TLS
 *
 * \author Adriaan de Jong <dejong@fox-it.com>
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#if defined(MBEDTLS_MD_C)

#define CHECK(f)                 \
    do                           \
    {                            \
        if( ( ret = (f) ) )      \
            goto cleanup;        \
    } while( 0 )

/*
 * Reminder: update profiles in x509_crt.c when adding a new hash!
 */
static const uint8_t supported_digests[] = {
        MBEDTLS_MD_BLAKE2B256,
#if defined(MBEDTLS_SHA512_C)
        MBEDTLS_MD_SHA512,
#if !defined(MBEDTLS_SHA512_NO_SHA384)
        MBEDTLS_MD_SHA384,
#endif
#endif
#if defined(MBEDTLS_SHA256_C)
        MBEDTLS_MD_SHA256,
        MBEDTLS_MD_SHA224,
#endif
#if defined(MBEDTLS_SHA1_C)
        MBEDTLS_MD_SHA1,
#endif
#if defined(MBEDTLS_MD5_C)
        MBEDTLS_MD_MD5,
#endif
#if defined(MBEDTLS_MD4_C)
        MBEDTLS_MD_MD4,
#endif
#if defined(MBEDTLS_MD2_C)
        MBEDTLS_MD_MD2,
#endif
        MBEDTLS_MD_NONE
};

/**
 * \brief           This function returns the list of digests supported by the
 *                  generic digest module.
 *
 * \note            The list starts with the strongest available hashes.
 *
 * \return          A statically allocated array of digests. Each element
 *                  in the returned list is an integer belonging to the
 *                  message-digest enumeration #mbedtls_md_type_t.
 *                  The last entry is 0.
 */
const uint8_t *mbedtls_md_list( void )
{
    return( supported_digests );
}

/**
 * \brief           This function returns the message-digest information
 *                  associated with the given digest name.
 *
 * \param md_name   The name of the digest to search for.
 *
 * \return          The message-digest information associated with \p md_name.
 * \return          NULL if the associated message-digest information is not found.
 */
const mbedtls_md_info_t *mbedtls_md_info_from_string( const char *md_name )
{
    if( NULL == md_name )
        return( NULL );
    /* Get the appropriate digest information */
#if defined(MBEDTLS_MD2_C)
    if( !strcasecmp( "MD2", md_name ) )
        return mbedtls_md_info_from_type( MBEDTLS_MD_MD2 );
#endif
#if defined(MBEDTLS_MD4_C)
    if( !strcasecmp( "MD4", md_name ) )
        return mbedtls_md_info_from_type( MBEDTLS_MD_MD4 );
#endif
#if defined(MBEDTLS_MD5_C)
    if( !strcasecmp( "MD5", md_name ) )
        return mbedtls_md_info_from_type( MBEDTLS_MD_MD5 );
#endif
#if defined(MBEDTLS_SHA1_C)
    if( !strcasecmp( "SHA1", md_name ) || !strcasecmp( "SHA", md_name ) )
        return mbedtls_md_info_from_type( MBEDTLS_MD_SHA1 );
#endif
#if defined(MBEDTLS_SHA256_C)
    if( !strcasecmp( "SHA224", md_name ) )
        return mbedtls_md_info_from_type( MBEDTLS_MD_SHA224 );
    if( !strcasecmp( "SHA256", md_name ) )
        return mbedtls_md_info_from_type( MBEDTLS_MD_SHA256 );
#endif
#if defined(MBEDTLS_SHA512_C)
#if !defined(MBEDTLS_SHA512_NO_SHA384)
    if( !strcasecmp( "SHA384", md_name ) )
        return mbedtls_md_info_from_type( MBEDTLS_MD_SHA384 );
#endif
    if( !strcasecmp( "SHA512", md_name ) )
        return mbedtls_md_info_from_type( MBEDTLS_MD_SHA512 );
#endif
    if( !strcasecmp( "BLAKE2B256", md_name ) )
        return mbedtls_md_info_from_type( MBEDTLS_MD_BLAKE2B256 );
    return( NULL );
}

/**
 * \brief           This function returns the message-digest information
 *                  associated with the given digest type.
 *
 * \param md_type   The type of digest to search for.
 *
 * \return          The message-digest information associated with \p md_type.
 * \return          NULL if the associated message-digest information is not found.
 */
const mbedtls_md_info_t *mbedtls_md_info_from_type( mbedtls_md_type_t md_type )
{
    switch( md_type )
    {
#if defined(MBEDTLS_MD2_C)
        case MBEDTLS_MD_MD2:
            return( &mbedtls_md2_info );
#endif
#if defined(MBEDTLS_MD4_C)
        case MBEDTLS_MD_MD4:
            return( &mbedtls_md4_info );
#endif
#if defined(MBEDTLS_MD5_C)
        case MBEDTLS_MD_MD5:
            return( &mbedtls_md5_info );
#endif
#if defined(MBEDTLS_SHA1_C)
        case MBEDTLS_MD_SHA1:
            return( &mbedtls_sha1_info );
#endif
#if defined(MBEDTLS_SHA256_C)
        case MBEDTLS_MD_SHA224:
            return( &mbedtls_sha224_info );
        case MBEDTLS_MD_SHA256:
            return( &mbedtls_sha256_info );
#endif
#if defined(MBEDTLS_SHA512_C)
#if !defined(MBEDTLS_SHA512_NO_SHA384)
        case MBEDTLS_MD_SHA384:
            return( &mbedtls_sha384_info );
#endif
        case MBEDTLS_MD_SHA512:
            return( &mbedtls_sha512_info );
#endif
        case MBEDTLS_MD_BLAKE2B256:
            return( &mbedtls_blake2b256_info );
        default:
            return( NULL );
    }
}

static int16_t GetMdContextSize(mbedtls_md_type_t t)
{
    switch( t )
    {
#if defined(MBEDTLS_MD2_C)
        case MBEDTLS_MD_MD2:
            return sizeof(mbedtls_md2_context);
#endif
#if defined(MBEDTLS_MD4_C)
        case MBEDTLS_MD_MD4:
            return sizeof(mbedtls_md4_context);
#endif
#if defined(MBEDTLS_MD5_C)
        case MBEDTLS_MD_MD5:
            return sizeof(mbedtls_md5_context);
#endif
#if defined(MBEDTLS_SHA1_C)
        case MBEDTLS_MD_SHA1:
            return sizeof(mbedtls_sha1_context);
#endif
#if defined(MBEDTLS_SHA256_C)
        case MBEDTLS_MD_SHA224:
        case MBEDTLS_MD_SHA256:
            return sizeof(mbedtls_sha256_context);
#endif
#if defined(MBEDTLS_SHA512_C)
#if !defined(MBEDTLS_SHA512_NO_SHA384)
        case MBEDTLS_MD_SHA384:
#endif
        case MBEDTLS_MD_SHA512:
            return sizeof(mbedtls_sha512_context);
#endif
        case MBEDTLS_MD_BLAKE2B256:
            return sizeof(struct Blake2b);
        default:
            return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    }
}

/**
 * \brief           This function initializes a message-digest context without
 *                  binding it to a particular message-digest algorithm.
 *
 *                  This function should always be called first. It prepares the
 *                  context for mbedtls_md_setup() for binding it to a
 *                  message-digest algorithm.
 */
void mbedtls_md_init( mbedtls_md_context_t *ctx )
{
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_md_context_t ) );
}

/**
 * \brief           This function clears the internal structure of \p ctx and
 *                  frees any embedded internal structure, but does not free
 *                  \p ctx itself.
 *
 *                  If you have called mbedtls_md_setup() on \p ctx, you must
 *                  call mbedtls_md_free() when you are no longer using the
 *                  context.
 *                  Calling this function if you have previously
 *                  called mbedtls_md_init() and nothing else is optional.
 *                  You must not call this function if you have not called
 *                  mbedtls_md_init().
 */
void mbedtls_md_free( mbedtls_md_context_t *ctx )
{
    int16_t csize;
    if( !ctx || !ctx->md_info )
        return;
    if( ctx->md_ctx )
    {
        if ( ( csize = GetMdContextSize( ctx->md_info->type ) ) > 0 )
            mbedtls_platform_zeroize( ctx->md_ctx, csize );
        mbedtls_free( ctx->md_ctx );
    }
    if( ctx->hmac_ctx )
    {
        mbedtls_platform_zeroize( ctx->hmac_ctx,
                                  2 * ctx->md_info->block_size );
        mbedtls_free( ctx->hmac_ctx );
    }
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_md_context_t ) );
}

/**
 * \brief           This function clones the state of an message-digest
 *                  context.
 *
 * \note            You must call mbedtls_md_setup() on \c dst before calling
 *                  this function.
 *
 * \note            The two contexts must have the same type,
 *                  for example, both are SHA-256.
 *
 * \warning         This function clones the message-digest state, not the
 *                  HMAC state.
 *
 * \param dst       The destination context.
 * \param src       The context to be cloned.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification failure.
 */
int mbedtls_md_clone( mbedtls_md_context_t *dst,
                      const mbedtls_md_context_t *src )
{
    int16_t csize;
    if( !dst || !dst->md_info ||
        !src || !src->md_info ||
        dst->md_info != src->md_info ||
        ( csize = GetMdContextSize( src->md_info->type ) ) < 0)
    {
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    }
    memcpy( dst->md_ctx, src->md_ctx, csize );
    return( 0 );
}

#define ALLOC( type )                                                   \
    do {                                                                \
        ctx->md_ctx = mbedtls_calloc( 1, sizeof( mbedtls_##type##_context ) ); \
        if( !ctx->md_ctx )                                              \
            return( MBEDTLS_ERR_MD_ALLOC_FAILED );                      \
    }                                                                   \
    while( 0 )

/**
 * \brief           This function selects the message digest algorithm to use,
 *                  and allocates internal structures.
 *
 *                  It should be called after mbedtls_md_init() or
 *                  mbedtls_md_free(). Makes it necessary to call
 *                  mbedtls_md_free() later.
 *
 * \param ctx       The context to set up.
 * \param md_info   The information structure of the message-digest algorithm
 *                  to use.
 * \param hmac      Defines if HMAC is used. 0: HMAC is not used (saves some memory),
 *                  or non-zero: HMAC is used with this context.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                  failure.
 * \return          #MBEDTLS_ERR_MD_ALLOC_FAILED on memory-allocation failure.
 */
int mbedtls_md_setup( mbedtls_md_context_t *ctx, const mbedtls_md_info_t *md_info, int hmac )
{
    int16_t csize;
    if( !md_info || !ctx )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    ctx->md_info = md_info;
    ctx->md_ctx = NULL;
    ctx->hmac_ctx = NULL;
    if ((csize = GetMdContextSize(md_info->type)) < 0) 
        return( csize );
    if( !( ctx->md_ctx = mbedtls_calloc( 1, csize ) ) )
        return( MBEDTLS_ERR_MD_ALLOC_FAILED );
    if( hmac )
    {
        ctx->hmac_ctx = mbedtls_calloc( 2, md_info->block_size );
        if( !ctx->hmac_ctx )
        {
            mbedtls_md_free( ctx );
            return( MBEDTLS_ERR_MD_ALLOC_FAILED );
        }
    }
    return( 0 );
}

/**
 * \brief          This function calculates the message-digest checksum
 *                 result of the contents of the provided file.
 *
 *                 The result is calculated as
 *                 Output = message_digest(file contents).
 *
 * \param md_info  The information structure of the message-digest algorithm
 *                 to use.
 * \param path     The input file name.
 * \param output   The generic message-digest checksum result.
 *
 * \return         \c 0 on success.
 * \return         #MBEDTLS_ERR_MD_FILE_IO_ERROR on an I/O error accessing
 *                 the file pointed by \p path.
 * \return         #MBEDTLS_ERR_MD_BAD_INPUT_DATA if \p md_info was NULL.
 */
int mbedtls_md_file( const mbedtls_md_info_t *md_info, const char *path, unsigned char *output )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    FILE *f;
    size_t n;
    mbedtls_md_context_t ctx;
    unsigned char buf[1024];
    if( !md_info )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    if( !( f = fopen( path, "rb" ) ) )
        return( MBEDTLS_ERR_MD_FILE_IO_ERROR );
    mbedtls_md_init( &ctx );
    CHECK( mbedtls_md_setup( &ctx, md_info, 0 ) );
    CHECK( mbedtls_md_starts( &ctx ) );
    while( ( n = fread( buf, 1, sizeof( buf ), f ) ) > 0 )
        CHECK( mbedtls_md_update( &ctx, buf, n ) );
    if( ferror( f ) )
        ret = MBEDTLS_ERR_MD_FILE_IO_ERROR;
    else
        ret = mbedtls_md_finish( &ctx, output );
cleanup:
    mbedtls_platform_zeroize( buf, sizeof( buf ) );
    mbedtls_md_free( &ctx );
    fclose( f );
    return( ret );
}

/**
 * \brief           This function sets the HMAC key and prepares to
 *                  authenticate a new message.
 *
 *                  Call this function after mbedtls_md_setup(), to use
 *                  the MD context for an HMAC calculation, then call
 *                  mbedtls_md_hmac_update() to provide the input data, and
 *                  mbedtls_md_hmac_finish() to get the HMAC value.
 *
 * \param ctx       The message digest context containing an embedded HMAC
 *                  context.
 * \param key       The HMAC secret key.
 * \param keylen    The length of the HMAC key in Bytes.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                  failure.
 */
int mbedtls_md_hmac_starts( mbedtls_md_context_t *ctx, const unsigned char *key, size_t keylen )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char sum[MBEDTLS_MD_MAX_SIZE];
    unsigned char *ipad, *opad;
    size_t i;
    if( !ctx || !ctx->md_info || !ctx->hmac_ctx )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    if( keylen > (size_t) ctx->md_info->block_size )
    {
        CHECK( mbedtls_md_starts( ctx ) );
        CHECK( mbedtls_md_update( ctx, key, keylen ) );
        CHECK( mbedtls_md_finish( ctx, sum ) );
        keylen = ctx->md_info->size;
        key = sum;
    }
    ipad = (unsigned char *) ctx->hmac_ctx;
    opad = (unsigned char *) ctx->hmac_ctx + ctx->md_info->block_size;
    memset( ipad, 0x36, ctx->md_info->block_size );
    memset( opad, 0x5C, ctx->md_info->block_size );
    for( i = 0; i < keylen; i++ )
    {
        ipad[i] = (unsigned char)( ipad[i] ^ key[i] );
        opad[i] = (unsigned char)( opad[i] ^ key[i] );
    }
    CHECK( mbedtls_md_starts( ctx ) );
    CHECK( mbedtls_md_update( ctx, ipad, ctx->md_info->block_size ) );
cleanup:
    mbedtls_platform_zeroize( sum, sizeof( sum ) );
    return( ret );
}

/**
 * \brief           This function finishes the HMAC operation, and writes
 *                  the result to the output buffer.
 *
 *                  Call this function after mbedtls_md_hmac_starts() and
 *                  mbedtls_md_hmac_update() to get the HMAC value. Afterwards
 *                  you may either call mbedtls_md_free() to clear the context,
 *                  or call mbedtls_md_hmac_reset() to reuse the context with
 *                  the same HMAC key.
 *
 * \param ctx       The message digest context containing an embedded HMAC
 *                  context.
 * \param output    The generic HMAC checksum result.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                  failure.
 */
int mbedtls_md_hmac_finish( mbedtls_md_context_t *ctx, unsigned char *output )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char tmp[MBEDTLS_MD_MAX_SIZE];
    unsigned char *opad;
    if( !ctx || !ctx->md_info || !ctx->hmac_ctx )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    opad = (unsigned char *) ctx->hmac_ctx + ctx->md_info->block_size;
    CHECK( mbedtls_md_finish( ctx, tmp ) );
    CHECK( mbedtls_md_starts( ctx ) );
    CHECK( mbedtls_md_update( ctx, opad, ctx->md_info->block_size ) );
    CHECK( mbedtls_md_update( ctx, tmp, ctx->md_info->size ) );
    return( mbedtls_md_finish( ctx, output ) );
cleanup:
    return( ret );
}

/**
 * \brief           This function prepares to authenticate a new message with
 *                  the same key as the previous HMAC operation.
 *
 *                  You may call this function after mbedtls_md_hmac_finish().
 *                  Afterwards call mbedtls_md_hmac_update() to pass the new
 *                  input.
 *
 * \param ctx       The message digest context containing an embedded HMAC
 *                  context.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                  failure.
 */
int mbedtls_md_hmac_reset( mbedtls_md_context_t *ctx )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char *ipad;
    if( !ctx || !ctx->md_info || !ctx->hmac_ctx )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    ipad = (unsigned char *) ctx->hmac_ctx;
    if( ( ret = mbedtls_md_starts( ctx ) ) )
        return( ret );
    return( mbedtls_md_update( ctx, ipad, ctx->md_info->block_size ) );
}

/**
 * \brief          This function calculates the full generic HMAC
 *                 on the input buffer with the provided key.
 *
 *                 The function allocates the context, performs the
 *                 calculation, and frees the context.
 *
 *                 The HMAC result is calculated as
 *                 output = generic HMAC(hmac key, input buffer).
 *
 * \param md_info  The information structure of the message-digest algorithm
 *                 to use.
 * \param key      The HMAC secret key.
 * \param keylen   The length of the HMAC secret key in Bytes.
 * \param input    The buffer holding the input data.
 * \param ilen     The length of the input data.
 * \param output   The generic HMAC result.
 *
 * \return         \c 0 on success.
 * \return         #MBEDTLS_ERR_MD_BAD_INPUT_DATA on parameter-verification
 *                 failure.
 */
int mbedtls_md_hmac( const mbedtls_md_info_t *md_info,
                     const unsigned char *key, size_t keylen,
                     const unsigned char *input, size_t ilen,
                     unsigned char *output )
{
    mbedtls_md_context_t ctx;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    if( !md_info )
        return( MBEDTLS_ERR_MD_BAD_INPUT_DATA );
    mbedtls_md_init( &ctx );
    CHECK( mbedtls_md_setup( &ctx, md_info, 1 ) );
    CHECK( mbedtls_md_hmac_starts( &ctx, key, keylen ) );
    CHECK( mbedtls_md_hmac_update( &ctx, input, ilen ) );
    CHECK( mbedtls_md_hmac_finish( &ctx, output ) );
cleanup:
    mbedtls_md_free( &ctx );
    return( ret );
}

#if defined(MBEDTLS_MD2_C)
const mbedtls_md_info_t mbedtls_md2_info = {
    "MD2",
    MBEDTLS_MD_MD2,
    16,
    16,
};
#endif

#if defined(MBEDTLS_MD4_C)
const mbedtls_md_info_t mbedtls_md4_info = {
    "MD4",
    MBEDTLS_MD_MD4,
    16,
    64,
};
#endif

#endif /* MBEDTLS_MD_C */
