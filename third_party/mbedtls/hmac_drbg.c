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
#include <libc/stdio/stdio.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/hmac_drbg.h"
#include "third_party/mbedtls/platform.h"
__static_yoink("mbedtls_notice");

/*
 *  HMAC_DRBG implementation (NIST SP 800-90)
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

/*
 *  The NIST SP 800-90A DRBGs are described in the following publication.
 *  http://csrc.nist.gov/publications/nistpubs/800-90A/SP800-90A.pdf
 *  References below are based on rev. 1 (January 2012).
 */

#if defined(MBEDTLS_HMAC_DRBG_C)

/**
 * \brief               HMAC_DRBG context initialization.
 *
 * This function makes the context ready for mbedtls_hmac_drbg_seed(),
 * mbedtls_hmac_drbg_seed_buf() or mbedtls_hmac_drbg_free().
 *
 * \note                The reseed interval is #MBEDTLS_HMAC_DRBG_RESEED_INTERVAL
 *                      by default. Override this value by calling
 *                      mbedtls_hmac_drbg_set_reseed_interval().
 *
 * \param ctx           HMAC_DRBG context to be initialized.
 */
void mbedtls_hmac_drbg_init( mbedtls_hmac_drbg_context *ctx )
{
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_hmac_drbg_context ) );
    ctx->reseed_interval = MBEDTLS_HMAC_DRBG_RESEED_INTERVAL;
}

/**
 * \brief               This function updates the state of the HMAC_DRBG context.
 *
 * \note                This function is not thread-safe. It is not safe
 *                      to call this function if another thread might be
 *                      concurrently obtaining random numbers from the same
 *                      context or updating or reseeding the same context.
 *
 * \param ctx           The HMAC_DRBG context.
 * \param additional    The data to update the state with.
 *                      If this is \c NULL, there is no additional data.
 * \param add_len       Length of \p additional in bytes.
 *                      Unused if \p additional is \c NULL.
 *
 * \return              \c 0 on success, or an error from the underlying
 *                      hash calculation.
 */
int mbedtls_hmac_drbg_update_ret( mbedtls_hmac_drbg_context *ctx,
                                  const unsigned char *additional,
                                  size_t add_len )
{
    size_t md_len = mbedtls_md_get_size( ctx->md_ctx.md_info );
    unsigned char rounds = ( additional != NULL && add_len != 0 ) ? 2 : 1;
    unsigned char sep[1];
    unsigned char K[MBEDTLS_MD_MAX_SIZE];
    int ret = MBEDTLS_ERR_MD_BAD_INPUT_DATA;

    for( sep[0] = 0; sep[0] < rounds; sep[0]++ )
    {
        /* Step 1 or 4 */
        if( ( ret = mbedtls_md_hmac_reset( &ctx->md_ctx ) ) != 0 )
            goto exit;
        if( ( ret = mbedtls_md_hmac_update( &ctx->md_ctx,
                                            ctx->V, md_len ) ) != 0 )
            goto exit;
        if( ( ret = mbedtls_md_hmac_update( &ctx->md_ctx,
                                            sep, 1 ) ) != 0 )
            goto exit;
        if( rounds == 2 )
        {
            if( ( ret = mbedtls_md_hmac_update( &ctx->md_ctx,
                                                additional, add_len ) ) != 0 )
            goto exit;
        }
        if( ( ret = mbedtls_md_hmac_finish( &ctx->md_ctx, K ) ) != 0 )
            goto exit;

        /* Step 2 or 5 */
        if( ( ret = mbedtls_md_hmac_starts( &ctx->md_ctx, K, md_len ) ) != 0 )
            goto exit;
        if( ( ret = mbedtls_md_hmac_update( &ctx->md_ctx,
                                            ctx->V, md_len ) ) != 0 )
            goto exit;
        if( ( ret = mbedtls_md_hmac_finish( &ctx->md_ctx, ctx->V ) ) != 0 )
            goto exit;
    }

exit:
    mbedtls_platform_zeroize( K, sizeof( K ) );
    return( ret );
}

/**
 * \brief               Initilisation of simpified HMAC_DRBG (never reseeds).
 *
 * This function is meant for use in algorithms that need a pseudorandom
 * input such as deterministic ECDSA.
 *
 * \param ctx           HMAC_DRBG context to be initialised.
 * \param md_info       MD algorithm to use for HMAC_DRBG.
 * \param data          Concatenation of the initial entropy string and
 *                      the additional data.
 * \param data_len      Length of \p data in bytes.
 *
 * \return              \c 0 if successful. or
 * \return              #MBEDTLS_ERR_MD_BAD_INPUT_DATA if \p md_info is
 *                      invalid.
 * \return              #MBEDTLS_ERR_MD_ALLOC_FAILED if there was not enough
 *                      memory to allocate context data.
 */
int mbedtls_hmac_drbg_seed_buf( mbedtls_hmac_drbg_context *ctx,
                                const mbedtls_md_info_t * md_info,
                                const unsigned char *data, size_t data_len )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;

    if( ( ret = mbedtls_md_setup( &ctx->md_ctx, md_info, 1 ) ) != 0 )
        return( ret );

    /*
     * Set initial working state.
     * Use the V memory location, which is currently all 0, to initialize the
     * MD context with an all-zero key. Then set V to its initial value.
     */
    if( ( ret = mbedtls_md_hmac_starts( &ctx->md_ctx, ctx->V,
                                        mbedtls_md_get_size( md_info ) ) ) != 0 )
        return( ret );
    memset( ctx->V, 0x01, mbedtls_md_get_size( md_info ) );

    if( ( ret = mbedtls_hmac_drbg_update_ret( ctx, data, data_len ) ) != 0 )
        return( ret );

    return( 0 );
}

/*
 * Internal function used both for seeding and reseeding the DRBG.
 * Comments starting with arabic numbers refer to section 10.1.2.4
 * of SP800-90A, while roman numbers refer to section 9.2.
 */
static int hmac_drbg_reseed_core( mbedtls_hmac_drbg_context *ctx,
                                  const unsigned char *additional, size_t len,
                                  int use_nonce )
{
    unsigned char seed[MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT];
    size_t seedlen = 0;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;

    {
        size_t total_entropy_len;

        if( use_nonce == 0 )
            total_entropy_len = ctx->entropy_len;
        else
            total_entropy_len = ctx->entropy_len * 3 / 2;

        /* III. Check input length */
        if( len > MBEDTLS_HMAC_DRBG_MAX_INPUT ||
            total_entropy_len + len > MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT )
        {
            return( MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG );
        }
    }

    mbedtls_platform_zeroize( seed, MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT );

    /* IV. Gather entropy_len bytes of entropy for the seed */
    if( ( ret = ctx->f_entropy( ctx->p_entropy,
                                seed, ctx->entropy_len ) ) != 0 )
    {
        return( MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED );
    }
    seedlen += ctx->entropy_len;

    /* For initial seeding, allow adding of nonce generated
     * from the entropy source. See Sect 8.6.7 in SP800-90A. */
    if( use_nonce )
    {
        /* Note: We don't merge the two calls to f_entropy() in order
         *       to avoid requesting too much entropy from f_entropy()
         *       at once. Specifically, if the underlying digest is not
         *       SHA-1, 3 / 2 * entropy_len is at least 36 Bytes, which
         *       is larger than the maximum of 32 Bytes that our own
         *       entropy source implementation can emit in a single
         *       call in configurations disabling SHA-512. */
        if( ( ret = ctx->f_entropy( ctx->p_entropy,
                                    seed + seedlen,
                                    ctx->entropy_len / 2 ) ) != 0 )
        {
            return( MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED );
        }

        seedlen += ctx->entropy_len / 2;
    }


    /* 1. Concatenate entropy and additional data if any */
    if( additional != NULL && len != 0 )
    {
        memcpy( seed + seedlen, additional, len );
        seedlen += len;
    }

    /* 2. Update state */
    if( ( ret = mbedtls_hmac_drbg_update_ret( ctx, seed, seedlen ) ) != 0 )
        goto exit;

    /* 3. Reset reseed_counter */
    ctx->reseed_counter = 1;

exit:
    /* 4. Done */
    mbedtls_platform_zeroize( seed, seedlen );
    return( ret );
}

/**
 * \brief               This function reseeds the HMAC_DRBG context, that is
 *                      extracts data from the entropy source.
 *
 * \note                This function is not thread-safe. It is not safe
 *                      to call this function if another thread might be
 *                      concurrently obtaining random numbers from the same
 *                      context or updating or reseeding the same context.
 *
 * \param ctx           The HMAC_DRBG context.
 * \param additional    Additional data to add to the state.
 *                      If this is \c NULL, there is no additional data
 *                      and \p len should be \c 0.
 * \param len           The length of the additional data.
 *                      This must be at most #MBEDTLS_HMAC_DRBG_MAX_INPUT
 *                      and also at most
 *                      #MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT - \p entropy_len
 *                      where \p entropy_len is the entropy length
 *                      (see mbedtls_hmac_drbg_set_entropy_len()).
 *
 * \return              \c 0 if successful.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED
 *                      if a call to the entropy function failed.
 */
int mbedtls_hmac_drbg_reseed( mbedtls_hmac_drbg_context *ctx,
                      const unsigned char *additional, size_t len )
{
    return( hmac_drbg_reseed_core( ctx, additional, len, 0 ) );
}

/**
 * \brief               HMAC_DRBG initial seeding.
 *
 * Set the initial seed and set up the entropy source for future reseeds.
 *
 * A typical choice for the \p f_entropy and \p p_entropy parameters is
 * to use the entropy module:
 * - \p f_entropy is mbedtls_entropy_func();
 * - \p p_entropy is an instance of ::mbedtls_entropy_context initialized
 *   with mbedtls_entropy_init() (which registers the platform's default
 *   entropy sources).
 *
 * You can provide a personalization string in addition to the
 * entropy source, to make this instantiation as unique as possible.
 *
 * \note                By default, the security strength as defined by NIST is:
 *                      - 128 bits if \p md_info is SHA-1;
 *                      - 192 bits if \p md_info is SHA-224;
 *                      - 256 bits if \p md_info is SHA-256, SHA-384 or SHA-512.
 *                      Note that SHA-256 is just as efficient as SHA-224.
 *                      The security strength can be reduced if a smaller
 *                      entropy length is set with
 *                      mbedtls_hmac_drbg_set_entropy_len().
 *
 * \note                The default entropy length is the security strength
 *                      (converted from bits to bytes). You can override
 *                      it by calling mbedtls_hmac_drbg_set_entropy_len().
 *
 * \note                During the initial seeding, this function calls
 *                      the entropy source to obtain a nonce
 *                      whose length is half the entropy length.
 *
 * \param ctx           HMAC_DRBG context to be seeded.
 * \param md_info       MD algorithm to use for HMAC_DRBG.
 * \param f_entropy     The entropy callback, taking as arguments the
 *                      \p p_entropy context, the buffer to fill, and the
 *                      length of the buffer.
 *                      \p f_entropy is always called with a length that is
 *                      less than or equal to the entropy length.
 * \param p_entropy     The entropy context to pass to \p f_entropy.
 * \param custom        The personalization string.
 *                      This can be \c NULL, in which case the personalization
 *                      string is empty regardless of the value of \p len.
 * \param len           The length of the personalization string.
 *                      This must be at most #MBEDTLS_HMAC_DRBG_MAX_INPUT
 *                      and also at most
 *                      #MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT - \p entropy_len * 3 / 2
 *                      where \p entropy_len is the entropy length
 *                      described above.
 *
 * \return              \c 0 if successful.
 * \return              #MBEDTLS_ERR_MD_BAD_INPUT_DATA if \p md_info is
 *                      invalid.
 * \return              #MBEDTLS_ERR_MD_ALLOC_FAILED if there was not enough
 *                      memory to allocate context data.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED
 *                      if the call to \p f_entropy failed.
 */
int mbedtls_hmac_drbg_seed( mbedtls_hmac_drbg_context *ctx,
                            const mbedtls_md_info_t * md_info,
                            int (*f_entropy)(void *, unsigned char *, size_t),
                            void *p_entropy,
                            const unsigned char *custom,
                            size_t len )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t md_size;

    if( ( ret = mbedtls_md_setup( &ctx->md_ctx, md_info, 1 ) ) != 0 )
        return( ret );

    md_size = mbedtls_md_get_size( md_info );

    /*
     * Set initial working state.
     * Use the V memory location, which is currently all 0, to initialize the
     * MD context with an all-zero key. Then set V to its initial value.
     */
    if( ( ret = mbedtls_md_hmac_starts( &ctx->md_ctx, ctx->V, md_size ) ) != 0 )
        return( ret );
    memset( ctx->V, 0x01, md_size );

    ctx->f_entropy = f_entropy;
    ctx->p_entropy = p_entropy;

    if( ctx->entropy_len == 0 )
    {
        /*
         * See SP800-57 5.6.1 (p. 65-66) for the security strength provided by
         * each hash function, then according to SP800-90A rev1 10.1 table 2,
         * min_entropy_len (in bits) is security_strength.
         *
         * (This also matches the sizes used in the NIST test vectors.)
         */
        ctx->entropy_len = md_size <= 20 ? 16 : /* 160-bits hash -> 128 bits */
                           md_size <= 28 ? 24 : /* 224-bits hash -> 192 bits */
                           32;  /* better (256+) -> 256 bits */
    }

    if( ( ret = hmac_drbg_reseed_core( ctx, custom, len,
                                       1 /* add nonce */ ) ) != 0 )
    {
        return( ret );
    }

    return( 0 );
}

/**
 * \brief               This function turns prediction resistance on or off.
 *                      The default value is off.
 *
 * \note                If enabled, entropy is gathered at the beginning of
 *                      every call to mbedtls_hmac_drbg_random_with_add()
 *                      or mbedtls_hmac_drbg_random().
 *                      Only use this if your entropy source has sufficient
 *                      throughput.
 *
 * \param ctx           The HMAC_DRBG context.
 * \param resistance    #MBEDTLS_HMAC_DRBG_PR_ON or #MBEDTLS_HMAC_DRBG_PR_OFF.
 */
void mbedtls_hmac_drbg_set_prediction_resistance( mbedtls_hmac_drbg_context *ctx,
                                          int resistance )
{
    ctx->prediction_resistance = resistance;
}

/**
 * \brief               This function sets the amount of entropy grabbed on each
 *                      seed or reseed.
 *
 * See the documentation of mbedtls_hmac_drbg_seed() for the default value.
 *
 * \param ctx           The HMAC_DRBG context.
 * \param len           The amount of entropy to grab, in bytes.
 */
void mbedtls_hmac_drbg_set_entropy_len( mbedtls_hmac_drbg_context *ctx, size_t len )
{
    ctx->entropy_len = len;
}

/**
 * \brief               Set the reseed interval.
 *
 * The reseed interval is the number of calls to mbedtls_hmac_drbg_random()
 * or mbedtls_hmac_drbg_random_with_add() after which the entropy function
 * is called again.
 *
 * The default value is #MBEDTLS_HMAC_DRBG_RESEED_INTERVAL.
 *
 * \param ctx           The HMAC_DRBG context.
 * \param interval      The reseed interval.
 */
void mbedtls_hmac_drbg_set_reseed_interval( mbedtls_hmac_drbg_context *ctx, int interval )
{
    ctx->reseed_interval = interval;
}

/**
 * \brief   This function updates an HMAC_DRBG instance with additional
 *          data and uses it to generate random data.
 *
 * This function automatically reseeds if the reseed counter is exceeded
 * or prediction resistance is enabled.
 *
 * \note                This function is not thread-safe. It is not safe
 *                      to call this function if another thread might be
 *                      concurrently obtaining random numbers from the same
 *                      context or updating or reseeding the same context.
 *
 * \param p_rng         The HMAC_DRBG context. This must be a pointer to a
 *                      #mbedtls_hmac_drbg_context structure.
 * \param output        The buffer to fill.
 * \param output_len    The length of the buffer in bytes.
 *                      This must be at most #MBEDTLS_HMAC_DRBG_MAX_REQUEST.
 * \param additional    Additional data to update with.
 *                      If this is \c NULL, there is no additional data
 *                      and \p add_len should be \c 0.
 * \param add_len       The length of the additional data.
 *                      This must be at most #MBEDTLS_HMAC_DRBG_MAX_INPUT.
 *
 * \return              \c 0 if successful.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED
 *                      if a call to the entropy source failed.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_REQUEST_TOO_BIG if
 *                      \p output_len > #MBEDTLS_HMAC_DRBG_MAX_REQUEST.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG if
 *                      \p add_len > #MBEDTLS_HMAC_DRBG_MAX_INPUT.
 */
int mbedtls_hmac_drbg_random_with_add( void *p_rng,
                                       unsigned char *output, size_t out_len,
                                       const unsigned char *additional, size_t add_len )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_hmac_drbg_context *ctx = (mbedtls_hmac_drbg_context *) p_rng;
    size_t md_len = mbedtls_md_get_size( ctx->md_ctx.md_info );
    size_t left = out_len;
    unsigned char *out = output;

    /* II. Check request length */
    if( out_len > MBEDTLS_HMAC_DRBG_MAX_REQUEST )
        return( MBEDTLS_ERR_HMAC_DRBG_REQUEST_TOO_BIG );

    /* III. Check input length */
    if( add_len > MBEDTLS_HMAC_DRBG_MAX_INPUT )
        return( MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG );

    /* 1. (aka VII and IX) Check reseed counter and PR */
    if( ctx->f_entropy != NULL && /* For no-reseeding instances */
        ( ctx->prediction_resistance == MBEDTLS_HMAC_DRBG_PR_ON ||
          ctx->reseed_counter > ctx->reseed_interval ) )
    {
        if( ( ret = mbedtls_hmac_drbg_reseed( ctx, additional, add_len ) ) != 0 )
            return( ret );

        add_len = 0; /* VII.4 */
    }

    /* 2. Use additional data if any */
    if( additional != NULL && add_len != 0 )
    {
        if( ( ret = mbedtls_hmac_drbg_update_ret( ctx,
                                                  additional, add_len ) ) != 0 )
            goto exit;
    }

    /* 3, 4, 5. Generate bytes */
    while( left != 0 )
    {
        size_t use_len = left > md_len ? md_len : left;

        if( ( ret = mbedtls_md_hmac_reset( &ctx->md_ctx ) ) != 0 )
            goto exit;
        if( ( ret = mbedtls_md_hmac_update( &ctx->md_ctx,
                                            ctx->V, md_len ) ) != 0 )
            goto exit;
        if( ( ret = mbedtls_md_hmac_finish( &ctx->md_ctx, ctx->V ) ) != 0 )
            goto exit;

        memcpy( out, ctx->V, use_len );
        out += use_len;
        left -= use_len;
    }

    /* 6. Update */
    if( ( ret = mbedtls_hmac_drbg_update_ret( ctx,
                                              additional, add_len ) ) != 0 )
        goto exit;

    /* 7. Update reseed counter */
    ctx->reseed_counter++;

exit:
    /* 8. Done */
    return( ret );
}

/**
 * \brief   This function uses HMAC_DRBG to generate random data.
 *
 * This function automatically reseeds if the reseed counter is exceeded
 * or prediction resistance is enabled.
 *
 * \param p_rng         The HMAC_DRBG context. This must be a pointer to a
 *                      #mbedtls_hmac_drbg_context structure.
 * \param output        The buffer to fill.
 * \param out_len       The length of the buffer in bytes.
 *                      This must be at most #MBEDTLS_HMAC_DRBG_MAX_REQUEST.
 *
 * \return              \c 0 if successful.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED
 *                      if a call to the entropy source failed.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_REQUEST_TOO_BIG if
 *                      \p out_len > #MBEDTLS_HMAC_DRBG_MAX_REQUEST.
 */
int mbedtls_hmac_drbg_random( void *p_rng, unsigned char *output, size_t out_len )
{
    mbedtls_hmac_drbg_context *ctx = (mbedtls_hmac_drbg_context *) p_rng;
    return mbedtls_hmac_drbg_random_with_add( ctx, output, out_len, NULL, 0 );
}

/**
 * \brief               This function resets HMAC_DRBG context to the state immediately
 *                      after initial call of mbedtls_hmac_drbg_init().
 *
 * \param ctx           The HMAC_DRBG context to free.
 */
void mbedtls_hmac_drbg_free( mbedtls_hmac_drbg_context *ctx )
{
    if( ctx == NULL )
        return;
    mbedtls_md_free( &ctx->md_ctx );
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_hmac_drbg_context ) );
    ctx->reseed_interval = MBEDTLS_HMAC_DRBG_RESEED_INTERVAL;
}

#if defined(MBEDTLS_FS_IO)

/**
 * \brief               This function writes a seed file.
 *
 * \param ctx           The HMAC_DRBG context.
 * \param path          The name of the file.
 *
 * \return              \c 0 on success.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_FILE_IO_ERROR on file error.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED on reseed
 *                      failure.
 */
int mbedtls_hmac_drbg_write_seed_file( mbedtls_hmac_drbg_context *ctx, const char *path )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    FILE *f;
    unsigned char buf[ MBEDTLS_HMAC_DRBG_MAX_INPUT ];

    if( ( f = fopen( path, "wb" ) ) == NULL )
        return( MBEDTLS_ERR_HMAC_DRBG_FILE_IO_ERROR );

    if( ( ret = mbedtls_hmac_drbg_random( ctx, buf, sizeof( buf ) ) ) != 0 )
        goto exit;

    if( fwrite( buf, 1, sizeof( buf ), f ) != sizeof( buf ) )
    {
        ret = MBEDTLS_ERR_HMAC_DRBG_FILE_IO_ERROR;
        goto exit;
    }

    ret = 0;

exit:
    fclose( f );
    mbedtls_platform_zeroize( buf, sizeof( buf ) );

    return( ret );
}

/**
 * \brief               This function reads and updates a seed file. The seed
 *                      is added to this instance.
 *
 * \param ctx           The HMAC_DRBG context.
 * \param path          The name of the file.
 *
 * \return              \c 0 on success.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_FILE_IO_ERROR on file error.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED on
 *                      reseed failure.
 * \return              #MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG if the existing
 *                      seed file is too large.
 */
int mbedtls_hmac_drbg_update_seed_file( mbedtls_hmac_drbg_context *ctx, const char *path )
{
    int ret = 0;
    FILE *f = NULL;
    size_t n;
    unsigned char buf[ MBEDTLS_HMAC_DRBG_MAX_INPUT ];
    unsigned char c;

    if( ( f = fopen( path, "rb" ) ) == NULL )
        return( MBEDTLS_ERR_HMAC_DRBG_FILE_IO_ERROR );

    n = fread( buf, 1, sizeof( buf ), f );
    if( fread( &c, 1, 1, f ) != 0 )
    {
        ret = MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG;
        goto exit;
    }
    if( n == 0 || ferror( f ) )
    {
        ret = MBEDTLS_ERR_HMAC_DRBG_FILE_IO_ERROR;
        goto exit;
    }
    fclose( f );
    f = NULL;

    ret = mbedtls_hmac_drbg_update_ret( ctx, buf, n );

exit:
    mbedtls_platform_zeroize( buf, sizeof( buf ) );
    if( f != NULL )
        fclose( f );
    if( ret != 0 )
        return( ret );
    return( mbedtls_hmac_drbg_write_seed_file( ctx, path ) );
}
#endif /* MBEDTLS_FS_IO */

#if defined(MBEDTLS_SELF_TEST)
#if defined(MBEDTLS_SHA1_C)

#define OUTPUT_LEN  80

/* From a NIST PR=true test vector */
static const unsigned char entropy_pr[] = {
    0xa0, 0xc9, 0xab, 0x58, 0xf1, 0xe2, 0xe5, 0xa4, 0xde, 0x3e, 0xbd, 0x4f,
    0xf7, 0x3e, 0x9c, 0x5b, 0x64, 0xef, 0xd8, 0xca, 0x02, 0x8c, 0xf8, 0x11,
    0x48, 0xa5, 0x84, 0xfe, 0x69, 0xab, 0x5a, 0xee, 0x42, 0xaa, 0x4d, 0x42,
    0x17, 0x60, 0x99, 0xd4, 0x5e, 0x13, 0x97, 0xdc, 0x40, 0x4d, 0x86, 0xa3,
    0x7b, 0xf5, 0x59, 0x54, 0x75, 0x69, 0x51, 0xe4 };
static const unsigned char result_pr[OUTPUT_LEN] = {
    0x9a, 0x00, 0xa2, 0xd0, 0x0e, 0xd5, 0x9b, 0xfe, 0x31, 0xec, 0xb1, 0x39,
    0x9b, 0x60, 0x81, 0x48, 0xd1, 0x96, 0x9d, 0x25, 0x0d, 0x3c, 0x1e, 0x94,
    0x10, 0x10, 0x98, 0x12, 0x93, 0x25, 0xca, 0xb8, 0xfc, 0xcc, 0x2d, 0x54,
    0x73, 0x19, 0x70, 0xc0, 0x10, 0x7a, 0xa4, 0x89, 0x25, 0x19, 0x95, 0x5e,
    0x4b, 0xc6, 0x00, 0x1d, 0x7f, 0x4e, 0x6a, 0x2b, 0xf8, 0xa3, 0x01, 0xab,
    0x46, 0x05, 0x5c, 0x09, 0xa6, 0x71, 0x88, 0xf1, 0xa7, 0x40, 0xee, 0xf3,
    0xe1, 0x5c, 0x02, 0x9b, 0x44, 0xaf, 0x03, 0x44 };

/* From a NIST PR=false test vector */
static const unsigned char entropy_nopr[] = {
    0x79, 0x34, 0x9b, 0xbf, 0x7c, 0xdd, 0xa5, 0x79, 0x95, 0x57, 0x86, 0x66,
    0x21, 0xc9, 0x13, 0x83, 0x11, 0x46, 0x73, 0x3a, 0xbf, 0x8c, 0x35, 0xc8,
    0xc7, 0x21, 0x5b, 0x5b, 0x96, 0xc4, 0x8e, 0x9b, 0x33, 0x8c, 0x74, 0xe3,
    0xe9, 0x9d, 0xfe, 0xdf };
static const unsigned char result_nopr[OUTPUT_LEN] = {
    0xc6, 0xa1, 0x6a, 0xb8, 0xd4, 0x20, 0x70, 0x6f, 0x0f, 0x34, 0xab, 0x7f,
    0xec, 0x5a, 0xdc, 0xa9, 0xd8, 0xca, 0x3a, 0x13, 0x3e, 0x15, 0x9c, 0xa6,
    0xac, 0x43, 0xc6, 0xf8, 0xa2, 0xbe, 0x22, 0x83, 0x4a, 0x4c, 0x0a, 0x0a,
    0xff, 0xb1, 0x0d, 0x71, 0x94, 0xf1, 0xc1, 0xa5, 0xcf, 0x73, 0x22, 0xec,
    0x1a, 0xe0, 0x96, 0x4e, 0xd4, 0xbf, 0x12, 0x27, 0x46, 0xe0, 0x87, 0xfd,
    0xb5, 0xb3, 0xe9, 0x1b, 0x34, 0x93, 0xd5, 0xbb, 0x98, 0xfa, 0xed, 0x49,
    0xe8, 0x5f, 0x13, 0x0f, 0xc8, 0xa4, 0x59, 0xb7 };

/* "Entropy" from buffer */
static size_t test_offset;
static int hmac_drbg_self_test_entropy( void *data,
                                        unsigned char *buf, size_t len )
{
    const unsigned char *p = data;
    memcpy( buf, p + test_offset, len );
    test_offset += len;
    return( 0 );
}

#define CHK( c )    if( (c) != 0 )                          \
                    {                                       \
                        if( verbose != 0 )                  \
                            mbedtls_printf( "failed\n" );  \
                        return( 1 );                        \
                    }

/**
 * \brief               The HMAC_DRBG Checkup routine.
 *
 * \return              \c 0 if successful.
 * \return              \c 1 if the test failed.
 */
int mbedtls_hmac_drbg_self_test( int verbose )
{
    mbedtls_hmac_drbg_context ctx;
    unsigned char buf[OUTPUT_LEN];
    const mbedtls_md_info_t *md_info = mbedtls_md_info_from_type( MBEDTLS_MD_SHA1 );

    mbedtls_hmac_drbg_init( &ctx );

    /*
     * PR = True
     */
    if( verbose != 0 )
        mbedtls_printf( "  HMAC_DRBG (PR = True) : " );

    test_offset = 0;
    CHK( mbedtls_hmac_drbg_seed( &ctx, md_info,
                         hmac_drbg_self_test_entropy, (void *) entropy_pr,
                         NULL, 0 ) );
    mbedtls_hmac_drbg_set_prediction_resistance( &ctx, MBEDTLS_HMAC_DRBG_PR_ON );
    CHK( mbedtls_hmac_drbg_random( &ctx, buf, OUTPUT_LEN ) );
    CHK( mbedtls_hmac_drbg_random( &ctx, buf, OUTPUT_LEN ) );
    CHK( timingsafe_bcmp( buf, result_pr, OUTPUT_LEN ) );
    mbedtls_hmac_drbg_free( &ctx );

    mbedtls_hmac_drbg_free( &ctx );

    if( verbose != 0 )
        mbedtls_printf( "passed\n" );

    /*
     * PR = False
     */
    if( verbose != 0 )
        mbedtls_printf( "  HMAC_DRBG (PR = False) : " );

    mbedtls_hmac_drbg_init( &ctx );

    test_offset = 0;
    CHK( mbedtls_hmac_drbg_seed( &ctx, md_info,
                         hmac_drbg_self_test_entropy, (void *) entropy_nopr,
                         NULL, 0 ) );
    CHK( mbedtls_hmac_drbg_reseed( &ctx, NULL, 0 ) );
    CHK( mbedtls_hmac_drbg_random( &ctx, buf, OUTPUT_LEN ) );
    CHK( mbedtls_hmac_drbg_random( &ctx, buf, OUTPUT_LEN ) );
    CHK( timingsafe_bcmp( buf, result_nopr, OUTPUT_LEN ) );
    mbedtls_hmac_drbg_free( &ctx );

    mbedtls_hmac_drbg_free( &ctx );

    if( verbose != 0 )
        mbedtls_printf( "passed\n" );

    if( verbose != 0 )
        mbedtls_printf( "\n" );

    return( 0 );
}

#endif /* MBEDTLS_SHA1_C */
#endif /* MBEDTLS_SELF_TEST */

#endif /* MBEDTLS_HMAC_DRBG_C */
