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
#include <libc/calls/calls.h>
#include <libc/stdio/stdio.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/entropy.h"
#include "third_party/mbedtls/entropy_poll.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/platform.h"
__static_yoink("mbedtls_notice");

/*
 *  Entropy accumulator implementation
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

#if defined(MBEDTLS_ENTROPY_C)

#if defined(MBEDTLS_TEST_NULL_ENTROPY)
#warning "**** WARNING!  MBEDTLS_TEST_NULL_ENTROPY defined! "
#warning "**** THIS BUILD HAS NO DEFINED ENTROPY SOURCES "
#warning "**** THIS BUILD IS *NOT* SUITABLE FOR PRODUCTION USE "
#endif

#define ENTROPY_MAX_LOOP    256     /*< Maximum amount to loop before error */

/**
 * \brief           Initialize the context
 *
 * \param ctx       Entropy context to initialize
 */
void mbedtls_entropy_init( mbedtls_entropy_context *ctx )
{
    ctx->source_count = 0;
    mbedtls_platform_zeroize( ctx->source, sizeof( ctx->source ) );

    ctx->accumulator_started = 0;
#if defined(MBEDTLS_ENTROPY_SHA512_ACCUMULATOR)
    mbedtls_sha512_init( &ctx->accumulator );
#else
    mbedtls_sha256_init( &ctx->accumulator );
#endif

    /* Reminder: Update ENTROPY_HAVE_STRONG in the test files
     *           when adding more strong entropy sources here. */
    mbedtls_entropy_add_source( ctx, mbedtls_hardware_poll, NULL,
                                MBEDTLS_ENTROPY_MIN_HARDWARE,
                                MBEDTLS_ENTROPY_SOURCE_STRONG );
}

/**
 * \brief           Free the data in the context
 *
 * \param ctx       Entropy context to free
 */
void mbedtls_entropy_free( mbedtls_entropy_context *ctx )
{
    /* If the context was already free, don't call free() again.
     * This is important for mutexes which don't allow double-free. */
    if( ctx->accumulator_started == -1 )
        return;
#if defined(MBEDTLS_ENTROPY_SHA512_ACCUMULATOR)
    mbedtls_sha512_free( &ctx->accumulator );
#else
    mbedtls_sha256_free( &ctx->accumulator );
#endif
#if defined(MBEDTLS_ENTROPY_NV_SEED)
    ctx->initial_entropy_run = 0;
#endif
    ctx->source_count = 0;
    mbedtls_platform_zeroize( ctx->source, sizeof( ctx->source ) );
    ctx->accumulator_started = -1;
}

/**
 * \brief           Adds an entropy source to poll
 *
 * \param ctx       Entropy context
 * \param f_source  Entropy function
 * \param p_source  Function data
 * \param threshold Minimum required from source before entropy is released
 *                  ( with mbedtls_entropy_func() ) (in bytes)
 * \param strong    MBEDTLS_ENTROPY_SOURCE_STRONG or
 *                  MBEDTLS_ENTROPY_SOURCE_WEAK.
 *                  At least one strong source needs to be added.
 *                  Weaker sources (such as the cycle counter) can be used as
 *                  a complement.
 *
 * \return          0 if successful or MBEDTLS_ERR_ENTROPY_MAX_SOURCES
 */
int mbedtls_entropy_add_source( mbedtls_entropy_context *ctx,
                                mbedtls_entropy_f_source_ptr f_source,
                                void *p_source, size_t threshold, int strong )
{
    int idx;
    idx = ctx->source_count;
    if( idx >= MBEDTLS_ENTROPY_MAX_SOURCES ) return MBEDTLS_ERR_ENTROPY_MAX_SOURCES;
    ctx->source[idx].f_source  = f_source;
    ctx->source[idx].p_source  = p_source;
    ctx->source[idx].threshold = threshold;
    ctx->source[idx].strong    = strong;
    ctx->source_count++;
    return 0;
}

/*
 * Entropy accumulator update
 */
static int entropy_update( mbedtls_entropy_context *ctx, unsigned char source_id,
                           const unsigned char *data, size_t len )
{
    unsigned char header[2];
    unsigned char tmp[MBEDTLS_ENTROPY_BLOCK_SIZE];
    size_t use_len = len;
    const unsigned char *p = data;
    int ret = 0;

    if( use_len > MBEDTLS_ENTROPY_BLOCK_SIZE )
    {
#if defined(MBEDTLS_ENTROPY_SHA512_ACCUMULATOR)
        if( ( ret = mbedtls_sha512_ret( data, len, tmp, 0 ) ) != 0 )
            goto cleanup;
#else
        if( ( ret = mbedtls_sha256_ret( data, len, tmp, 0 ) ) != 0 )
            goto cleanup;
#endif
        p = tmp;
        use_len = MBEDTLS_ENTROPY_BLOCK_SIZE;
    }

    header[0] = source_id;
    header[1] = use_len & 0xFF;

    /*
     * Start the accumulator if this has not already happened. Note that
     * it is sufficient to start the accumulator here only because all calls to
     * gather entropy eventually execute this code.
     */
#if defined(MBEDTLS_ENTROPY_SHA512_ACCUMULATOR)
    if( ctx->accumulator_started == 0 &&
        ( ret = mbedtls_sha512_starts_ret( &ctx->accumulator, 0 ) ) != 0 )
        goto cleanup;
    else
        ctx->accumulator_started = 1;
    if( ( ret = mbedtls_sha512_update_ret( &ctx->accumulator, header, 2 ) ) != 0 )
        goto cleanup;
    ret = mbedtls_sha512_update_ret( &ctx->accumulator, p, use_len );
#else
    if( ctx->accumulator_started == 0 &&
        ( ret = mbedtls_sha256_starts_ret( &ctx->accumulator, 0 ) ) != 0 )
        goto cleanup;
    else
        ctx->accumulator_started = 1;
    if( ( ret = mbedtls_sha256_update_ret( &ctx->accumulator, header, 2 ) ) != 0 )
        goto cleanup;
    ret = mbedtls_sha256_update_ret( &ctx->accumulator, p, use_len );
#endif

cleanup:
    mbedtls_platform_zeroize( tmp, sizeof( tmp ) );

    return( ret );
}

/**
 * \brief           Add data to the accumulator manually
 *
 * \param ctx       Entropy context
 * \param data      Data to add
 * \param len       Length of data
 *
 * \return          0 if successful
 */
int mbedtls_entropy_update_manual( mbedtls_entropy_context *ctx,
                           const unsigned char *data, size_t len )
{
    return entropy_update( ctx, MBEDTLS_ENTROPY_SOURCE_MANUAL, data, len );
}

/*
 * Run through the different sources to add entropy to our accumulator
 */
static int entropy_gather_internal( mbedtls_entropy_context *ctx )
{
    int ret = MBEDTLS_ERR_ENTROPY_SOURCE_FAILED;
    int i;
    int have_one_strong = 0;
    unsigned char buf[MBEDTLS_ENTROPY_MAX_GATHER];
    size_t olen;

    if( ctx->source_count == 0 )
        return( MBEDTLS_ERR_ENTROPY_NO_SOURCES_DEFINED );

    /*
     * Run through our entropy sources
     */
    for( i = 0; i < ctx->source_count; i++ )
    {
        if( ctx->source[i].strong == MBEDTLS_ENTROPY_SOURCE_STRONG )
            have_one_strong = 1;

        olen = 0;
        if( ( ret = ctx->source[i].f_source( ctx->source[i].p_source,
                        buf, MBEDTLS_ENTROPY_MAX_GATHER, &olen ) ) != 0 )
        {
            goto cleanup;
        }

        /*
         * Add if we actually gathered something
         */
        if( olen > 0 )
        {
            if( ( ret = entropy_update( ctx, (unsigned char) i,
                                        buf, olen ) ) != 0 )
                return( ret );
            ctx->source[i].size += olen;
        }
    }

    if( have_one_strong == 0 )
        ret = MBEDTLS_ERR_ENTROPY_NO_STRONG_SOURCE;

cleanup:
    mbedtls_platform_zeroize( buf, sizeof( buf ) );

    return( ret );
}

/**
 * \brief           Trigger an extra gather poll for the accumulator
 *
 * \param ctx       Entropy context
 *
 * \return          0 if successful, or MBEDTLS_ERR_ENTROPY_SOURCE_FAILED
 */
int mbedtls_entropy_gather( mbedtls_entropy_context *ctx )
{
    return entropy_gather_internal( ctx );
}

/**
 * \brief           Retrieve entropy from the accumulator
 *                  (Maximum length: MBEDTLS_ENTROPY_BLOCK_SIZE)
 *
 * \param data      Entropy context
 * \param output    Buffer to fill
 * \param len       Number of bytes desired, must be at most MBEDTLS_ENTROPY_BLOCK_SIZE
 *
 * \return          0 if successful, or MBEDTLS_ERR_ENTROPY_SOURCE_FAILED
 */
int mbedtls_entropy_func( void *data, unsigned char *output, size_t len )
{
    int ret, count = 0, i, thresholds_reached;
    size_t strong_size;
    mbedtls_entropy_context *ctx = (mbedtls_entropy_context *) data;
    unsigned char buf[MBEDTLS_ENTROPY_BLOCK_SIZE];

    if( len > MBEDTLS_ENTROPY_BLOCK_SIZE )
        return( MBEDTLS_ERR_ENTROPY_SOURCE_FAILED );

#if defined(MBEDTLS_ENTROPY_NV_SEED)
    /* Update the NV entropy seed before generating any entropy for outside
     * use.
     */
    if( ctx->initial_entropy_run == 0 )
    {
        ctx->initial_entropy_run = 1;
        if( ( ret = mbedtls_entropy_update_nv_seed( ctx ) ) != 0 )
            return( ret );
    }
#endif

    /*
     * Always gather extra entropy before a call
     */
    do
    {
        if( count++ > ENTROPY_MAX_LOOP )
        {
            ret = MBEDTLS_ERR_ENTROPY_SOURCE_FAILED;
            goto exit;
        }

        if( ( ret = entropy_gather_internal( ctx ) ) != 0 )
            goto exit;

        thresholds_reached = 1;
        strong_size = 0;
        for( i = 0; i < ctx->source_count; i++ )
        {
            if( ctx->source[i].size < ctx->source[i].threshold )
                thresholds_reached = 0;
            if( ctx->source[i].strong == MBEDTLS_ENTROPY_SOURCE_STRONG )
                strong_size += ctx->source[i].size;
        }
    }
    while( ! thresholds_reached || strong_size < MBEDTLS_ENTROPY_BLOCK_SIZE );

    mbedtls_platform_zeroize( buf, MBEDTLS_ENTROPY_BLOCK_SIZE );

#if defined(MBEDTLS_ENTROPY_SHA512_ACCUMULATOR)
    /*
     * Note that at this stage it is assumed that the accumulator was started
     * in a previous call to entropy_update(). If this is not guaranteed, the
     * code below will fail.
     */
    if( ( ret = mbedtls_sha512_finish_ret( &ctx->accumulator, buf ) ) != 0 )
        goto exit;

    /*
     * Reset accumulator and counters and recycle existing entropy
     */
    mbedtls_sha512_free( &ctx->accumulator );
    mbedtls_sha512_init( &ctx->accumulator );
    if( ( ret = mbedtls_sha512_starts_ret( &ctx->accumulator, 0 ) ) != 0 )
        goto exit;
    if( ( ret = mbedtls_sha512_update_ret( &ctx->accumulator, buf,
                                           MBEDTLS_ENTROPY_BLOCK_SIZE ) ) != 0 )
        goto exit;

    /*
     * Perform second SHA-512 on entropy
     */
    if( ( ret = mbedtls_sha512_ret( buf, MBEDTLS_ENTROPY_BLOCK_SIZE,
                                    buf, 0 ) ) != 0 )
        goto exit;
#else /* MBEDTLS_ENTROPY_SHA512_ACCUMULATOR */
    if( ( ret = mbedtls_sha256_finish_ret( &ctx->accumulator, buf ) ) != 0 )
        goto exit;

    /*
     * Reset accumulator and counters and recycle existing entropy
     */
    mbedtls_sha256_free( &ctx->accumulator );
    mbedtls_sha256_init( &ctx->accumulator );
    if( ( ret = mbedtls_sha256_starts_ret( &ctx->accumulator, 0 ) ) != 0 )
        goto exit;
    if( ( ret = mbedtls_sha256_update_ret( &ctx->accumulator, buf,
                                           MBEDTLS_ENTROPY_BLOCK_SIZE ) ) != 0 )
        goto exit;

    /*
     * Perform second SHA-256 on entropy
     */
    if( ( ret = mbedtls_sha256_ret( buf, MBEDTLS_ENTROPY_BLOCK_SIZE,
                                    buf, 0 ) ) != 0 )
        goto exit;
#endif /* MBEDTLS_ENTROPY_SHA512_ACCUMULATOR */

    for( i = 0; i < ctx->source_count; i++ )
        ctx->source[i].size = 0;

    memcpy( output, buf, len );

    ret = 0;

exit:
    mbedtls_platform_zeroize( buf, sizeof( buf ) );

    return( ret );
}

#if defined(MBEDTLS_ENTROPY_NV_SEED)
/**
 * \brief           Trigger an update of the seed file in NV by using the
 *                  current entropy pool.
 *
 * \param ctx       Entropy context
 *
 * \return          0 if successful
 */
int mbedtls_entropy_update_nv_seed( mbedtls_entropy_context *ctx )
{
    int ret = MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR;
    unsigned char buf[MBEDTLS_ENTROPY_BLOCK_SIZE];
    /* Read new seed  and write it to NV */
    if( ( ret = mbedtls_entropy_func( ctx, buf, MBEDTLS_ENTROPY_BLOCK_SIZE ) ) != 0 )
        return( ret );
    if( mbedtls_nv_seed_write( buf, MBEDTLS_ENTROPY_BLOCK_SIZE ) < 0 )
        return( MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR );
    /* Manually update the remaining stream with a separator value to diverge */
    mbedtls_platform_zeroize( buf, MBEDTLS_ENTROPY_BLOCK_SIZE );
    ret = mbedtls_entropy_update_manual( ctx, buf, MBEDTLS_ENTROPY_BLOCK_SIZE );
    return( ret );
}
#endif /* MBEDTLS_ENTROPY_NV_SEED */

#if defined(MBEDTLS_FS_IO)
/**
 * \brief               Write a seed file
 *
 * \param ctx           Entropy context
 * \param path          Name of the file
 *
 * \return              0 if successful,
 *                      MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR on file error, or
 *                      MBEDTLS_ERR_ENTROPY_SOURCE_FAILED
 */
int mbedtls_entropy_write_seed_file( mbedtls_entropy_context *ctx, const char *path )
{
    int ret = MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR;
    FILE *f;
    unsigned char buf[MBEDTLS_ENTROPY_BLOCK_SIZE];

    if( ( f = fopen( path, "wb" ) ) == NULL )
        return( MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR );

    if( ( ret = mbedtls_entropy_func( ctx, buf, MBEDTLS_ENTROPY_BLOCK_SIZE ) ) != 0 )
        goto exit;

    if( fwrite( buf, 1, MBEDTLS_ENTROPY_BLOCK_SIZE, f ) != MBEDTLS_ENTROPY_BLOCK_SIZE )
    {
        ret = MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR;
        goto exit;
    }

    ret = 0;

exit:
    mbedtls_platform_zeroize( buf, sizeof( buf ) );

    fclose( f );
    return( ret );
}

/**
 * \brief               Read and update a seed file. Seed is added to this
 *                      instance. No more than MBEDTLS_ENTROPY_MAX_SEED_SIZE bytes are
 *                      read from the seed file. The rest is ignored.
 *
 * \param ctx           Entropy context
 * \param path          Name of the file
 *
 * \return              0 if successful,
 *                      MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR on file error,
 *                      MBEDTLS_ERR_ENTROPY_SOURCE_FAILED
 */
int mbedtls_entropy_update_seed_file( mbedtls_entropy_context *ctx, const char *path )
{
    int ret = 0;
    FILE *f;
    size_t n;
    unsigned char buf[ MBEDTLS_ENTROPY_MAX_SEED_SIZE ];

    if( ( f = fopen( path, "rb" ) ) == NULL )
        return( MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR );

    fseek( f, 0, SEEK_END );
    n = (size_t) ftell( f );
    fseek( f, 0, SEEK_SET );

    if( n > MBEDTLS_ENTROPY_MAX_SEED_SIZE )
        n = MBEDTLS_ENTROPY_MAX_SEED_SIZE;

    if( fread( buf, 1, n, f ) != n )
        ret = MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR;
    else
        ret = mbedtls_entropy_update_manual( ctx, buf, n );

    fclose( f );

    mbedtls_platform_zeroize( buf, sizeof( buf ) );

    if( ret != 0 )
        return( ret );

    return( mbedtls_entropy_write_seed_file( ctx, path ) );
}
#endif /* MBEDTLS_FS_IO */

#if defined(MBEDTLS_SELF_TEST)
#if !defined(MBEDTLS_TEST_NULL_ENTROPY)
/*
 * Dummy source function
 */
static int entropy_dummy_source( void *data, unsigned char *output,
                                 size_t len, size_t *olen )
{
    ((void) data);

    memset( output, 0x2a, len );
    *olen = len;

    return( 0 );
}
#endif /* !MBEDTLS_TEST_NULL_ENTROPY */

#if defined(MBEDTLS_ENTROPY_HARDWARE_ALT)

static int mbedtls_entropy_source_self_test_gather( unsigned char *buf, size_t buf_len )
{
    int ret = 0;
    size_t entropy_len = 0;
    size_t olen = 0;
    size_t attempts = buf_len;

    while( attempts > 0 && entropy_len < buf_len )
    {
        if( ( ret = mbedtls_hardware_poll( NULL, buf + entropy_len,
            buf_len - entropy_len, &olen ) ) != 0 )
            return( ret );

        entropy_len += olen;
        attempts--;
    }

    if( entropy_len < buf_len )
    {
        ret = 1;
    }

    return( ret );
}


static int mbedtls_entropy_source_self_test_check_bits( const unsigned char *buf,
                                                        size_t buf_len )
{
    unsigned char set= 0xFF;
    unsigned char unset = 0x00;
    size_t i;

    for( i = 0; i < buf_len; i++ )
    {
        set &= buf[i];
        unset |= buf[i];
    }

    return( set == 0xFF || unset == 0x00 );
}

/**
 * \brief          Checkup routine
 *
 *                 Verifies the integrity of the hardware entropy source
 *                 provided by the function 'mbedtls_hardware_poll()'.
 *
 *                 Note this is the only hardware entropy source that is known
 *                 at link time, and other entropy sources configured
 *                 dynamically at runtime by the function
 *                 mbedtls_entropy_add_source() will not be tested.
 *
 * \return         0 if successful, or 1 if a test failed
 */
int mbedtls_entropy_source_self_test( int verbose )
{
    /*
     * A test to ensure hat the entropy sources are functioning correctly
     * and there is no obvious failure. The test performs the following checks:
     *  - The entropy source is not providing only 0s (all bits unset) or 1s (all
     *    bits set).
     *  - The entropy source is not providing values in a pattern. Because the
     *    hardware could be providing data in an arbitrary length, this check polls
     *    the hardware entropy source twice and compares the result to ensure they
     *    are not equal.
     *  - The error code returned by the entropy source is not an error.
     */
    int ret = 0;
    unsigned char buf0[2 * sizeof( unsigned long long int )];
    unsigned char buf1[2 * sizeof( unsigned long long int )];

    if( verbose != 0 )
        mbedtls_printf( "  ENTROPY_BIAS test: " );

    mbedtls_platform_zeroize( buf0, sizeof( buf0 ) );
    mbedtls_platform_zeroize( buf1, sizeof( buf1 ) );

    if( ( ret = mbedtls_entropy_source_self_test_gather( buf0, sizeof( buf0 ) ) ) != 0 )
        goto cleanup;
    if( ( ret = mbedtls_entropy_source_self_test_gather( buf1, sizeof( buf1 ) ) ) != 0 )
        goto cleanup;

    /* Make sure that the returned values are not all 0 or 1 */
    if( ( ret = mbedtls_entropy_source_self_test_check_bits( buf0, sizeof( buf0 ) ) ) != 0 )
        goto cleanup;
    if( ( ret = mbedtls_entropy_source_self_test_check_bits( buf1, sizeof( buf1 ) ) ) != 0 )
        goto cleanup;

    /* Make sure that the entropy source is not returning values in a
     * pattern */
    ret = timingsafe_bcmp( buf0, buf1, sizeof( buf0 ) ) == 0;

cleanup:
    if( verbose != 0 )
    {
        if( ret != 0 )
            mbedtls_printf( "failed\n" );
        else
            mbedtls_printf( "passed\n" );

        mbedtls_printf( "\n" );
    }

    return( ret != 0 );
}

#endif /* MBEDTLS_ENTROPY_HARDWARE_ALT */

/**
 * \brief          Checkup routine
 *
 *                 The actual entropy quality is hard to test, but we
 *                 can at least test that the functions don't cause
 *                 errors and write the correct amount of data to
 *                 buffers.
 *
 *                 This module self-test also calls the entropy self-test,
 *                 mbedtls_entropy_source_self_test();
 *
 * \return         0 if successful, or 1 if a test failed
 */
int mbedtls_entropy_self_test( int verbose )
{
    int ret = 1;
#if !defined(MBEDTLS_TEST_NULL_ENTROPY)
    mbedtls_entropy_context ctx;
    unsigned char buf[MBEDTLS_ENTROPY_BLOCK_SIZE] = { 0 };
    unsigned char acc[MBEDTLS_ENTROPY_BLOCK_SIZE] = { 0 };
    size_t i, j;
#endif /* !MBEDTLS_TEST_NULL_ENTROPY */

    if( verbose != 0 )
        mbedtls_printf( "  ENTROPY test: " );

#if !defined(MBEDTLS_TEST_NULL_ENTROPY)
    mbedtls_entropy_init( &ctx );

    /* First do a gather to make sure we have default sources */
    if( ( ret = mbedtls_entropy_gather( &ctx ) ) != 0 )
        goto cleanup;

    ret = mbedtls_entropy_add_source( &ctx, entropy_dummy_source, NULL, 16,
                                      MBEDTLS_ENTROPY_SOURCE_WEAK );
    if( ret != 0 )
        goto cleanup;

    if( ( ret = mbedtls_entropy_update_manual( &ctx, buf, sizeof buf ) ) != 0 )
        goto cleanup;

    /*
     * To test that mbedtls_entropy_func writes correct number of bytes:
     * - use the whole buffer and rely on ASan to detect overruns
     * - collect entropy 8 times and OR the result in an accumulator:
     *   any byte should then be 0 with probably 2^(-64), so requiring
     *   each of the 32 or 64 bytes to be non-zero has a false failure rate
     *   of at most 2^(-58) which is acceptable.
     */
    for( i = 0; i < 8; i++ )
    {
        if( ( ret = mbedtls_entropy_func( &ctx, buf, sizeof( buf ) ) ) != 0 )
            goto cleanup;

        for( j = 0; j < sizeof( buf ); j++ )
            acc[j] |= buf[j];
    }

    for( j = 0; j < sizeof( buf ); j++ )
    {
        if( acc[j] == 0 )
        {
            ret = 1;
            goto cleanup;
        }
    }

#if defined(MBEDTLS_ENTROPY_HARDWARE_ALT)
    if( ( ret = mbedtls_entropy_source_self_test( 0 ) ) != 0 )
        goto cleanup;
#endif

cleanup:
    mbedtls_entropy_free( &ctx );
#endif /* !MBEDTLS_TEST_NULL_ENTROPY */

    if( verbose != 0 )
    {
        if( ret != 0 )
            mbedtls_printf( "failed\n" );
        else
            mbedtls_printf( "passed\n" );

        mbedtls_printf( "\n" );
    }

    return( ret != 0 );
}
#endif /* MBEDTLS_SELF_TEST */

#endif /* MBEDTLS_ENTROPY_C */
