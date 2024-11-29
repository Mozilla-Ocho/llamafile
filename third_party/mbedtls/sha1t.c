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
#include <libc/str/str.h>
#include "third_party/mbedtls/platform.h"
#include "third_party/mbedtls/sha1.h"
__static_yoink("mbedtls_notice");

/*
 * FIPS-180-1 test vectors
 */
static const unsigned char sha1_test_buf[3][57] =
{
    { "abc" },
    { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" },
    { "" }
};

static const size_t sha1_test_buflen[3] =
{
    3, 56, 1000
};

static const unsigned char sha1_test_sum[3][20] =
{
    { 0xA9, 0x99, 0x3E, 0x36, 0x47, 0x06, 0x81, 0x6A, 0xBA, 0x3E,
      0x25, 0x71, 0x78, 0x50, 0xC2, 0x6C, 0x9C, 0xD0, 0xD8, 0x9D },
    { 0x84, 0x98, 0x3E, 0x44, 0x1C, 0x3B, 0xD2, 0x6E, 0xBA, 0xAE,
      0x4A, 0xA1, 0xF9, 0x51, 0x29, 0xE5, 0xE5, 0x46, 0x70, 0xF1 },
    { 0x34, 0xAA, 0x97, 0x3C, 0xD4, 0xC4, 0xDA, 0xA4, 0xF6, 0x1E,
      0xEB, 0x2B, 0xDB, 0xAD, 0x27, 0x31, 0x65, 0x34, 0x01, 0x6F }
};

/**
 * \brief          The SHA-1 checkup routine.
 *
 * \warning        SHA-1 is considered a weak message digest and its use
 *                 constitutes a security risk. We recommend considering
 *                 stronger message digests instead.
 *
 * \return         \c 0 on success.
 * \return         \c 1 on failure.
 *
 */
int mbedtls_sha1_self_test( int verbose )
{
    int i, j, buflen, ret = 0;
    unsigned char buf[1024];
    unsigned char sha1sum[20];
    mbedtls_sha1_context ctx;
    mbedtls_sha1_init( &ctx );
    /*
     * SHA-1
     */
    for( i = 0; i < 3; i++ )
    {
        if( verbose != 0 )
            mbedtls_printf( "  SHA-1 test #%d: ", i + 1 );
        if( ( ret = mbedtls_sha1_starts_ret( &ctx ) ) != 0 )
            goto fail;
        if( i == 2 )
        {
            memset( buf, 'a', buflen = 1000 );
            for( j = 0; j < 1000; j++ )
            {
                ret = mbedtls_sha1_update_ret( &ctx, buf, buflen );
                if( ret != 0 )
                    goto fail;
            }
        }
        else
        {
            ret = mbedtls_sha1_update_ret( &ctx, sha1_test_buf[i],
                                           sha1_test_buflen[i] );
            if( ret != 0 )
                goto fail;
        }
        if( ( ret = mbedtls_sha1_finish_ret( &ctx, sha1sum ) ) != 0 )
            goto fail;
        if( timingsafe_bcmp( sha1sum, sha1_test_sum[i], 20 ) != 0 )
        {
            ret = 1;
            goto fail;
        }
        if( verbose != 0 )
            mbedtls_printf( "passed\n" );
    }
    if( verbose != 0 )
        mbedtls_printf( "\n" );
    goto exit;
fail:
    if( verbose != 0 )
        mbedtls_printf( "failed\n" );
exit:
    mbedtls_sha1_free( &ctx );
    return( ret );
}
