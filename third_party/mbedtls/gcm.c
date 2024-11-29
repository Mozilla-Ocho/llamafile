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
#include "third_party/mbedtls/gcm.h"
#include <libc/serialize.h>
#include <libc/intrin/likely.h>
#include <libc/log/log.h>
#include <libc/nexgen32e/x86feature.h>
#include <libc/runtime/runtime.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/aes.h"
#include "third_party/mbedtls/aesni.h"
#include "third_party/mbedtls/cipher.h"
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/endian.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/platform.h"
__static_yoink("mbedtls_notice");

/*
 *  NIST SP800-38D compliant GCM implementation
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
 * http://csrc.nist.gov/publications/nistpubs/800-38D/SP-800-38D.pdf
 *
 * See also:
 * [MGV] http://csrc.nist.gov/groups/ST/toolkit/BCM/documents/proposedmodes/gcm/gcm-revised-spec.pdf
 *
 * We use the algorithm described as Shoup's method with 4-bit tables in
 * [MGV] 4.1, pp. 12-13, to enhance speed without using too much memory.
 */

#if !defined(MBEDTLS_GCM_ALT)

/* Parameter validation macros */
#define GCM_VALIDATE_RET( cond ) \
    MBEDTLS_INTERNAL_VALIDATE_RET( cond, MBEDTLS_ERR_GCM_BAD_INPUT )
#define GCM_VALIDATE( cond ) \
    MBEDTLS_INTERNAL_VALIDATE( cond )

/**
 * \brief           This function initializes the specified GCM context,
 *                  to make references valid, and prepares the context
 *                  for mbedtls_gcm_setkey() or mbedtls_gcm_free().
 *
 *                  The function does not bind the GCM context to a particular
 *                  cipher, nor set the key. For this purpose, use
 *                  mbedtls_gcm_setkey().
 *
 * \param ctx       The GCM context to initialize. This must not be \c NULL.
 */
void mbedtls_gcm_init( mbedtls_gcm_context *ctx )
{
    GCM_VALIDATE( ctx != NULL );
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_gcm_context ) );
}

/*
 * Precompute small multiples of H, that is set
 *      HH[i] || HL[i] = H times i,
 * where i is seen as a field element as in [MGV], ie high-order bits
 * correspond to low powers of P. The result is stored in the same way, that
 * is the high-order bit of HH corresponds to P^0 and the low-order bit of HL
 * corresponds to P^127.
 */
static int gcm_gen_table( mbedtls_gcm_context *ctx )
{
    int ret, i, j;
    uint64_t vl, vh;
    unsigned char h[16];
    size_t olen = 0;
    mbedtls_platform_zeroize( h, 16 );
    if( ( ret = mbedtls_cipher_update( &ctx->cipher_ctx, h, 16, h, &olen ) ) != 0 )
        return( ret );
    vh = READ64BE( h + 0 );
    vl = READ64BE( h + 8 );
#if defined(MBEDTLS_AESNI_C) && defined(MBEDTLS_HAVE_X86_64)
    /* With CLMUL support, we need only h, not the rest of the table */
    if (X86_HAVE(AES) && X86_HAVE(PCLMUL)) {
        ctx->H8[0] = vl;
        ctx->H8[1] = vh;
        return 0;
    }
#endif
    /* 0 corresponds to 0 in GF(2^128) */
    ctx->HH[0] = 0;
    ctx->HL[0] = 0;
    /* 8 = 1000 corresponds to 1 in GF(2^128) */
    ctx->HL[8] = vl;
    ctx->HH[8] = vh;
    for( i = 4; i > 0; i >>= 1 ) {
        uint32_t T = ( vl & 1 ) * 0xe1000000U;
        vl  = ( vh << 63 ) | ( vl >> 1 );
        vh  = ( vh >> 1 ) ^ ( (uint64_t) T << 32);
        ctx->HL[i] = vl;
        ctx->HH[i] = vh;
    }
    for( i = 2; i <= 8; i *= 2 ) {
        uint64_t *HiL = ctx->HL + i, *HiH = ctx->HH + i;
        vh = *HiH;
        vl = *HiL;
        for( j = 1; j < i; j++ ) {
            HiH[j] = vh ^ ctx->HH[j];
            HiL[j] = vl ^ ctx->HL[j];
        }
    }
    return( 0 );
}

/**
 * \brief           This function associates a GCM context with a
 *                  cipher algorithm and a key.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param cipher    The 128-bit block cipher to use.
 * \param key       The encryption key. This must be a readable buffer of at
 *                  least \p keybits bits.
 * \param keybits   The key size in bits. Valid options are:
 *                  <ul><li>128 bits</li>
 *                  <li>192 bits</li>
 *                  <li>256 bits</li></ul>
 *
 * \return          \c 0 on success.
 * \return          A cipher-specific error code on failure.
 */
int mbedtls_gcm_setkey( mbedtls_gcm_context *ctx,
                        mbedtls_cipher_id_t cipher,
                        const unsigned char *key,
                        unsigned int keybits )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    const mbedtls_cipher_info_t *cipher_info;
    GCM_VALIDATE_RET( ctx != NULL );
    GCM_VALIDATE_RET( key != NULL );
    GCM_VALIDATE_RET( keybits == 128 || keybits == 192 || keybits == 256 );
    cipher_info = mbedtls_cipher_info_from_values( cipher, keybits,
                                                   MBEDTLS_MODE_ECB );
    if( cipher_info == NULL )
        return( MBEDTLS_ERR_GCM_BAD_INPUT );
    if( cipher_info->block_size != 16 )
        return( MBEDTLS_ERR_GCM_BAD_INPUT );
    mbedtls_cipher_free( &ctx->cipher_ctx );
    ctx->cipher = cipher;
    if( ( ret = mbedtls_cipher_setup( &ctx->cipher_ctx, cipher_info ) ) != 0 )
        return( ret );
    if( ( ret = mbedtls_cipher_setkey( &ctx->cipher_ctx, key, keybits,
                                       MBEDTLS_ENCRYPT ) ) != 0 ) {
        return( ret );
    }
    if( ( ret = gcm_gen_table( ctx ) ) != 0 )
        return( ret );
    return( 0 );
}

/*
 * Shoup's method for multiplication use this table with
 *      last4[x] = x times P^128
 * where x and last4[x] are seen as elements of GF(2^128) as in [MGV]
 */
static const uint64_t last4[16] =
{
    0x0000, 0x1c20, 0x3840, 0x2460,
    0x7080, 0x6ca0, 0x48c0, 0x54e0,
    0xe100, 0xfd20, 0xd940, 0xc560,
    0x9180, 0x8da0, 0xa9c0, 0xb5e0
};

/*
 * Sets output to x times H using the precomputed tables.
 * x and output are seen as elements of GF(2^128) as in [MGV].
 */
static void gcm_mult( mbedtls_gcm_context *ctx,  unsigned char x[16] )
{
    int i;
    uint64_t zh, zl;
    unsigned char lo, hi, rem;
#if defined(MBEDTLS_AESNI_C) && defined(MBEDTLS_HAVE_X86_64)
    if (LIKELY(X86_HAVE(AES) && X86_HAVE(PCLMUL))) {
        mbedtls_aesni_gcm_mult( x, ctx->H8 );
        return;
    }
#endif /* MBEDTLS_AESNI_C && MBEDTLS_HAVE_X86_64 */
    lo = x[15] & 0xf;
    zh = ctx->HH[lo];
    zl = ctx->HL[lo];
    for( i = 15; i >= 0; i-- ) {
        lo = x[i] & 0xf;
        hi = ( x[i] >> 4 ) & 0xf;
        if( i != 15 ) {
            rem = (unsigned char) zl & 0xf;
            zl = ( zh << 60 ) | ( zl >> 4 );
            zh = ( zh >> 4 );
            zh ^= (uint64_t) last4[rem] << 48;
            zh ^= ctx->HH[lo];
            zl ^= ctx->HL[lo];
        }
        rem = (unsigned char) zl & 0xf;
        zl = ( zh << 60 ) | ( zl >> 4 );
        zh = ( zh >> 4 );
        zh ^= (uint64_t) last4[rem] << 48;
        zh ^= ctx->HH[hi];
        zl ^= ctx->HL[hi];
    }
    PUT_UINT64_BE( zh, x, 0 );
    PUT_UINT64_BE( zl, x, 8 );
}

/**
 * \brief           This function starts a GCM encryption or decryption
 *                  operation.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param mode      The operation to perform: #MBEDTLS_GCM_ENCRYPT or
 *                  #MBEDTLS_GCM_DECRYPT.
 * \param iv        The initialization vector. This must be a readable buffer of
 *                  at least \p iv_len Bytes.
 * \param iv_len    The length of the IV.
 * \param add       The buffer holding the additional data, or \c NULL
 *                  if \p add_len is \c 0.
 * \param add_len   The length of the additional data. If \c 0,
 *                  \p add may be \c NULL.
 *
 * \return          \c 0 on success.
 */
int mbedtls_gcm_starts( mbedtls_gcm_context *ctx,
                        int mode,
                        const unsigned char *iv,
                        size_t iv_len,
                        const unsigned char *add,
                        size_t add_len )
{
    size_t i;
    const unsigned char *p;
    size_t use_len, olen = 0;
    unsigned char work_buf[16];
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    GCM_VALIDATE_RET( ctx != NULL );
    GCM_VALIDATE_RET( iv != NULL );
    GCM_VALIDATE_RET( add_len == 0 || add != NULL );
    /* IV and AD are limited to 2^64 bits, so 2^61 bytes */
    /* IV is not allowed to be zero length */
    if( iv_len == 0 ||
      ( (uint64_t) iv_len  ) >> 61 != 0 ||
      ( (uint64_t) add_len ) >> 61 != 0 ) {
        return( MBEDTLS_ERR_GCM_BAD_INPUT );
    }
    mbedtls_platform_zeroize( ctx->y, sizeof(ctx->y) );
    mbedtls_platform_zeroize( ctx->buf, sizeof(ctx->buf) );
    ctx->mode = mode;
    ctx->len = 0;
    ctx->add_len = 0;
    if( iv_len == 12 ) {
        memcpy( ctx->y, iv, iv_len );
        ctx->y[15] = 1;
    } else {
        mbedtls_platform_zeroize( work_buf, 16 );
        PUT_UINT32_BE( iv_len * 8, work_buf, 12 );
        p = iv;
        while( iv_len > 0 ) {
            use_len = ( iv_len < 16 ) ? iv_len : 16;
            for( i = 0; i < use_len; i++ )
                ctx->y[i] ^= p[i];
            gcm_mult( ctx, ctx->y );
            iv_len -= use_len;
            p += use_len;
        }
        for( i = 0; i < 16; i++ )
            ctx->y[i] ^= work_buf[i];
        gcm_mult( ctx, ctx->y );
    }
    if( ( ret = mbedtls_cipher_update( &ctx->cipher_ctx, ctx->y, 16,
                                       ctx->base_ectr, &olen ) ) != 0 ) {
        return( ret );
    }
    ctx->add_len = add_len;
    p = add;
    while( add_len > 0 ) {
        use_len = ( add_len < 16 ) ? add_len : 16;
        for( i = 0; i < use_len; i++ )
            ctx->buf[i] ^= p[i];
        gcm_mult( ctx, ctx->buf );
        add_len -= use_len;
        p += use_len;
    }
    return( 0 );
}

/**
 * \brief           This function feeds an input buffer into an ongoing GCM
 *                  encryption or decryption operation.
 *
 *                  The function expects input to be a multiple of 16
 *                  Bytes. Only the last call before calling
 *                  mbedtls_gcm_finish() can be less than 16 Bytes.
 *
 * \note            For decryption, the output buffer cannot be the same as
 *                  input buffer. If the buffers overlap, the output buffer
 *                  must trail at least 8 Bytes behind the input buffer.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param length    The length of the input data. This must be a multiple of
 *                  16 except in the last call before mbedtls_gcm_finish().
 * \param input     The buffer holding the input data. If \p length is greater
 *                  than zero, this must be a readable buffer of at least that
 *                  size in Bytes.
 * \param output    The buffer for holding the output data. If \p length is
 *                  greater than zero, this must be a writable buffer of at
 *                  least that size in Bytes.
 *
 * \return         \c 0 on success.
 * \return         #MBEDTLS_ERR_GCM_BAD_INPUT on failure.
 */
int mbedtls_gcm_update( mbedtls_gcm_context *ctx,
                        size_t length,
                        const unsigned char *input,
                        unsigned char *output )
{
    size_t i, j;
    uint64_t a, b;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char ectr[16];
    const unsigned char *p;
    unsigned char *q, *out_p = output;
    size_t olen = 0;
    GCM_VALIDATE_RET( ctx );
    GCM_VALIDATE_RET( !length || input );
    GCM_VALIDATE_RET( !length || output );
    if( output > input && (size_t) ( output - input ) < length )
        return( MBEDTLS_ERR_GCM_BAD_INPUT );
    /* Total length is restricted to 2^39 - 256 bits, ie 2^36 - 2^5 bytes
     * Also check for possible overflow */
    if( ctx->len + length < ctx->len ||
        (uint64_t) ctx->len + length > 0xFFFFFFFE0ull ) {
        return( MBEDTLS_ERR_GCM_BAD_INPUT );
    }
    ctx->len += length;
    p = input;
    q = ctx->buf;
    for( j = 0; j + 16 <= length; j += 16 ){
        for( i = 16; i > 12; i-- )
            if( ++ctx->y[i - 1] != 0 )
                break;
        if( !( ret = mbedtls_cipher_update( &ctx->cipher_ctx, ctx->y, 16,
                                            ectr, &olen ) ) ) {
            if( ctx->mode == MBEDTLS_GCM_DECRYPT ) {
                __builtin_memcpy(&a, p+j, 8);
                __builtin_memcpy(&b, q, 8);
                b ^= a;
                __builtin_memcpy(q, &b, 8);
                __builtin_memcpy(&b, ectr, 8);
                b ^= a;
                __builtin_memcpy(out_p+j, &b, 8);
                __builtin_memcpy(&a, p+j+8, 8);
                __builtin_memcpy(&b, q+8, 8);
                b ^= a;
                __builtin_memcpy(q+8, &b, 8);
                __builtin_memcpy(&b, ectr+8, 8);
                b ^= a;
                __builtin_memcpy(out_p+j+8, &b, 8);
                /* for( i = 0; i < 16; i++ ) ctx->buf[i] ^= p[i]; */
                /* for( i = 0; i < 16; i++ ) out_p[i] = ectr[i] ^ p[i]; */
            } else {
                __builtin_memcpy(&a, ectr, 8);
                __builtin_memcpy(&b, p+j, 8);
                b ^= a;
                __builtin_memcpy(out_p+j, &b, 8);
                __builtin_memcpy(&a, q, 8);
                b ^= a;
                __builtin_memcpy(q, &b, 8);
                __builtin_memcpy(&a, ectr+8, 8);
                __builtin_memcpy(&b, p+j+8, 8);
                b ^= a;
                __builtin_memcpy(out_p+j+8, &b, 8);
                __builtin_memcpy(&a, q+8, 8);
                b ^= a;
                __builtin_memcpy(q+8, &b, 8);
                /* for( i = 0; i < 16; i++ ) out_p[i] = ectr[i] ^ p[i]; */
                /* for( i = 0; i < 16; i++ ) ctx->buf[i] ^= out_p[i]; */
            }
            gcm_mult( ctx, q );
        } else {
            return( ret );
        }
    }
    length -= j;
    out_p += j;
    p += j;
    if( length ) {
        for( i = 16; i > 12; i-- )
            if( ++ctx->y[i - 1] != 0 )
                break;
        if( !( ret = mbedtls_cipher_update( &ctx->cipher_ctx, ctx->y, 16, ectr,
                                            &olen ) ) ) {
            if( ctx->mode == MBEDTLS_GCM_DECRYPT ) {
                for( i = 0; i < length; i++ ){
                    q[i] ^= p[i];
                    out_p[i] = ectr[i] ^ p[i];
                }
            } else {
                for( i = 0; i < length; i++ ){
                    out_p[i] = ectr[i] ^ p[i];
                    q[i] ^= out_p[i];
                }
            }
            gcm_mult( ctx, q );
        } else {
            return( ret );
        }
    }
    return( 0 );
}

/**
 * \brief           This function finishes the GCM operation and generates
 *                  the authentication tag.
 *
 *                  It wraps up the GCM stream, and generates the
 *                  tag. The tag can have a maximum length of 16 Bytes.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param tag       The buffer for holding the tag. This must be a writable
 *                  buffer of at least \p tag_len Bytes.
 * \param tag_len   The length of the tag to generate. This must be at least
 *                  four.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_GCM_BAD_INPUT on failure.
 */
int mbedtls_gcm_finish( mbedtls_gcm_context *ctx,
                        unsigned char *tag,
                        size_t tag_len )
{
    size_t i;
    uint64_t orig_len;
    uint64_t orig_add_len;
    GCM_VALIDATE_RET( ctx != NULL );
    GCM_VALIDATE_RET( tag != NULL );
    orig_len = ctx->len * 8;
    orig_add_len = ctx->add_len * 8;
    if( tag_len > 16 || tag_len < 4 )
        return( MBEDTLS_ERR_GCM_BAD_INPUT );
    memcpy( tag, ctx->base_ectr, tag_len );
    if( orig_len || orig_add_len ) {
        Write64be( ctx->buf + 0, READ64BE( ctx->buf + 0 ) ^ orig_add_len );
        Write64be( ctx->buf + 8, READ64BE( ctx->buf + 8 ) ^ orig_len     );
        gcm_mult( ctx, ctx->buf );
        for( i = 0; i < tag_len; i++ ) tag[i] ^= ctx->buf[i];
    }
    return( 0 );
}

/**
 * \brief           This function performs GCM encryption or decryption of a buffer.
 *
 * \note            For encryption, the output buffer can be the same as the
 *                  input buffer. For decryption, the output buffer cannot be
 *                  the same as input buffer. If the buffers overlap, the output
 *                  buffer must trail at least 8 Bytes behind the input buffer.
 *
 * \warning         When this function performs a decryption, it outputs the
 *                  authentication tag and does not verify that the data is
 *                  authentic. You should use this function to perform encryption
 *                  only. For decryption, use mbedtls_gcm_auth_decrypt() instead.
 *
 * \param ctx       The GCM context to use for encryption or decryption. This
 *                  must be initialized.
 * \param mode      The operation to perform:
 *                  - #MBEDTLS_GCM_ENCRYPT to perform authenticated encryption.
 *                    The ciphertext is written to \p output and the
 *                    authentication tag is written to \p tag.
 *                  - #MBEDTLS_GCM_DECRYPT to perform decryption.
 *                    The plaintext is written to \p output and the
 *                    authentication tag is written to \p tag.
 *                    Note that this mode is not recommended, because it does
 *                    not verify the authenticity of the data. For this reason,
 *                    you should use mbedtls_gcm_auth_decrypt() instead of
 *                    calling this function in decryption mode.
 * \param length    The length of the input data, which is equal to the length
 *                  of the output data.
 * \param iv        The initialization vector. This must be a readable buffer of
 *                  at least \p iv_len Bytes.
 * \param iv_len    The length of the IV.
 * \param add       The buffer holding the additional data. This must be of at
 *                  least that size in Bytes.
 * \param add_len   The length of the additional data.
 * \param input     The buffer holding the input data. If \p length is greater
 *                  than zero, this must be a readable buffer of at least that
 *                  size in Bytes.
 * \param output    The buffer for holding the output data. If \p length is greater
 *                  than zero, this must be a writable buffer of at least that
 *                  size in Bytes.
 * \param tag_len   The length of the tag to generate.
 * \param tag       The buffer for holding the tag. This must be a writable
 *                  buffer of at least \p tag_len Bytes.
 *
 * \return          \c 0 if the encryption or decryption was performed
 *                  successfully. Note that in #MBEDTLS_GCM_DECRYPT mode,
 *                  this does not indicate that the data is authentic.
 * \return          #MBEDTLS_ERR_GCM_BAD_INPUT if the lengths or pointers are
 *                  not valid or a cipher-specific error code if the encryption
 *                  or decryption failed.
 */
int mbedtls_gcm_crypt_and_tag( mbedtls_gcm_context *ctx,
                               int mode,
                               size_t length,
                               const unsigned char *iv,
                               size_t iv_len,
                               const unsigned char *add,
                               size_t add_len,
                               const unsigned char *input,
                               unsigned char *output,
                               size_t tag_len,
                               unsigned char *tag )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    GCM_VALIDATE_RET( ctx != NULL );
    GCM_VALIDATE_RET( iv != NULL );
    GCM_VALIDATE_RET( add_len == 0 || add != NULL );
    GCM_VALIDATE_RET( length == 0 || input != NULL );
    GCM_VALIDATE_RET( length == 0 || output != NULL );
    GCM_VALIDATE_RET( tag != NULL );
    if( ( ret = mbedtls_gcm_starts( ctx, mode, iv, iv_len, add, add_len ) ) != 0 )
        return( ret );
    if( ( ret = mbedtls_gcm_update( ctx, length, input, output ) ) != 0 )
        return( ret );
    if( ( ret = mbedtls_gcm_finish( ctx, tag, tag_len ) ) != 0 )
        return( ret );
    return( 0 );
}

/**
 * \brief           This function performs a GCM authenticated decryption of a
 *                  buffer.
 *
 * \note            For decryption, the output buffer cannot be the same as
 *                  input buffer. If the buffers overlap, the output buffer
 *                  must trail at least 8 Bytes behind the input buffer.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param length    The length of the ciphertext to decrypt, which is also
 *                  the length of the decrypted plaintext.
 * \param iv        The initialization vector. This must be a readable buffer
 *                  of at least \p iv_len Bytes.
 * \param iv_len    The length of the IV.
 * \param add       The buffer holding the additional data. This must be of at
 *                  least that size in Bytes.
 * \param add_len   The length of the additional data.
 * \param tag       The buffer holding the tag to verify. This must be a
 *                  readable buffer of at least \p tag_len Bytes.
 * \param tag_len   The length of the tag to verify.
 * \param input     The buffer holding the ciphertext. If \p length is greater
 *                  than zero, this must be a readable buffer of at least that
 *                  size.
 * \param output    The buffer for holding the decrypted plaintext. If \p length
 *                  is greater than zero, this must be a writable buffer of at
 *                  least that size.
 *
 * \return          \c 0 if successful and authenticated.
 * \return          #MBEDTLS_ERR_GCM_AUTH_FAILED if the tag does not match.
 * \return          #MBEDTLS_ERR_GCM_BAD_INPUT if the lengths or pointers are
 *                  not valid or a cipher-specific error code if the decryption
 *                  failed.
 */
int mbedtls_gcm_auth_decrypt( mbedtls_gcm_context *ctx,
                              size_t length,
                              const unsigned char *iv,
                              size_t iv_len,
                              const unsigned char *add,
                              size_t add_len,
                              const unsigned char *tag,
                              size_t tag_len,
                              const unsigned char *input,
                              unsigned char *output )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char check_tag[16];
    size_t i;
    int diff;
    GCM_VALIDATE_RET( ctx != NULL );
    GCM_VALIDATE_RET( iv != NULL );
    GCM_VALIDATE_RET( add_len == 0 || add != NULL );
    GCM_VALIDATE_RET( tag != NULL );
    GCM_VALIDATE_RET( length == 0 || input != NULL );
    GCM_VALIDATE_RET( length == 0 || output != NULL );
    if( ( ret = mbedtls_gcm_crypt_and_tag( ctx, MBEDTLS_GCM_DECRYPT, length,
                                   iv, iv_len, add, add_len,
                                   input, output, tag_len, check_tag ) ) != 0 ) {
        return( ret );
    }
    /* Check tag in "constant-time" */
    for( diff = 0, i = 0; i < tag_len; i++ )
        diff |= tag[i] ^ check_tag[i];
    if( diff != 0 ) {
        mbedtls_platform_zeroize( output, length );
        return( MBEDTLS_ERR_GCM_AUTH_FAILED );
    }
    return( 0 );
}

/**
 * \brief           This function clears a GCM context and the underlying
 *                  cipher sub-context.
 *
 * \param ctx       The GCM context to clear. If this is \c NULL, the call has
 *                  no effect. Otherwise, this must be initialized.
 */
void mbedtls_gcm_free( mbedtls_gcm_context *ctx )
{
    if( ctx == NULL )
        return;
    mbedtls_cipher_free( &ctx->cipher_ctx );
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_gcm_context ) );
}

#endif /* !MBEDTLS_GCM_ALT */

#if defined(MBEDTLS_SELF_TEST) && defined(MBEDTLS_AES_C)
/*
 * AES-GCM test vectors from:
 *
 * http://csrc.nist.gov/groups/STM/cavp/documents/mac/gcmtestvectors.zip
 */
#define MAX_TESTS   6

static const int key_index_test_data[MAX_TESTS] =
    { 0, 0, 1, 1, 1, 1 };

static const unsigned char key_test_data[MAX_TESTS][32] =
{
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xfe, 0xff, 0xe9, 0x92, 0x86, 0x65, 0x73, 0x1c,
      0x6d, 0x6a, 0x8f, 0x94, 0x67, 0x30, 0x83, 0x08,
      0xfe, 0xff, 0xe9, 0x92, 0x86, 0x65, 0x73, 0x1c,
      0x6d, 0x6a, 0x8f, 0x94, 0x67, 0x30, 0x83, 0x08 },
};

static const size_t iv_len_test_data[MAX_TESTS] =
    { 12, 12, 12, 12, 8, 60 };

static const int iv_index_test_data[MAX_TESTS] =
    { 0, 0, 1, 1, 1, 2 };

static const unsigned char iv_test_data[MAX_TESTS][64] =
{
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00 },
    { 0xca, 0xfe, 0xba, 0xbe, 0xfa, 0xce, 0xdb, 0xad,
      0xde, 0xca, 0xf8, 0x88 },
    { 0x93, 0x13, 0x22, 0x5d, 0xf8, 0x84, 0x06, 0xe5,
      0x55, 0x90, 0x9c, 0x5a, 0xff, 0x52, 0x69, 0xaa,
      0x6a, 0x7a, 0x95, 0x38, 0x53, 0x4f, 0x7d, 0xa1,
      0xe4, 0xc3, 0x03, 0xd2, 0xa3, 0x18, 0xa7, 0x28,
      0xc3, 0xc0, 0xc9, 0x51, 0x56, 0x80, 0x95, 0x39,
      0xfc, 0xf0, 0xe2, 0x42, 0x9a, 0x6b, 0x52, 0x54,
      0x16, 0xae, 0xdb, 0xf5, 0xa0, 0xde, 0x6a, 0x57,
      0xa6, 0x37, 0xb3, 0x9b },
};

static const size_t add_len_test_data[MAX_TESTS] =
    { 0, 0, 0, 20, 20, 20 };

static const int add_index_test_data[MAX_TESTS] =
    { 0, 0, 0, 1, 1, 1 };

static const unsigned char additional_test_data[MAX_TESTS][64] =
{
    { 0x00 },
    { 0xfe, 0xed, 0xfa, 0xce, 0xde, 0xad, 0xbe, 0xef,
      0xfe, 0xed, 0xfa, 0xce, 0xde, 0xad, 0xbe, 0xef,
      0xab, 0xad, 0xda, 0xd2 },
};

static const size_t pt_len_test_data[MAX_TESTS] =
    { 0, 16, 64, 60, 60, 60 };

static const int pt_index_test_data[MAX_TESTS] =
    { 0, 0, 1, 1, 1, 1 };

static const unsigned char pt_test_data[MAX_TESTS][64] =
{
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xd9, 0x31, 0x32, 0x25, 0xf8, 0x84, 0x06, 0xe5,
      0xa5, 0x59, 0x09, 0xc5, 0xaf, 0xf5, 0x26, 0x9a,
      0x86, 0xa7, 0xa9, 0x53, 0x15, 0x34, 0xf7, 0xda,
      0x2e, 0x4c, 0x30, 0x3d, 0x8a, 0x31, 0x8a, 0x72,
      0x1c, 0x3c, 0x0c, 0x95, 0x95, 0x68, 0x09, 0x53,
      0x2f, 0xcf, 0x0e, 0x24, 0x49, 0xa6, 0xb5, 0x25,
      0xb1, 0x6a, 0xed, 0xf5, 0xaa, 0x0d, 0xe6, 0x57,
      0xba, 0x63, 0x7b, 0x39, 0x1a, 0xaf, 0xd2, 0x55 },
};

static const unsigned char ct_test_data[MAX_TESTS * 3][64] =
{
    { 0x00 },
    { 0x03, 0x88, 0xda, 0xce, 0x60, 0xb6, 0xa3, 0x92,
      0xf3, 0x28, 0xc2, 0xb9, 0x71, 0xb2, 0xfe, 0x78 },
    { 0x42, 0x83, 0x1e, 0xc2, 0x21, 0x77, 0x74, 0x24,
      0x4b, 0x72, 0x21, 0xb7, 0x84, 0xd0, 0xd4, 0x9c,
      0xe3, 0xaa, 0x21, 0x2f, 0x2c, 0x02, 0xa4, 0xe0,
      0x35, 0xc1, 0x7e, 0x23, 0x29, 0xac, 0xa1, 0x2e,
      0x21, 0xd5, 0x14, 0xb2, 0x54, 0x66, 0x93, 0x1c,
      0x7d, 0x8f, 0x6a, 0x5a, 0xac, 0x84, 0xaa, 0x05,
      0x1b, 0xa3, 0x0b, 0x39, 0x6a, 0x0a, 0xac, 0x97,
      0x3d, 0x58, 0xe0, 0x91, 0x47, 0x3f, 0x59, 0x85 },
    { 0x42, 0x83, 0x1e, 0xc2, 0x21, 0x77, 0x74, 0x24,
      0x4b, 0x72, 0x21, 0xb7, 0x84, 0xd0, 0xd4, 0x9c,
      0xe3, 0xaa, 0x21, 0x2f, 0x2c, 0x02, 0xa4, 0xe0,
      0x35, 0xc1, 0x7e, 0x23, 0x29, 0xac, 0xa1, 0x2e,
      0x21, 0xd5, 0x14, 0xb2, 0x54, 0x66, 0x93, 0x1c,
      0x7d, 0x8f, 0x6a, 0x5a, 0xac, 0x84, 0xaa, 0x05,
      0x1b, 0xa3, 0x0b, 0x39, 0x6a, 0x0a, 0xac, 0x97,
      0x3d, 0x58, 0xe0, 0x91 },
    { 0x61, 0x35, 0x3b, 0x4c, 0x28, 0x06, 0x93, 0x4a,
      0x77, 0x7f, 0xf5, 0x1f, 0xa2, 0x2a, 0x47, 0x55,
      0x69, 0x9b, 0x2a, 0x71, 0x4f, 0xcd, 0xc6, 0xf8,
      0x37, 0x66, 0xe5, 0xf9, 0x7b, 0x6c, 0x74, 0x23,
      0x73, 0x80, 0x69, 0x00, 0xe4, 0x9f, 0x24, 0xb2,
      0x2b, 0x09, 0x75, 0x44, 0xd4, 0x89, 0x6b, 0x42,
      0x49, 0x89, 0xb5, 0xe1, 0xeb, 0xac, 0x0f, 0x07,
      0xc2, 0x3f, 0x45, 0x98 },
    { 0x8c, 0xe2, 0x49, 0x98, 0x62, 0x56, 0x15, 0xb6,
      0x03, 0xa0, 0x33, 0xac, 0xa1, 0x3f, 0xb8, 0x94,
      0xbe, 0x91, 0x12, 0xa5, 0xc3, 0xa2, 0x11, 0xa8,
      0xba, 0x26, 0x2a, 0x3c, 0xca, 0x7e, 0x2c, 0xa7,
      0x01, 0xe4, 0xa9, 0xa4, 0xfb, 0xa4, 0x3c, 0x90,
      0xcc, 0xdc, 0xb2, 0x81, 0xd4, 0x8c, 0x7c, 0x6f,
      0xd6, 0x28, 0x75, 0xd2, 0xac, 0xa4, 0x17, 0x03,
      0x4c, 0x34, 0xae, 0xe5 },
    { 0x00 },
    { 0x98, 0xe7, 0x24, 0x7c, 0x07, 0xf0, 0xfe, 0x41,
      0x1c, 0x26, 0x7e, 0x43, 0x84, 0xb0, 0xf6, 0x00 },
    { 0x39, 0x80, 0xca, 0x0b, 0x3c, 0x00, 0xe8, 0x41,
      0xeb, 0x06, 0xfa, 0xc4, 0x87, 0x2a, 0x27, 0x57,
      0x85, 0x9e, 0x1c, 0xea, 0xa6, 0xef, 0xd9, 0x84,
      0x62, 0x85, 0x93, 0xb4, 0x0c, 0xa1, 0xe1, 0x9c,
      0x7d, 0x77, 0x3d, 0x00, 0xc1, 0x44, 0xc5, 0x25,
      0xac, 0x61, 0x9d, 0x18, 0xc8, 0x4a, 0x3f, 0x47,
      0x18, 0xe2, 0x44, 0x8b, 0x2f, 0xe3, 0x24, 0xd9,
      0xcc, 0xda, 0x27, 0x10, 0xac, 0xad, 0xe2, 0x56 },
    { 0x39, 0x80, 0xca, 0x0b, 0x3c, 0x00, 0xe8, 0x41,
      0xeb, 0x06, 0xfa, 0xc4, 0x87, 0x2a, 0x27, 0x57,
      0x85, 0x9e, 0x1c, 0xea, 0xa6, 0xef, 0xd9, 0x84,
      0x62, 0x85, 0x93, 0xb4, 0x0c, 0xa1, 0xe1, 0x9c,
      0x7d, 0x77, 0x3d, 0x00, 0xc1, 0x44, 0xc5, 0x25,
      0xac, 0x61, 0x9d, 0x18, 0xc8, 0x4a, 0x3f, 0x47,
      0x18, 0xe2, 0x44, 0x8b, 0x2f, 0xe3, 0x24, 0xd9,
      0xcc, 0xda, 0x27, 0x10 },
    { 0x0f, 0x10, 0xf5, 0x99, 0xae, 0x14, 0xa1, 0x54,
      0xed, 0x24, 0xb3, 0x6e, 0x25, 0x32, 0x4d, 0xb8,
      0xc5, 0x66, 0x63, 0x2e, 0xf2, 0xbb, 0xb3, 0x4f,
      0x83, 0x47, 0x28, 0x0f, 0xc4, 0x50, 0x70, 0x57,
      0xfd, 0xdc, 0x29, 0xdf, 0x9a, 0x47, 0x1f, 0x75,
      0xc6, 0x65, 0x41, 0xd4, 0xd4, 0xda, 0xd1, 0xc9,
      0xe9, 0x3a, 0x19, 0xa5, 0x8e, 0x8b, 0x47, 0x3f,
      0xa0, 0xf0, 0x62, 0xf7 },
    { 0xd2, 0x7e, 0x88, 0x68, 0x1c, 0xe3, 0x24, 0x3c,
      0x48, 0x30, 0x16, 0x5a, 0x8f, 0xdc, 0xf9, 0xff,
      0x1d, 0xe9, 0xa1, 0xd8, 0xe6, 0xb4, 0x47, 0xef,
      0x6e, 0xf7, 0xb7, 0x98, 0x28, 0x66, 0x6e, 0x45,
      0x81, 0xe7, 0x90, 0x12, 0xaf, 0x34, 0xdd, 0xd9,
      0xe2, 0xf0, 0x37, 0x58, 0x9b, 0x29, 0x2d, 0xb3,
      0xe6, 0x7c, 0x03, 0x67, 0x45, 0xfa, 0x22, 0xe7,
      0xe9, 0xb7, 0x37, 0x3b },
    { 0x00 },
    { 0xce, 0xa7, 0x40, 0x3d, 0x4d, 0x60, 0x6b, 0x6e,
      0x07, 0x4e, 0xc5, 0xd3, 0xba, 0xf3, 0x9d, 0x18 },
    { 0x52, 0x2d, 0xc1, 0xf0, 0x99, 0x56, 0x7d, 0x07,
      0xf4, 0x7f, 0x37, 0xa3, 0x2a, 0x84, 0x42, 0x7d,
      0x64, 0x3a, 0x8c, 0xdc, 0xbf, 0xe5, 0xc0, 0xc9,
      0x75, 0x98, 0xa2, 0xbd, 0x25, 0x55, 0xd1, 0xaa,
      0x8c, 0xb0, 0x8e, 0x48, 0x59, 0x0d, 0xbb, 0x3d,
      0xa7, 0xb0, 0x8b, 0x10, 0x56, 0x82, 0x88, 0x38,
      0xc5, 0xf6, 0x1e, 0x63, 0x93, 0xba, 0x7a, 0x0a,
      0xbc, 0xc9, 0xf6, 0x62, 0x89, 0x80, 0x15, 0xad },
    { 0x52, 0x2d, 0xc1, 0xf0, 0x99, 0x56, 0x7d, 0x07,
      0xf4, 0x7f, 0x37, 0xa3, 0x2a, 0x84, 0x42, 0x7d,
      0x64, 0x3a, 0x8c, 0xdc, 0xbf, 0xe5, 0xc0, 0xc9,
      0x75, 0x98, 0xa2, 0xbd, 0x25, 0x55, 0xd1, 0xaa,
      0x8c, 0xb0, 0x8e, 0x48, 0x59, 0x0d, 0xbb, 0x3d,
      0xa7, 0xb0, 0x8b, 0x10, 0x56, 0x82, 0x88, 0x38,
      0xc5, 0xf6, 0x1e, 0x63, 0x93, 0xba, 0x7a, 0x0a,
      0xbc, 0xc9, 0xf6, 0x62 },
    { 0xc3, 0x76, 0x2d, 0xf1, 0xca, 0x78, 0x7d, 0x32,
      0xae, 0x47, 0xc1, 0x3b, 0xf1, 0x98, 0x44, 0xcb,
      0xaf, 0x1a, 0xe1, 0x4d, 0x0b, 0x97, 0x6a, 0xfa,
      0xc5, 0x2f, 0xf7, 0xd7, 0x9b, 0xba, 0x9d, 0xe0,
      0xfe, 0xb5, 0x82, 0xd3, 0x39, 0x34, 0xa4, 0xf0,
      0x95, 0x4c, 0xc2, 0x36, 0x3b, 0xc7, 0x3f, 0x78,
      0x62, 0xac, 0x43, 0x0e, 0x64, 0xab, 0xe4, 0x99,
      0xf4, 0x7c, 0x9b, 0x1f },
    { 0x5a, 0x8d, 0xef, 0x2f, 0x0c, 0x9e, 0x53, 0xf1,
      0xf7, 0x5d, 0x78, 0x53, 0x65, 0x9e, 0x2a, 0x20,
      0xee, 0xb2, 0xb2, 0x2a, 0xaf, 0xde, 0x64, 0x19,
      0xa0, 0x58, 0xab, 0x4f, 0x6f, 0x74, 0x6b, 0xf4,
      0x0f, 0xc0, 0xc3, 0xb7, 0x80, 0xf2, 0x44, 0x45,
      0x2d, 0xa3, 0xeb, 0xf1, 0xc5, 0xd8, 0x2c, 0xde,
      0xa2, 0x41, 0x89, 0x97, 0x20, 0x0e, 0xf8, 0x2e,
      0x44, 0xae, 0x7e, 0x3f },
};

static const unsigned char tag_test_data[MAX_TESTS * 3][16] =
{
    { 0x58, 0xe2, 0xfc, 0xce, 0xfa, 0x7e, 0x30, 0x61,
      0x36, 0x7f, 0x1d, 0x57, 0xa4, 0xe7, 0x45, 0x5a },
    { 0xab, 0x6e, 0x47, 0xd4, 0x2c, 0xec, 0x13, 0xbd,
      0xf5, 0x3a, 0x67, 0xb2, 0x12, 0x57, 0xbd, 0xdf },
    { 0x4d, 0x5c, 0x2a, 0xf3, 0x27, 0xcd, 0x64, 0xa6,
      0x2c, 0xf3, 0x5a, 0xbd, 0x2b, 0xa6, 0xfa, 0xb4 },
    { 0x5b, 0xc9, 0x4f, 0xbc, 0x32, 0x21, 0xa5, 0xdb,
      0x94, 0xfa, 0xe9, 0x5a, 0xe7, 0x12, 0x1a, 0x47 },
    { 0x36, 0x12, 0xd2, 0xe7, 0x9e, 0x3b, 0x07, 0x85,
      0x56, 0x1b, 0xe1, 0x4a, 0xac, 0xa2, 0xfc, 0xcb },
    { 0x61, 0x9c, 0xc5, 0xae, 0xff, 0xfe, 0x0b, 0xfa,
      0x46, 0x2a, 0xf4, 0x3c, 0x16, 0x99, 0xd0, 0x50 },
    { 0xcd, 0x33, 0xb2, 0x8a, 0xc7, 0x73, 0xf7, 0x4b,
      0xa0, 0x0e, 0xd1, 0xf3, 0x12, 0x57, 0x24, 0x35 },
    { 0x2f, 0xf5, 0x8d, 0x80, 0x03, 0x39, 0x27, 0xab,
      0x8e, 0xf4, 0xd4, 0x58, 0x75, 0x14, 0xf0, 0xfb },
    { 0x99, 0x24, 0xa7, 0xc8, 0x58, 0x73, 0x36, 0xbf,
      0xb1, 0x18, 0x02, 0x4d, 0xb8, 0x67, 0x4a, 0x14 },
    { 0x25, 0x19, 0x49, 0x8e, 0x80, 0xf1, 0x47, 0x8f,
      0x37, 0xba, 0x55, 0xbd, 0x6d, 0x27, 0x61, 0x8c },
    { 0x65, 0xdc, 0xc5, 0x7f, 0xcf, 0x62, 0x3a, 0x24,
      0x09, 0x4f, 0xcc, 0xa4, 0x0d, 0x35, 0x33, 0xf8 },
    { 0xdc, 0xf5, 0x66, 0xff, 0x29, 0x1c, 0x25, 0xbb,
      0xb8, 0x56, 0x8f, 0xc3, 0xd3, 0x76, 0xa6, 0xd9 },
    { 0x53, 0x0f, 0x8a, 0xfb, 0xc7, 0x45, 0x36, 0xb9,
      0xa9, 0x63, 0xb4, 0xf1, 0xc4, 0xcb, 0x73, 0x8b },
    { 0xd0, 0xd1, 0xc8, 0xa7, 0x99, 0x99, 0x6b, 0xf0,
      0x26, 0x5b, 0x98, 0xb5, 0xd4, 0x8a, 0xb9, 0x19 },
    { 0xb0, 0x94, 0xda, 0xc5, 0xd9, 0x34, 0x71, 0xbd,
      0xec, 0x1a, 0x50, 0x22, 0x70, 0xe3, 0xcc, 0x6c },
    { 0x76, 0xfc, 0x6e, 0xce, 0x0f, 0x4e, 0x17, 0x68,
      0xcd, 0xdf, 0x88, 0x53, 0xbb, 0x2d, 0x55, 0x1b },
    { 0x3a, 0x33, 0x7d, 0xbf, 0x46, 0xa7, 0x92, 0xc4,
      0x5e, 0x45, 0x49, 0x13, 0xfe, 0x2e, 0xa8, 0xf2 },
    { 0xa4, 0x4a, 0x82, 0x66, 0xee, 0x1c, 0x8e, 0xb0,
      0xc8, 0xb5, 0xd4, 0xcf, 0x5a, 0xe9, 0xf1, 0x9a },
};

/**
 * \brief          The GCM checkup routine.
 *
 * \return         \c 0 on success.
 * \return         \c 1 on failure.
 */
int mbedtls_gcm_self_test( int verbose )
{
    mbedtls_gcm_context ctx;
    unsigned char buf[64];
    unsigned char tag_buf[16];
    int i, j, ret;
    mbedtls_cipher_id_t cipher = MBEDTLS_CIPHER_ID_AES;

    for( j = 0; j < 3; j++ )
    {
        int key_len = 128 + 64 * j;

        for( i = 0; i < MAX_TESTS; i++ )
        {
            mbedtls_gcm_init( &ctx );

            if( verbose != 0 )
                mbedtls_printf( "  AES-GCM-%3d #%d (%s): ",
                                key_len, i, "enc" );

            ret = mbedtls_gcm_setkey( &ctx, cipher,
                                      key_test_data[key_index_test_data[i]],
                                      key_len );
            if( ret != 0 )
            {
                goto exit;
            }

            ret = mbedtls_gcm_crypt_and_tag( &ctx, MBEDTLS_GCM_ENCRYPT,
                                pt_len_test_data[i],
                                iv_test_data[iv_index_test_data[i]],
                                iv_len_test_data[i],
                                additional_test_data[add_index_test_data[i]],
                                add_len_test_data[i],
                                pt_test_data[pt_index_test_data[i]],
                                buf, 16, tag_buf );
            if( ret != 0 )
                goto exit;

            if ( timingsafe_bcmp( buf, ct_test_data[j * 6 + i],
                                  pt_len_test_data[i] ) != 0 ||
                 timingsafe_bcmp( tag_buf, tag_test_data[j * 6 + i], 16 ) != 0 )
            {
                ret = 1;
                goto exit;
            }

            mbedtls_gcm_free( &ctx );

            if( verbose != 0 )
                mbedtls_printf( "passed\n" );

            mbedtls_gcm_init( &ctx );

            if( verbose != 0 )
                mbedtls_printf( "  AES-GCM-%3d #%d (%s): ",
                                key_len, i, "dec" );

            ret = mbedtls_gcm_setkey( &ctx, cipher,
                                      key_test_data[key_index_test_data[i]],
                                      key_len );
            if( ret != 0 )
                goto exit;

            ret = mbedtls_gcm_crypt_and_tag( &ctx, MBEDTLS_GCM_DECRYPT,
                                pt_len_test_data[i],
                                iv_test_data[iv_index_test_data[i]],
                                iv_len_test_data[i],
                                additional_test_data[add_index_test_data[i]],
                                add_len_test_data[i],
                                ct_test_data[j * 6 + i], buf, 16, tag_buf );

            if( ret != 0 )
                goto exit;

            if( timingsafe_bcmp( buf, pt_test_data[pt_index_test_data[i]],
                                 pt_len_test_data[i] ) != 0 ||
                timingsafe_bcmp( tag_buf, tag_test_data[j * 6 + i], 16 ) != 0 )
            {
                ret = 1;
                goto exit;
            }

            mbedtls_gcm_free( &ctx );

            if( verbose != 0 )
                mbedtls_printf( "passed\n" );

            mbedtls_gcm_init( &ctx );

            if( verbose != 0 )
                mbedtls_printf( "  AES-GCM-%3d #%d split (%s): ",
                                key_len, i, "enc" );

            ret = mbedtls_gcm_setkey( &ctx, cipher,
                                      key_test_data[key_index_test_data[i]],
                                      key_len );
            if( ret != 0 )
                goto exit;

            ret = mbedtls_gcm_starts( &ctx, MBEDTLS_GCM_ENCRYPT,
                                  iv_test_data[iv_index_test_data[i]],
                                  iv_len_test_data[i],
                                  additional_test_data[add_index_test_data[i]],
                                  add_len_test_data[i] );
            if( ret != 0 )
                goto exit;

            if( pt_len_test_data[i] > 32 )
            {
                size_t rest_len = pt_len_test_data[i] - 32;
                ret = mbedtls_gcm_update( &ctx, 32,
                                          pt_test_data[pt_index_test_data[i]],
                                          buf );
                if( ret != 0 )
                    goto exit;

                ret = mbedtls_gcm_update( &ctx, rest_len,
                                      pt_test_data[pt_index_test_data[i]] + 32,
                                      buf + 32 );
                if( ret != 0 )
                    goto exit;
            }
            else
            {
                ret = mbedtls_gcm_update( &ctx, pt_len_test_data[i],
                                          pt_test_data[pt_index_test_data[i]],
                                          buf );
                if( ret != 0 )
                    goto exit;
            }

            ret = mbedtls_gcm_finish( &ctx, tag_buf, 16 );
            if( ret != 0 )
                goto exit;

            if( timingsafe_bcmp( buf, ct_test_data[j * 6 + i],
                                 pt_len_test_data[i] ) != 0 ||
                timingsafe_bcmp( tag_buf, tag_test_data[j * 6 + i], 16 ) != 0 )
            {
                ret = 1;
                goto exit;
            }

            mbedtls_gcm_free( &ctx );

            if( verbose != 0 )
                mbedtls_printf( "passed\n" );

            mbedtls_gcm_init( &ctx );

            if( verbose != 0 )
                mbedtls_printf( "  AES-GCM-%3d #%d split (%s): ",
                                key_len, i, "dec" );

            ret = mbedtls_gcm_setkey( &ctx, cipher,
                                      key_test_data[key_index_test_data[i]],
                                      key_len );
            if( ret != 0 )
                goto exit;

            ret = mbedtls_gcm_starts( &ctx, MBEDTLS_GCM_DECRYPT,
                              iv_test_data[iv_index_test_data[i]],
                              iv_len_test_data[i],
                              additional_test_data[add_index_test_data[i]],
                              add_len_test_data[i] );
            if( ret != 0 )
                goto exit;

            if( pt_len_test_data[i] > 32 )
            {
                size_t rest_len = pt_len_test_data[i] - 32;
                ret = mbedtls_gcm_update( &ctx, 32, ct_test_data[j * 6 + i],
                                          buf );
                if( ret != 0 )
                    goto exit;

                ret = mbedtls_gcm_update( &ctx, rest_len,
                                          ct_test_data[j * 6 + i] + 32,
                                          buf + 32 );
                if( ret != 0 )
                    goto exit;
            }
            else
            {
                ret = mbedtls_gcm_update( &ctx, pt_len_test_data[i],
                                          ct_test_data[j * 6 + i],
                                          buf );
                if( ret != 0 )
                    goto exit;
            }

            ret = mbedtls_gcm_finish( &ctx, tag_buf, 16 );
            if( ret != 0 )
                goto exit;

            if( timingsafe_bcmp( buf, pt_test_data[pt_index_test_data[i]],
                                 pt_len_test_data[i] ) != 0 ||
                timingsafe_bcmp( tag_buf, tag_test_data[j * 6 + i], 16 ) != 0 )
            {
                ret = 1;
                goto exit;
            }

            mbedtls_gcm_free( &ctx );

            if( verbose != 0 )
                mbedtls_printf( "passed\n" );
        }
    }

    if( verbose != 0 )
        mbedtls_printf( "\n" );

    ret = 0;

exit:
    if( ret != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "failed\n" );
        mbedtls_gcm_free( &ctx );
    }

    return( ret );
}

#endif /* MBEDTLS_SELF_TEST && MBEDTLS_AES_C */
