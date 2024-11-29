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
#include "third_party/mbedtls/ccm.h"
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/platform.h"
__static_yoink("mbedtls_notice");

/**
 * @fileoverview NIST SP800-38C compliant CCM implementation
 *
 * This file provides an API for the CCM authenticated encryption mode
 * for block ciphers.
 *
 * CCM combines Counter mode encryption with CBC-MAC authentication
 * for 128-bit block ciphers.
 *
 * Input to CCM includes the following elements:
 * <ul><li>Payload - data that is both authenticated and encrypted.</li>
 * <li>Associated data (Adata) - data that is authenticated but not
 * encrypted, For example, a header.</li>
 * <li>Nonce - A unique value that is assigned to the payload and the
 * associated data.</li></ul>
 *
 * Definition of CCM:
 * http://csrc.nist.gov/publications/nistpubs/800-38C/SP800-38C_updated-July20_2007.pdf
 * RFC 3610 "Counter with CBC-MAC (CCM)"
 *
 * Related:
 * RFC 5116 "An Interface and Algorithms for Authenticated Encryption"
 *
 * Definition of CCM*:
 * IEEE 802.15.4 - IEEE Standard for Local and metropolitan area networks
 * Integer representation is fixed most-significant-octet-first order and
 * the representation of octets is most-significant-bit-first order. This is
 * consistent with RFC 3610.
 */

#define CCM_VALIDATE_RET( cond ) \
    MBEDTLS_INTERNAL_VALIDATE_RET( cond, MBEDTLS_ERR_CCM_BAD_INPUT )
#define CCM_VALIDATE( cond ) \
    MBEDTLS_INTERNAL_VALIDATE( cond )

#define CCM_ENCRYPT 0
#define CCM_DECRYPT 1

/**
 * \brief           This function initializes the specified CCM context,
 *                  to make references valid, and prepare the context
 *                  for mbedtls_ccm_setkey() or mbedtls_ccm_free().
 *
 * \param ctx       The CCM context to initialize. This must not be \c NULL.
 */
void mbedtls_ccm_init( mbedtls_ccm_context *ctx )
{
    CCM_VALIDATE( ctx );
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_ccm_context ) );
}

/**
 * \brief           This function initializes the CCM context set in the
 *                  \p ctx parameter and sets the encryption key.
 *
 * \param ctx       The CCM context to initialize. This must be an initialized
 *                  context.
 * \param cipher    The 128-bit block cipher to use.
 * \param key       The encryption key. This must not be \c NULL.
 * \param keybits   The key size in bits. This must be acceptable by the cipher.
 *
 * \return          \c 0 on success.
 * \return          A CCM or cipher-specific error code on failure.
 */
int mbedtls_ccm_setkey( mbedtls_ccm_context *ctx,
                        mbedtls_cipher_id_t cipher,
                        const unsigned char *key,
                        unsigned int keybits )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    const mbedtls_cipher_info_t *cipher_info;
    CCM_VALIDATE_RET( ctx );
    CCM_VALIDATE_RET( key );
    cipher_info = mbedtls_cipher_info_from_values( cipher, keybits,
                                                   MBEDTLS_MODE_ECB );
    if( cipher_info == NULL )
        return( MBEDTLS_ERR_CCM_BAD_INPUT );
    if( cipher_info->block_size != 16 )
        return( MBEDTLS_ERR_CCM_BAD_INPUT );
    mbedtls_cipher_free( &ctx->cipher_ctx );
    if( ( ret = mbedtls_cipher_setup( &ctx->cipher_ctx, cipher_info ) ) != 0 )
        return( ret );
    if( ( ret = mbedtls_cipher_setkey( &ctx->cipher_ctx, key, keybits,
                               MBEDTLS_ENCRYPT ) ) != 0 )
    {
        return( ret );
    }
    return( 0 );
}

/**
 * \brief   This function releases and clears the specified CCM context
 *          and underlying cipher sub-context.
 *
 * \param ctx       The CCM context to clear. If this is \c NULL, the function
 *                  has no effect. Otherwise, this must be initialized.
 */
void mbedtls_ccm_free( mbedtls_ccm_context *ctx )
{
    if( ctx == NULL )
        return;
    mbedtls_cipher_free( &ctx->cipher_ctx );
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_ccm_context ) );
}

/*
 * Macros for common operations.
 * Results in smaller compiled code than static inline functions.
 */

/*
 * Update the CBC-MAC state in y using a block in b
 * (Always using b as the source helps the compiler optimise a bit better.)
 */
#define UPDATE_CBC_MAC                                                      \
    for( i = 0; i < 16; i++ )                                               \
        y[i] ^= b[i];                                                       \
                                                                            \
    if( ( ret = mbedtls_cipher_update( &ctx->cipher_ctx, y, 16, y, &olen ) ) != 0 ) \
        return( ret );

/*
 * Encrypt or decrypt a partial block with CTR
 * Warning: using b for temporary storage! src and dst must not be b!
 * This avoids allocating one more 16 bytes buffer while allowing src == dst.
 */
#define CTR_CRYPT( dst, src, len  )                                            \
    do                                                                  \
    {                                                                   \
        if( ( ret = mbedtls_cipher_update( &ctx->cipher_ctx, ctr,       \
                                           16, b, &olen ) ) != 0 )      \
        {                                                               \
            return( ret );                                              \
        }                                                               \
                                                                        \
        for( i = 0; i < (len); i++ )                                    \
            (dst)[i] = (src)[i] ^ b[i];                                 \
    } while( 0 )

/*
 * Authenticated encryption or decryption
 */
static int ccm_auth_crypt( mbedtls_ccm_context *ctx, int mode, size_t length,
                           const unsigned char *iv, size_t iv_len,
                           const unsigned char *add, size_t add_len,
                           const unsigned char *input, unsigned char *output,
                           unsigned char *tag, size_t tag_len )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char i;
    unsigned char q;
    size_t len_left, olen;
    unsigned char b[16];
    unsigned char y[16];
    unsigned char ctr[16];
    const unsigned char *src;
    unsigned char *dst;

    /*
     * Check length requirements: SP800-38C A.1
     * Additional requirement: a < 2^16 - 2^8 to simplify the code.
     * 'length' checked later (when writing it to the first block)
     *
     * Also, loosen the requirements to enable support for CCM* (IEEE 802.15.4).
     */
    if( tag_len == 2 || tag_len > 16 || tag_len % 2 != 0 )
        return( MBEDTLS_ERR_CCM_BAD_INPUT );

    /* Also implies q is within bounds */
    if( iv_len < 7 || iv_len > 13 )
        return( MBEDTLS_ERR_CCM_BAD_INPUT );

    if( add_len >= 0xFF00 )
        return( MBEDTLS_ERR_CCM_BAD_INPUT );

    q = 16 - 1 - (unsigned char) iv_len;

    /*
     * First block B_0:
     * 0        .. 0        flags
     * 1        .. iv_len   nonce (aka iv)
     * iv_len+1 .. 15       length
     *
     * With flags as (bits):
     * 7        0
     * 6        add present?
     * 5 .. 3   (t - 2) / 2
     * 2 .. 0   q - 1
     */
    b[0] = 0;
    b[0] |= ( add_len > 0 ) << 6;
    b[0] |= ( ( tag_len - 2 ) / 2 ) << 3;
    b[0] |= q - 1;

    memcpy( b + 1, iv, iv_len );

    for( i = 0, len_left = length; i < q; i++, len_left >>= 8 )
        b[15-i] = (unsigned char)( len_left & 0xFF );

    if( len_left > 0 )
        return( MBEDTLS_ERR_CCM_BAD_INPUT );


    /* Start CBC-MAC with first block */
    memset( y, 0, 16 );
    UPDATE_CBC_MAC;

    /*
     * If there is additional data, update CBC-MAC with
     * add_len, add, 0 (padding to a block boundary)
     */
    if( add_len > 0 )
    {
        size_t use_len;
        len_left = add_len;
        src = add;

        memset( b, 0, 16 );
        b[0] = (unsigned char)( ( add_len >> 8 ) & 0xFF );
        b[1] = (unsigned char)( ( add_len      ) & 0xFF );

        use_len = len_left < 16 - 2 ? len_left : 16 - 2;
        memcpy( b + 2, src, use_len );
        len_left -= use_len;
        src += use_len;

        UPDATE_CBC_MAC;

        while( len_left > 0 )
        {
            use_len = len_left > 16 ? 16 : len_left;

            memset( b, 0, 16 );
            memcpy( b, src, use_len );
            UPDATE_CBC_MAC;

            len_left -= use_len;
            src += use_len;
        }
    }

    /*
     * Prepare counter block for encryption:
     * 0        .. 0        flags
     * 1        .. iv_len   nonce (aka iv)
     * iv_len+1 .. 15       counter (initially 1)
     *
     * With flags as (bits):
     * 7 .. 3   0
     * 2 .. 0   q - 1
     */
    ctr[0] = q - 1;
    memcpy( ctr + 1, iv, iv_len );
    memset( ctr + 1 + iv_len, 0, q );
    ctr[15] = 1;

    /*
     * Authenticate and {en,de}crypt the message.
     *
     * The only difference between encryption and decryption is
     * the respective order of authentication and {en,de}cryption.
     */
    len_left = length;
    src = input;
    dst = output;

    while( len_left > 0 )
    {
        size_t use_len = len_left > 16 ? 16 : len_left;

        if( mode == CCM_ENCRYPT )
        {
            memset( b, 0, 16 );
            memcpy( b, src, use_len );
            UPDATE_CBC_MAC;
        }

        CTR_CRYPT( dst, src, use_len );

        if( mode == CCM_DECRYPT )
        {
            memset( b, 0, 16 );
            memcpy( b, dst, use_len );
            UPDATE_CBC_MAC;
        }

        dst += use_len;
        src += use_len;
        len_left -= use_len;

        /*
         * Increment counter.
         * No need to check for overflow thanks to the length check above.
         */
        for( i = 0; i < q; i++ )
            if( ++ctr[15-i] != 0 )
                break;
    }

    /*
     * Authentication: reset counter and crypt/mask internal tag
     */
    for( i = 0; i < q; i++ )
        ctr[15-i] = 0;

    CTR_CRYPT( y, y, 16 );
    memcpy( tag, y, tag_len );

    return( 0 );
}

/**
 * \brief           This function encrypts a buffer using CCM*.
 *
 * \note            The tag is written to a separate buffer. To concatenate
 *                  the \p tag with the \p output, as done in <em>RFC-3610:
 *                  Counter with CBC-MAC (CCM)</em>, use
 *                  \p tag = \p output + \p length, and make sure that the
 *                  output buffer is at least \p length + \p tag_len wide.
 *
 * \note            When using this function in a variable tag length context,
 *                  the tag length has to be encoded into the \p iv passed to
 *                  this function.
 *
 * \param ctx       The CCM context to use for encryption. This must be
 *                  initialized and bound to a key.
 * \param length    The length of the input data in Bytes.
 * \param iv        The initialization vector (nonce). This must be a readable
 *                  buffer of at least \p iv_len Bytes.
 * \param iv_len    The length of the nonce in Bytes: 7, 8, 9, 10, 11, 12,
 *                  or 13. The length L of the message length field is
 *                  15 - \p iv_len.
 * \param add       The additional data field. This must be a readable buffer of
 *                  at least \p add_len Bytes.
 * \param add_len   The length of additional data in Bytes.
 *                  This must be less than 2^16 - 2^8.
 * \param input     The buffer holding the input data. If \p length is greater
 *                  than zero, \p input must be a readable buffer of at least
 *                  that length.
 * \param output    The buffer holding the output data. If \p length is greater
 *                  than zero, \p output must be a writable buffer of at least
 *                  that length.
 * \param tag       The buffer holding the authentication field. This must be a
 *                  writable buffer of at least \p tag_len Bytes.
 * \param tag_len   The length of the authentication field to generate in Bytes:
 *                  0, 4, 6, 8, 10, 12, 14 or 16.
 *
 * \warning         Passing \c 0 as \p tag_len means that the message is no
 *                  longer authenticated.
 *
 * \return          \c 0 on success.
 * \return          A CCM or cipher-specific error code on failure.
 */
int mbedtls_ccm_star_encrypt_and_tag( mbedtls_ccm_context *ctx, size_t length,
                                      const unsigned char *iv, size_t iv_len,
                                      const unsigned char *add, size_t add_len,
                                      const unsigned char *input, unsigned char *output,
                                      unsigned char *tag, size_t tag_len )
{
    CCM_VALIDATE_RET( ctx );
    CCM_VALIDATE_RET( iv );
    CCM_VALIDATE_RET( add_len == 0 || add );
    CCM_VALIDATE_RET( length == 0 || input );
    CCM_VALIDATE_RET( length == 0 || output );
    CCM_VALIDATE_RET( tag_len == 0 || tag );
    return( ccm_auth_crypt( ctx, CCM_ENCRYPT, length, iv, iv_len,
                            add, add_len, input, output, tag, tag_len ) );
}

/**
 * \brief           This function encrypts a buffer using CCM.
 *
 * \note            The tag is written to a separate buffer. To concatenate
 *                  the \p tag with the \p output, as done in <em>RFC-3610:
 *                  Counter with CBC-MAC (CCM)</em>, use
 *                  \p tag = \p output + \p length, and make sure that the
 *                  output buffer is at least \p length + \p tag_len wide.
 *
 * \param ctx       The CCM context to use for encryption. This must be
 *                  initialized and bound to a key.
 * \param length    The length of the input data in Bytes.
 * \param iv        The initialization vector (nonce). This must be a readable
 *                  buffer of at least \p iv_len Bytes.
 * \param iv_len    The length of the nonce in Bytes: 7, 8, 9, 10, 11, 12,
 *                  or 13. The length L of the message length field is
 *                  15 - \p iv_len.
 * \param add       The additional data field. If \p add_len is greater than
 *                  zero, \p add must be a readable buffer of at least that
 *                  length.
 * \param add_len   The length of additional data in Bytes.
 *                  This must be less than `2^16 - 2^8`.
 * \param input     The buffer holding the input data. If \p length is greater
 *                  than zero, \p input must be a readable buffer of at least
 *                  that length.
 * \param output    The buffer holding the output data. If \p length is greater
 *                  than zero, \p output must be a writable buffer of at least
 *                  that length.
 * \param tag       The buffer holding the authentication field. This must be a
 *                  writable buffer of at least \p tag_len Bytes.
 * \param tag_len   The length of the authentication field to generate in Bytes:
 *                  4, 6, 8, 10, 12, 14 or 16.
 *
 * \return          \c 0 on success.
 * \return          A CCM or cipher-specific error code on failure.
 */
int mbedtls_ccm_encrypt_and_tag( mbedtls_ccm_context *ctx, size_t length,
                                 const unsigned char *iv, size_t iv_len,
                                 const unsigned char *add, size_t add_len,
                                 const unsigned char *input, unsigned char *output,
                                 unsigned char *tag, size_t tag_len )
{
    CCM_VALIDATE_RET( ctx );
    CCM_VALIDATE_RET( iv );
    CCM_VALIDATE_RET( add_len == 0 || add );
    CCM_VALIDATE_RET( length == 0 || input );
    CCM_VALIDATE_RET( length == 0 || output );
    CCM_VALIDATE_RET( tag_len == 0 || tag );
    if( tag_len == 0 )
        return( MBEDTLS_ERR_CCM_BAD_INPUT );

    return( mbedtls_ccm_star_encrypt_and_tag( ctx, length, iv, iv_len, add,
                                              add_len, input, output, tag, tag_len ) );
}

/**
 * \brief           This function performs a CCM* authenticated decryption of a
 *                  buffer.
 *
 * \note            When using this function in a variable tag length context,
 *                  the tag length has to be decoded from \p iv and passed to
 *                  this function as \p tag_len. (\p tag needs to be adjusted
 *                  accordingly.)
 *
 * \param ctx       The CCM context to use for decryption. This must be
 *                  initialized and bound to a key.
 * \param length    The length of the input data in Bytes.
 * \param iv        The initialization vector (nonce). This must be a readable
 *                  buffer of at least \p iv_len Bytes.
 * \param iv_len    The length of the nonce in Bytes: 7, 8, 9, 10, 11, 12,
 *                  or 13. The length L of the message length field is
 *                  15 - \p iv_len.
 * \param add       The additional data field. This must be a readable buffer of
 *                  at least that \p add_len Bytes.
 * \param add_len   The length of additional data in Bytes.
 *                  This must be less than 2^16 - 2^8.
 * \param input     The buffer holding the input data. If \p length is greater
 *                  than zero, \p input must be a readable buffer of at least
 *                  that length.
 * \param output    The buffer holding the output data. If \p length is greater
 *                  than zero, \p output must be a writable buffer of at least
 *                  that length.
 * \param tag       The buffer holding the authentication field. This must be a
 *                  readable buffer of at least \p tag_len Bytes.
 * \param tag_len   The length of the authentication field in Bytes.
 *                  0, 4, 6, 8, 10, 12, 14 or 16.
 *
 * \warning         Passing \c 0 as \p tag_len means that the message is nos
 *                  longer authenticated.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_CCM_AUTH_FAILED if the tag does not match.
 * \return          A cipher-specific error code on calculation failure.
 */
int mbedtls_ccm_star_auth_decrypt( mbedtls_ccm_context *ctx, size_t length,
                                   const unsigned char *iv, size_t iv_len,
                                   const unsigned char *add, size_t add_len,
                                   const unsigned char *input, unsigned char *output,
                                   const unsigned char *tag, size_t tag_len )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char check_tag[16];
    unsigned char i;
    int diff;

    CCM_VALIDATE_RET( ctx );
    CCM_VALIDATE_RET( iv );
    CCM_VALIDATE_RET( add_len == 0 || add );
    CCM_VALIDATE_RET( length == 0 || input );
    CCM_VALIDATE_RET( length == 0 || output );
    CCM_VALIDATE_RET( tag_len == 0 || tag );

    if( ( ret = ccm_auth_crypt( ctx, CCM_DECRYPT, length,
                                iv, iv_len, add, add_len,
                                input, output, check_tag, tag_len ) ) != 0 )
    {
        return( ret );
    }

    /* Check tag in "constant-time" */
    for( diff = 0, i = 0; i < tag_len; i++ )
        diff |= tag[i] ^ check_tag[i];

    if( diff != 0 )
    {
        mbedtls_platform_zeroize( output, length );
        return( MBEDTLS_ERR_CCM_AUTH_FAILED );
    }

    return( 0 );
}

/**
 * \brief           This function performs a CCM authenticated decryption of a
 *                  buffer.
 *
 * \param ctx       The CCM context to use for decryption. This must be
 *                  initialized and bound to a key.
 * \param length    The length of the input data in Bytes.
 * \param iv        The initialization vector (nonce). This must be a readable
 *                  buffer of at least \p iv_len Bytes.
 * \param iv_len    The length of the nonce in Bytes: 7, 8, 9, 10, 11, 12,
 *                  or 13. The length L of the message length field is
 *                  15 - \p iv_len.
 * \param add       The additional data field. This must be a readable buffer
 *                  of at least that \p add_len Bytes..
 * \param add_len   The length of additional data in Bytes.
 *                  This must be less than 2^16 - 2^8.
 * \param input     The buffer holding the input data. If \p length is greater
 *                  than zero, \p input must be a readable buffer of at least
 *                  that length.
 * \param output    The buffer holding the output data. If \p length is greater
 *                  than zero, \p output must be a writable buffer of at least
 *                  that length.
 * \param tag       The buffer holding the authentication field. This must be a
 *                  readable buffer of at least \p tag_len Bytes.
 * \param tag_len   The length of the authentication field to generate in Bytes:
 *                  4, 6, 8, 10, 12, 14 or 16.
 *
 * \return          \c 0 on success. This indicates that the message is
 * authentic. \return          #MBEDTLS_ERR_CCM_AUTH_FAILED if the tag does not
 * match. \return          A cipher-specific error code on calculation failure.
 */
int mbedtls_ccm_auth_decrypt( mbedtls_ccm_context *ctx, size_t length,
                              const unsigned char *iv, size_t iv_len,
                              const unsigned char *add, size_t add_len,
                              const unsigned char *input, unsigned char *output,
                              const unsigned char *tag, size_t tag_len )
{
    CCM_VALIDATE_RET( ctx );
    CCM_VALIDATE_RET( iv );
    CCM_VALIDATE_RET( add    || !add_len );
    CCM_VALIDATE_RET( input  || !length  );
    CCM_VALIDATE_RET( output || !length  );
    CCM_VALIDATE_RET( tag    || !tag_len );
    if( !tag_len )
        return( MBEDTLS_ERR_CCM_BAD_INPUT );
    return( mbedtls_ccm_star_auth_decrypt( ctx, length, iv, iv_len, add,
                add_len, input, output, tag, tag_len ) );
}

#if defined(MBEDTLS_SELF_TEST) && defined(MBEDTLS_AES_C)
/*
 * Examples 1 to 3 from SP800-38C Appendix C
 */

#define NB_TESTS 3
#define CCM_SELFTEST_PT_MAX_LEN 24
#define CCM_SELFTEST_CT_MAX_LEN 32
/*
 * The data is the same for all tests, only the used length changes
 */
static const unsigned char key_test_data[] = {
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f
};

static const unsigned char iv_test_data[] = {
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b
};

static const unsigned char ad_test_data[] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13
};

static const unsigned char msg_test_data[CCM_SELFTEST_PT_MAX_LEN] = {
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
};

static const size_t iv_len_test_data [NB_TESTS] = { 7, 8,  12 };
static const size_t add_len_test_data[NB_TESTS] = { 8, 16, 20 };
static const size_t msg_len_test_data[NB_TESTS] = { 4, 16, 24 };
static const size_t tag_len_test_data[NB_TESTS] = { 4, 6,  8  };

static const unsigned char res_test_data[NB_TESTS][CCM_SELFTEST_CT_MAX_LEN] = {
    {   0x71, 0x62, 0x01, 0x5b, 0x4d, 0xac, 0x25, 0x5d },
    {   0xd2, 0xa1, 0xf0, 0xe0, 0x51, 0xea, 0x5f, 0x62,
        0x08, 0x1a, 0x77, 0x92, 0x07, 0x3d, 0x59, 0x3d,
        0x1f, 0xc6, 0x4f, 0xbf, 0xac, 0xcd },
    {   0xe3, 0xb2, 0x01, 0xa9, 0xf5, 0xb7, 0x1a, 0x7a,
        0x9b, 0x1c, 0xea, 0xec, 0xcd, 0x97, 0xe7, 0x0b,
        0x61, 0x76, 0xaa, 0xd9, 0xa4, 0x42, 0x8a, 0xa5,
        0x48, 0x43, 0x92, 0xfb, 0xc1, 0xb0, 0x99, 0x51 }
};

/**
 * \brief          The CCM checkup routine.
 *
 * \return         \c 0 on success.
 * \return         \c 1 on failure.
 */
int mbedtls_ccm_self_test( int verbose )
{
    mbedtls_ccm_context ctx;
    /*
     * Some hardware accelerators require the input and output buffers
     * would be in RAM, because the flash is not accessible.
     * Use buffers on the stack to hold the test vectors data.
     */
    unsigned char plaintext[CCM_SELFTEST_PT_MAX_LEN];
    unsigned char ciphertext[CCM_SELFTEST_CT_MAX_LEN];
    size_t i;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;

    mbedtls_ccm_init( &ctx );

    if( mbedtls_ccm_setkey( &ctx, MBEDTLS_CIPHER_ID_AES, key_test_data,
                            8 * sizeof key_test_data ) != 0 )
    {
        if( verbose != 0 )
            mbedtls_printf( "  CCM: setup failed" );

        return( 1 );
    }

    for( i = 0; i < NB_TESTS; i++ )
    {
        if( verbose != 0 )
            mbedtls_printf( "  CCM-AES #%u: ", (unsigned int) i + 1 );

        mbedtls_platform_zeroize( plaintext, CCM_SELFTEST_PT_MAX_LEN );
        mbedtls_platform_zeroize( ciphertext, CCM_SELFTEST_CT_MAX_LEN );
        memcpy( plaintext, msg_test_data, msg_len_test_data[i] );

        ret = mbedtls_ccm_encrypt_and_tag( &ctx, msg_len_test_data[i],
                                           iv_test_data, iv_len_test_data[i],
                                           ad_test_data, add_len_test_data[i],
                                           plaintext, ciphertext,
                                           ciphertext + msg_len_test_data[i],
                                           tag_len_test_data[i] );

        if( ret != 0 ||
            timingsafe_bcmp( ciphertext, res_test_data[i],
                             msg_len_test_data[i] + tag_len_test_data[i] ) != 0 )
        {
            if( verbose != 0 )
                mbedtls_printf( "failed\n" );

            return( 1 );
        }
        mbedtls_platform_zeroize( plaintext, CCM_SELFTEST_PT_MAX_LEN );

        ret = mbedtls_ccm_auth_decrypt( &ctx, msg_len_test_data[i],
                                        iv_test_data, iv_len_test_data[i],
                                        ad_test_data, add_len_test_data[i],
                                        ciphertext, plaintext,
                                        ciphertext + msg_len_test_data[i],
                                        tag_len_test_data[i] );

        if( ret != 0 ||
            timingsafe_bcmp( plaintext, msg_test_data, msg_len_test_data[i] ) != 0 )
        {
            if( verbose != 0 )
                mbedtls_printf( "failed\n" );

            return( 1 );
        }

        if( verbose != 0 )
            mbedtls_printf( "passed\n" );
    }

    mbedtls_ccm_free( &ctx );

    if( verbose != 0 )
        mbedtls_printf( "\n" );

    return( 0 );
}

#endif /* MBEDTLS_SELF_TEST && MBEDTLS_AES_C */
