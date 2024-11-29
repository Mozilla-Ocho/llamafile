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
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/hkdf.h"
#include "third_party/mbedtls/platform.h"
__static_yoink("mbedtls_notice");

/**
 * @fileoverview HKDF implementation (RFC 5869)
 */

/**
 * \brief            HMAC-based Extract-and-Expand Key Derivation Function
 *
 * \param  md        A hash function; md.size denotes the length of the hash
 *                   function output in bytes.
 * \param  salt      An optional salt value (a non-secret random value);
 *                   if the salt is not provided, a string of all zeros of
 *                   md.size length is used as the salt.
 * \param  salt_len  The length in bytes of the optional \p salt.
 * \param  ikm       The input keying material.
 * \param  ikm_len   The length in bytes of \p ikm.
 * \param  info      An optional context and application specific information
 *                   string. This can be a zero-length string.
 * \param  info_len  The length of \p info in bytes.
 * \param  okm       The output keying material of \p okm_len bytes.
 * \param  okm_len   The length of the output keying material in bytes. This
 *                   must be less than or equal to 255 * md.size bytes.
 *
 * \return 0 on success.
 * \return #MBEDTLS_ERR_HKDF_BAD_INPUT_DATA when the parameters are invalid.
 * \return An MBEDTLS_ERR_MD_* error for errors returned from the underlying
 *         MD layer.
 */
int mbedtls_hkdf( const mbedtls_md_info_t *md, const unsigned char *salt,
                  size_t salt_len, const unsigned char *ikm, size_t ikm_len,
                  const unsigned char *info, size_t info_len,
                  unsigned char *okm, size_t okm_len )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char prk[MBEDTLS_MD_MAX_SIZE];
    ret = mbedtls_hkdf_extract( md, salt, salt_len, ikm, ikm_len, prk );
    if( ret == 0 )
    {
        ret = mbedtls_hkdf_expand( md, prk, mbedtls_md_get_size( md ),
                                   info, info_len, okm, okm_len );
    }
    mbedtls_platform_zeroize( prk, sizeof( prk ) );
    return( ret );
}

/**
 * \brief  Takes input keying material \p ikm and extract from it a
 *         fixed-length pseudorandom key \p prk.
 *
 * \warning    This function should only be used if the security of it has been
 *             studied and established in that particular context (eg. TLS 1.3
 *             key schedule). For standard HKDF security guarantees use
 *             \c mbedtls_hkdf instead.
 *
 * \param       md        A hash function; md.size denotes the length of the
 *                        hash function output in bytes.
 * \param       salt      An optional salt value (a non-secret random value);
 *                        if the salt is not provided, a string of all zeros
 *                        of md.size length is used as the salt.
 * \param       salt_len  The length in bytes of the optional \p salt.
 * \param       ikm       The input keying material.
 * \param       ikm_len   The length in bytes of \p ikm.
 * \param[out]  prk       A pseudorandom key of at least md.size bytes.
 *
 * \return 0 on success.
 * \return #MBEDTLS_ERR_HKDF_BAD_INPUT_DATA when the parameters are invalid.
 * \return An MBEDTLS_ERR_MD_* error for errors returned from the underlying
 *         MD layer.
 */
int mbedtls_hkdf_extract( const mbedtls_md_info_t *md,
                          const unsigned char *salt, size_t salt_len,
                          const unsigned char *ikm, size_t ikm_len,
                          unsigned char *prk )
{
    unsigned char null_salt[MBEDTLS_MD_MAX_SIZE] = { '\0' };
    if( salt == NULL )
    {
        size_t hash_len;
        if( salt_len != 0 )
        {
            return MBEDTLS_ERR_HKDF_BAD_INPUT_DATA;
        }
        hash_len = mbedtls_md_get_size( md );
        if( hash_len == 0 )
        {
            return MBEDTLS_ERR_HKDF_BAD_INPUT_DATA;
        }
        salt = null_salt;
        salt_len = hash_len;
    }
    return( mbedtls_md_hmac( md, salt, salt_len, ikm, ikm_len, prk ) );
}

/**
 * \brief            Expand the supplied \p prk into several additional
 *                   pseudorandom keys, which is the output of the HKDF.
 *
 * \param  md        A hash function; md.size denotes the length of the hash
 *                   function output in bytes.
 * \param  prk       A pseudorandom key of at least md.size bytes. \p prk is
 *                   usually the output from the HKDF extract step.
 * \param  prk_len   The length in bytes of \p prk.
 * \param  info      An optional context and application specific information
 *                   string. This can be a zero-length string.
 * \param  info_len  The length of \p info in bytes.
 * \param  okm       The output keying material of \p okm_len bytes.
 * \param  okm_len   The length of the output keying material in bytes. This
 *                   must be less than or equal to 255 * md.size bytes.
 *
 * \return           0 on success
 * \return           #MBEDTLS_ERR_HKDF_BAD_INPUT_DATA when the
 *                   parameters are invalid.
 * \return           An MBEDTLS_ERR_MD_* error for errors returned from
 *                   the underlying MD layer.
 *
 * \warning          This function should only be used if its security has
 *                   been studied and established in that particular context
 *                   (eg. TLS 1.3 key schedule). For standard HKDF security
 *                   guarantees use \c mbedtls_hkdf instead.
 */
int mbedtls_hkdf_expand( const mbedtls_md_info_t *md, const unsigned char *prk,
                         size_t prk_len, const unsigned char *info,
                         size_t info_len, unsigned char *okm, size_t okm_len )
{
    size_t hash_len;
    size_t where = 0;
    size_t n;
    size_t t_len = 0;
    size_t i;
    int ret = 0;
    mbedtls_md_context_t ctx;
    unsigned char t[MBEDTLS_MD_MAX_SIZE];
    if( !okm ) return( MBEDTLS_ERR_HKDF_BAD_INPUT_DATA );
    hash_len = mbedtls_md_get_size( md );
    if( prk_len < hash_len || hash_len == 0 )
    {
        return( MBEDTLS_ERR_HKDF_BAD_INPUT_DATA );
    }
    if( info == NULL )
    {
        info = (const unsigned char *) "";
        info_len = 0;
    }
    n = okm_len / hash_len;
    if( okm_len % hash_len != 0 )
    {
        n++;
    }
    /*
     * Per RFC 5869 Section 2.3, okm_len must not exceed
     * 255 times the hash length
     */
    if( n > 255 ) return( MBEDTLS_ERR_HKDF_BAD_INPUT_DATA );
    mbedtls_md_init( &ctx );
    if(( ret = mbedtls_md_setup( &ctx, md, 1 ) )) goto exit;
    mbedtls_platform_zeroize( t, hash_len );
    /*
     * Compute T = T(1) | T(2) | T(3) | ... | T(N)
     * Where T(N) is defined in RFC 5869 Section 2.3
     */
    for( i = 1; i <= n; i++ )
    {
        size_t num_to_copy;
        unsigned char c = i & 0xff;
        if(( ret = mbedtls_md_hmac_starts( &ctx, prk, prk_len ) )) goto exit;
        if(( ret = mbedtls_md_hmac_update( &ctx, t, t_len ) )) goto exit;
        if(( ret = mbedtls_md_hmac_update( &ctx, info, info_len ) )) goto exit;
        if(( ret = mbedtls_md_hmac_update( &ctx, &c, 1 ) )) goto exit;
        if(( ret = mbedtls_md_hmac_finish( &ctx, t ) )) goto exit;
        num_to_copy = i != n ? hash_len : okm_len - where;
        memcpy( okm + where, t, num_to_copy );
        where += hash_len;
        t_len = hash_len;
    }
exit:
    mbedtls_md_free( &ctx );
    mbedtls_platform_zeroize( t, sizeof( t ) );
    return( ret );
}
