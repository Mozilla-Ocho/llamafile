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
#include "third_party/mbedtls/asn1write.h"
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/oid.h"
#include "third_party/mbedtls/pem.h"
#include "third_party/mbedtls/platform.h"
#include "third_party/mbedtls/x509_csr.h"
__static_yoink("mbedtls_notice");

/*
 *  X.509 Certificate Signing Request writing
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
 * References:
 * - CSRs: PKCS#10 v1.7 aka RFC 2986
 * - attributes: PKCS#9 v2.0 aka RFC 2985
 */

#if defined(MBEDTLS_X509_CSR_WRITE_C)

/**
 * \brief           Initialize a CSR context
 *
 * \param ctx       CSR context to initialize
 */
void mbedtls_x509write_csr_init( mbedtls_x509write_csr *ctx )
{
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_x509write_csr ) );
}

/**
 * \brief           Free the contents of a CSR context
 *
 * \param ctx       CSR context to free
 */
void mbedtls_x509write_csr_free( mbedtls_x509write_csr *ctx )
{
    mbedtls_asn1_free_named_data_list( &ctx->subject );
    mbedtls_asn1_free_named_data_list( &ctx->extensions );

    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_x509write_csr ) );
}

/**
 * \brief           Set the MD algorithm to use for the signature
 *                  (e.g. MBEDTLS_MD_SHA1)
 *
 * \param ctx       CSR context to use
 * \param md_alg    MD algorithm to use
 */
void mbedtls_x509write_csr_set_md_alg( mbedtls_x509write_csr *ctx, mbedtls_md_type_t md_alg )
{
    ctx->md_alg = md_alg;
}

/**
 * \brief           Set the key for a CSR (public key will be included,
 *                  private key used to sign the CSR when writing it)
 *
 * \param ctx       CSR context to use
 * \param key       Asymetric key to include
 */
void mbedtls_x509write_csr_set_key( mbedtls_x509write_csr *ctx, mbedtls_pk_context *key )
{
    ctx->key = key;
}

/**
 * \brief           Set the subject name for a CSR
 *                  Subject names should contain a comma-separated list
 *                  of OID types and values:
 *                  e.g. "C=UK,O=ARM,CN=mbed TLS Server 1"
 *
 * \param ctx           CSR context to use
 * \param subject_name  subject name to set
 *
 * \return          0 if subject name was parsed successfully, or
 *                  a specific error code
 */
int mbedtls_x509write_csr_set_subject_name( mbedtls_x509write_csr *ctx,
                                            const char *subject_name )
{
    return mbedtls_x509_string_to_names( &ctx->subject, subject_name );
}

/**
 * \brief           Generic function to add to or replace an extension in the
 *                  CSR
 *
 * \param ctx       CSR context to use
 * \param oid       OID of the extension
 * \param oid_len   length of the OID
 * \param val       value of the extension OCTET STRING
 * \param val_len   length of the value data
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_csr_set_extension( mbedtls_x509write_csr *ctx,
                                         const char *oid, size_t oid_len,
                                         const unsigned char *val, size_t val_len )
{
    return mbedtls_x509_set_extension( &ctx->extensions, oid, oid_len,
                                       0, val, val_len );
}

/**
 * \brief           Set the Key Usage Extension flags
 *                  (e.g. MBEDTLS_X509_KU_DIGITAL_SIGNATURE | MBEDTLS_X509_KU_KEY_CERT_SIGN)
 *
 * \param ctx       CSR context to use
 * \param key_usage key usage flags to set
 *
 * \return          0 if successful, or MBEDTLS_ERR_X509_ALLOC_FAILED
 *
 * \note            The <code>decipherOnly</code> flag from the Key Usage
 *                  extension is represented by bit 8 (i.e.
 *                  <code>0x8000</code>), which cannot typically be represented
 *                  in an unsigned char. Therefore, the flag
 *                  <code>decipherOnly</code> (i.e.
 *                  #MBEDTLS_X509_KU_DECIPHER_ONLY) cannot be set using this
 *                  function.
 */
int mbedtls_x509write_csr_set_key_usage( mbedtls_x509write_csr *ctx, unsigned char key_usage )
{
    unsigned char buf[4];
    unsigned char *c;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    c = buf + 4;
    ret = mbedtls_asn1_write_named_bitstring( &c, buf, &key_usage, 8 );
    if( ret < 3 || ret > 4 )
        return ret;
    ret = mbedtls_x509write_csr_set_extension( ctx, MBEDTLS_OID_KEY_USAGE,
                                               MBEDTLS_OID_SIZE( MBEDTLS_OID_KEY_USAGE ),
                                               c, (size_t)ret );
    if( ret != 0 )
        return ret;
    return 0;
}

/**
 * \brief           Set the Netscape Cert Type flags
 *                  (e.g. MBEDTLS_X509_NS_CERT_TYPE_SSL_CLIENT | MBEDTLS_X509_NS_CERT_TYPE_EMAIL)
 *
 * \param ctx           CSR context to use
 * \param ns_cert_type  Netscape Cert Type flags to set
 *
 * \return          0 if successful, or MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_csr_set_ns_cert_type( mbedtls_x509write_csr *ctx,
                                            unsigned char ns_cert_type )
{
    unsigned char buf[4];
    unsigned char *c;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    c = buf + 4;
    ret = mbedtls_asn1_write_named_bitstring( &c, buf, &ns_cert_type, 8 );
    if( ret < 3 || ret > 4 )
        return ret;
    ret = mbedtls_x509write_csr_set_extension( ctx, MBEDTLS_OID_NS_CERT_TYPE,
                                       MBEDTLS_OID_SIZE( MBEDTLS_OID_NS_CERT_TYPE ),
                                       c, (size_t)ret );
    if( ret != 0 )
        return ret;
    return 0;
}

static int x509write_csr_der_internal( mbedtls_x509write_csr *ctx,
                                 unsigned char *buf,
                                 size_t size,
                                 unsigned char *sig,
                                 int (*f_rng)(void *, unsigned char *, size_t),
                                 void *p_rng )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    const char *sig_oid;
    size_t sig_oid_len = 0;
    unsigned char *c, *c2;
    unsigned char hash[64];
    size_t pub_len = 0, sig_and_oid_len = 0, sig_len;
    size_t len = 0;
    mbedtls_pk_type_t pk_alg;
    /* Write the CSR backwards starting from the end of buf */
    c = buf + size;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_x509_write_extensions( &c, buf,
                                                           ctx->extensions ) );
    if( len )
    {
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
        MBEDTLS_ASN1_CHK_ADD( len,
            mbedtls_asn1_write_tag(
                &c, buf,
                MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
        MBEDTLS_ASN1_CHK_ADD( len,
            mbedtls_asn1_write_tag(
                &c, buf,
                MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SET ) );
        MBEDTLS_ASN1_CHK_ADD( len,
            mbedtls_asn1_write_oid(
                &c, buf, MBEDTLS_OID_PKCS9_CSR_EXT_REQ,
                MBEDTLS_OID_SIZE( MBEDTLS_OID_PKCS9_CSR_EXT_REQ ) ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
        MBEDTLS_ASN1_CHK_ADD( len,
            mbedtls_asn1_write_tag(
                &c, buf,
                MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) );
    }
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len,
        mbedtls_asn1_write_tag(
            &c, buf,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_CONTEXT_SPECIFIC ) );
    MBEDTLS_ASN1_CHK_ADD( pub_len, mbedtls_pk_write_pubkey_der( ctx->key,
                                                              buf, c - buf ) );
    c -= pub_len;
    len += pub_len;
    /*
     *  Subject  ::=  Name
     */
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_x509_write_names( &c, buf,
                                                         ctx->subject ) );
    /*
     *  Version  ::=  INTEGER  {  v1(0), v2(1), v3(2)  }
     */
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_int( &c, buf, 0 ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len,
        mbedtls_asn1_write_tag(
            &c, buf,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) );
    /*
     * Sign the written CSR data into the sig buffer
     * Note: hash errors can happen only after an internal error
     */
    ret = mbedtls_md( mbedtls_md_info_from_type( ctx->md_alg ), c, len, hash );
    if( ret != 0 )
        return ret;
    if( ( ret = mbedtls_pk_sign( ctx->key, ctx->md_alg, hash, 0, sig, &sig_len,
                                 f_rng, p_rng ) ) != 0 )
    {
        return ret;
    }
    if( mbedtls_pk_can_do( ctx->key, MBEDTLS_PK_RSA ) )
        pk_alg = MBEDTLS_PK_RSA;
    else if( mbedtls_pk_can_do( ctx->key, MBEDTLS_PK_ECDSA ) )
        pk_alg = MBEDTLS_PK_ECDSA;
    else
        return( MBEDTLS_ERR_X509_INVALID_ALG );
    if( ( ret = mbedtls_oid_get_oid_by_sig_alg( pk_alg, ctx->md_alg,
                                              &sig_oid, &sig_oid_len ) ) != 0 )
    {
        return ret;
    }
    /*
     * Move the written CSR data to the start of buf to create space for
     * writing the signature into buf.
     */
    memmove( buf, c, len );
    /*
     * Write sig and its OID into buf backwards from the end of buf.
     * Note: mbedtls_x509_write_sig will check for c2 - ( buf + len ) < sig_len
     * and return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL if needed.
     */
    c2 = buf + size;
    MBEDTLS_ASN1_CHK_ADD( sig_and_oid_len,
        mbedtls_x509_write_sig( &c2, buf + len, sig_oid, sig_oid_len,
                                sig, sig_len ) );
    /*
     * Compact the space between the CSR data and signature by moving the
     * CSR data to the start of the signature.
     */
    c2 -= len;
    memmove( c2, buf, len );
    /* ASN encode the total size and tag the CSR data with it. */
    len += sig_and_oid_len;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c2, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len,
        mbedtls_asn1_write_tag(
            &c2, buf,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) );
    /* Zero the unused bytes at the start of buf */
    mbedtls_platform_zeroize( buf, c2 - buf);
    return( (int) len );
}

/**
 * \brief           Write a CSR (Certificate Signing Request) to a
 *                  DER structure
 *                  Note: data is written at the end of the buffer! Use the
 *                        return value to determine where you should start
 *                        using the buffer
 *
 * \param ctx       CSR to write away
 * \param buf       buffer to write to
 * \param size      size of the buffer
 * \param f_rng     RNG function (for signature, see note)
 * \param p_rng     RNG parameter
 *
 * \return          length of data written if successful, or a specific
 *                  error code
 *
 * \note            f_rng may be NULL if RSA is used for signature and the
 *                  signature is made offline (otherwise f_rng is desirable
 *                  for countermeasures against timing attacks).
 *                  ECDSA signatures always require a non-NULL f_rng.
 */
int mbedtls_x509write_csr_der( mbedtls_x509write_csr *ctx, unsigned char *buf,
                               size_t size,
                               int (*f_rng)(void *, unsigned char *, size_t),
                               void *p_rng )
{
    int ret;
    unsigned char *sig;
    if( ( sig = mbedtls_calloc( 1, MBEDTLS_PK_SIGNATURE_MAX_SIZE ) ) == NULL )
    {
        return( MBEDTLS_ERR_X509_ALLOC_FAILED );
    }
    ret = x509write_csr_der_internal( ctx, buf, size, sig, f_rng, p_rng );
    mbedtls_free( sig );
    return ret;
}

#define PEM_BEGIN_CSR           "-----BEGIN CERTIFICATE REQUEST-----\n"
#define PEM_END_CSR             "-----END CERTIFICATE REQUEST-----\n"

/**
 * \brief           Write a CSR (Certificate Signing Request) to a
 *                  PEM string
 *
 * \param ctx       CSR to write away
 * \param buf       buffer to write to
 * \param size      size of the buffer
 * \param f_rng     RNG function (for signature, see note)
 * \param p_rng     RNG parameter
 *
 * \return          0 if successful, or a specific error code
 *
 * \note            f_rng may be NULL if RSA is used for signature and the
 *                  signature is made offline (otherwise f_rng is desirable
 *                  for countermeasures against timing attacks).
 *                  ECDSA signatures always require a non-NULL f_rng.
 */
int mbedtls_x509write_csr_pem( mbedtls_x509write_csr *ctx, unsigned char *buf, size_t size,
                       int (*f_rng)(void *, unsigned char *, size_t),
                       void *p_rng )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t olen = 0;
    if( ( ret = mbedtls_x509write_csr_der( ctx, buf, size,
                                   f_rng, p_rng ) ) < 0 )
    {
        return ret;
    }
    if( ( ret = mbedtls_pem_write_buffer( PEM_BEGIN_CSR, PEM_END_CSR,
                                  buf + size - ret,
                                  ret, buf, size, &olen ) ) != 0 )
    {
        return ret;
    }
    return 0;
}

#endif /* MBEDTLS_X509_CSR_WRITE_C */
