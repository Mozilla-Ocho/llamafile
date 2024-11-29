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
#include "third_party/mbedtls/sha1.h"
#include "third_party/mbedtls/x509_crt.h"
__static_yoink("mbedtls_notice");

/*
 *  X.509 certificate writing
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
 * - certificates: RFC 5280, updated by RFC 6818
 * - CSRs: PKCS#10 v1.7 aka RFC 2986
 * - attributes: PKCS#9 v2.0 aka RFC 2985
 */

#if defined(MBEDTLS_X509_CRT_WRITE_C)

/**
 * \brief           Initialize a CRT writing context
 *
 * \param ctx       CRT context to initialize
 */
void mbedtls_x509write_crt_init( mbedtls_x509write_cert *ctx )
{
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_x509write_cert ) );
    mbedtls_mpi_init( &ctx->serial );
    ctx->version = MBEDTLS_X509_CRT_VERSION_3;
}

/**
 * \brief           Free the contents of a CRT write context
 *
 * \param ctx       CRT context to free
 */
void mbedtls_x509write_crt_free( mbedtls_x509write_cert *ctx )
{
    mbedtls_mpi_free( &ctx->serial );

    mbedtls_asn1_free_named_data_list( &ctx->subject );
    mbedtls_asn1_free_named_data_list( &ctx->issuer );
    mbedtls_asn1_free_named_data_list( &ctx->extensions );

    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_x509write_cert ) );
}

/**
 * \brief           Set the verion for a Certificate
 *                  Default: MBEDTLS_X509_CRT_VERSION_3
 *
 * \param ctx       CRT context to use
 * \param version   version to set (MBEDTLS_X509_CRT_VERSION_1, MBEDTLS_X509_CRT_VERSION_2 or
 *                                  MBEDTLS_X509_CRT_VERSION_3)
 */
void mbedtls_x509write_crt_set_version( mbedtls_x509write_cert *ctx,
                                        int version )
{
    ctx->version = version;
}

/**
 * \brief           Set the MD algorithm to use for the signature
 *                  (e.g. MBEDTLS_MD_SHA1)
 *
 * \param ctx       CRT context to use
 * \param md_alg    MD algorithm to use
 */
void mbedtls_x509write_crt_set_md_alg( mbedtls_x509write_cert *ctx,
                                       mbedtls_md_type_t md_alg )
{
    ctx->md_alg = md_alg;
}

/**
 * \brief           Set the subject public key for the certificate
 *
 * \param ctx       CRT context to use
 * \param key       public key to include
 */
void mbedtls_x509write_crt_set_subject_key( mbedtls_x509write_cert *ctx,
                                            mbedtls_pk_context *key )
{
    ctx->subject_key = key;
}

/**
 * \brief           Set the issuer key used for signing the certificate
 *
 * \param ctx       CRT context to use
 * \param key       private key to sign with
 */
void mbedtls_x509write_crt_set_issuer_key( mbedtls_x509write_cert *ctx,
                                           mbedtls_pk_context *key )
{
    ctx->issuer_key = key;
}

/**
 * \brief           Set the subject name for a Certificate
 *                  Subject names should contain a comma-separated list
 *                  of OID types and values:
 *                  e.g. "C=UK,O=ARM,CN=mbed TLS Server 1"
 *
 * \param ctx           CRT context to use
 * \param subject_name  subject name to set
 *
 * \return          0 if subject name was parsed successfully, or
 *                  a specific error code
 */
int mbedtls_x509write_crt_set_subject_name( mbedtls_x509write_cert *ctx,
                                            const char *subject_name )
{
    return mbedtls_x509_string_to_names( &ctx->subject, subject_name );
}

/**
 * \brief           Set the issuer name for a Certificate
 *                  Issuer names should contain a comma-separated list
 *                  of OID types and values:
 *                  e.g. "C=UK,O=ARM,CN=mbed TLS CA"
 *
 * \param ctx           CRT context to use
 * \param issuer_name   issuer name to set
 *
 * \return          0 if issuer name was parsed successfully, or
 *                  a specific error code
 */
int mbedtls_x509write_crt_set_issuer_name( mbedtls_x509write_cert *ctx,
                                           const char *issuer_name )
{
    return mbedtls_x509_string_to_names( &ctx->issuer, issuer_name );
}

/**
 * \brief           Set the serial number for a Certificate.
 *
 * \param ctx       CRT context to use
 * \param serial    serial number to set
 *
 * \return          0 if successful
 */
int mbedtls_x509write_crt_set_serial( mbedtls_x509write_cert *ctx,
                                      const mbedtls_mpi *serial )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;

    if( ( ret = mbedtls_mpi_copy( &ctx->serial, serial ) ) != 0 )
        return( ret );

    return( 0 );
}

/**
 * \brief           Set the validity period for a Certificate
 *                  Timestamps should be in string format for UTC timezone
 *                  i.e. "YYYYMMDDhhmmss"
 *                  e.g. "20131231235959" for December 31st 2013
 *                       at 23:59:59
 *
 * \param ctx       CRT context to use
 * \param not_before    not_before timestamp
 * \param not_after     not_after timestamp
 *
 * \return          0 if timestamp was parsed successfully, or
 *                  a specific error code
 */
int mbedtls_x509write_crt_set_validity( mbedtls_x509write_cert *ctx,
                                        const char *not_before,
                                        const char *not_after )
{
    if( strlen( not_before ) != MBEDTLS_X509_RFC5280_UTC_TIME_LEN - 1 ||
        strlen( not_after )  != MBEDTLS_X509_RFC5280_UTC_TIME_LEN - 1 )
    {
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
    }
    strncpy( ctx->not_before, not_before, MBEDTLS_X509_RFC5280_UTC_TIME_LEN );
    strncpy( ctx->not_after , not_after , MBEDTLS_X509_RFC5280_UTC_TIME_LEN );
    ctx->not_before[MBEDTLS_X509_RFC5280_UTC_TIME_LEN - 1] = 'Z';
    ctx->not_after[MBEDTLS_X509_RFC5280_UTC_TIME_LEN - 1] = 'Z';
    return( 0 );
}

/**
 * \brief           Generic function to add to or replace an extension in the
 *                  CRT
 *
 * \param ctx       CRT context to use
 * \param oid       OID of the extension
 * \param oid_len   length of the OID
 * \param critical  if the extension is critical (per the RFC's definition)
 * \param val       value of the extension OCTET STRING
 * \param val_len   length of the value data
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_extension( mbedtls_x509write_cert *ctx,
                                         const char *oid, size_t oid_len,
                                         int critical, const unsigned char *val, 
                                         size_t val_len )
{
    return( mbedtls_x509_set_extension( &ctx->extensions, oid, oid_len,
                                        critical, val, val_len ) );
}

/**
 * \brief           Set the basicConstraints extension for a CRT
 *
 * \param ctx       CRT context to use
 * \param is_ca     is this a CA certificate
 * \param max_pathlen   maximum length of certificate chains below this
 *                      certificate (only for CA certificates, -1 is
 *                      inlimited)
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_basic_constraints( mbedtls_x509write_cert *ctx,
                                                 int is_ca, int max_pathlen )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char buf[9];
    unsigned char *c = buf + sizeof(buf);
    size_t len = 0;
    mbedtls_platform_zeroize( buf, sizeof(buf) );
    if( is_ca && max_pathlen > 127 )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
    if( is_ca )
    {
        if( max_pathlen >= 0 )
        {
            MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_int( &c, buf,
                                                               max_pathlen ) );
        }
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_bool( &c, buf, 1 ) );
    }
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( &c, buf,
                                                       MBEDTLS_ASN1_CONSTRUCTED |
                                                       MBEDTLS_ASN1_SEQUENCE ) );
    return(
        mbedtls_x509write_crt_set_extension( ctx, MBEDTLS_OID_BASIC_CONSTRAINTS,
                             MBEDTLS_OID_SIZE( MBEDTLS_OID_BASIC_CONSTRAINTS ),
                             is_ca, buf + sizeof(buf) - len, len ) );
}

#if defined(MBEDTLS_SHA1_C)
/**
 * \brief           Set the subjectKeyIdentifier extension for a CRT
 *                  Requires that mbedtls_x509write_crt_set_subject_key() has been
 *                  called before
 *
 * \param ctx       CRT context to use
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_subject_key_identifier( mbedtls_x509write_cert *ctx )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char buf[MBEDTLS_MPI_MAX_SIZE * 2 + 20]; /* tag, length + 2xMPI */
    unsigned char *c = buf + sizeof(buf);
    size_t len = 0;
    mbedtls_platform_zeroize( buf, sizeof(buf) );
    MBEDTLS_ASN1_CHK_ADD( len,
                mbedtls_pk_write_pubkey( &c, buf, ctx->subject_key ) );
    ret = mbedtls_sha1_ret( buf + sizeof( buf ) - len, len,
                            buf + sizeof( buf ) - 20 );
    if( ret != 0 )
        return( ret );
    c = buf + sizeof( buf ) - 20;
    len = 20;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len,
            mbedtls_asn1_write_tag( &c, buf, MBEDTLS_ASN1_OCTET_STRING ) );
    return mbedtls_x509write_crt_set_extension( ctx,
                 MBEDTLS_OID_SUBJECT_KEY_IDENTIFIER,
                 MBEDTLS_OID_SIZE( MBEDTLS_OID_SUBJECT_KEY_IDENTIFIER ),
                 0, buf + sizeof(buf) - len, len );
}

/**
 * \brief           Set the authorityKeyIdentifier extension for a CRT
 *                  Requires that mbedtls_x509write_crt_set_issuer_key() has been
 *                  called before
 *
 * \param ctx       CRT context to use
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_authority_key_identifier( mbedtls_x509write_cert *ctx )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned char buf[MBEDTLS_MPI_MAX_SIZE * 2 + 20]; /* tag, length + 2xMPI */
    unsigned char *c = buf + sizeof( buf );
    size_t len = 0;
    mbedtls_platform_zeroize( buf, sizeof(buf) );
    MBEDTLS_ASN1_CHK_ADD( len,
                          mbedtls_pk_write_pubkey( &c, buf, ctx->issuer_key ) );
    ret = mbedtls_sha1_ret( buf + sizeof( buf ) - len, len,
                            buf + sizeof( buf ) - 20 );
    if( ret != 0 )
        return( ret );
    c = buf + sizeof( buf ) - 20;
    len = 20;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len,
        mbedtls_asn1_write_tag( &c, buf, MBEDTLS_ASN1_CONTEXT_SPECIFIC | 0 ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len,
                          mbedtls_asn1_write_tag( &c, buf,
                                                  MBEDTLS_ASN1_CONSTRUCTED |
                                                  MBEDTLS_ASN1_SEQUENCE ) );
    return mbedtls_x509write_crt_set_extension(
        ctx, MBEDTLS_OID_AUTHORITY_KEY_IDENTIFIER,
        MBEDTLS_OID_SIZE( MBEDTLS_OID_AUTHORITY_KEY_IDENTIFIER ),
        0, buf + sizeof( buf ) - len, len );
}
#endif /* MBEDTLS_SHA1_C */

/**
 * \brief           Set the Key Usage Extension flags
 *                  (e.g. MBEDTLS_X509_KU_DIGITAL_SIGNATURE | MBEDTLS_X509_KU_KEY_CERT_SIGN)
 *
 * \param ctx       CRT context to use
 * \param key_usage key usage flags to set
 *
 * \return          0 if successful, or MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_key_usage( mbedtls_x509write_cert *ctx,
                                         unsigned int key_usage )
{
    unsigned char buf[5], ku[2];
    unsigned char *c;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    const unsigned int allowed_bits = MBEDTLS_X509_KU_DIGITAL_SIGNATURE |
        MBEDTLS_X509_KU_NON_REPUDIATION   |
        MBEDTLS_X509_KU_KEY_ENCIPHERMENT  |
        MBEDTLS_X509_KU_DATA_ENCIPHERMENT |
        MBEDTLS_X509_KU_KEY_AGREEMENT     |
        MBEDTLS_X509_KU_KEY_CERT_SIGN     |
        MBEDTLS_X509_KU_CRL_SIGN          |
        MBEDTLS_X509_KU_ENCIPHER_ONLY     |
        MBEDTLS_X509_KU_DECIPHER_ONLY;
    /* Check that nothing other than the allowed flags is set */
    if( ( key_usage & ~allowed_bits ) != 0 )
        return( MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE );
    c = buf + 5;
    ku[0] = (unsigned char)( key_usage      );
    ku[1] = (unsigned char)( key_usage >> 8 );
    ret = mbedtls_asn1_write_named_bitstring( &c, buf, ku, 9 );
    if( ret < 0 )
        return( ret );
    else if( ret < 3 || ret > 5 )
        return( MBEDTLS_ERR_X509_INVALID_FORMAT );
    ret = mbedtls_x509write_crt_set_extension( ctx, MBEDTLS_OID_KEY_USAGE,
                                   MBEDTLS_OID_SIZE( MBEDTLS_OID_KEY_USAGE ),
                                   1, c, (size_t)ret );
    if( ret != 0 )
        return( ret );
    return( 0 );
}

/**
 * \brief           Set the Netscape Cert Type flags
 *                  (e.g. MBEDTLS_X509_NS_CERT_TYPE_SSL_CLIENT | MBEDTLS_X509_NS_CERT_TYPE_EMAIL)
 *
 * \param ctx           CRT context to use
 * \param ns_cert_type  Netscape Cert Type flags to set
 *
 * \return          0 if successful, or MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_ns_cert_type( mbedtls_x509write_cert *ctx,
                                            unsigned char ns_cert_type )
{
    unsigned char buf[4];
    unsigned char *c;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    c = buf + 4;
    ret = mbedtls_asn1_write_named_bitstring( &c, buf, &ns_cert_type, 8 );
    if( ret < 3 || ret > 4 )
        return( ret );
    ret = mbedtls_x509write_crt_set_extension( ctx, MBEDTLS_OID_NS_CERT_TYPE,
                                   MBEDTLS_OID_SIZE( MBEDTLS_OID_NS_CERT_TYPE ),
                                   0, c, (size_t)ret );
    if( ret != 0 )
        return( ret );
    return( 0 );
}

/**
 * Writes Extended Key Usage section to certificate.
 *
 * @see mbedtls_x509write_crt_set_ns_cert_type()
 * @see RFC5280 §4.2.1.12
 */
int mbedtls_x509write_crt_set_ext_key_usage(mbedtls_x509write_cert *ctx,
                                            int ns_cert_type) {
  int ret;
  size_t len;
  unsigned char buf[256];
  unsigned char *c;
  if (!ns_cert_type) return 0;
  if (ns_cert_type & ~(MBEDTLS_X509_NS_CERT_TYPE_SSL_CLIENT |
                       MBEDTLS_X509_NS_CERT_TYPE_SSL_SERVER |
                       MBEDTLS_X509_NS_CERT_TYPE_EMAIL)) {
    return MBEDTLS_ERR_X509_BAD_INPUT_DATA;
  }
  len = 0;
  c = buf + sizeof(buf);
  mbedtls_platform_zeroize(buf, sizeof(buf));
  if (ns_cert_type & MBEDTLS_X509_NS_CERT_TYPE_SSL_CLIENT) {
    MBEDTLS_ASN1_CHK_ADD(
        len, mbedtls_asn1_write_oid(&c, buf, MBEDTLS_OID_CLIENT_AUTH,
                                    MBEDTLS_OID_SIZE(MBEDTLS_OID_CLIENT_AUTH)));
  }
  if (ns_cert_type & MBEDTLS_X509_NS_CERT_TYPE_SSL_SERVER) {
    MBEDTLS_ASN1_CHK_ADD(
        len, mbedtls_asn1_write_oid(&c, buf, MBEDTLS_OID_SERVER_AUTH,
                                    MBEDTLS_OID_SIZE(MBEDTLS_OID_SERVER_AUTH)));
  }
  if (ns_cert_type & MBEDTLS_X509_NS_CERT_TYPE_EMAIL) {
    MBEDTLS_ASN1_CHK_ADD(
        len,
        mbedtls_asn1_write_oid(&c, buf, MBEDTLS_OID_EMAIL_PROTECTION,
                               MBEDTLS_OID_SIZE(MBEDTLS_OID_EMAIL_PROTECTION)));
  }
  MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
  MBEDTLS_ASN1_CHK_ADD(
      len, mbedtls_asn1_write_tag(
               &c, buf, MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE));
  return mbedtls_x509write_crt_set_extension(
      ctx, MBEDTLS_OID_EXTENDED_KEY_USAGE,
      MBEDTLS_OID_SIZE(MBEDTLS_OID_EXTENDED_KEY_USAGE), false,
      buf + sizeof(buf) - len, len);
}

static int x509_write_time( unsigned char **p, unsigned char *start,
                            const char *t, size_t size )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    /*
     * write MBEDTLS_ASN1_UTC_TIME if year < 2050 (2 bytes shorter)
     */
    if( t[0] == '2' && t[1] == '0' && t[2] < '5' )
    {
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start,
                                             (const unsigned char *) t + 2,
                                             size - 2 ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start,
                                             MBEDTLS_ASN1_UTC_TIME ) );
    }
    else
    {
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start,
                                                  (const unsigned char *) t,
                                                  size ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start,
                                             MBEDTLS_ASN1_GENERALIZED_TIME ) );
    }
    return( (int) len );
}

/**
 * \brief           Write a built up certificate to a X509 DER structure
 *                  Note: data is written at the end of the buffer! Use the
 *                        return value to determine where you should start
 *                        using the buffer
 *
 * \param ctx       certificate to write away
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
int mbedtls_x509write_crt_der( mbedtls_x509write_cert *ctx,
                               unsigned char *buf, size_t size,
                               int (*f_rng)(void *, unsigned char *, size_t),
                               void *p_rng )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    const char *sig_oid;
    size_t sig_oid_len = 0;
    unsigned char *c, *c2;
    unsigned char hash[64];
    unsigned char sig[MBEDTLS_PK_SIGNATURE_MAX_SIZE];
    size_t sub_len = 0, pub_len = 0, sig_and_oid_len = 0, sig_len;
    size_t len = 0;
    mbedtls_pk_type_t pk_alg;
    /*
     * Prepare data to be signed at the end of the target buffer
     */
    c = buf + size;
    /* Signature algorithm needed in TBS, and later for actual signature */
    /* There's no direct way of extracting a signature algorithm
     * (represented as an element of mbedtls_pk_type_t) from a PK instance. */
    if( mbedtls_pk_can_do( ctx->issuer_key, MBEDTLS_PK_RSA ) )
        pk_alg = MBEDTLS_PK_RSA;
    else if( mbedtls_pk_can_do( ctx->issuer_key, MBEDTLS_PK_ECDSA ) )
        pk_alg = MBEDTLS_PK_ECDSA;
    else
        return( MBEDTLS_ERR_X509_INVALID_ALG );
    if( ( ret = mbedtls_oid_get_oid_by_sig_alg( pk_alg, ctx->md_alg,
                                          &sig_oid, &sig_oid_len ) ) != 0 )
    {
        return( ret );
    }
    /*
     *  Extensions  ::=  SEQUENCE SIZE (1..MAX) OF Extension
     */
    /* Only for v3 */
    if( ctx->version == MBEDTLS_X509_CRT_VERSION_3 )
    {
        MBEDTLS_ASN1_CHK_ADD( len,
                              mbedtls_x509_write_extensions( &c,
                                                      buf, ctx->extensions ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
        MBEDTLS_ASN1_CHK_ADD( len,
                              mbedtls_asn1_write_tag( &c, buf,
                                                      MBEDTLS_ASN1_CONSTRUCTED |
                                                      MBEDTLS_ASN1_SEQUENCE ) );
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
        MBEDTLS_ASN1_CHK_ADD( len,
                              mbedtls_asn1_write_tag( &c, buf,
                                               MBEDTLS_ASN1_CONTEXT_SPECIFIC |
                                               MBEDTLS_ASN1_CONSTRUCTED | 3 ) );
    }
    /*
     *  SubjectPublicKeyInfo
     */
    MBEDTLS_ASN1_CHK_ADD( pub_len,
                          mbedtls_pk_write_pubkey_der( ctx->subject_key,
                                                       buf, c - buf ) );
    c -= pub_len;
    len += pub_len;
    /*
     *  Subject  ::=  Name
     */
    MBEDTLS_ASN1_CHK_ADD( len,
                          mbedtls_x509_write_names( &c, buf,
                                                    ctx->subject ) );
    /*
     *  Validity ::= SEQUENCE {
     *       notBefore      Time,
     *       notAfter       Time }
     */
    sub_len = 0;
    MBEDTLS_ASN1_CHK_ADD( sub_len,
                          x509_write_time( &c, buf, ctx->not_after,
                                        MBEDTLS_X509_RFC5280_UTC_TIME_LEN ) );
    MBEDTLS_ASN1_CHK_ADD( sub_len,
                          x509_write_time( &c, buf, ctx->not_before,
                                        MBEDTLS_X509_RFC5280_UTC_TIME_LEN ) );
    len += sub_len;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, sub_len ) );
    MBEDTLS_ASN1_CHK_ADD( len,
                          mbedtls_asn1_write_tag( &c, buf,
                                                  MBEDTLS_ASN1_CONSTRUCTED |
                                                  MBEDTLS_ASN1_SEQUENCE ) );
    /*
     *  Issuer  ::=  Name
     */
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_x509_write_names( &c, buf,
                                                         ctx->issuer ) );
    /*
     *  Signature   ::=  AlgorithmIdentifier
     */
    MBEDTLS_ASN1_CHK_ADD( len,
                          mbedtls_asn1_write_algorithm_identifier( &c, buf,
                                              sig_oid, strlen( sig_oid ), 0 ) );
    /*
     *  Serial   ::=  INTEGER
     */
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_mpi( &c, buf,
                                                       &ctx->serial ) );
    /*
     *  Version  ::=  INTEGER  {  v1(0), v2(1), v3(2)  }
     */
    /* Can be omitted for v1 */
    if( ctx->version != MBEDTLS_X509_CRT_VERSION_1 )
    {
        sub_len = 0;
        MBEDTLS_ASN1_CHK_ADD( sub_len,
                              mbedtls_asn1_write_int( &c, buf, ctx->version ) );
        len += sub_len;
        MBEDTLS_ASN1_CHK_ADD( len,
                              mbedtls_asn1_write_len( &c, buf, sub_len ) );
        MBEDTLS_ASN1_CHK_ADD( len,
                              mbedtls_asn1_write_tag( &c, buf,
                                               MBEDTLS_ASN1_CONTEXT_SPECIFIC |
                                               MBEDTLS_ASN1_CONSTRUCTED | 0 ) );
    }
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len,
                mbedtls_asn1_write_tag( &c, buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                     MBEDTLS_ASN1_SEQUENCE ) );
    /*
     * Make signature
     */
    /* Compute hash of CRT. */
    if( ( ret = mbedtls_md( mbedtls_md_info_from_type( ctx->md_alg ), c,
                            len, hash ) ) != 0 )
    {
        return( ret );
    }
    if( ( ret = mbedtls_pk_sign( ctx->issuer_key, ctx->md_alg,
                                 hash, 0, sig, &sig_len,
                                 f_rng, p_rng ) ) != 0 )
    {
        return( ret );
    }
    /* Move CRT to the front of the buffer to have space
     * for the signature. */
    memmove( buf, c, len );
    c = buf + len;
    /* Add signature at the end of the buffer,
     * making sure that it doesn't underflow
     * into the CRT buffer. */
    c2 = buf + size;
    MBEDTLS_ASN1_CHK_ADD( sig_and_oid_len, mbedtls_x509_write_sig( &c2, c,
                                        sig_oid, sig_oid_len, sig, sig_len ) );
    /*
     * Memory layout after this step:
     *
     * buf       c=buf+len                c2            buf+size
     * [CRT0,...,CRTn, UNUSED, ..., UNUSED, SIG0, ..., SIGm]
     */
    /* Move raw CRT to just before the signature. */
    c = c2 - len;
    memmove( c, buf, len );
    len += sig_and_oid_len;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( &c, buf, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( &c, buf,
                                                 MBEDTLS_ASN1_CONSTRUCTED |
                                                 MBEDTLS_ASN1_SEQUENCE ) );
    return( (int) len );
}

#define PEM_BEGIN_CRT           "-----BEGIN CERTIFICATE-----\n"
#define PEM_END_CRT             "-----END CERTIFICATE-----\n"

/**
 * \brief           Write a built up certificate to a X509 PEM string
 *
 * \param ctx       certificate to write away
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
int mbedtls_x509write_crt_pem( mbedtls_x509write_cert *crt,
                               unsigned char *buf, size_t size,
                               int (*f_rng)(void *, unsigned char *, size_t),
                               void *p_rng )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t olen;
    if( ( ret = mbedtls_x509write_crt_der( crt, buf, size,
                                   f_rng, p_rng ) ) < 0 )
    {
        return( ret );
    }
    if( ( ret = mbedtls_pem_write_buffer( PEM_BEGIN_CRT, PEM_END_CRT,
                                          buf + size - ret, ret,
                                          buf, size, &olen ) ) != 0 )
    {
        return( ret );
    }
    return( 0 );
}

#endif /* MBEDTLS_X509_CRT_WRITE_C */
