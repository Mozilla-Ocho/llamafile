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
#include <stdbool.h>
#include "third_party/mbedtls/x509_crt.h"
#include <libc/calls/calls.h>
#include <libc/calls/struct/dirent.h>
#include <libc/calls/struct/stat.h>
#include <libc/serialize.h>
#include <libc/limits.h>
#include <libc/log/log.h>
#include <libc/mem/mem.h>
#include <libc/stdio/stdio.h>
#include <libc/sysv/consts/s.h>
#include <net/http/http.h>
#include <net/http/ip.h>
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/oid.h"
#include "third_party/mbedtls/pem.h"
#include "third_party/mbedtls/platform.h"
__static_yoink("mbedtls_notice");

/*
 *  X.509 certificate parsing and verification
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
 *  The ITU-T X.509 standard defines a certificate format for PKI.
 *
 *  http://www.ietf.org/rfc/rfc5280.txt (Certificates and CRLs)
 *  http://www.ietf.org/rfc/rfc3279.txt (Alg IDs for CRLs)
 *  http://www.ietf.org/rfc/rfc2986.txt (CSRs, aka PKCS#10)
 *
 *  http://www.itu.int/ITU-T/studygroups/com17/languages/X.680-0207.pdf
 *  http://www.itu.int/ITU-T/studygroups/com17/languages/X.690-0207.pdf
 *
 *  [SIRO] https://cabforum.org/wp-content/uploads/Chunghwatelecom201503cabforumV4.pdf
 */

#if defined(MBEDTLS_X509_CRT_PARSE_C)

/*
 * Item in a verification chain: cert and flags for it
 */
typedef struct {
    mbedtls_x509_crt *crt;
    uint32_t flags;
} x509_crt_verify_chain_item;

/*
 * Max size of verification chain: end-entity + intermediates + trusted root
 */
#define X509_MAX_VERIFY_CHAIN_SIZE    ( MBEDTLS_X509_MAX_INTERMEDIATE_CA + 2 )

/*
 * Default profile
 */
const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_default = {
#if defined(MBEDTLS_TLS_DEFAULT_ALLOW_SHA1_IN_CERTIFICATES)
    /* Allow SHA-1 (weak, but still safe in controlled environments) */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA1 ) |
#endif
    /* Only SHA-2 hashes */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA224 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA256 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA384 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA512 ),
    0xFFFFFFF, /* Any PK alg    */
    0xFFFFFFF, /* Any curve     */
    2048,
};

/*
 * Next-default profile
 */
const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_next = {
    /* Hashes from SHA-256 and above */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA256 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA384 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA512 ),
    0xFFFFFFF, /* Any PK alg    */
#if defined(MBEDTLS_ECP_C)
    /* Curves at or above 128-bit security level */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP256R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP384R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP521R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_BP256R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_BP384R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_BP512R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP256K1 ),
#else
    0,
#endif
    2048,
};

/*
 * NSA Suite B Profile
 */
const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_suiteb = {
    /* Only SHA-256 and 384 */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA256 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_MD_SHA384 ),
    /* Only ECDSA */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_PK_ECDSA ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_PK_ECKEY ),
#if defined(MBEDTLS_ECP_C)
    /* Only NIST P-256 and P-384 */
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP256R1 ) |
    MBEDTLS_X509_ID_FLAG( MBEDTLS_ECP_DP_SECP384R1 ),
#else
    0,
#endif
    0,
};

/*
 * Check md_alg against profile
 * Return 0 if md_alg is acceptable for this profile, -1 otherwise
 */
static int x509_profile_check_md_alg( const mbedtls_x509_crt_profile *profile,
                                      mbedtls_md_type_t md_alg )
{
    if( md_alg == MBEDTLS_MD_NONE )
        return -1;
    if( ( profile->allowed_mds & MBEDTLS_X509_ID_FLAG( md_alg ) ) )
        return 0;
    return -1;
}

/*
 * Check pk_alg against profile
 * Return 0 if pk_alg is acceptable for this profile, -1 otherwise
 */
static int x509_profile_check_pk_alg( const mbedtls_x509_crt_profile *profile,
                                      mbedtls_pk_type_t pk_alg )
{
    if( pk_alg == MBEDTLS_PK_NONE )
        return -1;
    if( ( profile->allowed_pks & MBEDTLS_X509_ID_FLAG( pk_alg ) ) )
        return 0;
    return -1;
}

/*
 * Check key against profile
 * Return 0 if pk is acceptable for this profile, -1 otherwise
 */
static int x509_profile_check_key( const mbedtls_x509_crt_profile *profile,
                                   const mbedtls_pk_context *pk )
{
    const mbedtls_pk_type_t pk_alg = mbedtls_pk_get_type( pk );
#if defined(MBEDTLS_RSA_C)
    if( pk_alg == MBEDTLS_PK_RSA || pk_alg == MBEDTLS_PK_RSASSA_PSS )
    {
        if( mbedtls_pk_get_bitlen( pk ) >= profile->rsa_min_bitlen )
            return 0;
        return -1;
    }
#endif
#if defined(MBEDTLS_ECP_C)
    if( pk_alg == MBEDTLS_PK_ECDSA ||
        pk_alg == MBEDTLS_PK_ECKEY ||
        pk_alg == MBEDTLS_PK_ECKEY_DH )
    {
        const mbedtls_ecp_group_id gid = mbedtls_pk_ec( *pk )->grp.id;
        if( gid == MBEDTLS_ECP_DP_NONE )
            return -1;
        if( ( profile->allowed_curves & MBEDTLS_X509_ID_FLAG( gid ) ) )
            return 0;
        return -1;
    }
#endif
    return -1;
}

/*
 * Return 0 if name matches wildcard, -1 otherwise
 */
static int x509_check_wildcard( const char *cn, const mbedtls_x509_buf *name )
{
    size_t i;
    size_t cn_idx = 0, cn_len = strlen( cn );
    /* We can't have a match if there is no wildcard to match */
    if( name->len < 3 || name->p[0] != '*' || name->p[1] != '.' )
        return -1;
    for( i = 0; i < cn_len; ++i )
    {
        if( cn[i] == '.' )
        {
            cn_idx = i;
            break;
        }
    }
    if( cn_idx == 0 )
        return -1;
    if( cn_len - cn_idx == name->len - 1 &&
        memcasecmp( name->p + 1, cn + cn_idx, name->len - 1 ) == 0 )
    {
        return 0;
    }
    return -1;
}

/*
 * Compare two X.509 strings, case-insensitive, and allowing for some encoding
 * variations (but not all).
 *
 * Return 0 if equal, -1 otherwise.
 */
static int x509_string_cmp( const mbedtls_x509_buf *a, const mbedtls_x509_buf *b )
{
    if( a->tag == b->tag &&
        a->len == b->len &&
        timingsafe_bcmp( a->p, b->p, b->len ) == 0 )
    {
        return 0;
    }
    if( ( a->tag == MBEDTLS_ASN1_UTF8_STRING || a->tag == MBEDTLS_ASN1_PRINTABLE_STRING ) &&
        ( b->tag == MBEDTLS_ASN1_UTF8_STRING || b->tag == MBEDTLS_ASN1_PRINTABLE_STRING ) &&
        a->len == b->len &&
        memcasecmp( a->p, b->p, b->len ) == 0 )
    {
        return 0;
    }
    return -1;
}

/*
 * Compare two X.509 Names (aka rdnSequence).
 *
 * See RFC 5280 section 7.1, though we don't implement the whole algorithm:
 * we sometimes return unequal when the full algorithm would return equal,
 * but never the other way. (In particular, we don't do Unicode normalisation
 * or space folding.)
 *
 * Return 0 if equal, -1 otherwise.
 */
int mbedtls_x509_name_cmp( const mbedtls_x509_name *a, const mbedtls_x509_name *b )
{
    /* Avoid recursion, it might not be optimised by the compiler */
    while( a != NULL || b != NULL )
    {
        if( a == NULL || b == NULL )
            return -1;
        /* type */
        if( a->oid.tag != b->oid.tag ||
            a->oid.len != b->oid.len ||
            timingsafe_bcmp( a->oid.p, b->oid.p, b->oid.len ) )
        {
            return -1;
        }
        /* value */
        if( x509_string_cmp( &a->val, &b->val ) )
            return -1;
        /* structure of the list of sets */
        if( a->next_merged != b->next_merged )
            return -1;
        a = a->next;
        b = b->next;
    }
    /* a == NULL == b */
    return 0;
}

/*
 * Reset (init or clear) a verify_chain
 */
static void x509_crt_verify_chain_reset(
    mbedtls_x509_crt_verify_chain *ver_chain )
{
    size_t i;
    for( i = 0; i < MBEDTLS_X509_MAX_VERIFY_CHAIN_SIZE; i++ )
    {
        ver_chain->items[i].crt = NULL;
        ver_chain->items[i].flags = (uint32_t) -1;
    }
    ver_chain->len = 0;
#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
    ver_chain->trust_ca_cb_result = NULL;
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */
}

/*
 *  Version  ::=  INTEGER  {  v1(0), v2(1), v3(2)  }
 */
static int x509_get_version( unsigned char **p,
                             const unsigned char *end,
                             int *ver )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len;
    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | 0 ) ) )
    {
        if( ret == MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
        {
            *ver = 0;
            return 0;
        }
        return( MBEDTLS_ERR_X509_INVALID_FORMAT + ret );
    }
    end = *p + len;
    if( ( ret = mbedtls_asn1_get_int( p, end, ver ) ) )
        return( MBEDTLS_ERR_X509_INVALID_VERSION + ret );
    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_VERSION +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    return 0;
}

/*
 *  Validity ::= SEQUENCE {
 *       notBefore      Time,
 *       notAfter       Time }
 */
static int x509_get_dates( unsigned char **p,
                           const unsigned char *end,
                           mbedtls_x509_time *from,
                           mbedtls_x509_time *to )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len;
    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
        return( MBEDTLS_ERR_X509_INVALID_DATE + ret );
    end = *p + len;
    if( ( ret = mbedtls_x509_get_time( p, end, from ) ) )
        return ret;
    if( ( ret = mbedtls_x509_get_time( p, end, to ) ) )
        return ret;
    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_DATE +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    return 0;
}

/*
 * X.509 v2/v3 unique identifier (not parsed)
 */
static int x509_get_uid( unsigned char **p,
                         const unsigned char *end,
                         mbedtls_x509_buf *uid, int n )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    if( *p == end )
        return 0;
    uid->tag = **p;
    if( ( ret = mbedtls_asn1_get_tag( p, end, &uid->len,
            MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_ASN1_CONSTRUCTED | n ) ) )
    {
        if( ret == MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
            return 0;
        return( MBEDTLS_ERR_X509_INVALID_FORMAT + ret );
    }
    uid->p = *p;
    *p += uid->len;
    return 0;
}

static int x509_get_basic_constraints( unsigned char **p,
                                       const unsigned char *end,
                                       int *ca_istrue,
                                       int *max_pathlen )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len;
    /*
     * BasicConstraints ::= SEQUENCE {
     *      cA                      BOOLEAN DEFAULT FALSE,
     *      pathLenConstraint       INTEGER (0..MAX) OPTIONAL }
     */
    *ca_istrue = 0; /* DEFAULT FALSE */
    *max_pathlen = 0; /* endless */
    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    if( *p == end )
        return 0;
    if( ( ret = mbedtls_asn1_get_bool( p, end, ca_istrue ) ) )
    {
        if( ret == MBEDTLS_ERR_ASN1_UNEXPECTED_TAG )
            ret = mbedtls_asn1_get_int( p, end, ca_istrue );
        if( ret )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
        if( *ca_istrue )
            *ca_istrue = 1;
    }
    if( *p == end )
        return 0;
    if( ( ret = mbedtls_asn1_get_int( p, end, max_pathlen ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    /* Do not accept max_pathlen equal to INT_MAX to avoid a signed integer
     * overflow, which is an undefined behavior. */
    if( *max_pathlen == INT_MAX )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_INVALID_LENGTH );
    (*max_pathlen)++;
    return 0;
}

static int x509_get_ns_cert_type( unsigned char **p,
                                  const unsigned char *end,
                                  unsigned char *ns_cert_type)
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_x509_bitstring bs = { 0, 0, NULL };
    if( ( ret = mbedtls_asn1_get_bitstring( p, end, &bs ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    if( bs.len != 1 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_INVALID_LENGTH );
    /* Get actual bitstring */
    *ns_cert_type = *bs.p;
    return 0;
}

static int x509_get_key_usage( unsigned char **p,
                               const unsigned char *end,
                               unsigned int *key_usage)
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t i;
    mbedtls_x509_bitstring bs = { 0, 0, NULL };
    if( ( ret = mbedtls_asn1_get_bitstring( p, end, &bs ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    if( bs.len < 1 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_INVALID_LENGTH );
    /* Get actual bitstring */
    *key_usage = 0;
    for( i = 0; i < bs.len && i < sizeof( unsigned int ); i++ )
    {
        *key_usage |= (unsigned int) bs.p[i] << (8*i);
    }
    return 0;
}

/*
 * ExtKeyUsageSyntax ::= SEQUENCE SIZE (1..MAX) OF KeyPurposeId
 *
 * KeyPurposeId ::= OBJECT IDENTIFIER
 */
static int x509_get_ext_key_usage( unsigned char **p,
                                   const unsigned char *end,
                                   mbedtls_x509_sequence *ext_key_usage)
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    if( ( ret = mbedtls_asn1_get_sequence_of( p, end, ext_key_usage, MBEDTLS_ASN1_OID ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    /* Sequence length must be >= 1 */
    if( ext_key_usage->buf.p == NULL )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_INVALID_LENGTH );
    return 0;
}

/*
 * SubjectAltName ::= GeneralNames
 *
 * GeneralNames ::= SEQUENCE SIZE (1..MAX) OF GeneralName
 *
 * GeneralName ::= CHOICE {
 *      otherName                       [0]     OtherName,
 *      rfc822Name                      [1]     IA5String,
 *      dNSName                         [2]     IA5String,
 *      x400Address                     [3]     ORAddress,
 *      directoryName                   [4]     Name,
 *      ediPartyName                    [5]     EDIPartyName,
 *      uniformResourceIdentifier       [6]     IA5String,
 *      iPAddress                       [7]     OCTET STRING,
 *      registeredID                    [8]     OBJECT IDENTIFIER }
 *
 * OtherName ::= SEQUENCE {
 *      type-id    OBJECT IDENTIFIER,
 *      value      [0] EXPLICIT ANY DEFINED BY type-id }
 *
 * EDIPartyName ::= SEQUENCE {
 *      nameAssigner            [0]     DirectoryString OPTIONAL,
 *      partyName               [1]     DirectoryString }
 *
 * NOTE: we list all types, but only use dNSName and otherName
 * of type HwModuleName, as defined in RFC 4108, at this point.
 */
static int x509_get_subject_alt_name( unsigned char **p,
                                      const unsigned char *end,
                                      mbedtls_x509_sequence *subject_alt_name )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len, tag_len;
    mbedtls_asn1_buf *buf;
    unsigned char tag;
    mbedtls_asn1_sequence *cur = subject_alt_name;
    /* Get main sequence tag */
    if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    if( *p + len != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    while( *p < end )
    {
        mbedtls_x509_subject_alternative_name dummy_san_buf;
        mbedtls_platform_zeroize( &dummy_san_buf, sizeof( dummy_san_buf ) );
        tag = **p;
        (*p)++;
        if( ( ret = mbedtls_asn1_get_len( p, end, &tag_len ) ) )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
        if( ( tag & MBEDTLS_ASN1_TAG_CLASS_MASK ) !=
                MBEDTLS_ASN1_CONTEXT_SPECIFIC )
        {
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                    MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );
        }
        /*
         * Check that the SAN is structured correctly.
         */
        ret = mbedtls_x509_parse_subject_alt_name( &(cur->buf), &dummy_san_buf );
        /*
         * In case the extension is malformed, return an error,
         * and clear the allocated sequences.
         */
        if( ret && ret != MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE )
        {
            mbedtls_x509_sequence *seq_cur = subject_alt_name->next;
            mbedtls_x509_sequence *seq_prv;
            while( seq_cur != NULL )
            {
                seq_prv = seq_cur;
                seq_cur = seq_cur->next;
                mbedtls_platform_zeroize( seq_prv,
                                          sizeof( mbedtls_x509_sequence ) );
                mbedtls_free( seq_prv );
            }
            subject_alt_name->next = NULL;
            return ret;
        }
        /* Allocate and assign next pointer */
        if( cur->buf.p != NULL )
        {
            if( cur->next != NULL )
                return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS );
            cur->next = mbedtls_calloc( 1, sizeof( mbedtls_asn1_sequence ) );
            if( cur->next == NULL )
                return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                        MBEDTLS_ERR_ASN1_ALLOC_FAILED );
            cur = cur->next;
        }
        buf = &(cur->buf);
        buf->tag = tag;
        buf->p = *p;
        buf->len = tag_len;
        *p += buf->len;
    }
    /* Set final sequence entry's next pointer to NULL */
    cur->next = NULL;
    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    return 0;
}

/*
 * id-ce-certificatePolicies OBJECT IDENTIFIER ::=  { id-ce 32 }
 *
 * anyPolicy OBJECT IDENTIFIER ::= { id-ce-certificatePolicies 0 }
 *
 * certificatePolicies ::= SEQUENCE SIZE (1..MAX) OF PolicyInformation
 *
 * PolicyInformation ::= SEQUENCE {
 *     policyIdentifier   CertPolicyId,
 *     policyQualifiers   SEQUENCE SIZE (1..MAX) OF
 *                             PolicyQualifierInfo OPTIONAL }
 *
 * CertPolicyId ::= OBJECT IDENTIFIER
 *
 * PolicyQualifierInfo ::= SEQUENCE {
 *      policyQualifierId  PolicyQualifierId,
 *      qualifier          ANY DEFINED BY policyQualifierId }
 *
 * -- policyQualifierIds for Internet policy qualifiers
 *
 * id-qt          OBJECT IDENTIFIER ::=  { id-pkix 2 }
 * id-qt-cps      OBJECT IDENTIFIER ::=  { id-qt 1 }
 * id-qt-unotice  OBJECT IDENTIFIER ::=  { id-qt 2 }
 *
 * PolicyQualifierId ::= OBJECT IDENTIFIER ( id-qt-cps | id-qt-unotice )
 *
 * Qualifier ::= CHOICE {
 *      cPSuri           CPSuri,
 *      userNotice       UserNotice }
 *
 * CPSuri ::= IA5String
 *
 * UserNotice ::= SEQUENCE {
 *      noticeRef        NoticeReference OPTIONAL,
 *      explicitText     DisplayText OPTIONAL }
 *
 * NoticeReference ::= SEQUENCE {
 *      organization     DisplayText,
 *      noticeNumbers    SEQUENCE OF INTEGER }
 *
 * DisplayText ::= CHOICE {
 *      ia5String        IA5String      (SIZE (1..200)),
 *      visibleString    VisibleString  (SIZE (1..200)),
 *      bmpString        BMPString      (SIZE (1..200)),
 *      utf8String       UTF8String     (SIZE (1..200)) }
 *
 * NOTE: we only parse and use anyPolicy without qualifiers at this point
 * as defined in RFC 5280.
 */
static int x509_get_certificate_policies( unsigned char **p,
                                          const unsigned char *end,
                                          mbedtls_x509_sequence *certificate_policies )
{
    int ret, parse_ret = 0;
    size_t len;
    mbedtls_asn1_buf *buf;
    mbedtls_asn1_sequence *cur = certificate_policies;
    /* Get main sequence tag */
    ret = mbedtls_asn1_get_tag( p, end, &len,
                             MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE );
    if( ret )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    if( *p + len != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    /*
     * Cannot be an empty sequence.
     */
    if( len == 0 )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    while( *p < end )
    {
        mbedtls_x509_buf policy_oid;
        const unsigned char *policy_end;
        /*
         * Get the policy sequence
         */
        if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
                MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
        policy_end = *p + len;
        if( ( ret = mbedtls_asn1_get_tag( p, policy_end, &len,
                                          MBEDTLS_ASN1_OID ) ) )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
        policy_oid.tag = MBEDTLS_ASN1_OID;
        policy_oid.len = len;
        policy_oid.p = *p;
        /*
         * Only AnyPolicy is currently supported when enforcing policy.
         */
        if( MBEDTLS_OID_CMP( MBEDTLS_OID_ANY_POLICY, &policy_oid ) )
        {
            /*
             * Set the parsing return code but continue parsing, in case this
             * extension is critical and MBEDTLS_X509_ALLOW_UNSUPPORTED_CRITICAL_EXTENSION
             * is configured.
             */
            parse_ret = MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE;
        }
        /* Allocate and assign next pointer */
        if( cur->buf.p != NULL )
        {
            if( cur->next != NULL )
                return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS );
            cur->next = mbedtls_calloc( 1, sizeof( mbedtls_asn1_sequence ) );
            if( cur->next == NULL )
                return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                        MBEDTLS_ERR_ASN1_ALLOC_FAILED );
            cur = cur->next;
        }
        buf = &( cur->buf );
        buf->tag = policy_oid.tag;
        buf->p = policy_oid.p;
        buf->len = policy_oid.len;
        *p += len;
       /*
        * If there is an optional qualifier, then *p < policy_end
        * Check the Qualifier len to verify it doesn't exceed policy_end.
        */
        if( *p < policy_end )
        {
            if( ( ret = mbedtls_asn1_get_tag( p, policy_end, &len,
                     MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
                return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
            /*
             * Skip the optional policy qualifiers.
             */
            *p += len;
        }
        if( *p != policy_end )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                    MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    /* Set final sequence entry's next pointer to NULL */
    cur->next = NULL;
    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    return( parse_ret );
}

/*
 * X.509 v3 extensions
 */
static int x509_get_crt_ext( unsigned char **p,
                             const unsigned char *end,
                             mbedtls_x509_crt *crt,
                             mbedtls_x509_crt_ext_cb_t cb,
                             void *p_ctx )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len;
    unsigned char *end_ext_data, *start_ext_octet, *end_ext_octet;
    if( *p == end )
        return 0;
    if( ( ret = mbedtls_x509_get_ext( p, end, &crt->v3_ext, 3 ) ) )
        return ret;
    end = crt->v3_ext.p + crt->v3_ext.len;
    while( *p < end )
    {
        /*
         * Extension  ::=  SEQUENCE  {
         *      extnID      OBJECT IDENTIFIER,
         *      critical    BOOLEAN DEFAULT FALSE,
         *      extnValue   OCTET STRING  }
         */
        mbedtls_x509_buf extn_oid = {0, 0, NULL};
        int is_critical = 0; /* DEFAULT FALSE */
        int ext_type = 0;
        if( ( ret = mbedtls_asn1_get_tag( p, end, &len,
                MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
        end_ext_data = *p + len;
        /* Get extension ID */
        if( ( ret = mbedtls_asn1_get_tag( p, end_ext_data, &extn_oid.len,
                                          MBEDTLS_ASN1_OID ) ) )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
        extn_oid.tag = MBEDTLS_ASN1_OID;
        extn_oid.p = *p;
        *p += extn_oid.len;
        /* Get optional critical */
        if( ( ret = mbedtls_asn1_get_bool( p, end_ext_data, &is_critical ) ) &&
            ( ret != MBEDTLS_ERR_ASN1_UNEXPECTED_TAG ) )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
        /* Data should be octet string type */
        if( ( ret = mbedtls_asn1_get_tag( p, end_ext_data, &len,
                MBEDTLS_ASN1_OCTET_STRING ) ) )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
        start_ext_octet = *p;
        end_ext_octet = *p + len;
        if( end_ext_octet != end_ext_data )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                    MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
        /*
         * Detect supported extensions
         */
        ret = mbedtls_oid_get_x509_ext_type( &extn_oid, &ext_type );
        if( ret )
        {
            /* Give the callback (if any) a chance to handle the extension */
            if( cb != NULL )
            {
                ret = cb( p_ctx, crt, &extn_oid, is_critical, *p, end_ext_octet );
                if( ret && is_critical )
                    return ret;
                *p = end_ext_octet;
                continue;
            }
            /* No parser found, skip extension */
            *p = end_ext_octet;
#if !defined(MBEDTLS_X509_ALLOW_UNSUPPORTED_CRITICAL_EXTENSION)
            if( is_critical )
            {
                /* Data is marked as critical: fail */
                return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                        MBEDTLS_ERR_ASN1_UNEXPECTED_TAG );
            }
#endif
            continue;
        }
        /* Forbid repeated extensions */
        if( ( crt->ext_types & ext_type ) )
            return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS );
        crt->ext_types |= ext_type;
        switch( ext_type )
        {
        case MBEDTLS_X509_EXT_BASIC_CONSTRAINTS:
            /* Parse basic constraints */
            if( ( ret = x509_get_basic_constraints( p, end_ext_octet,
                    &crt->ca_istrue, &crt->max_pathlen ) ) )
                return ret;
            break;
        case MBEDTLS_X509_EXT_KEY_USAGE:
            /* Parse key usage */
            if( ( ret = x509_get_key_usage( p, end_ext_octet,
                    &crt->key_usage ) ) )
                return ret;
            break;
        case MBEDTLS_X509_EXT_EXTENDED_KEY_USAGE:
            /* Parse extended key usage */
            if( ( ret = x509_get_ext_key_usage( p, end_ext_octet,
                    &crt->ext_key_usage ) ) )
                return ret;
            break;
        case MBEDTLS_X509_EXT_SUBJECT_ALT_NAME:
            /* Parse subject alt name */
            if( ( ret = x509_get_subject_alt_name( p, end_ext_octet,
                    &crt->subject_alt_names ) ) )
                return ret;
            break;
        case MBEDTLS_X509_EXT_NS_CERT_TYPE:
            /* Parse netscape certificate type */
            if( ( ret = x509_get_ns_cert_type( p, end_ext_octet,
                    &crt->ns_cert_type ) ) )
                return ret;
            break;
        case MBEDTLS_OID_X509_EXT_CERTIFICATE_POLICIES:
            /* Parse certificate policies type */
            if( ( ret = x509_get_certificate_policies( p, end_ext_octet,
                    &crt->certificate_policies ) ) )
            {
                /* Give the callback (if any) a chance to handle the extension
                 * if it contains unsupported policies */
                if( ret == MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE && cb != NULL &&
                    cb( p_ctx, crt, &extn_oid, is_critical,
                        start_ext_octet, end_ext_octet ) == 0 )
                    break;
#if !defined(MBEDTLS_X509_ALLOW_UNSUPPORTED_CRITICAL_EXTENSION)
                if( is_critical )
                    return ret;
                else
#endif
                /*
                 * If MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE is returned, then we
                 * cannot interpret or enforce the policy. However, it is up to
                 * the user to choose how to enforce the policies,
                 * unless the extension is critical.
                 */
                if( ret != MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE )
                    return ret;
            }
            break;
        default:
            /*
             * If this is a non-critical extension, which the oid layer
             * supports, but there isn't an x509 parser for it,
             * skip the extension.
             */
#if !defined(MBEDTLS_X509_ALLOW_UNSUPPORTED_CRITICAL_EXTENSION)
            if( is_critical )
                return( MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE );
            else
#endif
                *p = end_ext_octet;
        }
    }
    if( *p != end )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    return 0;
}

/*
 * Parse and fill a single X.509 certificate in DER format
 */
static int x509_crt_parse_der_core( mbedtls_x509_crt *crt,
                                    const unsigned char *buf,
                                    size_t buflen,
                                    int make_copy,
                                    mbedtls_x509_crt_ext_cb_t cb,
                                    void *p_ctx )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len;
    unsigned char *p, *end, *crt_end;
    mbedtls_x509_buf sig_params1, sig_params2, sig_oid2;
    mbedtls_platform_zeroize( &sig_params1, sizeof( mbedtls_x509_buf ) );
    mbedtls_platform_zeroize( &sig_params2, sizeof( mbedtls_x509_buf ) );
    mbedtls_platform_zeroize( &sig_oid2, sizeof( mbedtls_x509_buf ) );
    /*
     * Check for valid input
     */
    if( crt == NULL || buf == NULL )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
    /* Use the original buffer until we figure out actual length. */
    p = (unsigned char*) buf;
    len = buflen;
    end = p + len;
    /*
     * Certificate  ::=  SEQUENCE  {
     *      tbsCertificate       TBSCertificate,
     *      signatureAlgorithm   AlgorithmIdentifier,
     *      signatureValue       BIT STRING  }
     */
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT );
    }
    end = crt_end = p + len;
    crt->raw.len = crt_end - buf;
    if( make_copy )
    {
        /* Create and populate a new buffer for the raw field. */
        crt->raw.p = p = mbedtls_calloc( 1, crt->raw.len );
        if( crt->raw.p == NULL )
            return( MBEDTLS_ERR_X509_ALLOC_FAILED );
        memcpy( crt->raw.p, buf, crt->raw.len );
        crt->own_buffer = 1;
        p += crt->raw.len - len;
        end = crt_end = p + len;
    }
    else
    {
        crt->raw.p = (unsigned char*) buf;
        crt->own_buffer = 0;
    }
    /*
     * TBSCertificate  ::=  SEQUENCE  {
     */
    crt->tbs.p = p;
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT + ret );
    }
    end = p + len;
    crt->tbs.len = end - crt->tbs.p;
    /*
     * Version  ::=  INTEGER  {  v1(0), v2(1), v3(2)  }
     *
     * CertificateSerialNumber  ::=  INTEGER
     *
     * signature            AlgorithmIdentifier
     */
    if( ( ret = x509_get_version(  &p, end, &crt->version  ) ) ||
        ( ret = mbedtls_x509_get_serial(   &p, end, &crt->serial   ) ) ||
        ( ret = mbedtls_x509_get_alg(      &p, end, &crt->sig_oid,
                                            &sig_params1 ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return ret;
    }
    if( crt->version < 0 || crt->version > 2 )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_UNKNOWN_VERSION );
    }
    crt->version++;
    if( ( ret = mbedtls_x509_get_sig_alg( &crt->sig_oid, &sig_params1,
                                  &crt->sig_md, &crt->sig_pk,
                                  &crt->sig_opts ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return ret;
    }
    /*
     * issuer               Name
     */
    crt->issuer_raw.p = p;
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT + ret );
    }
    if( ( ret = mbedtls_x509_get_name( &p, p + len, &crt->issuer ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return ret;
    }
    crt->issuer_raw.len = p - crt->issuer_raw.p;
    /*
     * Validity ::= SEQUENCE {
     *      notBefore      Time,
     *      notAfter       Time }
     *
     */
    if( ( ret = x509_get_dates( &p, end, &crt->valid_from,
                                         &crt->valid_to ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return ret;
    }
    /*
     * subject              Name
     */
    crt->subject_raw.p = p;
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT + ret );
    }
    if( len && ( ret = mbedtls_x509_get_name( &p, p + len, &crt->subject ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return ret;
    }
    crt->subject_raw.len = p - crt->subject_raw.p;
    /*
     * SubjectPublicKeyInfo
     */
    crt->pk_raw.p = p;
    if( ( ret = mbedtls_pk_parse_subpubkey( &p, end, &crt->pk ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return ret;
    }
    crt->pk_raw.len = p - crt->pk_raw.p;
    /*
     *  issuerUniqueID  [1]  IMPLICIT UniqueIdentifier OPTIONAL,
     *                       -- If present, version shall be v2 or v3
     *  subjectUniqueID [2]  IMPLICIT UniqueIdentifier OPTIONAL,
     *                       -- If present, version shall be v2 or v3
     *  extensions      [3]  EXPLICIT Extensions OPTIONAL
     *                       -- If present, version shall be v3
     */
    if( crt->version == 2 || crt->version == 3 )
    {
        ret = x509_get_uid( &p, end, &crt->issuer_id,  1 );
        if( ret )
        {
            mbedtls_x509_crt_free( crt );
            return ret;
        }
    }
    if( crt->version == 2 || crt->version == 3 )
    {
        ret = x509_get_uid( &p, end, &crt->subject_id,  2 );
        if( ret )
        {
            mbedtls_x509_crt_free( crt );
            return ret;
        }
    }
#if !defined(MBEDTLS_X509_ALLOW_EXTENSIONS_NON_V3)
    if( crt->version == 3 )
#endif
    {
        ret = x509_get_crt_ext( &p, end, crt, cb, p_ctx );
        if( ret )
        {
            mbedtls_x509_crt_free( crt );
            return ret;
        }
    }
    if( p != end )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    end = crt_end;
    /*
     *  }
     *  -- end of TBSCertificate
     *
     *  signatureAlgorithm   AlgorithmIdentifier,
     *  signatureValue       BIT STRING
     */
    if( ( ret = mbedtls_x509_get_alg( &p, end, &sig_oid2, &sig_params2 ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return ret;
    }
    if( crt->sig_oid.len != sig_oid2.len ||
        timingsafe_bcmp( crt->sig_oid.p, sig_oid2.p, crt->sig_oid.len ) ||
        sig_params1.tag != sig_params2.tag ||
        sig_params1.len != sig_params2.len ||
        ( sig_params1.len &&
          timingsafe_bcmp( sig_params1.p, sig_params2.p, sig_params1.len ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_SIG_MISMATCH );
    }
    if( ( ret = mbedtls_x509_get_sig( &p, end, &crt->sig ) ) )
    {
        mbedtls_x509_crt_free( crt );
        return ret;
    }
    if( p != end )
    {
        mbedtls_x509_crt_free( crt );
        return( MBEDTLS_ERR_X509_INVALID_FORMAT +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    return 0;
}

/*
 * Parse one X.509 certificate in DER format from a buffer and add them to a
 * chained list
 */
static int mbedtls_x509_crt_parse_der_internal( mbedtls_x509_crt *chain,
                                                const unsigned char *buf,
                                                size_t buflen,
                                                int make_copy,
                                                mbedtls_x509_crt_ext_cb_t cb,
                                                void *p_ctx )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_x509_crt *crt = chain, *prev = NULL;
    /*
     * Check for valid input
     */
    if( crt == NULL || buf == NULL )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
    while( crt->version && crt->next != NULL )
    {
        prev = crt;
        crt = crt->next;
    }
    /*
     * Add new certificate on the end of the chain if needed.
     */
    if( crt->version && crt->next == NULL )
    {
        crt->next = mbedtls_calloc( 1, sizeof( mbedtls_x509_crt ) );
        if( crt->next == NULL )
            return( MBEDTLS_ERR_X509_ALLOC_FAILED );
        prev = crt;
        mbedtls_x509_crt_init( crt->next );
        crt = crt->next;
    }
    ret = x509_crt_parse_der_core( crt, buf, buflen, make_copy, cb, p_ctx );
    if( ret )
    {
        if( prev )
            prev->next = NULL;
        if( crt != chain )
            mbedtls_free( crt );
        return ret;
    }
    return 0;
}

/**
 * \brief          Parse a single DER formatted certificate and add it
 *                 to the end of the provided chained list. This is a
 *                 variant of mbedtls_x509_crt_parse_der() which takes
 *                 temporary ownership of the CRT buffer until the CRT
 *                 is destroyed.
 *
 * \param chain    The pointer to the start of the CRT chain to attach to.
 *                 When parsing the first CRT in a chain, this should point
 *                 to an instance of ::mbedtls_x509_crt initialized through
 *                 mbedtls_x509_crt_init().
 * \param buf      The address of the readable buffer holding the DER encoded
 *                 certificate to use. On success, this buffer must be
 *                 retained and not be changed for the liftetime of the
 *                 CRT chain \p chain, that is, until \p chain is destroyed
 *                 through a call to mbedtls_x509_crt_free().
 * \param buflen   The size in Bytes of \p buf.
 *
 * \note           This call is functionally equivalent to
 *                 mbedtls_x509_crt_parse_der(), but it avoids creating a
 *                 copy of the input buffer at the cost of stronger lifetime
 *                 constraints. This is useful in constrained environments
 *                 where duplication of the CRT cannot be tolerated.
 *
 * \return         \c 0 if successful.
 * \return         A negative error code on failure.
 */
int mbedtls_x509_crt_parse_der_nocopy( mbedtls_x509_crt *chain,
                                       const unsigned char *buf,
                                       size_t buflen )
{
    return( mbedtls_x509_crt_parse_der_internal( chain, buf, buflen, 0, NULL, NULL ) );
}

/**
 * \brief            Parse a single DER formatted certificate and add it
 *                   to the end of the provided chained list.
 *
 * \param chain      The pointer to the start of the CRT chain to attach to.
 *                   When parsing the first CRT in a chain, this should point
 *                   to an instance of ::mbedtls_x509_crt initialized through
 *                   mbedtls_x509_crt_init().
 * \param buf        The buffer holding the DER encoded certificate.
 * \param buflen     The size in Bytes of \p buf.
 * \param make_copy  When not zero this function makes an internal copy of the
 *                   CRT buffer \p buf. In particular, \p buf may be destroyed
 *                   or reused after this call returns.
 *                   When zero this function avoids duplicating the CRT buffer
 *                   by taking temporary ownership thereof until the CRT
 *                   is destroyed (like mbedtls_x509_crt_parse_der_nocopy())
 * \param cb         A callback invoked for every unsupported certificate
 *                   extension.
 * \param p_ctx      An opaque context passed to the callback.
 *
 * \note             This call is functionally equivalent to
 *                   mbedtls_x509_crt_parse_der(), and/or
 *                   mbedtls_x509_crt_parse_der_nocopy()
 *                   but it calls the callback with every unsupported
 *                   certificate extension and additionally the
 *                   "certificate policies" extension if it contains any
 *                   unsupported certificate policies.
 *                   The callback must return a negative error code if it
 *                   does not know how to handle such an extension.
 *                   When the callback fails to parse a critical extension
 *                   mbedtls_x509_crt_parse_der_with_ext_cb() also fails.
 *                   When the callback fails to parse a non critical extension
 *                   mbedtls_x509_crt_parse_der_with_ext_cb() simply skips
 *                   the extension and continues parsing.
 *                   Future versions of the library may invoke the callback
 *                   in other cases, if and when the need arises.
 *
 * \return           \c 0 if successful.
 * \return           A negative error code on failure.
 */
int mbedtls_x509_crt_parse_der_with_ext_cb( mbedtls_x509_crt *chain,
                                            const unsigned char *buf,
                                            size_t buflen,
                                            int make_copy,
                                            mbedtls_x509_crt_ext_cb_t cb,
                                            void *p_ctx )
{
    return( mbedtls_x509_crt_parse_der_internal( chain, buf, buflen, make_copy, cb, p_ctx ) );
}

/**
 * \brief          Parse a single DER formatted certificate and add it
 *                 to the end of the provided chained list.
 *
 * \param chain    The pointer to the start of the CRT chain to attach to.
 *                 When parsing the first CRT in a chain, this should point
 *                 to an instance of ::mbedtls_x509_crt initialized through
 *                 mbedtls_x509_crt_init().
 * \param buf      The buffer holding the DER encoded certificate.
 * \param buflen   The size in Bytes of \p buf.
 *
 * \note           This function makes an internal copy of the CRT buffer
 *                 \p buf. In particular, \p buf may be destroyed or reused
 *                 after this call returns. To avoid duplicating the CRT
 *                 buffer (at the cost of stricter lifetime constraints),
 *                 use mbedtls_x509_crt_parse_der_nocopy() instead.
 *
 * \return         \c 0 if successful.
 * \return         A negative error code on failure.
 */
int mbedtls_x509_crt_parse_der( mbedtls_x509_crt *chain,
                                const unsigned char *buf,
                                size_t buflen )
{
    return( mbedtls_x509_crt_parse_der_internal( chain, buf, buflen, 1, NULL, NULL ) );
}

/**
 * \brief          Parse one DER-encoded or one or more concatenated PEM-encoded
 *                 certificates and add them to the chained list.
 *
 *                 For CRTs in PEM encoding, the function parses permissively:
 *                 if at least one certificate can be parsed, the function
 *                 returns the number of certificates for which parsing failed
 *                 (hence \c 0 if all certificates were parsed successfully).
 *                 If no certificate could be parsed, the function returns
 *                 the first (negative) error encountered during parsing.
 *
 *                 PEM encoded certificates may be interleaved by other data
 *                 such as human readable descriptions of their content, as
 *                 long as the certificates are enclosed in the PEM specific
 *                 '-----{BEGIN/END} CERTIFICATE-----' delimiters.
 *
 * \param chain    The chain to which to add the parsed certificates.
 * \param buf      The buffer holding the certificate data in PEM or DER format.
 *                 For certificates in PEM encoding, this may be a concatenation
 *                 of multiple certificates; for DER encoding, the buffer must
 *                 comprise exactly one certificate.
 * \param buflen   The size of \p buf, including the terminating \c NULL byte
 *                 in case of PEM encoded data.
 *
 * \return         \c 0 if all certificates were parsed successfully.
 * \return         The (positive) number of certificates that couldn't
 *                 be parsed if parsing was partly successful (see above).
 * \return         A negative X509 or PEM error code otherwise.
 *
 */
int mbedtls_x509_crt_parse( mbedtls_x509_crt *chain,
                            const unsigned char *buf,
                            size_t buflen )
{
#if defined(MBEDTLS_PEM_PARSE_C)
    int success = 0, first_error = 0, total_failed = 0;
    int buf_format = MBEDTLS_X509_FORMAT_DER;
#endif
    /*
     * Check for valid input
     */
    if( chain == NULL || buf == NULL )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
    /*
     * Determine buffer content. Buffer contains either one DER certificate or
     * one or more PEM certificates.
     */
#if defined(MBEDTLS_PEM_PARSE_C)
    if( buflen && buf[buflen - 1] == '\0' &&
        strstr( (const char *) buf, "-----BEGIN CERTIFICATE-----" ) != NULL )
    {
        buf_format = MBEDTLS_X509_FORMAT_PEM;
    }
    if( buf_format == MBEDTLS_X509_FORMAT_DER )
        return mbedtls_x509_crt_parse_der( chain, buf, buflen );
#else
    return mbedtls_x509_crt_parse_der( chain, buf, buflen );
#endif
#if defined(MBEDTLS_PEM_PARSE_C)
    if( buf_format == MBEDTLS_X509_FORMAT_PEM )
    {
        int ret = MBEDTLS_ERR_THIS_CORRUPTION;
        mbedtls_pem_context pem;
        /* 1 rather than 0 since the terminating NULL byte is counted in */
        while( buflen > 1 )
        {
            size_t use_len;
            mbedtls_pem_init( &pem );
            /* If we get there, we know the string is null-terminated */
            ret = mbedtls_pem_read_buffer( &pem,
                           "-----BEGIN CERTIFICATE-----",
                           "-----END CERTIFICATE-----",
                           buf, NULL, 0, &use_len );
            if( ret == 0 )
            {
                /*
                 * Was PEM encoded
                 */
                buflen -= use_len;
                buf += use_len;
            }
            else if( ret == MBEDTLS_ERR_PEM_BAD_INPUT_DATA )
            {
                return ret;
            }
            else if( ret != MBEDTLS_ERR_PEM_NO_HEADER_FOOTER_PRESENT )
            {
                mbedtls_pem_free( &pem );
                /*
                 * PEM header and footer were found
                 */
                buflen -= use_len;
                buf += use_len;
                if( first_error == 0 )
                    first_error = ret;
                total_failed++;
                continue;
            }
            else
                break;
            ret = mbedtls_x509_crt_parse_der( chain, pem.buf, pem.buflen );
            mbedtls_pem_free( &pem );
            if( ret )
            {
                /*
                 * Quit parsing on a memory error
                 */
                if( ret == MBEDTLS_ERR_X509_ALLOC_FAILED )
                    return ret;
                if( first_error == 0 )
                    first_error = ret;
                total_failed++;
                continue;
            }
            success = 1;
        }
    }
    if( success )
        return( total_failed );
    else if( first_error )
        return( first_error );
    else
        return( MBEDTLS_ERR_X509_CERT_UNKNOWN_FORMAT );
#endif /* MBEDTLS_PEM_PARSE_C */
}

/**
 * \brief          Load one or more certificates and add them
 *                 to the chained list. Parses permissively. If some
 *                 certificates can be parsed, the result is the number
 *                 of failed certificates it encountered. If none complete
 *                 correctly, the first error is returned.
 *
 * \param chain    points to the start of the chain
 * \param path     filename to read the certificates from
 *
 * \return         0 if all certificates parsed successfully, a positive number
 *                 if partly successful or a specific X509 or PEM error code
 */
int mbedtls_x509_crt_parse_file( mbedtls_x509_crt *chain, const char *path )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t n;
    unsigned char *buf;
    if( ( ret = mbedtls_pk_load_file( path, &buf, &n ) ) )
        return ret;
    ret = mbedtls_x509_crt_parse( chain, buf, n );
    mbedtls_platform_zeroize( buf, n );
    mbedtls_free( buf );
    return ret;
}

/**
 * \brief          Load one or more certificate files from a path and add them
 *                 to the chained list. Parses permissively. If some
 *                 certificates can be parsed, the result is the number
 *                 of failed certificates it encountered. If none complete
 *                 correctly, the first error is returned.
 *
 * \param chain    points to the start of the chain
 * \param path     directory / folder to read the certificate files from
 *
 * \return         0 if all certificates parsed successfully, a positive number
 *                 if partly successful or a specific X509 or PEM error code
 */
int mbedtls_x509_crt_parse_path( mbedtls_x509_crt *chain, const char *path )
{
    int ret = 0;
    int t_ret;
    int snp_ret;
    struct stat sb;
    struct dirent *entry;
    char entry_name[MBEDTLS_X509_MAX_FILE_PATH_LEN];
    DIR *dir = opendir( path );
    if( dir == NULL )
        return( MBEDTLS_ERR_X509_FILE_IO_ERROR );
    mbedtls_platform_zeroize( &sb, sizeof( sb ) );
    while( ( entry = readdir( dir ) ) != NULL )
    {
        snp_ret = mbedtls_snprintf( entry_name, sizeof entry_name,
                                    "%s/%s", path, entry->d_name );
        if( snp_ret < 0 || (size_t)snp_ret >= sizeof entry_name )
        {
            ret = MBEDTLS_ERR_X509_BUFFER_TOO_SMALL;
            goto cleanup;
        }
        else if( stat( entry_name, &sb ) == -1 )
        {
            ret = MBEDTLS_ERR_X509_FILE_IO_ERROR;
            goto cleanup;
        }
        if( !S_ISREG( sb.st_mode ) )
            continue;
        // Ignore parse errors
        //
        t_ret = mbedtls_x509_crt_parse_file( chain, entry_name );
        if( t_ret < 0 )
            ret++;
        else
            ret += t_ret;
    }
cleanup:
    closedir( dir );
    return ret;
}

/*
 * OtherName ::= SEQUENCE {
 *      type-id    OBJECT IDENTIFIER,
 *      value      [0] EXPLICIT ANY DEFINED BY type-id }
 *
 * HardwareModuleName ::= SEQUENCE {
 *                           hwType OBJECT IDENTIFIER,
 *                           hwSerialNum OCTET STRING }
 *
 * NOTE: we currently only parse and use otherName of type HwModuleName,
 * as defined in RFC 4108.
 */
static int x509_get_other_name( const mbedtls_x509_buf *subject_alt_name,
                                mbedtls_x509_san_other_name *other_name )
{
    int ret = 0;
    size_t len;
    unsigned char *p = subject_alt_name->p;
    const unsigned char *end = p + subject_alt_name->len;
    mbedtls_x509_buf cur_oid;
    if( ( subject_alt_name->tag &
        ( MBEDTLS_ASN1_TAG_CLASS_MASK | MBEDTLS_ASN1_TAG_VALUE_MASK ) ) !=
        ( MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_X509_SAN_OTHER_NAME ) )
    {
        /*
         * The given subject alternative name is not of type "othername".
         */
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
    }
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
                                      MBEDTLS_ASN1_OID ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    cur_oid.tag = MBEDTLS_ASN1_OID;
    cur_oid.p = p;
    cur_oid.len = len;
    /*
     * Only HwModuleName is currently supported.
     */
    if( MBEDTLS_OID_CMP( MBEDTLS_OID_ON_HW_MODULE_NAME, &cur_oid ) )
    {
        return( MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE );
    }
    if( p + len >= end )
    {
        mbedtls_platform_zeroize( other_name, sizeof( *other_name ) );
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    p += len;
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
            MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_CONTEXT_SPECIFIC ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
                     MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) ) )
       return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len, MBEDTLS_ASN1_OID ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    other_name->value.hardware_module_name.oid.tag = MBEDTLS_ASN1_OID;
    other_name->value.hardware_module_name.oid.p = p;
    other_name->value.hardware_module_name.oid.len = len;
    if( p + len >= end )
    {
        mbedtls_platform_zeroize( other_name, sizeof( *other_name ) );
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    p += len;
    if( ( ret = mbedtls_asn1_get_tag( &p, end, &len,
                                      MBEDTLS_ASN1_OCTET_STRING ) ) )
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS + ret );
    other_name->value.hardware_module_name.val.tag = MBEDTLS_ASN1_OCTET_STRING;
    other_name->value.hardware_module_name.val.p = p;
    other_name->value.hardware_module_name.val.len = len;
    p += len;
    if( p != end )
    {
        mbedtls_platform_zeroize( other_name,
                                  sizeof( *other_name ) );
        return( MBEDTLS_ERR_X509_INVALID_EXTENSIONS +
                MBEDTLS_ERR_ASN1_LENGTH_MISMATCH );
    }
    return 0;
}

static int x509_info_subject_alt_name( char **buf, size_t *size,
                                       const mbedtls_x509_sequence
                                                    *subject_alt_name,
                                       const char *prefix )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t n = *size;
    char *p = *buf;
    const mbedtls_x509_sequence *cur = subject_alt_name;
    mbedtls_x509_subject_alternative_name san;
    int parse_ret;
    while( cur != NULL )
    {
        mbedtls_platform_zeroize( &san, sizeof( san ) );
        parse_ret = mbedtls_x509_parse_subject_alt_name( &cur->buf, &san );
        if( parse_ret )
        {
            if( parse_ret == MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE )
            {
                ret = mbedtls_snprintf( p, n, "\n%s    <unsupported>", prefix );
                MBEDTLS_X509_SAFE_SNPRINTF;
            }
            else
            {
                ret = mbedtls_snprintf( p, n, "\n%s    <malformed>", prefix );
                MBEDTLS_X509_SAFE_SNPRINTF;
            }
            cur = cur->next;
            continue;
        }
        switch( san.type )
        {
            /*
             * otherName
             */
            case MBEDTLS_X509_SAN_OTHER_NAME:
            {
                mbedtls_x509_san_other_name *other_name = &san.san.other_name;
                ret = mbedtls_snprintf( p, n, "\n%s    otherName :", prefix );
                MBEDTLS_X509_SAFE_SNPRINTF;
                if( MBEDTLS_OID_CMP( MBEDTLS_OID_ON_HW_MODULE_NAME,
                                     &other_name->value.hardware_module_name.oid ) )
                {
                    ret = mbedtls_snprintf( p, n, "\n%s        hardware module name :", prefix );
                    MBEDTLS_X509_SAFE_SNPRINTF;
                    ret = mbedtls_snprintf( p, n, "\n%s            hardware type          : ", prefix );
                    MBEDTLS_X509_SAFE_SNPRINTF;
                    ret = mbedtls_oid_get_numeric_string( p, n, &other_name->value.hardware_module_name.oid );
                    MBEDTLS_X509_SAFE_SNPRINTF;
                    ret = mbedtls_snprintf( p, n, "\n%s            hardware serial number : ", prefix );
                    MBEDTLS_X509_SAFE_SNPRINTF;
                    if( other_name->value.hardware_module_name.val.len >= n )
                    {
                        *p = '\0';
                        return( MBEDTLS_ERR_X509_BUFFER_TOO_SMALL );
                    }
                    memcpy( p, other_name->value.hardware_module_name.val.p,
                            other_name->value.hardware_module_name.val.len );
                    p += other_name->value.hardware_module_name.val.len;
                    n -= other_name->value.hardware_module_name.val.len;
                }/* MBEDTLS_OID_ON_HW_MODULE_NAME */
            }
            break;
            /*
             * dNSName
             */
            case MBEDTLS_X509_SAN_DNS_NAME: {
                ret = mbedtls_snprintf( p, n, "\n%s    dNSName : ", prefix );
                MBEDTLS_X509_SAFE_SNPRINTF;
                if( san.san.unstructured_name.len >= n ) {
                    *p = '\0';
                    return( MBEDTLS_ERR_X509_BUFFER_TOO_SMALL );
                }
                memcpy( p, san.san.unstructured_name.p, san.san.unstructured_name.len );
                p += san.san.unstructured_name.len;
                n -= san.san.unstructured_name.len;
            }
            break;
            /*
             * rfc822Name
             */
            case MBEDTLS_X509_SAN_RFC822_NAME: {
                ret = mbedtls_snprintf( p, n, "\n%s    rfc822Name : ", prefix );
                MBEDTLS_X509_SAFE_SNPRINTF;
                if( san.san.unstructured_name.len >= n ) {
                    *p = '\0';
                    return( MBEDTLS_ERR_X509_BUFFER_TOO_SMALL );
                }
                memcpy( p, san.san.unstructured_name.p, san.san.unstructured_name.len );
                p += san.san.unstructured_name.len;
                n -= san.san.unstructured_name.len;
            }
            break;
            /*
             * uniformResourceIdentifier
             */
            case MBEDTLS_X509_SAN_UNIFORM_RESOURCE_IDENTIFIER: {
                ret = mbedtls_snprintf( p, n, "\n%s    uniformResourceIdentifier : ", prefix );
                MBEDTLS_X509_SAFE_SNPRINTF;
                if( san.san.unstructured_name.len >= n ) {
                    *p = '\0';
                    return( MBEDTLS_ERR_X509_BUFFER_TOO_SMALL );
                }
                memcpy( p, san.san.unstructured_name.p, san.san.unstructured_name.len );
                p += san.san.unstructured_name.len;
                n -= san.san.unstructured_name.len;
            }
            break;
            /*
             * iPAddress
             */
            case MBEDTLS_X509_SAN_IP_ADDRESS:
            {
                ret = mbedtls_snprintf( p, n, "\n%s    iPAddress : %hhu.%hhu.%hhu.%hhu",
                                        prefix, san.san.ip>>24,
                                        san.san.ip>>16,
                                        san.san.ip>>8,
                                        san.san.ip);
                MBEDTLS_X509_SAFE_SNPRINTF;
            }
            break;
            /*
             * Type not supported, skip item.
             */
            default:
                ret = mbedtls_snprintf( p, n, "\n%s    <unsupported:%d>", prefix, san.type );
                MBEDTLS_X509_SAFE_SNPRINTF;
                break;
        }
        cur = cur->next;
    }
    *p = '\0';
    *size = n;
    *buf = p;
    return 0;
}

/**
 * \brief          This function parses an item in the SubjectAlternativeNames
 *                 extension.
 *
 * \param san_buf  The buffer holding the raw data item of the subject
 *                 alternative name.
 * \param san      The target structure to populate with the parsed presentation
 *                 of the subject alternative name encoded in \p san_raw.
 *
 * \note           Only "dnsName" and "otherName" of type hardware_module_name
 *                 as defined in RFC 4180 is supported.
 *
 * \note           This function should be called on a single raw data of
 *                 subject alternative name. For example, after successful
 *                 certificate parsing, one must iterate on every item in the
 *                 \p crt->subject_alt_names sequence, and pass it to
 *                 this function.
 *
 * \warning        The target structure contains pointers to the raw data of the
 *                 parsed certificate, and its lifetime is restricted by the
 *                 lifetime of the certificate.
 *
 * \return         \c 0 on success
 * \return         #MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE for an unsupported
 *                 SAN type.
 * \return         Another negative value for any other failure.
 */
int mbedtls_x509_parse_subject_alt_name( const mbedtls_x509_buf *san_buf,
                                         mbedtls_x509_subject_alternative_name *san )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    switch( san_buf->tag &
            ( MBEDTLS_ASN1_TAG_CLASS_MASK |
              MBEDTLS_ASN1_TAG_VALUE_MASK ) )
    {
        /*
         * otherName
         */
        case( MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_X509_SAN_OTHER_NAME ):
        {
            mbedtls_x509_san_other_name other_name;
            ret = x509_get_other_name( san_buf, &other_name );
            if( ret )
                return ret;
            mbedtls_platform_zeroize( san, sizeof( mbedtls_x509_subject_alternative_name ) );
            san->type = MBEDTLS_X509_SAN_OTHER_NAME;
            memcpy( &san->san.other_name,
                    &other_name, sizeof( other_name ) );
        }
        break;
        /*
         * dNSName
         */
        case( MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_X509_SAN_DNS_NAME ):
        {
            mbedtls_platform_zeroize( san, sizeof( mbedtls_x509_subject_alternative_name ) );
            san->type = MBEDTLS_X509_SAN_DNS_NAME;
            memcpy( &san->san.unstructured_name, san_buf, sizeof( *san_buf ) );
        }
        break;
        /*
         * uniformResourceIdentifier
         */
        case( MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_X509_SAN_UNIFORM_RESOURCE_IDENTIFIER ):
        {
            mbedtls_platform_zeroize( san, sizeof( mbedtls_x509_subject_alternative_name ) );
            san->type = MBEDTLS_X509_SAN_UNIFORM_RESOURCE_IDENTIFIER;
            memcpy( &san->san.unstructured_name, san_buf, sizeof( *san_buf ) );
        }
        break;
        /*
         * rfc822Name
         */
        case( MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_X509_SAN_RFC822_NAME ):
        {
            mbedtls_platform_zeroize( san, sizeof( mbedtls_x509_subject_alternative_name ) );
            san->type = MBEDTLS_X509_SAN_RFC822_NAME;
            memcpy( &san->san.unstructured_name, san_buf, sizeof( *san_buf ) );
        }
        break;
        /*
         * iPAddress
         */
        case( MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_X509_SAN_IP_ADDRESS ):
        {
            mbedtls_platform_zeroize( san, sizeof( mbedtls_x509_subject_alternative_name ) );
            san->type = MBEDTLS_X509_SAN_IP_ADDRESS;
            san->san.ip = READ32BE(san_buf->p);
        }
        break;
        /*
         * Type not supported
         */
        default:
            return( MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE );
    }
    return 0;
}

#define PRINT_ITEM(i)                           \
    {                                           \
        ret = mbedtls_snprintf( p, n, "%s" i, sep );    \
        MBEDTLS_X509_SAFE_SNPRINTF;                        \
        sep = ", ";                             \
    }

#define CERT_TYPE(type,name)                    \
    if( ns_cert_type & (type) )                 \
        PRINT_ITEM( name );

static int x509_info_cert_type( char **buf, size_t *size,
                                unsigned char ns_cert_type )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t n = *size;
    char *p = *buf;
    const char *sep = "";
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_SSL_CLIENT,         "SSL Client" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_SSL_SERVER,         "SSL Server" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_EMAIL,              "Email" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_OBJECT_SIGNING,     "Object Signing" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_RESERVED,           "Reserved" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_SSL_CA,             "SSL CA" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_EMAIL_CA,           "Email CA" );
    CERT_TYPE( MBEDTLS_X509_NS_CERT_TYPE_OBJECT_SIGNING_CA,  "Object Signing CA" );
    *size = n;
    *buf = p;
    return 0;
}

#define KEY_USAGE(code,name)    \
    if( key_usage & (code) )    \
        PRINT_ITEM( name );

static int x509_info_key_usage( char **buf, size_t *size,
                                unsigned int key_usage )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t n = *size;
    char *p = *buf;
    const char *sep = "";
    KEY_USAGE( MBEDTLS_X509_KU_DIGITAL_SIGNATURE,    "Digital Signature" );
    KEY_USAGE( MBEDTLS_X509_KU_NON_REPUDIATION,      "Non Repudiation" );
    KEY_USAGE( MBEDTLS_X509_KU_KEY_ENCIPHERMENT,     "Key Encipherment" );
    KEY_USAGE( MBEDTLS_X509_KU_DATA_ENCIPHERMENT,    "Data Encipherment" );
    KEY_USAGE( MBEDTLS_X509_KU_KEY_AGREEMENT,        "Key Agreement" );
    KEY_USAGE( MBEDTLS_X509_KU_KEY_CERT_SIGN,        "Key Cert Sign" );
    KEY_USAGE( MBEDTLS_X509_KU_CRL_SIGN,             "CRL Sign" );
    KEY_USAGE( MBEDTLS_X509_KU_ENCIPHER_ONLY,        "Encipher Only" );
    KEY_USAGE( MBEDTLS_X509_KU_DECIPHER_ONLY,        "Decipher Only" );
    *size = n;
    *buf = p;
    return 0;
}

static int x509_info_ext_key_usage( char **buf, size_t *size,
                                    const mbedtls_x509_sequence *extended_key_usage )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    const char *desc;
    size_t n = *size;
    char *p = *buf;
    char tmp[48];
    const mbedtls_x509_sequence *cur = extended_key_usage;
    const char *sep = "";
    while( cur )
    {
        if( mbedtls_oid_get_extended_key_usage( &cur->buf, &desc ) )
        {
            mbedtls_oid_get_numeric_string(tmp, sizeof(tmp), &cur->buf);
            desc = tmp;
        }
        ret = mbedtls_snprintf( p, n, "%s%s", sep, desc );
        MBEDTLS_X509_SAFE_SNPRINTF;
        sep = ", ";
        cur = cur->next;
    }
    *size = n;
    *buf = p;
    return 0;
}

static int x509_info_cert_policies( char **buf, size_t *size,
                                    const mbedtls_x509_sequence *certificate_policies )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    const char *desc;
    size_t n = *size;
    char *p = *buf;
    char tmp[48];
    const mbedtls_x509_sequence *cur = certificate_policies;
    const char *sep = "";
    while( cur ) {
        if( mbedtls_oid_get_certificate_policies( &cur->buf, &desc ) )
        {
            mbedtls_oid_get_numeric_string(tmp, sizeof(tmp), &cur->buf);
            desc = tmp;
        }
        ret = mbedtls_snprintf( p, n, "%s%s", sep, desc );
        MBEDTLS_X509_SAFE_SNPRINTF;
        sep = ", ";
        cur = cur->next;
    }
    *size = n;
    *buf = p;
    return 0;
}

#define BEFORE_COLON    18
#define BC              "18"

/**
 * \brief          Returns an informational string about the
 *                 certificate.
 *
 * \param buf      Buffer to write to
 * \param size     Maximum size of buffer
 * \param prefix   A line prefix
 * \param crt      The X509 certificate to represent
 *
 * \return         The length of the string written (not including the
 *                 terminated nul byte), or a negative error code.
 */
int mbedtls_x509_crt_info( char *buf, size_t size, const char *prefix,
                           const mbedtls_x509_crt *crt )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t n;
    char *p;
    char key_size_str[BEFORE_COLON];
    p = buf;
    n = size;
    if( !crt ) {
        ret = mbedtls_snprintf( p, n, "\nCertificate is uninitialised!\n" );
        MBEDTLS_X509_SAFE_SNPRINTF;
        return( (int) ( size - n ) );
    }
    ret = mbedtls_snprintf( p, n, "%scert. version     : %d\n",
                            prefix, crt->version );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_snprintf( p, n, "%sserial number     : ",
                               prefix );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_x509_serial_gets( p, n, &crt->serial );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_snprintf( p, n, "\n%sissuer name       : ", prefix );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_x509_dn_gets( p, n, &crt->issuer  );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_snprintf( p, n, "\n%ssubject name      : ", prefix );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_x509_dn_gets( p, n, &crt->subject );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_snprintf( p, n, "\n%sissued  on        : " \
                   "%04d-%02d-%02d %02d:%02d:%02d", prefix,
                   crt->valid_from.year, crt->valid_from.mon,
                   crt->valid_from.day,  crt->valid_from.hour,
                   crt->valid_from.min,  crt->valid_from.sec );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_snprintf( p, n, "\n%sexpires on        : " \
                   "%04d-%02d-%02d %02d:%02d:%02d", prefix,
                   crt->valid_to.year, crt->valid_to.mon,
                   crt->valid_to.day,  crt->valid_to.hour,
                   crt->valid_to.min,  crt->valid_to.sec );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_snprintf( p, n, "\n%ssigned using      : ", prefix );
    MBEDTLS_X509_SAFE_SNPRINTF;
    ret = mbedtls_x509_sig_alg_gets( p, n, &crt->sig_oid, crt->sig_pk,
                             crt->sig_md, crt->sig_opts );
    MBEDTLS_X509_SAFE_SNPRINTF;
    /* Key size */
    if( ( ret = mbedtls_x509_key_size_helper( key_size_str, BEFORE_COLON,
                                              mbedtls_pk_get_name( &crt->pk ) ) ) )
    {
        return ret;
    }
    ret = mbedtls_snprintf( p, n, "\n%s%-" BC "s: %d bits", prefix, key_size_str,
                          (int) mbedtls_pk_get_bitlen( &crt->pk ) );
    MBEDTLS_X509_SAFE_SNPRINTF;
    /*
     * Optional extensions
     */
    if( crt->ext_types & MBEDTLS_X509_EXT_BASIC_CONSTRAINTS )
    {
        ret = mbedtls_snprintf( p, n, "\n%sbasic constraints : CA=%s", prefix,
                        crt->ca_istrue ? "true" : "false" );
        MBEDTLS_X509_SAFE_SNPRINTF;
        if( crt->max_pathlen > 0 )
        {
            ret = mbedtls_snprintf( p, n, ", max_pathlen=%d", crt->max_pathlen - 1 );
            MBEDTLS_X509_SAFE_SNPRINTF;
        }
    }
    if( crt->ext_types & MBEDTLS_X509_EXT_SUBJECT_ALT_NAME )
    {
        ret = mbedtls_snprintf( p, n, "\n%ssubject alt name  :", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;
        if( ( ret = x509_info_subject_alt_name( &p, &n,
                                                &crt->subject_alt_names,
                                                prefix ) ) )
            return ret;
    }
    if( crt->ext_types & MBEDTLS_X509_EXT_NS_CERT_TYPE )
    {
        ret = mbedtls_snprintf( p, n, "\n%scert. type        : ", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;
        if( ( ret = x509_info_cert_type( &p, &n, crt->ns_cert_type ) ) )
            return ret;
    }
    if( crt->ext_types & MBEDTLS_X509_EXT_KEY_USAGE )
    {
        ret = mbedtls_snprintf( p, n, "\n%skey usage         : ", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;
        if( ( ret = x509_info_key_usage( &p, &n, crt->key_usage ) ) )
            return ret;
    }
    if( crt->ext_types & MBEDTLS_X509_EXT_EXTENDED_KEY_USAGE )
    {
        ret = mbedtls_snprintf( p, n, "\n%sext key usage     : ", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;
        if( ( ret = x509_info_ext_key_usage( &p, &n,
                                             &crt->ext_key_usage ) ) )
            return ret;
    }
    if( crt->ext_types & MBEDTLS_OID_X509_EXT_CERTIFICATE_POLICIES )
    {
        ret = mbedtls_snprintf( p, n, "\n%scert policies     : ", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;
        if( ( ret = x509_info_cert_policies( &p, &n,
                                             &crt->certificate_policies ) ) )
            return ret;
    }
    ret = mbedtls_snprintf( p, n, "\n" );
    MBEDTLS_X509_SAFE_SNPRINTF;
    return( (int) ( size - n ) );
}

struct x509_crt_verify_string {
    int code;
    const char *string;
};

static const struct x509_crt_verify_string x509_crt_verify_strings[] = {
    { MBEDTLS_X509_BADCERT_EXPIRED,       "The certificate validity has expired" },
    { MBEDTLS_X509_BADCERT_REVOKED,       "The certificate has been revoked (is on a CRL)" },
    { MBEDTLS_X509_BADCERT_CN_MISMATCH,   "The certificate Common Name (CN) does not match with the expected CN" },
    { MBEDTLS_X509_BADCERT_NOT_TRUSTED,   "The certificate is not correctly signed by the trusted CA" },
    { MBEDTLS_X509_BADCRL_NOT_TRUSTED,    "The CRL is not correctly signed by the trusted CA" },
    { MBEDTLS_X509_BADCRL_EXPIRED,        "The CRL is expired" },
    { MBEDTLS_X509_BADCERT_MISSING,       "Certificate was missing" },
    { MBEDTLS_X509_BADCERT_SKIP_VERIFY,   "Certificate verification was skipped" },
    { MBEDTLS_X509_BADCERT_OTHER,         "Other reason (can be used by verify callback)" },
    { MBEDTLS_X509_BADCERT_FUTURE,        "The certificate validity starts in the future" },
    { MBEDTLS_X509_BADCRL_FUTURE,         "The CRL is from the future" },
    { MBEDTLS_X509_BADCERT_KEY_USAGE,     "Usage does not match the keyUsage extension" },
    { MBEDTLS_X509_BADCERT_EXT_KEY_USAGE, "Usage does not match the extendedKeyUsage extension" },
    { MBEDTLS_X509_BADCERT_NS_CERT_TYPE,  "Usage does not match the nsCertType extension" },
    { MBEDTLS_X509_BADCERT_BAD_MD,        "The certificate is signed with an unacceptable hash." },
    { MBEDTLS_X509_BADCERT_BAD_PK,        "The certificate is signed with an unacceptable PK alg (eg RSA vs ECDSA)." },
    { MBEDTLS_X509_BADCERT_BAD_KEY,       "The certificate is signed with an unacceptable key (eg bad curve, RSA too short)." },
    { MBEDTLS_X509_BADCRL_BAD_MD,         "The CRL is signed with an unacceptable hash." },
    { MBEDTLS_X509_BADCRL_BAD_PK,         "The CRL is signed with an unacceptable PK alg (eg RSA vs ECDSA)." },
    { MBEDTLS_X509_BADCRL_BAD_KEY,        "The CRL is signed with an unacceptable key (eg bad curve, RSA too short)." },
    { 0, NULL }
};

/**
 * \brief          Returns an informational string about the
 *                 verification status of a certificate.
 *
 * \param buf      Buffer to write to
 * \param size     Maximum size of buffer
 * \param prefix   A line prefix
 * \param flags    Verification flags created by mbedtls_x509_crt_verify()
 *
 * \return         The length of the string written (not including the
 *                 terminated nul byte), or a negative error code.
 */
int mbedtls_x509_crt_verify_info( char *buf, size_t size, const char *prefix,
                                  uint32_t flags )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    const struct x509_crt_verify_string *cur;
    char *p = buf;
    size_t n = size;
    for( cur = x509_crt_verify_strings; cur->string != NULL ; cur++ )
    {
        if( ( flags & cur->code ) == 0 )
            continue;
        ret = mbedtls_snprintf( p, n, "%s%s\n", prefix, cur->string );
        MBEDTLS_X509_SAFE_SNPRINTF;
        flags ^= cur->code;
    }
    if( flags )
    {
        ret = mbedtls_snprintf( p, n, "%sUnknown reason "
                                       "(this should not happen)\n", prefix );
        MBEDTLS_X509_SAFE_SNPRINTF;
    }
    return( (int) ( size - n ) );
}

/**
 * \brief          Check usage of certificate against keyUsage extension.
 *
 * \param crt      Leaf certificate used.
 * \param usage    Intended usage(s) (eg MBEDTLS_X509_KU_KEY_ENCIPHERMENT
 *                 before using the certificate to perform an RSA key
 *                 exchange).
 *
 * \note           Except for decipherOnly and encipherOnly, a bit set in the
 *                 usage argument means this bit MUST be set in the
 *                 certificate. For decipherOnly and encipherOnly, it means
 *                 that bit MAY be set.
 *
 * \return         0 is these uses of the certificate are allowed,
 *                 MBEDTLS_ERR_X509_BAD_INPUT_DATA if the keyUsage extension
 *                 is present but does not match the usage argument.
 *
 * \note           You should only call this function on leaf certificates, on
 *                 (intermediate) CAs the keyUsage extension is automatically
 *                 checked by \c mbedtls_x509_crt_verify().
 */
int mbedtls_x509_crt_check_key_usage( const mbedtls_x509_crt *crt,
                                      unsigned int usage )
{
    unsigned int usage_must, usage_may;
    unsigned int may_mask = MBEDTLS_X509_KU_ENCIPHER_ONLY
                          | MBEDTLS_X509_KU_DECIPHER_ONLY;
    if( ( crt->ext_types & MBEDTLS_X509_EXT_KEY_USAGE ) == 0 )
        return 0;
    usage_must = usage & ~may_mask;
    if( ( ( crt->key_usage & ~may_mask ) & usage_must ) != usage_must )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
    usage_may = usage & may_mask;
    if( ( ( crt->key_usage & may_mask ) | usage_may ) != usage_may )
        return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
    return 0;
}

/**
 * \brief           Check usage of certificate against extendedKeyUsage.
 *
 * \param crt       Leaf certificate used.
 * \param usage_oid Intended usage (eg MBEDTLS_OID_SERVER_AUTH or
 *                  MBEDTLS_OID_CLIENT_AUTH).
 * \param usage_len Length of usage_oid (eg given by MBEDTLS_OID_SIZE()).
 *
 * \return          0 if this use of the certificate is allowed,
 *                  MBEDTLS_ERR_X509_BAD_INPUT_DATA if not.
 *
 * \note            Usually only makes sense on leaf certificates.
 */
int mbedtls_x509_crt_check_extended_key_usage( const mbedtls_x509_crt *crt,
                                               const char *usage_oid,
                                               size_t usage_len )
{
    const mbedtls_x509_sequence *cur;
    /* Extension is not mandatory, absent means no restriction */
    if( ( crt->ext_types & MBEDTLS_X509_EXT_EXTENDED_KEY_USAGE ) == 0 )
        return 0;
    /*
     * Look for the requested usage (or wildcard ANY) in our list
     */
    for( cur = &crt->ext_key_usage; cur != NULL; cur = cur->next )
    {
        const mbedtls_x509_buf *cur_oid = &cur->buf;
        if( cur_oid->len == usage_len &&
            timingsafe_bcmp( cur_oid->p, usage_oid, usage_len ) == 0 )
        {
            return 0;
        }
        if( MBEDTLS_OID_CMP( MBEDTLS_OID_ANY_EXTENDED_KEY_USAGE, cur_oid ) == 0 )
            return 0;
    }
    return( MBEDTLS_ERR_X509_BAD_INPUT_DATA );
}

/**
 * \brief          Verify the certificate revocation status
 *
 * \param crt      a certificate to be verified
 * \param crl      the CRL to verify against
 *
 * \return         1 if the certificate is revoked, 0 otherwise
 *
 */
int mbedtls_x509_crt_is_revoked( const mbedtls_x509_crt *crt, const mbedtls_x509_crl *crl )
{
    const mbedtls_x509_crl_entry *cur = &crl->entry;
    while( cur && cur->serial.len )
    {
        if( crt->serial.len == cur->serial.len &&
            timingsafe_bcmp( crt->serial.p, cur->serial.p, crt->serial.len ) == 0 )
        {
            return( 1 );
        }
        cur = cur->next;
    }
    return 0;
}

/*
 * Check that the given certificate is not revoked according to the CRL.
 * Skip validation if no CRL for the given CA is present.
 */
static int x509_crt_verifycrl( mbedtls_x509_crt *crt, mbedtls_x509_crt *ca,
                               mbedtls_x509_crl *crl_list,
                               const mbedtls_x509_crt_profile *profile )
{
    int flags = 0;
    unsigned char hash[MBEDTLS_MD_MAX_SIZE];
    const mbedtls_md_info_t *md_info;
    if( ca == NULL )
        return( flags );
    while( crl_list )
    {
        if( crl_list->version == 0 ||
            mbedtls_x509_name_cmp( &crl_list->issuer, &ca->subject ) )
        {
            crl_list = crl_list->next;
            continue;
        }
        /*
         * Check if the CA is configured to sign CRLs
         */
#if defined(MBEDTLS_X509_CHECK_KEY_USAGE)
        if( mbedtls_x509_crt_check_key_usage( ca,
                                              MBEDTLS_X509_KU_CRL_SIGN ) )
        {
            flags |= MBEDTLS_X509_BADCRL_NOT_TRUSTED;
            break;
        }
#endif
        /*
         * Check if CRL is correctly signed by the trusted CA
         */
        if( x509_profile_check_md_alg( profile, crl_list->sig_md ) )
            flags |= MBEDTLS_X509_BADCRL_BAD_MD;
        if( x509_profile_check_pk_alg( profile, crl_list->sig_pk ) )
            flags |= MBEDTLS_X509_BADCRL_BAD_PK;
        md_info = mbedtls_md_info_from_type( crl_list->sig_md );
        if( mbedtls_md( md_info, crl_list->tbs.p, crl_list->tbs.len, hash ) )
        {
            /* Note: this can't happen except after an internal error */
            flags |= MBEDTLS_X509_BADCRL_NOT_TRUSTED;
            break;
        }
        if( x509_profile_check_key( profile, &ca->pk ) )
            flags |= MBEDTLS_X509_BADCERT_BAD_KEY;
        if( mbedtls_pk_verify_ext( crl_list->sig_pk, crl_list->sig_opts, &ca->pk,
                           crl_list->sig_md, hash, mbedtls_md_get_size( md_info ),
                           crl_list->sig.p, crl_list->sig.len ) )
        {
            flags |= MBEDTLS_X509_BADCRL_NOT_TRUSTED;
            break;
        }
        /*
         * Check for validity of CRL (Do not drop out)
         */
        if( mbedtls_x509_time_is_past( &crl_list->next_update ) )
            flags |= MBEDTLS_X509_BADCRL_EXPIRED;
        if( mbedtls_x509_time_is_future( &crl_list->this_update ) )
            flags |= MBEDTLS_X509_BADCRL_FUTURE;
        /*
         * Check if certificate is revoked
         */
        if( mbedtls_x509_crt_is_revoked( crt, crl_list ) )
        {
            flags |= MBEDTLS_X509_BADCERT_REVOKED;
            break;
        }
        crl_list = crl_list->next;
    }
    return( flags );
}

/*
 * Check the signature of a certificate by its parent
 */
int mbedtls_x509_crt_check_signature( const mbedtls_x509_crt *child,
                                      mbedtls_x509_crt *parent,
                                      mbedtls_x509_crt_restart_ctx *rs_ctx )
{
    unsigned char hash[MBEDTLS_MD_MAX_SIZE];
    size_t hash_len;
    const mbedtls_md_info_t *md_info;
    md_info = mbedtls_md_info_from_type( child->sig_md );
    hash_len = mbedtls_md_get_size( md_info );
    /* Note: hash errors can happen only after an internal error */
    if( mbedtls_md( md_info, child->tbs.p, child->tbs.len, hash ) != 0 )
        return( -1 );
    /* Skip expensive computation on obvious mismatch */
    if( ! mbedtls_pk_can_do( &parent->pk, child->sig_pk ) )
        return -1;
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
    if( rs_ctx && child->sig_pk == MBEDTLS_PK_ECDSA )
    {
        return( mbedtls_pk_verify_restartable( &parent->pk,
                    child->sig_md, hash, hash_len,
                    child->sig.p, child->sig.len, &rs_ctx->pk ) );
    }
#else
    (void) rs_ctx;
#endif
    return( mbedtls_pk_verify_ext( child->sig_pk, child->sig_opts, &parent->pk,
                child->sig_md, hash, hash_len,
                child->sig.p, child->sig.len ) );
}

/*
 * Checks if 'parent' is a suitable parent (signing CA) for 'child'.
 * Return 0 if yes, -1 if not.
 *
 * top means parent is a locally-trusted certificate
 */
int mbedtls_x509_crt_check_parent( const mbedtls_x509_crt *child,
                                   const mbedtls_x509_crt *parent,
                                   int top )
{
    int need_ca_bit;
    /* Parent must be the issuer */
    if( mbedtls_x509_name_cmp( &child->issuer, &parent->subject ) )
        return -1;
    /* Parent must have the basicConstraints CA bit set as a general rule */
    need_ca_bit = 1;
    /* Exception: v1/v2 certificates that are locally trusted. */
    if( top && parent->version < 3 )
        need_ca_bit = 0;
    if( need_ca_bit && ! parent->ca_istrue )
        return -1;
#if defined(MBEDTLS_X509_CHECK_KEY_USAGE)
    if( need_ca_bit &&
        mbedtls_x509_crt_check_key_usage( parent, MBEDTLS_X509_KU_KEY_CERT_SIGN ) )
    {
        return -1;
    }
#endif
    return 0;
}

/*
 * Find a suitable parent for child in candidates, or return NULL.
 *
 * Here suitable is defined as:
 *  1. subject name matches child's issuer
 *  2. if necessary, the CA bit is set and key usage allows signing certs
 *  3. for trusted roots, the signature is correct
 *     (for intermediates, the signature is checked and the result reported)
 *  4. pathlen constraints are satisfied
 *
 * If there's a suitable candidate which is also time-valid, return the first
 * such. Otherwise, return the first suitable candidate (or NULL if there is
 * none).
 *
 * The rationale for this rule is that someone could have a list of trusted
 * roots with two versions on the same root with different validity periods.
 * (At least one user reported having such a list and wanted it to just work.)
 * The reason we don't just require time-validity is that generally there is
 * only one version, and if it's expired we want the flags to state that
 * rather than NOT_TRUSTED, as would be the case if we required it here.
 *
 * The rationale for rule 3 (signature for trusted roots) is that users might
 * have two versions of the same CA with different keys in their list, and the
 * way we select the correct one is by checking the signature (as we don't
 * rely on key identifier extensions). (This is one way users might choose to
 * handle key rollover, another relies on self-issued certs, see [SIRO].)
 *
 * Arguments:
 *  - [in] child: certificate for which we're looking for a parent
 *  - [in] candidates: chained list of potential parents
 *  - [out] r_parent: parent found (or NULL)
 *  - [out] r_signature_is_good: 1 if child signature by parent is valid, or 0
 *  - [in] top: 1 if candidates consists of trusted roots, ie we're at the top
 *         of the chain, 0 otherwise
 *  - [in] path_cnt: number of intermediates seen so far
 *  - [in] self_cnt: number of self-signed intermediates seen so far
 *         (will never be greater than path_cnt)
 *  - [in-out] rs_ctx: context for restarting operations
 *
 * Return value:
 *  - 0 on success
 *  - MBEDTLS_ERR_ECP_IN_PROGRESS otherwise
 */
static int x509_crt_find_parent_in(
                        mbedtls_x509_crt *child,
                        mbedtls_x509_crt *candidates,
                        mbedtls_x509_crt **r_parent,
                        int *r_signature_is_good,
                        int top,
                        unsigned path_cnt,
                        unsigned self_cnt,
                        mbedtls_x509_crt_restart_ctx *rs_ctx )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_x509_crt *parent, *fallback_parent;
    int signature_is_good = 0, fallback_signature_is_good;
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
    /* did we have something in progress? */
    if( rs_ctx && rs_ctx->parent )
    {
        /* restore saved state */
        parent = rs_ctx->parent;
        fallback_parent = rs_ctx->fallback_parent;
        fallback_signature_is_good = rs_ctx->fallback_signature_is_good;
        /* clear saved state */
        rs_ctx->parent = NULL;
        rs_ctx->fallback_parent = NULL;
        rs_ctx->fallback_signature_is_good = 0;
        /* resume where we left */
        goto check_signature;
    }
#endif
    fallback_parent = NULL;
    fallback_signature_is_good = 0;
    for( parent = candidates; parent; parent = parent->next )
    {
        /* basic parenting skills (name, CA bit, key usage) */
        if( mbedtls_x509_crt_check_parent( child, parent, top ) )
            continue;
        /* +1 because stored max_pathlen is 1 higher that the actual value */
        if( parent->max_pathlen > 0 &&
            (size_t) parent->max_pathlen < 1 + path_cnt - self_cnt )
        {
            continue;
        }
        /* Signature */
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
check_signature:
#endif
        ret = mbedtls_x509_crt_check_signature( child, parent, rs_ctx );
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
        if( rs_ctx && ret == MBEDTLS_ERR_ECP_IN_PROGRESS )
        {
            /* save state */
            rs_ctx->parent = parent;
            rs_ctx->fallback_parent = fallback_parent;
            rs_ctx->fallback_signature_is_good = fallback_signature_is_good;
            return ret;
        }
#else
        (void) ret;
#endif
        signature_is_good = ret == 0;
        if( top && ! signature_is_good )
            continue;
        /* optional time check */
        if( mbedtls_x509_time_is_past( &parent->valid_to ) ||
            mbedtls_x509_time_is_future( &parent->valid_from ) )
        {
            if( fallback_parent == NULL )
            {
                fallback_parent = parent;
                fallback_signature_is_good = signature_is_good;
            }
            continue;
        }
        *r_parent = parent;
        *r_signature_is_good = signature_is_good;
        break;
    }
    if( parent == NULL )
    {
        *r_parent = fallback_parent;
        *r_signature_is_good = fallback_signature_is_good;
    }
    return 0;
}

/*
 * Find a parent in trusted CAs or the provided chain, or return NULL.
 *
 * Searches in trusted CAs first, and return the first suitable parent found
 * (see find_parent_in() for definition of suitable).
 *
 * Arguments:
 *  - [in] child: certificate for which we're looking for a parent, followed
 *         by a chain of possible intermediates
 *  - [in] trust_ca: list of locally trusted certificates
 *  - [out] parent: parent found (or NULL)
 *  - [out] parent_is_trusted: 1 if returned `parent` is trusted, or 0
 *  - [out] signature_is_good: 1 if child signature by parent is valid, or 0
 *  - [in] path_cnt: number of links in the chain so far (EE -> ... -> child)
 *  - [in] self_cnt: number of self-signed certs in the chain so far
 *         (will always be no greater than path_cnt)
 *  - [in-out] rs_ctx: context for restarting operations
 *
 * Return value:
 *  - 0 on success
 *  - MBEDTLS_ERR_ECP_IN_PROGRESS otherwise
 */
static int x509_crt_find_parent(
                        mbedtls_x509_crt *child,
                        mbedtls_x509_crt *trust_ca,
                        mbedtls_x509_crt **parent,
                        int *parent_is_trusted,
                        int *signature_is_good,
                        unsigned path_cnt,
                        unsigned self_cnt,
                        mbedtls_x509_crt_restart_ctx *rs_ctx )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_x509_crt *search_list;
    *parent_is_trusted = 1;
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
    /* restore then clear saved state if we have some stored */
    if( rs_ctx && rs_ctx->parent_is_trusted != -1 )
    {
        *parent_is_trusted = rs_ctx->parent_is_trusted;
        rs_ctx->parent_is_trusted = -1;
    }
#endif
    while( 1 ) {
        search_list = *parent_is_trusted ? trust_ca : child->next;
        ret = x509_crt_find_parent_in( child, search_list,
                                       parent, signature_is_good,
                                       *parent_is_trusted,
                                       path_cnt, self_cnt, rs_ctx );
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
        if( rs_ctx && ret == MBEDTLS_ERR_ECP_IN_PROGRESS )
        {
            /* save state */
            rs_ctx->parent_is_trusted = *parent_is_trusted;
            return ret;
        }
#else
        (void) ret;
#endif
        /* stop here if found or already in second iteration */
        if( *parent || *parent_is_trusted == 0 )
            break;
        /* prepare second iteration */
        *parent_is_trusted = 0;
    }
    /* extra precaution against mistakes in the caller */
    if( *parent == NULL )
    {
        *parent_is_trusted = 0;
        *signature_is_good = 0;
    }
    return 0;
}

/*
 * Check if an end-entity certificate is locally trusted
 *
 * Currently we require such certificates to be self-signed (actually only
 * check for self-issued as self-signatures are not checked)
 */
static int x509_crt_check_ee_locally_trusted(
                    mbedtls_x509_crt *crt,
                    mbedtls_x509_crt *trust_ca )
{
    mbedtls_x509_crt *cur;
    /* must be self-issued */
    if( mbedtls_x509_name_cmp( &crt->issuer, &crt->subject ) )
        return -1;
    /* look for an exact match with trusted cert */
    for( cur = trust_ca; cur; cur = cur->next )
    {
        if( crt->raw.len == cur->raw.len &&
            timingsafe_bcmp( crt->raw.p, cur->raw.p, crt->raw.len ) == 0 )
        {
            return 0;
        }
    }
    /* too bad */
    return -1;
}

/*
 * Build and verify a certificate chain
 *
 * Given a peer-provided list of certificates EE, C1, ..., Cn and
 * a list of trusted certs R1, ... Rp, try to build and verify a chain
 *      EE, Ci1, ... Ciq [, Rj]
 * such that every cert in the chain is a child of the next one,
 * jumping to a trusted root as early as possible.
 *
 * Verify that chain and return it with flags for all issues found.
 *
 * Special cases:
 * - EE == Rj -> return a one-element list containing it
 * - EE, Ci1, ..., Ciq cannot be continued with a trusted root
 *   -> return that chain with NOT_TRUSTED set on Ciq
 *
 * Tests for (aspects of) this function should include at least:
 * - trusted EE
 * - EE -> trusted root
 * - EE -> intermediate CA -> trusted root
 * - if relevant: EE untrusted
 * - if relevant: EE -> intermediate, untrusted
 * with the aspect under test checked at each relevant level (EE, int, root).
 * For some aspects longer chains are required, but usually length 2 is
 * enough (but length 1 is not in general).
 *
 * Arguments:
 *  - [in] crt: the cert list EE, C1, ..., Cn
 *  - [in] trust_ca: the trusted list R1, ..., Rp
 *  - [in] ca_crl, profile: as in verify_with_profile()
 *  - [out] ver_chain: the built and verified chain
 *      Only valid when return value is 0, may contain garbage otherwise!
 *      Restart note: need not be the same when calling again to resume.
 *  - [in-out] rs_ctx: context for restarting operations
 *
 * Return value:
 *  - non-zero if the chain could not be fully built and examined
 *  - 0 is the chain was successfully built and examined,
 *      even if it was found to be invalid
 */
static int x509_crt_verify_chain(
                mbedtls_x509_crt *crt,
                mbedtls_x509_crt *trust_ca,
                mbedtls_x509_crl *ca_crl,
                mbedtls_x509_crt_ca_cb_t f_ca_cb,
                void *p_ca_cb,
                const mbedtls_x509_crt_profile *profile,
                mbedtls_x509_crt_verify_chain *ver_chain,
                mbedtls_x509_crt_restart_ctx *rs_ctx )
{
    /* Don't initialize any of those variables here, so that the compiler can
     * catch potential issues with jumping ahead when restarting */
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    uint32_t *flags;
    mbedtls_x509_crt_verify_chain_item *cur;
    mbedtls_x509_crt *child;
    mbedtls_x509_crt *parent;
    int parent_is_trusted;
    int child_is_trusted;
    int signature_is_good;
    unsigned self_cnt;
    mbedtls_x509_crt *cur_trust_ca = NULL;
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
    /* resume if we had an operation in progress */
    if( rs_ctx && rs_ctx->in_progress == x509_crt_rs_find_parent )
    {
        /* restore saved state */
        *ver_chain = rs_ctx->ver_chain; /* struct copy */
        self_cnt = rs_ctx->self_cnt;
        /* restore derived state */
        cur = &ver_chain->items[ver_chain->len - 1];
        child = cur->crt;
        flags = &cur->flags;
        goto find_parent;
    }
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */
    child = crt;
    self_cnt = 0;
    parent_is_trusted = 0;
    child_is_trusted = 0;
    while( 1 ) {
        /* Add certificate to the verification chain */
        cur = &ver_chain->items[ver_chain->len];
        cur->crt = child;
        cur->flags = 0;
        ver_chain->len++;
        flags = &cur->flags;
        /* Check time-validity (all certificates) */
        if( mbedtls_x509_time_is_past( &child->valid_to ) )
            *flags |= MBEDTLS_X509_BADCERT_EXPIRED;
        if( mbedtls_x509_time_is_future( &child->valid_from ) )
            *flags |= MBEDTLS_X509_BADCERT_FUTURE;
        /* Stop here for trusted roots (but not for trusted EE certs) */
        if( child_is_trusted )
            return 0;
        /* Check signature algorithm: MD & PK algs */
        if( x509_profile_check_md_alg( profile, child->sig_md ) )
            *flags |= MBEDTLS_X509_BADCERT_BAD_MD;
        if( x509_profile_check_pk_alg( profile, child->sig_pk ) )
            *flags |= MBEDTLS_X509_BADCERT_BAD_PK;
        /* Special case: EE certs that are locally trusted */
        if( ver_chain->len == 1 &&
            x509_crt_check_ee_locally_trusted( child, trust_ca ) == 0 )
        {
            return 0;
        }
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
find_parent:
#endif
        /* Obtain list of potential trusted signers from CA callback,
         * or use statically provided list. */
#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
        if( f_ca_cb )
        {
            mbedtls_x509_crt_free( ver_chain->trust_ca_cb_result );
            mbedtls_free( ver_chain->trust_ca_cb_result );
            ver_chain->trust_ca_cb_result = NULL;
            ret = f_ca_cb( p_ca_cb, child, &ver_chain->trust_ca_cb_result );
            if( ret )
                return( MBEDTLS_ERR_X509_FATAL_ERROR );
            cur_trust_ca = ver_chain->trust_ca_cb_result;
        }
        else
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */
        {
            ((void) f_ca_cb);
            ((void) p_ca_cb);
            cur_trust_ca = trust_ca;
        }
        /* Look for a parent in trusted CAs or up the chain */
        ret = x509_crt_find_parent( child, cur_trust_ca, &parent,
                                       &parent_is_trusted, &signature_is_good,
                                       ver_chain->len - 1, self_cnt, rs_ctx );
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
        if( rs_ctx && ret == MBEDTLS_ERR_ECP_IN_PROGRESS )
        {
            /* save state */
            rs_ctx->in_progress = x509_crt_rs_find_parent;
            rs_ctx->self_cnt = self_cnt;
            rs_ctx->ver_chain = *ver_chain; /* struct copy */
            return ret;
        }
#else
        (void) ret;
#endif
        /* No parent? We're done here */
        if( parent == NULL )
        {
            *flags |= MBEDTLS_X509_BADCERT_NOT_TRUSTED;
            return 0;
        }
        /* Count intermediate self-issued (not necessarily self-signed) certs.
         * These can occur with some strategies for key rollover, see [SIRO],
         * and should be excluded from max_pathlen checks. */
        if( ver_chain->len != 1 &&
            mbedtls_x509_name_cmp( &child->issuer, &child->subject ) == 0 )
        {
            self_cnt++;
        }
        /* path_cnt is 0 for the first intermediate CA,
         * and if parent is trusted it's not an intermediate CA */
        if( ! parent_is_trusted &&
            ver_chain->len > MBEDTLS_X509_MAX_INTERMEDIATE_CA )
        {
            /* return immediately to avoid overflow the chain array */
            return( MBEDTLS_ERR_X509_FATAL_ERROR );
        }
        /* signature was checked while searching parent */
        if( ! signature_is_good )
            *flags |= MBEDTLS_X509_BADCERT_NOT_TRUSTED;
        /* check size of signing key */
        if( x509_profile_check_key( profile, &parent->pk ) )
            *flags |= MBEDTLS_X509_BADCERT_BAD_KEY;
#if defined(MBEDTLS_X509_CRL_PARSE_C)
        /* Check trusted CA's CRL for the given crt */
        *flags |= x509_crt_verifycrl( child, parent, ca_crl, profile );
#else
        (void) ca_crl;
#endif
        /* prepare for next iteration */
        child = parent;
        parent = NULL;
        child_is_trusted = parent_is_trusted;
        signature_is_good = 0;
    }
}

/*
 * Check for CN match
 */
static int x509_crt_check_cn( const mbedtls_x509_buf *name,
                              const char *cn, size_t cn_len )
{
    /* try exact match */
    if( name->len == cn_len &&
        memcasecmp( cn, name->p, cn_len ) == 0 )
    {
        return 0;
    }
    /* try wildcard match */
    if( x509_check_wildcard( cn, name ) == 0 )
    {
        return 0;
    }
    return -1;
}

/*
 * Check for SAN match, see RFC 5280 Section 4.2.1.6
 */
static int x509_crt_check_san( const mbedtls_x509_buf *name,
                               const char *cn, size_t cn_len )
{
    int64_t ip;
    const unsigned char san_type = (unsigned char) name->tag &
                                   MBEDTLS_ASN1_TAG_VALUE_MASK;
    /* dNSName */
    if( san_type == MBEDTLS_X509_SAN_DNS_NAME )
        return( x509_crt_check_cn( name, cn, cn_len ) );
    if( san_type == MBEDTLS_X509_SAN_IP_ADDRESS &&
        name->len == 4 && ( ip = ParseIp( cn, cn_len ) ) != -1 &&
        ip == READ32BE( name->p ) ) {
        return( 0 );
    }
    /* (We may handle other types here later.) */
    /* Unrecognized type */
    return -1;
}

/*
 * Verify the requested CN - only call this if cn is not NULL!
 */
static void x509_crt_verify_name( const mbedtls_x509_crt *crt,
                                  const char *cn,
                                  uint32_t *flags )
{
    const mbedtls_x509_name *name;
    const mbedtls_x509_sequence *cur;
    size_t cn_len = strlen( cn );
    if( crt->ext_types & MBEDTLS_X509_EXT_SUBJECT_ALT_NAME )
    {
        for( cur = &crt->subject_alt_names; cur; cur = cur->next )
        {
            if( x509_crt_check_san( &cur->buf, cn, cn_len ) == 0 )
                break;
        }
        if( cur == NULL )
            *flags |= MBEDTLS_X509_BADCERT_CN_MISMATCH;
    }
    else
    {
        for( name = &crt->subject; name; name = name->next )
        {
            if( MBEDTLS_OID_CMP( MBEDTLS_OID_AT_CN, &name->oid ) == 0 &&
                x509_crt_check_cn( &name->val, cn, cn_len ) == 0 )
            {
                break;
            }
        }
        if( name == NULL )
            *flags |= MBEDTLS_X509_BADCERT_CN_MISMATCH;
    }
}

/*
 * Merge the flags for all certs in the chain, after calling callback
 */
static int x509_crt_merge_flags_with_cb(
           uint32_t *flags,
           const mbedtls_x509_crt_verify_chain *ver_chain,
           int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
           void *p_vrfy )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    unsigned i;
    uint32_t cur_flags;
    const mbedtls_x509_crt_verify_chain_item *cur;
    for( i = ver_chain->len; i; --i )
    {
        cur = &ver_chain->items[i-1];
        cur_flags = cur->flags;
        if( NULL != f_vrfy )
            if( ( ret = f_vrfy( p_vrfy, cur->crt, (int) i-1, &cur_flags ) ) )
                return ret;
        *flags |= cur_flags;
    }
    return 0;
}

/*
 * Verify the certificate validity, with profile, restartable version
 *
 * This function:
 *  - checks the requested CN (if any)
 *  - checks the type and size of the EE cert's key,
 *    as that isn't done as part of chain building/verification currently
 *  - builds and verifies the chain
 *  - then calls the callback and merges the flags
 *
 * The parameters pairs `trust_ca`, `ca_crl` and `f_ca_cb`, `p_ca_cb`
 * are mutually exclusive: If `f_ca_cb`, it will be used by the
 * verification routine to search for trusted signers, and CRLs will
 * be disabled. Otherwise, `trust_ca` will be used as the static list
 * of trusted signers, and `ca_crl` will be use as the static list
 * of CRLs.
 */
static int x509_crt_verify_restartable_ca_cb(
    mbedtls_x509_crt *crt,
    mbedtls_x509_crt *trust_ca,
    mbedtls_x509_crl *ca_crl,
    mbedtls_x509_crt_ca_cb_t f_ca_cb,
    void *p_ca_cb,
    const mbedtls_x509_crt_profile *profile,
    const char *cn, uint32_t *flags,
    int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
    void *p_vrfy,
    mbedtls_x509_crt_restart_ctx *rs_ctx )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    mbedtls_pk_type_t pk_type;
    mbedtls_x509_crt_verify_chain ver_chain;
    uint32_t ee_flags;
    *flags = 0;
    ee_flags = 0;
    x509_crt_verify_chain_reset( &ver_chain );
    if( profile == NULL )
    {
        ret = MBEDTLS_ERR_X509_BAD_INPUT_DATA;
        goto exit;
    }
    /* check name if requested */
    if( cn )
        x509_crt_verify_name( crt, cn, &ee_flags );
    /* Check the type and size of the key */
    pk_type = mbedtls_pk_get_type( &crt->pk );
    if( x509_profile_check_pk_alg( profile, pk_type ) )
        ee_flags |= MBEDTLS_X509_BADCERT_BAD_PK;
    if( x509_profile_check_key( profile, &crt->pk ) )
        ee_flags |= MBEDTLS_X509_BADCERT_BAD_KEY;
    /* Check the chain */
    ret = x509_crt_verify_chain( crt, trust_ca, ca_crl,
                                 f_ca_cb, p_ca_cb, profile,
                                 &ver_chain, rs_ctx );
    if( ret )
        goto exit;
    /* Merge end-entity flags */
    ver_chain.items[0].flags |= ee_flags;
    /* Build final flags, calling callback on the way if any */
    ret = x509_crt_merge_flags_with_cb( flags, &ver_chain, f_vrfy, p_vrfy );
exit:
#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
    mbedtls_x509_crt_free( ver_chain.trust_ca_cb_result );
    mbedtls_free( ver_chain.trust_ca_cb_result );
    ver_chain.trust_ca_cb_result = NULL;
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
    if( rs_ctx && ret != MBEDTLS_ERR_ECP_IN_PROGRESS )
        mbedtls_x509_crt_restart_free( rs_ctx );
#endif
    /* prevent misuse of the vrfy callback - VERIFY_FAILED would be ignored by
     * the SSL module for authmode optional, but non-zero return from the
     * callback means a fatal error so it shouldn't be ignored */
    if( ret == MBEDTLS_ERR_X509_CERT_VERIFY_FAILED )
        ret = MBEDTLS_ERR_X509_FATAL_ERROR;
    if( ret ) {
        *flags = (uint32_t) -1;
        return ret;
    }
    if( *flags )
        return( MBEDTLS_ERR_X509_CERT_VERIFY_FAILED );
    return 0;
}

/**
 * \brief          Verify a chain of certificates.
 *
 *                 The verify callback is a user-supplied callback that
 *                 can clear / modify / add flags for a certificate. If set,
 *                 the verification callback is called for each
 *                 certificate in the chain (from the trust-ca down to the
 *                 presented crt). The parameters for the callback are:
 *                 (void *parameter, mbedtls_x509_crt *crt, int certificate_depth,
 *                 int *flags). With the flags representing current flags for
 *                 that specific certificate and the certificate depth from
 *                 the bottom (Peer cert depth = 0).
 *
 *                 All flags left after returning from the callback
 *                 are also returned to the application. The function should
 *                 return 0 for anything (including invalid certificates)
 *                 other than fatal error, as a non-zero return code
 *                 immediately aborts the verification process. For fatal
 *                 errors, a specific error code should be used (different
 *                 from MBEDTLS_ERR_X509_CERT_VERIFY_FAILED which should not
 *                 be returned at this point), or MBEDTLS_ERR_X509_FATAL_ERROR
 *                 can be used if no better code is available.
 *
 * \note           In case verification failed, the results can be displayed
 *                 using \c mbedtls_x509_crt_verify_info()
 *
 * \note           Same as \c mbedtls_x509_crt_verify_with_profile() with the
 *                 default security profile.
 *
 * \note           It is your responsibility to provide up-to-date CRLs for
 *                 all trusted CAs. If no CRL is provided for the CA that was
 *                 used to sign the certificate, CRL verification is skipped
 *                 silently, that is *without* setting any flag.
 *
 * \note           The \c trust_ca list can contain two types of certificates:
 *                 (1) those of trusted root CAs, so that certificates
 *                 chaining up to those CAs will be trusted, and (2)
 *                 self-signed end-entity certificates to be trusted (for
 *                 specific peers you know) - in that case, the self-signed
 *                 certificate doesn't need to have the CA bit set.
 *
 * \param crt      The certificate chain to be verified.
 * \param trust_ca The list of trusted CAs.
 * \param ca_crl   The list of CRLs for trusted CAs.
 * \param cn       The expected Common Name. This will be checked to be
 *                 present in the certificate's subjectAltNames extension or,
 *                 if this extension is absent, as a CN component in its
 *                 Subject name. Currently only DNS names are supported. This
 *                 may be \c NULL if the CN need not be verified.
 * \param flags    The address at which to store the result of the verification.
 *                 If the verification couldn't be completed, the flag value is
 *                 set to (uint32_t) -1.
 * \param f_vrfy   The verification callback to use. See the documentation
 *                 of mbedtls_x509_crt_verify() for more information.
 * \param p_vrfy   The context to be passed to \p f_vrfy.
 *
 * \return         \c 0 if the chain is valid with respect to the
 *                 passed CN, CAs, CRLs and security profile.
 * \return         #MBEDTLS_ERR_X509_CERT_VERIFY_FAILED in case the
 *                 certificate chain verification failed. In this case,
 *                 \c *flags will have one or more
 *                 \c MBEDTLS_X509_BADCERT_XXX or \c MBEDTLS_X509_BADCRL_XXX
 *                 flags set.
 * \return         Another negative error code in case of a fatal error
 *                 encountered during the verification process.
 */
int mbedtls_x509_crt_verify( mbedtls_x509_crt *crt,
                             mbedtls_x509_crt *trust_ca,
                             mbedtls_x509_crl *ca_crl,
                             const char *cn, uint32_t *flags,
                             int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                             void *p_vrfy)
{
    return( x509_crt_verify_restartable_ca_cb( crt, trust_ca, ca_crl,
                                         NULL, NULL,
                                         &mbedtls_x509_crt_profile_default,
                                         cn, flags,
                                         f_vrfy, p_vrfy, NULL ) );
}

/**
 * \brief          Verify a chain of certificates with respect to
 *                 a configurable security profile.
 *
 * \note           Same as \c mbedtls_x509_crt_verify(), but with explicit
 *                 security profile.
 *
 * \note           The restrictions on keys (RSA minimum size, allowed curves
 *                 for ECDSA) apply to all certificates: trusted root,
 *                 intermediate CAs if any, and end entity certificate.
 *
 * \param crt      The certificate chain to be verified.
 * \param trust_ca The list of trusted CAs.
 * \param ca_crl   The list of CRLs for trusted CAs.
 * \param profile  The security profile to use for the verification.
 * \param cn       The expected Common Name. This may be \c NULL if the
 *                 CN need not be verified.
 * \param flags    The address at which to store the result of the verification.
 *                 If the verification couldn't be completed, the flag value is
 *                 set to (uint32_t) -1.
 * \param f_vrfy   The verification callback to use. See the documentation
 *                 of mbedtls_x509_crt_verify() for more information.
 * \param p_vrfy   The context to be passed to \p f_vrfy.
 *
 * \return         \c 0 if the chain is valid with respect to the
 *                 passed CN, CAs, CRLs and security profile.
 * \return         #MBEDTLS_ERR_X509_CERT_VERIFY_FAILED in case the
 *                 certificate chain verification failed. In this case,
 *                 \c *flags will have one or more
 *                 \c MBEDTLS_X509_BADCERT_XXX or \c MBEDTLS_X509_BADCRL_XXX
 *                 flags set.
 * \return         Another negative error code in case of a fatal error
 *                 encountered during the verification process.
 */
int mbedtls_x509_crt_verify_with_profile( mbedtls_x509_crt *crt,
                                          mbedtls_x509_crt *trust_ca,
                                          mbedtls_x509_crl *ca_crl,
                                          const mbedtls_x509_crt_profile *profile,
                                          const char *cn, uint32_t *flags,
                                          int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                                          void *p_vrfy )
{
    return( x509_crt_verify_restartable_ca_cb( crt, trust_ca, ca_crl,
                                                 NULL, NULL,
                                                 profile, cn, flags,
                                                 f_vrfy, p_vrfy, NULL ) );
}

#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
/**
 * \brief          Version of \c mbedtls_x509_crt_verify_with_profile() which
 *                 uses a callback to acquire the list of trusted CA
 *                 certificates.
 *
 * \param crt      The certificate chain to be verified.
 * \param f_ca_cb  The callback to be used to query for potential signers
 *                 of a given child certificate. See the documentation of
 *                 ::mbedtls_x509_crt_ca_cb_t for more information.
 * \param p_ca_cb  The opaque context to be passed to \p f_ca_cb.
 * \param profile  The security profile for the verification.
 * \param cn       The expected Common Name. This may be \c NULL if the
 *                 CN need not be verified.
 * \param flags    The address at which to store the result of the verification.
 *                 If the verification couldn't be completed, the flag value is
 *                 set to (uint32_t) -1.
 * \param f_vrfy   The verification callback to use. See the documentation
 *                 of mbedtls_x509_crt_verify() for more information.
 * \param p_vrfy   The context to be passed to \p f_vrfy.
 *
 * \return         See \c mbedtls_crt_verify_with_profile().
 */
int mbedtls_x509_crt_verify_with_ca_cb( mbedtls_x509_crt *crt,
                     mbedtls_x509_crt_ca_cb_t f_ca_cb,
                     void *p_ca_cb,
                     const mbedtls_x509_crt_profile *profile,
                     const char *cn, uint32_t *flags,
                     int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                     void *p_vrfy )
{
    return( x509_crt_verify_restartable_ca_cb( crt, NULL, NULL,
                                                 f_ca_cb, p_ca_cb,
                                                 profile, cn, flags,
                                                 f_vrfy, p_vrfy, NULL ) );
}
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */

/**
 * \brief          Restartable version of \c mbedtls_crt_verify_with_profile()
 *
 * \note           Performs the same job as \c mbedtls_crt_verify_with_profile()
 *                 but can return early and restart according to the limit
 *                 set with \c mbedtls_ecp_set_max_ops() to reduce blocking.
 *
 * \param crt      The certificate chain to be verified.
 * \param trust_ca The list of trusted CAs.
 * \param ca_crl   The list of CRLs for trusted CAs.
 * \param profile  The security profile to use for the verification.
 * \param cn       The expected Common Name. This may be \c NULL if the
 *                 CN need not be verified.
 * \param flags    The address at which to store the result of the verification.
 *                 If the verification couldn't be completed, the flag value is
 *                 set to (uint32_t) -1.
 * \param f_vrfy   The verification callback to use. See the documentation
 *                 of mbedtls_x509_crt_verify() for more information.
 * \param p_vrfy   The context to be passed to \p f_vrfy.
 * \param rs_ctx   The restart context to use. This may be set to \c NULL
 *                 to disable restartable ECC.
 *
 * \return         See \c mbedtls_crt_verify_with_profile(), or
 * \return         #MBEDTLS_ERR_ECP_IN_PROGRESS if maximum number of
 *                 operations was reached: see \c mbedtls_ecp_set_max_ops().
 */
int mbedtls_x509_crt_verify_restartable( mbedtls_x509_crt *crt,
                     mbedtls_x509_crt *trust_ca,
                     mbedtls_x509_crl *ca_crl,
                     const mbedtls_x509_crt_profile *profile,
                     const char *cn, uint32_t *flags,
                     int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                     void *p_vrfy,
                     mbedtls_x509_crt_restart_ctx *rs_ctx )
{
    return( x509_crt_verify_restartable_ca_cb( crt, trust_ca, ca_crl,
                                                 NULL, NULL,
                                                 profile, cn, flags,
                                                 f_vrfy, p_vrfy, rs_ctx ) );
}


/**
 * \brief          Initialize a certificate (chain)
 *
 * \param crt      Certificate chain to initialize
 */
void mbedtls_x509_crt_init( mbedtls_x509_crt *crt )
{
    mbedtls_platform_zeroize( crt, sizeof(mbedtls_x509_crt) );
}

/**
 * \brief          Unallocate all certificate data
 *
 * \param crt      Certificate chain to free
 */
void mbedtls_x509_crt_free( mbedtls_x509_crt *crt )
{
    mbedtls_x509_crt *cert_cur = crt;
    mbedtls_x509_crt *cert_prv;
    mbedtls_x509_name *name_cur;
    mbedtls_x509_name *name_prv;
    mbedtls_x509_sequence *seq_cur;
    mbedtls_x509_sequence *seq_prv;
    if( crt == NULL )
        return;
    do
    {
        mbedtls_pk_free( &cert_cur->pk );
        name_cur = cert_cur->issuer.next;
        while( name_cur )
        {
            name_prv = name_cur;
            name_cur = name_cur->next;
            mbedtls_platform_zeroize( name_prv, sizeof( mbedtls_x509_name ) );
            mbedtls_free( name_prv );
        }
        name_cur = cert_cur->subject.next;
        while( name_cur )
        {
            name_prv = name_cur;
            name_cur = name_cur->next;
            mbedtls_platform_zeroize( name_prv, sizeof( mbedtls_x509_name ) );
            mbedtls_free( name_prv );
        }
        seq_cur = cert_cur->ext_key_usage.next;
        while( seq_cur )
        {
            seq_prv = seq_cur;
            seq_cur = seq_cur->next;
            mbedtls_platform_zeroize( seq_prv,
                                      sizeof( mbedtls_x509_sequence ) );
            mbedtls_free( seq_prv );
        }
        seq_cur = cert_cur->subject_alt_names.next;
        while( seq_cur )
        {
            seq_prv = seq_cur;
            seq_cur = seq_cur->next;
            mbedtls_platform_zeroize( seq_prv,
                                      sizeof( mbedtls_x509_sequence ) );
            mbedtls_free( seq_prv );
        }
        seq_cur = cert_cur->certificate_policies.next;
        while( seq_cur )
        {
            seq_prv = seq_cur;
            seq_cur = seq_cur->next;
            mbedtls_platform_zeroize( seq_prv,
                                      sizeof( mbedtls_x509_sequence ) );
            mbedtls_free( seq_prv );
        }
        if( cert_cur->raw.p && cert_cur->own_buffer )
        {
            mbedtls_platform_zeroize( cert_cur->raw.p, cert_cur->raw.len );
            mbedtls_free( cert_cur->raw.p );
        }
        cert_cur = cert_cur->next;
    }
    while( cert_cur );
    cert_cur = crt;
    do
    {
        cert_prv = cert_cur;
        cert_cur = cert_cur->next;
        mbedtls_platform_zeroize( cert_prv, sizeof( mbedtls_x509_crt ) );
        if( cert_prv != crt )
            mbedtls_free( cert_prv );
    }
    while( cert_cur );
}

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
/**
 * \brief           Initialize a restart context
 */
void mbedtls_x509_crt_restart_init( mbedtls_x509_crt_restart_ctx *ctx )
{
    mbedtls_pk_restart_init( &ctx->pk );
    ctx->parent = NULL;
    ctx->fallback_parent = NULL;
    ctx->fallback_signature_is_good = 0;
    ctx->parent_is_trusted = -1;
    ctx->in_progress = x509_crt_rs_none;
    ctx->self_cnt = 0;
    x509_crt_verify_chain_reset( &ctx->ver_chain );
}

/**
 * \brief           Free the components of a restart context
 */
void mbedtls_x509_crt_restart_free( mbedtls_x509_crt_restart_ctx *ctx ) {
    if( ctx == NULL )
        return;
    mbedtls_pk_restart_free( &ctx->pk );
    mbedtls_x509_crt_restart_init( ctx );
}
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

#endif /* MBEDTLS_X509_CRT_PARSE_C */
