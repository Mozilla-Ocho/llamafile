#ifndef MBEDTLS_X509_H_
#define MBEDTLS_X509_H_
#include "third_party/mbedtls/asn1.h"
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/pk.h"
#include "third_party/mbedtls/rsa.h"
COSMOPOLITAN_C_START_
 
#define MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE              -0x2080  /*< Unavailable feature, e.g. RSA hashing/encryption combination. */
#define MBEDTLS_ERR_X509_UNKNOWN_OID                      -0x2100  /*< Requested OID is unknown. */
#define MBEDTLS_ERR_X509_INVALID_FORMAT                   -0x2180  /*< The CRT/CRL/CSR format is invalid, e.g. different type expected. */
#define MBEDTLS_ERR_X509_INVALID_VERSION                  -0x2200  /*< The CRT/CRL/CSR version element is invalid. */
#define MBEDTLS_ERR_X509_INVALID_SERIAL                   -0x2280  /*< The serial tag or value is invalid. */
#define MBEDTLS_ERR_X509_INVALID_ALG                      -0x2300  /*< The algorithm tag or value is invalid. */
#define MBEDTLS_ERR_X509_INVALID_NAME                     -0x2380  /*< The name tag or value is invalid. */
#define MBEDTLS_ERR_X509_INVALID_DATE                     -0x2400  /*< The date tag or value is invalid. */
#define MBEDTLS_ERR_X509_INVALID_SIGNATURE                -0x2480  /*< The signature tag or value invalid. */
#define MBEDTLS_ERR_X509_INVALID_EXTENSIONS               -0x2500  /*< The extension tag or value is invalid. */
#define MBEDTLS_ERR_X509_UNKNOWN_VERSION                  -0x2580  /*< CRT/CRL/CSR has an unsupported version number. */
#define MBEDTLS_ERR_X509_UNKNOWN_SIG_ALG                  -0x2600  /*< Signature algorithm (oid) is unsupported. */
#define MBEDTLS_ERR_X509_SIG_MISMATCH                     -0x2680  /*< Signature algorithms do not match. (see \c ::mbedtls_x509_crt sig_oid) */
#define MBEDTLS_ERR_X509_CERT_VERIFY_FAILED               -0x2700  /*< Certificate verification failed, e.g. CRL, CA or signature check failed. */
#define MBEDTLS_ERR_X509_CERT_UNKNOWN_FORMAT              -0x2780  /*< Format not recognized as DER or PEM. */
#define MBEDTLS_ERR_X509_BAD_INPUT_DATA                   -0x2800  /*< Input invalid. */
#define MBEDTLS_ERR_X509_ALLOC_FAILED                     -0x2880  /*< Allocation of memory failed. */
#define MBEDTLS_ERR_X509_FILE_IO_ERROR                    -0x2900  /*< Read/write of file failed. */
#define MBEDTLS_ERR_X509_BUFFER_TOO_SMALL                 -0x2980  /*< Destination buffer is too small. */
#define MBEDTLS_ERR_X509_FATAL_ERROR                      -0x3000  /*< A fatal error occurred, eg the chain is too long or the vrfy callback failed. */

/* Reminder: update x509_crt_verify_strings[] in library/x509_crt.c */
#define MBEDTLS_X509_BADCERT_EXPIRED             0x01  /*< The certificate validity has expired. */
#define MBEDTLS_X509_BADCERT_REVOKED             0x02  /*< The certificate has been revoked (is on a CRL). */
#define MBEDTLS_X509_BADCERT_CN_MISMATCH         0x04  /*< The certificate Common Name (CN) does not match with the expected CN. */
#define MBEDTLS_X509_BADCERT_NOT_TRUSTED         0x08  /*< The certificate is not correctly signed by the trusted CA. */
#define MBEDTLS_X509_BADCRL_NOT_TRUSTED          0x10  /*< The CRL is not correctly signed by the trusted CA. */
#define MBEDTLS_X509_BADCRL_EXPIRED              0x20  /*< The CRL is expired. */
#define MBEDTLS_X509_BADCERT_MISSING             0x40  /*< Certificate was missing. */
#define MBEDTLS_X509_BADCERT_SKIP_VERIFY         0x80  /*< Certificate verification was skipped. */
#define MBEDTLS_X509_BADCERT_OTHER             0x0100  /*< Other reason (can be used by verify callback) */
#define MBEDTLS_X509_BADCERT_FUTURE            0x0200  /*< The certificate validity starts in the future. */
#define MBEDTLS_X509_BADCRL_FUTURE             0x0400  /*< The CRL is from the future */
#define MBEDTLS_X509_BADCERT_KEY_USAGE         0x0800  /*< Usage does not match the keyUsage extension. */
#define MBEDTLS_X509_BADCERT_EXT_KEY_USAGE     0x1000  /*< Usage does not match the extendedKeyUsage extension. */
#define MBEDTLS_X509_BADCERT_NS_CERT_TYPE      0x2000  /*< Usage does not match the nsCertType extension. */
#define MBEDTLS_X509_BADCERT_BAD_MD            0x4000  /*< The certificate is signed with an unacceptable hash. */
#define MBEDTLS_X509_BADCERT_BAD_PK            0x8000  /*< The certificate is signed with an unacceptable PK alg (eg RSA vs ECDSA). */
#define MBEDTLS_X509_BADCERT_BAD_KEY         0x010000  /*< The certificate is signed with an unacceptable key (eg bad curve, RSA too short). */
#define MBEDTLS_X509_BADCRL_BAD_MD           0x020000  /*< The CRL is signed with an unacceptable hash. */
#define MBEDTLS_X509_BADCRL_BAD_PK           0x040000  /*< The CRL is signed with an unacceptable PK alg (eg RSA vs ECDSA). */
#define MBEDTLS_X509_BADCRL_BAD_KEY          0x080000  /*< The CRL is signed with an unacceptable key (eg bad curve, RSA too short). */

/*
 * X.509 v3 Subject Alternative Name types.
 *      otherName                       [0]     OtherName,
 *      rfc822Name                      [1]     IA5String,
 *      dNSName                         [2]     IA5String,
 *      x400Address                     [3]     ORAddress,
 *      directoryName                   [4]     Name,
 *      ediPartyName                    [5]     EDIPartyName,
 *      uniformResourceIdentifier       [6]     IA5String,
 *      iPAddress                       [7]     OCTET STRING,
 *      registeredID                    [8]     OBJECT IDENTIFIER
 */
#define MBEDTLS_X509_SAN_OTHER_NAME                      0
#define MBEDTLS_X509_SAN_RFC822_NAME                     1
#define MBEDTLS_X509_SAN_DNS_NAME                        2
#define MBEDTLS_X509_SAN_X400_ADDRESS_NAME               3
#define MBEDTLS_X509_SAN_DIRECTORY_NAME                  4
#define MBEDTLS_X509_SAN_EDI_PARTY_NAME                  5
#define MBEDTLS_X509_SAN_UNIFORM_RESOURCE_IDENTIFIER     6
#define MBEDTLS_X509_SAN_IP_ADDRESS                      7
#define MBEDTLS_X509_SAN_REGISTERED_ID                   8

/*
 * X.509 v3 Key Usage Extension flags
 * Reminder: update x509_info_key_usage() when adding new flags.
 */
#define MBEDTLS_X509_KU_DIGITAL_SIGNATURE            (0x80)  /* bit 0 */
#define MBEDTLS_X509_KU_NON_REPUDIATION              (0x40)  /* bit 1 */
#define MBEDTLS_X509_KU_KEY_ENCIPHERMENT             (0x20)  /* bit 2 */
#define MBEDTLS_X509_KU_DATA_ENCIPHERMENT            (0x10)  /* bit 3 */
#define MBEDTLS_X509_KU_KEY_AGREEMENT                (0x08)  /* bit 4 */
#define MBEDTLS_X509_KU_KEY_CERT_SIGN                (0x04)  /* bit 5 */
#define MBEDTLS_X509_KU_CRL_SIGN                     (0x02)  /* bit 6 */
#define MBEDTLS_X509_KU_ENCIPHER_ONLY                (0x01)  /* bit 7 */
#define MBEDTLS_X509_KU_DECIPHER_ONLY              (0x8000)  /* bit 8 */

/*
 * Netscape certificate types
 * (http://www.mozilla.org/projects/security/pki/nss/tech-notes/tn3.html)
 */

#define MBEDTLS_X509_NS_CERT_TYPE_SSL_CLIENT         (0x80)  /* bit 0 */
#define MBEDTLS_X509_NS_CERT_TYPE_SSL_SERVER         (0x40)  /* bit 1 */
#define MBEDTLS_X509_NS_CERT_TYPE_EMAIL              (0x20)  /* bit 2 */
#define MBEDTLS_X509_NS_CERT_TYPE_OBJECT_SIGNING     (0x10)  /* bit 3 */
#define MBEDTLS_X509_NS_CERT_TYPE_RESERVED           (0x08)  /* bit 4 */
#define MBEDTLS_X509_NS_CERT_TYPE_SSL_CA             (0x04)  /* bit 5 */
#define MBEDTLS_X509_NS_CERT_TYPE_EMAIL_CA           (0x02)  /* bit 6 */
#define MBEDTLS_X509_NS_CERT_TYPE_OBJECT_SIGNING_CA  (0x01)  /* bit 7 */

/*
 * X.509 extension types
 *
 * Comments refer to the status for using certificates. Status can be
 * different for writing certificates or reading CRLs or CSRs.
 *
 * Those are defined in oid.h as oid.c needs them in a data structure. Since
 * these were previously defined here, let's have aliases for compatibility.
 */
#define MBEDTLS_X509_EXT_AUTHORITY_KEY_IDENTIFIER MBEDTLS_OID_X509_EXT_AUTHORITY_KEY_IDENTIFIER
#define MBEDTLS_X509_EXT_SUBJECT_KEY_IDENTIFIER   MBEDTLS_OID_X509_EXT_SUBJECT_KEY_IDENTIFIER
#define MBEDTLS_X509_EXT_KEY_USAGE                MBEDTLS_OID_X509_EXT_KEY_USAGE
#define MBEDTLS_X509_EXT_CERTIFICATE_POLICIES     MBEDTLS_OID_X509_EXT_CERTIFICATE_POLICIES
#define MBEDTLS_X509_EXT_POLICY_MAPPINGS          MBEDTLS_OID_X509_EXT_POLICY_MAPPINGS
#define MBEDTLS_X509_EXT_SUBJECT_ALT_NAME         MBEDTLS_OID_X509_EXT_SUBJECT_ALT_NAME         /* Supported (DNS) */
#define MBEDTLS_X509_EXT_ISSUER_ALT_NAME          MBEDTLS_OID_X509_EXT_ISSUER_ALT_NAME
#define MBEDTLS_X509_EXT_SUBJECT_DIRECTORY_ATTRS  MBEDTLS_OID_X509_EXT_SUBJECT_DIRECTORY_ATTRS
#define MBEDTLS_X509_EXT_BASIC_CONSTRAINTS        MBEDTLS_OID_X509_EXT_BASIC_CONSTRAINTS        /* Supported */
#define MBEDTLS_X509_EXT_NAME_CONSTRAINTS         MBEDTLS_OID_X509_EXT_NAME_CONSTRAINTS
#define MBEDTLS_X509_EXT_POLICY_CONSTRAINTS       MBEDTLS_OID_X509_EXT_POLICY_CONSTRAINTS
#define MBEDTLS_X509_EXT_EXTENDED_KEY_USAGE       MBEDTLS_OID_X509_EXT_EXTENDED_KEY_USAGE
#define MBEDTLS_X509_EXT_CRL_DISTRIBUTION_POINTS  MBEDTLS_OID_X509_EXT_CRL_DISTRIBUTION_POINTS
#define MBEDTLS_X509_EXT_INIHIBIT_ANYPOLICY       MBEDTLS_OID_X509_EXT_INIHIBIT_ANYPOLICY
#define MBEDTLS_X509_EXT_FRESHEST_CRL             MBEDTLS_OID_X509_EXT_FRESHEST_CRL
#define MBEDTLS_X509_EXT_NS_CERT_TYPE             MBEDTLS_OID_X509_EXT_NS_CERT_TYPE

/*
 * Storage format identifiers
 * Recognized formats: PEM and DER
 */
#define MBEDTLS_X509_FORMAT_DER                 1
#define MBEDTLS_X509_FORMAT_PEM                 2

#define MBEDTLS_X509_MAX_DN_NAME_SIZE         256 /*< Maximum value size of a DN entry */

/**
 * Type-length-value structure that allows for ASN1 using DER.
 */
typedef mbedtls_asn1_buf mbedtls_x509_buf;

/**
 * Container for ASN1 bit strings.
 */
typedef mbedtls_asn1_bitstring mbedtls_x509_bitstring;

/**
 * Container for ASN1 named information objects.
 * It allows for Relative Distinguished Names (e.g. cn=localhost,ou=code,etc.).
 */
typedef mbedtls_asn1_named_data mbedtls_x509_name;

/**
 * Container for a sequence of ASN.1 items
 */
typedef mbedtls_asn1_sequence mbedtls_x509_sequence;

/** Container for date and time (precision in seconds). */
typedef struct mbedtls_x509_time
{
    int year, mon, day;         /*< Date. */
    int hour, min, sec;         /*< Time. */
}
mbedtls_x509_time;

int mbedtls_x509_dn_gets( char *, size_t, const mbedtls_x509_name * );
int mbedtls_x509_get_alg( unsigned char **, const unsigned char *, mbedtls_x509_buf *, mbedtls_x509_buf * );
int mbedtls_x509_get_alg_null( unsigned char **, const unsigned char *, mbedtls_x509_buf * );
int mbedtls_x509_get_ext( unsigned char **, const unsigned char *, mbedtls_x509_buf *, int );
int mbedtls_x509_get_name( unsigned char **, const unsigned char *, mbedtls_x509_name * );
int mbedtls_x509_get_serial( unsigned char **, const unsigned char *, mbedtls_x509_buf * );
int mbedtls_x509_get_sig( unsigned char **, const unsigned char *, mbedtls_x509_buf * );
int mbedtls_x509_get_sig_alg( const mbedtls_x509_buf *, const mbedtls_x509_buf *, mbedtls_md_type_t *, mbedtls_pk_type_t *, void ** );
int mbedtls_x509_get_time( unsigned char **, const unsigned char *, mbedtls_x509_time * );
int mbedtls_x509_key_size_helper( char *, size_t, const char * );
int mbedtls_x509_self_test( int );
int mbedtls_x509_serial_gets( char *, size_t, const mbedtls_x509_buf * );
int mbedtls_x509_set_extension( mbedtls_asn1_named_data **, const char *, size_t, int, const unsigned char *, size_t );
int mbedtls_x509_sig_alg_gets( char *, size_t, const mbedtls_x509_buf *, mbedtls_pk_type_t, mbedtls_md_type_t, const void * );
int mbedtls_x509_string_to_names( mbedtls_asn1_named_data **, const char * );
int mbedtls_x509_time_is_future( const mbedtls_x509_time * );
int mbedtls_x509_time_is_past( const mbedtls_x509_time * );
int mbedtls_x509_write_extensions( unsigned char **, unsigned char *, mbedtls_asn1_named_data * );
int mbedtls_x509_write_names( unsigned char **, unsigned char *, mbedtls_asn1_named_data * );
int mbedtls_x509_write_sig( unsigned char **, unsigned char *, const char *, size_t, unsigned char *, size_t );

#define MBEDTLS_X509_SAFE_SNPRINTF                          \
    do {                                                    \
        if( ret < 0 || (size_t) ret >= n )                  \
            return( MBEDTLS_ERR_X509_BUFFER_TOO_SMALL );    \
         n -= (size_t) ret;                                 \
        p += (size_t) ret;                                  \
    } while( 0 )

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_X509_H_ */
