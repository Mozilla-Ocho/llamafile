#ifndef MBEDTLS_X509_CRT_H_
#define MBEDTLS_X509_CRT_H_
#include "third_party/mbedtls/bignum.h"
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/x509.h"
#include "third_party/mbedtls/x509_crl.h"
COSMOPOLITAN_C_START_

/**
 * Container for an X.509 certificate. The certificate may be chained.
 */
typedef struct mbedtls_x509_crt
{
    int own_buffer;                     /*< Indicates if \c raw is owned
                                         *   by the structure or not.        */
    mbedtls_x509_buf raw;               /*< The raw certificate data (DER). */
    mbedtls_x509_buf tbs;               /*< The raw certificate body (DER). The part that is To Be Signed. */

    int version;                        /*< The X.509 version. (1=v1, 2=v2, 3=v3) */
    mbedtls_x509_buf serial;            /*< Unique id for certificate issued by a specific CA. */
    mbedtls_x509_buf sig_oid;           /*< Signature algorithm, e.g. sha1RSA */

    mbedtls_x509_buf issuer_raw;        /*< The raw issuer data (DER). Used for quick comparison. */
    mbedtls_x509_buf subject_raw;       /*< The raw subject data (DER). Used for quick comparison. */

    mbedtls_x509_name issuer;           /*< The parsed issuer data (named information object). */
    mbedtls_x509_name subject;          /*< The parsed subject data (named information object). */

    mbedtls_x509_time valid_from;       /*< Start time of certificate validity. */
    mbedtls_x509_time valid_to;         /*< End time of certificate validity. */

    mbedtls_x509_buf pk_raw;
    mbedtls_pk_context pk;              /*< Container for the public key context. */

    mbedtls_x509_buf issuer_id;         /*< Optional X.509 v2/v3 issuer unique identifier. */
    mbedtls_x509_buf subject_id;        /*< Optional X.509 v2/v3 subject unique identifier. */
    mbedtls_x509_buf v3_ext;            /*< Optional X.509 v3 extensions.  */
    mbedtls_x509_sequence subject_alt_names;    /*< Optional list of raw entries of Subject Alternative Names extension (currently only dNSName and OtherName are listed). */

    mbedtls_x509_sequence certificate_policies; /*< Optional list of certificate policies (Only anyPolicy is printed and enforced, however the rest of the policies are still listed). */

    int ext_types;              /*< Bit string containing detected and parsed extensions */
    int ca_istrue;              /*< Optional Basic Constraint extension value: 1 if this certificate belongs to a CA, 0 otherwise. */
    int max_pathlen;            /*< Optional Basic Constraint extension value: The maximum path length to the root certificate. Path length is 1 higher than RFC 5280 'meaning', so 1+ */

    unsigned int key_usage;     /*< Optional key usage extension value: See the values in x509.h */

    mbedtls_x509_sequence ext_key_usage; /*< Optional list of extended key usage OIDs. */

    unsigned char ns_cert_type; /*< Optional Netscape certificate type extension value: See the values in x509.h */

    mbedtls_x509_buf sig;               /*< Signature: hash of the tbs part signed with the private key. */
    mbedtls_md_type_t sig_md;           /*< Internal representation of the MD algorithm of the signature algorithm, e.g. MBEDTLS_MD_SHA256 */
    mbedtls_pk_type_t sig_pk;           /*< Internal representation of the Public Key algorithm of the signature algorithm, e.g. MBEDTLS_PK_RSA */
    void *sig_opts;             /*< Signature options to be passed to mbedtls_pk_verify_ext(), e.g. for RSASSA-PSS */

    struct mbedtls_x509_crt *next;     /*< Next certificate in the CA-chain. */
}
mbedtls_x509_crt;

/**
 * From RFC 5280 section 4.2.1.6:
 * OtherName ::= SEQUENCE {
 *      type-id    OBJECT IDENTIFIER,
 *      value      [0] EXPLICIT ANY DEFINED BY type-id }
 */
typedef struct mbedtls_x509_san_other_name
{
    /**
     * The type_id is an OID as deifned in RFC 5280.
     * To check the value of the type id, you should use
     * \p MBEDTLS_OID_CMP with a known OID mbedtls_x509_buf.
     */
    mbedtls_x509_buf type_id;                   /*< The type id. */
    union
    {
        /**
         * From RFC 4108 section 5:
         * HardwareModuleName ::= SEQUENCE {
         *                         hwType OBJECT IDENTIFIER,
         *                         hwSerialNum OCTET STRING }
         */
        struct
        {
            mbedtls_x509_buf oid;               /*< The object identifier. */
            mbedtls_x509_buf val;               /*< The named value. */
        }
        hardware_module_name;
    }
    value;
}
mbedtls_x509_san_other_name;

/**
 * A structure for holding the parsed Subject Alternative Name, according to type
 */
typedef struct mbedtls_x509_subject_alternative_name
{
    int type;                              /*< The SAN type, value of MBEDTLS_X509_SAN_XXX. */
    union {
        mbedtls_x509_san_other_name other_name; /*< The otherName supported type. */
        mbedtls_x509_buf   unstructured_name; /*< The buffer for the un constructed types. Only dnsName currently supported */
        uint32_t ip;
    }
    san; /*< A union of the supported SAN types */
}
mbedtls_x509_subject_alternative_name;

/**
 * Build flag from an algorithm/curve identifier (pk, md, ecp)
 * Since 0 is always XXX_NONE, ignore it.
 */
#define MBEDTLS_X509_ID_FLAG( id )   ( 1 << ( (id) - 1 ) )

/**
 * Security profile for certificate verification.
 *
 * All lists are bitfields, built by ORing flags from MBEDTLS_X509_ID_FLAG().
 */
typedef struct mbedtls_x509_crt_profile
{
    uint32_t allowed_mds;       /*< MDs for signatures         */
    uint32_t allowed_pks;       /*< PK algs for signatures     */
    uint32_t allowed_curves;    /*< Elliptic curves for ECDSA  */
    uint32_t rsa_min_bitlen;    /*< Minimum size for RSA keys  */
}
mbedtls_x509_crt_profile;

#define MBEDTLS_X509_CRT_VERSION_1              0
#define MBEDTLS_X509_CRT_VERSION_2              1
#define MBEDTLS_X509_CRT_VERSION_3              2

#define MBEDTLS_X509_RFC5280_MAX_SERIAL_LEN 32
#define MBEDTLS_X509_RFC5280_UTC_TIME_LEN   15

#if !defined( MBEDTLS_X509_MAX_FILE_PATH_LEN )
#define MBEDTLS_X509_MAX_FILE_PATH_LEN 512
#endif

/**
 * Container for writing a certificate (CRT)
 */
typedef struct mbedtls_x509write_cert
{
    int version;
    mbedtls_mpi serial;
    mbedtls_pk_context *subject_key;
    mbedtls_pk_context *issuer_key;
    mbedtls_asn1_named_data *subject;
    mbedtls_asn1_named_data *issuer;
    mbedtls_md_type_t md_alg;
    char not_before[MBEDTLS_X509_RFC5280_UTC_TIME_LEN + 1];
    char not_after[MBEDTLS_X509_RFC5280_UTC_TIME_LEN + 1];
    mbedtls_asn1_named_data *extensions;
}
mbedtls_x509write_cert;

/**
 * Item in a verification chain: cert and flags for it
 */
typedef struct {
    mbedtls_x509_crt *crt;
    uint32_t flags;
} mbedtls_x509_crt_verify_chain_item;

/**
 * Max size of verification chain: end-entity + intermediates + trusted root
 */
#define MBEDTLS_X509_MAX_VERIFY_CHAIN_SIZE  ( MBEDTLS_X509_MAX_INTERMEDIATE_CA + 2 )

/**
 * Verification chain as built by \c mbedtls_crt_verify_chain()
 */
typedef struct
{
    mbedtls_x509_crt_verify_chain_item items[MBEDTLS_X509_MAX_VERIFY_CHAIN_SIZE];
    unsigned len;

#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
    /* This stores the list of potential trusted signers obtained from
     * the CA callback used for the CRT verification, if configured.
     * We must track it somewhere because the callback passes its
     * ownership to the caller. */
    mbedtls_x509_crt *trust_ca_cb_result;
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */
} mbedtls_x509_crt_verify_chain;

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)

/**
 * \brief       Context for resuming X.509 verify operations
 */
typedef struct
{
    /* for check_signature() */
    mbedtls_pk_restart_ctx pk;

    /* for find_parent_in() */
    mbedtls_x509_crt *parent; /* non-null iff parent_in in progress */
    mbedtls_x509_crt *fallback_parent;
    int fallback_signature_is_good;

    /* for find_parent() */
    int parent_is_trusted; /* -1 if find_parent is not in progress */

    /* for verify_chain() */
    enum {
        x509_crt_rs_none,
        x509_crt_rs_find_parent,
    } in_progress;  /* none if no operation is in progress */
    int self_cnt;
    mbedtls_x509_crt_verify_chain ver_chain;

} mbedtls_x509_crt_restart_ctx;

#else /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

/* Now we can declare functions that take a pointer to that */
typedef void mbedtls_x509_crt_restart_ctx;

#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

/**
 * Default security profile. Should provide a good balance between security
 * and compatibility with current deployments.
 */
extern const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_default;

/**
 * Expected next default profile. Recommended for new deployments.
 * Currently targets a 128-bit security level, except for RSA-2048.
 */
extern const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_next;

/**
 * NSA Suite B profile.
 */
extern const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_suiteb;

/**
 * \brief          The type of certificate extension callbacks.
 *
 *                 Callbacks of this type are passed to and used by the
 *                 mbedtls_x509_crt_parse_der_with_ext_cb() routine when
 *                 it encounters either an unsupported extension or a
 *                 "certificate policies" extension containing any
 *                 unsupported certificate policies.
 *                 Future versions of the library may invoke the callback
 *                 in other cases, if and when the need arises.
 *
 * \param p_ctx    An opaque context passed to the callback.
 * \param crt      The certificate being parsed.
 * \param oid      The OID of the extension.
 * \param critical Whether the extension is critical.
 * \param p        Pointer to the start of the extension value
 *                 (the content of the OCTET STRING).
 * \param end      End of extension value.
 *
 * \note           The callback must fail and return a negative error code
 *                 if it can not parse or does not support the extension.
 *                 When the callback fails to parse a critical extension
 *                 mbedtls_x509_crt_parse_der_with_ext_cb() also fails.
 *                 When the callback fails to parse a non critical extension
 *                 mbedtls_x509_crt_parse_der_with_ext_cb() simply skips
 *                 the extension and continues parsing.
 *
 * \return         \c 0 on success.
 * \return         A negative error code on failure.
 */
typedef int (*mbedtls_x509_crt_ext_cb_t)( void *p_ctx,
                                          mbedtls_x509_crt const *crt,
                                          mbedtls_x509_buf const *oid,
                                          int critical,
                                          const unsigned char *p,
                                          const unsigned char *end );

/**
 * \brief               The type of trusted certificate callbacks.
 *
 *                      Callbacks of this type are passed to and used by the CRT
 *                      verification routine mbedtls_x509_crt_verify_with_ca_cb()
 *                      when looking for trusted signers of a given certificate.
 *
 *                      On success, the callback returns a list of trusted
 *                      certificates to be considered as potential signers
 *                      for the input certificate.
 *
 * \param p_ctx         An opaque context passed to the callback.
 * \param child         The certificate for which to search a potential signer.
 *                      This will point to a readable certificate.
 * \param candidate_cas The address at which to store the address of the first
 *                      entry in the generated linked list of candidate signers.
 *                      This will not be \c NULL.
 *
 * \note                The callback must only return a non-zero value on a
 *                      fatal error. If, in contrast, the search for a potential
 *                      signer completes without a single candidate, the
 *                      callback must return \c 0 and set \c *candidate_cas
 *                      to \c NULL.
 *
 * \return              \c 0 on success. In this case, \c *candidate_cas points
 *                      to a heap-allocated linked list of instances of
 *                      ::mbedtls_x509_crt, and ownership of this list is passed
 *                      to the caller.
 * \return              A negative error code on failure.
 */
typedef int (*mbedtls_x509_crt_ca_cb_t)( void *p_ctx,
                                         mbedtls_x509_crt const *child,
                                         mbedtls_x509_crt **candidate_cas );

int mbedtls_x509_crt_check_extended_key_usage( const mbedtls_x509_crt *, const char *, size_t );
int mbedtls_x509_crt_check_key_usage( const mbedtls_x509_crt *, unsigned int );
int mbedtls_x509_crt_check_parent( const mbedtls_x509_crt *, const mbedtls_x509_crt *, int );
int mbedtls_x509_crt_check_signature( const mbedtls_x509_crt *, mbedtls_x509_crt *, mbedtls_x509_crt_restart_ctx * );
int mbedtls_x509_crt_info( char *, size_t, const char *, const mbedtls_x509_crt * );
int mbedtls_x509_crt_is_revoked( const mbedtls_x509_crt *, const mbedtls_x509_crl * );
int mbedtls_x509_crt_parse( mbedtls_x509_crt *, const unsigned char *, size_t );
int mbedtls_x509_crt_parse_der( mbedtls_x509_crt *, const unsigned char *, size_t );
int mbedtls_x509_crt_parse_der_nocopy( mbedtls_x509_crt *, const unsigned char *, size_t );
int mbedtls_x509_crt_parse_der_with_ext_cb( mbedtls_x509_crt *, const unsigned char *, size_t, int, mbedtls_x509_crt_ext_cb_t, void * );
int mbedtls_x509_crt_parse_file( mbedtls_x509_crt *, const char * );
int mbedtls_x509_crt_parse_path( mbedtls_x509_crt *, const char * );
int mbedtls_x509_crt_verify( mbedtls_x509_crt *, mbedtls_x509_crt *, mbedtls_x509_crl *, const char *, uint32_t *, int (*)(void *, mbedtls_x509_crt *, int, uint32_t *), void * );
int mbedtls_x509_crt_verify_info( char *, size_t, const char *, uint32_t );
int mbedtls_x509_crt_verify_restartable( mbedtls_x509_crt *, mbedtls_x509_crt *, mbedtls_x509_crl *, const mbedtls_x509_crt_profile *, const char *, uint32_t *, int (*)(void *, mbedtls_x509_crt *, int, uint32_t *), void *, mbedtls_x509_crt_restart_ctx * );
int mbedtls_x509_crt_verify_with_ca_cb( mbedtls_x509_crt *, mbedtls_x509_crt_ca_cb_t, void *, const mbedtls_x509_crt_profile *, const char *, uint32_t *, int (*)(void *, mbedtls_x509_crt *, int, uint32_t *), void * );
int mbedtls_x509_crt_verify_with_profile( mbedtls_x509_crt *, mbedtls_x509_crt *, mbedtls_x509_crl *, const mbedtls_x509_crt_profile *, const char *, uint32_t *, int (*)(void *, mbedtls_x509_crt *, int, uint32_t *), void * );
int mbedtls_x509_name_cmp( const mbedtls_x509_name *, const mbedtls_x509_name * );
int mbedtls_x509_parse_subject_alt_name( const mbedtls_x509_buf *, mbedtls_x509_subject_alternative_name * );
int mbedtls_x509write_crt_der( mbedtls_x509write_cert *, unsigned char *, size_t, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_x509write_crt_pem( mbedtls_x509write_cert *, unsigned char *, size_t, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_x509write_crt_set_authority_key_identifier( mbedtls_x509write_cert * );
int mbedtls_x509write_crt_set_basic_constraints( mbedtls_x509write_cert *, int, int );
int mbedtls_x509write_crt_set_ext_key_usage(mbedtls_x509write_cert *, int);
int mbedtls_x509write_crt_set_extension( mbedtls_x509write_cert *, const char *, size_t, int, const unsigned char *, size_t );
int mbedtls_x509write_crt_set_issuer_name( mbedtls_x509write_cert *, const char * );
int mbedtls_x509write_crt_set_key_usage( mbedtls_x509write_cert *, unsigned int );
int mbedtls_x509write_crt_set_ns_cert_type( mbedtls_x509write_cert *, unsigned char );
int mbedtls_x509write_crt_set_serial( mbedtls_x509write_cert *, const mbedtls_mpi * );
int mbedtls_x509write_crt_set_subject_key_identifier( mbedtls_x509write_cert * );
int mbedtls_x509write_crt_set_subject_name( mbedtls_x509write_cert *, const char * );
int mbedtls_x509write_crt_set_validity( mbedtls_x509write_cert *, const char *, const char * );
void mbedtls_x509_crt_free( mbedtls_x509_crt * );
void mbedtls_x509_crt_init( mbedtls_x509_crt * );
void mbedtls_x509_crt_restart_free( mbedtls_x509_crt_restart_ctx * );
void mbedtls_x509_crt_restart_init( mbedtls_x509_crt_restart_ctx * );
void mbedtls_x509write_crt_free( mbedtls_x509write_cert * );
void mbedtls_x509write_crt_init( mbedtls_x509write_cert * );
void mbedtls_x509write_crt_set_issuer_key( mbedtls_x509write_cert *, mbedtls_pk_context * );
void mbedtls_x509write_crt_set_md_alg( mbedtls_x509write_cert *, mbedtls_md_type_t );
void mbedtls_x509write_crt_set_subject_key( mbedtls_x509write_cert *, mbedtls_pk_context * );
void mbedtls_x509write_crt_set_version( mbedtls_x509write_cert *, int );

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_X509_CRT_H */
