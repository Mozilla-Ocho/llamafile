#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_PK_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_PK_H_
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/ecdsa.h"
#include "third_party/mbedtls/ecp.h"
#include "third_party/mbedtls/md.h"
#include "third_party/mbedtls/rsa.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_PK_ALLOC_FAILED        -0x3F80  /*< Memory allocation failed. */
#define MBEDTLS_ERR_PK_TYPE_MISMATCH       -0x3F00  /*< Type mismatch, eg attempt to encrypt with an ECDSA key */
#define MBEDTLS_ERR_PK_BAD_INPUT_DATA      -0x3E80  /*< Bad input parameters to function. */
#define MBEDTLS_ERR_PK_FILE_IO_ERROR       -0x3E00  /*< Read/write of file failed. */
#define MBEDTLS_ERR_PK_KEY_INVALID_VERSION -0x3D80  /*< Unsupported key version */
#define MBEDTLS_ERR_PK_KEY_INVALID_FORMAT  -0x3D00  /*< Invalid key tag or value. */
#define MBEDTLS_ERR_PK_UNKNOWN_PK_ALG      -0x3C80  /*< Key algorithm is unsupported (only RSA and EC are supported). */
#define MBEDTLS_ERR_PK_PASSWORD_REQUIRED   -0x3C00  /*< Private key password can't be empty. */
#define MBEDTLS_ERR_PK_PASSWORD_MISMATCH   -0x3B80  /*< Given private key password does not allow for correct decryption. */
#define MBEDTLS_ERR_PK_INVALID_PUBKEY      -0x3B00  /*< The pubkey tag or value is invalid (only RSA and EC are supported). */
#define MBEDTLS_ERR_PK_INVALID_ALG         -0x3A80  /*< The algorithm tag or value is invalid. */
#define MBEDTLS_ERR_PK_UNKNOWN_NAMED_CURVE -0x3A00  /*< Elliptic curve is unsupported (only NIST curves are supported). */
#define MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE -0x3980  /*< Unavailable feature, e.g. RSA disabled for RSA key. */
#define MBEDTLS_ERR_PK_SIG_LEN_MISMATCH    -0x3900  /*< The buffer contains a valid signature followed by more data. */

/* MBEDTLS_ERR_PK_HW_ACCEL_FAILED is deprecated and should not be used. */
#define MBEDTLS_ERR_PK_HW_ACCEL_FAILED     -0x3880  /*< PK hardware accelerator failed. */

/**
 * \brief          Public key types
 */
typedef enum {
    MBEDTLS_PK_NONE=0,
    MBEDTLS_PK_RSA,
    MBEDTLS_PK_ECKEY,
    MBEDTLS_PK_ECKEY_DH,
    MBEDTLS_PK_ECDSA,
    MBEDTLS_PK_RSA_ALT,
    MBEDTLS_PK_RSASSA_PSS,
    MBEDTLS_PK_OPAQUE,
} mbedtls_pk_type_t;

/**
 * \brief           Options for RSASSA-PSS signature verification.
 *                  See \c mbedtls_rsa_rsassa_pss_verify_ext()
 */
typedef struct mbedtls_pk_rsassa_pss_options
{
    mbedtls_md_type_t mgf1_hash_id;
    int expected_salt_len;

} mbedtls_pk_rsassa_pss_options;

/**
 * \brief           Maximum size of a signature made by mbedtls_pk_sign().
 */
/* We need to set MBEDTLS_PK_SIGNATURE_MAX_SIZE to the maximum signature
 * size among the supported signature types. Do it by starting at 0,
 * then incrementally increasing to be large enough for each supported
 * signature mechanism.
 *
 * The resulting value can be 0, for example if MBEDTLS_ECDH_C is enabled
 * (which allows the pk module to be included) but neither MBEDTLS_ECDSA_C
 * nor MBEDTLS_RSA_C nor any opaque signature mechanism (PSA or RSA_ALT).
 */
#define MBEDTLS_PK_SIGNATURE_MAX_SIZE 0

#if ( defined(MBEDTLS_RSA_C) || defined(MBEDTLS_PK_RSA_ALT_SUPPORT) ) && \
    MBEDTLS_MPI_MAX_SIZE > MBEDTLS_PK_SIGNATURE_MAX_SIZE
/* For RSA, the signature can be as large as the bignum module allows.
 * For RSA_ALT, the signature size is not necessarily tied to what the
 * bignum module can do, but in the absence of any specific setting,
 * we use that (rsa_alt_sign_wrap in pk_wrap will check). */
#undef MBEDTLS_PK_SIGNATURE_MAX_SIZE
#define MBEDTLS_PK_SIGNATURE_MAX_SIZE MBEDTLS_MPI_MAX_SIZE
#endif

#if defined(MBEDTLS_ECDSA_C) &&                                 \
    MBEDTLS_ECDSA_MAX_LEN > MBEDTLS_PK_SIGNATURE_MAX_SIZE
/* For ECDSA, the ecdsa module exports a constant for the maximum
 * signature size. */
#undef MBEDTLS_PK_SIGNATURE_MAX_SIZE
#define MBEDTLS_PK_SIGNATURE_MAX_SIZE MBEDTLS_ECDSA_MAX_LEN
#endif

#if defined(MBEDTLS_USE_PSA_CRYPTO)
#if PSA_SIGNATURE_MAX_SIZE > MBEDTLS_PK_SIGNATURE_MAX_SIZE
/* PSA_SIGNATURE_MAX_SIZE is the maximum size of a signature made
 * through the PSA API in the PSA representation. */
#undef MBEDTLS_PK_SIGNATURE_MAX_SIZE
#define MBEDTLS_PK_SIGNATURE_MAX_SIZE PSA_SIGNATURE_MAX_SIZE
#endif

#if PSA_VENDOR_ECDSA_SIGNATURE_MAX_SIZE + 11 > MBEDTLS_PK_SIGNATURE_MAX_SIZE
/* The Mbed TLS representation is different for ECDSA signatures:
 * PSA uses the raw concatenation of r and s,
 * whereas Mbed TLS uses the ASN.1 representation (SEQUENCE of two INTEGERs).
 * Add the overhead of ASN.1: up to (1+2) + 2 * (1+2+1) for the
 * types, lengths (represented by up to 2 bytes), and potential leading
 * zeros of the INTEGERs and the SEQUENCE. */
#undef MBEDTLS_PK_SIGNATURE_MAX_SIZE
#define MBEDTLS_PK_SIGNATURE_MAX_SIZE ( PSA_VENDOR_ECDSA_SIGNATURE_MAX_SIZE + 11 )
#endif
#endif /* defined(MBEDTLS_USE_PSA_CRYPTO) */

/**
 * \brief           Types for interfacing with the debug module
 */
typedef enum
{
    MBEDTLS_PK_DEBUG_NONE = 0,
    MBEDTLS_PK_DEBUG_MPI,
    MBEDTLS_PK_DEBUG_ECP,
} mbedtls_pk_debug_type;

/**
 * \brief           Item to send to the debug module
 */
typedef struct mbedtls_pk_debug_item
{
    mbedtls_pk_debug_type type;
    const char *name;
    void *value;
} mbedtls_pk_debug_item;

/** Maximum number of item send for debugging, plus 1 */
#define MBEDTLS_PK_DEBUG_MAX_ITEMS 3

/**
 * \brief           Public key information and operations
 */
typedef struct mbedtls_pk_info_t mbedtls_pk_info_t;

/**
 * \brief           Public key container
 */
typedef struct mbedtls_pk_context
{
    const mbedtls_pk_info_t *   pk_info; /*< Public key information         */
    void *                      pk_ctx;  /*< Underlying public key context  */
} mbedtls_pk_context;

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
/**
 * \brief           Context for resuming operations
 */
typedef struct
{
    const mbedtls_pk_info_t *   pk_info; /*< Public key information         */
    void *                      rs_ctx;  /*< Underlying restart context     */
} mbedtls_pk_restart_ctx;
#else /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */
/* Now we can declare functions that take a pointer to that */
typedef void mbedtls_pk_restart_ctx;
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

/**
 * Quick access to an RSA context inside a PK context.
 *
 * \warning You must make sure the PK context actually holds an RSA context
 * before using this function!
 */
static inline mbedtls_rsa_context *mbedtls_pk_rsa( const mbedtls_pk_context pk )
{
    return( (mbedtls_rsa_context *) (pk).pk_ctx );
}

/**
 * Quick access to an EC context inside a PK context.
 *
 * \warning You must make sure the PK context actually holds an EC context
 * before using this function!
 */
static inline mbedtls_ecp_keypair *mbedtls_pk_ec( const mbedtls_pk_context pk )
{
    return( (mbedtls_ecp_keypair *) (pk).pk_ctx );
}

/**
 * \brief           Types for RSA-alt abstraction
 */
typedef int (*mbedtls_pk_rsa_alt_decrypt_func)( void *ctx, int mode, size_t *olen,
                    const unsigned char *input, unsigned char *output,
                    size_t output_max_len );
typedef int (*mbedtls_pk_rsa_alt_sign_func)( void *ctx,
                    int (*f_rng)(void *, unsigned char *, size_t), void *p_rng,
                    int mode, mbedtls_md_type_t md_alg, unsigned int hashlen,
                    const unsigned char *hash, unsigned char *sig );
typedef size_t (*mbedtls_pk_rsa_alt_key_len_func)( void *ctx );

const mbedtls_pk_info_t *mbedtls_pk_info_from_type( mbedtls_pk_type_t );
void mbedtls_pk_init( mbedtls_pk_context * );
void mbedtls_pk_free( mbedtls_pk_context * );
void mbedtls_pk_restart_init( mbedtls_pk_restart_ctx * );
void mbedtls_pk_restart_free( mbedtls_pk_restart_ctx * );
int mbedtls_pk_setup( mbedtls_pk_context *, const mbedtls_pk_info_t * );
int mbedtls_pk_setup_rsa_alt( mbedtls_pk_context *, void *, mbedtls_pk_rsa_alt_decrypt_func, mbedtls_pk_rsa_alt_sign_func, mbedtls_pk_rsa_alt_key_len_func );
size_t mbedtls_pk_get_bitlen( const mbedtls_pk_context * );
const char * mbedtls_pk_get_name( const mbedtls_pk_context * );
const char *mbedtls_pk_type_name(mbedtls_pk_type_t);
int mbedtls_pk_can_do( const mbedtls_pk_context *, mbedtls_pk_type_t );
int mbedtls_pk_check_pair( const mbedtls_pk_context *, const mbedtls_pk_context * );
int mbedtls_pk_debug( const mbedtls_pk_context *, mbedtls_pk_debug_item * );
int mbedtls_pk_decrypt( mbedtls_pk_context *, const unsigned char *, size_t, unsigned char *, size_t *, size_t, int (*)(void *, unsigned char *, size_t),  void * );
int mbedtls_pk_encrypt( mbedtls_pk_context *, const unsigned char *, size_t, unsigned char *, size_t *, size_t, int (*)(void *, unsigned char *, size_t),  void * );
int mbedtls_pk_load_file( const char *, unsigned char **, size_t * );
int mbedtls_pk_parse_key( mbedtls_pk_context *, const unsigned char *, size_t, const unsigned char *, size_t );
int mbedtls_pk_parse_keyfile( mbedtls_pk_context *, const char *, const char * );
int mbedtls_pk_parse_public_key( mbedtls_pk_context *, const unsigned char *, size_t );
int mbedtls_pk_parse_public_keyfile( mbedtls_pk_context *, const char * );
int mbedtls_pk_parse_subpubkey( unsigned char **, const unsigned char *, mbedtls_pk_context * );
int mbedtls_pk_sign( mbedtls_pk_context *, mbedtls_md_type_t, const unsigned char *, size_t, unsigned char *, size_t *, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_pk_sign_restartable( mbedtls_pk_context *, mbedtls_md_type_t, const unsigned char *, size_t, unsigned char *, size_t *, int (*)(void *, unsigned char *, size_t),  void *, mbedtls_pk_restart_ctx * );
int mbedtls_pk_verify( mbedtls_pk_context *, mbedtls_md_type_t, const unsigned char *, size_t, const unsigned char *, size_t );
int mbedtls_pk_verify_ext( mbedtls_pk_type_t, const void *, mbedtls_pk_context *, mbedtls_md_type_t, const unsigned char *, size_t, const unsigned char *, size_t );
int mbedtls_pk_verify_restartable( mbedtls_pk_context *, mbedtls_md_type_t, const unsigned char *, size_t, const unsigned char *, size_t, mbedtls_pk_restart_ctx * );
int mbedtls_pk_write_key_der( mbedtls_pk_context *, unsigned char *, size_t );
int mbedtls_pk_write_key_pem( mbedtls_pk_context *, unsigned char *, size_t );
int mbedtls_pk_write_pubkey( unsigned char **, unsigned char *, const mbedtls_pk_context * );
int mbedtls_pk_write_pubkey_der( mbedtls_pk_context *, unsigned char *, size_t );
int mbedtls_pk_write_pubkey_pem( mbedtls_pk_context *, unsigned char *, size_t );
mbedtls_pk_type_t mbedtls_pk_get_type( const mbedtls_pk_context * );

/**
 * \brief           Get the length in bytes of the underlying key
 *
 * \param ctx       The context to query. It must have been initialized.
 *
 * \return          Key length in bytes, or 0 on error
 */
static inline size_t mbedtls_pk_get_len( const mbedtls_pk_context *ctx )
{
    return( ( mbedtls_pk_get_bitlen( ctx ) + 7 ) / 8 );
}

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_PK_H_ */
