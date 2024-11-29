#ifndef MBEDTLS_X509_CSR_H_
#define MBEDTLS_X509_CSR_H_
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/x509.h"
COSMOPOLITAN_C_START_

/**
 * Certificate Signing Request (CSR) structure.
 */
typedef struct mbedtls_x509_csr {
    mbedtls_x509_buf raw;           /*< The raw CSR data (DER). */
    mbedtls_x509_buf cri;           /*< The raw CertificateRequestInfo body (DER). */
    int version;                    /*< CSR version (1=v1). */
    mbedtls_x509_buf  subject_raw;  /*< The raw subject data (DER). */
    mbedtls_x509_name subject;      /*< The parsed subject data (named information object). */
    mbedtls_pk_context pk;          /*< Container for the public key context. */
    mbedtls_x509_buf sig_oid;
    mbedtls_x509_buf sig;
    mbedtls_md_type_t sig_md;       /*< Internal representation of the MD algorithm of the signature algorithm, e.g. MBEDTLS_MD_SHA256 */
    mbedtls_pk_type_t sig_pk;       /*< Internal representation of the Public Key algorithm of the signature algorithm, e.g. MBEDTLS_PK_RSA */
    void *sig_opts;                 /*< Signature options to be passed to mbedtls_pk_verify_ext(), e.g. for RSASSA-PSS */
} mbedtls_x509_csr;

/**
 * Container for writing a CSR
 */
typedef struct mbedtls_x509write_csr {
    mbedtls_pk_context *key;
    mbedtls_asn1_named_data *subject;
    mbedtls_md_type_t md_alg;
    mbedtls_asn1_named_data *extensions;
} mbedtls_x509write_csr;

int mbedtls_x509_csr_info( char *, size_t, const char *, const mbedtls_x509_csr * );
int mbedtls_x509_csr_parse( mbedtls_x509_csr *, const unsigned char *, size_t );
int mbedtls_x509_csr_parse_der( mbedtls_x509_csr *, const unsigned char *, size_t );
int mbedtls_x509_csr_parse_file( mbedtls_x509_csr *, const char * );
int mbedtls_x509write_csr_der( mbedtls_x509write_csr *, unsigned char *, size_t, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_x509write_csr_pem( mbedtls_x509write_csr *, unsigned char *, size_t, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_x509write_csr_set_extension( mbedtls_x509write_csr *, const char *, size_t, const unsigned char *, size_t );
int mbedtls_x509write_csr_set_key_usage( mbedtls_x509write_csr *, unsigned char );
int mbedtls_x509write_csr_set_ns_cert_type( mbedtls_x509write_csr *, unsigned char );
int mbedtls_x509write_csr_set_subject_name( mbedtls_x509write_csr *, const char * );
void mbedtls_x509_csr_free( mbedtls_x509_csr * );
void mbedtls_x509_csr_init( mbedtls_x509_csr * );
void mbedtls_x509write_csr_free( mbedtls_x509write_csr * );
void mbedtls_x509write_csr_init( mbedtls_x509write_csr * );
void mbedtls_x509write_csr_set_key( mbedtls_x509write_csr *, mbedtls_pk_context * );
void mbedtls_x509write_csr_set_md_alg( mbedtls_x509write_csr *, mbedtls_md_type_t );

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_X509_CSR_H_ */
