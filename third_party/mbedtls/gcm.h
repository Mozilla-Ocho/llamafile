#ifndef MBEDTLS_GCM_H_
#define MBEDTLS_GCM_H_
#include "third_party/mbedtls/cipher.h"
#include "third_party/mbedtls/config.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_GCM_ENCRYPT     1
#define MBEDTLS_GCM_DECRYPT     0

#define MBEDTLS_ERR_GCM_AUTH_FAILED                       -0x0012  /*< Authenticated decryption failed. */
#define MBEDTLS_ERR_GCM_BAD_INPUT                         -0x0014  /*< Bad input parameters to function. */

typedef struct mbedtls_gcm_context {
    mbedtls_cipher_context_t cipher_ctx;  /*!< The cipher context used. */
    uint64_t len;                         /*!< The total length of the encrypted data. */
    uint64_t add_len;                     /*!< The total length of the additional data. */
    unsigned char base_ectr[16];          /*!< The first ECTR for tag. */
    unsigned char y[16];                  /*!< The Y working value. */
    unsigned char buf[16];                /*!< The buf working value. */
    int mode;                             /*!< The operation to perform: #MBEDTLS_GCM_ENCRYPT or #MBEDTLS_GCM_DECRYPT. */
    uint64_t H8[2];                       /*!< For AES-NI. */
    uint64_t HL[16];                      /*!< Precalculated HTable low. */
    uint64_t HH[16];                      /*!< Precalculated HTable high. */
    mbedtls_cipher_id_t cipher;           /*!< The cipher being used. */
} mbedtls_gcm_context;

void mbedtls_gcm_init( mbedtls_gcm_context * );
int mbedtls_gcm_setkey( mbedtls_gcm_context *, mbedtls_cipher_id_t, const unsigned char *, unsigned int );
int mbedtls_gcm_crypt_and_tag( mbedtls_gcm_context *, int, size_t, const unsigned char *, size_t, const unsigned char *, size_t, const unsigned char *, unsigned char *, size_t, unsigned char * );
int mbedtls_gcm_auth_decrypt( mbedtls_gcm_context *, size_t, const unsigned char *, size_t, const unsigned char *, size_t, const unsigned char *, size_t, const unsigned char *, unsigned char * );
int mbedtls_gcm_starts( mbedtls_gcm_context *, int, const unsigned char *, size_t, const unsigned char *, size_t );
int mbedtls_gcm_update( mbedtls_gcm_context *, size_t, const unsigned char *, unsigned char * );
int mbedtls_gcm_finish( mbedtls_gcm_context *, unsigned char *, size_t );
void mbedtls_gcm_free( mbedtls_gcm_context * );
int mbedtls_gcm_self_test( int );

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_GCM_H_ */
