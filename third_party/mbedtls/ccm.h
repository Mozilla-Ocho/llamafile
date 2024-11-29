#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_CCM_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_CCM_H_
#include "third_party/mbedtls/cipher.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_CCM_BAD_INPUT \
  -0x000D /*< Bad input parameters to the function. */
#define MBEDTLS_ERR_CCM_AUTH_FAILED \
  -0x000F /*< Authenticated decryption failed. */

/* MBEDTLS_ERR_CCM_HW_ACCEL_FAILED is deprecated and should not be used. */
#define MBEDTLS_ERR_CCM_HW_ACCEL_FAILED \
  -0x0011 /*< CCM hardware accelerator failed. */

/**
 * \brief    The CCM context-type definition. The CCM context is passed
 *           to the APIs called.
 */
typedef struct mbedtls_ccm_context {
  mbedtls_cipher_context_t cipher_ctx; /*!< The cipher context used. */
} mbedtls_ccm_context;

void mbedtls_ccm_init(mbedtls_ccm_context *);
int mbedtls_ccm_setkey(mbedtls_ccm_context *, mbedtls_cipher_id_t,
                       const unsigned char *, unsigned int);
void mbedtls_ccm_free(mbedtls_ccm_context *);
int mbedtls_ccm_encrypt_and_tag(mbedtls_ccm_context *, size_t,
                                const unsigned char *, size_t,
                                const unsigned char *, size_t,
                                const unsigned char *, unsigned char *,
                                unsigned char *, size_t);
int mbedtls_ccm_star_encrypt_and_tag(mbedtls_ccm_context *, size_t,
                                     const unsigned char *, size_t,
                                     const unsigned char *, size_t,
                                     const unsigned char *, unsigned char *,
                                     unsigned char *, size_t);
int mbedtls_ccm_auth_decrypt(mbedtls_ccm_context *, size_t,
                             const unsigned char *, size_t,
                             const unsigned char *, size_t,
                             const unsigned char *, unsigned char *,
                             const unsigned char *, size_t);
int mbedtls_ccm_star_auth_decrypt(mbedtls_ccm_context *, size_t,
                                  const unsigned char *, size_t,
                                  const unsigned char *, size_t,
                                  const unsigned char *, unsigned char *,
                                  const unsigned char *, size_t);

int mbedtls_ccm_self_test(int);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_CCM_H_ */
