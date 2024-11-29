#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_AESCE_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_AESCE_H_
#include "third_party/mbedtls/aes.h"
COSMOPOLITAN_C_START_

/**
 * \brief          Internal AES-ECB block encryption and decryption
 *
 * \param ctx      AES context
 * \param mode     MBEDTLS_AES_ENCRYPT or MBEDTLS_AES_DECRYPT
 * \param input    16-byte input block
 * \param output   16-byte output block
 *
 * \return         0 on success (cannot fail)
 */
int mbedtls_aesce_crypt_ecb(mbedtls_aes_context *ctx, int mode,
                            const unsigned char input[16],
                            unsigned char output[16]);

/**
 * \brief          Internal GCM multiplication: c = a * b in GF(2^128)
 *
 * \note           This function is only for internal use by other library
 *                 functions; you must not call it directly.
 *
 * \param c        Result
 * \param a        First operand
 * \param b        Second operand
 *
 * \note           Both operands and result are bit strings interpreted as
 *                 elements of GF(2^128) as per the GCM spec.
 */
void mbedtls_aesce_gcm_mult(unsigned char c[16], const unsigned char a[16],
                            const unsigned char b[16]);

/**
 * \brief           Internal round key inversion. This function computes
 *                  decryption round keys from the encryption round keys.
 *
 * \param invkey    Round keys for the equivalent inverse cipher
 * \param fwdkey    Original round keys (for encryption)
 * \param nr        Number of rounds (that is, number of round keys minus one)
 */
void mbedtls_aesce_inverse_key(unsigned char *invkey,
                               const unsigned char *fwdkey, int nr);

/**
 * \brief           Internal key expansion for encryption
 *
 * \param rk        Destination buffer where the round keys are written
 * \param key       Encryption key
 * \param bits      Key size in bits (must be 128, 192 or 256)
 *
 * \return          0 if successful, or MBEDTLS_ERR_AES_INVALID_KEY_LENGTH
 */
int mbedtls_aesce_setkey_enc(unsigned char *rk, const unsigned char *key,
                             size_t bits);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_AESCE_H_ */
