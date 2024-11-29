#ifndef MBEDTLS_AESNI_H_
#define MBEDTLS_AESNI_H_
#include "third_party/mbedtls/aes.h"
#include "third_party/mbedtls/config.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_AESNI_AES      0x02000000u
#define MBEDTLS_AESNI_CLMUL    0x00000002u

int mbedtls_aesni_crypt_ecb( mbedtls_aes_context *, int, const unsigned char[16], unsigned char[16] );
void mbedtls_aesni_gcm_mult( unsigned char[16], const uint64_t[2] );
void mbedtls_aesni_inverse_key( unsigned char *, const unsigned char *, int );
int mbedtls_aesni_setkey_enc( unsigned char *, const unsigned char *, size_t );

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_AESNI_H_ */
