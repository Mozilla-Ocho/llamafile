#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_BASE64_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_BASE64_H_
#include "third_party/mbedtls/config.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL   -0x002A  /*< Output buffer too small. */
#define MBEDTLS_ERR_BASE64_INVALID_CHARACTER  -0x002C  /*< Invalid character in input. */

int mbedtls_base64_encode(unsigned char *, size_t, size_t *, const unsigned char *, size_t);
int mbedtls_base64_decode(unsigned char *, size_t, size_t *, const unsigned char *, size_t);
int mbedtls_base64_self_test(int);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_BASE64_H_ */
