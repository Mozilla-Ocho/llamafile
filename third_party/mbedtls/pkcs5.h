#ifndef MBEDTLS_PKCS5_H
#define MBEDTLS_PKCS5_H
#include "third_party/mbedtls/asn1.h"
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/md.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_PKCS5_BAD_INPUT_DATA                  -0x2f80  /*< Bad input parameters to function. */
#define MBEDTLS_ERR_PKCS5_INVALID_FORMAT                  -0x2f00  /*< Unexpected ASN.1 data. */
#define MBEDTLS_ERR_PKCS5_FEATURE_UNAVAILABLE             -0x2e80  /*< Requested encryption or digest alg not available. */
#define MBEDTLS_ERR_PKCS5_PASSWORD_MISMATCH               -0x2e00  /*< Given private key password does not allow for correct decryption. */

#define MBEDTLS_PKCS5_DECRYPT      0
#define MBEDTLS_PKCS5_ENCRYPT      1

int mbedtls_pkcs5_pbes2(const mbedtls_asn1_buf *, int, const unsigned char *,
                        size_t, const unsigned char *, size_t, unsigned char *);
int mbedtls_pkcs5_pbkdf2_hmac(mbedtls_md_context_t *, const void *, size_t,
                              const void *, size_t, unsigned, uint32_t,
                              unsigned char *);
int mbedtls_pkcs5_self_test(int);

COSMOPOLITAN_C_END_
#endif /* pkcs5.h */
