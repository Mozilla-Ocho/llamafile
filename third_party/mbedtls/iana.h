#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_IANA_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_IANA_H_
#include "third_party/mbedtls/ssl.h"
COSMOPOLITAN_C_START_

bool IsCipherSuiteGood(uint16_t);
const char *GetCipherSuiteName(uint16_t);
const char *DescribeMbedtlsErrorCode(int);
const char *GetAlertDescription(unsigned char);
char *FormatSslClientCiphers(const mbedtls_ssl_context *) __wur;
const char *DescribeSslClientHandshakeError(const mbedtls_ssl_context *, int);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_IANA_H_ */
