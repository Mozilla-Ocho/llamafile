#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SAN_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SAN_H_
#include "third_party/mbedtls/x509_crt.h"
COSMOPOLITAN_C_START_

struct mbedtls_san {
  int tag;
  union {
    const char *val;
    uint32_t ip4;
  };
};

int mbedtls_x509write_crt_set_subject_alternative_name(
    mbedtls_x509write_cert *, const struct mbedtls_san *, size_t);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SAN_H_ */
