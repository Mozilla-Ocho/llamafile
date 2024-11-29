#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_CHK_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_CHK_H_

#define MBEDTLS_CHK(f)             \
  do {                             \
    if ((ret = (f))) goto cleanup; \
  } while (0)

#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_CHK_H_ */
