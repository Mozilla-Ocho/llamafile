#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SELECT_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SELECT_H_
COSMOPOLITAN_C_START_

static inline uint64_t Select(uint64_t a, uint64_t b, uint64_t mask) {
  return (__conceal("r", mask) & a) | (__conceal("r", ~mask) & b);
}

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SELECT_H_ */
