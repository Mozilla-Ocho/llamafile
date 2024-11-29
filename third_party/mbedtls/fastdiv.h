#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_FASTDIV_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_FASTDIV_H_
#include <libc/macros.h>
COSMOPOLITAN_C_START_

struct Divisor {
  uint64_t m;
  uint8_t s;
  uint8_t t;
};

static inline struct Divisor GetDivisor(uint64_t d) {
  int b;
  uint128_t x;
  b = __builtin_clzll(d) ^ 63;
  x = -d & (((1ull << b) - 1) | (1ull << b));
  return (struct Divisor){(x << 64) / d + 1, MIN(1, b + 1), MAX(0, b)};
}

forceinline uint64_t Divide(uint64_t x, struct Divisor d) {
  uint128_t t;
  uint64_t l, h;
  t = d.m;
  t *= x;
  l = t;
  h = t >> 64;
  l = (x - h) >> d.s;
  return (h + l) >> d.t;
}

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_FASTDIV_H_ */
