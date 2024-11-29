#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_MATH_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_MATH_H_

#define ADC(R, A, B, CI, CO) \
  do {                       \
    uint64_t Ta = A;         \
    uint64_t Tb = B;         \
    CO = (Ta += CI) < CI;    \
    CO += (Ta += Tb) < Tb;   \
    R = Ta;                  \
  } while (0)

#define SBB(R, A, B, CI, CO) \
  do {                       \
    uint64_t Ta = A;         \
    uint64_t Tb = B;         \
    uint64_t Tc = Ta < CI;   \
    Ta -= CI;                \
    CO = (Ta < Tb) + Tc;     \
    Ta -= Tb;                \
    R = Ta;                  \
  } while (0)

#define MADD(a, b, c0, c1, c2) \
  t = (uint128_t)a * b;        \
  t += c0;                     \
  c0 = t;                      \
  h = t >> 64;                 \
  c1 += h;                     \
  if (c1 < h) c2++

#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_MATH_H_ */
