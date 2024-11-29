#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_BIGNUM_INTERNAL_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_BIGNUM_INTERNAL_H_
#include "third_party/mbedtls/platform.h"
COSMOPOLITAN_C_START_

#define MPI_VALIDATE_RET(cond) \
  MBEDTLS_INTERNAL_VALIDATE_RET(cond, MBEDTLS_ERR_MPI_BAD_INPUT_DATA)
#define MPI_VALIDATE(cond) MBEDTLS_INTERNAL_VALIDATE(cond)

#define ciL (sizeof(mbedtls_mpi_uint)) /* chars in limb  */
#define biL (ciL << 3)                 /* bits  in limb  */
#define biH (ciL << 2)                 /* half limb size */

#define MPI_SIZE_T_MAX ((size_t)-1) /* SIZE_T_MAX is not standard */

/*
 * Convert between bits/chars and number of limbs
 * Divide first in order to avoid potential overflows
 */
#define BITS_TO_LIMBS(i)  ((i) / biL + ((i) % biL != 0))
#define CHARS_TO_LIMBS(i) ((i) / ciL + ((i) % ciL != 0))

extern void (*Mul4x4)(uint64_t[8], const uint64_t[4], const uint64_t[4]);
extern void (*ShiftRight)(uint64_t *, size_t, unsigned char);

void ShiftRightAvx(uint64_t *, size_t, unsigned char);
void ShiftRightPure(uint64_t *, size_t, unsigned char);
void Mul4x4Adx(uint64_t[8], const uint64_t[4], const uint64_t[4]);
void Mul6x6Adx(uint64_t[12], const uint64_t[6], const uint64_t[6]);
void Mul8x8Adx(uint64_t[16], const uint64_t[8], const uint64_t[8]);
void Mul4x4Pure(uint64_t[16], const uint64_t[8], const uint64_t[8]);
void Mul(uint64_t *, const uint64_t *, unsigned, const uint64_t *, unsigned);
void Karatsuba(uint64_t *, uint64_t *, uint64_t *, size_t, uint64_t *);
void mbedtls_mpi_mul_hlp(size_t, const uint64_t *, uint64_t *, uint64_t);
void mbedtls_mpi_mul_hlp1(size_t, const uint64_t *, uint64_t *, uint64_t);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_BIGNUM_INTERNAL_H_ */
