#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ENTROPY_POLL_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ENTROPY_POLL_H_
COSMOPOLITAN_C_START_

int mbedtls_null_entropy_poll(void *, unsigned char *, size_t, size_t *);
int mbedtls_platform_entropy_poll(void *, unsigned char *, size_t, size_t *);
int mbedtls_hardclock_poll(void *, unsigned char *, size_t, size_t *);
int mbedtls_hardware_poll(void *, unsigned char *, size_t, size_t *);

COSMOPOLITAN_C_END_
#define MBEDTLS_ENTROPY_MIN_PLATFORM     32     /*< Minimum for platform source    */
#define MBEDTLS_ENTROPY_MIN_HARDCLOCK     4     /*< Minimum for mbedtls_timing_hardclock()        */
#define MBEDTLS_ENTROPY_MIN_HARDWARE     32     /*< Minimum for the hardware source */
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ENTROPY_POLL_H_ */
