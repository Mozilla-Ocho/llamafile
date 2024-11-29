#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_X25519_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_X25519_H_
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/ecp.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ECP_TLS_CURVE25519    0x1d
#define MBEDTLS_X25519_KEY_SIZE_BYTES 32

typedef enum {
  MBEDTLS_EVEREST_ECDH_OURS,
  MBEDTLS_EVEREST_ECDH_THEIRS,
} mbedtls_everest_ecdh_side;

typedef struct {
  unsigned char our_secret[MBEDTLS_X25519_KEY_SIZE_BYTES];
  unsigned char peer_point[MBEDTLS_X25519_KEY_SIZE_BYTES];
} mbedtls_ecdh_context_everest;

int mbedtls_everest_setup(mbedtls_ecdh_context_everest *, int);
void mbedtls_everest_free(mbedtls_ecdh_context_everest *);
int mbedtls_everest_make_params(mbedtls_ecdh_context_everest *, size_t *,
                                unsigned char *, size_t,
                                int (*)(void *, unsigned char *, size_t),
                                void *);
int mbedtls_everest_read_params(mbedtls_ecdh_context_everest *,
                                const unsigned char **, const unsigned char *);
int mbedtls_everest_get_params(mbedtls_ecdh_context_everest *,
                               const mbedtls_ecp_keypair *,
                               mbedtls_everest_ecdh_side);
int mbedtls_everest_make_public(mbedtls_ecdh_context_everest *, size_t *,
                                unsigned char *, size_t,
                                int (*)(void *, unsigned char *, size_t),
                                void *);
int mbedtls_everest_read_public(mbedtls_ecdh_context_everest *,
                                const unsigned char *, size_t);
int mbedtls_everest_calc_secret(mbedtls_ecdh_context_everest *, size_t *,
                                unsigned char *, size_t,
                                int (*)(void *, unsigned char *, size_t),
                                void *);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_X25519_H_ */
