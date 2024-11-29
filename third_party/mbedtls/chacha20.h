#ifndef MBEDTLS_CHACHA20_H_
#define MBEDTLS_CHACHA20_H_
#include "third_party/mbedtls/config.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_CHACHA20_BAD_INPUT_DATA         -0x0051 /*< Invalid input parameter(s). */

typedef struct mbedtls_chacha20_context
{
    uint32_t state[16];          /*! The state (before round operations). */
    uint8_t  keystream8[64];     /*! Leftover keystream bytes. */
    size_t keystream_bytes_used; /*! Number of keystream bytes already used. */
}
mbedtls_chacha20_context;

void mbedtls_chacha20_init( mbedtls_chacha20_context * );
void mbedtls_chacha20_free( mbedtls_chacha20_context * );
int mbedtls_chacha20_setkey( mbedtls_chacha20_context *, const unsigned char[32] );
int mbedtls_chacha20_starts( mbedtls_chacha20_context *, const unsigned char[12], uint32_t );
int mbedtls_chacha20_update( mbedtls_chacha20_context *, size_t, const unsigned char *, unsigned char * );
int mbedtls_chacha20_crypt( const unsigned char[32], const unsigned char[12], uint32_t, size_t, const unsigned char *, unsigned char * );
int mbedtls_chacha20_self_test( int );

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_CHACHA20_H_ */
