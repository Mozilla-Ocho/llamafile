#ifndef MBEDTLS_HMAC_DRBG_H_
#define MBEDTLS_HMAC_DRBG_H_
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/md.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_HMAC_DRBG_REQUEST_TOO_BIG              -0x0003  /*< Too many random requested in single call. */
#define MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG                -0x0005  /*< Input too large (Entropy + additional). */
#define MBEDTLS_ERR_HMAC_DRBG_FILE_IO_ERROR                -0x0007  /*< Read/write error in file. */
#define MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED        -0x0009  /*< The entropy source failed. */

#if !defined(MBEDTLS_HMAC_DRBG_RESEED_INTERVAL)
#define MBEDTLS_HMAC_DRBG_RESEED_INTERVAL   10000   /*< Interval before reseed is performed by default */
#endif

#if !defined(MBEDTLS_HMAC_DRBG_MAX_INPUT)
#define MBEDTLS_HMAC_DRBG_MAX_INPUT         256     /*< Maximum number of additional input bytes */
#endif

#if !defined(MBEDTLS_HMAC_DRBG_MAX_REQUEST)
#define MBEDTLS_HMAC_DRBG_MAX_REQUEST       1024    /*< Maximum number of requested bytes per call */
#endif

#if !defined(MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT)
#define MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT    384     /*< Maximum size of (re)seed buffer */
#endif

#define MBEDTLS_HMAC_DRBG_PR_OFF   0   /*< No prediction resistance       */
#define MBEDTLS_HMAC_DRBG_PR_ON    1   /*< Prediction resistance enabled  */

typedef struct mbedtls_hmac_drbg_context
{
    mbedtls_md_context_t md_ctx;
    unsigned char V[MBEDTLS_MD_MAX_SIZE];
    int reseed_counter;
    size_t entropy_len;
    int prediction_resistance;
    int reseed_interval;
    int (*f_entropy)(void *, unsigned char *, size_t);
    void *p_entropy;
} mbedtls_hmac_drbg_context;

void mbedtls_hmac_drbg_init( mbedtls_hmac_drbg_context * );
int mbedtls_hmac_drbg_seed( mbedtls_hmac_drbg_context *, const mbedtls_md_info_t * , int (*)(void *, unsigned char *, size_t), void *, const unsigned char *, size_t );
int mbedtls_hmac_drbg_seed_buf( mbedtls_hmac_drbg_context *, const mbedtls_md_info_t *, const unsigned char *, size_t );
void mbedtls_hmac_drbg_set_prediction_resistance( mbedtls_hmac_drbg_context *, int );
void mbedtls_hmac_drbg_set_entropy_len( mbedtls_hmac_drbg_context *, size_t );
void mbedtls_hmac_drbg_set_reseed_interval( mbedtls_hmac_drbg_context *, int );
int mbedtls_hmac_drbg_update_ret( mbedtls_hmac_drbg_context *, const unsigned char *, size_t );
int mbedtls_hmac_drbg_reseed( mbedtls_hmac_drbg_context *, const unsigned char *, size_t );
int mbedtls_hmac_drbg_random_with_add( void *, unsigned char *, size_t , const unsigned char *, size_t );
int mbedtls_hmac_drbg_random( void *, unsigned char *, size_t );
void mbedtls_hmac_drbg_free( mbedtls_hmac_drbg_context * );
int mbedtls_hmac_drbg_write_seed_file( mbedtls_hmac_drbg_context *, const char * );
int mbedtls_hmac_drbg_update_seed_file( mbedtls_hmac_drbg_context *, const char * );
int mbedtls_hmac_drbg_self_test( int );

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_HMAC_DRBG_H_ */
