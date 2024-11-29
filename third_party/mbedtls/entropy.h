#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ENTROPY_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ENTROPY_H_
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/sha256.h"
#include "third_party/mbedtls/sha512.h"
COSMOPOLITAN_C_START_

#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_ENTROPY_FORCE_SHA256)
#define MBEDTLS_ENTROPY_SHA512_ACCUMULATOR
#else
#if defined(MBEDTLS_SHA256_C)
#define MBEDTLS_ENTROPY_SHA256_ACCUMULATOR
#endif
#endif

#define MBEDTLS_ERR_ENTROPY_SOURCE_FAILED                 -0x003C  /*< Critical entropy source failure. */
#define MBEDTLS_ERR_ENTROPY_MAX_SOURCES                   -0x003E  /*< No more sources can be added. */
#define MBEDTLS_ERR_ENTROPY_NO_SOURCES_DEFINED            -0x0040  /*< No sources have been added to poll. */
#define MBEDTLS_ERR_ENTROPY_NO_STRONG_SOURCE              -0x003D  /*< No strong sources have been added to poll. */
#define MBEDTLS_ERR_ENTROPY_FILE_IO_ERROR                 -0x003F  /*< Read/write error in file. */

#if !defined(MBEDTLS_ENTROPY_MAX_SOURCES)
#define MBEDTLS_ENTROPY_MAX_SOURCES     20      /*< Maximum number of sources supported */
#endif

#if !defined(MBEDTLS_ENTROPY_MAX_GATHER)
#define MBEDTLS_ENTROPY_MAX_GATHER      128     /*< Maximum amount requested from entropy sources */
#endif

#if defined(MBEDTLS_ENTROPY_SHA512_ACCUMULATOR)
#define MBEDTLS_ENTROPY_BLOCK_SIZE      64      /*< Block size of entropy accumulator (SHA-512) */
#else
#define MBEDTLS_ENTROPY_BLOCK_SIZE      32      /*< Block size of entropy accumulator (SHA-256) */
#endif

#define MBEDTLS_ENTROPY_MAX_SEED_SIZE   1024    /*< Maximum size of seed we read from seed file */
#define MBEDTLS_ENTROPY_SOURCE_MANUAL   MBEDTLS_ENTROPY_MAX_SOURCES

#define MBEDTLS_ENTROPY_SOURCE_STRONG   1       /*< Entropy source is strong   */
#define MBEDTLS_ENTROPY_SOURCE_WEAK     0       /*< Entropy source is weak     */

/**
 * \brief           Entropy poll callback pointer
 *
 * \param data      Callback-specific data pointer
 * \param output    Data to fill
 * \param len       Maximum size to provide
 * \param olen      The actual amount of bytes put into the buffer (Can be 0)
 *
 * \return          0 if no critical failures occurred,
 *                  MBEDTLS_ERR_ENTROPY_SOURCE_FAILED otherwise
 */
typedef int (*mbedtls_entropy_f_source_ptr)(void *data, unsigned char *output, size_t len, size_t *olen);

/**
 * \brief           Entropy source state
 */
typedef struct mbedtls_entropy_source_state
{
    mbedtls_entropy_f_source_ptr    f_source;   /*< The entropy source callback */
    void *          p_source;   /*< The callback data pointer */
    size_t          size;       /*< Amount received in bytes */
    size_t          threshold;  /*< Minimum bytes required before release */
    int             strong;     /*< Is the source strong? */
}
mbedtls_entropy_source_state;

/**
 * \brief           Entropy context structure
 */
typedef struct mbedtls_entropy_context
{
    int accumulator_started; /* 0 after init.
                              * 1 after the first update.
                              * -1 after free. */
#if defined(MBEDTLS_ENTROPY_SHA512_ACCUMULATOR)
    mbedtls_sha512_context  accumulator;
#else
    mbedtls_sha256_context  accumulator;
#endif
    int             source_count; /* Number of entries used in source. */
    mbedtls_entropy_source_state    source[MBEDTLS_ENTROPY_MAX_SOURCES];
#if defined(MBEDTLS_ENTROPY_NV_SEED)
    int initial_entropy_run;
#endif
}
mbedtls_entropy_context;

void mbedtls_entropy_init( mbedtls_entropy_context * );
void mbedtls_entropy_free( mbedtls_entropy_context * );
int mbedtls_entropy_add_source( mbedtls_entropy_context *, mbedtls_entropy_f_source_ptr, void *, size_t, int );
int mbedtls_entropy_gather( mbedtls_entropy_context * );
int mbedtls_entropy_func( void *, unsigned char *, size_t );
int mbedtls_entropy_update_manual( mbedtls_entropy_context *, const unsigned char *, size_t );
int mbedtls_entropy_update_nv_seed( mbedtls_entropy_context * );
int mbedtls_entropy_write_seed_file( mbedtls_entropy_context *, const char * );
int mbedtls_entropy_update_seed_file( mbedtls_entropy_context *, const char * );
int mbedtls_entropy_self_test( int );
int mbedtls_entropy_source_self_test( int );

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ENTROPY_H_ */
