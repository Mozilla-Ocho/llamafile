#ifndef MBEDTLS_SSL_TICKET_H
#define MBEDTLS_SSL_TICKET_H
#include "third_party/mbedtls/cipher.h"
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/ssl.h"

/*
 * This implementation of the session ticket callbacks includes key
 * management, rotating the keys periodically in order to preserve forward
 * secrecy, when MBEDTLS_HAVE_TIME is defined.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief   Information for session ticket protection
 */
typedef struct mbedtls_ssl_ticket_key
{
    unsigned char name[4];          /*!< random key identifier              */
    uint32_t generation_time;       /*!< key generation timestamp (seconds) */
    mbedtls_cipher_context_t ctx;   /*!< context for auth enc/decryption    */
}
mbedtls_ssl_ticket_key;

/**
 * \brief   Context for session ticket handling functions
 */
typedef struct mbedtls_ssl_ticket_context
{
    mbedtls_ssl_ticket_key keys[2]; /*!< ticket protection keys             */
    unsigned char active;           /*!< index of the currently active key  */

    uint32_t ticket_lifetime;       /*!< lifetime of tickets in seconds     */

    /** Callback for getting (pseudo-)random numbers                        */
    int  (*f_rng)(void *, unsigned char *, size_t);
    void *p_rng;                    /*!< context for the RNG function       */
}
mbedtls_ssl_ticket_context;

/**
 * \brief           Initialize a ticket context.
 *                  (Just make it ready for mbedtls_ssl_ticket_setup()
 *                  or mbedtls_ssl_ticket_free().)
 *
 * \param ctx       Context to be initialized
 */
void mbedtls_ssl_ticket_init( mbedtls_ssl_ticket_context *ctx );

int mbedtls_ssl_ticket_setup( mbedtls_ssl_ticket_context *ctx,
    int (*f_rng)(void *, unsigned char *, size_t), void *p_rng,
    mbedtls_cipher_type_t cipher,
    uint32_t lifetime );

/**
 * \brief           Implementation of the ticket write callback
 *
 * \note            See \c mbedtls_ssl_ticket_write_t for description
 */
extern mbedtls_ssl_ticket_write_t mbedtls_ssl_ticket_write;

/**
 * \brief           Implementation of the ticket parse callback
 *
 * \note            See \c mbedtls_ssl_ticket_parse_t for description
 */
extern mbedtls_ssl_ticket_parse_t mbedtls_ssl_ticket_parse;

/**
 * \brief           Free a context's content and zeroize it.
 *
 * \param ctx       Context to be cleaned up
 */
void mbedtls_ssl_ticket_free( mbedtls_ssl_ticket_context *ctx );

#ifdef __cplusplus
}
#endif

#endif /* ssl_ticket.h */
