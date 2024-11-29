#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_INVASIVE_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_INVASIVE_H_
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/md.h"
COSMOPOLITAN_C_START_
#if defined(MBEDTLS_TEST_HOOKS) && defined(MBEDTLS_SSL_SOME_SUITES_USE_TLS_CBC)

/** \brief Compute the HMAC of variable-length data with constant flow.
 *
 * This function computes the HMAC of the concatenation of \p add_data and \p
 * data, and does with a code flow and memory access pattern that does not
 * depend on \p data_len_secret, but only on \p min_data_len and \p
 * max_data_len. In particular, this function always reads exactly \p
 * max_data_len bytes from \p data.
 *
 * \param ctx               The HMAC context. It must have keys configured
 *                          with mbedtls_md_hmac_starts() and use one of the
 *                          following hashes: SHA-384, SHA-256, SHA-1 or MD-5.
 *                          It is reset using mbedtls_md_hmac_reset() after
 *                          the computation is complete to prepare for the
 *                          next computation.
 * \param add_data          The additional data prepended to \p data. This
 *                          must point to a readable buffer of \p add_data_len
 *                          bytes.
 * \param add_data_len      The length of \p add_data in bytes.
 * \param data              The data appended to \p add_data. This must point
 *                          to a readable buffer of \p max_data_len bytes.
 * \param data_len_secret   The length of the data to process in \p data.
 *                          This must be no less than \p min_data_len and no
 *                          greater than \p max_data_len.
 * \param min_data_len      The minimal length of \p data in bytes.
 * \param max_data_len      The maximal length of \p data in bytes.
 * \param output            The HMAC will be written here. This must point to
 *                          a writable buffer of sufficient size to hold the
 *                          HMAC value.
 *
 * \retval 0
 *         Success.
 * \retval MBEDTLS_ERR_PLATFORM_HW_ACCEL_FAILED
 *         The hardware accelerator failed.
 */
int mbedtls_ssl_cf_hmac(mbedtls_md_context_t *ctx,
                        const unsigned char *add_data, size_t add_data_len,
                        const unsigned char *data, size_t data_len_secret,
                        size_t min_data_len, size_t max_data_len,
                        unsigned char *output);

/**
 * \brief Copy data from a secret position with constant flow.
 *
 * This function copies \p len bytes from \p src_base + \p offset_secret to \p
 * dst, with a code flow and memory access pattern that does not depend on \p
 * offset_secret, but only on \p offset_min, \p offset_max and \p len.
 *
 * \param dst           The destination buffer. This must point to a writable
 *                      buffer of at least \p len bytes.
 * \param src_base      The base of the source buffer. This must point to a
 *                      readable buffer of at least \p offset_max + \p len
 *                      bytes.
 * \param offset_secret The offset in the source buffer from which to copy.
 *                      This must be no less than \p offset_min and no greater
 *                      than \p offset_max.
 * \param offset_min    The minimal value of \p offset_secret.
 * \param offset_max    The maximal value of \p offset_secret.
 * \param len           The number of bytes to copy.
 */
void mbedtls_ssl_cf_memcpy_offset(unsigned char *dst,
                                  const unsigned char *src_base,
                                  size_t offset_secret, size_t offset_min,
                                  size_t offset_max, size_t len);

#endif /* MBEDTLS_TEST_HOOKS && MBEDTLS_SSL_SOME_SUITES_USE_TLS_CBC */
COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_INVASIVE_H_ */
