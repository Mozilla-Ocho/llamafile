/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;tab-width:4;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright The Mbed TLS Contributors                                          │
│                                                                              │
│ Licensed under the Apache License, Version 2.0 (the "License");              │
│ you may not use this file except in compliance with the License.             │
│ You may obtain a copy of the License at                                      │
│                                                                              │
│     http://www.apache.org/licenses/LICENSE-2.0                               │
│                                                                              │
│ Unless required by applicable law or agreed to in writing, software          │
│ distributed under the License is distributed on an "AS IS" BASIS,            │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
╚─────────────────────────────────────────────────────────────────────────────*/
#include "third_party/mbedtls/ecdh_everest.h"
#include <libc/str/str.h>
#include "third_party/mbedtls/everest.h"
#if defined(MBEDTLS_ECDH_C) && defined(MBEDTLS_ECDH_VARIANT_EVEREST_ENABLED)
#define KEYSIZE 32
__static_yoink("mbedtls_notice");

/**
 * \brief           This function sets up the ECDH context with the information
 *                  given.
 *
 *                  This function should be called after mbedtls_ecdh_init() but
 *                  before mbedtls_ecdh_make_params(). There is no need to call
 *                  this function before mbedtls_ecdh_read_params().
 *
 *                  This is the first function used by a TLS server for
 *                  ECDHE ciphersuites.
 *
 * \param ctx       The ECDH context to set up.
 * \param grp_id    The group id of the group to set up the context for.
 *
 * \return          \c 0 on success.
 */
int mbedtls_everest_setup(mbedtls_ecdh_context_everest *ctx, int grp_id)
{
  if (grp_id != MBEDTLS_ECP_DP_CURVE25519)
    return MBEDTLS_ERR_ECP_BAD_INPUT_DATA;
  mbedtls_platform_zeroize(ctx, sizeof(*ctx));
  return 0;
}

/**
 * \brief           This function frees a context.
 *
 * \param ctx       The context to free.
 */
void mbedtls_everest_free(mbedtls_ecdh_context_everest *ctx)
{
  if (!ctx) return;
  mbedtls_platform_zeroize(ctx, sizeof(*ctx));
}

/**
 * \brief           This function generates a public key and a TLS
 *                  ServerKeyExchange payload.
 *
 *                  This is the second function used by a TLS server for ECDHE
 *                  ciphersuites. (It is called after mbedtls_ecdh_setup().)
 *
 * \note            This function assumes that the ECP group (grp) of the
 *                  \p ctx context has already been properly set,
 *                  for example, using mbedtls_ecp_group_load().
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context.
 * \param olen      The number of characters written.
 * \param buf       The destination buffer.
 * \param blen      The length of the destination buffer.
 * \param f_rng     The RNG function.
 * \param p_rng     The RNG context.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_everest_make_params(mbedtls_ecdh_context_everest *ctx, size_t *olen,
                                unsigned char *buf, size_t blen,
                                int (*f_rng)(void *, unsigned char *, size_t),
                                void *p_rng)
{
  int ret = 0;
  uint8_t base[KEYSIZE] = {9};
  if ((ret = f_rng(p_rng, ctx->our_secret, KEYSIZE)) != 0) return ret;
  *olen = KEYSIZE + 4;
  if (blen < *olen) return MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL;
  *buf++ = MBEDTLS_ECP_TLS_NAMED_CURVE;
  *buf++ = MBEDTLS_ECP_TLS_CURVE25519 >> 8;
  *buf++ = MBEDTLS_ECP_TLS_CURVE25519 & 0xFF;
  *buf++ = KEYSIZE;
  curve25519(buf, ctx->our_secret, base);
  base[0] = 0;
  if (!timingsafe_bcmp(buf, base, KEYSIZE))
    return MBEDTLS_ERR_ECP_RANDOM_FAILED;
  return 0;
}

/**
 * \brief           This function parses and processes a TLS ServerKeyExhange
 *                  payload.
 *
 *                  This is the first function used by a TLS client for ECDHE
 *                  ciphersuites.
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context.
 * \param buf       The pointer to the start of the input buffer.
 * \param end       The address for one Byte past the end of the buffer.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_everest_read_params(mbedtls_ecdh_context_everest *ctx,
                                const unsigned char **buf,
                                const unsigned char *end)
{
  if (end - *buf < KEYSIZE + 1) return MBEDTLS_ERR_ECP_BAD_INPUT_DATA;
  if ((*(*buf)++ != KEYSIZE)) return MBEDTLS_ERR_ECP_BAD_INPUT_DATA;
  memcpy(ctx->peer_point, *buf, KEYSIZE);
  *buf += KEYSIZE;
  return 0;
}

/**
 * \brief           This function sets up an ECDH context from an EC key.
 *
 *                  It is used by clients and servers in place of the
 *                  ServerKeyEchange for static ECDH, and imports ECDH
 *                  parameters from the EC key information of a certificate.
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context to set up.
 * \param key       The EC key to use.
 * \param side      Defines the source of the key: 1: Our key, or
 *                  0: The key of the peer.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_everest_get_params(mbedtls_ecdh_context_everest *ctx,
                               const mbedtls_ecp_keypair *key,
                               mbedtls_everest_ecdh_side side)
{
  size_t olen = 0;
  switch (side)
  {
    case MBEDTLS_EVEREST_ECDH_THEIRS:
      return mbedtls_ecp_point_write_binary(&key->grp, &key->Q,
                                            MBEDTLS_ECP_PF_COMPRESSED, &olen,
                                            ctx->peer_point, KEYSIZE);
    case MBEDTLS_EVEREST_ECDH_OURS:
      return mbedtls_mpi_write_binary_le(&key->d, ctx->our_secret, KEYSIZE);
    default:
      return MBEDTLS_ERR_ECP_BAD_INPUT_DATA;
  }
}

/**
 * \brief           This function generates a public key and a TLS
 *                  ClientKeyExchange payload.
 *
 *                  This is the second function used by a TLS client for ECDH(E)
 *                  ciphersuites.
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context.
 * \param olen      The number of Bytes written.
 * \param buf       The destination buffer.
 * \param blen      The size of the destination buffer.
 * \param f_rng     The RNG function.
 * \param p_rng     The RNG context.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_everest_make_public(mbedtls_ecdh_context_everest *ctx, size_t *olen,
                                unsigned char *buf, size_t blen,
                                int (*f_rng)(void *, unsigned char *, size_t),
                                void *p_rng)
{
  int ret = 0;
  unsigned char base[KEYSIZE] = {9};
  if (!ctx) return MBEDTLS_ERR_ECP_BAD_INPUT_DATA;
  if ((ret = f_rng(p_rng, ctx->our_secret, KEYSIZE))) return ret;
  *olen = KEYSIZE + 1;
  if (blen < *olen) return MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL;
  *buf++ = KEYSIZE;
  curve25519(buf, ctx->our_secret, base);
  base[0] = 0;
  if (!timingsafe_bcmp(buf, base, KEYSIZE))
    return MBEDTLS_ERR_ECP_RANDOM_FAILED;
  return ret;
}

/**
 * \brief       This function parses and processes a TLS ClientKeyExchange
 *              payload.
 *
 *              This is the third function used by a TLS server for ECDH(E)
 *              ciphersuites. (It is called after mbedtls_ecdh_setup() and
 *              mbedtls_ecdh_make_params().)
 *
 * \see         ecp.h
 *
 * \param ctx   The ECDH context.
 * \param buf   The start of the input buffer.
 * \param blen  The length of the input buffer.
 *
 * \return      \c 0 on success.
 * \return      An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_everest_read_public(mbedtls_ecdh_context_everest *ctx,
                                const unsigned char *buf, size_t blen)
{
  if (blen < KEYSIZE + 1) return MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL;
  if ((*buf++ != KEYSIZE)) return MBEDTLS_ERR_ECP_BAD_INPUT_DATA;
  memcpy(ctx->peer_point, buf, KEYSIZE);
  return 0;
}

/**
 * \brief           This function derives and exports the shared secret.
 *
 *                  This is the last function used by both TLS client
 *                  and servers.
 *
 * \note            If \p f_rng is not NULL, it is used to implement
 *                  countermeasures against side-channel attacks.
 *                  For more information, see mbedtls_ecp_mul().
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context.
 * \param olen      The number of Bytes written.
 * \param buf       The destination buffer.
 * \param blen      The length of the destination buffer.
 * \param f_rng     The RNG function.
 * \param p_rng     The RNG context.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_everest_calc_secret(mbedtls_ecdh_context_everest *ctx, size_t *olen,
                                unsigned char *buf, size_t blen,
                                int (*f_rng)(void *, unsigned char *, size_t),
                                void *p_rng)
{
  /* f_rng and p_rng are not used here because this implementation does not
     need blinding since it has constant trace. (todo(jart): wut?) */
  *olen = KEYSIZE;
  if (blen < *olen) return MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL;
  curve25519(buf, ctx->our_secret, ctx->peer_point);
  if (!timingsafe_bcmp(buf, ctx->our_secret, KEYSIZE)) goto wut;
  /* Wipe the DH secret and don't let the peer chose a small subgroup point */
  mbedtls_platform_zeroize(ctx->our_secret, KEYSIZE);
  if (!timingsafe_bcmp(buf, ctx->our_secret, KEYSIZE)) goto wut;
  return 0;
wut:
  mbedtls_platform_zeroize(buf, KEYSIZE);
  mbedtls_platform_zeroize(ctx->our_secret, KEYSIZE);
  return MBEDTLS_ERR_ECP_RANDOM_FAILED;
}

#endif
