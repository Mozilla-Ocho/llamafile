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
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/ecdsa.h"
#include "third_party/mbedtls/ecp.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/pk.h"
#include "third_party/mbedtls/pk_internal.h"
#include "third_party/mbedtls/platform.h"
#include "third_party/mbedtls/rsa.h"
__static_yoink("mbedtls_notice");

/**
 * @fileoverview Public Key abstraction layer
 */

#if defined(MBEDTLS_PK_C)

#define PK_VALIDATE_RET( cond )    \
    MBEDTLS_INTERNAL_VALIDATE_RET( cond, MBEDTLS_ERR_PK_BAD_INPUT_DATA )
#define PK_VALIDATE( cond )        \
    MBEDTLS_INTERNAL_VALIDATE( cond )

/**
 * \brief           Initialize a #mbedtls_pk_context (as NONE).
 *
 * \param ctx       The context to initialize.
 *                  This must not be \c NULL.
 */
void mbedtls_pk_init( mbedtls_pk_context *ctx )
{
    PK_VALIDATE( ctx );
    ctx->pk_info = NULL;
    ctx->pk_ctx = NULL;
}

/**
 * \brief           Free the components of a #mbedtls_pk_context.
 *
 * \param ctx       The context to clear. It must have been initialized.
 *                  If this is \c NULL, this function does nothing.
 *
 * \note            For contexts that have been set up with
 *                  mbedtls_pk_setup_opaque(), this does not free the underlying
 *                  PSA key and you still need to call psa_destroy_key()
 *                  independently if you want to destroy that key.
 */
void mbedtls_pk_free( mbedtls_pk_context *ctx )
{
    if( ctx == NULL )
        return;
    if ( ctx->pk_info )
        ctx->pk_info->ctx_free_func( ctx->pk_ctx );
    mbedtls_platform_zeroize( ctx, sizeof( mbedtls_pk_context ) );
}

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
/**
 * \brief           Initialize a restart context
 *
 * \param ctx       The context to initialize.
 *                  This must not be \c NULL.
 */
void mbedtls_pk_restart_init( mbedtls_pk_restart_ctx *ctx )
{
    PK_VALIDATE( ctx );
    ctx->pk_info = NULL;
    ctx->rs_ctx = NULL;
}

/**
 * \brief           Free the components of a restart context
 *
 * \param ctx       The context to clear. It must have been initialized.
 *                  If this is \c NULL, this function does nothing.
 */
void mbedtls_pk_restart_free( mbedtls_pk_restart_ctx *ctx )
{
    if( ctx == NULL || ctx->pk_info == NULL ||
        ctx->pk_info->rs_free_func == NULL )
    {
        return;
    }
    ctx->pk_info->rs_free_func( ctx->rs_ctx );
    ctx->pk_info = NULL;
    ctx->rs_ctx = NULL;
}
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

/**
 * \brief           Return information associated with the given PK type
 *
 * \param pk_type   PK type to search for.
 *
 * \return          The PK info associated with the type or NULL if not found.
 */
const mbedtls_pk_info_t * mbedtls_pk_info_from_type( mbedtls_pk_type_t pk_type )
{
    switch( pk_type ) {
#if defined(MBEDTLS_RSA_C)
        case MBEDTLS_PK_RSA:
            return( &mbedtls_rsa_info );
#endif
#if defined(MBEDTLS_ECP_C)
        case MBEDTLS_PK_ECKEY:
            return( &mbedtls_eckey_info );
        case MBEDTLS_PK_ECKEY_DH:
            return( &mbedtls_eckeydh_info );
#endif
#if defined(MBEDTLS_ECDSA_C)
        case MBEDTLS_PK_ECDSA:
            return( &mbedtls_ecdsa_info );
#endif
        /* MBEDTLS_PK_RSA_ALT omitted on purpose */
        default:
            return( NULL );
    }
}

/**
 * \brief           Initialize a PK context with the information given
 *                  and allocates the type-specific PK subcontext.
 *
 * \param ctx       Context to initialize. It must not have been set
 *                  up yet (type #MBEDTLS_PK_NONE).
 * \param info      Information to use
 *
 * \return          0 on success,
 *                  MBEDTLS_ERR_PK_BAD_INPUT_DATA on invalid input,
 *                  MBEDTLS_ERR_PK_ALLOC_FAILED on allocation failure.
 *
 * \note            For contexts holding an RSA-alt key, use
 *                  \c mbedtls_pk_setup_rsa_alt() instead.
 */
int mbedtls_pk_setup( mbedtls_pk_context *ctx, const mbedtls_pk_info_t *info )
{
    PK_VALIDATE_RET( ctx );
    if( info == NULL || ctx->pk_info )
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );

    if( ( ctx->pk_ctx = info->ctx_alloc_func() ) == NULL )
        return( MBEDTLS_ERR_PK_ALLOC_FAILED );

    ctx->pk_info = info;

    return( 0 );
}

/**
 * \brief           Initialize an RSA-alt context
 *
 * \param ctx       Context to initialize. It must not have been set
 *                  up yet (type #MBEDTLS_PK_NONE).
 * \param key       RSA key pointer
 * \param decrypt_func  Decryption function
 * \param sign_func     Signing function
 * \param key_len_func  Function returning key length in bytes
 *
 * \return          0 on success, or MBEDTLS_ERR_PK_BAD_INPUT_DATA if the
 *                  context wasn't already initialized as RSA_ALT.
 *
 * \note            This function replaces \c mbedtls_pk_setup() for RSA-alt.
 */
int mbedtls_pk_setup_rsa_alt( mbedtls_pk_context *ctx, void * key,
                              mbedtls_pk_rsa_alt_decrypt_func decrypt_func,
                              mbedtls_pk_rsa_alt_sign_func sign_func,
                              mbedtls_pk_rsa_alt_key_len_func key_len_func )
{
    mbedtls_rsa_alt_context *rsa_alt;
    const mbedtls_pk_info_t *info = &mbedtls_rsa_alt_info;

    PK_VALIDATE_RET( ctx );
    if( ctx->pk_info )
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );

    if( ( ctx->pk_ctx = info->ctx_alloc_func() ) == NULL )
        return( MBEDTLS_ERR_PK_ALLOC_FAILED );

    ctx->pk_info = info;

    rsa_alt = (mbedtls_rsa_alt_context *) ctx->pk_ctx;

    rsa_alt->key = key;
    rsa_alt->decrypt_func = decrypt_func;
    rsa_alt->sign_func = sign_func;
    rsa_alt->key_len_func = key_len_func;

    return( 0 );
}

/**
 * \brief           Tell if a context can do the operation given by type
 *
 * \param ctx       The context to query. It must have been initialized.
 * \param type      The desired type.
 *
 * \return          1 if the context can do operations on the given type.
 * \return          0 if the context cannot do the operations on the given
 *                  type. This is always the case for a context that has
 *                  been initialized but not set up, or that has been
 *                  cleared with mbedtls_pk_free().
 */
int mbedtls_pk_can_do( const mbedtls_pk_context *ctx, mbedtls_pk_type_t type )
{
    /* A context with null pk_info is not set up yet and can't do anything.
     * For backward compatibility, also accept NULL instead of a context
     * pointer. */
    if( ctx == NULL || ctx->pk_info == NULL )
        return( 0 );
    return( ctx->pk_info->can_do( type ) );
}

/*
 * Helper for mbedtls_pk_sign and mbedtls_pk_verify
 */
static inline int pk_hashlen_helper( mbedtls_md_type_t md_alg, size_t *hash_len )
{
    const mbedtls_md_info_t *md_info;
    if( *hash_len != 0 )
        return( 0 );
    if( ( md_info = mbedtls_md_info_from_type( md_alg ) ) == NULL )
        return( -1 );
    *hash_len = mbedtls_md_get_size( md_info );
    return( 0 );
}

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
/*
 * Helper to set up a restart context if needed
 */
static int pk_restart_setup( mbedtls_pk_restart_ctx *ctx,
                             const mbedtls_pk_info_t *info )
{
    /* Don't do anything if already set up or invalid */
    if( ctx == NULL || ctx->pk_info )
        return( 0 );

    /* Should never happen when we're called */
    if( info->rs_alloc_func == NULL || info->rs_free_func == NULL )
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );

    if( ( ctx->rs_ctx = info->rs_alloc_func() ) == NULL )
        return( MBEDTLS_ERR_PK_ALLOC_FAILED );

    ctx->pk_info = info;

    return( 0 );
}
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

/**
 * \brief           Restartable version of \c mbedtls_pk_verify()
 *
 * \note            Performs the same job as \c mbedtls_pk_verify(), but can
 *                  return early and restart according to the limit set with
 *                  \c mbedtls_ecp_set_max_ops() to reduce blocking for ECC
 *                  operations. For RSA, same as \c mbedtls_pk_verify().
 *
 * \param ctx       The PK context to use. It must have been set up.
 * \param md_alg    Hash algorithm used (see notes)
 * \param hash      Hash of the message to sign
 * \param hash_len  Hash length or 0 (see notes)
 * \param sig       Signature to verify
 * \param sig_len   Signature length
 * \param rs_ctx    Restart context (NULL to disable restart)
 *
 * \return          See \c mbedtls_pk_verify(), or
 * \return          #MBEDTLS_ERR_ECP_IN_PROGRESS if maximum number of
 *                  operations was reached: see \c mbedtls_ecp_set_max_ops().
 */
int mbedtls_pk_verify_restartable( mbedtls_pk_context *ctx,
                                   mbedtls_md_type_t md_alg,
                                   const unsigned char *hash, size_t hash_len,
                                   const unsigned char *sig, size_t sig_len,
                                   mbedtls_pk_restart_ctx *rs_ctx )
{
    PK_VALIDATE_RET( ctx );
    PK_VALIDATE_RET( ( md_alg == MBEDTLS_MD_NONE && hash_len == 0 ) ||
                     hash );
    PK_VALIDATE_RET( sig );

    if( ctx->pk_info == NULL ||
        pk_hashlen_helper( md_alg, &hash_len ) != 0 )
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
    /* optimization: use non-restartable version if restart disabled */
    if( rs_ctx &&
        mbedtls_ecp_restart_is_enabled() &&
        ctx->pk_info->verify_rs_func )
    {
        int ret = MBEDTLS_ERR_THIS_CORRUPTION;

        if( ( ret = pk_restart_setup( rs_ctx, ctx->pk_info ) ) != 0 )
            return( ret );

        ret = ctx->pk_info->verify_rs_func( ctx->pk_ctx,
                   md_alg, hash, hash_len, sig, sig_len, rs_ctx->rs_ctx );

        if( ret != MBEDTLS_ERR_ECP_IN_PROGRESS )
            mbedtls_pk_restart_free( rs_ctx );

        return( ret );
    }
#else /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */
    (void) rs_ctx;
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

    if( ctx->pk_info->verify_func == NULL )
        return( MBEDTLS_ERR_PK_TYPE_MISMATCH );

    return( ctx->pk_info->verify_func( ctx->pk_ctx, md_alg, hash, hash_len,
                                       sig, sig_len ) );
}

/**
 * \brief           Verify signature (including padding if relevant).
 *
 * \param ctx       The PK context to use. It must have been set up.
 * \param md_alg    Hash algorithm used (see notes)
 * \param hash      Hash of the message to sign
 * \param hash_len  Hash length or 0 (see notes)
 * \param sig       Signature to verify
 * \param sig_len   Signature length
 *
 * \return          0 on success (signature is valid),
 *                  #MBEDTLS_ERR_PK_SIG_LEN_MISMATCH if there is a valid
 *                  signature in sig but its length is less than \p siglen,
 *                  or a specific error code.
 *
 * \note            For RSA keys, the default padding type is PKCS#1 v1.5.
 *                  Use \c mbedtls_pk_verify_ext( MBEDTLS_PK_RSASSA_PSS, ... )
 *                  to verify RSASSA_PSS signatures.
 *
 * \note            If hash_len is 0, then the length associated with md_alg
 *                  is used instead, or an error returned if it is invalid.
 *
 * \note            md_alg may be MBEDTLS_MD_NONE, only if hash_len != 0
 */
int mbedtls_pk_verify( mbedtls_pk_context *ctx, mbedtls_md_type_t md_alg,
                       const unsigned char *hash, size_t hash_len,
                       const unsigned char *sig, size_t sig_len )
{
    return( mbedtls_pk_verify_restartable( ctx, md_alg, hash, hash_len,
                                           sig, sig_len, NULL ) );
}

/**
 * \brief           Verify signature, with options.
 *                  (Includes verification of the padding depending on type.)
 *
 * \param type      Signature type (inc. possible padding type) to verify
 * \param options   Pointer to type-specific options, or NULL
 * \param ctx       The PK context to use. It must have been set up.
 * \param md_alg    Hash algorithm used (see notes)
 * \param hash      Hash of the message to sign
 * \param hash_len  Hash length or 0 (see notes)
 * \param sig       Signature to verify
 * \param sig_len   Signature length
 *
 * \return          0 on success (signature is valid),
 *                  #MBEDTLS_ERR_PK_TYPE_MISMATCH if the PK context can't be
 *                  used for this type of signatures,
 *                  #MBEDTLS_ERR_PK_SIG_LEN_MISMATCH if there is a valid
 *                  signature in sig but its length is less than \p siglen,
 *                  or a specific error code.
 *
 * \note            If hash_len is 0, then the length associated with md_alg
 *                  is used instead, or an error returned if it is invalid.
 *
 * \note            md_alg may be MBEDTLS_MD_NONE, only if hash_len != 0
 *
 * \note            If type is MBEDTLS_PK_RSASSA_PSS, then options must point
 *                  to a mbedtls_pk_rsassa_pss_options structure,
 *                  otherwise it must be NULL.
 */
int mbedtls_pk_verify_ext( mbedtls_pk_type_t type, const void *options,
                           mbedtls_pk_context *ctx, mbedtls_md_type_t md_alg,
                           const unsigned char *hash, size_t hash_len,
                           const unsigned char *sig, size_t sig_len )
{
  PK_VALIDATE_RET( ctx );
  PK_VALIDATE_RET( ( md_alg == MBEDTLS_MD_NONE && hash_len == 0 ) ||
                   hash );
  PK_VALIDATE_RET( sig );
  if( ctx->pk_info == NULL )
    return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );
  if( ! mbedtls_pk_can_do( ctx, type ) )
    return( MBEDTLS_ERR_PK_TYPE_MISMATCH );
  if( type == MBEDTLS_PK_RSASSA_PSS )
  {
#if defined(MBEDTLS_RSA_C) && defined(MBEDTLS_PKCS1_V21)
        int ret = MBEDTLS_ERR_THIS_CORRUPTION;
        const mbedtls_pk_rsassa_pss_options *pss_opts;
#if SIZE_MAX > UINT_MAX
        if( md_alg == MBEDTLS_MD_NONE && UINT_MAX < hash_len )
            return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );
#endif /* SIZE_MAX > UINT_MAX */
        if( options == NULL )
            return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );
        pss_opts = (const mbedtls_pk_rsassa_pss_options *) options;
        if( sig_len < mbedtls_pk_get_len( ctx ) )
            return( MBEDTLS_ERR_RSA_VERIFY_FAILED );
        ret = mbedtls_rsa_rsassa_pss_verify_ext( mbedtls_pk_rsa( *ctx ),
                NULL, NULL, MBEDTLS_RSA_PUBLIC,
                md_alg, (unsigned int) hash_len, hash,
                pss_opts->mgf1_hash_id,
                pss_opts->expected_salt_len,
                sig );
        if( ret != 0 )
            return( ret );
        if( sig_len > mbedtls_pk_get_len( ctx ) )
            return( MBEDTLS_ERR_PK_SIG_LEN_MISMATCH );
        return( 0 );
#else
        return( MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE );
#endif /* MBEDTLS_RSA_C && MBEDTLS_PKCS1_V21 */
    }
    /* General case: no options */
    if( options )
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );
    return( mbedtls_pk_verify( ctx, md_alg, hash, hash_len, sig, sig_len ) );
}

/**
 * \brief           Restartable version of \c mbedtls_pk_sign()
 *
 * \note            Performs the same job as \c mbedtls_pk_sign(), but can
 *                  return early and restart according to the limit set with
 *                  \c mbedtls_ecp_set_max_ops() to reduce blocking for ECC
 *                  operations. For RSA, same as \c mbedtls_pk_sign().
 *
 * \param ctx       The PK context to use. It must have been set up
 *                  with a private key.
 * \param md_alg    Hash algorithm used (see notes for mbedtls_pk_sign())
 * \param hash      Hash of the message to sign
 * \param hash_len  Hash length or 0 (see notes for mbedtls_pk_sign())
 * \param sig       Place to write the signature.
 *                  It must have enough room for the signature.
 *                  #MBEDTLS_PK_SIGNATURE_MAX_SIZE is always enough.
 *                  You may use a smaller buffer if it is large enough
 *                  given the key type.
 * \param sig_len   On successful return,
 *                  the number of bytes written to \p sig.
 * \param f_rng     RNG function
 * \param p_rng     RNG parameter
 * \param rs_ctx    Restart context (NULL to disable restart)
 *
 * \return          See \c mbedtls_pk_sign().
 * \return          #MBEDTLS_ERR_ECP_IN_PROGRESS if maximum number of
 *                  operations was reached: see \c mbedtls_ecp_set_max_ops().
 */
int mbedtls_pk_sign_restartable( mbedtls_pk_context *ctx,
                                 mbedtls_md_type_t md_alg,
                                 const unsigned char *hash, size_t hash_len,
                                 unsigned char *sig, size_t *sig_len,
                                 int (*f_rng)(void *, unsigned char *, size_t), 
                                 void *p_rng, mbedtls_pk_restart_ctx *rs_ctx )
{
    PK_VALIDATE_RET( ctx );
    PK_VALIDATE_RET( ( md_alg == MBEDTLS_MD_NONE && hash_len == 0 ) ||
                     hash );
    PK_VALIDATE_RET( sig );
    if( ctx->pk_info == NULL ||
        pk_hashlen_helper( md_alg, &hash_len ) != 0 )
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );
#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
    /* optimization: use non-restartable version if restart disabled */
    if( rs_ctx &&
        mbedtls_ecp_restart_is_enabled() &&
        ctx->pk_info->sign_rs_func )
    {
        int ret = MBEDTLS_ERR_THIS_CORRUPTION;
        if( ( ret = pk_restart_setup( rs_ctx, ctx->pk_info ) ) != 0 )
            return( ret );
        ret = ctx->pk_info->sign_rs_func( ctx->pk_ctx, md_alg,
                hash, hash_len, sig, sig_len, f_rng, p_rng, rs_ctx->rs_ctx );
        if( ret != MBEDTLS_ERR_ECP_IN_PROGRESS )
            mbedtls_pk_restart_free( rs_ctx );
        return( ret );
    }
#else /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */
    (void) rs_ctx;
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */
    if( ctx->pk_info->sign_func == NULL )
        return( MBEDTLS_ERR_PK_TYPE_MISMATCH );
    return( ctx->pk_info->sign_func( ctx->pk_ctx, md_alg, hash, hash_len,
                                     sig, sig_len, f_rng, p_rng ) );
}

/**
 * \brief           Make signature, including padding if relevant.
 *
 * \param ctx       The PK context to use. It must have been set up
 *                  with a private key.
 * \param md_alg    Hash algorithm used (see notes)
 * \param hash      Hash of the message to sign
 * \param hash_len  Hash length or 0 (see notes)
 * \param sig       Place to write the signature.
 *                  It must have enough room for the signature.
 *                  #MBEDTLS_PK_SIGNATURE_MAX_SIZE is always enough.
 *                  You may use a smaller buffer if it is large enough
 *                  given the key type.
 * \param sig_len   On successful return,
 *                  the number of bytes written to \p sig.
 * \param f_rng     RNG function
 * \param p_rng     RNG parameter
 *
 * \return          0 on success, or a specific error code.
 *
 * \note            For RSA keys, the default padding type is PKCS#1 v1.5.
 *                  There is no interface in the PK module to make RSASSA-PSS
 *                  signatures yet.
 *
 * \note            If hash_len is 0, then the length associated with md_alg
 *                  is used instead, or an error returned if it is invalid.
 *
 * \note            For RSA, md_alg may be MBEDTLS_MD_NONE if hash_len != 0.
 *                  For ECDSA, md_alg may never be MBEDTLS_MD_NONE.
 */
int mbedtls_pk_sign( mbedtls_pk_context *ctx, mbedtls_md_type_t md_alg,
                     const unsigned char *hash, size_t hash_len,
                     unsigned char *sig, size_t *sig_len,
                     int (*f_rng)(void *, unsigned char *, size_t), void *p_rng )
{
    return( mbedtls_pk_sign_restartable( ctx, md_alg, hash, hash_len,
                                         sig, sig_len, f_rng, p_rng, NULL ) );
}

/**
 * \brief           Decrypt message (including padding if relevant).
 *
 * \param ctx       The PK context to use. It must have been set up
 *                  with a private key.
 * \param input     Input to decrypt
 * \param ilen      Input size
 * \param output    Decrypted output
 * \param olen      Decrypted message length
 * \param osize     Size of the output buffer
 * \param f_rng     RNG function
 * \param p_rng     RNG parameter
 *
 * \note            For RSA keys, the default padding type is PKCS#1 v1.5.
 *
 * \return          0 on success, or a specific error code.
 */
int mbedtls_pk_decrypt( mbedtls_pk_context *ctx,
                        const unsigned char *input, size_t ilen,
                        unsigned char *output, size_t *olen, size_t osize,
                        int (*f_rng)(void *, unsigned char *, size_t), void *p_rng )
{
    PK_VALIDATE_RET( ctx );
    PK_VALIDATE_RET( input || ilen == 0 );
    PK_VALIDATE_RET( output || osize == 0 );
    PK_VALIDATE_RET( olen );
    if( ctx->pk_info == NULL )
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );
    if( ctx->pk_info->decrypt_func == NULL )
        return( MBEDTLS_ERR_PK_TYPE_MISMATCH );
    return( ctx->pk_info->decrypt_func( ctx->pk_ctx, input, ilen,
                output, olen, osize, f_rng, p_rng ) );
}

/**
 * \brief           Encrypt message (including padding if relevant).
 *
 * \param ctx       The PK context to use. It must have been set up.
 * \param input     Message to encrypt
 * \param ilen      Message size
 * \param output    Encrypted output
 * \param olen      Encrypted output length
 * \param osize     Size of the output buffer
 * \param f_rng     RNG function
 * \param p_rng     RNG parameter
 *
 * \note            For RSA keys, the default padding type is PKCS#1 v1.5.
 *
 * \return          0 on success, or a specific error code.
 */
int mbedtls_pk_encrypt( mbedtls_pk_context *ctx,
                        const unsigned char *input, size_t ilen,
                        unsigned char *output, size_t *olen, size_t osize,
                        int (*f_rng)(void *, unsigned char *, size_t), void *p_rng )
{
    PK_VALIDATE_RET( ctx );
    PK_VALIDATE_RET( input || ilen == 0 );
    PK_VALIDATE_RET( output || osize == 0 );
    PK_VALIDATE_RET( olen );
    if( ctx->pk_info == NULL )
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );
    if( ctx->pk_info->encrypt_func == NULL )
        return( MBEDTLS_ERR_PK_TYPE_MISMATCH );
    return( ctx->pk_info->encrypt_func( ctx->pk_ctx, input, ilen,
                output, olen, osize, f_rng, p_rng ) );
}

/**
 * \brief           Check if a public-private pair of keys matches.
 *
 * \param pub       Context holding a public key.
 * \param prv       Context holding a private (and public) key.
 *
 * \return          \c 0 on success (keys were checked and match each other).

 * \return          #MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE if the keys could not
 *                  be checked - in that case they may or may not match.
 * \return          #MBEDTLS_ERR_PK_BAD_INPUT_DATA if a context is invalid.
 * \return          Another non-zero value if the keys do not match.
 */
int mbedtls_pk_check_pair( const mbedtls_pk_context *pub, const mbedtls_pk_context *prv )
{
    PK_VALIDATE_RET( pub );
    PK_VALIDATE_RET( prv );
    if( pub->pk_info == NULL ||
        prv->pk_info == NULL )
    {
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );
    }
    if( prv->pk_info->check_pair_func == NULL )
        return( MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE );
    if( prv->pk_info->type == MBEDTLS_PK_RSA_ALT )
    {
        if( pub->pk_info->type != MBEDTLS_PK_RSA )
            return( MBEDTLS_ERR_PK_TYPE_MISMATCH );
    }
    else
    {
        if( pub->pk_info != prv->pk_info )
            return( MBEDTLS_ERR_PK_TYPE_MISMATCH );
    }
    return( prv->pk_info->check_pair_func( pub->pk_ctx, prv->pk_ctx ) );
}

/**
 * \brief           Get the size in bits of the underlying key
 *
 * \param ctx       The context to query. It must have been initialized.
 *
 * \return          Key size in bits, or 0 on error
 */
size_t mbedtls_pk_get_bitlen( const mbedtls_pk_context *ctx )
{
    /* For backward compatibility, accept NULL or a context that
     * isn't set up yet, and return a fake value that should be safe. */
    if( ctx == NULL || ctx->pk_info == NULL )
        return( 0 );
    return( ctx->pk_info->get_bitlen( ctx->pk_ctx ) );
}

/**
 * \brief           Export debug information
 *
 * \param ctx       The PK context to use. It must have been initialized.
 * \param items     Place to write debug items
 *
 * \return          0 on success or MBEDTLS_ERR_PK_BAD_INPUT_DATA
 */
int mbedtls_pk_debug( const mbedtls_pk_context *ctx, mbedtls_pk_debug_item *items )
{
    PK_VALIDATE_RET( ctx );
    if( ctx->pk_info == NULL )
        return( MBEDTLS_ERR_PK_BAD_INPUT_DATA );
    if( ctx->pk_info->debug_func == NULL )
        return( MBEDTLS_ERR_PK_TYPE_MISMATCH );
    ctx->pk_info->debug_func( ctx->pk_ctx, items );
    return( 0 );
}

/**
 * \brief           Access the type name
 *
 * \param ctx       The PK context to use. It must have been initialized.
 *
 * \return          Type name on success, or "invalid PK"
 */
const char *mbedtls_pk_get_name( const mbedtls_pk_context *ctx )
{
    if( ctx == NULL || ctx->pk_info == NULL )
        return( "invalid PK" );

    return( ctx->pk_info->name );
}

/**
 * \brief           Get the key type
 *
 * \param ctx       The PK context to use. It must have been initialized.
 *
 * \return          Type on success.
 * \return          #MBEDTLS_PK_NONE for a context that has not been set up.
 */
mbedtls_pk_type_t mbedtls_pk_get_type( const mbedtls_pk_context *ctx )
{
    if( ctx == NULL || ctx->pk_info == NULL )
        return( MBEDTLS_PK_NONE );

    return( ctx->pk_info->type );
}

#endif /* MBEDTLS_PK_C */
