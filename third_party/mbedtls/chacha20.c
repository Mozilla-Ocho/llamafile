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
#include "third_party/mbedtls/chacha20.h"
#include <libc/serialize.h>
#include <libc/stdio/stdio.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/platform.h"
__static_yoink("mbedtls_notice");

/* Parameter validation macros */
#define CHACHA20_VALIDATE_RET( cond )                                       \
    MBEDTLS_INTERNAL_VALIDATE_RET( cond, MBEDTLS_ERR_CHACHA20_BAD_INPUT_DATA )
#define CHACHA20_VALIDATE( cond )                                           \
    MBEDTLS_INTERNAL_VALIDATE( cond )

#define BYTES_TO_U32_LE( data, offset ) READ32LE((data) + (offset))

#define ROTL32( value, amount ) \
    ( (uint32_t) ( (value) << (amount) ) | ( (value) >> ( 32 - (amount) ) ) )

#define CHACHA20_CTR_INDEX ( 12U )

#define CHACHA20_BLOCK_SIZE_BYTES ( 4U * 16U )

/**
 * \brief               Generates a keystream block.
 *
 * \param s             The initial ChaCha20 state (key, nonce, counter).
 * \param k             Generated keystream bytes are written to this buffer.
 */
static void chacha20_block( const uint32_t s[16], unsigned char k[64] )
{
    int i;
    uint8_t *p;
    uint32_t A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P;
    A = s[ 0];
    B = s[ 1];
    C = s[ 2];
    D = s[ 3];
    E = s[ 4];
    F = s[ 5];
    G = s[ 6];
    H = s[ 7];
    I = s[ 8];
    J = s[ 9];
    K = s[10];
    L = s[11];
    M = s[12];
    N = s[13];
    O = s[14];
    P = s[15];
    for (i = 0; i < 10; ++i) {
        A += E; M = ROTL32(M ^ A, 16);
        B += F; N = ROTL32(N ^ B, 16);
        C += G; O = ROTL32(O ^ C, 16);
        D += H; P = ROTL32(P ^ D, 16);
        I += M; E = ROTL32(E ^ I, 12);
        J += N; F = ROTL32(F ^ J, 12);
        K += O; G = ROTL32(G ^ K, 12);
        L += P; H = ROTL32(H ^ L, 12);
        A += E; M = ROTL32(M ^ A,  8);
        B += F; N = ROTL32(N ^ B,  8);
        C += G; O = ROTL32(O ^ C,  8);
        D += H; P = ROTL32(P ^ D,  8);
        I += M; E = ROTL32(E ^ I,  7);
        J += N; F = ROTL32(F ^ J,  7);
        K += O; G = ROTL32(G ^ K,  7);
        L += P; H = ROTL32(H ^ L,  7);
        A += F; P = ROTL32(P ^ A, 16);
        B += G; M = ROTL32(M ^ B, 16);
        C += H; N = ROTL32(N ^ C, 16);
        D += E; O = ROTL32(O ^ D, 16);
        K += P; F = ROTL32(F ^ K, 12);
        L += M; G = ROTL32(G ^ L, 12);
        I += N; H = ROTL32(H ^ I, 12);
        J += O; E = ROTL32(E ^ J, 12);
        A += F; P = ROTL32(P ^ A,  8);
        B += G; M = ROTL32(M ^ B,  8);
        C += H; N = ROTL32(N ^ C,  8);
        D += E; O = ROTL32(O ^ D,  8);
        K += P; F = ROTL32(F ^ K,  7);
        L += M; G = ROTL32(G ^ L,  7);
        I += N; H = ROTL32(H ^ I,  7);
        J += O; E = ROTL32(E ^ J,  7);
    }
    p = k;
    A += s[ 0]; p = WRITE32LE(p, A);
    B += s[ 1]; p = WRITE32LE(p, B);
    C += s[ 2]; p = WRITE32LE(p, C);
    D += s[ 3]; p = WRITE32LE(p, D);
    E += s[ 4]; p = WRITE32LE(p, E);
    F += s[ 5]; p = WRITE32LE(p, F);
    G += s[ 6]; p = WRITE32LE(p, G);
    H += s[ 7]; p = WRITE32LE(p, H);
    I += s[ 8]; p = WRITE32LE(p, I);
    J += s[ 9]; p = WRITE32LE(p, J);
    K += s[10]; p = WRITE32LE(p, K);
    L += s[11]; p = WRITE32LE(p, L);
    M += s[12]; p = WRITE32LE(p, M);
    N += s[13]; p = WRITE32LE(p, N);
    O += s[14]; p = WRITE32LE(p, O);
    P += s[15]; p = WRITE32LE(p, P);
}

/**
 * \brief           This function initializes the specified ChaCha20 context.
 *
 *                  It must be the first API called before using
 *                  the context.
 *
 *                  It is usually followed by calls to
 *                  \c mbedtls_chacha20_setkey() and
 *                  \c mbedtls_chacha20_starts(), then one or more calls to
 *                  to \c mbedtls_chacha20_update(), and finally to
 *                  \c mbedtls_chacha20_free().
 *
 * \param ctx       The ChaCha20 context to initialize.
 *                  This must not be \c NULL.
 */
void mbedtls_chacha20_init( mbedtls_chacha20_context *ctx )
{
    CHACHA20_VALIDATE( ctx != NULL );
    mbedtls_platform_zeroize( ctx->state, sizeof( ctx->state ) );
    mbedtls_platform_zeroize( ctx->keystream8, sizeof( ctx->keystream8 ) );
    /* Initially, there's no keystream bytes available */
    ctx->keystream_bytes_used = CHACHA20_BLOCK_SIZE_BYTES;
}

/**
 * \brief           This function releases and clears the specified
 *                  ChaCha20 context.
 *
 * \param ctx       The ChaCha20 context to clear. This may be \c NULL,
 *                  in which case this function is a no-op. If it is not
 *                  \c NULL, it must point to an initialized context.
 *
 */
void mbedtls_chacha20_free( mbedtls_chacha20_context *ctx )
{
    if( ctx != NULL )
    {
        mbedtls_platform_zeroize( ctx, sizeof( mbedtls_chacha20_context ) );
    }
}

/**
 * \brief           This function sets the encryption/decryption key.
 *
 * \note            After using this function, you must also call
 *                  \c mbedtls_chacha20_starts() to set a nonce before you
 *                  start encrypting/decrypting data with
 *                  \c mbedtls_chacha_update().
 *
 * \param ctx       The ChaCha20 context to which the key should be bound.
 *                  It must be initialized.
 * \param key       The encryption/decryption key. This must be \c 32 Bytes
 *                  in length.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_CHACHA20_BAD_INPUT_DATA if ctx or key is NULL.
 */
int mbedtls_chacha20_setkey( mbedtls_chacha20_context *ctx,
                             const unsigned char key[32] )
{
    CHACHA20_VALIDATE_RET( ctx != NULL );
    CHACHA20_VALIDATE_RET( key != NULL );

    /* ChaCha20 constants - the string "expand 32-byte k" */
    ctx->state[0] = 0x61707865;
    ctx->state[1] = 0x3320646e;
    ctx->state[2] = 0x79622d32;
    ctx->state[3] = 0x6b206574;

    /* Set key */
    ctx->state[4]  = BYTES_TO_U32_LE( key, 0 );
    ctx->state[5]  = BYTES_TO_U32_LE( key, 4 );
    ctx->state[6]  = BYTES_TO_U32_LE( key, 8 );
    ctx->state[7]  = BYTES_TO_U32_LE( key, 12 );
    ctx->state[8]  = BYTES_TO_U32_LE( key, 16 );
    ctx->state[9]  = BYTES_TO_U32_LE( key, 20 );
    ctx->state[10] = BYTES_TO_U32_LE( key, 24 );
    ctx->state[11] = BYTES_TO_U32_LE( key, 28 );

    return( 0 );
}

/**
 * \brief           This function sets the nonce and initial counter value.
 *
 * \note            A ChaCha20 context can be re-used with the same key by
 *                  calling this function to change the nonce.
 *
 * \warning         You must never use the same nonce twice with the same key.
 *                  This would void any confidentiality guarantees for the
 *                  messages encrypted with the same nonce and key.
 *
 * \param ctx       The ChaCha20 context to which the nonce should be bound.
 *                  It must be initialized and bound to a key.
 * \param nonce     The nonce. This must be \c 12 Bytes in size.
 * \param counter   The initial counter value. This is usually \c 0.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_CHACHA20_BAD_INPUT_DATA if ctx or nonce is
 *                  NULL.
 */
int mbedtls_chacha20_starts( mbedtls_chacha20_context* ctx,
                             const unsigned char nonce[12],
                             uint32_t counter )
{
    CHACHA20_VALIDATE_RET( ctx != NULL );
    CHACHA20_VALIDATE_RET( nonce != NULL );

    /* Counter */
    ctx->state[12] = counter;

    /* Nonce */
    ctx->state[13] = BYTES_TO_U32_LE( nonce, 0 );
    ctx->state[14] = BYTES_TO_U32_LE( nonce, 4 );
    ctx->state[15] = BYTES_TO_U32_LE( nonce, 8 );

    mbedtls_platform_zeroize( ctx->keystream8, sizeof( ctx->keystream8 ) );

    /* Initially, there's no keystream bytes available */
    ctx->keystream_bytes_used = CHACHA20_BLOCK_SIZE_BYTES;

    return( 0 );
}

/**
 * \brief           This function encrypts or decrypts data.
 *
 *                  Since ChaCha20 is a stream cipher, the same operation is
 *                  used for encrypting and decrypting data.
 *
 * \note            The \p input and \p output pointers must either be equal or
 *                  point to non-overlapping buffers.
 *
 * \note            \c mbedtls_chacha20_setkey() and
 *                  \c mbedtls_chacha20_starts() must be called at least once
 *                  to setup the context before this function can be called.
 *
 * \note            This function can be called multiple times in a row in
 *                  order to encrypt of decrypt data piecewise with the same
 *                  key and nonce.
 *
 * \param ctx       The ChaCha20 context to use for encryption or decryption.
 *                  It must be initialized and bound to a key and nonce.
 * \param size      The length of the input data in Bytes.
 * \param input     The buffer holding the input data.
 *                  This pointer can be \c NULL if `size == 0`.
 * \param output    The buffer holding the output data.
 *                  This must be able to hold \p size Bytes.
 *                  This pointer can be \c NULL if `size == 0`.
 *
 * \return          \c 0 on success.
 * \return          A negative error code on failure.
 */
int mbedtls_chacha20_update( mbedtls_chacha20_context *ctx,
                             size_t size,
                             const unsigned char *input,
                             unsigned char *output )
{
    size_t offset = 0U;
    size_t i;

    CHACHA20_VALIDATE_RET( ctx != NULL );
    CHACHA20_VALIDATE_RET( size == 0 || input  != NULL );
    CHACHA20_VALIDATE_RET( size == 0 || output != NULL );

    /* Use leftover keystream bytes, if available */
    while( size > 0U && ctx->keystream_bytes_used < CHACHA20_BLOCK_SIZE_BYTES )
    {
        output[offset] = input[offset]
                       ^ ctx->keystream8[ctx->keystream_bytes_used];

        ctx->keystream_bytes_used++;
        offset++;
        size--;
    }

    /* Process full blocks */
    while( size >= CHACHA20_BLOCK_SIZE_BYTES )
    {
        /* Generate new keystream block and increment counter */
        chacha20_block( ctx->state, ctx->keystream8 );
        ctx->state[CHACHA20_CTR_INDEX]++;

        for( i = 0U; i < 64U; i += 8U )
        {
            output[offset + i  ] = input[offset + i  ] ^ ctx->keystream8[i  ];
            output[offset + i+1] = input[offset + i+1] ^ ctx->keystream8[i+1];
            output[offset + i+2] = input[offset + i+2] ^ ctx->keystream8[i+2];
            output[offset + i+3] = input[offset + i+3] ^ ctx->keystream8[i+3];
            output[offset + i+4] = input[offset + i+4] ^ ctx->keystream8[i+4];
            output[offset + i+5] = input[offset + i+5] ^ ctx->keystream8[i+5];
            output[offset + i+6] = input[offset + i+6] ^ ctx->keystream8[i+6];
            output[offset + i+7] = input[offset + i+7] ^ ctx->keystream8[i+7];
        }

        offset += CHACHA20_BLOCK_SIZE_BYTES;
        size   -= CHACHA20_BLOCK_SIZE_BYTES;
    }

    /* Last (partial) block */
    if( size > 0U )
    {
        /* Generate new keystream block and increment counter */
        chacha20_block( ctx->state, ctx->keystream8 );
        ctx->state[CHACHA20_CTR_INDEX]++;

        for( i = 0U; i < size; i++)
        {
            output[offset + i] = input[offset + i] ^ ctx->keystream8[i];
        }

        ctx->keystream_bytes_used = size;

    }

    return( 0 );
}

/**
 * \brief           This function encrypts or decrypts data with ChaCha20 and
 *                  the given key and nonce.
 *
 *                  Since ChaCha20 is a stream cipher, the same operation is
 *                  used for encrypting and decrypting data.
 *
 * \warning         You must never use the same (key, nonce) pair more than
 *                  once. This would void any confidentiality guarantees for
 *                  the messages encrypted with the same nonce and key.
 *
 * \note            The \p input and \p output pointers must either be equal or
 *                  point to non-overlapping buffers.
 *
 * \param key       The encryption/decryption key.
 *                  This must be \c 32 Bytes in length.
 * \param nonce     The nonce. This must be \c 12 Bytes in size.
 * \param counter   The initial counter value. This is usually \c 0.
 * \param size      The length of the input data in Bytes.
 * \param input     The buffer holding the input data.
 *                  This pointer can be \c NULL if `size == 0`.
 * \param output    The buffer holding the output data.
 *                  This must be able to hold \p size Bytes.
 *                  This pointer can be \c NULL if `size == 0`.
 *
 * \return          \c 0 on success.
 * \return          A negative error code on failure.
 */
int mbedtls_chacha20_crypt( const unsigned char key[32],
                            const unsigned char nonce[12],
                            uint32_t counter,
                            size_t data_len,
                            const unsigned char* input,
                            unsigned char* output )
{
    mbedtls_chacha20_context ctx;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;

    CHACHA20_VALIDATE_RET( key != NULL );
    CHACHA20_VALIDATE_RET( nonce != NULL );
    CHACHA20_VALIDATE_RET( data_len == 0 || input  != NULL );
    CHACHA20_VALIDATE_RET( data_len == 0 || output != NULL );

    mbedtls_chacha20_init( &ctx );

    ret = mbedtls_chacha20_setkey( &ctx, key );
    if( ret != 0 )
        goto cleanup;

    ret = mbedtls_chacha20_starts( &ctx, nonce, counter );
    if( ret != 0 )
        goto cleanup;

    ret = mbedtls_chacha20_update( &ctx, data_len, input, output );

cleanup:
    mbedtls_chacha20_free( &ctx );
    return( ret );
}

#if defined(MBEDTLS_SELF_TEST)

static const unsigned char test_keys[2][32] =
{
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01
    }
};

static const unsigned char test_nonces[2][12] =
{
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x02
    }
};

static const uint32_t test_counters[2] =
{
    0U,
    1U
};

static const unsigned char test_input[2][375] =
{
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    },
    {
        0x41, 0x6e, 0x79, 0x20, 0x73, 0x75, 0x62, 0x6d,
        0x69, 0x73, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x74,
        0x6f, 0x20, 0x74, 0x68, 0x65, 0x20, 0x49, 0x45,
        0x54, 0x46, 0x20, 0x69, 0x6e, 0x74, 0x65, 0x6e,
        0x64, 0x65, 0x64, 0x20, 0x62, 0x79, 0x20, 0x74,
        0x68, 0x65, 0x20, 0x43, 0x6f, 0x6e, 0x74, 0x72,
        0x69, 0x62, 0x75, 0x74, 0x6f, 0x72, 0x20, 0x66,
        0x6f, 0x72, 0x20, 0x70, 0x75, 0x62, 0x6c, 0x69,
        0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x20, 0x61,
        0x73, 0x20, 0x61, 0x6c, 0x6c, 0x20, 0x6f, 0x72,
        0x20, 0x70, 0x61, 0x72, 0x74, 0x20, 0x6f, 0x66,
        0x20, 0x61, 0x6e, 0x20, 0x49, 0x45, 0x54, 0x46,
        0x20, 0x49, 0x6e, 0x74, 0x65, 0x72, 0x6e, 0x65,
        0x74, 0x2d, 0x44, 0x72, 0x61, 0x66, 0x74, 0x20,
        0x6f, 0x72, 0x20, 0x52, 0x46, 0x43, 0x20, 0x61,
        0x6e, 0x64, 0x20, 0x61, 0x6e, 0x79, 0x20, 0x73,
        0x74, 0x61, 0x74, 0x65, 0x6d, 0x65, 0x6e, 0x74,
        0x20, 0x6d, 0x61, 0x64, 0x65, 0x20, 0x77, 0x69,
        0x74, 0x68, 0x69, 0x6e, 0x20, 0x74, 0x68, 0x65,
        0x20, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x78, 0x74,
        0x20, 0x6f, 0x66, 0x20, 0x61, 0x6e, 0x20, 0x49,
        0x45, 0x54, 0x46, 0x20, 0x61, 0x63, 0x74, 0x69,
        0x76, 0x69, 0x74, 0x79, 0x20, 0x69, 0x73, 0x20,
        0x63, 0x6f, 0x6e, 0x73, 0x69, 0x64, 0x65, 0x72,
        0x65, 0x64, 0x20, 0x61, 0x6e, 0x20, 0x22, 0x49,
        0x45, 0x54, 0x46, 0x20, 0x43, 0x6f, 0x6e, 0x74,
        0x72, 0x69, 0x62, 0x75, 0x74, 0x69, 0x6f, 0x6e,
        0x22, 0x2e, 0x20, 0x53, 0x75, 0x63, 0x68, 0x20,
        0x73, 0x74, 0x61, 0x74, 0x65, 0x6d, 0x65, 0x6e,
        0x74, 0x73, 0x20, 0x69, 0x6e, 0x63, 0x6c, 0x75,
        0x64, 0x65, 0x20, 0x6f, 0x72, 0x61, 0x6c, 0x20,
        0x73, 0x74, 0x61, 0x74, 0x65, 0x6d, 0x65, 0x6e,
        0x74, 0x73, 0x20, 0x69, 0x6e, 0x20, 0x49, 0x45,
        0x54, 0x46, 0x20, 0x73, 0x65, 0x73, 0x73, 0x69,
        0x6f, 0x6e, 0x73, 0x2c, 0x20, 0x61, 0x73, 0x20,
        0x77, 0x65, 0x6c, 0x6c, 0x20, 0x61, 0x73, 0x20,
        0x77, 0x72, 0x69, 0x74, 0x74, 0x65, 0x6e, 0x20,
        0x61, 0x6e, 0x64, 0x20, 0x65, 0x6c, 0x65, 0x63,
        0x74, 0x72, 0x6f, 0x6e, 0x69, 0x63, 0x20, 0x63,
        0x6f, 0x6d, 0x6d, 0x75, 0x6e, 0x69, 0x63, 0x61,
        0x74, 0x69, 0x6f, 0x6e, 0x73, 0x20, 0x6d, 0x61,
        0x64, 0x65, 0x20, 0x61, 0x74, 0x20, 0x61, 0x6e,
        0x79, 0x20, 0x74, 0x69, 0x6d, 0x65, 0x20, 0x6f,
        0x72, 0x20, 0x70, 0x6c, 0x61, 0x63, 0x65, 0x2c,
        0x20, 0x77, 0x68, 0x69, 0x63, 0x68, 0x20, 0x61,
        0x72, 0x65, 0x20, 0x61, 0x64, 0x64, 0x72, 0x65,
        0x73, 0x73, 0x65, 0x64, 0x20, 0x74, 0x6f
    }
};

static const unsigned char test_output[2][375] =
{
    {
        0x76, 0xb8, 0xe0, 0xad, 0xa0, 0xf1, 0x3d, 0x90,
        0x40, 0x5d, 0x6a, 0xe5, 0x53, 0x86, 0xbd, 0x28,
        0xbd, 0xd2, 0x19, 0xb8, 0xa0, 0x8d, 0xed, 0x1a,
        0xa8, 0x36, 0xef, 0xcc, 0x8b, 0x77, 0x0d, 0xc7,
        0xda, 0x41, 0x59, 0x7c, 0x51, 0x57, 0x48, 0x8d,
        0x77, 0x24, 0xe0, 0x3f, 0xb8, 0xd8, 0x4a, 0x37,
        0x6a, 0x43, 0xb8, 0xf4, 0x15, 0x18, 0xa1, 0x1c,
        0xc3, 0x87, 0xb6, 0x69, 0xb2, 0xee, 0x65, 0x86
    },
    {
        0xa3, 0xfb, 0xf0, 0x7d, 0xf3, 0xfa, 0x2f, 0xde,
        0x4f, 0x37, 0x6c, 0xa2, 0x3e, 0x82, 0x73, 0x70,
        0x41, 0x60, 0x5d, 0x9f, 0x4f, 0x4f, 0x57, 0xbd,
        0x8c, 0xff, 0x2c, 0x1d, 0x4b, 0x79, 0x55, 0xec,
        0x2a, 0x97, 0x94, 0x8b, 0xd3, 0x72, 0x29, 0x15,
        0xc8, 0xf3, 0xd3, 0x37, 0xf7, 0xd3, 0x70, 0x05,
        0x0e, 0x9e, 0x96, 0xd6, 0x47, 0xb7, 0xc3, 0x9f,
        0x56, 0xe0, 0x31, 0xca, 0x5e, 0xb6, 0x25, 0x0d,
        0x40, 0x42, 0xe0, 0x27, 0x85, 0xec, 0xec, 0xfa,
        0x4b, 0x4b, 0xb5, 0xe8, 0xea, 0xd0, 0x44, 0x0e,
        0x20, 0xb6, 0xe8, 0xdb, 0x09, 0xd8, 0x81, 0xa7,
        0xc6, 0x13, 0x2f, 0x42, 0x0e, 0x52, 0x79, 0x50,
        0x42, 0xbd, 0xfa, 0x77, 0x73, 0xd8, 0xa9, 0x05,
        0x14, 0x47, 0xb3, 0x29, 0x1c, 0xe1, 0x41, 0x1c,
        0x68, 0x04, 0x65, 0x55, 0x2a, 0xa6, 0xc4, 0x05,
        0xb7, 0x76, 0x4d, 0x5e, 0x87, 0xbe, 0xa8, 0x5a,
        0xd0, 0x0f, 0x84, 0x49, 0xed, 0x8f, 0x72, 0xd0,
        0xd6, 0x62, 0xab, 0x05, 0x26, 0x91, 0xca, 0x66,
        0x42, 0x4b, 0xc8, 0x6d, 0x2d, 0xf8, 0x0e, 0xa4,
        0x1f, 0x43, 0xab, 0xf9, 0x37, 0xd3, 0x25, 0x9d,
        0xc4, 0xb2, 0xd0, 0xdf, 0xb4, 0x8a, 0x6c, 0x91,
        0x39, 0xdd, 0xd7, 0xf7, 0x69, 0x66, 0xe9, 0x28,
        0xe6, 0x35, 0x55, 0x3b, 0xa7, 0x6c, 0x5c, 0x87,
        0x9d, 0x7b, 0x35, 0xd4, 0x9e, 0xb2, 0xe6, 0x2b,
        0x08, 0x71, 0xcd, 0xac, 0x63, 0x89, 0x39, 0xe2,
        0x5e, 0x8a, 0x1e, 0x0e, 0xf9, 0xd5, 0x28, 0x0f,
        0xa8, 0xca, 0x32, 0x8b, 0x35, 0x1c, 0x3c, 0x76,
        0x59, 0x89, 0xcb, 0xcf, 0x3d, 0xaa, 0x8b, 0x6c,
        0xcc, 0x3a, 0xaf, 0x9f, 0x39, 0x79, 0xc9, 0x2b,
        0x37, 0x20, 0xfc, 0x88, 0xdc, 0x95, 0xed, 0x84,
        0xa1, 0xbe, 0x05, 0x9c, 0x64, 0x99, 0xb9, 0xfd,
        0xa2, 0x36, 0xe7, 0xe8, 0x18, 0xb0, 0x4b, 0x0b,
        0xc3, 0x9c, 0x1e, 0x87, 0x6b, 0x19, 0x3b, 0xfe,
        0x55, 0x69, 0x75, 0x3f, 0x88, 0x12, 0x8c, 0xc0,
        0x8a, 0xaa, 0x9b, 0x63, 0xd1, 0xa1, 0x6f, 0x80,
        0xef, 0x25, 0x54, 0xd7, 0x18, 0x9c, 0x41, 0x1f,
        0x58, 0x69, 0xca, 0x52, 0xc5, 0xb8, 0x3f, 0xa3,
        0x6f, 0xf2, 0x16, 0xb9, 0xc1, 0xd3, 0x00, 0x62,
        0xbe, 0xbc, 0xfd, 0x2d, 0xc5, 0xbc, 0xe0, 0x91,
        0x19, 0x34, 0xfd, 0xa7, 0x9a, 0x86, 0xf6, 0xe6,
        0x98, 0xce, 0xd7, 0x59, 0xc3, 0xff, 0x9b, 0x64,
        0x77, 0x33, 0x8f, 0x3d, 0xa4, 0xf9, 0xcd, 0x85,
        0x14, 0xea, 0x99, 0x82, 0xcc, 0xaf, 0xb3, 0x41,
        0xb2, 0x38, 0x4d, 0xd9, 0x02, 0xf3, 0xd1, 0xab,
        0x7a, 0xc6, 0x1d, 0xd2, 0x9c, 0x6f, 0x21, 0xba,
        0x5b, 0x86, 0x2f, 0x37, 0x30, 0xe3, 0x7c, 0xfd,
        0xc4, 0xfd, 0x80, 0x6c, 0x22, 0xf2, 0x21
    }
};

static const size_t test_lengths[2] =
{
    64U,
    375U
};

/* Make sure no other definition is already present. */
#undef ASSERT

#define ASSERT( cond, args )            \
    do                                  \
    {                                   \
        if( ! ( cond ) )                \
        {                               \
            if( verbose != 0 )          \
                mbedtls_printf args;    \
                                        \
            return( -1 );               \
        }                               \
    }                                   \
    while( 0 )

/**
 * \brief           The ChaCha20 checkup routine.
 *
 * \return          \c 0 on success.
 * \return          \c 1 on failure.
 */
int mbedtls_chacha20_self_test( int verbose )
{
    unsigned char output[381];
    unsigned i;
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;

    for( i = 0U; i < 2U; i++ )
    {
        if( verbose != 0 )
            mbedtls_printf( "  ChaCha20 test %u ", i );

        ret = mbedtls_chacha20_crypt( test_keys[i],
                                      test_nonces[i],
                                      test_counters[i],
                                      test_lengths[i],
                                      test_input[i],
                                      output );

        ASSERT( 0 == ret, ( "error code: %i\n", ret ) );

        ASSERT( 0 == timingsafe_bcmp( output, test_output[i], test_lengths[i] ),
                ( "failed (output)\n" ) );

        if( verbose != 0 )
            mbedtls_printf( "passed\n" );
    }

    if( verbose != 0 )
        mbedtls_printf( "\n" );

    return( 0 );
}

#endif /* MBEDTLS_SELF_TEST */
