#ifndef MBEDTLS_CONFIG_H_
#define MBEDTLS_CONFIG_H_
#include <stdbool.h>

/* protocols */
#define MBEDTLS_SSL_PROTO_TLS1_2
#ifndef TINY
#define MBEDTLS_SSL_PROTO_TLS1_1
#define MBEDTLS_SSL_PROTO_TLS1
/*#define MBEDTLS_SSL_PROTO_TLS1_3_EXPERIMENTAL*/
/*#define MBEDTLS_SSL_PROTO_DTLS*/
/*#define MBEDTLS_SSL_PROTO_SSL3*/
#endif

/* hash functions */
#define MBEDTLS_MD5_C
#define MBEDTLS_SHA1_C
#define MBEDTLS_SHA256_C
#define MBEDTLS_SHA512_C

/* random numbers */
#define ENTROPY_HAVE_STRONG
#define MBEDTLS_CTR_DRBG_C
#define MBEDTLS_HMAC_DRBG_C
/*#define MBEDTLS_ENTROPY_FORCE_SHA256*/
/*#define MBEDTLS_TEST_NULL_ENTROPY*/

/* ciphers */
#define MBEDTLS_AES_C
#define MBEDTLS_CHACHA20_C
#define MBEDTLS_POLY1305_C
#define MBEDTLS_CHACHAPOLY_C
#ifdef MBEDTLS_SSL_PROTO_TLS1
#define MBEDTLS_DES_C
#endif
/*#define MBEDTLS_CIPHER_NULL_CIPHER*/
/*#define MBEDTLS_ENABLE_WEAK_CIPHERSUITES*/
/*#define MBEDTLS_REMOVE_3DES_CIPHERSUITES*/

/* block modes */
#define MBEDTLS_GCM_C
#ifndef TINY
#define MBEDTLS_CIPHER_MODE_CBC
/*#define MBEDTLS_CCM_C*/
/*#define MBEDTLS_CIPHER_MODE_CFB*/
/*#define MBEDTLS_CIPHER_MODE_CTR*/
/*#define MBEDTLS_CIPHER_MODE_OFB*/
/*#define MBEDTLS_CIPHER_MODE_XTS*/
#endif

/* key exchange */
#define MBEDTLS_RSA_C
#define MBEDTLS_KEY_EXCHANGE_RSA_ENABLED
#define MBEDTLS_KEY_EXCHANGE_PSK_ENABLED
#define MBEDTLS_ECP_C
#define MBEDTLS_ECDH_C
#define MBEDTLS_ECDSA_C
#define MBEDTLS_ECDSA_DETERMINISTIC
#define MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED
#define MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED
#define MBEDTLS_ECDH_VARIANT_EVEREST_ENABLED
#ifndef TINY
#define MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED
#define MBEDTLS_DHM_C
#define MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED
/*#define MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED*/
/*#define MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED*/
/*#define MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED*/
/*#define MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED*/
#endif

/* eliptic curves */
#define MBEDTLS_ECP_DP_SECP256R1_ENABLED
#define MBEDTLS_ECP_DP_SECP384R1_ENABLED
#define MBEDTLS_ECP_DP_CURVE25519_ENABLED
#ifndef TINY
#define MBEDTLS_ECP_DP_CURVE448_ENABLED
/*#define MBEDTLS_ECP_DP_SECP521R1_ENABLED*/
/*#define MBEDTLS_ECP_DP_BP384R1_ENABLED*/
/*#define MBEDTLS_ECP_DP_SECP192R1_ENABLED*/
/*#define MBEDTLS_ECP_DP_SECP224R1_ENABLED*/
/*#define MBEDTLS_ECP_DP_SECP192K1_ENABLED*/
/*#define MBEDTLS_ECP_DP_SECP224K1_ENABLED*/
/*#define MBEDTLS_ECP_DP_SECP256K1_ENABLED*/
/*#define MBEDTLS_ECP_DP_BP256R1_ENABLED*/
/*#define MBEDTLS_ECP_DP_BP512R1_ENABLED*/
#endif

#define MBEDTLS_X509_CHECK_KEY_USAGE
#define MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE
/*#define MBEDTLS_X509_ALLOW_EXTENSIONS_NON_V3*/
/*#define MBEDTLS_X509_ALLOW_UNSUPPORTED_CRITICAL_EXTENSION*/

/* boringssl and mbedtls hold considerable disagreement */
#define MBEDTLS_CTR_DRBG_RESEED_INTERVAL  4096
#define MBEDTLS_HMAC_DRBG_RESEED_INTERVAL 4096
#define MBEDTLS_ENTROPY_MAX_SOURCES       4
#define MBEDTLS_X509_MAX_INTERMEDIATE_CA  8

/*
 * Boosts performance from 230k qps to 330k
 * Hardens against against sbox side channels
 */
#define MBEDTLS_AESNI_C
#define MBEDTLS_AESCE_C

#ifdef __x86_64__
#define MBEDTLS_HAVE_X86_64
#define MBEDTLS_HAVE_SSE2
#endif

#ifndef TINY
/*
 * TODO(jart): RHEL5 sends SSLv2 hello even though it supports TLS. Is
 *             DROWN really a problem if we turn this on? Since Google
 *             supports it on their website. SSLLabs says we're OK.
 */
#define MBEDTLS_SSL_SRV_SUPPORT_SSLV2_CLIENT_HELLO
#endif

#ifndef TINY
/*
 * The CIA says "messages should be compressed prior to encryption"
 * because "compression reduces the amount of information to be
 * encrypted, thereby decreasing the amount of material available for
 * cryptanalysis. Additionally, compression is designed to eliminate
 * redundancies in the message, further complicating cryptanalysis."
 *
 * Google says that if you (1) have the ability to record encrypted
 * communications made by a machine and (2) have the ability to run code
 * on that machine which injects plaintext repeatedly into the encrypted
 * messages, then you can extract other small parts of the mesasge which
 * the code execution sandbox doesn't allow you to see, and that the
 * only solution to stop using compression.
 *
 * Since we pay $0.12/gb for GCP bandwidth we choose to believe the CIA.
 */
#define MBEDTLS_ZLIB_SUPPORT
#endif

#ifdef MODE_DBG
#define MBEDTLS_CHECK_PARAMS
#endif

#define MBEDTLS_MD5_SMALLER
#define MBEDTLS_SHA1_SMALLER
#define MBEDTLS_SHA256_SMALLER
#define MBEDTLS_SHA512_SMALLER
#define MBEDTLS_ECP_NIST_OPTIM
#ifdef TINY
#define MBEDTLS_AES_ROM_TABLES
#define MBEDTLS_AES_FEWER_TABLES
#endif

#define MBEDTLS_PLATFORM_C
#define MBEDTLS_HAVE_TIME
#define MBEDTLS_HAVE_TIME_DATE
#define MBEDTLS_DEPRECATED_REMOVED
#define MBEDTLS_NO_PLATFORM_ENTROPY

/**
 * \def MBEDTLS_PLATFORM_MEMORY
 *
 * Enable the memory allocation layer.
 *
 * By default mbed TLS uses the system-provided calloc() and free().
 * This allows different allocators (self-implemented or provided) to be
 * provided to the platform abstraction layer.
 *
 * Enabling MBEDTLS_PLATFORM_MEMORY without the
 * MBEDTLS_PLATFORM_{FREE,CALLOC}_MACROs will provide
 * "mbedtls_platform_set_calloc_free()" allowing you to set an alternative
 * calloc() and free() function pointer at runtime.
 *
 * Enabling MBEDTLS_PLATFORM_MEMORY and specifying
 * MBEDTLS_PLATFORM_{CALLOC,FREE}_MACROs will allow you to specify the
 * alternate function at compile time.
 *
 * Enable this layer to allow use of alternative memory allocators.
 */
/*#define MBEDTLS_PLATFORM_MEMORY*/

/**
 * \def MBEDTLS_ENTROPY_HARDWARE_ALT
 *
 * Uncomment this macro to let mbed TLS use your own implementation of a
 * hardware entropy collector.
 *
 * Your function must be called \c mbedtls_hardware_poll(), have the same
 * prototype as declared in entropy_poll.h, and accept NULL as first argument.
 *
 * Uncomment to use your own hardware entropy collector.
 */
#define MBEDTLS_ENTROPY_HARDWARE_ALT

/**
 * Enables PKCS#5 functions, e.g. PBKDF2.
 */
#define MBEDTLS_PKCS5_C

/**
 * \def MBEDTLS_CIPHER_PADDING_PKCS7
 *
 * MBEDTLS_CIPHER_PADDING_XXX: Uncomment or comment macros to add support for
 * specific padding modes in the cipher layer with cipher modes that support
 * padding (e.g. CBC)
 *
 * If you disable all padding modes, only full blocks can be used with CBC.
 *
 * Enable padding modes in the cipher layer.
 */
#define MBEDTLS_CIPHER_PADDING_PKCS7
#define MBEDTLS_CIPHER_PADDING_ONE_AND_ZEROS
#define MBEDTLS_CIPHER_PADDING_ZEROS_AND_LEN
#define MBEDTLS_CIPHER_PADDING_ZEROS

/**
 * \def MBEDTLS_CTR_DRBG_USE_128_BIT_KEY
 *
 * Uncomment this macro to use a 128-bit key in the CTR_DRBG module.
 * By default, CTR_DRBG uses a 256-bit key.
 */
/*#define MBEDTLS_CTR_DRBG_USE_128_BIT_KEY*/

/**
 * \def MBEDTLS_ECP_NO_INTERNAL_RNG
 *
 * When this option is disabled, mbedtls_ecp_mul() will make use of an
 * internal RNG when called with a NULL \c f_rng argument, in order to protect
 * against some side-channel attacks.
 *
 * This protection introduces a dependency of the ECP module on one of the
 * DRBG modules. For very constrained implementations that don't require this
 * protection (for example, because you're only doing signature verification,
 * so not manipulating any secret, or because local/physical side-channel
 * attacks are outside your threat model), it might be desirable to get rid of
 * that dependency.
 *
 * \warning Enabling this option makes some uses of ECP vulnerable to some
 * side-channel attacks. Only enable it if you know that's not a problem for
 * your use case.
 *
 * Uncomment this macro to disable some counter-measures in ECP.
 */
/*#define MBEDTLS_ECP_NO_INTERNAL_RNG*/

/**
 * \def MBEDTLS_ECP_RESTARTABLE
 *
 * Enable "non-blocking" ECC operations that can return early and be resumed.
 *
 * This allows various functions to pause by returning
 * #MBEDTLS_ERR_ECP_IN_PROGRESS (or, for functions in the SSL module,
 * #MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS) and then be called later again in
 * order to further progress and eventually complete their operation. This is
 * controlled through mbedtls_ecp_set_max_ops() which limits the maximum
 * number of ECC operations a function may perform before pausing; see
 * mbedtls_ecp_set_max_ops() for more information.
 *
 * This is useful in non-threaded environments if you want to avoid blocking
 * for too long on ECC (and, hence, X.509 or SSL/TLS) operations.
 *
 * Uncomment this macro to enable restartable ECC computations.
 *
 * \note  This option only works with the default software implementation of
 *        elliptic curve functionality. It is incompatible with
 *        MBEDTLS_ECP_ALT, MBEDTLS_ECDH_XXX_ALT, MBEDTLS_ECDSA_XXX_ALT
 *        and MBEDTLS_ECDH_LEGACY_CONTEXT.
 */
/*#define MBEDTLS_ECP_RESTARTABLE*/

/**
 * \def MBEDTLS_ECDH_LEGACY_CONTEXT
 *
 * Use a backward compatible ECDH context.
 *
 * Mbed TLS supports two formats for ECDH contexts (#mbedtls_ecdh_context
 * defined in `ecdh.h`). For most applications, the choice of format makes
 * no difference, since all library functions can work with either format,
 * except that the new format is incompatible with MBEDTLS_ECP_RESTARTABLE.
 *
 * The new format used when this option is disabled is smaller
 * (56 bytes on a 32-bit platform). In future versions of the library, it
 * will support alternative implementations of ECDH operations.
 * The new format is incompatible with applications that access
 * context fields directly and with restartable ECP operations.
 *
 * Define this macro if you enable MBEDTLS_ECP_RESTARTABLE or if you
 * want to access ECDH context fields directly. Otherwise you should
 * comment out this macro definition.
 *
 * This option has no effect if #MBEDTLS_ECDH_C is not enabled.
 *
 * \note This configuration option is experimental. Future versions of the
 *       library may modify the way the ECDH context layout is configured
 *       and may modify the layout of the new context type.
 */
/*#define MBEDTLS_ECDH_LEGACY_CONTEXT*/

/**
 * \def MBEDTLS_PK_PARSE_EC_EXTENDED
 *
 * Enhance support for reading EC keys using variants of SEC1 not allowed by
 * RFC 5915 and RFC 5480.
 *
 * Currently this means parsing the SpecifiedECDomain choice of EC
 * parameters (only known groups are supported, not arbitrary domains, to
 * avoid validation issues).
 *
 * Disable if you only need to support RFC 5915 + 5480 key formats.
 */
/*#define MBEDTLS_PK_PARSE_EC_EXTENDED*/

/**
 * \def MBEDTLS_ERROR_STRERROR_DUMMY
 *
 * Enable a dummy error function to make use of mbedtls_strerror() in
 * third party libraries easier when MBEDTLS_ERROR_C is disabled
 * (no effect when MBEDTLS_ERROR_C is enabled).
 *
 * You can safely disable this if MBEDTLS_ERROR_C is enabled, or if you're
 * not using mbedtls_strerror() or error_strerror() in your application.
 *
 * Disable if you run into name conflicts and want to really remove the
 * mbedtls_strerror()
 */
#define MBEDTLS_ERROR_STRERROR_DUMMY

/**
 * \def MBEDTLS_GENPRIME
 *
 * Enable the prime-number generation code.
 *
 * Requires: MBEDTLS_BIGNUM_C
 */
#define MBEDTLS_GENPRIME

/**
 * \def MBEDTLS_FS_IO
 *
 * Enable functions that use the filesystem.
 */
#define MBEDTLS_FS_IO

/**
 * \def MBEDTLS_MEMORY_DEBUG
 *
 * Enable debugging of buffer allocator memory issues. Automatically prints
 * (to stderr) all (fatal) messages on memory allocation issues. Enables
 * function for 'debug output' of allocated memory.
 *
 * Requires: MBEDTLS_MEMORY_BUFFER_ALLOC_C
 *
 * Uncomment this macro to let the buffer allocator print out error messages.
 */
/*#define MBEDTLS_MEMORY_DEBUG*/

/**
 * \def MBEDTLS_MEMORY_BACKTRACE
 *
 * Include backtrace information with each allocated block.
 *
 * Requires: MBEDTLS_MEMORY_BUFFER_ALLOC_C
 *           GLIBC-compatible backtrace() an backtrace_symbols() support
 *
 * Uncomment this macro to include backtrace information
 */
/*#define MBEDTLS_MEMORY_BACKTRACE*/

/**
 * \def MBEDTLS_PK_RSA_ALT_SUPPORT
 *
 * Support external private RSA keys (eg from a HSM) in the PK layer.
 *
 * Comment this macro to disable support for external private RSA keys.
 */
/*#define MBEDTLS_PK_RSA_ALT_SUPPORT*/

/**
 * \def MBEDTLS_PKCS1_V15
 *
 * Enable support for PKCS#1 v1.5 encoding.
 *
 * Requires: MBEDTLS_RSA_C
 *
 * This enables support for PKCS#1 v1.5 operations.
 */
#define MBEDTLS_PKCS1_V15

/**
 * \def MBEDTLS_PKCS1_V21
 *
 * Enable support for PKCS#1 v2.1 encoding.
 *
 * Requires: MBEDTLS_MD_C, MBEDTLS_RSA_C
 *
 * This enables support for RSAES-OAEP and RSASSA-PSS operations.
 */
/*#define MBEDTLS_PKCS1_V21*/

/**
 * \def MBEDTLS_RSA_NO_CRT
 *
 * Do not use the Chinese Remainder Theorem
 * for the RSA private operation.
 *
 * Uncomment this macro to disable the use of CRT in RSA.
 */
/*#define MBEDTLS_RSA_NO_CRT*/

/**
 * \def MBEDTLS_SELF_TEST
 *
 * Enable the checkup functions (*_self_test).
 */
#define MBEDTLS_SELF_TEST

/**
 * \def MBEDTLS_CERTS_C
 *
 * Enable the test certificates.
 *
 * Module:  library/certs.c
 * Caller:
 *
 * This module is used for testing (ssl_client/server).
 */
#define MBEDTLS_CERTS_C

/**
 * \def MBEDTLS_SHA512_NO_SHA384
 *
 * Disable the SHA-384 option of the SHA-512 module. Use this to save some
 * code size on devices that don't use SHA-384.
 *
 * Requires: MBEDTLS_SHA512_C
 *
 * Uncomment to disable SHA-384
 */
/*#define MBEDTLS_SHA512_NO_SHA384*/

/**
 * \def MBEDTLS_SSL_ALL_ALERT_MESSAGES
 *
 * Enable sending of alert messages in case of encountered errors as per RFC.
 * If you choose not to send the alert messages, mbed TLS can still communicate
 * with other servers, only debugging of failures is harder.
 *
 * The advantage of not sending alert messages, is that no information is given
 * about reasons for failures thus preventing adversaries of gaining intel.
 *
 * Enable sending of all alert messages
 */
#define MBEDTLS_SSL_ALL_ALERT_MESSAGES

#ifdef MBEDTLS_SSL_PROTO_DTLS
/**
 * \def MBEDTLS_SSL_RECORD_CHECKING
 *
 * Enable the function mbedtls_ssl_check_record() which can be used to check
 * the validity and authenticity of an incoming record, to verify that it has
 * not been seen before. These checks are performed without modifying the
 * externally visible state of the SSL context.
 *
 * See mbedtls_ssl_check_record() for more information.
 *
 * Uncomment to enable support for record checking.
 */
#define MBEDTLS_SSL_RECORD_CHECKING
#endif

/**
 * \def MBEDTLS_SSL_DTLS_CONNECTION_ID
 *
 * Enable support for the DTLS Connection ID extension
 * (version draft-ietf-tls-dtls-connection-id-05,
 * https://tools.ietf.org/html/draft-ietf-tls-dtls-connection-id-05)
 * which allows to identify DTLS connections across changes
 * in the underlying transport.
 *
 * Setting this option enables the SSL APIs `mbedtls_ssl_set_cid()`,
 * `mbedtls_ssl_get_peer_cid()` and `mbedtls_ssl_conf_cid()`.
 * See the corresponding documentation for more information.
 *
 * \warning The Connection ID extension is still in draft state.
 *          We make no stability promises for the availability
 *          or the shape of the API controlled by this option.
 *
 * The maximum lengths of outgoing and incoming CIDs can be configured
 * through the options
 * - MBEDTLS_SSL_CID_OUT_LEN_MAX
 * - MBEDTLS_SSL_CID_IN_LEN_MAX.
 *
 * Requires: MBEDTLS_SSL_PROTO_DTLS
 *
 * Uncomment to enable the Connection ID extension.
 */
/*#define MBEDTLS_SSL_DTLS_CONNECTION_ID*/

/**
 * \def MBEDTLS_SSL_ASYNC_PRIVATE
 *
 * Enable asynchronous external private key operations in SSL. This allows
 * you to configure an SSL connection to call an external cryptographic
 * module to perform private key operations instead of performing the
 * operation inside the library.
 */
/*#define MBEDTLS_SSL_ASYNC_PRIVATE*/

/**
 * \def MBEDTLS_SSL_CONTEXT_SERIALIZATION
 *
 * Enable serialization of the TLS context structures, through use of the
 * functions mbedtls_ssl_context_save() and mbedtls_ssl_context_load().
 *
 * This pair of functions allows one side of a connection to serialize the
 * context associated with the connection, then free or re-use that context
 * while the serialized state is persisted elsewhere, and finally deserialize
 * that state to a live context for resuming read/write operations on the
 * connection. From a protocol perspective, the state of the connection is
 * unaffected, in particular this is entirely transparent to the peer.
 *
 * Note: this is distinct from TLS session resumption, which is part of the
 * protocol and fully visible by the peer. TLS session resumption enables
 * establishing new connections associated to a saved session with shorter,
 * lighter handshakes, while context serialization is a local optimization in
 * handling a single, potentially long-lived connection.
 *
 * Enabling these APIs makes some SSL structures larger, as 64 extra bytes are
 * saved after the handshake to allow for more efficient serialization, so if
 * you don't need this feature you'll save RAM by disabling it.
 *
 * Comment to disable the context serialization APIs.
 */
/*#define MBEDTLS_SSL_CONTEXT_SERIALIZATION*/

/**
 * \def MBEDTLS_SSL_DEBUG_ALL
 *
 * Enable the debug messages in SSL module for all issues.
 * Debug messages have been disabled in some places to prevent timing
 * attacks due to (unbalanced) debugging function calls.
 *
 * If you need all error reporting you should enable this during debugging,
 * but remove this for production servers that should log as well.
 *
 * Uncomment this macro to report all debug messages on errors introducing
 * a timing side-channel.
 */
/*#define MBEDTLS_SSL_DEBUG_ALL*/

/**
 * \def MBEDTLS_SSL_ENCRYPT_THEN_MAC
 *
 * Enable support for Encrypt-then-MAC, RFC 7366.
 *
 * This allows peers that both support it to use a more robust protection for
 * ciphersuites using CBC, providing deep resistance against timing attacks
 * on the padding or underlying cipher.
 *
 * This only affects CBC ciphersuites, and is useless if none is defined.
 *
 * Requires: MBEDTLS_SSL_PROTO_TLS1    or
 *           MBEDTLS_SSL_PROTO_TLS1_1  or
 *           MBEDTLS_SSL_PROTO_TLS1_2
 *
 * Comment this macro to disable support for Encrypt-then-MAC
 */
#define MBEDTLS_SSL_ENCRYPT_THEN_MAC

/**
 * \def MBEDTLS_SSL_EXTENDED_MASTER_SECRET
 *
 * Enable support for RFC 7627: Session Hash and Extended Master Secret
 * Extension.
 *
 * This was introduced as "the proper fix" to the Triple Handshake familiy of
 * attacks, but it is recommended to always use it (even if you disable
 * renegotiation), since it actually fixes a more fundamental issue in the
 * original SSL/TLS design, and has implications beyond Triple Handshake.
 *
 * Requires: MBEDTLS_SSL_PROTO_TLS1    or
 *           MBEDTLS_SSL_PROTO_TLS1_1  or
 *           MBEDTLS_SSL_PROTO_TLS1_2
 *
 * Comment this macro to disable support for Extended Master Secret.
 */
#define MBEDTLS_SSL_EXTENDED_MASTER_SECRET

#if (MBEDTLS_SSL_PROTO_SSL3 + MBEDTLS_SSL_PROTO_TLS1 +     \
     MBEDTLS_SSL_PROTO_TLS1_1 + MBEDTLS_SSL_PROTO_TLS1_2 + \
     MBEDTLS_SSL_PROTO_TLS1_3_EXPERIMENTAL + 0) > 1
/**
 * \def MBEDTLS_SSL_FALLBACK_SCSV
 *
 * Enable support for RFC 7507: Fallback Signaling Cipher Suite Value (SCSV)
 * for Preventing Protocol Downgrade Attacks.
 *
 * For servers, it is recommended to always enable this, unless you support
 * only one version of TLS, or know for sure that none of your clients
 * implements a fallback strategy.
 *
 * For clients, you only need this if you're using a fallback strategy, which
 * is not recommended in the first place, unless you absolutely need it to
 * interoperate with buggy (version-intolerant) servers.
 *
 * Comment this macro to disable support for FALLBACK_SCSV
 */
#define MBEDTLS_SSL_FALLBACK_SCSV
#endif

/**
 * \def MBEDTLS_SSL_KEEP_PEER_CERTIFICATE
 *
 * This option controls the availability of the API mbedtls_ssl_get_peer_cert()
 * giving access to the peer's certificate after completion of the handshake.
 *
 * Unless you need mbedtls_ssl_peer_cert() in your application, it is
 * recommended to disable this option for reduced RAM usage.
 *
 * \note If this option is disabled, mbedtls_ssl_get_peer_cert() is still
 *       defined, but always returns \c NULL.
 *
 * \note This option has no influence on the protection against the
 *       triple handshake attack. Even if it is disabled, Mbed TLS will
 *       still ensure that certificates do not change during renegotiation,
 *       for exaple by keeping a hash of the peer's certificate.
 *
 * Comment this macro to disable storing the peer's certificate
 * after the handshake.
 */
#define MBEDTLS_SSL_KEEP_PEER_CERTIFICATE

/**
 * \def MBEDTLS_SSL_CBC_RECORD_SPLITTING
 *
 * Enable 1/n-1 record splitting for CBC mode in SSLv3 and TLS 1.0.
 *
 * This is a countermeasure to the BEAST attack, which also minimizes the risk
 * of interoperability issues compared to sending 0-length records.
 *
 * Comment this macro to disable 1/n-1 record splitting.
 */
#define MBEDTLS_SSL_CBC_RECORD_SPLITTING

/**
 * \def MBEDTLS_SSL_RENEGOTIATION
 *
 * Enable support for TLS renegotiation.
 *
 * The two main uses of renegotiation are (1) refresh keys on long-lived
 * connections and (2) client authentication after the initial handshake.
 * If you don't need renegotiation, it's probably better to disable it, since
 * it has been associated with security issues in the past and is easy to
 * misuse/misunderstand.
 *
 * Comment this to disable support for renegotiation.
 *
 * \note   Even if this option is disabled, both client and server are aware
 *         of the Renegotiation Indication Extension (RFC 5746) used to
 *         prevent the SSL renegotiation attack (see RFC 5746 Sect. 1).
 *         (See \c mbedtls_ssl_conf_legacy_renegotiation for the
 *          configuration of this extension).
 */
/*#define MBEDTLS_SSL_RENEGOTIATION*/

/**
 * \def MBEDTLS_SSL_SRV_RESPECT_CLIENT_PREFERENCE
 *
 * Pick the ciphersuite according to the client's preferences rather than ours
 * in the SSL Server module (MBEDTLS_SSL_SRV_C).
 *
 * Uncomment this macro to respect client's ciphersuite order
 */
#define MBEDTLS_SSL_SRV_RESPECT_CLIENT_PREFERENCE

/**
 * \def MBEDTLS_SSL_MAX_FRAGMENT_LENGTH
 *
 * Enable support for RFC 6066 max_fragment_length extension in SSL.
 *
 * Comment this macro to disable support for the max_fragment_length extension
 */
#define MBEDTLS_SSL_MAX_FRAGMENT_LENGTH

/**
 * \def MBEDTLS_SSL_ALPN
 *
 * Enable support for RFC 7301 Application Layer Protocol Negotiation.
 *
 * Comment this macro to disable support for ALPN.
 */
#define MBEDTLS_SSL_ALPN

#ifdef MBEDTLS_SSL_PROTO_DTLS
/**
 * \def MBEDTLS_SSL_DTLS_ANTI_REPLAY
 *
 * Enable support for the anti-replay mechanism in DTLS.
 *
 * Requires: MBEDTLS_SSL_TLS_C
 *           MBEDTLS_SSL_PROTO_DTLS
 *
 * \warning Disabling this is often a security risk!
 * See mbedtls_ssl_conf_dtls_anti_replay() for details.
 *
 * Comment this to disable anti-replay in DTLS.
 */
#define MBEDTLS_SSL_DTLS_ANTI_REPLAY
#endif

#ifdef MBEDTLS_SSL_PROTO_DTLS
/**
 * \def MBEDTLS_SSL_DTLS_HELLO_VERIFY
 *
 * Enable support for HelloVerifyRequest on DTLS servers.
 *
 * This feature is highly recommended to prevent DTLS servers being used as
 * amplifiers in DoS attacks against other hosts. It should always be enabled
 * unless you know for sure amplification cannot be a problem in the
 * environment in which your server operates.
 *
 * \warning Disabling this can ba a security risk! (see above)
 *
 * Requires: MBEDTLS_SSL_PROTO_DTLS
 *
 * Comment this to disable support for HelloVerifyRequest.
 */
#define MBEDTLS_SSL_DTLS_HELLO_VERIFY
#endif

/**
 * \def MBEDTLS_SSL_DTLS_SRTP
 *
 * Enable support for negotation of DTLS-SRTP (RFC 5764)
 * through the use_srtp extension.
 *
 * \note This feature provides the minimum functionality required
 * to negotiate the use of DTLS-SRTP and to allow the derivation of
 * the associated SRTP packet protection key material.
 * In particular, the SRTP packet protection itself, as well as the
 * demultiplexing of RTP and DTLS packets at the datagram layer
 * (see Section 5 of RFC 5764), are not handled by this feature.
 * Instead, after successful completion of a handshake negotiating
 * the use of DTLS-SRTP, the extended key exporter API
 * mbedtls_ssl_conf_export_keys_ext_cb() should be used to implement
 * the key exporter described in Section 4.2 of RFC 5764 and RFC 5705
 * (this is implemented in the SSL example programs).
 * The resulting key should then be passed to an SRTP stack.
 *
 * Setting this option enables the runtime API
 * mbedtls_ssl_conf_dtls_srtp_protection_profiles()
 * through which the supported DTLS-SRTP protection
 * profiles can be configured. You must call this API at
 * runtime if you wish to negotiate the use of DTLS-SRTP.
 *
 * Requires: MBEDTLS_SSL_PROTO_DTLS
 *
 * Uncomment this to enable support for use_srtp extension.
 */
/*#define MBEDTLS_SSL_DTLS_SRTP*/

#ifdef MBEDTLS_SSL_PROTO_DTLS
/**
 * \def MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE
 *
 * Enable server-side support for clients that reconnect from the same port.
 *
 * Some clients unexpectedly close the connection and try to reconnect using the
 * same source port. This needs special support from the server to handle the
 * new connection securely, as described in section 4.2.8 of RFC 6347. This
 * flag enables that support.
 *
 * Requires: MBEDTLS_SSL_DTLS_HELLO_VERIFY
 *
 * Comment this to disable support for clients reusing the source port.
 */
#define MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE
#endif

/**
 * \def MBEDTLS_SSL_DTLS_BADMAC_LIMIT
 *
 * Enable support for a limit of records with bad MAC.
 *
 * See mbedtls_ssl_conf_dtls_badmac_limit().
 *
 * Requires: MBEDTLS_SSL_PROTO_DTLS
 */
/*#define MBEDTLS_SSL_DTLS_BADMAC_LIMIT*/

/**
 * \def MBEDTLS_SSL_SESSION_TICKETS
 *
 * Enable support for RFC 5077 session tickets in SSL.
 * Client-side, provides full support for session tickets (maintenance of a
 * session store remains the responsibility of the application, though).
 * Server-side, you also need to provide callbacks for writing and parsing
 * tickets, including authenticated encryption and key management. Example
 * callbacks are provided by MBEDTLS_SSL_TICKET_C.
 *
 * Comment this macro to disable support for SSL session tickets
 */
#define MBEDTLS_SSL_SESSION_TICKETS

/**
 * \def MBEDTLS_SSL_EXPORT_KEYS
 *
 * Enable support for exporting key block and master secret.
 * This is required for certain users of TLS, e.g. EAP-TLS.
 *
 * Comment this macro to disable support for key export
 */
/*#define MBEDTLS_SSL_EXPORT_KEYS*/

/**
 * \def MBEDTLS_SSL_SERVER_NAME_INDICATION
 *
 * Enable support for RFC 6066 server name indication (SNI) in SSL.
 *
 * Requires: MBEDTLS_X509_CRT_PARSE_C
 *
 * Comment this macro to disable support for server name indication in SSL
 */
#define MBEDTLS_SSL_SERVER_NAME_INDICATION

/**
 * \def MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH
 *
 * When this option is enabled, the SSL buffer will be resized automatically
 * based on the negotiated maximum fragment length in each direction.
 *
 * Requires: MBEDTLS_SSL_MAX_FRAGMENT_LENGTH
 */
/*#define MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH*/

/**
 * \def MBEDTLS_TEST_CONSTANT_FLOW_MEMSAN
 *
 * Enable testing of the constant-flow nature of some sensitive functions with
 * clang's MemorySanitizer. This causes some existing tests to also test
 * this non-functional property of the code under test.
 *
 * This setting requires compiling with clang -fsanitize=memory. The test
 * suites can then be run normally.
 *
 * \warning This macro is only used for extended testing; it is not considered
 * part of the library's API, so it may change or disappear at any time.
 *
 * Uncomment to enable testing of the constant-flow nature of selected code.
 */
/*#define MBEDTLS_TEST_CONSTANT_FLOW_MEMSAN*/

/**
 * \def MBEDTLS_TEST_CONSTANT_FLOW_VALGRIND
 *
 * Enable testing of the constant-flow nature of some sensitive functions with
 * valgrind's memcheck tool. This causes some existing tests to also test
 * this non-functional property of the code under test.
 *
 * This setting requires valgrind headers for building, and is only useful for
 * testing if the tests suites are run with valgrind's memcheck. This can be
 * done for an individual test suite with 'valgrind ./test_suite_xxx', or when
 * using CMake, this can be done for all test suites with 'make memcheck'.
 *
 * \warning This macro is only used for extended testing; it is not considered
 * part of the library's API, so it may change or disappear at any time.
 *
 * Uncomment to enable testing of the constant-flow nature of selected code.
 */
/*#define MBEDTLS_TEST_CONSTANT_FLOW_VALGRIND*/

/**
 * \def MBEDTLS_TEST_HOOKS
 *
 * Enable features for invasive testing such as introspection functions and
 * hooks for fault injection. This enables additional unit tests.
 *
 * Merely enabling this feature should not change the behavior of the product.
 * It only adds new code, and new branching points where the default behavior
 * is the same as when this feature is disabled.
 * However, this feature increases the attack surface: there is an added
 * risk of vulnerabilities, and more gadgets that can make exploits easier.
 * Therefore this feature must never be enabled in production.
 *
 * See `docs/architecture/testing/mbed-crypto-invasive-testing.md` for more
 * information.
 *
 * Uncomment to enable invasive tests.
 */
/*#define MBEDTLS_TEST_HOOKS*/

/**
 * \def MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK
 *
 * If set, this enables the X.509 API `mbedtls_x509_crt_verify_with_ca_cb()`
 * and the SSL API `mbedtls_ssl_conf_ca_cb()` which allow users to configure
 * the set of trusted certificates through a callback instead of a linked
 * list.
 *
 * This is useful for example in environments where a large number of trusted
 * certificates is present and storing them in a linked list isn't efficient
 * enough, or when the set of trusted certificates changes frequently.
 *
 * See the documentation of `mbedtls_x509_crt_verify_with_ca_cb()` and
 * `mbedtls_ssl_conf_ca_cb()` for more information.
 *
 * Uncomment to enable trusted certificate callbacks.
 */
/*#define MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK*/

/**
 * \def MBEDTLS_ASN1_PARSE_C
 *
 * Enable the generic ASN1 parser.
 *
 * Module:  library/asn1.c
 * Caller:  library/x509.c
 *          library/dhm.c
 *          library/pkcs12.c
 *          library/pkcs5.c
 *          library/pkparse.c
 */
#define MBEDTLS_ASN1_PARSE_C

/**
 * \def MBEDTLS_ASN1_WRITE_C
 *
 * Enable the generic ASN1 writer.
 *
 * Module:  library/asn1write.c
 * Caller:  library/ecdsa.c
 *          library/pkwrite.c
 *          library/x509_create.c
 *          library/x509write_crt.c
 *          library/x509write_csr.c
 */
#define MBEDTLS_ASN1_WRITE_C

/**
 * \def MBEDTLS_BASE64_C
 *
 * Enable the Base64 module.
 *
 * Module:  library/base64.c
 * Caller:  library/pem.c
 *
 * This module is required for PEM support (required by X.509).
 */
#define MBEDTLS_BASE64_C

/**
 * \def MBEDTLS_BIGNUM_C
 *
 * Enable the multi-precision integer library.
 *
 * Module:  library/bignum.c
 * Caller:  library/dhm.c
 *          library/ecp.c
 *          library/ecdsa.c
 *          library/rsa.c
 *          library/rsa_internal.c
 *          library/ssl_tls.c
 *
 * This module is required for RSA, DHM and ECC (ECDH, ECDSA) support.
 */
#define MBEDTLS_BIGNUM_C

/**
 * \def MBEDTLS_CIPHER_C
 *
 * Enable the generic cipher layer.
 *
 * Module:  library/cipher.c
 * Caller:  library/ssl_tls.c
 *
 * Uncomment to enable generic cipher wrappers.
 */
#define MBEDTLS_CIPHER_C

#ifndef TINY
/**
 * \def MBEDTLS_DEBUG_C
 *
 * Enable the debug functions.
 *
 * Module:  library/debug.c
 * Caller:  library/ssl_cli.c
 *          library/ssl_srv.c
 *          library/ssl_tls.c
 *
 * This module provides debugging functions.
 */
#define MBEDTLS_DEBUG_C
#endif

/**
 * \def MBEDTLS_ENTROPY_C
 *
 * Enable the platform-specific entropy code.
 *
 * Module:  library/entropy.c
 * Caller:
 *
 * Requires: MBEDTLS_SHA512_C or MBEDTLS_SHA256_C
 *
 * This module provides a generic entropy pool
 */
#define MBEDTLS_ENTROPY_C

/**
 * \def MBEDTLS_ERROR_C
 *
 * Enable error code to error string conversion.
 *
 * Module:  library/error.c
 * Caller:
 *
 * This module enables mbedtls_strerror().
 */
#define MBEDTLS_ERROR_C

/**
 * \def MBEDTLS_HKDF_C
 *
 * Enable the HKDF algorithm (RFC 5869).
 *
 * Module:  library/hkdf.c
 * Caller:
 *
 * Requires: MBEDTLS_MD_C
 *
 * This module adds support for the Hashed Message Authentication Code
 * (HMAC)-based key derivation function (HKDF).
 */
/*#define MBEDTLS_HKDF_C*/

/**
 * \def MBEDTLS_NIST_KW_C
 *
 * Enable the Key Wrapping mode for 128-bit block ciphers,
 * as defined in NIST SP 800-38F. Only KW and KWP modes
 * are supported. At the moment, only AES is approved by NIST.
 *
 * Module:  library/nist_kw.c
 *
 * Requires: MBEDTLS_AES_C and MBEDTLS_CIPHER_C
 */
#define MBEDTLS_NIST_KW_C

/**
 * \def MBEDTLS_MD_C
 *
 * Enable the generic message digest layer.
 *
 * Module:  library/md.c
 * Caller:
 *
 * Uncomment to enable generic message digest wrappers.
 */
#define MBEDTLS_MD_C

#define MBEDTLS_OID_C
#define MBEDTLS_PEM_PARSE_C
#define MBEDTLS_PEM_WRITE_C
#define MBEDTLS_PK_C
#define MBEDTLS_PK_PARSE_C
#define MBEDTLS_PK_WRITE_C
#define MBEDTLS_X509_USE_C
#define MBEDTLS_X509_CREATE_C
#define MBEDTLS_X509_CRT_WRITE_C
#define MBEDTLS_X509_CRT_PARSE_C
#define MBEDTLS_X509_CSR_PARSE_C
#define MBEDTLS_X509_CSR_WRITE_C
#define MBEDTLS_X509_CRL_PARSE_C

#define MBEDTLS_SSL_TLS_C
#define MBEDTLS_SSL_CLI_C
#define MBEDTLS_SSL_SRV_C
#define MBEDTLS_SSL_TICKET_C
#define MBEDTLS_SSL_CACHE_C
/*#define MBEDTLS_SSL_COOKIE_C*/

/**
 * \def MBEDTLS_SSL_MAX_CONTENT_LEN
 *
 * Maximum length (in bytes) of incoming and outgoing plaintext fragments.
 *
 * This determines the size of both the incoming and outgoing TLS I/O buffers
 * in such a way that both are capable of holding the specified amount of
 * plaintext data, regardless of the protection mechanism used.
 *
 * To configure incoming and outgoing I/O buffers separately, use
 * #MBEDTLS_SSL_IN_CONTENT_LEN and #MBEDTLS_SSL_OUT_CONTENT_LEN,
 * which overwrite the value set by this option.
 *
 * \note When using a value less than the default of 16KB on the client, it is
 *       recommended to use the Maximum Fragment Length (MFL) extension to
 *       inform the server about this limitation. On the server, there
 *       is no supported, standardized way of informing the client about
 *       restriction on the maximum size of incoming messages, and unless
 *       the limitation has been communicated by other means, it is recommended
 *       to only change the outgoing buffer size #MBEDTLS_SSL_OUT_CONTENT_LEN
 *       while keeping the default value of 16KB for the incoming buffer.
 *
 * Uncomment to set the maximum plaintext size of both
 * incoming and outgoing I/O buffers.
 */
/*#define MBEDTLS_SSL_MAX_CONTENT_LEN             16384*/

/**
 * \def MBEDTLS_SSL_IN_CONTENT_LEN
 *
 * Maximum length (in bytes) of incoming plaintext fragments.
 *
 * This determines the size of the incoming TLS I/O buffer in such a way
 * that it is capable of holding the specified amount of plaintext data,
 * regardless of the protection mechanism used.
 *
 * If this option is undefined, it inherits its value from
 * #MBEDTLS_SSL_MAX_CONTENT_LEN.
 *
 * \note When using a value less than the default of 16KB on the client, it is
 *       recommended to use the Maximum Fragment Length (MFL) extension to
 *       inform the server about this limitation. On the server, there
 *       is no supported, standardized way of informing the client about
 *       restriction on the maximum size of incoming messages, and unless
 *       the limitation has been communicated by other means, it is recommended
 *       to only change the outgoing buffer size #MBEDTLS_SSL_OUT_CONTENT_LEN
 *       while keeping the default value of 16KB for the incoming buffer.
 *
 * Uncomment to set the maximum plaintext size of the incoming I/O buffer
 * independently of the outgoing I/O buffer.
 */
/*#define MBEDTLS_SSL_IN_CONTENT_LEN              16384*/

/**
 * \def MBEDTLS_SSL_CID_IN_LEN_MAX
 *
 * The maximum length of CIDs used for incoming DTLS messages.
 */
/*#define MBEDTLS_SSL_CID_IN_LEN_MAX 32*/

/**
 * \def MBEDTLS_SSL_CID_OUT_LEN_MAX
 *
 * The maximum length of CIDs used for outgoing DTLS messages.
 */
/*#define MBEDTLS_SSL_CID_OUT_LEN_MAX 32*/

/**
 * \def MBEDTLS_SSL_CID_PADDING_GRANULARITY
 *
 * This option controls the use of record plaintext padding
 * when using the Connection ID extension in DTLS 1.2.
 *
 * The padding will always be chosen so that the length of the
 * padded plaintext is a multiple of the value of this option.
 *
 * Note: A value of \c 1 means that no padding will be used
 *       for outgoing records.
 *
 * Note: On systems lacking division instructions,
 *       a power of two should be preferred.
 */
/*#define MBEDTLS_SSL_CID_PADDING_GRANULARITY 16*/

/**
 * \def MBEDTLS_SSL_TLS1_3_PADDING_GRANULARITY
 *
 * This option controls the use of record plaintext padding
 * in TLS 1.3.
 *
 * The padding will always be chosen so that the length of the
 * padded plaintext is a multiple of the value of this option.
 *
 * Note: A value of \c 1 means that no padding will be used
 *       for outgoing records.
 *
 * Note: On systems lacking division instructions,
 *       a power of two should be preferred.
 */
/*#define MBEDTLS_SSL_TLS1_3_PADDING_GRANULARITY 1*/

/**
 * \def MBEDTLS_SSL_OUT_CONTENT_LEN
 *
 * Maximum length (in bytes) of outgoing plaintext fragments.
 *
 * This determines the size of the outgoing TLS I/O buffer in such a way
 * that it is capable of holding the specified amount of plaintext data,
 * regardless of the protection mechanism used.
 *
 * If this option undefined, it inherits its value from
 * #MBEDTLS_SSL_MAX_CONTENT_LEN.
 *
 * It is possible to save RAM by setting a smaller outward buffer, while keeping
 * the default inward 16384 byte buffer to conform to the TLS specification.
 *
 * The minimum required outward buffer size is determined by the handshake
 * protocol's usage. Handshaking will fail if the outward buffer is too small.
 * The specific size requirement depends on the configured ciphers and any
 * certificate data which is sent during the handshake.
 *
 * Uncomment to set the maximum plaintext size of the outgoing I/O buffer
 * independently of the incoming I/O buffer.
 */
/*#define MBEDTLS_SSL_OUT_CONTENT_LEN             16384*/

/**
 * \def MBEDTLS_SSL_DTLS_MAX_BUFFERING
 *
 * Maximum number of heap-allocated bytes for the purpose of
 * DTLS handshake message reassembly and future message buffering.
 *
 * This should be at least 9/8 * MBEDTLSSL_IN_CONTENT_LEN
 * to account for a reassembled handshake message of maximum size,
 * together with its reassembly bitmap.
 *
 * A value of 2 * MBEDTLS_SSL_IN_CONTENT_LEN (32768 by default)
 * should be sufficient for all practical situations as it allows
 * to reassembly a large handshake message (such as a certificate)
 * while buffering multiple smaller handshake messages.
 */
/*#define MBEDTLS_SSL_DTLS_MAX_BUFFERING             32768*/

/**
 * Allow SHA-1 in the default TLS configuration for certificate signing.
 * Without this build-time option, SHA-1 support must be activated explicitly
 * through mbedtls_ssl_conf_cert_profile. Turning on this option is not
 * recommended because of it is possible to generate SHA-1 collisions, however
 * this may be safe for legacy infrastructure where additional controls apply.
 *
 * \warning   SHA-1 is considered a weak message digest and its use constitutes
 *            a security risk. If possible, we recommend avoiding dependencies
 *            on it, and considering stronger message digests instead.
 */
/*#define MBEDTLS_TLS_DEFAULT_ALLOW_SHA1_IN_CERTIFICATES*/

/**
 * Allow SHA-1 in the default TLS configuration for TLS 1.2 handshake
 * signature and ciphersuite selection. Without this build-time option, SHA-1
 * support must be activated explicitly through mbedtls_ssl_conf_sig_hashes.
 * The use of SHA-1 in TLS <= 1.1 and in HMAC-SHA-1 is always allowed by
 * default. At the time of writing, there is no practical attack on the use
 * of SHA-1 in handshake signatures, hence this option is turned on by default
 * to preserve compatibility with existing peers, but the general
 * warning applies nonetheless:
 *
 * \warning   SHA-1 is considered a weak message digest and its use constitutes
 *            a security risk. If possible, we recommend avoiding dependencies
 *            on it, and considering stronger message digests instead.
 */
#define MBEDTLS_TLS_DEFAULT_ALLOW_SHA1_IN_KEY_EXCHANGE

#define mbedtls_t_udbl uint128_t
#define MBEDTLS_HAVE_UDBL

#include <libc/dce.h>
#include "third_party/mbedtls/check.inc"
#endif /* MBEDTLS_CONFIG_H_ */
