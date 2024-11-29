
#if defined(TARGET_LIKE_MBED) && \
    ( defined(MBEDTLS_NET_C) || defined(MBEDTLS_TIMING_C) )
#error "The NET and TIMING modules are not available for mbed OS - please use the network and timing functions provided by mbed OS"
#endif

#if defined(MBEDTLS_DEPRECATED_WARNING) && \
    !defined(__GNUC__) && !defined(__clang__)
#error "MBEDTLS_DEPRECATED_WARNING only works with GCC and Clang"
#endif

#if defined(MBEDTLS_HAVE_TIME_DATE) && !defined(MBEDTLS_HAVE_TIME)
#error "MBEDTLS_HAVE_TIME_DATE without MBEDTLS_HAVE_TIME does not make sense"
#endif

#if defined(MBEDTLS_CTR_DRBG_C) && !defined(MBEDTLS_AES_C)
#error "MBEDTLS_CTR_DRBG_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_DHM_C) && !defined(MBEDTLS_BIGNUM_C)
#error "MBEDTLS_DHM_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_TRUNCATED_HMAC_COMPAT) && !defined(MBEDTLS_SSL_TRUNCATED_HMAC)
#error "MBEDTLS_SSL_TRUNCATED_HMAC_COMPAT defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_CMAC_C) && \
    !defined(MBEDTLS_AES_C) && !defined(MBEDTLS_DES_C)
#error "MBEDTLS_CMAC_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_NIST_KW_C) && \
    ( !defined(MBEDTLS_AES_C) || !defined(MBEDTLS_CIPHER_C) )
#error "MBEDTLS_NIST_KW_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECDH_C) && !defined(MBEDTLS_ECP_C)
#error "MBEDTLS_ECDH_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECDSA_C) &&            \
    ( !defined(MBEDTLS_ECP_C) ||           \
      !( defined(MBEDTLS_ECP_DP_SECP192R1_ENABLED) || \
         defined(MBEDTLS_ECP_DP_SECP224R1_ENABLED) || \
         defined(MBEDTLS_ECP_DP_SECP256R1_ENABLED) || \
         defined(MBEDTLS_ECP_DP_SECP384R1_ENABLED) || \
         defined(MBEDTLS_ECP_DP_SECP521R1_ENABLED) || \
         defined(MBEDTLS_ECP_DP_SECP192K1_ENABLED) || \
         defined(MBEDTLS_ECP_DP_SECP224K1_ENABLED) || \
         defined(MBEDTLS_ECP_DP_SECP256K1_ENABLED) || \
         defined(MBEDTLS_ECP_DP_BP256R1_ENABLED) ||   \
         defined(MBEDTLS_ECP_DP_BP384R1_ENABLED) ||   \
         defined(MBEDTLS_ECP_DP_BP512R1_ENABLED) ) || \
      !defined(MBEDTLS_ASN1_PARSE_C) ||    \
      !defined(MBEDTLS_ASN1_WRITE_C) )
#error "MBEDTLS_ECDSA_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECJPAKE_C) &&           \
    ( !defined(MBEDTLS_ECP_C) || !defined(MBEDTLS_MD_C) )
#error "MBEDTLS_ECJPAKE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_RESTARTABLE)           && \
    ( defined(MBEDTLS_ECDH_COMPUTE_SHARED_ALT) || \
      defined(MBEDTLS_ECDH_GEN_PUBLIC_ALT)     || \
      defined(MBEDTLS_ECDSA_SIGN_ALT)          || \
      defined(MBEDTLS_ECDSA_VERIFY_ALT)        || \
      defined(MBEDTLS_ECDSA_GENKEY_ALT)        || \
      defined(MBEDTLS_ECP_INTERNAL_ALT)        || \
      defined(MBEDTLS_ECP_ALT) )
#error "MBEDTLS_ECP_RESTARTABLE defined, but it cannot coexist with an alternative or PSA-based ECP implementation"
#endif

#if defined(MBEDTLS_ECP_RESTARTABLE)           && \
    ! defined(MBEDTLS_ECDH_LEGACY_CONTEXT)
#error "MBEDTLS_ECP_RESTARTABLE defined, but not MBEDTLS_ECDH_LEGACY_CONTEXT"
#endif

#if defined(MBEDTLS_ECDH_VARIANT_EVEREST_ENABLED)           && \
    defined(MBEDTLS_ECDH_LEGACY_CONTEXT)
#error "MBEDTLS_ECDH_VARIANT_EVEREST_ENABLED defined, but MBEDTLS_ECDH_LEGACY_CONTEXT not disabled"
#endif

#if defined(MBEDTLS_ECDSA_DETERMINISTIC) && !defined(MBEDTLS_HMAC_DRBG_C)
#error "MBEDTLS_ECDSA_DETERMINISTIC defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_C) && ( !defined(MBEDTLS_BIGNUM_C) || (    \
    !defined(MBEDTLS_ECP_DP_SECP192R1_ENABLED) &&                  \
    !defined(MBEDTLS_ECP_DP_SECP224R1_ENABLED) &&                  \
    !defined(MBEDTLS_ECP_DP_SECP256R1_ENABLED) &&                  \
    !defined(MBEDTLS_ECP_DP_SECP384R1_ENABLED) &&                  \
    !defined(MBEDTLS_ECP_DP_SECP521R1_ENABLED) &&                  \
    !defined(MBEDTLS_ECP_DP_BP256R1_ENABLED)   &&                  \
    !defined(MBEDTLS_ECP_DP_BP384R1_ENABLED)   &&                  \
    !defined(MBEDTLS_ECP_DP_BP512R1_ENABLED)   &&                  \
    !defined(MBEDTLS_ECP_DP_SECP192K1_ENABLED) &&                  \
    !defined(MBEDTLS_ECP_DP_SECP224K1_ENABLED) &&                  \
    !defined(MBEDTLS_ECP_DP_SECP256K1_ENABLED) &&                  \
    !defined(MBEDTLS_ECP_DP_CURVE25519_ENABLED) &&                 \
    !defined(MBEDTLS_ECP_DP_CURVE448_ENABLED) ) )
#error "MBEDTLS_ECP_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_C) && !(            \
    defined(MBEDTLS_ECP_ALT) ||             \
    defined(MBEDTLS_CTR_DRBG_C) ||          \
    defined(MBEDTLS_HMAC_DRBG_C) ||         \
    defined(MBEDTLS_ECP_NO_INTERNAL_RNG))
#error "MBEDTLS_ECP_C requires a DRBG module unless MBEDTLS_ECP_NO_INTERNAL_RNG is defined or an alternative implementation is used"
#endif

#if defined(MBEDTLS_PK_PARSE_C) && !defined(MBEDTLS_ASN1_PARSE_C)
#error "MBEDTLS_PK_PARSE_C defined, but not all prerequesites"
#endif

#if defined(MBEDTLS_ENTROPY_C) && (!defined(MBEDTLS_SHA512_C) &&      \
                                    !defined(MBEDTLS_SHA256_C))
#error "MBEDTLS_ENTROPY_C defined, but not all prerequisites"
#endif
#if defined(MBEDTLS_ENTROPY_C) && defined(MBEDTLS_SHA512_C) &&         \
    defined(MBEDTLS_CTR_DRBG_ENTROPY_LEN) && (MBEDTLS_CTR_DRBG_ENTROPY_LEN > 64)
#error "MBEDTLS_CTR_DRBG_ENTROPY_LEN value too high"
#endif
#if defined(MBEDTLS_ENTROPY_C) &&                                            \
    ( !defined(MBEDTLS_SHA512_C) || defined(MBEDTLS_ENTROPY_FORCE_SHA256) ) \
    && defined(MBEDTLS_CTR_DRBG_ENTROPY_LEN) && (MBEDTLS_CTR_DRBG_ENTROPY_LEN > 32)
#error "MBEDTLS_CTR_DRBG_ENTROPY_LEN value too high"
#endif
#if defined(MBEDTLS_ENTROPY_C) && \
    defined(MBEDTLS_ENTROPY_FORCE_SHA256) && !defined(MBEDTLS_SHA256_C)
#error "MBEDTLS_ENTROPY_FORCE_SHA256 defined, but not all prerequisites"
#endif

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define MBEDTLS_HAS_MEMSAN
#endif
#endif
#if defined(MBEDTLS_TEST_CONSTANT_FLOW_MEMSAN) &&  !defined(MBEDTLS_HAS_MEMSAN)
#error "MBEDTLS_TEST_CONSTANT_FLOW_MEMSAN requires building with MemorySanitizer"
#endif
#undef MBEDTLS_HAS_MEMSAN

#if defined(MBEDTLS_TEST_NULL_ENTROPY) && \
    ( !defined(MBEDTLS_ENTROPY_C) || !defined(MBEDTLS_NO_DEFAULT_ENTROPY_SOURCES) )
#error "MBEDTLS_TEST_NULL_ENTROPY defined, but not all prerequisites"
#endif
#if defined(MBEDTLS_TEST_NULL_ENTROPY) && \
     ( defined(MBEDTLS_ENTROPY_NV_SEED) || defined(MBEDTLS_ENTROPY_HARDWARE_ALT) || \
    defined(MBEDTLS_HAVEGE_C) )
#error "MBEDTLS_TEST_NULL_ENTROPY defined, but entropy sources too"
#endif

#if defined(MBEDTLS_GCM_C) && (                                        \
        !defined(MBEDTLS_AES_C) && !defined(MBEDTLS_CAMELLIA_C) && !defined(MBEDTLS_ARIA_C) )
#error "MBEDTLS_GCM_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_RANDOMIZE_JAC_ALT) && !defined(MBEDTLS_ECP_INTERNAL_ALT)
#error "MBEDTLS_ECP_RANDOMIZE_JAC_ALT defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_ADD_MIXED_ALT) && !defined(MBEDTLS_ECP_INTERNAL_ALT)
#error "MBEDTLS_ECP_ADD_MIXED_ALT defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_DOUBLE_JAC_ALT) && !defined(MBEDTLS_ECP_INTERNAL_ALT)
#error "MBEDTLS_ECP_DOUBLE_JAC_ALT defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_NORMALIZE_JAC_MANY_ALT) && !defined(MBEDTLS_ECP_INTERNAL_ALT)
#error "MBEDTLS_ECP_NORMALIZE_JAC_MANY_ALT defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_NORMALIZE_JAC_ALT) && !defined(MBEDTLS_ECP_INTERNAL_ALT)
#error "MBEDTLS_ECP_NORMALIZE_JAC_ALT defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_DOUBLE_ADD_MXZ_ALT) && !defined(MBEDTLS_ECP_INTERNAL_ALT)
#error "MBEDTLS_ECP_DOUBLE_ADD_MXZ_ALT defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_RANDOMIZE_MXZ_ALT) && !defined(MBEDTLS_ECP_INTERNAL_ALT)
#error "MBEDTLS_ECP_RANDOMIZE_MXZ_ALT defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_NORMALIZE_MXZ_ALT) && !defined(MBEDTLS_ECP_INTERNAL_ALT)
#error "MBEDTLS_ECP_NORMALIZE_MXZ_ALT defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_ECP_NO_FALLBACK) && !defined(MBEDTLS_ECP_INTERNAL_ALT)
#error "MBEDTLS_ECP_NO_FALLBACK defined, but no alternative implementation enabled"
#endif

#if defined(MBEDTLS_HKDF_C) && !defined(MBEDTLS_MD_C)
#error "MBEDTLS_HKDF_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_HMAC_DRBG_C) && !defined(MBEDTLS_MD_C)
#error "MBEDTLS_HMAC_DRBG_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED) &&                 \
    ( !defined(MBEDTLS_ECDH_C) || !defined(MBEDTLS_ECDSA_C) ||          \
      !defined(MBEDTLS_X509_CRT_PARSE_C) )
#error "MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED) &&                 \
    ( !defined(MBEDTLS_ECDH_C) || !defined(MBEDTLS_RSA_C) ||          \
      !defined(MBEDTLS_X509_CRT_PARSE_C) )
#error "MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED) && !defined(MBEDTLS_DHM_C)
#error "MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED) &&                     \
    !defined(MBEDTLS_ECDH_C)
#error "MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED) &&                   \
    ( !defined(MBEDTLS_DHM_C) || !defined(MBEDTLS_RSA_C) ||           \
      !defined(MBEDTLS_X509_CRT_PARSE_C) || !defined(MBEDTLS_PKCS1_V15) )
#error "MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED) &&                 \
    ( !defined(MBEDTLS_ECDH_C) || !defined(MBEDTLS_RSA_C) ||          \
      !defined(MBEDTLS_X509_CRT_PARSE_C) || !defined(MBEDTLS_PKCS1_V15) )
#error "MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED) &&                 \
    ( !defined(MBEDTLS_ECDH_C) || !defined(MBEDTLS_ECDSA_C) ||          \
      !defined(MBEDTLS_X509_CRT_PARSE_C) )
#error "MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED) &&                   \
    ( !defined(MBEDTLS_RSA_C) || !defined(MBEDTLS_X509_CRT_PARSE_C) || \
      !defined(MBEDTLS_PKCS1_V15) )
#error "MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED) &&                       \
    ( !defined(MBEDTLS_RSA_C) || !defined(MBEDTLS_X509_CRT_PARSE_C) || \
      !defined(MBEDTLS_PKCS1_V15) )
#error "MBEDTLS_KEY_EXCHANGE_RSA_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED) &&                    \
    ( !defined(MBEDTLS_ECJPAKE_C) || !defined(MBEDTLS_SHA256_C) ||      \
      !defined(MBEDTLS_ECP_DP_SECP256R1_ENABLED) )
#error "MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED) &&        \
    !defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE) &&              \
    ( !defined(MBEDTLS_SHA256_C) &&                             \
      !defined(MBEDTLS_SHA512_C) &&                             \
      !defined(MBEDTLS_SHA1_C) )
#error "!MBEDTLS_SSL_KEEP_PEER_CERTIFICATE requires MBEDTLS_SHA512_C, MBEDTLS_SHA256_C or MBEDTLS_SHA1_C"
#endif

#if defined(MBEDTLS_MEMORY_BACKTRACE) && !defined(MBEDTLS_MEMORY_BUFFER_ALLOC_C)
#error "MBEDTLS_MEMORY_BACKTRACE defined, but not all prerequesites"
#endif

#if defined(MBEDTLS_MEMORY_DEBUG) && !defined(MBEDTLS_MEMORY_BUFFER_ALLOC_C)
#error "MBEDTLS_MEMORY_DEBUG defined, but not all prerequesites"
#endif

#if defined(MBEDTLS_PEM_PARSE_C) && !defined(MBEDTLS_BASE64_C)
#error "MBEDTLS_PEM_PARSE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_PEM_WRITE_C) && !defined(MBEDTLS_BASE64_C)
#error "MBEDTLS_PEM_WRITE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_PK_C) && \
    ( !defined(MBEDTLS_RSA_C) && !defined(MBEDTLS_ECP_C) )
#error "MBEDTLS_PK_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_PK_PARSE_C) && !defined(MBEDTLS_PK_C)
#error "MBEDTLS_PK_PARSE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_PK_WRITE_C) && !defined(MBEDTLS_PK_C)
#error "MBEDTLS_PK_WRITE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_PKCS11_C) && !defined(MBEDTLS_PK_C)
#error "MBEDTLS_PKCS11_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_PKCS11_C)
#if defined(MBEDTLS_DEPRECATED_REMOVED)
#error "MBEDTLS_PKCS11_C is deprecated and will be removed in a future version of Mbed TLS"
#elif defined(MBEDTLS_DEPRECATED_WARNING)
#warning "MBEDTLS_PKCS11_C is deprecated and will be removed in a future version of Mbed TLS"
#endif
#endif /* MBEDTLS_PKCS11_C */

#if defined(MBEDTLS_RSA_C) && ( !defined(MBEDTLS_BIGNUM_C) ||         \
    !defined(MBEDTLS_OID_C) )
#error "MBEDTLS_RSA_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_RSA_C) && ( !defined(MBEDTLS_PKCS1_V21) &&         \
    !defined(MBEDTLS_PKCS1_V15) )
#error "MBEDTLS_RSA_C defined, but none of the PKCS1 versions enabled"
#endif

#if defined(MBEDTLS_SHA512_NO_SHA384) && !defined(MBEDTLS_SHA512_C)
#error "MBEDTLS_SHA512_NO_SHA384 defined without MBEDTLS_SHA512_C"
#endif

#if defined(MBEDTLS_SSL_PROTO_SSL3) && ( !defined(MBEDTLS_MD5_C) ||     \
    !defined(MBEDTLS_SHA1_C) )
#error "MBEDTLS_SSL_PROTO_SSL3 defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1) && ( !defined(MBEDTLS_MD5_C) ||     \
    !defined(MBEDTLS_SHA1_C) )
#error "MBEDTLS_SSL_PROTO_TLS1 defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1_1) && ( !defined(MBEDTLS_MD5_C) ||     \
    !defined(MBEDTLS_SHA1_C) )
#error "MBEDTLS_SSL_PROTO_TLS1_1 defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && ( !defined(MBEDTLS_SHA1_C) &&     \
    !defined(MBEDTLS_SHA256_C) && !defined(MBEDTLS_SHA512_C) )
#error "MBEDTLS_SSL_PROTO_TLS1_2 defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1_3_EXPERIMENTAL) && ( !defined(MBEDTLS_HKDF_C) && \
    !defined(MBEDTLS_SHA256_C) && !defined(MBEDTLS_SHA512_C) )
#error "MBEDTLS_SSL_PROTO_TLS1_3_EXPERIMENTAL defined, but not all prerequisites"
#endif

#if (defined(MBEDTLS_SSL_PROTO_SSL3) || defined(MBEDTLS_SSL_PROTO_TLS1) ||  \
     defined(MBEDTLS_SSL_PROTO_TLS1_1) || defined(MBEDTLS_SSL_PROTO_TLS1_2)) && \
    !(defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED) ||                          \
      defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED) ||                      \
      defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED) ||                    \
      defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED) ||                  \
      defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED) ||                     \
      defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED) ||                   \
      defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED) ||                          \
      defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED) ||                      \
      defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED) ||                      \
      defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED) ||                    \
      defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED) )
#error "One or more versions of the TLS protocol are enabled " \
        "but no key exchange methods defined with MBEDTLS_KEY_EXCHANGE_xxxx"
#endif

#if defined(MBEDTLS_SSL_PROTO_DTLS)     && \
    !defined(MBEDTLS_SSL_PROTO_TLS1_1)  && \
    !defined(MBEDTLS_SSL_PROTO_TLS1_2)
#error "MBEDTLS_SSL_PROTO_DTLS defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_CLI_C) && !defined(MBEDTLS_SSL_TLS_C)
#error "MBEDTLS_SSL_CLI_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_TLS_C) && ( !defined(MBEDTLS_CIPHER_C) ||     \
    !defined(MBEDTLS_MD_C) )
#error "MBEDTLS_SSL_TLS_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_SRV_C) && !defined(MBEDTLS_SSL_TLS_C)
#error "MBEDTLS_SSL_SRV_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_TLS_C) && (!defined(MBEDTLS_SSL_PROTO_SSL3) && \
    !defined(MBEDTLS_SSL_PROTO_TLS1) && !defined(MBEDTLS_SSL_PROTO_TLS1_1) && \
    !defined(MBEDTLS_SSL_PROTO_TLS1_2))
#error "MBEDTLS_SSL_TLS_C defined, but no protocols are active"
#endif

#if defined(MBEDTLS_SSL_TLS_C) && (defined(MBEDTLS_SSL_PROTO_SSL3) && \
    defined(MBEDTLS_SSL_PROTO_TLS1_1) && !defined(MBEDTLS_SSL_PROTO_TLS1))
#error "Illegal protocol selection"
#endif

#if defined(MBEDTLS_SSL_TLS_C) && (defined(MBEDTLS_SSL_PROTO_TLS1) && \
    defined(MBEDTLS_SSL_PROTO_TLS1_2) && !defined(MBEDTLS_SSL_PROTO_TLS1_1))
#error "Illegal protocol selection"
#endif

#if defined(MBEDTLS_SSL_TLS_C) && (defined(MBEDTLS_SSL_PROTO_SSL3) && \
    defined(MBEDTLS_SSL_PROTO_TLS1_2) && (!defined(MBEDTLS_SSL_PROTO_TLS1) || \
    !defined(MBEDTLS_SSL_PROTO_TLS1_1)))
#error "Illegal protocol selection"
#endif

#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && !defined(MBEDTLS_SSL_PROTO_DTLS)
#error "MBEDTLS_SSL_DTLS_HELLO_VERIFY  defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE) && \
    !defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY)
#error "MBEDTLS_SSL_DTLS_CLIENT_PORT_REUSE  defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY) &&                              \
    ( !defined(MBEDTLS_SSL_TLS_C) || !defined(MBEDTLS_SSL_PROTO_DTLS) )
#error "MBEDTLS_SSL_DTLS_ANTI_REPLAY  defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID) &&                              \
    ( !defined(MBEDTLS_SSL_TLS_C) || !defined(MBEDTLS_SSL_PROTO_DTLS) )
#error "MBEDTLS_SSL_DTLS_CONNECTION_ID  defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)            &&                 \
    defined(MBEDTLS_SSL_CID_IN_LEN_MAX) &&                 \
    MBEDTLS_SSL_CID_IN_LEN_MAX > 255
#error "MBEDTLS_SSL_CID_IN_LEN_MAX too large (max 255)"
#endif

#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)            &&                  \
    defined(MBEDTLS_SSL_CID_OUT_LEN_MAX) &&                 \
    MBEDTLS_SSL_CID_OUT_LEN_MAX > 255
#error "MBEDTLS_SSL_CID_OUT_LEN_MAX too large (max 255)"
#endif

#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT) &&                              \
    ( !defined(MBEDTLS_SSL_TLS_C) || !defined(MBEDTLS_SSL_PROTO_DTLS) )
#error "MBEDTLS_SSL_DTLS_BADMAC_LIMIT  defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC) &&   \
    !defined(MBEDTLS_SSL_PROTO_TLS1)   &&      \
    !defined(MBEDTLS_SSL_PROTO_TLS1_1) &&      \
    !defined(MBEDTLS_SSL_PROTO_TLS1_2)
#error "MBEDTLS_SSL_ENCRYPT_THEN_MAC defined, but not all prerequsites"
#endif

#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET) && \
    !defined(MBEDTLS_SSL_PROTO_TLS1)   &&          \
    !defined(MBEDTLS_SSL_PROTO_TLS1_1) &&          \
    !defined(MBEDTLS_SSL_PROTO_TLS1_2)
#error "MBEDTLS_SSL_EXTENDED_MASTER_SECRET defined, but not all prerequsites"
#endif

#if defined(MBEDTLS_SSL_TICKET_C) && !defined(MBEDTLS_CIPHER_C)
#error "MBEDTLS_SSL_TICKET_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_CBC_RECORD_SPLITTING) && \
    !defined(MBEDTLS_SSL_PROTO_SSL3) && !defined(MBEDTLS_SSL_PROTO_TLS1)
#undef MBEDTLS_SSL_CBC_RECORD_SPLITTING
#endif

#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION) && \
        !defined(MBEDTLS_X509_CRT_PARSE_C)
#error "MBEDTLS_SSL_SERVER_NAME_INDICATION defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_VERSION_FEATURES) && !defined(MBEDTLS_VERSION_C)
#error "MBEDTLS_VERSION_FEATURES defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_X509_USE_C) && ( !defined(MBEDTLS_BIGNUM_C) ||  \
    !defined(MBEDTLS_OID_C) || !defined(MBEDTLS_ASN1_PARSE_C) ||      \
    !defined(MBEDTLS_PK_PARSE_C) )
#error "MBEDTLS_X509_USE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_X509_CREATE_C) && ( !defined(MBEDTLS_BIGNUM_C) ||  \
    !defined(MBEDTLS_OID_C) || !defined(MBEDTLS_ASN1_WRITE_C) ||       \
    !defined(MBEDTLS_PK_WRITE_C) )
#error "MBEDTLS_X509_CREATE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_CERTS_C) && !defined(MBEDTLS_X509_USE_C)
#error "MBEDTLS_CERTS_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_X509_CRT_PARSE_C) && ( !defined(MBEDTLS_X509_USE_C) )
#error "MBEDTLS_X509_CRT_PARSE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_X509_CRL_PARSE_C) && ( !defined(MBEDTLS_X509_USE_C) )
#error "MBEDTLS_X509_CRL_PARSE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_X509_CSR_PARSE_C) && ( !defined(MBEDTLS_X509_USE_C) )
#error "MBEDTLS_X509_CSR_PARSE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_X509_CRT_WRITE_C) && ( !defined(MBEDTLS_X509_CREATE_C) )
#error "MBEDTLS_X509_CRT_WRITE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_X509_CSR_WRITE_C) && ( !defined(MBEDTLS_X509_CREATE_C) )
#error "MBEDTLS_X509_CSR_WRITE_C defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_HAVE_INT32) && defined(MBEDTLS_HAVE_INT64)
#error "MBEDTLS_HAVE_INT32 and MBEDTLS_HAVE_INT64 cannot be defined simultaneously"
#endif /* MBEDTLS_HAVE_INT32 && MBEDTLS_HAVE_INT64 */

#if defined(MBEDTLS_SSL_PROTO_SSL3)
#if defined(MBEDTLS_DEPRECATED_REMOVED)
#error "MBEDTLS_SSL_PROTO_SSL3 is deprecated and will be removed in a future version of Mbed TLS"
#elif defined(MBEDTLS_DEPRECATED_WARNING)
#warning "MBEDTLS_SSL_PROTO_SSL3 is deprecated and will be removed in a future version of Mbed TLS"
#endif
#endif /* MBEDTLS_SSL_PROTO_SSL3 */

#if defined(MBEDTLS_SSL_DTLS_SRTP) && ( !defined(MBEDTLS_SSL_PROTO_DTLS) )
#error "MBEDTLS_SSL_DTLS_SRTP defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH) && ( !defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH) )
#error "MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_GCM_C) || defined(MBEDTLS_CCM_C) || defined(MBEDTLS_CHACHAPOLY_C)
#define MBEDTLS_CIPHER_MODE_AEAD
#endif

#if defined(MBEDTLS_CIPHER_MODE_CBC)
#define MBEDTLS_CIPHER_MODE_WITH_PADDING
#endif

#if defined(MBEDTLS_ARC4_C) || defined(MBEDTLS_CIPHER_NULL_CIPHER) || \
    defined(MBEDTLS_CHACHA20_C)
#define MBEDTLS_CIPHER_MODE_STREAM
#endif

/* Key exchanges using a certificate */
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED)           || \
    defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)       || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED)     || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)   || \
    defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)       || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED)      || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)
#define MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED
#endif

/* Key exchanges allowing client certificate requests */
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED)           ||       \
    defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)       ||       \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED)      ||       \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED)     ||       \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)    ||       \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)
#define MBEDTLS_KEY_EXCHANGE_CERT_REQ_ALLOWED_ENABLED
#endif

/* Key exchanges involving server signature in ServerKeyExchange */
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)       || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED)     || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)
#define MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED
#endif

/* Key exchanges using ECDH */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED)      || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)
#define MBEDTLS_KEY_EXCHANGE_SOME_ECDH_ENABLED
#endif

/* Key exchanges that don't involve ephemeral keys */
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED)           || \
    defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED)           || \
    defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)       || \
    defined(MBEDTLS_KEY_EXCHANGE_SOME_ECDH_ENABLED)
#define MBEDTLS_KEY_EXCHANGE_SOME_NON_PFS_ENABLED
#endif

/* Key exchanges that involve ephemeral keys */
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)       || \
    defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)       || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED)     || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)     || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)   || \
    defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
#define MBEDTLS_KEY_EXCHANGE_SOME_PFS_ENABLED
#endif

/* Key exchanges using a PSK */
#if defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED)           || \
    defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)       || \
    defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)       || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
#define MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED
#endif

/* Key exchanges using DHE */
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)       || \
    defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)
#define MBEDTLS_KEY_EXCHANGE_SOME_DHE_ENABLED
#endif

/* Key exchanges using ECDHE */
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED)     || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)   || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
#define MBEDTLS_KEY_EXCHANGE_SOME_ECDHE_ENABLED
#endif

/* Shorthand for restartable ECC */
#if defined(MBEDTLS_ECP_RESTARTABLE) && \
    defined(MBEDTLS_SSL_CLI_C) && \
    defined(MBEDTLS_SSL_PROTO_TLS1_2) && \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)
#define MBEDTLS_SSL_ECP_RESTARTABLE_ENABLED
#endif

/* Flags indicating whether to include code that is specific to certain
 * types of curves. These flags are for internal library use only. */
#if defined(MBEDTLS_ECP_DP_SECP192R1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_SECP224R1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_SECP256R1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_SECP384R1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_SECP521R1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_BP256R1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_BP384R1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_BP512R1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_SECP192K1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_SECP224K1_ENABLED) || \
    defined(MBEDTLS_ECP_DP_SECP256K1_ENABLED)
#define MBEDTLS_ECP_SHORT_WEIERSTRASS_ENABLED
#endif
#if defined(MBEDTLS_ECP_DP_CURVE25519_ENABLED) || \
    defined(MBEDTLS_ECP_DP_CURVE448_ENABLED)
#define MBEDTLS_ECP_MONTGOMERY_ENABLED
#endif

/* This macro determines whether CBC is supported. */
#if defined(MBEDTLS_CIPHER_MODE_CBC) &&                               \
    ( defined(MBEDTLS_AES_C)      ||                                  \
      defined(MBEDTLS_CAMELLIA_C) ||                                  \
      defined(MBEDTLS_ARIA_C)     ||                                  \
      defined(MBEDTLS_DES_C) )
#define MBEDTLS_SSL_SOME_SUITES_USE_CBC
#endif

/* This macro determines whether the CBC construct used in TLS 1.0-1.2 (as
 * opposed to the very different CBC construct used in SSLv3) is supported. */
#if defined(MBEDTLS_SSL_SOME_SUITES_USE_CBC) && \
    ( defined(MBEDTLS_SSL_PROTO_TLS1) ||        \
      defined(MBEDTLS_SSL_PROTO_TLS1_1) ||      \
      defined(MBEDTLS_SSL_PROTO_TLS1_2) )
#define MBEDTLS_SSL_SOME_SUITES_USE_TLS_CBC
#endif

#if defined(MBEDTLS_ARC4_C) || defined(MBEDTLS_CIPHER_NULL_CIPHER) ||   \
    defined(MBEDTLS_SSL_SOME_SUITES_USE_CBC)
#define MBEDTLS_SSL_SOME_MODES_USE_MAC
#endif

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED) &&        \
    !defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
#define MBEDTLS_SSL_PEER_CERT_DIGEST_MAX_LEN  48
#if defined(MBEDTLS_SHA256_C)
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_TYPE MBEDTLS_MD_SHA256
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_LEN  32
#elif defined(MBEDTLS_SHA512_C)
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_TYPE MBEDTLS_MD_SHA384
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_LEN  48
#elif defined(MBEDTLS_SHA1_C)
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_TYPE MBEDTLS_MD_SHA1
#define MBEDTLS_SSL_PEER_CERT_DIGEST_DFL_LEN  20
#else
/* This is already checked in check.h, but be sure. */
#error "Bad configuration - need SHA-1, SHA-256 or SHA-512 enabled to compute digest of peer CRT."
#endif
#endif

/*
 * Avoid warning from -pedantic. This is a convenient place for this
 * workaround since this is included by every single file before the
 * #if defined(MBEDTLS_xxx_C) that results in empty translation units.
 */
typedef int mbedtls_iso_c_forbids_empty_translation_units;
