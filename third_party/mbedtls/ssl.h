#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_H_
#include "third_party/mbedtls/bignum.h"
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/dhm.h"
#include "third_party/mbedtls/ecdh.h"
#include "third_party/mbedtls/ecp.h"
#include "third_party/mbedtls/platform.h"
#include "third_party/mbedtls/ssl_ciphersuites.h"
#include "third_party/mbedtls/x509_crl.h"
#include "third_party/mbedtls/x509_crt.h"
COSMOPOLITAN_C_START_

/*
 * SSL Error codes
 */
#define MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE               -0x7080  /*< The requested feature is not available. */
#define MBEDTLS_ERR_SSL_BAD_INPUT_DATA                    -0x7100  /*< Bad input parameters to function. */
#define MBEDTLS_ERR_SSL_INVALID_MAC                       -0x7180  /*< Verification of the message MAC failed. */
#define MBEDTLS_ERR_SSL_INVALID_RECORD                    -0x7200  /*< An invalid SSL record was received. */
#define MBEDTLS_ERR_SSL_CONN_EOF                          -0x7280  /*< The connection indicated an EOF. */
#define MBEDTLS_ERR_SSL_UNKNOWN_CIPHER                    -0x7300  /*< An unknown cipher was received. */
#define MBEDTLS_ERR_SSL_NO_CIPHER_CHOSEN                  -0x7380  /*< The server has no ciphersuites in common with the client. */
#define MBEDTLS_ERR_SSL_NO_RNG                            -0x7400  /*< No RNG was provided to the SSL module. */
#define MBEDTLS_ERR_SSL_NO_CLIENT_CERTIFICATE             -0x7480  /*< No client certification received from the client, but required by the authentication mode. */
#define MBEDTLS_ERR_SSL_CERTIFICATE_TOO_LARGE             -0x7500  /*< Our own certificate(s) is/are too large to send in an SSL message. */
#define MBEDTLS_ERR_SSL_CERTIFICATE_REQUIRED              -0x7580  /*< The own certificate is not set, but needed by the server. */
#define MBEDTLS_ERR_SSL_PRIVATE_KEY_REQUIRED              -0x7600  /*< The own private key or pre-shared key is not set, but needed. */
#define MBEDTLS_ERR_SSL_CA_CHAIN_REQUIRED                 -0x7680  /*< No CA Chain is set, but required to operate. */
#define MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE                -0x7700  /*< An unexpected message was received from our peer. */
#define MBEDTLS_ERR_SSL_FATAL_ALERT_MESSAGE               -0x7780  /*< A fatal alert message was received from our peer. */
#define MBEDTLS_ERR_SSL_PEER_VERIFY_FAILED                -0x7800  /*< Verification of our peer failed. */
#define MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY                 -0x7880  /*< The peer notified us that the connection is going to be closed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CLIENT_HELLO               -0x7900  /*< Processing of the ClientHello handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO               -0x7980  /*< Processing of the ServerHello handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE                -0x7A00  /*< Processing of the Certificate handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_REQUEST        -0x7A80  /*< Processing of the CertificateRequest handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE        -0x7B00  /*< Processing of the ServerKeyExchange handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_SERVER_HELLO_DONE          -0x7B80  /*< Processing of the ServerHelloDone handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE        -0x7C00  /*< Processing of the ClientKeyExchange handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_RP     -0x7C80  /*< Processing of the ClientKeyExchange handshake message failed in DHM / ECDH Read Public. */
#define MBEDTLS_ERR_SSL_BAD_HS_CLIENT_KEY_EXCHANGE_CS     -0x7D00  /*< Processing of the ClientKeyExchange handshake message failed in DHM / ECDH Calculate Secret. */
#define MBEDTLS_ERR_SSL_BAD_HS_CERTIFICATE_VERIFY         -0x7D80  /*< Processing of the CertificateVerify handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_CHANGE_CIPHER_SPEC         -0x7E00  /*< Processing of the ChangeCipherSpec handshake message failed. */
#define MBEDTLS_ERR_SSL_BAD_HS_FINISHED                   -0x7E80  /*< Processing of the Finished handshake message failed. */
#define MBEDTLS_ERR_SSL_ALLOC_FAILED                      -0x7F00  /*< Memory allocation failed */
#define MBEDTLS_ERR_SSL_HW_ACCEL_FAILED                   -0x7F80  /*< Hardware acceleration function returned with error */
#define MBEDTLS_ERR_SSL_HW_ACCEL_FALLTHROUGH              -0x6F80  /*< Hardware acceleration function skipped / left alone data */
#define MBEDTLS_ERR_SSL_COMPRESSION_FAILED                -0x6F00  /*< Processing of the compression / decompression failed */
#define MBEDTLS_ERR_SSL_BAD_HS_PROTOCOL_VERSION           -0x6E80  /*< Handshake protocol not within min/max boundaries */
#define MBEDTLS_ERR_SSL_BAD_HS_NEW_SESSION_TICKET         -0x6E00  /*< Processing of the NewSessionTicket handshake message failed. */
#define MBEDTLS_ERR_SSL_SESSION_TICKET_EXPIRED            -0x6D80  /*< Session ticket has expired. */
#define MBEDTLS_ERR_SSL_PK_TYPE_MISMATCH                  -0x6D00  /*< Public key type mismatch (eg, asked for RSA key exchange and presented EC key) */
#define MBEDTLS_ERR_SSL_UNKNOWN_IDENTITY                  -0x6C80  /*< Unknown identity received (eg, PSK identity) */
#define MBEDTLS_ERR_SSL_INTERNAL_ERROR                    -0x6C00  /*< Internal error (eg, unexpected failure in lower-level module) */
#define MBEDTLS_ERR_SSL_COUNTER_WRAPPING                  -0x6B80  /*< A counter would wrap (eg, too many messages exchanged). */
#define MBEDTLS_ERR_SSL_WAITING_SERVER_HELLO_RENEGO       -0x6B00  /*< Unexpected message at ServerHello in renegotiation. */
#define MBEDTLS_ERR_SSL_HELLO_VERIFY_REQUIRED             -0x6A80  /*< DTLS client must retry for hello verification */
#define MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL                  -0x6A00  /*< A buffer is too small to receive or write a message */
#define MBEDTLS_ERR_SSL_NO_USABLE_CIPHERSUITE             -0x6980  /*< None of the common ciphersuites is usable (eg, no suitable certificate, see debug messages). */
#define MBEDTLS_ERR_SSL_WANT_READ                         -0x6900  /*< No data of requested type currently available on underlying transport. */
#define MBEDTLS_ERR_SSL_WANT_WRITE                        -0x6880  /*< Connection requires a write call. */
#define MBEDTLS_ERR_SSL_CANCELED                          -0x9900  /*< The POSIX thread was canceled. */
#define MBEDTLS_ERR_SSL_TIMEOUT                           -0x6800  /*< The operation timed out. */
#define MBEDTLS_ERR_SSL_CLIENT_RECONNECT                  -0x6780  /*< The client initiated a reconnect from the same port. */
#define MBEDTLS_ERR_SSL_UNEXPECTED_RECORD                 -0x6700  /*< Record header looks valid but is not expected. */
#define MBEDTLS_ERR_SSL_NON_FATAL                         -0x6680  /*< The alert message received indicates a non-fatal error. */
#define MBEDTLS_ERR_SSL_INVALID_VERIFY_HASH               -0x6600  /*< Couldn't set the hash for verifying CertificateVerify */
#define MBEDTLS_ERR_SSL_CONTINUE_PROCESSING               -0x6580  /*< Internal-only message signaling that further message-processing should be done */
#define MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS                 -0x6500  /*< The asynchronous operation is not completed yet. */
#define MBEDTLS_ERR_SSL_EARLY_MESSAGE                     -0x6480  /*< Internal-only message signaling that a message arrived early. */
#define MBEDTLS_ERR_SSL_UNEXPECTED_CID                    -0x6000  /*< An encrypted DTLS-frame with an unexpected CID was received. */
#define MBEDTLS_ERR_SSL_VERSION_MISMATCH                  -0x5F00  /*< An operation failed due to an unexpected version or configuration. */
#define MBEDTLS_ERR_SSL_CRYPTO_IN_PROGRESS                -0x7000  /*< A cryptographic operation is in progress. Try again later. */
#define MBEDTLS_ERR_SSL_BAD_CONFIG                        -0x5E80  /*< Invalid value in SSL config */

/*
 * Various constants
 */
#define MBEDTLS_SSL_MAJOR_VERSION_3             3
#define MBEDTLS_SSL_MINOR_VERSION_0             0   /*!< SSL v3.0 */
#define MBEDTLS_SSL_MINOR_VERSION_1             1   /*!< TLS v1.0 */
#define MBEDTLS_SSL_MINOR_VERSION_2             2   /*!< TLS v1.1 */
#define MBEDTLS_SSL_MINOR_VERSION_3             3   /*!< TLS v1.2 */
#define MBEDTLS_SSL_MINOR_VERSION_4             4   /*!< TLS v1.3 (experimental) */

#define MBEDTLS_SSL_TRANSPORT_STREAM            0   /*!< TLS      */
#define MBEDTLS_SSL_TRANSPORT_DATAGRAM          1   /*!< DTLS     */

#define MBEDTLS_SSL_MAX_HOST_NAME_LEN           255 /*!< Maximum host name defined in RFC 1035 */
#define MBEDTLS_SSL_MAX_ALPN_NAME_LEN           255 /*!< Maximum size in bytes of a protocol name in alpn ext., RFC 7301 */

#define MBEDTLS_SSL_MAX_ALPN_LIST_LEN           65535 /*!< Maximum size in bytes of list in alpn ext., RFC 7301          */

/* RFC 6066 section 4, see also mfl_code_to_length in ssl_tls.c
 * NONE must be zero so that memset()ing structure to zero works */
#define MBEDTLS_SSL_MAX_FRAG_LEN_NONE           0   /*!< don't use this extension   */
#define MBEDTLS_SSL_MAX_FRAG_LEN_512            1   /*!< MaxFragmentLength 2^9      */
#define MBEDTLS_SSL_MAX_FRAG_LEN_1024           2   /*!< MaxFragmentLength 2^10     */
#define MBEDTLS_SSL_MAX_FRAG_LEN_2048           3   /*!< MaxFragmentLength 2^11     */
#define MBEDTLS_SSL_MAX_FRAG_LEN_4096           4   /*!< MaxFragmentLength 2^12     */
#define MBEDTLS_SSL_MAX_FRAG_LEN_INVALID        5   /*!< first invalid value        */

#define MBEDTLS_SSL_IS_CLIENT                   0
#define MBEDTLS_SSL_IS_SERVER                   1

#define MBEDTLS_SSL_IS_NOT_FALLBACK             0
#define MBEDTLS_SSL_IS_FALLBACK                 1

#define MBEDTLS_SSL_EXTENDED_MS_DISABLED        0
#define MBEDTLS_SSL_EXTENDED_MS_ENABLED         1

#define MBEDTLS_SSL_CID_DISABLED                0
#define MBEDTLS_SSL_CID_ENABLED                 1

#define MBEDTLS_SSL_ETM_DISABLED                0
#define MBEDTLS_SSL_ETM_ENABLED                 1

#define MBEDTLS_SSL_COMPRESS_NULL               0
#define MBEDTLS_SSL_COMPRESS_DEFLATE            1

#define MBEDTLS_SSL_VERIFY_NONE                 0
#define MBEDTLS_SSL_VERIFY_OPTIONAL             1
#define MBEDTLS_SSL_VERIFY_REQUIRED             2
#define MBEDTLS_SSL_VERIFY_UNSET                3 /* Used only for sni_authmode */

#define MBEDTLS_SSL_LEGACY_RENEGOTIATION        0
#define MBEDTLS_SSL_SECURE_RENEGOTIATION        1

#define MBEDTLS_SSL_RENEGOTIATION_DISABLED      0
#define MBEDTLS_SSL_RENEGOTIATION_ENABLED       1

#define MBEDTLS_SSL_ANTI_REPLAY_DISABLED        0
#define MBEDTLS_SSL_ANTI_REPLAY_ENABLED         1

#define MBEDTLS_SSL_RENEGOTIATION_NOT_ENFORCED  -1
#define MBEDTLS_SSL_RENEGO_MAX_RECORDS_DEFAULT  16

#define MBEDTLS_SSL_LEGACY_NO_RENEGOTIATION     0
#define MBEDTLS_SSL_LEGACY_ALLOW_RENEGOTIATION  1
#define MBEDTLS_SSL_LEGACY_BREAK_HANDSHAKE      2

#define MBEDTLS_SSL_TRUNC_HMAC_DISABLED         0
#define MBEDTLS_SSL_TRUNC_HMAC_ENABLED          1
#define MBEDTLS_SSL_TRUNCATED_HMAC_LEN          10  /* 80 bits, rfc 6066 section 7 */

#define MBEDTLS_SSL_SESSION_TICKETS_DISABLED     0
#define MBEDTLS_SSL_SESSION_TICKETS_ENABLED      1

#define MBEDTLS_SSL_CBC_RECORD_SPLITTING_DISABLED    0
#define MBEDTLS_SSL_CBC_RECORD_SPLITTING_ENABLED     1

#define MBEDTLS_SSL_ARC4_ENABLED                0
#define MBEDTLS_SSL_ARC4_DISABLED               1

#define MBEDTLS_SSL_PRESET_DEFAULT              MBEDTLS_SSL_PRESET_SUITEC
#define MBEDTLS_SSL_PRESET_SUITEB               2
#define MBEDTLS_SSL_PRESET_SUITEC               0

#define MBEDTLS_SSL_CERT_REQ_CA_LIST_ENABLED       1
#define MBEDTLS_SSL_CERT_REQ_CA_LIST_DISABLED      0

#define MBEDTLS_SSL_DTLS_SRTP_MKI_UNSUPPORTED    0
#define MBEDTLS_SSL_DTLS_SRTP_MKI_SUPPORTED      1

#define MBEDTLS_SSL_UNEXPECTED_CID_IGNORE 0
#define MBEDTLS_SSL_UNEXPECTED_CID_FAIL   1

/*
 * Default range for DTLS retransmission timer value, in milliseconds.
 * RFC 6347 4.2.4.1 says from 1 second to 60 seconds.
 */
#define MBEDTLS_SSL_DTLS_TIMEOUT_DFL_MIN    1000
#define MBEDTLS_SSL_DTLS_TIMEOUT_DFL_MAX   60000

/**
 * \name SECTION: Module settings
 *
 * The configuration options you can set for this module are in this section.
 * Either change them in config.h or define them on the compiler command line.
 * \{
 */

#if !defined(MBEDTLS_SSL_DEFAULT_TICKET_LIFETIME)
#define MBEDTLS_SSL_DEFAULT_TICKET_LIFETIME     86400 /*< Lifetime of session tickets (if enabled) */
#endif

/*
 * Maximum fragment length in bytes,
 * determines the size of each of the two internal I/O buffers.
 *
 * Note: the RFC defines the default size of SSL / TLS messages. If you
 * change the value here, other clients / servers may not be able to
 * communicate with you anymore. Only change this value if you control
 * both sides of the connection and have it reduced at both sides, or
 * if you're using the Max Fragment Length extension and you know all your
 * peers are using it too!
 */
#if !defined(MBEDTLS_SSL_MAX_CONTENT_LEN)
#define MBEDTLS_SSL_MAX_CONTENT_LEN         16384   /*< Size of the input / output buffer */
#endif

#if !defined(MBEDTLS_SSL_IN_CONTENT_LEN)
#define MBEDTLS_SSL_IN_CONTENT_LEN MBEDTLS_SSL_MAX_CONTENT_LEN
#endif

#if !defined(MBEDTLS_SSL_OUT_CONTENT_LEN)
#define MBEDTLS_SSL_OUT_CONTENT_LEN MBEDTLS_SSL_MAX_CONTENT_LEN
#endif

/*
 * Maximum number of heap-allocated bytes for the purpose of
 * DTLS handshake message reassembly and future message buffering.
 */
#if !defined(MBEDTLS_SSL_DTLS_MAX_BUFFERING)
#define MBEDTLS_SSL_DTLS_MAX_BUFFERING 32768
#endif

/*
 * Maximum length of CIDs for incoming and outgoing messages.
 */
#if !defined(MBEDTLS_SSL_CID_IN_LEN_MAX)
#define MBEDTLS_SSL_CID_IN_LEN_MAX          32
#endif

#if !defined(MBEDTLS_SSL_CID_OUT_LEN_MAX)
#define MBEDTLS_SSL_CID_OUT_LEN_MAX         32
#endif

#if !defined(MBEDTLS_SSL_CID_PADDING_GRANULARITY)
#define MBEDTLS_SSL_CID_PADDING_GRANULARITY 16
#endif

#if !defined(MBEDTLS_SSL_TLS1_3_PADDING_GRANULARITY)
#define MBEDTLS_SSL_TLS1_3_PADDING_GRANULARITY 1
#endif

/*
 * Length of the verify data for secure renegotiation
 */
#if defined(MBEDTLS_SSL_PROTO_SSL3)
#define MBEDTLS_SSL_VERIFY_DATA_MAX_LEN 36
#else
#define MBEDTLS_SSL_VERIFY_DATA_MAX_LEN 12
#endif

/*
 * Signaling ciphersuite values (SCSV)
 */
#define MBEDTLS_SSL_EMPTY_RENEGOTIATION_INFO    0xFF   /*< renegotiation info ext */
#define MBEDTLS_SSL_FALLBACK_SCSV_VALUE         0x5600 /*< RFC 7507 section 2 */

/*
 * Supported Signature and Hash algorithms (For TLS 1.2)
 * RFC 5246 section 7.4.1.4.1
 */
#define MBEDTLS_SSL_HASH_NONE                0
#define MBEDTLS_SSL_HASH_MD5                 1
#define MBEDTLS_SSL_HASH_SHA1                2
#define MBEDTLS_SSL_HASH_SHA224              3
#define MBEDTLS_SSL_HASH_SHA256              4
#define MBEDTLS_SSL_HASH_SHA384              5
#define MBEDTLS_SSL_HASH_SHA512              6

#define MBEDTLS_SSL_SIG_ANON                 0
#define MBEDTLS_SSL_SIG_RSA                  1
#define MBEDTLS_SSL_SIG_ECDSA                3

/*
 * Client Certificate Types
 * RFC 5246 section 7.4.4 plus RFC 4492 section 5.5
 */
#define MBEDTLS_SSL_CERT_TYPE_RSA_SIGN       1
#define MBEDTLS_SSL_CERT_TYPE_ECDSA_SIGN    64

/*
 * Message, alert and handshake types
 */
#define MBEDTLS_SSL_MSG_CHANGE_CIPHER_SPEC     20
#define MBEDTLS_SSL_MSG_ALERT                  21
#define MBEDTLS_SSL_MSG_HANDSHAKE              22
#define MBEDTLS_SSL_MSG_APPLICATION_DATA       23
#define MBEDTLS_SSL_MSG_CID                    25

#define MBEDTLS_SSL_ALERT_LEVEL_WARNING         1
#define MBEDTLS_SSL_ALERT_LEVEL_FATAL           2

#define MBEDTLS_SSL_ALERT_MSG_CLOSE_NOTIFY           0  /* 0x00 */
#define MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE    10  /* 0x0A */
#define MBEDTLS_SSL_ALERT_MSG_BAD_RECORD_MAC        20  /* 0x14 */
#define MBEDTLS_SSL_ALERT_MSG_DECRYPTION_FAILED     21  /* 0x15 */
#define MBEDTLS_SSL_ALERT_MSG_RECORD_OVERFLOW       22  /* 0x16 */
#define MBEDTLS_SSL_ALERT_MSG_DECOMPRESSION_FAILURE 30  /* 0x1E */
#define MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE     40  /* 0x28 */
#define MBEDTLS_SSL_ALERT_MSG_NO_CERT               41  /* 0x29 */
#define MBEDTLS_SSL_ALERT_MSG_BAD_CERT              42  /* 0x2A */
#define MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_CERT      43  /* 0x2B */
#define MBEDTLS_SSL_ALERT_MSG_CERT_REVOKED          44  /* 0x2C */
#define MBEDTLS_SSL_ALERT_MSG_CERT_EXPIRED          45  /* 0x2D */
#define MBEDTLS_SSL_ALERT_MSG_CERT_UNKNOWN          46  /* 0x2E */
#define MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER     47  /* 0x2F */
#define MBEDTLS_SSL_ALERT_MSG_UNKNOWN_CA            48  /* 0x30 */
#define MBEDTLS_SSL_ALERT_MSG_ACCESS_DENIED         49  /* 0x31 */
#define MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR          50  /* 0x32 */
#define MBEDTLS_SSL_ALERT_MSG_DECRYPT_ERROR         51  /* 0x33 */
#define MBEDTLS_SSL_ALERT_MSG_EXPORT_RESTRICTION    60  /* 0x3C */
#define MBEDTLS_SSL_ALERT_MSG_PROTOCOL_VERSION      70  /* 0x46 */
#define MBEDTLS_SSL_ALERT_MSG_INSUFFICIENT_SECURITY 71  /* 0x47 */
#define MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR        80  /* 0x50 */
#define MBEDTLS_SSL_ALERT_MSG_INAPROPRIATE_FALLBACK 86  /* 0x56 */
#define MBEDTLS_SSL_ALERT_MSG_USER_CANCELED         90  /* 0x5A */
#define MBEDTLS_SSL_ALERT_MSG_NO_RENEGOTIATION     100  /* 0x64 */
#define MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_EXT      110  /* 0x6E */
#define MBEDTLS_SSL_ALERT_MSG_UNRECOGNIZED_NAME    112  /* 0x70 */
#define MBEDTLS_SSL_ALERT_MSG_UNKNOWN_PSK_IDENTITY 115  /* 0x73 */
#define MBEDTLS_SSL_ALERT_MSG_NO_APPLICATION_PROTOCOL 120 /* 0x78 */

#define MBEDTLS_SSL_HS_HELLO_REQUEST            0
#define MBEDTLS_SSL_HS_CLIENT_HELLO             1
#define MBEDTLS_SSL_HS_SERVER_HELLO             2
#define MBEDTLS_SSL_HS_HELLO_VERIFY_REQUEST     3
#define MBEDTLS_SSL_HS_NEW_SESSION_TICKET       4
#define MBEDTLS_SSL_HS_CERTIFICATE             11
#define MBEDTLS_SSL_HS_SERVER_KEY_EXCHANGE     12
#define MBEDTLS_SSL_HS_CERTIFICATE_REQUEST     13
#define MBEDTLS_SSL_HS_SERVER_HELLO_DONE       14
#define MBEDTLS_SSL_HS_CERTIFICATE_VERIFY      15
#define MBEDTLS_SSL_HS_CLIENT_KEY_EXCHANGE     16
#define MBEDTLS_SSL_HS_FINISHED                20

/*
 * TLS extensions
 */
#define MBEDTLS_TLS_EXT_SERVERNAME                   0
#define MBEDTLS_TLS_EXT_SERVERNAME_HOSTNAME          0

#define MBEDTLS_TLS_EXT_MAX_FRAGMENT_LENGTH          1

#define MBEDTLS_TLS_EXT_TRUNCATED_HMAC               4

#define MBEDTLS_TLS_EXT_SUPPORTED_ELLIPTIC_CURVES   10
#define MBEDTLS_TLS_EXT_SUPPORTED_POINT_FORMATS     11

#define MBEDTLS_TLS_EXT_SIG_ALG                     13

#define MBEDTLS_TLS_EXT_USE_SRTP                    14

#define MBEDTLS_TLS_EXT_ALPN                        16

#define MBEDTLS_TLS_EXT_ENCRYPT_THEN_MAC            22 /* 0x16 */
#define MBEDTLS_TLS_EXT_EXTENDED_MASTER_SECRET  0x0017 /* 23 */

#define MBEDTLS_TLS_EXT_SESSION_TICKET              35

/* The value of the CID extension is still TBD as of
 * draft-ietf-tls-dtls-connection-id-05
 * (https://tools.ietf.org/html/draft-ietf-tls-dtls-connection-id-05) */
#define MBEDTLS_TLS_EXT_CID                        254 /* TBD */

#define MBEDTLS_TLS_EXT_ECJPAKE_KKPP               256 /* experimental */

#define MBEDTLS_TLS_EXT_RENEGOTIATION_INFO      0xFF01

/*
 * Size defines
 */
#if !defined(MBEDTLS_PSK_MAX_LEN)
#define MBEDTLS_PSK_MAX_LEN            32 /* 256 bits */
#endif

/* Dummy type used only for its size */
union mbedtls_ssl_premaster_secret
{
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_ENABLED)
    unsigned char _pms_rsa[48];                         /* RFC 5246 8.1.1 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED)
    unsigned char _pms_dhm[MBEDTLS_MPI_MAX_SIZE];      /* RFC 5246 8.1.2 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED)    || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED)  || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED)     || \
    defined(MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED)
    unsigned char _pms_ecdh[MBEDTLS_ECP_MAX_BYTES];    /* RFC 4492 5.10 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_PSK_ENABLED)
    unsigned char _pms_psk[4 + 2 * MBEDTLS_PSK_MAX_LEN];       /* RFC 4279 2 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED)
    unsigned char _pms_dhe_psk[4 + MBEDTLS_MPI_MAX_SIZE
                                 + MBEDTLS_PSK_MAX_LEN];       /* RFC 4279 3 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED)
    unsigned char _pms_rsa_psk[52 + MBEDTLS_PSK_MAX_LEN];      /* RFC 4279 4 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED)
    unsigned char _pms_ecdhe_psk[4 + MBEDTLS_ECP_MAX_BYTES
                                   + MBEDTLS_PSK_MAX_LEN];     /* RFC 5489 2 */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED)
    unsigned char _pms_ecjpake[32];     /* Thread spec: SHA-256 output */
#endif
};

#define MBEDTLS_PREMASTER_SIZE     sizeof( union mbedtls_ssl_premaster_secret )

/*
 * SSL state machine
 */
typedef enum
{
    MBEDTLS_SSL_HELLO_REQUEST,
    MBEDTLS_SSL_CLIENT_HELLO,
    MBEDTLS_SSL_SERVER_HELLO,
    MBEDTLS_SSL_SERVER_CERTIFICATE,
    MBEDTLS_SSL_SERVER_KEY_EXCHANGE,
    MBEDTLS_SSL_CERTIFICATE_REQUEST,
    MBEDTLS_SSL_SERVER_HELLO_DONE,
    MBEDTLS_SSL_CLIENT_CERTIFICATE,
    MBEDTLS_SSL_CLIENT_KEY_EXCHANGE,
    MBEDTLS_SSL_CERTIFICATE_VERIFY,
    MBEDTLS_SSL_CLIENT_CHANGE_CIPHER_SPEC,
    MBEDTLS_SSL_CLIENT_FINISHED,
    MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC,
    MBEDTLS_SSL_SERVER_FINISHED,
    MBEDTLS_SSL_FLUSH_BUFFERS,
    MBEDTLS_SSL_HANDSHAKE_WRAPUP,
    MBEDTLS_SSL_HANDSHAKE_OVER,
    MBEDTLS_SSL_SERVER_NEW_SESSION_TICKET,
    MBEDTLS_SSL_SERVER_HELLO_VERIFY_REQUEST_SENT,
}
mbedtls_ssl_states;

/*
 * The tls_prf function types.
 */
typedef enum
{
   MBEDTLS_SSL_TLS_PRF_NONE,
   MBEDTLS_SSL_TLS_PRF_SSL3,
   MBEDTLS_SSL_TLS_PRF_TLS1,
   MBEDTLS_SSL_TLS_PRF_SHA384,
   MBEDTLS_SSL_TLS_PRF_SHA256
}
mbedtls_tls_prf_types;

/**
 * \brief          Callback type: send data on the network.
 *
 * \note           That callback may be either blocking or non-blocking.
 *
 * \param ctx      Context for the send callback (typically a file descriptor)
 * \param buf      Buffer holding the data to send
 * \param len      Length of the data to send
 *
 * \return         The callback must return the number of bytes sent if any,
 *                 or a non-zero error code.
 *                 If performing non-blocking I/O, \c MBEDTLS_ERR_SSL_WANT_WRITE
 *                 must be returned when the operation would block.
 *
 * \note           The callback is allowed to send fewer bytes than requested.
 *                 It must always return the number of bytes actually sent.
 */
typedef int mbedtls_ssl_send_t( void *ctx,
                                const unsigned char *buf,
                                size_t len );

/**
 * \brief          Callback type: receive data from the network.
 *
 * \note           That callback may be either blocking or non-blocking.
 *
 * \param ctx      Context for the receive callback (typically a file
 *                 descriptor)
 * \param buf      Buffer to write the received data to
 * \param len      Length of the receive buffer
 *
 * \return         The callback must return the number of bytes received,
 *                 or a non-zero error code.
 *                 If performing non-blocking I/O, \c MBEDTLS_ERR_SSL_WANT_READ
 *                 must be returned when the operation would block.
 *
 * \note           The callback may receive fewer bytes than the length of the
 *                 buffer. It must always return the number of bytes actually
 *                 received and written to the buffer.
 */
typedef int mbedtls_ssl_recv_t( void *ctx,
                                unsigned char *buf,
                                size_t len );

/**
 * \brief          Callback type: receive data from the network, with timeout
 *
 * \note           That callback must block until data is received, or the
 *                 timeout delay expires, or the operation is interrupted by a
 *                 signal.
 *
 * \param ctx      Context for the receive callback (typically a file descriptor)
 * \param buf      Buffer to write the received data to
 * \param len      Length of the receive buffer
 * \param timeout  Maximum nomber of millisecondes to wait for data
 *                 0 means no timeout (potentially waiting forever)
 *
 * \return         The callback must return the number of bytes received,
 *                 or a non-zero error code:
 *                 \c MBEDTLS_ERR_SSL_TIMEOUT if the operation timed out,
 *                 \c MBEDTLS_ERR_SSL_WANT_READ if interrupted by a signal.
 *
 * \note           The callback may receive fewer bytes than the length of the
 *                 buffer. It must always return the number of bytes actually
 *                 received and written to the buffer.
 */
typedef int mbedtls_ssl_recv_timeout_t( void *ctx,
                                        unsigned char *buf,
                                        size_t len,
                                        uint32_t timeout );
/**
 * \brief          Callback type: set a pair of timers/delays to watch
 *
 * \param ctx      Context pointer
 * \param int_ms   Intermediate delay in milliseconds
 * \param fin_ms   Final delay in milliseconds
 *                 0 cancels the current timer.
 *
 * \note           This callback must at least store the necessary information
 *                 for the associated \c mbedtls_ssl_get_timer_t callback to
 *                 return correct information.
 *
 * \note           If using a event-driven style of programming, an event must
 *                 be generated when the final delay is passed. The event must
 *                 cause a call to \c mbedtls_ssl_handshake() with the proper
 *                 SSL context to be scheduled. Care must be taken to ensure
 *                 that at most one such call happens at a time.
 *
 * \note           Only one timer at a time must be running. Calling this
 *                 function while a timer is running must cancel it. Cancelled
 *                 timers must not generate any event.
 */
typedef void mbedtls_ssl_set_timer_t( void * ctx,
                                      uint32_t int_ms,
                                      uint32_t fin_ms );

/**
 * \brief          Callback type: get status of timers/delays
 *
 * \param ctx      Context pointer
 *
 * \return         This callback must return:
 *                 -1 if cancelled (fin_ms == 0),
 *                  0 if none of the delays have passed,
 *                  1 if only the intermediate delay has passed,
 *                  2 if the final delay has passed.
 */
typedef int mbedtls_ssl_get_timer_t( void * ctx );

/* Defined below */
typedef struct mbedtls_ssl_session mbedtls_ssl_session;
typedef struct mbedtls_ssl_context mbedtls_ssl_context;
typedef struct mbedtls_ssl_config  mbedtls_ssl_config;

/* Defined in ssl_internal.h */
typedef struct mbedtls_ssl_transform mbedtls_ssl_transform;
typedef struct mbedtls_ssl_handshake_params mbedtls_ssl_handshake_params;
typedef struct mbedtls_ssl_sig_hash_set_t mbedtls_ssl_sig_hash_set_t;
typedef struct mbedtls_ssl_key_cert mbedtls_ssl_key_cert;
typedef struct mbedtls_ssl_flight_item mbedtls_ssl_flight_item;

/**
 * \brief           Callback type: start external signature operation.
 *
 *                  This callback is called during an SSL handshake to start
 *                  a signature decryption operation using an
 *                  external processor. The parameter \p cert contains
 *                  the public key; it is up to the callback function to
 *                  determine how to access the associated private key.
 *
 *                  This function typically sends or enqueues a request, and
 *                  does not wait for the operation to complete. This allows
 *                  the handshake step to be non-blocking.
 *
 *                  The parameters \p ssl and \p cert are guaranteed to remain
 *                  valid throughout the handshake. On the other hand, this
 *                  function must save the contents of \p hash if the value
 *                  is needed for later processing, because the \p hash buffer
 *                  is no longer valid after this function returns.
 *
 *                  This function may call mbedtls_ssl_set_async_operation_data()
 *                  to store an operation context for later retrieval
 *                  by the resume or cancel callback.
 *
 * \note            For RSA signatures, this function must produce output
 *                  that is consistent with PKCS#1 v1.5 in the same way as
 *                  mbedtls_rsa_pkcs1_sign(). Before the private key operation,
 *                  apply the padding steps described in RFC 8017, section 9.2
 *                  "EMSA-PKCS1-v1_5" as follows.
 *                  - If \p md_alg is #MBEDTLS_MD_NONE, apply the PKCS#1 v1.5
 *                    encoding, treating \p hash as the DigestInfo to be
 *                    padded. In other words, apply EMSA-PKCS1-v1_5 starting
 *                    from step 3, with `T = hash` and `tLen = hash_len`.
 *                  - If `md_alg != MBEDTLS_MD_NONE`, apply the PKCS#1 v1.5
 *                    encoding, treating \p hash as the hash to be encoded and
 *                    padded. In other words, apply EMSA-PKCS1-v1_5 starting
 *                    from step 2, with `digestAlgorithm` obtained by calling
 *                    mbedtls_oid_get_oid_by_md() on \p md_alg.
 *
 * \note            For ECDSA signatures, the output format is the DER encoding
 *                  `Ecdsa-Sig-Value` defined in
 *                  [RFC 4492 section 5.4](https://tools.ietf.org/html/rfc4492#section-5.4).
 *
 * \param ssl             The SSL connection instance. It should not be
 *                        modified other than via
 *                        mbedtls_ssl_set_async_operation_data().
 * \param cert            Certificate containing the public key.
 *                        In simple cases, this is one of the pointers passed to
 *                        mbedtls_ssl_conf_own_cert() when configuring the SSL
 *                        connection. However, if other callbacks are used, this
 *                        property may not hold. For example, if an SNI callback
 *                        is registered with mbedtls_ssl_conf_sni(), then
 *                        this callback determines what certificate is used.
 * \param md_alg          Hash algorithm.
 * \param hash            Buffer containing the hash. This buffer is
 *                        no longer valid when the function returns.
 * \param hash_len        Size of the \c hash buffer in bytes.
 *
 * \return          0 if the operation was started successfully and the SSL
 *                  stack should call the resume callback immediately.
 * \return          #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS if the operation
 *                  was started successfully and the SSL stack should return
 *                  immediately without calling the resume callback yet.
 * \return          #MBEDTLS_ERR_SSL_HW_ACCEL_FALLTHROUGH if the external
 *                  processor does not support this key. The SSL stack will
 *                  use the private key object instead.
 * \return          Any other error indicates a fatal failure and is
 *                  propagated up the call chain. The callback should
 *                  use \c MBEDTLS_ERR_PK_xxx error codes, and <b>must not</b>
 *                  use \c MBEDTLS_ERR_SSL_xxx error codes except as
 *                  directed in the documentation of this callback.
 */
typedef int mbedtls_ssl_async_sign_t( mbedtls_ssl_context *ssl,
                                      mbedtls_x509_crt *cert,
                                      mbedtls_md_type_t md_alg,
                                      const unsigned char *hash,
                                      size_t hash_len );

/**
 * \brief           Callback type: start external decryption operation.
 *
 *                  This callback is called during an SSL handshake to start
 *                  an RSA decryption operation using an
 *                  external processor. The parameter \p cert contains
 *                  the public key; it is up to the callback function to
 *                  determine how to access the associated private key.
 *
 *                  This function typically sends or enqueues a request, and
 *                  does not wait for the operation to complete. This allows
 *                  the handshake step to be non-blocking.
 *
 *                  The parameters \p ssl and \p cert are guaranteed to remain
 *                  valid throughout the handshake. On the other hand, this
 *                  function must save the contents of \p input if the value
 *                  is needed for later processing, because the \p input buffer
 *                  is no longer valid after this function returns.
 *
 *                  This function may call mbedtls_ssl_set_async_operation_data()
 *                  to store an operation context for later retrieval
 *                  by the resume or cancel callback.
 *
 * \warning         RSA decryption as used in TLS is subject to a potential
 *                  timing side channel attack first discovered by Bleichenbacher
 *                  in 1998. This attack can be remotely exploitable
 *                  in practice. To avoid this attack, you must ensure that
 *                  if the callback performs an RSA decryption, the time it
 *                  takes to execute and return the result does not depend
 *                  on whether the RSA decryption succeeded or reported
 *                  invalid padding.
 *
 * \param ssl             The SSL connection instance. It should not be
 *                        modified other than via
 *                        mbedtls_ssl_set_async_operation_data().
 * \param cert            Certificate containing the public key.
 *                        In simple cases, this is one of the pointers passed to
 *                        mbedtls_ssl_conf_own_cert() when configuring the SSL
 *                        connection. However, if other callbacks are used, this
 *                        property may not hold. For example, if an SNI callback
 *                        is registered with mbedtls_ssl_conf_sni(), then
 *                        this callback determines what certificate is used.
 * \param input           Buffer containing the input ciphertext. This buffer
 *                        is no longer valid when the function returns.
 * \param input_len       Size of the \p input buffer in bytes.
 *
 * \return          0 if the operation was started successfully and the SSL
 *                  stack should call the resume callback immediately.
 * \return          #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS if the operation
 *                  was started successfully and the SSL stack should return
 *                  immediately without calling the resume callback yet.
 * \return          #MBEDTLS_ERR_SSL_HW_ACCEL_FALLTHROUGH if the external
 *                  processor does not support this key. The SSL stack will
 *                  use the private key object instead.
 * \return          Any other error indicates a fatal failure and is
 *                  propagated up the call chain. The callback should
 *                  use \c MBEDTLS_ERR_PK_xxx error codes, and <b>must not</b>
 *                  use \c MBEDTLS_ERR_SSL_xxx error codes except as
 *                  directed in the documentation of this callback.
 */
typedef int mbedtls_ssl_async_decrypt_t( mbedtls_ssl_context *ssl,
                                         mbedtls_x509_crt *cert,
                                         const unsigned char *input,
                                         size_t input_len );

/**
 * \brief           Callback type: resume external operation.
 *
 *                  This callback is called during an SSL handshake to resume
 *                  an external operation started by the
 *                  ::mbedtls_ssl_async_sign_t or
 *                  ::mbedtls_ssl_async_decrypt_t callback.
 *
 *                  This function typically checks the status of a pending
 *                  request or causes the request queue to make progress, and
 *                  does not wait for the operation to complete. This allows
 *                  the handshake step to be non-blocking.
 *
 *                  This function may call mbedtls_ssl_get_async_operation_data()
 *                  to retrieve an operation context set by the start callback.
 *                  It may call mbedtls_ssl_set_async_operation_data() to modify
 *                  this context.
 *
 *                  Note that when this function returns a status other than
 *                  #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS, it must free any
 *                  resources associated with the operation.
 *
 * \param ssl             The SSL connection instance. It should not be
 *                        modified other than via
 *                        mbedtls_ssl_set_async_operation_data().
 * \param output          Buffer containing the output (signature or decrypted
 *                        data) on success.
 * \param output_len      On success, number of bytes written to \p output.
 * \param output_size     Size of the \p output buffer in bytes.
 *
 * \return          0 if output of the operation is available in the
 *                  \p output buffer.
 * \return          #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS if the operation
 *                  is still in progress. Subsequent requests for progress
 *                  on the SSL connection will call the resume callback
 *                  again.
 * \return          Any other error means that the operation is aborted.
 *                  The SSL handshake is aborted. The callback should
 *                  use \c MBEDTLS_ERR_PK_xxx error codes, and <b>must not</b>
 *                  use \c MBEDTLS_ERR_SSL_xxx error codes except as
 *                  directed in the documentation of this callback.
 */
typedef int mbedtls_ssl_async_resume_t( mbedtls_ssl_context *ssl,
                                        unsigned char *output,
                                        size_t *output_len,
                                        size_t output_size );

/**
 * \brief           Callback type: cancel external operation.
 *
 *                  This callback is called if an SSL connection is closed
 *                  while an asynchronous operation is in progress. Note that
 *                  this callback is not called if the
 *                  ::mbedtls_ssl_async_resume_t callback has run and has
 *                  returned a value other than
 *                  #MBEDTLS_ERR_SSL_ASYNC_IN_PROGRESS, since in that case
 *                  the asynchronous operation has already completed.
 *
 *                  This function may call mbedtls_ssl_get_async_operation_data()
 *                  to retrieve an operation context set by the start callback.
 *
 * \param ssl             The SSL connection instance. It should not be
 *                        modified.
 */
typedef void mbedtls_ssl_async_cancel_t( mbedtls_ssl_context *ssl );

#define MBEDTLS_TLS_SRTP_MAX_MKI_LENGTH             255
#define MBEDTLS_TLS_SRTP_MAX_PROFILE_LIST_LENGTH    4
/*
 * For code readability use a typedef for DTLS-SRTP profiles
 *
 * Use_srtp extension protection profiles values as defined in
 * http://www.iana.org/assignments/srtp-protection/srtp-protection.xhtml
 *
 * Reminder: if this list is expanded mbedtls_ssl_check_srtp_profile_value
 * must be updated too.
 */
#define MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_80     ( (uint16_t) 0x0001)
#define MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_32     ( (uint16_t) 0x0002)
#define MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_80          ( (uint16_t) 0x0005)
#define MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_32          ( (uint16_t) 0x0006)
/* This one is not iana defined, but for code readability. */
#define MBEDTLS_TLS_SRTP_UNSET                      ( (uint16_t) 0x0000)

typedef uint16_t mbedtls_ssl_srtp_profile;

typedef struct mbedtls_dtls_srtp_info_t
{
    /*! The SRTP profile that was negotiated. */
    mbedtls_ssl_srtp_profile chosen_dtls_srtp_profile;
    /*! The length of mki_value. */
    uint16_t mki_len;
    /*! The mki_value used, with max size of 256 bytes. */
    unsigned char mki_value[MBEDTLS_TLS_SRTP_MAX_MKI_LENGTH];
}
mbedtls_dtls_srtp_info;

/*
 * This structure is used for storing current session data.
 *
 * Note: when changing this definition, we need to check and update:
 *  - in tests/suites/test_suite_ssl.function:
 *      ssl_populate_session() and ssl_serialize_session_save_load()
 *  - in library/ssl_tls.c:
 *      mbedtls_ssl_session_init() and mbedtls_ssl_session_free()
 *      mbedtls_ssl_session_save() and ssl_session_load()
 *      ssl_session_copy()
 */
struct mbedtls_ssl_session
{
#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t start;       /*!< starting time      */
#endif
    int ciphersuite;            /*!< chosen ciphersuite */
    int compression;            /*!< chosen compression */
    size_t id_len;              /*!< session id length  */
    unsigned char id[32];       /*!< session identifier */
    unsigned char master[48];   /*!< the master secret  */
#if defined(MBEDTLS_X509_CRT_PARSE_C)
#if defined(MBEDTLS_SSL_KEEP_PEER_CERTIFICATE)
    mbedtls_x509_crt *peer_cert;       /*!< peer X.509 cert chain */
#else /* MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
    /*! The digest of the peer's end-CRT. This must be kept to detect CRT
     *  changes during renegotiation, mitigating the triple handshake attack. */
    unsigned char *peer_cert_digest;
    size_t peer_cert_digest_len;
    mbedtls_md_type_t peer_cert_digest_type;
#endif /* !MBEDTLS_SSL_KEEP_PEER_CERTIFICATE */
#endif /* MBEDTLS_X509_CRT_PARSE_C */
    uint32_t verify_result;          /*!<  verification result     */
#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_CLI_C)
    unsigned char *ticket;      /*!< RFC 5077 session ticket */
    size_t ticket_len;          /*!< session ticket length   */
    uint32_t ticket_lifetime;   /*!< ticket lifetime hint    */
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_CLI_C */
#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    unsigned char mfl_code;     /*!< MaxFragmentLength negotiated by peer */
#endif /* MBEDTLS_SSL_MAX_FRAGMENT_LENGTH */
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
    int encrypt_then_mac;       /*!< flag for EtM activation                */
#endif
};

/**
 * SSL/TLS configuration to be shared between mbedtls_ssl_context structures.
 */
struct mbedtls_ssl_config
{
    /* Group items by size (largest first) to minimize padding overhead */
    /*
     * Pointers
     */
    const uint16_t *ciphersuite_list[4]; /*!< allowed ciphersuites per version   */
    /** Callback to SSL handshake step                                      */
    int (*f_step)( mbedtls_ssl_context * );
    /** Callback for printing debug output                                  */
    void (*f_dbg)(void *, int, const char *, int, const char *);
    void *p_dbg;                    /*!< context for the debug function     */
    /** Callback for getting (pseudo-)random numbers                        */
    int  (*f_rng)(void *, unsigned char *, size_t);
    void *p_rng;                    /*!< context for the RNG function       */
    /** Callback to retrieve a session from the cache                       */
    int (*f_get_cache)(void *, mbedtls_ssl_session *);
    /** Callback to store a session into the cache                          */
    int (*f_set_cache)(void *, const mbedtls_ssl_session *);
    void *p_cache;                  /*!< context for cache callbacks        */
#if defined(MBEDTLS_SSL_SERVER_NAME_INDICATION)
    /** Callback for setting cert according to SNI extension                */
    int (*f_sni)(void *, mbedtls_ssl_context *, const unsigned char *, size_t);
    void *p_sni;                    /*!< context for SNI callback           */
#endif
#if defined(MBEDTLS_X509_CRT_PARSE_C)
    /** Callback to customize X.509 certificate chain verification          */
    int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *);
    void *p_vrfy;                   /*!< context for X.509 verify calllback */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
    /** Callback to retrieve PSK key from identity                          */
    int (*f_psk)(void *, mbedtls_ssl_context *, const unsigned char *, size_t);
    void *p_psk;                    /*!< context for PSK callback           */
#endif
#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && defined(MBEDTLS_SSL_SRV_C)
    /** Callback to create & write a cookie for ClientHello veirifcation    */
    int (*f_cookie_write)( void *, unsigned char **, unsigned char *,
                           const unsigned char *, size_t );
    /** Callback to verify validity of a ClientHello cookie                 */
    int (*f_cookie_check)( void *, const unsigned char *, size_t,
                           const unsigned char *, size_t );
    void *p_cookie;                 /*!< context for the cookie callbacks   */
#endif
#if defined(MBEDTLS_SSL_SESSION_TICKETS) && defined(MBEDTLS_SSL_SRV_C)
    /** Callback to create & write a session ticket                         */
    int (*f_ticket_write)( void *, const mbedtls_ssl_session *,
            unsigned char *, const unsigned char *, size_t *, uint32_t * );
    /** Callback to parse a session ticket into a session structure         */
    int (*f_ticket_parse)( void *, mbedtls_ssl_session *, unsigned char *, size_t);
    void *p_ticket;                 /*!< context for the ticket callbacks   */
#endif /* MBEDTLS_SSL_SESSION_TICKETS && MBEDTLS_SSL_SRV_C */
#if defined(MBEDTLS_SSL_EXPORT_KEYS)
    /** Callback to export key block and master secret                      */
    int (*f_export_keys)( void *, const unsigned char *,
            const unsigned char *, size_t, size_t, size_t );
    /** Callback to export key block, master secret,
     *  tls_prf and random bytes. Should replace f_export_keys    */
    int (*f_export_keys_ext)( void *, const unsigned char *,
                const unsigned char *, size_t, size_t, size_t,
                const unsigned char[32], const unsigned char[32],
                mbedtls_tls_prf_types );
    void *p_export_keys;            /*!< context for key export callback    */
#endif
#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    size_t cid_len; /*!< The length of CIDs for incoming DTLS records.      */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
#if defined(MBEDTLS_X509_CRT_PARSE_C)
    const mbedtls_x509_crt_profile *cert_profile; /*!< verification profile */
    mbedtls_ssl_key_cert *key_cert; /*!< own certificate/key pair(s)        */
    mbedtls_x509_crt *ca_chain;     /*!< trusted CAs                        */
    mbedtls_x509_crl *ca_crl;       /*!< trusted CAs CRLs                   */
#if defined(MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK)
    mbedtls_x509_crt_ca_cb_t f_ca_cb;
    void *p_ca_cb;
#endif /* MBEDTLS_X509_TRUSTED_CERTIFICATE_CALLBACK */
#endif /* MBEDTLS_X509_CRT_PARSE_C */
#if defined(MBEDTLS_SSL_ASYNC_PRIVATE)
#if defined(MBEDTLS_X509_CRT_PARSE_C)
    mbedtls_ssl_async_sign_t *f_async_sign_start; /*!< start asynchronous signature operation */
    mbedtls_ssl_async_decrypt_t *f_async_decrypt_start; /*!< start asynchronous decryption operation */
#endif /* MBEDTLS_X509_CRT_PARSE_C */
    mbedtls_ssl_async_resume_t *f_async_resume; /*!< resume asynchronous operation */
    mbedtls_ssl_async_cancel_t *f_async_cancel; /*!< cancel asynchronous operation */
    void *p_async_config_data; /*!< Configuration data set by mbedtls_ssl_conf_async_private_cb(). */
#endif /* MBEDTLS_SSL_ASYNC_PRIVATE */
#if defined(MBEDTLS_KEY_EXCHANGE_WITH_CERT_ENABLED)
    const uint8_t *sig_hashes;          /*!< allowed signature hashes           */
#endif
#if defined(MBEDTLS_ECP_C)
    const mbedtls_ecp_group_id *curve_list; /*!< allowed curves             */
#endif
#if defined(MBEDTLS_DHM_C)
    mbedtls_mpi dhm_P;              /*!< prime modulus for DHM              */
    mbedtls_mpi dhm_G;              /*!< generator for DHM                  */
#endif
#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED)
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_key_id_t psk_opaque; /*!< PSA key slot holding opaque PSK. This field
                              *   should only be set via
                              *   mbedtls_ssl_conf_psk_opaque().
                              *   If either no PSK or a raw PSK have been
                              *   configured, this has value \c 0.
                              */
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    unsigned char *psk;      /*!< The raw pre-shared key. This field should
                              *   only be set via mbedtls_ssl_conf_psk().
                              *   If either no PSK or an opaque PSK
                              *   have been configured, this has value NULL. */
    size_t         psk_len;  /*!< The length of the raw pre-shared key.
                              *   This field should only be set via
                              *   mbedtls_ssl_conf_psk().
                              *   Its value is non-zero if and only if
                              *   \c psk is not \c NULL. */
    unsigned char *psk_identity;    /*!< The PSK identity for PSK negotiation.
                                     *   This field should only be set via
                                     *   mbedtls_ssl_conf_psk().
                                     *   This is set if and only if either
                                     *   \c psk or \c psk_opaque are set. */
    size_t         psk_identity_len;/*!< The length of PSK identity.
                                     *   This field should only be set via
                                     *   mbedtls_ssl_conf_psk().
                                     *   Its value is non-zero if and only if
                                     *   \c psk is not \c NULL or \c psk_opaque
                                     *   is not \c 0. */
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_PSK_ENABLED */
#if defined(MBEDTLS_SSL_ALPN)
    const char **alpn_list;         /*!< ordered list of protocols          */
#endif
#if defined(MBEDTLS_SSL_DTLS_SRTP)
    /*! ordered list of supported srtp profile */
    const mbedtls_ssl_srtp_profile *dtls_srtp_profile_list;
    /*! number of supported profiles */
    size_t dtls_srtp_profile_list_len;
#endif /* MBEDTLS_SSL_DTLS_SRTP */
    /*
     * Numerical settings (int then char)
     */
    uint32_t read_timeout;          /*!< timeout for mbedtls_ssl_read (ms)  */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    uint32_t hs_timeout_min;        /*!< initial value of the handshake
                                         retransmission timeout (ms)        */
    uint32_t hs_timeout_max;        /*!< maximum value of the handshake
                                         retransmission timeout (ms)        */
#endif
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    int renego_max_records;         /*!< grace period for renegotiation     */
    unsigned char renego_period[8]; /*!< value of the record counters
                                         that triggers renegotiation        */
#endif
#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT)
    unsigned int badmac_limit;      /*!< limit of records with a bad MAC    */
#endif
#if defined(MBEDTLS_DHM_C) && defined(MBEDTLS_SSL_CLI_C)
    unsigned int dhm_min_bitlen;    /*!< min. bit length of the DHM prime   */
#endif
    unsigned char max_major_ver;    /*!< max. major version used            */
    unsigned char max_minor_ver;    /*!< max. minor version used            */
    unsigned char min_major_ver;    /*!< min. major version used            */
    unsigned char min_minor_ver;    /*!< min. minor version used            */
    /*
     * Flags (bitfields)
     */
    unsigned int endpoint : 1;      /*!< 0: client, 1: server               */
    unsigned int transport : 1;     /*!< stream (TLS) or datagram (DTLS)    */
    unsigned int authmode : 2;      /*!< MBEDTLS_SSL_VERIFY_XXX             */
    /* needed even with renego disabled for LEGACY_BREAK_HANDSHAKE          */
    unsigned int allow_legacy_renegotiation : 2 ; /*!< MBEDTLS_LEGACY_XXX   */
#if defined(MBEDTLS_ARC4_C)
    unsigned int arc4_disabled : 1; /*!< blacklist RC4 ciphersuites?        */
#endif
#if defined(MBEDTLS_SSL_MAX_FRAGMENT_LENGTH)
    unsigned int mfl_code : 3;      /*!< desired fragment length            */
#endif
#if defined(MBEDTLS_SSL_ENCRYPT_THEN_MAC)
    unsigned int encrypt_then_mac : 1 ; /*!< negotiate encrypt-then-mac?    */
#endif
#if defined(MBEDTLS_SSL_EXTENDED_MASTER_SECRET)
    unsigned int extended_ms : 1;   /*!< negotiate extended master secret?  */
#endif
#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
    unsigned int anti_replay : 1;   /*!< detect and prevent replay?         */
#endif
#if defined(MBEDTLS_SSL_CBC_RECORD_SPLITTING)
    unsigned int cbc_record_splitting : 1;  /*!< do cbc record splitting    */
#endif
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    unsigned int disable_renegotiation : 1; /*!< disable renegotiation?     */
#endif
#if defined(MBEDTLS_SSL_SESSION_TICKETS)
    unsigned int session_tickets : 1;   /*!< use session tickets?           */
#endif
#if defined(MBEDTLS_SSL_FALLBACK_SCSV) && defined(MBEDTLS_SSL_CLI_C)
    unsigned int fallback : 1;      /*!< is this a fallback?                */
#endif
#if defined(MBEDTLS_SSL_SRV_C)
    unsigned int cert_req_ca_list : 1;  /*!< enable sending CA list in
                                          Certificate Request messages?     */
#endif
#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    unsigned int ignore_unexpected_cid : 1; /*!< Determines whether DTLS
                                             *   record with unexpected CID
                                             *   should lead to failure.    */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
#if defined(MBEDTLS_SSL_DTLS_SRTP)
    unsigned int dtls_srtp_mki_support : 1; /* support having mki_value
                                               in the use_srtp extension     */
#endif
    bool disable_compression;
};

struct mbedtls_ssl_context
{
    const mbedtls_ssl_config *conf; /*!< configuration information          */
    /*
     * Miscellaneous
     */
    int state;                  /*!< SSL handshake: current state     */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    int renego_status;          /*!< Initial, in progress, pending?   */
    int renego_records_seen;    /*!< Records since renego request, or with DTLS,
                                  number of retransmissions of request if
                                  renego_max_records is < 0           */
#endif /* MBEDTLS_SSL_RENEGOTIATION */
    int major_ver;              /*!< equal to  MBEDTLS_SSL_MAJOR_VERSION_3    */
    int minor_ver;              /*!< either 0 (SSL3) or 1 (TLS1.0)    */
#if defined(MBEDTLS_SSL_DTLS_BADMAC_LIMIT)
    unsigned badmac_seen;       /*!< records with a bad MAC received    */
#endif /* MBEDTLS_SSL_DTLS_BADMAC_LIMIT */
#if defined(MBEDTLS_X509_CRT_PARSE_C)
    /** Callback to customize X.509 certificate chain verification          */
    int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *);
    void *p_vrfy;                   /*!< context for X.509 verify callback */
#endif
    mbedtls_ssl_send_t *f_send; /*!< Callback for network send */
    mbedtls_ssl_recv_t *f_recv; /*!< Callback for network receive */
    mbedtls_ssl_recv_timeout_t *f_recv_timeout;
                                /*!< Callback for network receive with timeout */
    void *p_bio;                /*!< context for I/O operations   */
    /*
     * Session layer
     */
    mbedtls_ssl_session *session_in;            /*!<  current session data (in)   */
    mbedtls_ssl_session *session_out;           /*!<  current session data (out)  */
    mbedtls_ssl_session *session;               /*!<  negotiated session data     */
    mbedtls_ssl_session *session_negotiate;     /*!<  session data in negotiation */
    mbedtls_ssl_handshake_params *handshake;    /*!<  params required only during
                                                      the handshake process        */
    const mbedtls_ecp_curve_info *curve;
    /*
     * Record layer transformations
     */
    mbedtls_ssl_transform *transform_in;        /*!<  current transform params (in)   */
    mbedtls_ssl_transform *transform_out;       /*!<  current transform params (in)   */
    mbedtls_ssl_transform *transform;           /*!<  negotiated transform params     */
    mbedtls_ssl_transform *transform_negotiate; /*!<  transform params in negotiation */
    /*
     * Timers
     */
    void *p_timer;              /*!< context for the timer callbacks */
    mbedtls_ssl_set_timer_t *f_set_timer;       /*!< set timer callback */
    mbedtls_ssl_get_timer_t *f_get_timer;       /*!< get timer callback */
    /*
     * Record layer (incoming data)
     */
    unsigned char *in_buf;      /*!< input buffer                     */
    unsigned char *in_ctr;      /*!< 64-bit incoming message counter
                                     TLS: maintained by us
                                     DTLS: read from peer             */
    unsigned char *in_hdr;      /*!< start of record header           */
#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    unsigned char *in_cid;      /*!< The start of the CID;
                                 *   (the end is marked by in_len).   */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
    unsigned char *in_len;      /*!< two-bytes message length field   */
    unsigned char *in_iv;       /*!< ivlen-byte IV                    */
    unsigned char *in_msg;      /*!< message contents (in_iv+ivlen)   */
    unsigned char *in_offt;     /*!< read offset in application data  */
    int in_msgtype;             /*!< record header: message type      */
    size_t in_msglen;           /*!< record header: message length    */
    size_t in_left;             /*!< amount of data read so far       */
#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    size_t in_buf_len;          /*!< length of input buffer           */
#endif
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    uint16_t in_epoch;          /*!< DTLS epoch for incoming records  */
    size_t next_record_offset;  /*!< offset of the next record in datagram
                                     (equal to in_left if none)       */
#endif /* MBEDTLS_SSL_PROTO_DTLS */
#if defined(MBEDTLS_SSL_DTLS_ANTI_REPLAY)
    uint64_t in_window_top;     /*!< last validated record seq_num    */
    uint64_t in_window;         /*!< bitmask for replay detection     */
#endif /* MBEDTLS_SSL_DTLS_ANTI_REPLAY */
    size_t in_hslen;            /*!< current handshake message length,
                                     including the handshake header   */
    int nb_zero;                /*!< # of 0-length encrypted messages */
    int keep_current_message;   /*!< drop or reuse current message
                                     on next call to record layer? */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    uint8_t disable_datagram_packing;  /*!< Disable packing multiple records
                                        *   within a single datagram.  */
#endif /* MBEDTLS_SSL_PROTO_DTLS */
    /*
     * Record layer (outgoing data)
     */
    unsigned char *out_buf;     /*!< output buffer                    */
    unsigned char *out_ctr;     /*!< 64-bit outgoing message counter  */
    unsigned char *out_hdr;     /*!< start of record header           */
#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    unsigned char *out_cid;     /*!< The start of the CID;
                                 *   (the end is marked by in_len).   */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
    unsigned char *out_len;     /*!< two-bytes message length field   */
    unsigned char *out_iv;      /*!< ivlen-byte IV                    */
    unsigned char *out_msg;     /*!< message contents (out_iv+ivlen)  */
    int out_msgtype;            /*!< record header: message type      */
    size_t out_msglen;          /*!< record header: message length    */
    size_t out_left;            /*!< amount of data not yet written   */
#if defined(MBEDTLS_SSL_VARIABLE_BUFFER_LENGTH)
    size_t out_buf_len;         /*!< length of output buffer          */
#endif
    uint8_t fatal_alert;
    unsigned char cur_out_ctr[8]; /*!<  Outgoing record sequence  number. */
#if defined(MBEDTLS_SSL_PROTO_DTLS)
    uint16_t mtu;               /*!< path mtu, used to fragment outgoing messages */
#endif /* MBEDTLS_SSL_PROTO_DTLS */
#if defined(MBEDTLS_ZLIB_SUPPORT)
    unsigned char *compress_buf;        /*!<  zlib data buffer        */
#endif /* MBEDTLS_ZLIB_SUPPORT */
#if defined(MBEDTLS_SSL_CBC_RECORD_SPLITTING)
    signed char split_done;     /*!< current record already splitted? */
#endif /* MBEDTLS_SSL_CBC_RECORD_SPLITTING */
    /*
     * PKI layer
     */
    int client_auth;                    /*!<  flag for client auth.   */
    /*
     * User settings
     */
#if defined(MBEDTLS_X509_CRT_PARSE_C)
    char *hostname;             /*!< expected peer CN for verification
                                     (and SNI if available)                 */
#endif /* MBEDTLS_X509_CRT_PARSE_C */
#if defined(MBEDTLS_SSL_ALPN)
    const char *alpn_chosen;    /*!<  negotiated protocol                   */
#endif /* MBEDTLS_SSL_ALPN */
#if defined(MBEDTLS_SSL_DTLS_SRTP)
    /*
     * use_srtp extension
     */
    mbedtls_dtls_srtp_info dtls_srtp_info;
#endif /* MBEDTLS_SSL_DTLS_SRTP */
    /*
     * Information for DTLS hello verify
     */
#if defined(MBEDTLS_SSL_DTLS_HELLO_VERIFY) && defined(MBEDTLS_SSL_SRV_C)
    unsigned char  *cli_id;         /*!<  transport-level ID of the client  */
    size_t          cli_id_len;     /*!<  length of cli_id                  */
#endif /* MBEDTLS_SSL_DTLS_HELLO_VERIFY && MBEDTLS_SSL_SRV_C */
    /*
     * Secure renegotiation
     */
    /* needed to know when to send extension on server */
    int secure_renegotiation;           /*!<  does peer support legacy or
                                              secure renegotiation           */
#if defined(MBEDTLS_SSL_RENEGOTIATION)
    size_t verify_data_len;             /*!<  length of verify data stored   */
    char own_verify_data[MBEDTLS_SSL_VERIFY_DATA_MAX_LEN]; /*!<  previous handshake verify data */
    char peer_verify_data[MBEDTLS_SSL_VERIFY_DATA_MAX_LEN]; /*!<  previous handshake verify data */
#endif /* MBEDTLS_SSL_RENEGOTIATION */
#if defined(MBEDTLS_SSL_DTLS_CONNECTION_ID)
    /* CID configuration to use in subsequent handshakes. */
    /*! The next incoming CID, chosen by the user and applying to
     *  all subsequent handshakes. This may be different from the
     *  CID currently used in case the user has re-configured the CID
     *  after an initial handshake. */
    unsigned char own_cid[ MBEDTLS_SSL_CID_IN_LEN_MAX ];
    uint8_t own_cid_len;   /*!< The length of \c own_cid. */
    uint8_t negotiate_cid; /*!< This indicates whether the CID extension should
                            *   be negotiated in the next handshake or not.
                            *   Possible values are #MBEDTLS_SSL_CID_ENABLED
                            *   and #MBEDTLS_SSL_CID_DISABLED. */
#endif /* MBEDTLS_SSL_DTLS_CONNECTION_ID */
  uint16_t client_ciphers[16]; /* [jart] clarifies MBEDTLS_ERR_SSL_NO_USABLE_CIPHERSUITE */
};

/**
 * \brief           Callback type: generate and write session ticket
 *
 * \note            This describes what a callback implementation should do.
 *                  This callback should generate an encrypted and
 *                  authenticated ticket for the session and write it to the
 *                  output buffer. Here, ticket means the opaque ticket part
 *                  of the NewSessionTicket structure of RFC 5077.
 *
 * \param p_ticket  Context for the callback
 * \param session   SSL session to be written in the ticket
 * \param start     Start of the output buffer
 * \param end       End of the output buffer
 * \param tlen      On exit, holds the length written
 * \param lifetime  On exit, holds the lifetime of the ticket in seconds
 *
 * \return          0 if successful, or
 *                  a specific MBEDTLS_ERR_XXX code.
 */
typedef int mbedtls_ssl_ticket_write_t( void *p_ticket,
                                        const mbedtls_ssl_session *session,
                                        unsigned char *start,
                                        const unsigned char *end,
                                        size_t *tlen,
                                        uint32_t *lifetime );

/**
 * \brief           Callback type: Export key block and master secret
 *
 * \note            This is required for certain uses of TLS, e.g. EAP-TLS
 *                  (RFC 5216) and Thread. The key pointers are ephemeral and
 *                  therefore must not be stored. The master secret and keys
 *                  should not be used directly except as an input to a key
 *                  derivation function.
 *
 * \param p_expkey  Context for the callback
 * \param ms        Pointer to master secret (fixed length: 48 bytes)
 * \param kb        Pointer to key block, see RFC 5246 section 6.3
 *                  (variable length: 2 * maclen + 2 * keylen + 2 * ivlen).
 * \param maclen    MAC length
 * \param keylen    Key length
 * \param ivlen     IV length
 *
 * \return          0 if successful, or
 *                  a specific MBEDTLS_ERR_XXX code.
 */
typedef int mbedtls_ssl_export_keys_t( void *p_expkey,
                                       const unsigned char *ms,
                                       const unsigned char *kb,
                                       size_t maclen,
                                       size_t keylen,
                                       size_t ivlen );

/**
 * \brief           Callback type: Export key block, master secret,
 *                                 handshake randbytes and the tls_prf function
 *                                 used to derive keys.
 *
 * \note            This is required for certain uses of TLS, e.g. EAP-TLS
 *                  (RFC 5216) and Thread. The key pointers are ephemeral and
 *                  therefore must not be stored. The master secret and keys
 *                  should not be used directly except as an input to a key
 *                  derivation function.
 *
 * \param p_expkey  Context for the callback.
 * \param ms        Pointer to master secret (fixed length: 48 bytes).
 * \param kb            Pointer to key block, see RFC 5246 section 6.3.
 *                      (variable length: 2 * maclen + 2 * keylen + 2 * ivlen).
 * \param maclen        MAC length.
 * \param keylen        Key length.
 * \param ivlen         IV length.
 * \param client_random The client random bytes.
 * \param server_random The server random bytes.
 * \param tls_prf_type The tls_prf enum type.
 *
 * \return          0 if successful, or
 *                  a specific MBEDTLS_ERR_XXX code.
 */
typedef int mbedtls_ssl_export_keys_ext_t( void *p_expkey,
                                           const unsigned char *ms,
                                           const unsigned char *kb,
                                           size_t maclen,
                                           size_t keylen,
                                           size_t ivlen,
                                           const unsigned char client_random[32],
                                           const unsigned char server_random[32],
                                           mbedtls_tls_prf_types tls_prf_type );

/**
 * \brief           Callback type: parse and load session ticket
 *
 * \note            This describes what a callback implementation should do.
 *                  This callback should parse a session ticket as generated
 *                  by the corresponding mbedtls_ssl_ticket_write_t function,
 *                  and, if the ticket is authentic and valid, load the
 *                  session.
 *
 * \note            The implementation is allowed to modify the first len
 *                  bytes of the input buffer, eg to use it as a temporary
 *                  area for the decrypted ticket contents.
 *
 * \param p_ticket  Context for the callback
 * \param session   SSL session to be loaded
 * \param buf       Start of the buffer containing the ticket
 * \param len       Length of the ticket.
 *
 * \return          0 if successful, or
 *                  MBEDTLS_ERR_SSL_INVALID_MAC if not authentic, or
 *                  MBEDTLS_ERR_SSL_SESSION_TICKET_EXPIRED if expired, or
 *                  any other non-zero code for other failures.
 */
typedef int mbedtls_ssl_ticket_parse_t( void *p_ticket,
                                        mbedtls_ssl_session *session,
                                        unsigned char *buf,
                                        size_t len );

/**
 * \brief          Callback type: generate a cookie
 *
 * \param ctx      Context for the callback
 * \param p        Buffer to write to,
 *                 must be updated to point right after the cookie
 * \param end      Pointer to one past the end of the output buffer
 * \param info     Client ID info that was passed to
 *                 \c mbedtls_ssl_set_client_transport_id()
 * \param ilen     Length of info in bytes
 *
 * \return         The callback must return 0 on success,
 *                 or a negative error code.
 */
typedef int mbedtls_ssl_cookie_write_t( void *ctx,
                                        unsigned char **p, unsigned char *end,
                                        const unsigned char *info, size_t ilen );

/**
 * \brief          Callback type: verify a cookie
 *
 * \param ctx      Context for the callback
 * \param cookie   Cookie to verify
 * \param clen     Length of cookie
 * \param info     Client ID info that was passed to
 *                 \c mbedtls_ssl_set_client_transport_id()
 * \param ilen     Length of info in bytes
 *
 * \return         The callback must return 0 if cookie is valid,
 *                 or a negative error code.
 */
typedef int mbedtls_ssl_cookie_check_t( void *ctx,
                                        const unsigned char *cookie, size_t clen,
                                        const unsigned char *info, size_t ilen );

const char *mbedtls_sig_alg_name(int);
const char *mbedtls_ssl_get_alpn_protocol( const mbedtls_ssl_context * );
const char *mbedtls_ssl_get_ciphersuite( const mbedtls_ssl_context * );
const char *mbedtls_ssl_get_ciphersuite_name( const int );
const char *mbedtls_ssl_get_srtp_profile_as_string( mbedtls_ssl_srtp_profile );
const char *mbedtls_ssl_get_version( const mbedtls_ssl_context * );
const mbedtls_ssl_session *mbedtls_ssl_get_session_pointer( const mbedtls_ssl_context * );
const mbedtls_x509_crt *mbedtls_ssl_get_peer_cert( const mbedtls_ssl_context * );
int mbedtls_ssl_check_pending( const mbedtls_ssl_context * );
int mbedtls_ssl_check_record( mbedtls_ssl_context const *, unsigned char *, size_t );
int mbedtls_ssl_close_notify( mbedtls_ssl_context * );
int mbedtls_ssl_conf_alpn_protocols( mbedtls_ssl_config *, const char ** );
int mbedtls_ssl_conf_cid( mbedtls_ssl_config *, size_t, int );
int mbedtls_ssl_conf_dh_param_bin( mbedtls_ssl_config *, const unsigned char *, size_t, const unsigned char *,  size_t );
int mbedtls_ssl_conf_dh_param_ctx( mbedtls_ssl_config *, mbedtls_dhm_context * );
int mbedtls_ssl_conf_dtls_srtp_protection_profiles( mbedtls_ssl_config *, const mbedtls_ssl_srtp_profile * );
int mbedtls_ssl_conf_max_frag_len( mbedtls_ssl_config *, unsigned char );
int mbedtls_ssl_conf_own_cert( mbedtls_ssl_config *, mbedtls_x509_crt *, mbedtls_pk_context * );
int mbedtls_ssl_conf_psk( mbedtls_ssl_config *, const void *, size_t, const void *, size_t );
int mbedtls_ssl_context_load( mbedtls_ssl_context *,  const unsigned char *, size_t );
int mbedtls_ssl_context_save( mbedtls_ssl_context *, unsigned char *, size_t, size_t * );
int mbedtls_ssl_get_ciphersuite_id( const char * );
int mbedtls_ssl_get_max_out_record_payload( const mbedtls_ssl_context * );
int mbedtls_ssl_get_peer_cid( mbedtls_ssl_context *, int *, unsigned char[ MBEDTLS_SSL_CID_OUT_LEN_MAX ], size_t * );
int mbedtls_ssl_get_record_expansion( const mbedtls_ssl_context * );
int mbedtls_ssl_get_session( const mbedtls_ssl_context *, mbedtls_ssl_session * );
int mbedtls_ssl_handshake( mbedtls_ssl_context * );
int mbedtls_ssl_handshake_client_step( mbedtls_ssl_context * );
int mbedtls_ssl_handshake_server_step( mbedtls_ssl_context * );
int mbedtls_ssl_handshake_step( mbedtls_ssl_context * );
int mbedtls_ssl_read( mbedtls_ssl_context *, void *, size_t );
int mbedtls_ssl_renegotiate( mbedtls_ssl_context * );
int mbedtls_ssl_send_alert_message( mbedtls_ssl_context *, unsigned char, unsigned char );
int mbedtls_ssl_session_load( mbedtls_ssl_session *, const unsigned char *, size_t );
int mbedtls_ssl_session_reset( mbedtls_ssl_context * );
int mbedtls_ssl_session_save( const mbedtls_ssl_session *, unsigned char *, size_t, size_t * );
int mbedtls_ssl_set_cid( mbedtls_ssl_context *, int, unsigned char const *, size_t );
int mbedtls_ssl_set_client_transport_id( mbedtls_ssl_context *, const unsigned char *, size_t );
int mbedtls_ssl_set_hostname( mbedtls_ssl_context *, const char * );
int mbedtls_ssl_set_hs_ecjpake_password( mbedtls_ssl_context *, const unsigned char *, size_t );
int mbedtls_ssl_set_hs_own_cert( mbedtls_ssl_context *, mbedtls_x509_crt *, mbedtls_pk_context * );
int mbedtls_ssl_set_hs_psk( mbedtls_ssl_context *, const void *, size_t );
int mbedtls_ssl_set_session( mbedtls_ssl_context *, const mbedtls_ssl_session * );
int mbedtls_ssl_setup( mbedtls_ssl_context *, const mbedtls_ssl_config * );
int mbedtls_ssl_tls_prf( const mbedtls_tls_prf_types , const unsigned char *, size_t, const char *, const unsigned char *, size_t, unsigned char *, size_t );
int mbedtls_ssl_write( mbedtls_ssl_context *, const void *, size_t );
size_t mbedtls_ssl_get_bytes_avail( const mbedtls_ssl_context * );
size_t mbedtls_ssl_get_input_max_frag_len( const mbedtls_ssl_context * );
size_t mbedtls_ssl_get_output_max_frag_len( const mbedtls_ssl_context * );
uint32_t mbedtls_ssl_get_verify_result( const mbedtls_ssl_context * );
void *mbedtls_ssl_conf_get_async_config_data( const mbedtls_ssl_config * );
void *mbedtls_ssl_get_async_operation_data( const mbedtls_ssl_context * );
void mbedtls_ssl_conf_async_private_cb( mbedtls_ssl_config *, mbedtls_ssl_async_sign_t *, mbedtls_ssl_async_decrypt_t *, mbedtls_ssl_async_resume_t *, mbedtls_ssl_async_cancel_t *, void * );
void mbedtls_ssl_conf_authmode( mbedtls_ssl_config *, int );
void mbedtls_ssl_conf_ca_cb( mbedtls_ssl_config *, mbedtls_x509_crt_ca_cb_t, void * );
void mbedtls_ssl_conf_ca_chain( mbedtls_ssl_config *, mbedtls_x509_crt *, mbedtls_x509_crl * );
void mbedtls_ssl_conf_cbc_record_splitting( mbedtls_ssl_config *, char );
void mbedtls_ssl_conf_cert_profile( mbedtls_ssl_config *, const mbedtls_x509_crt_profile * );
void mbedtls_ssl_conf_cert_req_ca_list( mbedtls_ssl_config *, char );
void mbedtls_ssl_conf_ciphersuites( mbedtls_ssl_config *, const uint16_t * );
void mbedtls_ssl_conf_ciphersuites_for_version( mbedtls_ssl_config *, const uint16_t *, int, int );
void mbedtls_ssl_conf_curves( mbedtls_ssl_config *, const mbedtls_ecp_group_id * );
void mbedtls_ssl_conf_dbg( mbedtls_ssl_config *, void (*)(void *, int, const char *, int, const char *), void * );
void mbedtls_ssl_conf_dhm_min_bitlen( mbedtls_ssl_config *, unsigned int );
void mbedtls_ssl_conf_dtls_anti_replay( mbedtls_ssl_config *, char );
void mbedtls_ssl_conf_dtls_badmac_limit( mbedtls_ssl_config *, unsigned );
void mbedtls_ssl_conf_dtls_cookies( mbedtls_ssl_config *, mbedtls_ssl_cookie_write_t *, mbedtls_ssl_cookie_check_t *, void * );
void mbedtls_ssl_conf_encrypt_then_mac( mbedtls_ssl_config *, char );
void mbedtls_ssl_conf_endpoint( mbedtls_ssl_config *, int );
void mbedtls_ssl_conf_export_keys_cb( mbedtls_ssl_config *, mbedtls_ssl_export_keys_t *, void * );
void mbedtls_ssl_conf_export_keys_ext_cb( mbedtls_ssl_config *, mbedtls_ssl_export_keys_ext_t *, void * );
void mbedtls_ssl_conf_extended_master_secret( mbedtls_ssl_config *, char );
void mbedtls_ssl_conf_fallback( mbedtls_ssl_config *, char );
void mbedtls_ssl_conf_handshake_timeout( mbedtls_ssl_config *, uint32_t, uint32_t );
void mbedtls_ssl_conf_legacy_renegotiation( mbedtls_ssl_config *, int );
void mbedtls_ssl_conf_max_version( mbedtls_ssl_config *, int, int );
void mbedtls_ssl_conf_min_version( mbedtls_ssl_config *, int, int );
void mbedtls_ssl_conf_psk_cb( mbedtls_ssl_config *, int (*)(void *, mbedtls_ssl_context *, const unsigned char *, size_t), void * );
void mbedtls_ssl_conf_read_timeout( mbedtls_ssl_config *, uint32_t );
void mbedtls_ssl_conf_renegotiation( mbedtls_ssl_config *, int );
void mbedtls_ssl_conf_renegotiation_enforced( mbedtls_ssl_config *, int );
void mbedtls_ssl_conf_renegotiation_period( mbedtls_ssl_config *, const unsigned char[8] );
void mbedtls_ssl_conf_rng( mbedtls_ssl_config *, int (*)(void *, unsigned char *, size_t), void * );
void mbedtls_ssl_conf_session_cache( mbedtls_ssl_config *, void *, int (*)(void *, mbedtls_ssl_session *), int (*)(void *, const mbedtls_ssl_session *) );
void mbedtls_ssl_conf_session_tickets( mbedtls_ssl_config *, int );
void mbedtls_ssl_conf_session_tickets_cb( mbedtls_ssl_config *, mbedtls_ssl_ticket_write_t *, mbedtls_ssl_ticket_parse_t *, void * );
void mbedtls_ssl_conf_sig_hashes( mbedtls_ssl_config *, const uint8_t * );
void mbedtls_ssl_conf_sni( mbedtls_ssl_config *, int (*)(void *, mbedtls_ssl_context *, const unsigned char *, size_t), void *);
void mbedtls_ssl_conf_srtp_mki_value_supported( mbedtls_ssl_config *, int );
void mbedtls_ssl_conf_transport( mbedtls_ssl_config *, int );
void mbedtls_ssl_conf_verify( mbedtls_ssl_config *, int (*)(void *, mbedtls_x509_crt *, int, uint32_t *), void * );
void mbedtls_ssl_config_free( mbedtls_ssl_config * );
void mbedtls_ssl_config_init( mbedtls_ssl_config * );
void mbedtls_ssl_free( mbedtls_ssl_context * );
void mbedtls_ssl_get_dtls_srtp_negotiation_result( const mbedtls_ssl_context *, mbedtls_dtls_srtp_info * );
void mbedtls_ssl_init( mbedtls_ssl_context * );
void mbedtls_ssl_key_cert_free( mbedtls_ssl_key_cert * );
void mbedtls_ssl_session_free( mbedtls_ssl_session * );
void mbedtls_ssl_session_init( mbedtls_ssl_session * );
void mbedtls_ssl_set_async_operation_data( mbedtls_ssl_context *, void * );
void mbedtls_ssl_set_bio( mbedtls_ssl_context *, void *, mbedtls_ssl_send_t *, mbedtls_ssl_recv_t *, mbedtls_ssl_recv_timeout_t * );
void mbedtls_ssl_set_datagram_packing( mbedtls_ssl_context *, unsigned );
void mbedtls_ssl_set_hs_authmode( mbedtls_ssl_context *, int );
void mbedtls_ssl_set_hs_ca_chain( mbedtls_ssl_context *, mbedtls_x509_crt *, mbedtls_x509_crl * );
void mbedtls_ssl_set_mtu( mbedtls_ssl_context *, uint16_t );
void mbedtls_ssl_set_timer_cb( mbedtls_ssl_context *, void *, mbedtls_ssl_set_timer_t *, mbedtls_ssl_get_timer_t * );
void mbedtls_ssl_set_verify( mbedtls_ssl_context *, int (*)(void *, mbedtls_x509_crt *, int, uint32_t *), void * );

/**
 * \brief           Load reasonnable default SSL configuration values.
 *                  (You need to call mbedtls_ssl_config_init() first.)
 *
 * \param conf      SSL configuration context
 * \param endpoint  MBEDTLS_SSL_IS_CLIENT or MBEDTLS_SSL_IS_SERVER
 * \param transport MBEDTLS_SSL_TRANSPORT_STREAM for TLS, or
 *                  MBEDTLS_SSL_TRANSPORT_DATAGRAM for DTLS
 * \param preset    a MBEDTLS_SSL_PRESET_XXX value
 *
 * \note            See \c mbedtls_ssl_conf_transport() for notes on DTLS.
 *
 * \return          0 if successful, or
 *                  MBEDTLS_ERR_XXX_ALLOC_FAILED on memory allocation error.
 */
forceinline int mbedtls_ssl_config_defaults( mbedtls_ssl_config *conf,
                                             int endpoint, int transport, 
                                             int preset ) {
  int mbedtls_ssl_config_defaults_impl(mbedtls_ssl_config *, int, int, int,
                                       int (*)(mbedtls_ssl_context *));
  switch (endpoint) {
#if defined(MBEDTLS_SSL_CLI_C)
    case MBEDTLS_SSL_IS_CLIENT:
      return mbedtls_ssl_config_defaults_impl(
          conf, endpoint, transport, preset, 
          mbedtls_ssl_handshake_client_step);
#endif
#if defined(MBEDTLS_SSL_SRV_C)
    case MBEDTLS_SSL_IS_SERVER:
      return mbedtls_ssl_config_defaults_impl(
          conf, endpoint, transport, preset, 
          mbedtls_ssl_handshake_server_step);
#endif
    default:
      return MBEDTLS_ERR_SSL_FEATURE_UNAVAILABLE;
  }
}

const char *GetSslStateName(mbedtls_ssl_states);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_H_ */
