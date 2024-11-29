#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_CIPHERSUITES_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_CIPHERSUITES_H_
#include "third_party/mbedtls/cipher.h"
#include "third_party/mbedtls/config.h"
#include "third_party/mbedtls/md.h"
#include "third_party/mbedtls/pk.h"
COSMOPOLITAN_C_START_

/*
 * Supported ciphersuites (Official IANA names)
 */
#define MBEDTLS_TLS_RSA_WITH_NULL_MD5                    0x0001   /*< Weak! */
#define MBEDTLS_TLS_RSA_WITH_NULL_SHA                    0x0002   /*< Weak! */

#define MBEDTLS_TLS_RSA_WITH_RC4_128_MD5                 0x0004
#define MBEDTLS_TLS_RSA_WITH_RC4_128_SHA                 0x0005
#define MBEDTLS_TLS_RSA_WITH_DES_CBC_SHA                 0x0009   /*< Weak! Not in TLS 1.2 */

#define MBEDTLS_TLS_RSA_WITH_3DES_EDE_CBC_SHA            0x000A

#define MBEDTLS_TLS_DHE_RSA_WITH_DES_CBC_SHA             0x0015   /*< Weak! Not in TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA        0x0016

#define MBEDTLS_TLS_PSK_WITH_NULL_SHA                    0x002C   /*< Weak! */
#define MBEDTLS_TLS_DHE_PSK_WITH_NULL_SHA                0x002D   /*< Weak! */
#define MBEDTLS_TLS_RSA_PSK_WITH_NULL_SHA                0x002E   /*< Weak! */
#define MBEDTLS_TLS_RSA_WITH_AES_128_CBC_SHA             0x002F

#define MBEDTLS_TLS_DHE_RSA_WITH_AES_128_CBC_SHA         0x0033
#define MBEDTLS_TLS_RSA_WITH_AES_256_CBC_SHA             0x0035
#define MBEDTLS_TLS_DHE_RSA_WITH_AES_256_CBC_SHA         0x0039

#define MBEDTLS_TLS_RSA_WITH_NULL_SHA256                 0x003B   /*< Weak! */
#define MBEDTLS_TLS_RSA_WITH_AES_128_CBC_SHA256          0x003C   /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_WITH_AES_256_CBC_SHA256          0x003D   /*< TLS 1.2 */

#define MBEDTLS_TLS_RSA_WITH_CAMELLIA_128_CBC_SHA        0x0041
#define MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA    0x0045

#define MBEDTLS_TLS_DHE_RSA_WITH_AES_128_CBC_SHA256      0x0067   /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_AES_256_CBC_SHA256      0x006B   /*< TLS 1.2 */

#define MBEDTLS_TLS_RSA_WITH_CAMELLIA_256_CBC_SHA        0x0084
#define MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA    0x0088

#define MBEDTLS_TLS_PSK_WITH_RC4_128_SHA                 0x008A
#define MBEDTLS_TLS_PSK_WITH_3DES_EDE_CBC_SHA            0x008B
#define MBEDTLS_TLS_PSK_WITH_AES_128_CBC_SHA             0x008C
#define MBEDTLS_TLS_PSK_WITH_AES_256_CBC_SHA             0x008D

#define MBEDTLS_TLS_DHE_PSK_WITH_RC4_128_SHA             0x008E
#define MBEDTLS_TLS_DHE_PSK_WITH_3DES_EDE_CBC_SHA        0x008F
#define MBEDTLS_TLS_DHE_PSK_WITH_AES_128_CBC_SHA         0x0090
#define MBEDTLS_TLS_DHE_PSK_WITH_AES_256_CBC_SHA         0x0091

#define MBEDTLS_TLS_RSA_PSK_WITH_RC4_128_SHA             0x0092
#define MBEDTLS_TLS_RSA_PSK_WITH_3DES_EDE_CBC_SHA        0x0093
#define MBEDTLS_TLS_RSA_PSK_WITH_AES_128_CBC_SHA         0x0094
#define MBEDTLS_TLS_RSA_PSK_WITH_AES_256_CBC_SHA         0x0095

#define MBEDTLS_TLS_RSA_WITH_AES_128_GCM_SHA256          0x009C   /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_WITH_AES_256_GCM_SHA384          0x009D   /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_AES_128_GCM_SHA256      0x009E   /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_AES_256_GCM_SHA384      0x009F   /*< TLS 1.2 */

#define MBEDTLS_TLS_PSK_WITH_AES_128_GCM_SHA256          0x00A8   /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_AES_256_GCM_SHA384          0x00A9   /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_AES_128_GCM_SHA256      0x00AA   /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_AES_256_GCM_SHA384      0x00AB   /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_PSK_WITH_AES_128_GCM_SHA256      0x00AC   /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_PSK_WITH_AES_256_GCM_SHA384      0x00AD   /*< TLS 1.2 */

#define MBEDTLS_TLS_PSK_WITH_AES_128_CBC_SHA256          0x00AE
#define MBEDTLS_TLS_PSK_WITH_AES_256_CBC_SHA384          0x00AF
#define MBEDTLS_TLS_PSK_WITH_NULL_SHA256                 0x00B0   /*< Weak! */
#define MBEDTLS_TLS_PSK_WITH_NULL_SHA384                 0x00B1   /*< Weak! */

#define MBEDTLS_TLS_DHE_PSK_WITH_AES_128_CBC_SHA256      0x00B2
#define MBEDTLS_TLS_DHE_PSK_WITH_AES_256_CBC_SHA384      0x00B3
#define MBEDTLS_TLS_DHE_PSK_WITH_NULL_SHA256             0x00B4   /*< Weak! */
#define MBEDTLS_TLS_DHE_PSK_WITH_NULL_SHA384             0x00B5   /*< Weak! */

#define MBEDTLS_TLS_RSA_PSK_WITH_AES_128_CBC_SHA256      0x00B6
#define MBEDTLS_TLS_RSA_PSK_WITH_AES_256_CBC_SHA384      0x00B7
#define MBEDTLS_TLS_RSA_PSK_WITH_NULL_SHA256             0x00B8   /*< Weak! */
#define MBEDTLS_TLS_RSA_PSK_WITH_NULL_SHA384             0x00B9   /*< Weak! */

#define MBEDTLS_TLS_RSA_WITH_CAMELLIA_128_CBC_SHA256     0x00BA   /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA256 0x00BE   /*< TLS 1.2 */

#define MBEDTLS_TLS_RSA_WITH_CAMELLIA_256_CBC_SHA256     0x00C0   /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA256 0x00C4   /*< TLS 1.2 */

#define MBEDTLS_TLS_ECDH_ECDSA_WITH_NULL_SHA             0xC001 /*< Weak! */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_RC4_128_SHA          0xC002 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_3DES_EDE_CBC_SHA     0xC003 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA      0xC004 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA      0xC005 /*< Not in SSL3! */

#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_NULL_SHA            0xC006 /*< Weak! */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_RC4_128_SHA         0xC007 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_3DES_EDE_CBC_SHA    0xC008 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA     0xC009 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA     0xC00A /*< Not in SSL3! */

#define MBEDTLS_TLS_ECDH_RSA_WITH_NULL_SHA               0xC00B /*< Weak! */
#define MBEDTLS_TLS_ECDH_RSA_WITH_RC4_128_SHA            0xC00C /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_RSA_WITH_3DES_EDE_CBC_SHA       0xC00D /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_RSA_WITH_AES_128_CBC_SHA        0xC00E /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_RSA_WITH_AES_256_CBC_SHA        0xC00F /*< Not in SSL3! */

#define MBEDTLS_TLS_ECDHE_RSA_WITH_NULL_SHA              0xC010 /*< Weak! */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_RC4_128_SHA           0xC011 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA      0xC012 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA       0xC013 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA       0xC014 /*< Not in SSL3! */

#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256  0xC023 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384  0xC024 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA256   0xC025 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA384   0xC026 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256    0xC027 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384    0xC028 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_AES_128_CBC_SHA256     0xC029 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_AES_256_CBC_SHA384     0xC02A /*< TLS 1.2 */

#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256  0xC02B /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384  0xC02C /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256   0xC02D /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_AES_256_GCM_SHA384   0xC02E /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256    0xC02F /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384    0xC030 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_AES_128_GCM_SHA256     0xC031 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_AES_256_GCM_SHA384     0xC032 /*< TLS 1.2 */

#define MBEDTLS_TLS_ECDHE_PSK_WITH_RC4_128_SHA           0xC033 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_3DES_EDE_CBC_SHA      0xC034 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA       0xC035 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA       0xC036 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA256    0xC037 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA384    0xC038 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_NULL_SHA              0xC039 /*< Weak! No SSL3! */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_NULL_SHA256           0xC03A /*< Weak! No SSL3! */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_NULL_SHA384           0xC03B /*< Weak! No SSL3! */

#define MBEDTLS_TLS_RSA_WITH_ARIA_128_CBC_SHA256         0xC03C /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_WITH_ARIA_256_CBC_SHA384         0xC03D /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_ARIA_128_CBC_SHA256     0xC044 /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_ARIA_256_CBC_SHA384     0xC045 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_ARIA_128_CBC_SHA256 0xC048 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_ARIA_256_CBC_SHA384 0xC049 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_ARIA_128_CBC_SHA256  0xC04A /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_ARIA_256_CBC_SHA384  0xC04B /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_ARIA_128_CBC_SHA256   0xC04C /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_ARIA_256_CBC_SHA384   0xC04D /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_ARIA_128_CBC_SHA256    0xC04E /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_ARIA_256_CBC_SHA384    0xC04F /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_WITH_ARIA_128_GCM_SHA256         0xC050 /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_WITH_ARIA_256_GCM_SHA384         0xC051 /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_ARIA_128_GCM_SHA256     0xC052 /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_ARIA_256_GCM_SHA384     0xC053 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_ARIA_128_GCM_SHA256 0xC05C /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_ARIA_256_GCM_SHA384 0xC05D /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_ARIA_128_GCM_SHA256  0xC05E /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_ARIA_256_GCM_SHA384  0xC05F /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_ARIA_128_GCM_SHA256   0xC060 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_ARIA_256_GCM_SHA384   0xC061 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_ARIA_128_GCM_SHA256    0xC062 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_ARIA_256_GCM_SHA384    0xC063 /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_ARIA_128_CBC_SHA256         0xC064 /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_ARIA_256_CBC_SHA384         0xC065 /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_ARIA_128_CBC_SHA256     0xC066 /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_ARIA_256_CBC_SHA384     0xC067 /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_PSK_WITH_ARIA_128_CBC_SHA256     0xC068 /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_PSK_WITH_ARIA_256_CBC_SHA384     0xC069 /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_ARIA_128_GCM_SHA256         0xC06A /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_ARIA_256_GCM_SHA384         0xC06B /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_ARIA_128_GCM_SHA256     0xC06C /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_ARIA_256_GCM_SHA384     0xC06D /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_PSK_WITH_ARIA_128_GCM_SHA256     0xC06E /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_PSK_WITH_ARIA_256_GCM_SHA384     0xC06F /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_ARIA_128_CBC_SHA256   0xC070 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_ARIA_256_CBC_SHA384   0xC071 /*< TLS 1.2 */

#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_CBC_SHA256 0xC072 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_CBC_SHA384 0xC073 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_128_CBC_SHA256  0xC074 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_256_CBC_SHA384  0xC075 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_128_CBC_SHA256   0xC076 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_256_CBC_SHA384   0xC077 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_128_CBC_SHA256    0xC078 /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_256_CBC_SHA384    0xC079 /*< Not in SSL3! */

#define MBEDTLS_TLS_RSA_WITH_CAMELLIA_128_GCM_SHA256         0xC07A /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_WITH_CAMELLIA_256_GCM_SHA384         0xC07B /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_128_GCM_SHA256     0xC07C /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_CAMELLIA_256_GCM_SHA384     0xC07D /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_GCM_SHA256 0xC086 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_GCM_SHA384 0xC087 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_128_GCM_SHA256  0xC088 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_ECDSA_WITH_CAMELLIA_256_GCM_SHA384  0xC089 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_128_GCM_SHA256   0xC08A /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_CAMELLIA_256_GCM_SHA384   0xC08B /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_128_GCM_SHA256    0xC08C /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDH_RSA_WITH_CAMELLIA_256_GCM_SHA384    0xC08D /*< TLS 1.2 */

#define MBEDTLS_TLS_PSK_WITH_CAMELLIA_128_GCM_SHA256       0xC08E /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_CAMELLIA_256_GCM_SHA384       0xC08F /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_128_GCM_SHA256   0xC090 /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_256_GCM_SHA384   0xC091 /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_128_GCM_SHA256   0xC092 /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_256_GCM_SHA384   0xC093 /*< TLS 1.2 */

#define MBEDTLS_TLS_PSK_WITH_CAMELLIA_128_CBC_SHA256       0xC094
#define MBEDTLS_TLS_PSK_WITH_CAMELLIA_256_CBC_SHA384       0xC095
#define MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_128_CBC_SHA256   0xC096
#define MBEDTLS_TLS_DHE_PSK_WITH_CAMELLIA_256_CBC_SHA384   0xC097
#define MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_128_CBC_SHA256   0xC098
#define MBEDTLS_TLS_RSA_PSK_WITH_CAMELLIA_256_CBC_SHA384   0xC099
#define MBEDTLS_TLS_ECDHE_PSK_WITH_CAMELLIA_128_CBC_SHA256 0xC09A /*< Not in SSL3! */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_CAMELLIA_256_CBC_SHA384 0xC09B /*< Not in SSL3! */

#define MBEDTLS_TLS_RSA_WITH_AES_128_CCM                0xC09C  /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_WITH_AES_256_CCM                0xC09D  /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_AES_128_CCM            0xC09E  /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_AES_256_CCM            0xC09F  /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_WITH_AES_128_CCM_8              0xC0A0  /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_WITH_AES_256_CCM_8              0xC0A1  /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_AES_128_CCM_8          0xC0A2  /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_AES_256_CCM_8          0xC0A3  /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_AES_128_CCM                0xC0A4  /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_AES_256_CCM                0xC0A5  /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_AES_128_CCM            0xC0A6  /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_AES_256_CCM            0xC0A7  /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_AES_128_CCM_8              0xC0A8  /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_AES_256_CCM_8              0xC0A9  /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_AES_128_CCM_8          0xC0AA  /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_AES_256_CCM_8          0xC0AB  /*< TLS 1.2 */
/* The last two are named with PSK_DHE in the RFC, which looks like a typo */

#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_CCM        0xC0AC  /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_CCM        0xC0AD  /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_128_CCM_8      0xC0AE  /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_AES_256_CCM_8      0xC0AF  /*< TLS 1.2 */

#define MBEDTLS_TLS_ECJPAKE_WITH_AES_128_CCM_8          0xC0FF  /*< experimental */

/* RFC 7905 */
#define MBEDTLS_TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256   0xCCA8 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256 0xCCA9 /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256     0xCCAA /*< TLS 1.2 */
#define MBEDTLS_TLS_PSK_WITH_CHACHA20_POLY1305_SHA256         0xCCAB /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_CHACHA20_POLY1305_SHA256   0xCCAC /*< TLS 1.2 */
#define MBEDTLS_TLS_DHE_PSK_WITH_CHACHA20_POLY1305_SHA256     0xCCAD /*< TLS 1.2 */
#define MBEDTLS_TLS_RSA_PSK_WITH_CHACHA20_POLY1305_SHA256     0xCCAE /*< TLS 1.2 */

/* RFC 8442 */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_AES_128_GCM_SHA256         0xD001 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_AES_256_GCM_SHA384         0xD002 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_AES_128_CCM_8_SHA256       0xD003 /*< TLS 1.2 */
#define MBEDTLS_TLS_ECDHE_PSK_WITH_AES_128_CCM_SHA256         0xD005 /*< TLS 1.2 */

/* Reminder: update mbedtls_ssl_premaster_secret when adding a new key exchange.
 * Reminder: update MBEDTLS_KEY_EXCHANGE__xxx below
 */
typedef enum {
    MBEDTLS_KEY_EXCHANGE_NONE = 0,
    MBEDTLS_KEY_EXCHANGE_RSA = 1,
    MBEDTLS_KEY_EXCHANGE_DHE_RSA = 2,
    MBEDTLS_KEY_EXCHANGE_ECDHE_RSA = 3,
    MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA = 4,
    MBEDTLS_KEY_EXCHANGE_PSK = 5,
    MBEDTLS_KEY_EXCHANGE_DHE_PSK = 6,
    MBEDTLS_KEY_EXCHANGE_RSA_PSK = 7,
    MBEDTLS_KEY_EXCHANGE_ECDHE_PSK = 8,
    MBEDTLS_KEY_EXCHANGE_ECDH_RSA = 9,
    MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA = 10,
    MBEDTLS_KEY_EXCHANGE_ECJPAKE = 11,
} mbedtls_key_exchange_type_t;

typedef struct mbedtls_ssl_ciphersuite_t mbedtls_ssl_ciphersuite_t;

#define MBEDTLS_CIPHERSUITE_WEAK       0x01    /*< Weak ciphersuite flag  */
#define MBEDTLS_CIPHERSUITE_SHORT_TAG  0x02    /*< Short authentication tag,
                                                     eg for CCM_8 */
#define MBEDTLS_CIPHERSUITE_NODTLS     0x04    /*< Can't be used with DTLS */

/**
 * \brief   This structure is used for storing ciphersuite information
 */
struct thatispacked mbedtls_ssl_ciphersuite_t
{
    uint16_t id;
    const char * name;
    unsigned char cipher; /* mbedtls_cipher_type_t */
    unsigned char mac; /* mbedtls_md_type_t */
    unsigned char key_exchange; /* mbedtls_key_exchange_type_t */
    unsigned char min_major_ver;
    unsigned char min_minor_ver;
    unsigned char max_major_ver;
    unsigned char max_minor_ver;
    unsigned char flags;
};

const uint16_t *mbedtls_ssl_list_ciphersuites( void );

const mbedtls_ssl_ciphersuite_t *mbedtls_ssl_ciphersuite_from_string( const char *ciphersuite_name );
const mbedtls_ssl_ciphersuite_t *mbedtls_ssl_ciphersuite_from_id( int ciphersuite_id );

#if defined(MBEDTLS_PK_C)
mbedtls_pk_type_t mbedtls_ssl_get_ciphersuite_sig_pk_alg( const mbedtls_ssl_ciphersuite_t *info );
mbedtls_pk_type_t mbedtls_ssl_get_ciphersuite_sig_alg( const mbedtls_ssl_ciphersuite_t *info );
#endif

int mbedtls_ssl_ciphersuite_uses_ec( const mbedtls_ssl_ciphersuite_t *info );
int mbedtls_ssl_ciphersuite_uses_psk( const mbedtls_ssl_ciphersuite_t *info );

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_PFS_ENABLED)
static inline int mbedtls_ssl_ciphersuite_has_pfs( const mbedtls_ssl_ciphersuite_t *info )
{
    switch( info->key_exchange )
    {
        case MBEDTLS_KEY_EXCHANGE_DHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_DHE_PSK:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_PSK:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA:
        case MBEDTLS_KEY_EXCHANGE_ECJPAKE:
            return( 1 );

        default:
            return( 0 );
    }
}
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_PFS_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_NON_PFS_ENABLED)
static inline int mbedtls_ssl_ciphersuite_no_pfs( const mbedtls_ssl_ciphersuite_t *info )
{
    switch( info->key_exchange )
    {
        case MBEDTLS_KEY_EXCHANGE_ECDH_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA:
        case MBEDTLS_KEY_EXCHANGE_RSA:
        case MBEDTLS_KEY_EXCHANGE_PSK:
        case MBEDTLS_KEY_EXCHANGE_RSA_PSK:
            return( 1 );

        default:
            return( 0 );
    }
}
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_NON_PFS_ENABLED */

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_ECDH_ENABLED)
static inline int mbedtls_ssl_ciphersuite_uses_ecdh( const mbedtls_ssl_ciphersuite_t *info )
{
    switch( info->key_exchange )
    {
        case MBEDTLS_KEY_EXCHANGE_ECDH_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA:
            return( 1 );

        default:
            return( 0 );
    }
}
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_ECDH_ENABLED */

static inline int mbedtls_ssl_ciphersuite_cert_req_allowed( const mbedtls_ssl_ciphersuite_t *info )
{
    switch( info->key_exchange )
    {
        case MBEDTLS_KEY_EXCHANGE_RSA:
        case MBEDTLS_KEY_EXCHANGE_DHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDH_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA:
            return( 1 );

        default:
            return( 0 );
    }
}

static inline int mbedtls_ssl_ciphersuite_uses_srv_cert( const mbedtls_ssl_ciphersuite_t *info )
{
    if (!info) return 0; /* TODO: wut */
    switch( info->key_exchange )
    {
        case MBEDTLS_KEY_EXCHANGE_RSA:
        case MBEDTLS_KEY_EXCHANGE_RSA_PSK:
        case MBEDTLS_KEY_EXCHANGE_DHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDH_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA:
            return( 1 );

        default:
            return( 0 );
    }
}

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_DHE_ENABLED)
static inline int mbedtls_ssl_ciphersuite_uses_dhe( const mbedtls_ssl_ciphersuite_t *info )
{
    switch( info->key_exchange )
    {
        case MBEDTLS_KEY_EXCHANGE_DHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_DHE_PSK:
            return( 1 );

        default:
            return( 0 );
    }
}
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_DHE_ENABLED) */

#if defined(MBEDTLS_KEY_EXCHANGE_SOME_ECDHE_ENABLED)
static inline int mbedtls_ssl_ciphersuite_uses_ecdhe( const mbedtls_ssl_ciphersuite_t *info )
{
    switch( info->key_exchange )
    {
        case MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_PSK:
            return( 1 );

        default:
            return( 0 );
    }
}
#endif /* MBEDTLS_KEY_EXCHANGE_SOME_ECDHE_ENABLED) */

#if defined(MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED)
static inline int mbedtls_ssl_ciphersuite_uses_server_signature( const mbedtls_ssl_ciphersuite_t *info )
{
    switch( info->key_exchange )
    {
        case MBEDTLS_KEY_EXCHANGE_DHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_RSA:
        case MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA:
            return( 1 );

        default:
            return( 0 );
    }
}
#endif /* MBEDTLS_KEY_EXCHANGE_WITH_SERVER_SIGNATURE_ENABLED */

const mbedtls_ssl_ciphersuite_t *GetCipherSuite(const char *);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_CIPHERSUITES_H_ */
