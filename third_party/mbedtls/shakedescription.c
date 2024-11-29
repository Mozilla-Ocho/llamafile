/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2023 Justine Alexandra Roberts Tunney                              │
│                                                                              │
│ Permission to use, copy, modify, and/or distribute this software for         │
│ any purpose with or without fee is hereby granted, provided that the         │
│ above copyright notice and this permission notice appear in all copies.      │
│                                                                              │
│ THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL                │
│ WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED                │
│ WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE             │
│ AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL         │
│ DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR        │
│ PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER               │
│ TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR             │
│ PERFORMANCE OF THIS SOFTWARE.                                                │
╚─────────────────────────────────────────────────────────────────────────────*/
#include "third_party/mbedtls/iana.h"
#include "third_party/mbedtls/net_sockets.h"
#include "third_party/mbedtls/ssl.h"
#include "third_party/mbedtls/x509.h"

const char *DescribeSslClientHandshakeError(const mbedtls_ssl_context *ssl,
                                            int ret) {
  switch (ret) {
    case MBEDTLS_ERR_SSL_CONN_EOF:
      return "connection eof";
    case MBEDTLS_ERR_NET_CONN_RESET:
      return "connection reset";
    case MBEDTLS_ERR_SSL_TIMEOUT:
      return "ssl timeout";
    case MBEDTLS_ERR_SSL_NO_CIPHER_CHOSEN:
      return "no cipher chosen";
    case MBEDTLS_ERR_SSL_NO_USABLE_CIPHERSUITE:
      return "no usable ciphersuite";
    case MBEDTLS_ERR_SSL_BAD_HS_PROTOCOL_VERSION:
      return "bad ssl version";
    case MBEDTLS_ERR_SSL_INVALID_MAC:
      return "bad ssl mac";
    case MBEDTLS_ERR_SSL_BAD_HS_SERVER_KEY_EXCHANGE:
      return "bad key exchange";
    case MBEDTLS_ERR_X509_CERT_VERIFY_FAILED:
      switch (ssl->session_negotiate->verify_result) {
        case MBEDTLS_X509_BADCERT_EXPIRED:
          return "the certificate validity has expired";
        case MBEDTLS_X509_BADCERT_REVOKED:
          return "the certificate has been revoked (is on a crl)";
        case MBEDTLS_X509_BADCERT_CN_MISMATCH:
          return "the certificate common name (cn) does not match with the "
                 "expected cn";
        case MBEDTLS_X509_BADCERT_NOT_TRUSTED:
          return "the certificate is not correctly signed by the trusted ca";
        case MBEDTLS_X509_BADCRL_NOT_TRUSTED:
          return "the crl is not correctly signed by the trusted ca";
        case MBEDTLS_X509_BADCRL_EXPIRED:
          return "the crl is expired";
        case MBEDTLS_X509_BADCERT_MISSING:
          return "certificate was missing";
        case MBEDTLS_X509_BADCERT_SKIP_VERIFY:
          return "certificate verification was skipped";
        case MBEDTLS_X509_BADCERT_OTHER:
          return "other reason (can be used by verify callback)";
        case MBEDTLS_X509_BADCERT_FUTURE:
          return "the certificate validity starts in the future";
        case MBEDTLS_X509_BADCRL_FUTURE:
          return "the crl is from the future";
        case MBEDTLS_X509_BADCERT_KEY_USAGE:
          return "usage does not match the keyusage extension";
        case MBEDTLS_X509_BADCERT_EXT_KEY_USAGE:
          return "usage does not match the extendedkeyusage extension";
        case MBEDTLS_X509_BADCERT_NS_CERT_TYPE:
          return "usage does not match the nscerttype extension";
        case MBEDTLS_X509_BADCERT_BAD_MD:
          return "the certificate is signed with an unacceptable hash";
        case MBEDTLS_X509_BADCERT_BAD_PK:
          return "the certificate is signed with an unacceptable pk alg (eg "
                 "rsa vs ecdsa)";
        case MBEDTLS_X509_BADCERT_BAD_KEY:
          return "the certificate is signed with an unacceptable key (eg bad "
                 "curve, rsa too short)";
        case MBEDTLS_X509_BADCRL_BAD_MD:
          return "the crl is signed with an unacceptable hash";
        case MBEDTLS_X509_BADCRL_BAD_PK:
          return "the crl is signed with an unacceptable pk alg (eg rsa vs "
                 "ecdsa)";
        case MBEDTLS_X509_BADCRL_BAD_KEY:
          return "the crl is signed with an unacceptable key (eg bad curve, "
                 "rsa too short)";
        default:
          return "verification failed";
      }
    case MBEDTLS_ERR_SSL_FATAL_ALERT_MESSAGE:
      return GetAlertDescription(ssl->fatal_alert);
    default:
      return DescribeMbedtlsErrorCode(ret);
  }
}
