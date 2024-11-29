/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:2;coding:utf-8 -*-│
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
#include <libc/fmt/itoa.h>
#include "third_party/mbedtls/iana.h"
#include "third_party/mbedtls/ssl.h"

/**
 * Returns SSL fatal alert description.
 * @see RFC5246 §7.2
 */
const char *GetAlertDescription(unsigned char x) {
  static _Thread_local char buf[21];
  switch (x) {
    case MBEDTLS_SSL_ALERT_MSG_CLOSE_NOTIFY: /* 0 */
      return "close_notify";
    case MBEDTLS_SSL_ALERT_MSG_UNEXPECTED_MESSAGE: /* 10 */
      return "unexpected_message";
    case MBEDTLS_SSL_ALERT_MSG_BAD_RECORD_MAC: /* 20 */
      return "bad_record_mac";
    case MBEDTLS_SSL_ALERT_MSG_DECRYPTION_FAILED: /* 21 */
      return "decryption_failed";
    case MBEDTLS_SSL_ALERT_MSG_RECORD_OVERFLOW: /* 22 */
      return "record_overflow";
    case MBEDTLS_SSL_ALERT_MSG_DECOMPRESSION_FAILURE: /* 30 */
      return "decompression_failure";
    case MBEDTLS_SSL_ALERT_MSG_HANDSHAKE_FAILURE: /* 40 */
      return "handshake_failure";
    case MBEDTLS_SSL_ALERT_MSG_NO_CERT: /* 41 */
      return "no_cert";
    case MBEDTLS_SSL_ALERT_MSG_BAD_CERT: /* 42 */
      return "bad_cert";
    case MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_CERT: /* 43 */
      return "unsupported_cert";
    case MBEDTLS_SSL_ALERT_MSG_CERT_REVOKED: /* 44 */
      return "cert_revoked";
    case MBEDTLS_SSL_ALERT_MSG_CERT_EXPIRED: /* 45 */
      return "cert_expired";
    case MBEDTLS_SSL_ALERT_MSG_CERT_UNKNOWN: /* 46 */
      return "cert_unknown";
    case MBEDTLS_SSL_ALERT_MSG_ILLEGAL_PARAMETER: /* 47 */
      return "illegal_parameter";
    case MBEDTLS_SSL_ALERT_MSG_UNKNOWN_CA: /* 48 */
      return "unknown_ca";
    case MBEDTLS_SSL_ALERT_MSG_ACCESS_DENIED: /* 49 */
      return "access_denied";
    case MBEDTLS_SSL_ALERT_MSG_DECODE_ERROR: /* 50 */
      return "decode_error";
    case MBEDTLS_SSL_ALERT_MSG_DECRYPT_ERROR: /* 51 */
      return "decrypt_error";
    case MBEDTLS_SSL_ALERT_MSG_EXPORT_RESTRICTION: /* 60 */
      return "export_restriction";
    case MBEDTLS_SSL_ALERT_MSG_PROTOCOL_VERSION: /* 70 */
      return "protocol_version";
    case MBEDTLS_SSL_ALERT_MSG_INSUFFICIENT_SECURITY: /* 71 */
      return "insufficient_security";
    case MBEDTLS_SSL_ALERT_MSG_INTERNAL_ERROR: /* 80 */
      return "internal_error";
    case MBEDTLS_SSL_ALERT_MSG_USER_CANCELED: /* 90 */
      return "user_canceled";
    case MBEDTLS_SSL_ALERT_MSG_NO_RENEGOTIATION: /* 100 */
      return "no_renegotiation";
    case MBEDTLS_SSL_ALERT_MSG_UNSUPPORTED_EXT: /* 110 */
      return "unsupported_extension";
    default:
      FormatUint32(buf, x);
      return buf;
  }
}
