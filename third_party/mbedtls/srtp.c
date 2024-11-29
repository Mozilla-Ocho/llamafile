/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2021 Justine Alexandra Roberts Tunney                              │
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
#include "third_party/mbedtls/ssl.h"

const char *mbedtls_ssl_get_srtp_profile_as_string(
    mbedtls_ssl_srtp_profile profile) {
  switch (profile) {
    case MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_80:
      return "MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_80";
    case MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_32:
      return "MBEDTLS_TLS_SRTP_AES128_CM_HMAC_SHA1_32";
    case MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_80:
      return "MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_80";
    case MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_32:
      return "MBEDTLS_TLS_SRTP_NULL_HMAC_SHA1_32";
    default:
      return "";
  }
}
