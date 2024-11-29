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

const char *GetSslStateName(mbedtls_ssl_states x) {
  switch (x) {
    case MBEDTLS_SSL_HELLO_REQUEST:
      return "HELLO_REQUEST";
    case MBEDTLS_SSL_CLIENT_HELLO:
      return "CLIENT_HELLO";
    case MBEDTLS_SSL_SERVER_HELLO:
      return "SERVER_HELLO";
    case MBEDTLS_SSL_SERVER_CERTIFICATE:
      return "SERVER_CERTIFICATE";
    case MBEDTLS_SSL_SERVER_KEY_EXCHANGE:
      return "SERVER_KEY_EXCHANGE";
    case MBEDTLS_SSL_CERTIFICATE_REQUEST:
      return "CERTIFICATE_REQUEST";
    case MBEDTLS_SSL_SERVER_HELLO_DONE:
      return "SERVER_HELLO_DONE";
    case MBEDTLS_SSL_CLIENT_CERTIFICATE:
      return "CLIENT_CERTIFICATE";
    case MBEDTLS_SSL_CLIENT_KEY_EXCHANGE:
      return "CLIENT_KEY_EXCHANGE";
    case MBEDTLS_SSL_CERTIFICATE_VERIFY:
      return "CERTIFICATE_VERIFY";
    case MBEDTLS_SSL_CLIENT_CHANGE_CIPHER_SPEC:
      return "CLIENT_CHANGE_CIPHER_SPEC";
    case MBEDTLS_SSL_CLIENT_FINISHED:
      return "CLIENT_FINISHED";
    case MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC:
      return "SERVER_CHANGE_CIPHER_SPEC";
    case MBEDTLS_SSL_SERVER_FINISHED:
      return "SERVER_FINISHED";
    case MBEDTLS_SSL_FLUSH_BUFFERS:
      return "FLUSH_BUFFERS";
    case MBEDTLS_SSL_HANDSHAKE_WRAPUP:
      return "HANDSHAKE_WRAPUP";
    case MBEDTLS_SSL_HANDSHAKE_OVER:
      return "HANDSHAKE_OVER";
    case MBEDTLS_SSL_SERVER_NEW_SESSION_TICKET:
      return "SERVER_NEW_SESSION_TICKET";
    case MBEDTLS_SSL_SERVER_HELLO_VERIFY_REQUEST_SENT:
      return "SERVER_HELLO_VERIFY_REQUEST_SENT";
    default:
      return NULL;
  }
}
