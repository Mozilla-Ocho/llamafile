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
#include <stdbool.h>
#include "third_party/mbedtls/pk.h"

const char *mbedtls_pk_type_name(mbedtls_pk_type_t t) {
  switch (t) {
    case MBEDTLS_PK_NONE:
      return "NONE";
    case MBEDTLS_PK_RSA:
      return "RSA";
    case MBEDTLS_PK_ECKEY:
      return "ECKEY";
    case MBEDTLS_PK_ECKEY_DH:
      return "ECKEY_DH";
    case MBEDTLS_PK_ECDSA:
      return "ECDSA";
    case MBEDTLS_PK_RSA_ALT:
      return "RSA_ALT";
    case MBEDTLS_PK_RSASSA_PSS:
      return "RSASSA_PSS";
    case MBEDTLS_PK_OPAQUE:
      return "OPAQUE";
    default:
      return 0;
  }
}
