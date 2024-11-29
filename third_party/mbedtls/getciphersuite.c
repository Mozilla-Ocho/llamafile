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
#include "third_party/mbedtls/ssl_ciphersuites.h"

#define S32(S) (S[0] << 24 | S[1] << 16 | S[2] << 8 | S[3])

/**
 * Returns ciphersuite info by IANA name.
 *
 * This API provides some wiggle room for naming, e.g.
 *
 * - ECDHE-ECDSA-AES256-GCM-SHA384 (preferred)
 * - ECDHE-ECDSA-WITH-AES-256-GCM-SHA384
 * - TLS-ECDHE-ECDSA-WITH-AES-256-GCM-SHA384
 * - TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 (canonical)
 *
 * All of the above are acceptable names for 0xC02C.
 */
const mbedtls_ssl_ciphersuite_t *GetCipherSuite(const char *s) {
  int i, j;
  char b[50];
  uint32_t w;
  unsigned char c;
  for (i = j = w = 0; (c = s[i++]);) {
    if (c == '_') c = '-';                     // _       → -
    if ('a' <= c && c <= 'z') c -= 'a' - 'A';  // a-z     → A-Z
    if (c == '-' && w == S32("WITH")) j -= 5;  // WITH-   → -
    if (w == S32("TLS-")) j -= 4;              // TLS-    →
    w = w << 8 | c;                            // -------   ------
    if (w == S32("AES-")) continue;            // AES-XXX → AESXXX
    if (w == S32("SHA1")) continue;            // SHA1    → SHA
    if (!(0 <= j && j + 1 < sizeof(b))) return 0;
    b[j++] = c;
  }
  b[j++] = 0;
  return mbedtls_ssl_ciphersuite_from_string(b);
}
