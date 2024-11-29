/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2022 Justine Alexandra Roberts Tunney                              │
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
#include <libc/serialize.h>
#include <libc/macros.h>
#include <libc/stdio/append.h>
#include "third_party/mbedtls/iana.h"

/**
 * Returns string of joined list of first 𝑘 client preferred ciphers.
 * @return string that must be free'd, or null if none set
 */
__wur char *FormatSslClientCiphers(const mbedtls_ssl_context *ssl) {
  int i;
  char *b = 0;
  for (i = 0; i < ARRAYLEN(ssl->client_ciphers); ++i) {
    if (!ssl->client_ciphers[i]) break;
    if (i) appendw(&b, READ16LE(", "));
    appendf(&b, "%s[0x%04x]", GetCipherSuiteName(ssl->client_ciphers[i]),
            ssl->client_ciphers[i]);
  }
  if (i == ARRAYLEN(ssl->client_ciphers)) {
    appends(&b, ", ...");
  }
  return b;
}
