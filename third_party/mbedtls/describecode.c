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
#include <stdbool.h>
#include <libc/fmt/itoa.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/iana.h"

const char *DescribeMbedtlsErrorCode(int ret) {
  static _Thread_local char sslerr[64];
  char *p = sslerr;
  p = stpcpy(p, "mbedtls error code ");
  if (-ret <= 0xffffu) {
    *p++ = '-';
    *p++ = '0';
    *p++ = 'x';
    *p++ = "0123456789abcdef"[(-ret & 0xf000) >> 12];
    *p++ = "0123456789abcdef"[(-ret & 0x0f00) >> 8];
    *p++ = "0123456789abcdef"[(-ret & 0x00f0) >> 4];
    *p++ = "0123456789abcdef"[(-ret & 0x000f) >> 0];
    *p = 0;
  } else {
    FormatInt32(p, ret);
  }
  return sslerr;
}
