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
#include <libc/calls/calls.h>
#include <libc/fmt/itoa.h>
#include <libc/mem/mem.h>
#include <libc/runtime/runtime.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/platform.h"

void mbedtls_param_failed(const char *msg, const char *file, int line) {
  char *p, *q;
  if ((p = q = malloc(128 + strlen(msg) + strlen(file)))) {
    p = stpcpy(p, file), *p++ = ':';
    p = FormatInt32(p, line);
    p = stpcpy(p, ": error: ");
    p = stpcpy(p, msg);
    p = stpcpy(p, "\r\n");
    write(2, q, p - q);
    free(q);
  }
  exit(1);
}
