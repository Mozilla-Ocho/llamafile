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
#include "third_party/mbedtls/bignum_internal.h"
#include "third_party/mbedtls/platform.h"

#ifdef __x86_64__

typedef uint64_t xmm_t __attribute__((__vector_size__(16), __aligned__(1)));

void ShiftRightAvx(uint64_t *p, size_t n, unsigned char k) {
  uint64_t p1;
  xmm_t o0, o1;
  xmm_t i0, i1;
  xmm_t cv = {0};
  MBEDTLS_ASSERT(!(k & ~63));
  p1 = n > 1 ? p[1] : 0;
  while (n >= 4) {
    n -= 4;
    i0 = *(xmm_t *)(p + n + 2);
    i1 = *(xmm_t *)(p + n + 0);
    o0 = i0 >> k | (xmm_t){i0[1], cv[0]} << (64 - k);
    o1 = i1 >> k | (xmm_t){i1[1], i0[0]} << (64 - k);
    cv = i1;
    *(xmm_t *)(p + n + 2) = o0;
    *(xmm_t *)(p + n + 0) = o1;
  }
  if (n >= 2) {
    n -= 2;
    i0 = *(xmm_t *)(p + n);
    o0 = i0 >> k | (xmm_t){i0[1], cv[0]} << (64 - k);
    cv = i0;
    *(xmm_t *)(p + n) = o0;
  }
  if (n) {
    p[0] = p[0] >> k | p1 << (64 - k);
  }
}

#endif /* __x86_64__ */
