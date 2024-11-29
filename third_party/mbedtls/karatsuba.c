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
#include <libc/nexgen32e/x86feature.h>
#include <libc/runtime/runtime.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/bignum_internal.h"
#include "third_party/mbedtls/math.h"
#include "third_party/mbedtls/platform.h"

forceinline int Cmp(uint64_t *a, uint64_t *b, size_t n) {
  uint64_t x, y;
  while (n--) {
    x = a[n];
    y = b[n];
    if (x != y) {
      return x > y ? 1 : -1;
    }
  }
  return 0;
}

forceinline bool Sub(uint64_t *C, uint64_t *A, uint64_t *B, size_t n) {
  bool cf;
  uint64_t i;
#ifdef __x86_64__
  uint64_t c;
  asm volatile("xor\t%1,%1\n\t"
               ".align\t16\n1:\t"
               "mov\t(%5,%3,8),%1\n\t"
               "sbb\t(%6,%3,8),%1\n\t"
               "mov\t%1,(%4,%3,8)\n\t"
               "lea\t1(%3),%3\n\t"
               "dec\t%2\n\t"
               "jnz\t1b"
               : "=@ccb"(cf), "=&r"(c), "+c"(n), "=r"(i)
               : "r"(C), "r"(A), "r"(B), "3"(0)
               : "cc", "memory");
#else
  for (cf = false, i = 0; i < n; ++i) {
    SBB(C[i], A[i], B[i], cf, cf);
  }
#endif
  return cf;
}

forceinline bool Add(uint64_t *C, uint64_t *A, uint64_t *B, size_t n) {
  bool cf;
  uint64_t i;
#ifdef __x86_64__
  uint64_t c;
  asm volatile("xor\t%1,%1\n\t"
               ".align\t16\n1:\t"
               "mov\t(%5,%3,8),%1\n\t"
               "adc\t(%6,%3,8),%1\n\t"
               "mov\t%1,(%4,%3,8)\n\t"
               "lea\t1(%3),%3\n\t"
               "dec\t%2\n\t"
               "jnz\t1b"
               : "=@ccc"(cf), "=&r"(c), "+c"(n), "=r"(i)
               : "r"(C), "r"(A), "r"(B), "3"(0)
               : "cc", "memory");
#else
  for (cf = false, i = 0; i < n; ++i) {
    ADC(C[i], A[i], B[i], cf, cf);
  }
#endif
  return cf;
}

/**
 * Multiplies huge numbers faster.
 *
 * For 4096 bit numbers it's twice as fast.
 * For 16384 bit numbers it's thrice as fast.
 */
void Karatsuba(uint64_t *C, uint64_t *A, uint64_t *B, size_t n, uint64_t *K) {
  size_t i;
  uint64_t c, t;
  if (n == 8) {
#ifdef __x86_64__
    if (X86_HAVE(BMI2) && X86_HAVE(ADX)) {
      Mul8x8Adx(C, A, B);
      return;
    }
#endif
    Mul(C, A, 8, B, 8);
    return;
  }
  switch (Cmp(A, A + n / 2, n / 2) * 3 + Cmp(B + n / 2, B, n / 2)) {
    case -1 * 3 + +0:
    case +0 * 3 + -1:
    case +0 * 3 + +0:
    case +0 * 3 + +1:
    case +1 * 3 + +0:
      Karatsuba(C, A, B, n / 2, K + n * 2);
      Karatsuba(C + n, A + n / 2, B + n / 2, n / 2, K + n * 2);
      c = Add(K, C, C + n, n);
      c += Add(C + n / 2, C + n / 2, K, n);
      break;
    case -1 * 3 + -1:
      Sub(K, A + n / 2, A, n / 2);
      Sub(K + n / 2, B, B + n / 2, n / 2);
      Karatsuba(K + n, K, K + n / 2, n / 2, K + n * 2);
      Karatsuba(C, A, B, n / 2, K + n * 2);
      Karatsuba(C + n, A + n / 2, B + n / 2, n / 2, K + n * 2);
      c = Add(K, C, C + n, n);
      c += Add(K + n, K, K + n, n);
      c += Add(C + n / 2, C + n / 2, K + n, n);
      break;
    case -1 * 3 + +1:
      Sub(K, A + n / 2, A, n / 2);
      Sub(K + n / 2, B + n / 2, B, n / 2);
      Karatsuba(K + n, K, K + n / 2, n / 2, K + n * 2);
      Karatsuba(C, A, B, n / 2, K + n * 2);
      Karatsuba(C + n, A + n / 2, B + n / 2, n / 2, K + n * 2);
      c = Add(K, C, C + n, n);
      c -= Sub(K + n, K, K + n, n);
      c += Add(C + n / 2, C + n / 2, K + n, n);
      break;
    case +1 * 3 + -1:
      Sub(K, A, A + n / 2, n / 2);
      Sub(K + n / 2, B, B + n / 2, n / 2);
      Karatsuba(K + n, K, K + n / 2, n / 2, K + n * 2);
      Karatsuba(C, A, B, n / 2, K + n * 2);
      Karatsuba(C + n, A + n / 2, B + n / 2, n / 2, K + n * 2);
      c = Add(K, C, C + n, n);
      c -= Sub(K + n, K, K + n, n);
      c += Add(C + n / 2, C + n / 2, K + n, n);
      break;
    case +1 * 3 + +1:
      Sub(K, A, A + n / 2, n / 2);
      Sub(K + n / 2, B + n / 2, B, n / 2);
      Karatsuba(K + n, K, K + n / 2, n / 2, K + n * 2);
      Karatsuba(C, A, B, n / 2, K + n * 2);
      Karatsuba(C + n, A + n / 2, B + n / 2, n / 2, K + n * 2);
      c = Add(K, C, C + n, n);
      c += Add(K + n, K, K + n, n);
      c += Add(C + n / 2, C + n / 2, K + n, n);
      break;
    default:
      __builtin_unreachable();
  }
  for (i = n / 2 + n; c && i < n + n; i++) {
    t = C[i];
    c = (C[i] = t + c) < t;
  }
}
