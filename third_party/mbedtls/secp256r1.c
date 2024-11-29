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
#include <libc/log/check.h>
#include "third_party/mbedtls/bignum.h"
#include "third_party/mbedtls/math.h"
#include "third_party/mbedtls/platform.h"

#define Q(i) p[i >> 1]
#define L(w) (w & 0x00000000ffffffff)
#define H(w) (w & 0xffffffff00000000)

/**
 * Fastest quasi-reduction modulo ℘256.
 *
 *     p  = 2²⁵⁶ - 2²²⁴ + 2¹⁹² + 2⁹⁶ - 1
 *     B  = T + 2×S₁ + 2×S₂ + S₃ + S₄ – D₁ – D₂ – D₃ – D₄ mod p
 *     T  = ( A₇  ‖ A₆  ‖ A₅  ‖ A₄  ‖ A₃  ‖ A₂  ‖ A₁  ‖ A₀  )
 *     S₁ = ( A₁₅ ‖ A₁₄ ‖ A₁₃ ‖ A₁₂ ‖ A₁₁ ‖ 0   ‖ 0   ‖ 0   )
 *     S₂ = ( 0   ‖ A₁₅ ‖ A₁₄‖ A₁₃ ‖ A₁₂ ‖ 0   ‖ 0   ‖ 0   )
 *     S₃ = ( A₁₅ ‖ A₁₄ ‖ 0   ‖ 0   ‖ 0   ‖ A₁₀ ‖ A₉  ‖ A₈  )
 *     S₄ = ( A₈  ‖ A₁₃ ‖ A₁₅ ‖ A₁₄ ‖ A₁₃ ‖ A₁₁ ‖ A₁₀ ‖ A₉  )
 *     D₁ = ( A₁₀ ‖ A₈  ‖ 0   ‖ 0   ‖ 0   ‖ A₁₃ ‖ A₁₂ ‖ A₁₁ )
 *     D₂ = ( A₁₁ ‖ A₉  ‖ 0   ‖ 0   ‖ A₁₅ ‖ A₁₄ ‖ A₁₃ ‖ A₁₂ )
 *     D₃ = ( A₁₂ ‖ 0   ‖ A₁₀ ‖ A₉  ‖ A₈  ‖ A₁₅ ‖ A₁₄ ‖ A₁₃ )
 *     D₄ = ( A₁₃ ‖ 0   ‖ A₁₁ ‖ A₁₀ ‖ A₉  ‖ 0   ‖ A₁₅ ‖ A₁₄ )
 *
 * @see FIPS 186-3 §D.2.3
 */
void secp256r1(uint64_t p[8]) {
  char o;
  signed char E;
  uint64_t A, B, C, D, b, c, d;
  A = Q(0);
  B = Q(2);
  C = Q(4);
  D = Q(6);
  E = 0;
#if !defined(__x86_64__) || defined(__STRICT_ANSI__)
  (void)b;
  (void)c;
  (void)d;
  ADC(B, B, H(Q(10)) << 1, 0, o);
  ADC(C, C, Q(12) << 1 | Q(10) >> 63, o, o);
  ADC(D, D, Q(14) << 1 | Q(12) >> 63, o, o);
  E += o + (Q(14) >> 63);
  ADC(B, B, Q(12) << 33, 0, o);
  ADC(C, C, Q(14) << 33 | Q(12) >> 31, o, o);
  ADC(D, D, Q(14) >> 31, o, o);
  E += o;
  ADC(A, A, Q(8), 0, o);
  ADC(B, B, L(Q(10)), o, o);
  ADC(C, C, 0, o, o);
  ADC(D, D, Q(14), o, o);
  E += o;
  ADC(A, A, Q(10) << 32 | Q(8) >> 32, 0, o);
  ADC(B, B, H(Q(12)) | Q(10) >> 32, o, o);
  ADC(C, C, Q(14), o, o);
  ADC(D, D, Q(8) << 32 | Q(12) >> 32, o, o);
  E += o;
  SBB(A, A, Q(12) << 32 | Q(10) >> 32, 0, o);
  SBB(B, B, Q(12) >> 32, o, o);
  SBB(C, C, 0, o, o);
  SBB(D, D, Q(10) << 32 | L(Q(8)), o, o);
  E -= o;
  SBB(A, A, Q(12), 0, o);
  SBB(B, B, Q(14), o, o);
  SBB(C, C, 0, o, o);
  SBB(D, D, H(Q(10)) | Q(8) >> 32, o, o);
  E -= o;
  SBB(A, A, Q(14) << 32 | Q(12) >> 32, 0, o);
  SBB(B, B, Q(8) << 32 | Q(14) >> 32, o, o);
  SBB(C, C, Q(10) << 32 | Q(8) >> 32, o, o);
  SBB(D, D, Q(12) << 32, o, o);
  E -= o;
  SBB(A, A, Q(14), 0, o);
  SBB(B, B, H(Q(8)), o, o);
  SBB(C, C, Q(10), o, o);
  SBB(D, D, H(Q(12)), o, o);
  E -= o;
#else
  (void)o;
  asm volatile(/* x += 2 × ( A₁₅ ‖ A₁₄ ‖ A₁₃ ‖ A₁₂ ‖ A₁₁ ‖ 0 ‖ 0 ‖ 0 ) */
               "mov\t11*4(%8),%k5\n\t"
               "mov\t12*4(%8),%6\n\t"
               "mov\t14*4(%8),%7\n\t"
               "shl\t$33,%5\n\t"
               "rcl\t%6\n\t"
               "rcl\t%7\n\t"
               "adc\t$0,%b4\n\t"
               "add\t%5,%1\n\t"
               "adc\t%6,%2\n\t"
               "adc\t%7,%3\n\t"
               "adc\t$0,%b4\n\t"
               /* x += 2 × ( 0 ‖ A₁₅ ‖ A₁₄‖ A₁₃ ‖ A₁₂ ‖ 0 ‖ 0 ‖ 0 ) */
               "mov\t12*4(%8),%k5\n\t"
               "mov\t13*4(%8),%6\n\t"
               "mov\t15*4(%8),%k7\n\t"
               "shl\t$33,%5\n\t"
               "rcl\t%6\n\t"
               "rcl\t%7\n\t"
               "add\t%5,%1\n\t"
               "adc\t%6,%2\n\t"
               "adc\t%7,%3\n\t"
               /* x += ( A₁₅ ‖ A₁₄ ‖ 0 ‖ 0 ‖ 0 ‖ A₁₀ ‖ A₉ ‖ A₈ ) */
               "mov\t10*4(%8),%k5\n\t"
               "add\t8*4(%8),%0\n\t"
               "adc\t%5,%1\n\t"
               "adc\t$0,%2\n\t"
               "adc\t14*4(%8),%3\n\t"
               "adc\t$0,%b4\n\t"
               /* x += ( A₈ ‖ A₁₃ ‖ A₁₅ ‖ A₁₄ ‖ A₁₃ ‖ A₁₁ ‖ A₁₀ ‖ A₉ ) */
               "mov\t8*4(%8),%k7\n\t"  /* A₈  ‖ A₁₃ */
               "mov\t13*4(%8),%k5\n\t" /* ...       */
               "shl\t$32,%7\n\t"       /* ...       */
               "or\t%5,%7\n\t"         /* ...       */
               "shl\t$32,%5\n\t"       /* A₁₃ ‖ A₁₁ */
               "mov\t11*4(%8),%k6\n\t" /* ...       */
               "or\t%6,%5\n\t"         /* ...       */
               "add\t9*4(%8),%0\n\t"   /* A₁₀ ‖ A₉  */
               "adc\t%5,%1\n\t"        /* ...       */
               "adc\t14*4(%8),%2\n\t"  /* A₁₅ ‖ A₁₄ */
               "adc\t%7,%3\n\t"
               "adc\t$0,%b4\n\t"
               /* x -= ( A₁₀ ‖ A₈ ‖ 0 ‖ 0 ‖ 0 ‖ A₁₃ ‖ A₁₂ ‖ A₁₁ ) */
               "mov\t10*4(%8),%k6\n\t"
               "mov\t8*4(%8),%k7\n\t"
               "shl\t$32,%6\n\t"
               "or\t%6,%7\n\t"
               "mov\t13*4(%8),%k5\n\t"
               "sub\t11*4(%8),%0\n\t"
               "sbb\t%5,%1\n\t"
               "sbb\t$0,%2\n\t"
               "sbb\t%7,%3\n\t"
               "sbb\t$0,%b4\n\t"
               /* x -= ( A₁₁ ‖ A₉ ‖ 0 ‖ 0 ‖ A₁₅ ‖ A₁₄ ‖ A₁₃ ‖ A₁₂ ) */
               "mov\t11*4(%8),%k6\n\t"
               "mov\t9*4(%8),%k7\n\t"
               "shl\t$32,%6\n\t"
               "or\t%6,%7\n\t"
               "sub\t12*4(%8),%0\n\t"
               "sbb\t14*4(%8),%1\n\t"
               "sbb\t$0,%2\n\t"
               "sbb\t%7,%3\n\t"
               "sbb\t$0,%b4\n\t"
               /* x -= ( A₁₂ ‖ 0 ‖ A₁₀ ‖ A₉ ‖ A₈ ‖ A₁₅ ‖ A₁₄ ‖ A₁₃ ) */
               "mov\t12*4(%8),%k7\n\t"
               "shl\t$32,%7\n\t"
               "mov\t15*4(%8),%k6\n\t"
               "mov\t8*4(%8),%k5\n\t"
               "shl\t$32,%5\n\t"
               "or\t%5,%6\n\t"
               "sub\t13*4(%8),%0\n\t"
               "sbb\t%6,%1\n\t"
               "sbb\t9*4(%8),%2\n\t"
               "sbb\t%7,%3\n\t"
               "sbb\t$0,%b4\n\t"
               /* x -= ( A₁₃ ‖ 0 ‖ A₁₁ ‖ A₁₀ ‖ A₉ ‖ 0 ‖ A₁₅ ‖ A₁₄ ) */
               "mov\t9*4(%8),%k6\n\t"
               "shl\t$32,%6\n\t"
               "mov\t13*4(%8),%k5\n\t"
               "shl\t$32,%5\n\t"
               "sub\t14*4(%8),%0\n\t"
               "sbb\t%6,%1\n\t"
               "sbb\t10*4(%8),%2\n\t"
               "sbb\t%5,%3\n\t"
               "sbb\t$0,%b4\n\t"
               : "+r"(A), "+r"(B), "+r"(C), "+r"(D), "+&q"(E), "=&r"(b),
                 "=&r"(c), "=&r"(d)
               : "r"(p)
               : "memory");
#endif
  p[0] = A;
  p[1] = B;
  p[2] = C;
  p[3] = D;
  p[4] = E;
  p[5] = 0;
  p[6] = 0;
  p[7] = 0;
}

int ecp_mod_p256(mbedtls_mpi *N) {
  int r;
  char o;
  if (N->n < 8 && (r = mbedtls_mpi_grow(N, 8))) return r;
  secp256r1(N->p);
  if ((int64_t)N->p[4] < 0) {
    N->s = -1;
    SBB(N->p[0], 0, N->p[0], 0, o);
    SBB(N->p[1], 0, N->p[1], o, o);
    SBB(N->p[2], 0, N->p[2], o, o);
    SBB(N->p[3], 0, N->p[3], o, o);
    N->p[4] = 0 - (N->p[4] + o);
  } else {
    N->s = 1;
  }
  return 0;
}
