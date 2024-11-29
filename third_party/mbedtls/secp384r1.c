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
#include "third_party/mbedtls/ecp_internal.h"
#include "third_party/mbedtls/math.h"

#define Q(i) p[i >> 1]

/**
 * Fastest quasi-reduction modulo ℘384.
 *
 *     p  = 2³⁸⁴ – 2¹²⁸ – 2⁶ + 2³² – 1
 *     B  = T + 2×S₁ + S₂ + S₃ + S₄ + S₅ + S₆ – D₁ – D₂ – D₃ mod p
 *     T  = (A₁₁‖A₁₀‖A₉ ‖A₈ ‖A₇ ‖A₆ ‖A₅ ‖A₄ ‖A₃ ‖A₂ ‖A₁ ‖A₀ )
 *     S₁ = (0  ‖0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₂‖A₂₁‖0  ‖0  ‖0  ‖0  )
 *     S₂ = (A₂₃‖A₂₂‖A₂₁‖A₂₀‖A₁₉‖A₁₈‖A₁₇‖A₁₆‖A₁₅‖A₁₄‖A₁₃‖A₁₂)
 *     S₃ = (A₂₀‖A₁₉‖A₁₈‖A₁₇‖A₁₆‖A₁₅‖A₁₄‖A₁₃‖A₁₂‖A₂₃‖A₂₂‖A₂₁)
 *     S₄ = (A₁₉‖A₁₈‖A₁₇‖A₁₆‖A₁₅‖A₁₄‖A₁₃‖A₁₂‖A₂₀‖0  ‖A₂₃‖0  )
 *     S₅ = (0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₂‖A₂₁‖A₂₀‖0  ‖0  ‖0  ‖0  )
 *     S₆ = (0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₂‖A₂₁‖0  ‖0  ‖A₂₀)
 *     D₁ = (A₂₂‖A₂₁‖A₂₀‖A₁₉‖A₁₈‖A₁₇‖A₁₆‖A₁₅‖A₁₄‖A₁₃‖A₁₂‖A₂₃)
 *     D₂ = (0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₂‖A₂₁‖A₂₀‖0  )
 *     D₃ = (0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₃‖0  ‖0  ‖0  )
 *
 * @see FIPS 186-3 §D.2.4
 */
void secp384r1(uint64_t p[12]) {
  uint64_t A, B, C, D, E, F, G, a, b, o;
  A = Q(0);
  B = Q(2);
  C = Q(4);
  D = Q(6);
  E = Q(8);
  F = Q(10);
  G = 0;
#if !defined(__x86_64__) || defined(__STRICT_ANSI__)
  a = Q(22) << 32 | Q(21) >> 32;
  b = Q(23) >> 32;
  ADC(C, C, a << 1, 0, o);
  ADC(D, D, b << 1 | a >> 63, o, o);
  ADC(E, E, b >> 63, o, o);
  ADC(F, F, 0, o, o);
  G += o;
  ADC(A, A, Q(12), 0, o);
  ADC(B, B, Q(14), o, o);
  ADC(C, C, Q(16), o, o);
  ADC(D, D, Q(18), o, o);
  ADC(E, E, Q(20), o, o);
  ADC(F, F, Q(22), o, o);
  G += o;
  ADC(A, A, Q(22) << 32 | Q(21) >> 32, 0, o);
  ADC(B, B, Q(12) << 32 | Q(23) >> 32, o, o);
  ADC(C, C, Q(14) << 32 | Q(13) >> 32, o, o);
  ADC(D, D, Q(16) << 32 | Q(15) >> 32, o, o);
  ADC(E, E, Q(18) << 32 | Q(17) >> 32, o, o);
  ADC(F, F, Q(20) << 32 | Q(19) >> 32, o, o);
  G += o;
  ADC(A, A, Q(23) >> 32 << 32, 0, o);
  ADC(B, B, Q(20) << 32, o, o);
  ADC(C, C, Q(12), o, o);
  ADC(D, D, Q(14), o, o);
  ADC(E, E, Q(16), o, o);
  ADC(F, F, Q(18), o, o);
  G += o;
  ADC(C, C, Q(20), 0, o);
  ADC(D, D, Q(22), o, o);
  ADC(E, E, 0, o, o);
  ADC(F, F, 0, o, o);
  G += o;
  ADC(A, A, Q(20) & 0xffffffff, 0, o);
  ADC(B, B, Q(21) >> 32 << 32, o, o);
  ADC(C, C, Q(22), o, o);
  ADC(D, D, 0, o, o);
  ADC(E, E, 0, o, o);
  ADC(F, F, 0, o, o);
  G += o;
  SBB(A, A, Q(12) << 32 | Q(23) >> 32, 0, o);
  SBB(B, B, Q(14) << 32 | Q(13) >> 32, o, o);
  SBB(C, C, Q(16) << 32 | Q(15) >> 32, o, o);
  SBB(D, D, Q(18) << 32 | Q(17) >> 32, o, o);
  SBB(E, E, Q(20) << 32 | Q(19) >> 32, o, o);
  SBB(F, F, Q(22) << 32 | Q(21) >> 32, o, o);
  G -= o;
  SBB(A, A, Q(20) << 32, 0, o);
  SBB(B, B, Q(22) << 32 | Q(21) >> 32, o, o);
  SBB(C, C, Q(23) >> 32, o, o);
  SBB(D, D, 0, o, o);
  SBB(E, E, 0, o, o);
  SBB(F, F, 0, o, o);
  G -= o;
  SBB(B, B, Q(23) >> 32 << 32, 0, o);
  SBB(C, C, Q(23) >> 32, o, o);
  SBB(D, D, 0, o, o);
  SBB(E, E, 0, o, o);
  SBB(F, F, 0, o, o);
  G -= o;
#else
  (void)o;
  asm volatile(/* S₁ = (0  ‖0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₂‖A₂₁‖0  ‖0  ‖0  ‖0  ) */
               "mov\t21*4(%9),%7\n\t"
               "mov\t23*4(%9),%k8\n\t"
               "mov\t%7,%%r12\n\t"
               "shr\t$63,%%r12\n\t"
               "shl\t%7\n\t"
               "shl\t%8\n\t"
               "or\t%%r12,%8\n\t"
               "mov\t13*4(%9),%%r12\n\t"
               "add\t%7,%2\n\t"
               "mov\t23*4(%9),%k7\n\t"
               "adc\t%8,%3\n\t"
               "mov\t15*4(%9),%%r13\n\t"
               "adc\t$0,%4\n\t"
               "mov\t12*4(%9),%k8\n\t"
               "adc\t$0,%5\n\t"
               "mov\t17*4(%9),%%r14\n\t"
               "adc\t$0,%6\n\t"
               "mov\t19*4(%9),%%r15\n\t"
               /* D₁ = (A₂₂‖A₂₁‖A₂₀‖A₁₉‖A₁₈‖A₁₇‖A₁₆‖A₁₅‖A₁₄‖A₁₃‖A₁₂‖A₂₃) */
               "shl\t$32,%8\n\t"
               "or\t%8,%7\n\t"
               "mov\t23*4(%9),%k8\n\t"
               "sub\t%7,%0\n\t"
               "mov\t21*4(%9),%7\n\t"
               "sbb\t%%r12,%1\n\t"
               "sbb\t%%r13,%2\n\t"
               "sbb\t%%r14,%3\n\t"
               "sbb\t%%r15,%4\n\t"
               "sbb\t%7,%5\n\t"
               "mov\t12*4(%9),%k7\n\t"
               "sbb\t$0,%6\n\t"
               /* S₃ = (A₂₀‖A₁₉‖A₁₈‖A₁₇‖A₁₆‖A₁₅‖A₁₄‖A₁₃‖A₁₂‖A₂₃‖A₂₂‖A₂₁) */
               "shl\t$32,%7\n\t"
               "or\t%7,%8\n\t"
               "add\t21*4(%9),%0\n\t"
               "mov\t23*4(%9),%k7\n\t"
               "adc\t%8,%1\n\t"
               "mov\t20*4(%9),%k8\n\t"
               "adc\t%%r12,%2\n\t"
               "mov\t12*4(%9),%%r12\n\t"
               "adc\t%%r13,%3\n\t"
               "mov\t14*4(%9),%%r13\n\t"
               "adc\t%%r14,%4\n\t"
               "mov\t16*4(%9),%%r14\n\t"
               "adc\t%%r15,%5\n\t"
               "mov\t18*4(%9),%%r15\n\t"
               "adc\t$0,%6\n\t"
               /* S₄ = (A₁₉‖A₁₈‖A₁₇‖A₁₆‖A₁₅‖A₁₄‖A₁₃‖A₁₂‖A₂₀‖0  ‖A₂₃‖0  ) */
               "shl\t$32,%7\n\t"
               "shl\t$32,%8\n\t"
               "add\t%7,%0\n\t"
               "adc\t%8,%1\n\t"
               "adc\t%%r12,%2\n\t"
               "adc\t%%r13,%3\n\t"
               "adc\t%%r14,%4\n\t"
               "adc\t%%r15,%5\n\t"
               "adc\t$0,%6\n\t"
               /* S₂ = (A₂₃‖A₂₂‖A₂₁‖A₂₀‖A₁₉‖A₁₈‖A₁₇‖A₁₆‖A₁₅‖A₁₄‖A₁₃‖A₁₂) */
               "add\t%%r12,%0\n\t"
               "mov\t20*4(%9),%%r12\n\t"
               "adc\t%%r13,%1\n\t"
               "mov\t22*4(%9),%%r13\n\t"
               "adc\t%%r14,%2\n\t"
               "adc\t%%r15,%3\n\t"
               "adc\t%%r12,%4\n\t"
               "adc\t%%r13,%5\n\t"
               "adc\t$0,%6\n\t"
               /* S₅ = (0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₂‖A₂₁‖A₂₀‖0  ‖0  ‖0  ‖0  ) */
               "add\t%%r12,%2\n\t"
               "adc\t%%r13,%3\n\t"
               "adc\t$0,%4\n\t"
               "adc\t$0,%5\n\t"
               "adc\t$0,%6\n\t"
               /* S₆ = (0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₂‖A₂₁‖0  ‖0  ‖A₂₀) */
               "mov\t%%r12d,%k7\n\t"
               "mov\t%%r12,%8\n\t"
               "shr\t$32,%8\n\t"
               "shl\t$32,%8\n\t"
               "add\t%7,%0\n\t"
               "adc\t%8,%1\n\t"
               "adc\t%%r13,%2\n\t"
               "adc\t$0,%3\n\t"
               "adc\t$0,%4\n\t"
               "adc\t$0,%5\n\t"
               "adc\t$0,%6\n\t"
               /* D₂ = (0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₂‖A₂₁‖A₂₀‖0  ) */
               "mov\t%%r12d,%k7\n\t"
               "mov\t21*4(%9),%%r12\n\t"
               "mov\t%%r13,%8\n\t"
               "shr\t$32,%8\n\t"
               "shl\t$32,%7\n\t"
               "sub\t%7,%0\n\t"
               "sbb\t%%r12,%1\n\t"
               "sbb\t%8,%2\n\t"
               "sbb\t$0,%3\n\t"
               "sbb\t$0,%4\n\t"
               "sbb\t$0,%5\n\t"
               "sbb\t$0,%6\n\t"
               /* D₃ = (0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖0  ‖A₂₃‖A₂₃‖0  ‖0  ‖0  ) */
               "mov\t%%r13,%7\n\t"
               "shr\t$32,%7\n\t"
               "mov\t%k7,%k8\n\t"
               "shl\t$32,%7\n\t"
               "sub\t%7,%1\n\t"
               "sbb\t%8,%2\n\t"
               "sbb\t$0,%3\n\t"
               "sbb\t$0,%4\n\t"
               "sbb\t$0,%5\n\t"
               "sbb\t$0,%6"
               : "+r"(A), "+r"(B), "+r"(C), "+r"(D), "+r"(E), "+r"(F), "+q"(G),
                 "=&r"(a), "=&r"(b)
               : "r"(p)
               : "memory", "r12", "r13", "r14", "r15");
#endif
  p[0] = A;
  p[1] = B;
  p[2] = C;
  p[3] = D;
  p[4] = E;
  p[5] = F;
  p[6] = G;
  G = __conceal("r", 0L);
  p[7] = G;
  p[8] = G;
  p[9] = G;
  p[10] = G;
  p[11] = G;
}

int ecp_mod_p384(mbedtls_mpi *N) {
  int r;
  char o;
  if (N->n < 12 && (r = mbedtls_mpi_grow(N, 12))) return r;
  secp384r1(N->p);
  if ((int64_t)N->p[6] < 0) {
    N->s = -1;
    SBB(N->p[0], 0, N->p[0], 0, o);
    SBB(N->p[1], 0, N->p[1], o, o);
    SBB(N->p[2], 0, N->p[2], o, o);
    SBB(N->p[3], 0, N->p[3], o, o);
    SBB(N->p[4], 0, N->p[4], o, o);
    SBB(N->p[5], 0, N->p[5], o, o);
    N->p[6] = 0 - (N->p[6] + o);
  } else {
    N->s = 1;
  }
  return 0;
}

/*
Instructions:      115
Total Cycles:      46
Total uOps:        116
uOps Per Cycle:    2.52
IPC:               2.50
Block RThroughput: 31.0

SIMULATION          0123456789          0123456789
Index     0123456789          0123456789          012345
[0,0]     DR   .    .    .    .    .    .    .    .    .   xorl	%r10d, %r10d
[0,1]     DeeeeeER  .    .    .    .    .    .    .    .   movq	(%rdi), %r9
[0,2]     DeeeeeER  .    .    .    .    .    .    .    .   movq	8(%rdi), %r8
[0,3]     D=eeeeeER .    .    .    .    .    .    .    .   movq	16(%rdi), %rsi
[0,4]     D=eeeeeER .    .    .    .    .    .    .    .   movq	24(%rdi), %rcx
[0,5]     D==eeeeeER.    .    .    .    .    .    .    .   movq	32(%rdi), %rdx
[0,6]     .D==eeeeeER    .    .    .    .    .    .    .   movq	40(%rdi), %rax
[0,7]     .D=eeeeeE-R    .    .    .    .    .    .    .   movq	84(%rdi), %r11
[0,8]     .D==eeeeeER    .    .    .    .    .    .    .   movl	92(%rdi), %ebx
[0,9]     .D======eER    .    .    .    .    .    .    .   movq	%r11, %r12
[0,10]    .D=======eER   .    .    .    .    .    .    .   shrq	$63, %r12
[0,11]    .D======eE-R   .    .    .    .    .    .    .   shlq	%r11
[0,12]    . D======eER   .    .    .    .    .    .    .   shlq	%rbx
[0,13]    . D=======eER  .    .    .    .    .    .    .   orq	%r12, %rbx
[0,14]    . D==eeeeeE-R  .    .    .    .    .    .    .   movq	52(%rdi), %r12
[0,15]    . D======eE-R  .    .    .    .    .    .    .   addq	%r11, %rsi
[0,16]    . D==eeeeeE-R  .    .    .    .    .    .    .   movl	92(%rdi), %r11d
[0,17]    . D========eER .    .    .    .    .    .    .   adcq	%rbx, %rcx
[0,18]    .  D==eeeeeE-R .    .    .    .    .    .    .   movq	60(%rdi), %r13
[0,19]    .  D========eER.    .    .    .    .    .    .   adcq	$0, %rdx
[0,20]    .  D==eeeeeE--R.    .    .    .    .    .    .   movl	48(%rdi), %ebx
[0,21]    .  D=========eER    .    .    .    .    .    .   adcq	$0, %rax
[0,22]    .  D===eeeeeE--R    .    .    .    .    .    .   movq	68(%rdi), %r14
[0,23]    .  D==========eER   .    .    .    .    .    .   adcq	$0, %r10
[0,24]    .   D==eeeeeE---R   .    .    .    .    .    .   movq	76(%rdi), %r15
[0,25]    .   D======eE---R   .    .    .    .    .    .   shlq	$32, %rbx
[0,26]    .   D=======eE--R   .    .    .    .    .    .   orq	%rbx, %r11
[0,27]    .   D===eeeeeE--R   .    .    .    .    .    .   movl	92(%rdi), %ebx
[0,28]    .   D========eE-R   .    .    .    .    .    .   subq	%r11, %r9
[0,29]    .   D===eeeeeE--R   .    .    .    .    .    .   movq	84(%rdi), %r11
[0,30]    .    D========eER   .    .    .    .    .    .   sbbq	%r12, %r8
[0,31]    .    D=========eER  .    .    .    .    .    .   sbbq	%r13, %rsi
[0,32]    .    D==========eER .    .    .    .    .    .   sbbq	%r14, %rcx
[0,33]    .    D===========eER.    .    .    .    .    .   sbbq	%r15, %rdx
[0,34]    .    D============eER    .    .    .    .    .   sbbq	%r11, %rax
[0,35]    .    D===eeeeeE-----R    .    .    .    .    .   movl	48(%rdi), %r11d
[0,36]    .    .D============eER   .    .    .    .    .   sbbq	$0, %r10
[0,37]    .    .D========eE----R   .    .    .    .    .   shlq	$32, %r11
[0,38]    .    .D=========eE---R   .    .    .    .    .   orq	%r11, %rbx
[0,39]    .    .D==eeeeeE------R   .    .    .    .    .   movl	92(%rdi), %r11d
[0,40]    .    .D======eeeeeeE-R   .    .    .    .    .   addq	84(%rdi), %r9
[0,41]    .    . D===========eER   .    .    .    .    .   adcq	%rbx, %r8
[0,42]    .    . D==eeeeeE-----R   .    .    .    .    .   movl	80(%rdi), %ebx
[0,43]    .    . D============eER  .    .    .    .    .   adcq	%r12, %rsi
[0,44]    .    . D==eeeeeE------R  .    .    .    .    .   movq	48(%rdi), %r12
[0,45]    .    . D=============eER .    .    .    .    .   adcq	%r13, %rcx
[0,46]    .    . D===eeeeeE------R .    .    .    .    .   movq	56(%rdi), %r13
[0,47]    .    .  D=============eER.    .    .    .    .   adcq	%r14, %rdx
[0,48]    .    .  D==eeeeeE-------R.    .    .    .    .   movq	64(%rdi), %r14
[0,49]    .    .  D==============eER    .    .    .    .   adcq	%r15, %rax
[0,50]    .    .  D===eeeeeE-------R    .    .    .    .   movq	72(%rdi), %r15
[0,51]    .    .  D===============eER   .    .    .    .   adcq	$0, %r10
[0,52]    .    .  D=======eE--------R   .    .    .    .   shlq	$32, %r11
[0,53]    .    .   D=======eE-------R   .    .    .    .   shlq	$32, %rbx
[0,54]    .    .   D=========eE-----R   .    .    .    .   addq	%r11, %r9
[0,55]    .    .   D==========eE----R   .    .    .    .   adcq	%rbx, %r8
[0,56]    .    .   D===========eE---R   .    .    .    .   adcq	%r12, %rsi
[0,57]    .    .   D============eE--R   .    .    .    .   adcq	%r13, %rcx
[0,58]    .    .   D=============eE-R   .    .    .    .   adcq	%r14, %rdx
[0,59]    .    .    D=============eER   .    .    .    .   adcq	%r15, %rax
[0,60]    .    .    D==============eER  .    .    .    .   adcq	$0, %r10
[0,61]    .    .    D=========eE-----R  .    .    .    .   addq	%r12, %r9
[0,62]    .    .    D=eeeeeE---------R  .    .    .    .   movq	80(%rdi), %r12
[0,63]    .    .    D==============eER  .    .    .    .   adcq	%r13, %r8
[0,64]    .    .    D==eeeeeE--------R  .    .    .    .   movq	88(%rdi), %r13
[0,65]    .    .    .D==============eER .    .    .    .   adcq	%r14, %rsi
[0,66]    .    .    .D===============eER.    .    .    .   adcq	%r15, %rcx
[0,67]    .    .    .D================eER    .    .    .   adcq	%r12, %rdx
[0,68]    .    .    .D=================eER   .    .    .   adcq	%r13, %rax
[0,69]    .    .    .D==================eER  .    .    .   adcq	$0, %r10
[0,70]    .    .    .D===============eE---R  .    .    .   addq	%r12, %rsi
[0,71]    .    .    . D===============eE--R  .    .    .   adcq	%r13, %rcx
[0,72]    .    .    . D================eE-R  .    .    .   adcq	$0, %rdx
[0,73]    .    .    . D=================eER  .    .    .   adcq	$0, %rax
[0,74]    .    .    . D==================eER .    .    .   adcq	$0, %r10
[0,75]    .    .    . D====eE--------------R .    .    .   movl	%r12d, %r11d
[0,76]    .    .    . D====eE--------------R .    .    .   movq	%r12, %rbx
[0,77]    .    .    .  D====eE-------------R .    .    .   shrq	$32, %rbx
[0,78]    .    .    .  D============eE-----R .    .    .   shlq	$32, %rbx
[0,79]    .    .    .  D=======eE----------R .    .    .   addq	%r11, %r9
[0,80]    .    .    .  D=============eE----R .    .    .   adcq	%rbx, %r8
[0,81]    .    .    .  D=================eER .    .    .   adcq	%r13, %rsi
[0,82]    .    .    .  D==================eER.    .    .   adcq	$0, %rcx
[0,83]    .    .    .   D==================eER    .    .   adcq	$0, %rdx
[0,84]    .    .    .   D===================eER   .    .   adcq	$0, %rax
[0,85]    .    .    .   D====================eER  .    .   adcq	$0, %r10
[0,86]    .    .    .   D===eE-----------------R  .    .   movl	%r12d, %r11d
[0,87]    .    .    .   DeeeeeE----------------R  .    .   movq	84(%rdi), %r12
[0,88]    .    .    .   D===eE-----------------R  .    .   movq	%r13, %rbx
[0,89]    .    .    .    D================eE---R  .    .   shrq	$32, %rbx
[0,90]    .    .    .    D=================eE--R  .    .   shlq	$32, %r11
[0,91]    .    .    .    D==================eE-R  .    .   subq	%r11, %r9
[0,92]    .    .    .    D===================eER  .    .   sbbq	%r12, %r8
[0,93]    .    .    .    D====================eER .    .   sbbq	%rbx, %rsi
[0,94]    .    .    .    D=====================eER.    .   sbbq	$0, %rcx
[0,95]    .    .    .    .D=====================eER    .   sbbq	$0, %rdx
[0,96]    .    .    .    .D======================eER   .   sbbq	$0, %rax
[0,97]    .    .    .    .D=======================eER  .   sbbq	$0, %r10
[0,98]    .    .    .    .D==eE---------------------R  .   movq	%r13, %r11
[0,99]    .    .    .    .D=================eE------R  .   shrq	$32, %r11
[0,100]   .    .    .    .D==================eE-----R  .   movl	%r11d, %ebx
[0,101]   .    .    .    . D==================eE----R  .   shlq	$32, %r11
[0,102]   .    .    .    . D===================eE---R  .   subq	%r11, %r8
[0,103]   .    .    .    . D====================eE--R  .   sbbq	%rbx, %rsi
[0,104]   .    .    .    . D=====================eE-R  .   sbbq	$0, %rcx
[0,105]   .    .    .    . D======================eER  .   sbbq	$0, %rdx
[0,106]   .    .    .    . D=======================eER .   sbbq	$0, %rax
[0,107]   .    .    .    .  D=======================eER.   sbbq	$0, %r10
[0,108]   .    .    .    .  D================eE-------R.   movq	%r9, (%rdi)
[0,109]   .    .    .    .  D===================eE----R.   movq	%r8, 8(%rdi)
[0,110]   .    .    .    .  D====================eE---R.   movq	%rsi, 16(%rdi)
[0,111]   .    .    .    .  D=====================eE--R.   movq	%rcx, 24(%rdi)
[0,112]   .    .    .    .  D======================eE-R.   movq	%rdx, 32(%rdi)
[0,113]   .    .    .    .   D======================eER.   movq	%rax, 40(%rdi)
[0,114]   .    .    .    .   D=======================eER   movq	%r10, 48(%rdi)
*/
