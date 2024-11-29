/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2016-2018 INRIA and Microsoft Corporation                          │
│                                                                              │
│ Licensed under the Apache License, Version 2.0 (the "License");              │
│ you may not use this file except in compliance with the License.             │
│ You may obtain a copy of the License at                                      │
│                                                                              │
│     http://www.apache.org/licenses/LICENSE-2.0                               │
│                                                                              │
│ Unless required by applicable law or agreed to in writing, software          │
│ distributed under the License is distributed on an "AS IS" BASIS,            │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
╚─────────────────────────────────────────────────────────────────────────────*/
#include <libc/serialize.h>
#include "third_party/mbedtls/endian.h"

__notice(cosmo_everest_notice, "\
Cosmopolitan Everest (Apache 2.0)\n\
Copyright 2024 Justine Alexndra Roberts Tunney\n\
Copyright 2016-2018 INRIA and Microsoft Corporation\n\
Changes: Made C code look nice and not have pointers");

#define DW(x)     (uint128_t)(x)
#define EQ(x, y)  ((((x ^ y) | (~(x ^ y) + 1)) >> 63) - 1)
#define GTE(x, y) (((x ^ ((x ^ y) | ((x - y) ^ y))) >> 63) - 1)

forceinline void HaclBignumCopy(uint64_t o[5], uint64_t p[5]) {
  for (unsigned i = 0; i < 5; ++i) {
    o[i] = p[i];
  }
}

forceinline void HaclBignumFsum(uint64_t o[5], uint64_t p[5]) {
  for (unsigned i = 0; i < 5; ++i) {
    o[i] += p[i];
  }
}

forceinline void HaclBignumTrunc(uint64_t o[5], uint128_t p[5]) {
  for (unsigned i = 0; i < 5; ++i) {
    o[i] = p[i];
  }
}

forceinline void HaclBignumCarry(uint64_t p[5]) {
  for (unsigned i = 0; i < 4; ++i) {
    p[i + 1] += p[i] >> 51;
    p[i] &= 0x7ffffffffffff;
  }
}

forceinline void HaclBignumCarryWide(uint128_t p[5]) {
  for (unsigned i = 0; i < 4; ++i) {
    p[i + 1] += p[i] >> 51;
    p[i] &= 0x7ffffffffffff;
  }
}

static void HaclBignumFmulReduce(uint128_t o[5], uint64_t p[5], uint64_t q[5]) {
  uint64_t t;
  unsigned i, j;
  for (i = 0;; ++i) {
    for (j = 0; j < 5; ++j) {
      o[j] += DW(p[j]) * q[i];
    }
    if (i == 4) break;
    t = p[4] * 19;
    p[4] = p[3];
    p[3] = p[2];
    p[2] = p[1];
    p[1] = p[0];
    p[0] = t;
  }
}

static void HaclBignumFmul(uint64_t o[5], uint64_t p[5], uint64_t q[5]) {
  uint128_t t[5] = {0};
  uint64_t u[5] = {p[0], p[1], p[2], p[3], p[4]};
  HaclBignumFmulReduce(t, u, q);
  HaclBignumCarryWide(t);
  t[0] += DW(19) * (uint64_t)(t[4] >> 51);
  HaclBignumTrunc(o, t);
  o[1] += o[0] >> 51;
  o[4] &= 0x7ffffffffffff;
  o[0] &= 0x7ffffffffffff;
}

static void HaclBignumFsquare(uint128_t t[5], uint64_t p[5]) {
  t[0] = DW(p[0] * 1) * p[0] + DW(p[4] * 38) * p[1] + DW(p[2] * 38) * p[3];
  t[1] = DW(p[0] * 2) * p[1] + DW(p[4] * 38) * p[2] + DW(p[3] * 19) * p[3];
  t[2] = DW(p[0] * 2) * p[2] + DW(p[1] * 01) * p[1] + DW(p[4] * 38) * p[3];
  t[3] = DW(p[0] * 2) * p[3] + DW(p[1] * 02) * p[2] + DW(p[4]) * (p[4] * 19);
  t[4] = DW(p[0] * 2) * p[4] + DW(p[1] * 02) * p[3] + DW(p[2]) * p[2];
}

static void HaclBignumFsqa(uint64_t o[5], uint32_t n) {
  uint128_t t[5];
  for (unsigned i = 0; i < n; ++i) {
    HaclBignumFsquare(t, o);
    HaclBignumCarryWide(t);
    t[0] += DW(19) * (uint64_t)(t[4] >> 51);
    HaclBignumTrunc(o, t);
    o[1] += o[0] >> 51;
    o[4] &= 0x7ffffffffffff;
    o[0] &= 0x7ffffffffffff;
  }
}

static void HaclBignumFsqr(uint64_t o[5], uint64_t p[5], uint32_t n) {
  HaclBignumCopy(o, p);
  HaclBignumFsqa(o, n);
}

static void HaclBignumCrecip(uint64_t o[5], uint64_t z[5]) {
  uint64_t b[4][5];
  HaclBignumFsqr(b[0], z, 1);
  HaclBignumFsqr(b[1], b[0], 2);
  HaclBignumFmul(b[2], b[1], z);
  HaclBignumFmul(b[0], b[2], b[0]);
  HaclBignumFsqr(b[1], b[0], 1);
  HaclBignumFmul(b[2], b[1], b[2]);
  HaclBignumFsqr(b[1], b[2], 5);
  HaclBignumFmul(b[2], b[1], b[2]);
  HaclBignumFsqr(b[1], b[2], 10);
  HaclBignumFmul(b[3], b[1], b[2]);
  HaclBignumFsqr(b[1], b[3], 20);
  HaclBignumFmul(b[1], b[1], b[3]);
  HaclBignumFsqa(b[1], 10);
  HaclBignumFmul(b[2], b[1], b[2]);
  HaclBignumFsqr(b[1], b[2], 50);
  HaclBignumFmul(b[3], b[1], b[2]);
  HaclBignumFsqr(b[1], b[3], 100);
  HaclBignumFmul(b[1], b[1], b[3]);
  HaclBignumFsqa(b[1], 50);
  HaclBignumFmul(b[1], b[1], b[2]);
  HaclBignumFsqa(b[1], 5);
  HaclBignumFmul(o, b[1], b[0]);
}

static void HaclBignumFdif(uint64_t a[5], uint64_t b[5]) {
  a[0] = b[0] + 0x3fffffffffff68 - a[0];
  a[1] = b[1] + 0x3ffffffffffff8 - a[1];
  a[2] = b[2] + 0x3ffffffffffff8 - a[2];
  a[3] = b[3] + 0x3ffffffffffff8 - a[3];
  a[4] = b[4] + 0x3ffffffffffff8 - a[4];
}

static void HaclBignumFscalar(uint64_t o[5], uint64_t p[5], uint64_t s) {
  unsigned i;
  uint128_t t[5];
  for (i = 0; i < 5; ++i) t[i] = DW(p[i]) * s;
  HaclBignumCarryWide(t);
  t[0] += DW(19) * (uint64_t)(t[4] >> 51);
  t[4] &= 0x7ffffffffffff;
  HaclBignumTrunc(o, t);
}

static void HaclEcPointSwap(uint64_t a[2][5], uint64_t b[2][5], uint64_t m) {
  unsigned i, j;
  uint64_t x, y;
  for (i = 0; i < 2; ++i) {
    for (j = 0; j < 5; ++j) {
      x = a[i][j] ^ (-m & (a[i][j] ^ b[i][j]));
      y = b[i][j] ^ (-m & (a[i][j] ^ b[i][j]));
      a[i][j] = x;
      b[i][j] = y;
    }
  }
}

static void HaclEcFormatFexpand(uint64_t o[5], const uint8_t p[32]) {
  o[0] = READ64LE(p + 000) >> 00 & 0x7ffffffffffff;
  o[1] = READ64LE(p + 006) >> 03 & 0x7ffffffffffff;
  o[2] = READ64LE(p + 014) >> 06 & 0x7ffffffffffff;
  o[3] = READ64LE(p + 023) >> 01 & 0x7ffffffffffff;
  o[4] = READ64LE(p + 030) >> 12 & 0x7ffffffffffff;
}

static void HaclEcFormatFcontract(uint8_t o[32], uint64_t p[5]) {
  uint64_t m;
  HaclBignumCarry(p);
  p[0] += 19 * (p[4] >> 51);
  p[4] &= 0x7ffffffffffff;
  HaclBignumCarry(p);
  p[0] += 19 * (p[4] >> 51);
  p[1] += p[0] >> 51;
  p[0] &= 0x7ffffffffffff;
  p[1] &= 0x7ffffffffffff;
  p[4] &= 0x7ffffffffffff;
  m = GTE(p[0], 0x7ffffffffffed);
  m &= EQ(p[1], 0x7ffffffffffff);
  m &= EQ(p[2], 0x7ffffffffffff);
  m &= EQ(p[3], 0x7ffffffffffff);
  m &= EQ(p[4], 0x7ffffffffffff);
  p[0] -= 0x7ffffffffffed & m;
  p[1] -= 0x7ffffffffffff & m;
  p[2] -= 0x7ffffffffffff & m;
  p[3] -= 0x7ffffffffffff & m;
  p[4] -= 0x7ffffffffffff & m;
  Write64le(o + 000, p[1] << 51 | p[0] >> 00);
  Write64le(o + 010, p[2] << 38 | p[1] >> 13);
  Write64le(o + 020, p[3] << 25 | p[2] >> 26);
  Write64le(o + 030, p[4] << 12 | p[3] >> 39);
}

static void HaclEcFormatScalarOfPoint(uint8_t o[32], uint64_t p[2][5]) {
  uint64_t t[2][5];
  HaclBignumCrecip(t[0], p[1]);
  HaclBignumFmul(t[1], p[0], t[0]);
  HaclEcFormatFcontract(o, t[1]);
}

static void HaclEcAddAndDoubleFmonty(uint64_t xz2[2][5], uint64_t xz3[2][5],
                                     uint64_t xz[2][5], uint64_t xzprime[2][5],
                                     uint64_t qx[5]) {
  uint64_t b[7][5];
  HaclBignumCopy(b[0], xz[0]);
  HaclBignumFsum(xz[0], xz[1]);
  HaclBignumFdif(xz[1], b[0]);
  HaclBignumCopy(b[0], xzprime[0]);
  HaclBignumFsum(xzprime[0], xzprime[1]);
  HaclBignumFdif(xzprime[1], b[0]);
  HaclBignumFmul(b[4], xzprime[0], xz[1]);
  HaclBignumFmul(b[5], xz[0], xzprime[1]);
  HaclBignumCopy(b[0], b[4]);
  HaclBignumFsum(b[4], b[5]);
  HaclBignumFdif(b[5], b[0]);
  HaclBignumFsqr(xz3[0], b[4], 1);
  HaclBignumFsqr(b[6], b[5], 1);
  HaclBignumFmul(xz3[1], b[6], qx);
  HaclBignumFsqr(b[2], xz[0], 1);
  HaclBignumFsqr(b[3], xz[1], 1);
  HaclBignumFmul(xz2[0], b[2], b[3]);
  HaclBignumFdif(b[3], b[2]);
  HaclBignumFscalar(b[1], b[3], 121665);
  HaclBignumFsum(b[1], b[2]);
  HaclBignumFmul(xz2[1], b[1], b[3]);
}

/**
 * Computes elliptic curve 25519.
 */
void curve25519(uint8_t mypublic[32], const uint8_t secret[32],
                const uint8_t basepoint[32]) {
  uint32_t i, j;
  uint8_t e[32], s;
  uint64_t q[5], t[4][2][5] = {{{1}}, {{0}, {1}}};
  HaclEcFormatFexpand(q, basepoint);
  for (j = 0; j < 32; ++j) e[j] = secret[j];
  e[0] &= 248;
  e[31] = (e[31] & 127) | 64;
  HaclBignumCopy(t[1][0], q);
  for (i = 32; i--;) {
    for (s = e[i], j = 4; j--;) {
      HaclEcPointSwap(t[0], t[1], s >> 7);
      HaclEcAddAndDoubleFmonty(t[2], t[3], t[0], t[1], q);
      HaclEcPointSwap(t[2], t[3], s >> 7);
      s <<= 1;
      HaclEcPointSwap(t[2], t[3], s >> 7);
      HaclEcAddAndDoubleFmonty(t[0], t[1], t[2], t[3], q);
      HaclEcPointSwap(t[0], t[1], s >> 7);
      s <<= 1;
    }
  }
  HaclEcFormatScalarOfPoint(mypublic, t[0]);
}
