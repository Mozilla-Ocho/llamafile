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
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// this file should not have dependencies, because everything will be
// re-downloaded if the o/tool/sha256sum artifact becomes invalidated

#define PROG "sha256sum"
#define USAGE \
  "\
Usage: " PROG " [-?hbctw] [PATH...]\n\
  -h          help\n\
  -c          check mode\n\
  -b          binary mode\n\
  -t          textual mode\n\
  -w          warning mode\n"

#define ROTR(a, b)   (((a) >> (b)) | ((a) << (32 - (b))))
#define CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)       (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x)       (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x)      (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x)      (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

struct Sha256Ctx {
  uint8_t data[64];
  uint32_t datalen;
  uint64_t bitlen;
  uint32_t state[8];
};

static const uint32_t kSha256Tab[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,  //
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,  //
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,  //
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,  //
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,  //
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,  //
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,  //
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,  //
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,  //
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,  //
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,  //
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,  //
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,  //
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,  //
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,  //
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,  //
};

static bool g_warn;
static char g_mode;
static bool g_check;
static int g_mismatches;

static void Sha256Transform(uint32_t state[8], const uint8_t data[64]) {
  unsigned i;
  uint32_t a, b, c, d, e, f, g, h, t1, t2, m[64];
  for (i = 0; i < 16; ++i, data += 4) {
    m[i] = (uint32_t)data[0] << 24 | data[1] << 16 | data[2] << 8 | data[3];
  }
  for (; i < 64; ++i) {
    m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
  }
  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];
  e = state[4];
  f = state[5];
  g = state[6];
  h = state[7];
  for (i = 0; i < 64; ++i) {
    t1 = h + EP1(e) + CH(e, f, g) + kSha256Tab[i] + m[i];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

static void Sha256Init(struct Sha256Ctx *ctx) {
  ctx->datalen = 0;
  ctx->bitlen = 0;
  ctx->state[0] = 0x6a09e667;
  ctx->state[1] = 0xbb67ae85;
  ctx->state[2] = 0x3c6ef372;
  ctx->state[3] = 0xa54ff53a;
  ctx->state[4] = 0x510e527f;
  ctx->state[5] = 0x9b05688c;
  ctx->state[6] = 0x1f83d9ab;
  ctx->state[7] = 0x5be0cd19;
}

static void Sha256Update(struct Sha256Ctx *ctx, const uint8_t *data,
                         long size) {
  long i;
  for (i = 0; i < size; ++i) {
    ctx->data[ctx->datalen] = data[i];
    ctx->datalen++;
    if (ctx->datalen == 64) {
      Sha256Transform(ctx->state, ctx->data);
      ctx->bitlen += 512;
      ctx->datalen = 0;
    }
  }
}

static void Sha256Final(struct Sha256Ctx *ctx, uint8_t *hash) {
  long i;
  i = ctx->datalen;
  ctx->data[i++] = 0x80;
  if (ctx->datalen < 56) {
    memset(ctx->data + i, 0, 56 - i);
  } else {
    memset(ctx->data + i, 0, 64 - i);
    Sha256Transform(ctx->state, ctx->data);
    memset(ctx->data, 0, 56);
  }
  ctx->bitlen += ctx->datalen * 8;
  ctx->data[63] = ctx->bitlen;
  ctx->data[62] = ctx->bitlen >> 8;
  ctx->data[61] = ctx->bitlen >> 16;
  ctx->data[60] = ctx->bitlen >> 24;
  ctx->data[59] = ctx->bitlen >> 32;
  ctx->data[58] = ctx->bitlen >> 40;
  ctx->data[57] = ctx->bitlen >> 48;
  ctx->data[56] = ctx->bitlen >> 56;
  Sha256Transform(ctx->state, ctx->data);
  for (i = 0; i < 4; ++i) {
    hash[i] = (ctx->state[0] >> (24 - i * 8)) & 0xff;
    hash[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0xff;
    hash[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0xff;
    hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0xff;
    hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0xff;
    hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0xff;
    hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0xff;
    hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0xff;
  }
}

static char *FormatUint32(char *p, uint32_t x) {
  char t;
  size_t i, a, b;
  i = 0;
  do {
    p[i++] = x % 10 + '0';
    x = x / 10;
  } while (x > 0);
  p[i] = '\0';
  if (i) {
    for (a = 0, b = i - 1; a < b; ++a, --b) {
      t = p[a];
      p[a] = p[b];
      p[b] = t;
    }
  }
  return p + i;
}

static char *FormatInt32(char *p, int32_t x) {
  if (x < 0) *p++ = '-', x = -(uint32_t)x;
  return FormatUint32(p, x);
}

static size_t StrCat(char *dst, const char *src, size_t dsize) {
  size_t m, n = dsize;
  const char *p = dst;
  const char *q = src;
  while (n-- != 0 && *dst != '\0') dst++;
  m = dst - p;
  n = dsize - m;
  if (n-- == 0) {
    return m + strlen(src);
  }
  while (*src != '\0') {
    if (n != 0) {
      *dst++ = *src;
      n--;
    }
    src++;
  }
  *dst = '\0';
  return m + (src - q);
}

static void GetOpts(int argc, char *argv[]) {
  int opt;
  g_mode = ' ';
  while ((opt = getopt(argc, argv, "?hbctw")) != -1) {
    switch (opt) {
      case 'w':
        g_warn = true;
        break;
      case 'c':
        g_check = true;
        break;
      case 't':
        g_mode = ' ';
        break;
      case 'b':
        g_mode = '*';
        break;
      case 'h':
      case '?':
        (void)write(1, USAGE, sizeof(USAGE) - 1);
        exit(0);
      default:
        (void)write(2, USAGE, sizeof(USAGE) - 1);
        exit(64);
    }
  }
}

static void Write(int fd, const char *s, ...) {
  va_list va;
  char buf[512];
  buf[0] = 0;
  va_start(va, s);
  do {
    StrCat(buf, s, sizeof(buf));
  } while ((s = va_arg(va, const char *)));
  va_end(va);
  (void)write(fd, buf, strlen(buf));
}

static bool IsModeCharacter(char c) {
  switch (c) {
    case ' ':
    case '*':
      return true;
    default:
      return false;
  }
}

static bool IsSupportedPath(const char *path) {
  size_t i;
  for (i = 0;; ++i) {
    switch (path[i]) {
      case 0:
        if (i) return true;
        // fallthrough
      case '\r':
      case '\n':
      case '\\':
        Write(2, PROG, ": ", path, ": unsupported path\n", NULL);
        return false;
      default:
        break;
    }
  }
}

static bool GetDigest(const char *path, FILE *f, uint8_t digest[32]) {
  size_t got;
  uint8_t buf[512];
  struct Sha256Ctx ctx;
  Sha256Init(&ctx);
  while ((got = fread(buf, 1, sizeof(buf), f))) {
    Sha256Update(&ctx, buf, got);
  }
  if (ferror(f)) {
    Write(2, PROG, ": ", path, ": ", strerror(errno), "\n", NULL);
    return false;
  }
  Sha256Final(&ctx, digest);
  return true;
}

static char *CopyHex(char *s, const void *p, size_t n) {
  const char *d, *e;
  for (d = (const char *)p, e = d + n; d < e; ++d) {
    *s++ = "0123456789abcdef"[(*d >> 4) & 15];
    *s++ = "0123456789abcdef"[(*d >> 0) & 15];
  }
  *s = 0;
  return s;
}

static bool ProduceDigest(const char *path, FILE *f) {
  char hexdigest[65];
  char mode[2] = {g_mode};
  unsigned char digest[32];
  if (!IsSupportedPath(path)) return false;
  if (!GetDigest(path, f, digest)) return false;
  CopyHex(hexdigest, digest, 32);
  Write(1, hexdigest, " ", mode, path, "\n", NULL);
  return true;
}

static char *Chomp(char *line) {
  size_t i;
  if (line) {
    for (i = strlen(line); i--;) {
      if (line[i] == '\r' || line[i] == '\n') {
        line[i] = '\0';
      } else {
        break;
      }
    }
  }
  return line;
}

static int HexToInt(int c) {
  if ('0' <= c && c <= '9') {
    return c - '0';
  } else if ('a' <= c && c <= 'f') {
    return c - 'a' + 10;
  } else if ('A' <= c && c <= 'F') {
    return c - 'A' + 10;
  } else {
    return -1;
  }
}

static bool CheckDigests(const char *path, FILE *f) {
  FILE *f2;
  bool k = true;
  int a, b, i, line;
  const char *path2, *status;
  uint8_t wantdigest[32], gotdigest[32];
  char buf[64 + 2 + PATH_MAX + 1 + 1], *p;
  for (line = 0; fgets(buf, sizeof(buf), f); ++line) {
    if (!*Chomp(buf)) continue;
    for (p = buf, i = 0; i < 32; ++i) {
      if ((a = HexToInt(*p++ & 255)) == -1) goto InvalidLine;
      if ((b = HexToInt(*p++ & 255)) == -1) goto InvalidLine;
      wantdigest[i] = a << 4 | b;
    }
    if (*p++ != ' ') goto InvalidLine;
    if (!IsModeCharacter(*p++)) goto InvalidLine;
    path2 = p;
    if (!*path2) goto InvalidLine;
    if (!IsSupportedPath(path2)) continue;
    if ((f2 = fopen(path2, "rb"))) {
      if (GetDigest(path2, f2, gotdigest)) {
        if (!memcmp(wantdigest, gotdigest, 32)) {
          status = "OK";
        } else {
          status = "FAILED";
          ++g_mismatches;
          k = false;
        }
        Write(1, path2, ": ", status, "\n", NULL);
      } else {
        k = false;
      }
      fclose(f2);
    } else {
      Write(2, PROG, ": ", path2, ": ", strerror(errno), "\n", NULL);
      k = false;
    }
    continue;
  InvalidLine:
    if (g_warn) {
      char linestr[12];
      FormatInt32(linestr, line + 1);
      Write(2, PROG, ": ", path, ":", linestr, ": ",
            "improperly formatted checksum line", "\n", NULL);
    }
  }
  if (ferror(f)) {
    Write(2, PROG, ": ", path, ": ", strerror(errno), "\n", NULL);
    k = false;
  }
  return k;
}

static bool Process(const char *path, FILE *f) {
  if (g_check) {
    return CheckDigests(path, f);
  } else {
    return ProduceDigest(path, f);
  }
}

int main(int argc, char *argv[]) {
  int i;
  FILE *f;
  bool k = true;
  GetOpts(argc, argv);
  if (optind == argc) {
    f = stdin;
    k &= Process("-", f);
  } else {
    for (i = optind; i < argc; ++i) {
      if ((f = fopen(argv[i], "rb"))) {
        k &= Process(argv[i], f);
        fclose(f);
      } else {
        Write(2, PROG, ": ", argv[i], ": ", strerror(errno), "\n", NULL);
        k = false;
      }
    }
  }
  if (g_mismatches) {
    char ibuf[12];
    FormatInt32(ibuf, g_mismatches);
    Write(2, PROG, ": WARNING: ", ibuf, " computed checksum did NOT match\n",
          NULL);
  }
  return !k;
}
