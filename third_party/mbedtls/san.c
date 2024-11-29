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
#include "third_party/mbedtls/san.h"
#include <libc/serialize.h>
#include <libc/sock/sock.h>
#include <libc/sysv/consts/af.h>
#include "third_party/mbedtls/asn1.h"
#include "third_party/mbedtls/asn1write.h"
#include "third_party/mbedtls/oid.h"
#include "third_party/mbedtls/platform.h"
#include "third_party/mbedtls/x509_crt.h"

/**
 * Writes Subject Alternative Name section to certificate.
 *
 * @see RFC5280 §4.2.1.6
 */
int mbedtls_x509write_crt_set_subject_alternative_name(
    mbedtls_x509write_cert *ctx, const struct mbedtls_san *san, size_t sanlen) {
  int ret;
  const unsigned char *item;
  size_t i, len, cap, itemlen;
  unsigned char *pc, *buf, ip4[4];
  if (!sanlen) return 0;
  cap = sanlen * (253 + 5 + 1) + 5 + 1;
  if (!(buf = mbedtls_calloc(1, cap))) return MBEDTLS_ERR_ASN1_ALLOC_FAILED;
  pc = buf + cap;
  len = 0;
  for (i = sanlen; i--;) {
    switch (san[i].tag) {
      case MBEDTLS_X509_SAN_RFC822_NAME:
      case MBEDTLS_X509_SAN_DNS_NAME:
      case MBEDTLS_X509_SAN_UNIFORM_RESOURCE_IDENTIFIER:
        item = (const unsigned char *)san[i].val;
        itemlen = strlen(san[i].val);
        break;
      case MBEDTLS_X509_SAN_IP_ADDRESS:
        WRITE32BE(ip4, san[i].ip4);
        item = ip4;
        itemlen = 4;
        break;
      default:
        ret = MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE;
        goto finish;
    }
    if (itemlen > 253) {
      ret = MBEDTLS_ERR_ASN1_INVALID_LENGTH;
      goto finish;
    }
    ret = mbedtls_asn1_write_raw_buffer(&pc, buf, item, itemlen);
    if (ret < 0) goto finish;
    len += ret;
    ret = mbedtls_asn1_write_len(&pc, buf, itemlen);
    if (ret < 0) goto finish;
    len += ret;
    ret = mbedtls_asn1_write_tag(&pc, buf,
                                 MBEDTLS_ASN1_CONTEXT_SPECIFIC | san[i].tag);
    if (ret < 0) goto finish;
    len += ret;
  }
  ret = mbedtls_asn1_write_len(&pc, buf, len);
  if (ret < 0) goto finish;
  len += ret;
  ret = mbedtls_asn1_write_tag(
      &pc, buf, MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE);
  if (ret < 0) goto finish;
  len += ret;
  ret = mbedtls_x509write_crt_set_extension(
      ctx, MBEDTLS_OID_SUBJECT_ALT_NAME,
      MBEDTLS_OID_SIZE(MBEDTLS_OID_SUBJECT_ALT_NAME), 0, buf + cap - len, len);
finish:
  mbedtls_free(buf);
  return ret;
}
