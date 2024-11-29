/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;tab-width:4;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright The Mbed TLS Contributors                                          │
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
#include "third_party/mbedtls/iana.h"

bool IsCipherSuiteGood(uint16_t x) {
  switch (x) {
    case 0x009E: /* TLS_DHE_RSA_WITH_AES_128_GCM_SHA256 (RFC5288) */
    case 0x009F: /* TLS_DHE_RSA_WITH_AES_256_GCM_SHA384 (RFC5288) */
    case 0x00AA: /* TLS_DHE_PSK_WITH_AES_128_GCM_SHA256 (RFC5487) */
    case 0x00AB: /* TLS_DHE_PSK_WITH_AES_256_GCM_SHA384 (RFC5487) */
    case 0x1301: /* TLS_AES_128_GCM_SHA256 (RFC8446) */
    case 0x1302: /* TLS_AES_256_GCM_SHA384 (RFC8446) */
    case 0x1303: /* TLS_CHACHA20_POLY1305_SHA256 (RFC8446) */
    case 0x1304: /* TLS_AES_128_CCM_SHA256 (RFC8446) */
    case 0xC02B: /* TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 (RFC5289) */
    case 0xC02C: /* TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 (RFC5289) */
    case 0xC02F: /* TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 (RFC5289) */
    case 0xC030: /* TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 (RFC5289) */
    case 0xC09E: /* TLS_DHE_RSA_WITH_AES_128_CCM (RFC6655) */
    case 0xC09F: /* TLS_DHE_RSA_WITH_AES_256_CCM (RFC6655) */
    case 0xC0A6: /* TLS_DHE_PSK_WITH_AES_128_CCM (RFC6655) */
    case 0xC0A7: /* TLS_DHE_PSK_WITH_AES_256_CCM (RFC6655) */
    case 0xCCA8: /* TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 (RFC7905) */
    case 0xCCA9: /* TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256 (RFC7905) */
    case 0xCCAA: /* TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256 (RFC7905) */
    case 0xCCAC: /* TLS_ECDHE_PSK_WITH_CHACHA20_POLY1305_SHA256 (RFC7905) */
    case 0xCCAD: /* TLS_DHE_PSK_WITH_CHACHA20_POLY1305_SHA256 (RFC7905) */
    case 0xD001: /* TLS_ECDHE_PSK_WITH_AES_128_GCM_SHA256 (RFC8442) */
    case 0xD002: /* TLS_ECDHE_PSK_WITH_AES_256_GCM_SHA384 (RFC8442) */
    case 0xD005: /* TLS_ECDHE_PSK_WITH_AES_128_CCM_SHA256 (RFC8442) */
      return true;
    default:
      return false;
  }
}
