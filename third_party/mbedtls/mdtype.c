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
#include "third_party/mbedtls/md.h"

const char *mbedtls_md_type_name(mbedtls_md_type_t t) {
  switch (t) {
    case MBEDTLS_MD_NONE:
      return "NONE";
    case MBEDTLS_MD_MD2:
      return "MD2";
    case MBEDTLS_MD_MD4:
      return "MD4";
    case MBEDTLS_MD_MD5:
      return "MD5";
    case MBEDTLS_MD_SHA1:
      return "SHA1";
    case MBEDTLS_MD_SHA224:
      return "SHA224";
    case MBEDTLS_MD_SHA256:
      return "SHA256";
    case MBEDTLS_MD_SHA384:
      return "SHA384";
    case MBEDTLS_MD_SHA512:
      return "SHA512";
    default:
      return 0;
  }
}
