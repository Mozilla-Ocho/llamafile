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
#include <libc/nexgen32e/rdtsc.h>
#include <libc/str/str.h>
#include "third_party/mbedtls/entropy_poll.h"

int mbedtls_hardclock_poll(void *data, unsigned char *output, size_t len,
                           size_t *olen) {
  unsigned long timer;
  timer = rdtsc();
  *olen = 0;
  if (len < sizeof(unsigned long)) return 0;
  memcpy(output, &timer, sizeof(unsigned long));
  *olen = sizeof(unsigned long);
  return 0;
}
