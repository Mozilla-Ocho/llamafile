// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2023 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// example usage
//
// /opt/cosmocc/bin/x86_64-linux-cosmo-g++
// -oo//llama.cpp/server/server.o -D__COSMOPOLITAN__ -D__COSMOCC__
// -D__FATCOSMOCC__ -include libc/integral/normalize.inc -fportcosmo
// -fno-dwarf2-cfi-asm -fno-unwind-tables
// -fno-asynchronous-unwind-tables -fno-semantic-interposition
// -fno-optimize-sibling-calls -mno-omit-leaf-frame-pointer -fno-rtti
// -fno-exceptions -fuse-cxa-atexit -mno-tls-direct-seg-refs
// -fpatchable-function-entry=18,16 -fno-pie -nostdinc -fno-math-errno
// -isystem /opt/cosmocc/bin/../include -mno-red-zone -D_COSMO_SOURCE -g
// -O3 -iquote. -c -c llama.cpp/server/server.cpp -mssse3
// -fno-omit-frame-pointer -Q |& o//llamafile/addnl

int main(int argc, char *argv[]) {
    char tbuf[128];
    static char buf[134217728];
    struct timespec t = timespec_real();
    for (;;) {
        ssize_t rc = read(0, buf, sizeof(buf));
        if (!rc)
            break;
        struct timespec u = timespec_real();
        struct timespec d = timespec_sub(u, t);
        t = u;
        snprintf(tbuf, sizeof(tbuf), "%012ld ", timespec_tomicros(d));
        write(1, tbuf, strlen(tbuf));
        write(1, buf, rc);
        write(1, "\n", 1);
    }
}
