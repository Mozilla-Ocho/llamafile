// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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

#include "crash.h"
#include <cosmo.h>
#include <stdlib.h>
#include <third_party/dlmalloc/dlmalloc.h>

void *memlog(void *p) {
    char bt[160];
    struct StackFrame *sf;
    sf = (struct StackFrame *)__builtin_frame_address(0);
    describe_backtrace(bt, sizeof(bt), sf);
    kprintf("alloc %p bt %s\n", p, bt);
    return p;
}

void *malloc(size_t n) {
    return memlog(dlmalloc(n));
}

void *calloc(size_t n, size_t z) {
    return memlog(dlcalloc(n, z));
}

void *realloc(void *p, size_t n) {
    return memlog(dlrealloc(p, n));
}

void *memalign(size_t a, size_t n) {
    return memlog(dlmemalign(a, n));
}
