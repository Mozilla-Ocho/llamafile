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

#include "x.h"
#include <cosmo.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static wontreturn void oom(void) {
    tinyprint(2, program_invocation_name, ": out of memory\n", NULL);
    exit(1);
}

char *xvasprintf(const char *fmt, va_list va) {
    char *buf;
    if (vasprintf(&buf, fmt, va) == -1)
        oom();
    return buf;
}

char *xasprintf(const char *fmt, ...) {
    char *res;
    va_list va;
    va_start(va, fmt);
    res = xvasprintf(fmt, va);
    va_end(va);
    return res;
}
