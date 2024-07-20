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

#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include "threadlocal.h"

// Call this from your thread start function
// Your thread needs to have a guard page (default)
// You also need SIGSEGV and SIGBUS handlers w/ SA_ONSTACK
void protect_against_stack_overflow(void) {
    static ThreadLocal<void> cleanup([](void *ptr) {
        struct sigaltstack ss;
        ss.ss_flags = SS_DISABLE;
        ss.ss_size = 0;
        ss.ss_sp = 0;
        sigaltstack(&ss, 0);
        free(ptr);
    });
    struct sigaltstack ss;
    ss.ss_flags = 0;
    ss.ss_size = sysconf(_SC_MINSIGSTKSZ) + 16384;
    if ((ss.ss_sp = malloc(ss.ss_size))) {
        if (sigaltstack(&ss, 0))
            __builtin_trap();
        cleanup.set(ss.ss_sp);
    }
}
