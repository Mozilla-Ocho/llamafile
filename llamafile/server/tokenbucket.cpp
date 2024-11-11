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

#include "tokenbucket.h"
#include "llamafile/llamafile.h"
#include "llamafile/server/log.h"
#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>

namespace lf {
namespace server {

union TokenBucket
{
    atomic_uint_fast64_t* w;
    atomic_schar* b;
};

static size_t g_words;
static pthread_t g_thread;
static TokenBucket g_tokens;

static void
replenish_tokens(atomic_uint_fast64_t* w, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        uint64_t a = atomic_load_explicit(w + i, memory_order_relaxed);
        if (a == 0x7f7f7f7f7f7f7f7f)
            continue;
        uint64_t b = 0x8080808080808080;
        uint64_t c = 0x7f7f7f7f7f7f7f7f ^ a;
        uint64_t d = ((((c >> 1 | b) - c) & b) ^ b) >> 7;
        atomic_fetch_add_explicit(w + i, d, memory_order_relaxed);
    }
}

static int
acquire_token(atomic_schar* b, uint32_t x, int c)
{
    uint32_t i = x >> (32 - c);
    int t = atomic_load_explicit(b + i, memory_order_relaxed);
    if (t > 0)
        t = atomic_fetch_add_explicit(b + i, -1, memory_order_relaxed);
    return -t + 127;
}

void
tokenbucket_replenish()
{
    replenish_tokens(g_tokens.w, g_words);
}

static void
tokenbucket_replenisher()
{
    timespec rate = timespec_frommicros(1e6 / FLAG_token_rate);
    timespec next = timespec_add(timespec_real(), rate);
    for (;;) {
        while (timespec_cmp(timespec_real(), next) > 0) {
            tokenbucket_replenish();
            next = timespec_add(next, rate);
        }
        clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &next, 0);
    }
}

static void*
tokenbucket_thread(void* arg)
{
    sigset_t ss;
    sigemptyset(&ss);
    sigaddset(&ss, SIGHUP);
    sigaddset(&ss, SIGINT);
    sigaddset(&ss, SIGQUIT);
    sigaddset(&ss, SIGTERM);
    sigaddset(&ss, SIGUSR1);
    sigaddset(&ss, SIGALRM);
    pthread_sigmask(SIG_SETMASK, &ss, 0);
    set_thread_name("tokenbucket");
    tokenbucket_replenisher();
    return nullptr;
}

void
tokenbucket_init()
{
    size_t bytes = 1;
    bytes <<= FLAG_token_cidr;
    g_words = bytes / 8;
    if (!(g_tokens.b = (atomic_schar*)malloc(bytes)))
        __builtin_trap();
    memset(g_tokens.b, 127, bytes);
    if (pthread_create(&g_thread, 0, tokenbucket_thread, 0))
        __builtin_trap();
}

void
tokenbucket_destroy()
{
    pthread_cancel(g_thread);
    if (pthread_join(g_thread, 0))
        __builtin_trap();
    free(g_tokens.b);
}

int
tokenbucket_acquire(unsigned ip)
{
    return acquire_token(g_tokens.b, ip, FLAG_token_cidr);
}

} // namespace server
} // namespace lf
