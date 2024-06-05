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

#include "signals.h"

#include <ucontext.h>

#include "log.h"
#include "server.h"

void
on_termination_signal(int sig)
{
    LOG("%G", sig);
    g_server->terminate();
}

void
on_crash_signal(int sig, siginfo_t* si, void* arg)
{
    LOG("crashed %G", sig);
    char message[256];
    describe_crash(message, sizeof(message), sig, si, arg);
    LOG("crashed %s", message);
    pthread_exit(PTHREAD_CANCELED);
}

void
setup_signals(void)
{
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = on_termination_signal;

    sigaction(SIGINT, &sa, 0); // ctrl-c
    sigaction(SIGHUP, &sa, 0); // terminal close
    sigaction(SIGTERM, &sa, 0); // kill

    sa.sa_sigaction = on_crash_signal;
    sigaddset(&sa.sa_mask, SIGABRT); // abort()
    sigaddset(&sa.sa_mask, SIGTRAP); // breakpoint
    sigaddset(&sa.sa_mask, SIGFPE); // illegal math
    sigaddset(&sa.sa_mask, SIGBUS); // illegal memory
    sigaddset(&sa.sa_mask, SIGSEGV); // illegal memory
    sigaddset(&sa.sa_mask, SIGILL); // illegal instruction
    sigaddset(&sa.sa_mask, SIGXCPU); // out of cpu quota
    sigaddset(&sa.sa_mask, SIGXFSZ); // file too large

    sigaction(SIGABRT, &sa, 0); // abort()
    sigaction(SIGTRAP, &sa, 0); // breakpoint
    sigaction(SIGFPE, &sa, 0); // illegal math
    sigaction(SIGBUS, &sa, 0); // illegal memory
    sigaction(SIGSEGV, &sa, 0); // illegal memory
    sigaction(SIGILL, &sa, 0); // illegal instruction
    sigaction(SIGXCPU, &sa, 0); // out of cpu quota
    sigaction(SIGXFSZ, &sa, 0); // file too large
}

void
restore_signals(void)
{
    struct sigaction sa;
    sa.sa_flags = 0;
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);

    sigaction(SIGINT, &sa, 0); // ctrl-c
    sigaction(SIGHUP, &sa, 0); // terminal close
    sigaction(SIGTERM, &sa, 0); // kill

    sigaction(SIGABRT, &sa, 0); // abort()
    sigaction(SIGTRAP, &sa, 0); // breakpoint
    sigaction(SIGFPE, &sa, 0); // illegal math
    sigaction(SIGBUS, &sa, 0); // illegal memory
    sigaction(SIGSEGV, &sa, 0); // illegal memory
    sigaction(SIGILL, &sa, 0); // illegal instruction
    sigaction(SIGXCPU, &sa, 0); // out of cpu quota
    sigaction(SIGXFSZ, &sa, 0); // file too large
}
