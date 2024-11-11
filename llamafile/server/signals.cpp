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
#include "llamafile/crash.h"
#include "llamafile/server/log.h"
#include "llamafile/server/server.h"
#include "llamafile/threadlocal.h"
#include <signal.h>
#include <ucontext.h>

namespace lf {
namespace server {

static struct
{
    struct sigaction sigint; // ctrl-c
    struct sigaction sighup; // terminal close
    struct sigaction sigterm; // kill
    struct sigaction sigabrt; // abort()
    struct sigaction sigtrap; // breakpoint
    struct sigaction sigfpe; // illegal math
    struct sigaction sigbus; // illegal memory
    struct sigaction sigsegv; // illegal memory
    struct sigaction sigill; // illegal instruction
    struct sigaction sigxcpu; // out of cpu quota
    struct sigaction sigxfsz; // file too large
} old;

void
on_termination_signal(int sig)
{
    SLOG("%G", sig);
    g_server->terminate();
}

void
on_crash_signal(int sig, siginfo_t* si, void* arg)
{
    SLOG("crashed %G", sig);
    char message[256];
    describe_crash(message, sizeof(message), sig, si, arg);
    SLOG("crashed %s", message);
    pthread_exit(PTHREAD_CANCELED);
}

void
signals_init(void)
{
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = on_termination_signal;

    sigaction(SIGINT, &sa, &old.sigint);
    sigaction(SIGHUP, &sa, &old.sighup);
    sigaction(SIGTERM, &sa, &old.sigterm);

    sa.sa_sigaction = on_crash_signal;
    sigaddset(&sa.sa_mask, SIGABRT);
    sigaddset(&sa.sa_mask, SIGTRAP);
    sigaddset(&sa.sa_mask, SIGFPE);
    sigaddset(&sa.sa_mask, SIGBUS);
    sigaddset(&sa.sa_mask, SIGSEGV);
    sigaddset(&sa.sa_mask, SIGILL);
    sigaddset(&sa.sa_mask, SIGXCPU);
    sigaddset(&sa.sa_mask, SIGXFSZ);

    sigaction(SIGABRT, &sa, &old.sigabrt);
    sigaction(SIGTRAP, &sa, &old.sigtrap);
    sigaction(SIGFPE, &sa, &old.sigfpe);
    sigaction(SIGILL, &sa, &old.sigill);
    sigaction(SIGXCPU, &sa, &old.sigxcpu);
    sigaction(SIGXFSZ, &sa, &old.sigxfsz);

    sa.sa_flags |= SA_ONSTACK;
    sigaction(SIGBUS, &sa, &old.sigbus);
    sigaction(SIGSEGV, &sa, &old.sigsegv);
}

void
signals_destroy(void)
{
    sigaction(SIGINT, &old.sigint, 0);
    sigaction(SIGHUP, &old.sighup, 0);
    sigaction(SIGTERM, &old.sigterm, 0);
    sigaction(SIGABRT, &old.sigabrt, 0);
    sigaction(SIGTRAP, &old.sigtrap, 0);
    sigaction(SIGFPE, &old.sigfpe, 0);
    sigaction(SIGBUS, &old.sigbus, 0);
    sigaction(SIGSEGV, &old.sigsegv, 0);
    sigaction(SIGILL, &old.sigill, 0);
    sigaction(SIGXCPU, &old.sigxcpu, 0);
    sigaction(SIGXFSZ, &old.sigxfsz, 0);
}

} // namespace server
} // namespace lf
