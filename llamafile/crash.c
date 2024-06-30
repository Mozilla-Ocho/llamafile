// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
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
#include <ucontext.h>

#ifdef __aarch64__
#define PC pc
#define BP regs[29]
#else
#define PC gregs[REG_RIP]
#define BP gregs[REG_RBP]
#endif

char *hexcpy(char *p, unsigned long x) {
    int k = x ? (__builtin_clzl(x) ^ 63) + 1 : 1;
    k = (k + 3) & -4;
    while (k > 0)
        *p++ = "0123456789abcdef"[(x >> (k -= 4)) & 15];
    *p = '\0';
    return p;
}

// returns true if `p` is preceded by x86 call instruction
// this is actually impossible to do but we'll do our best
int is_call(const unsigned char *p) {
    if (p[-5] == 0xe8)
        return 5; // call Jvds
    if (p[-2] == 0xff && (p[-1] & 070) == 020)
        return 2; // call %reg
    if (p[-4] == 0xff && (p[-3] & 070) == 020)
        return 4; // call disp8(%reg,%reg)
    if (p[-3] == 0xff && (p[-2] & 070) == 020)
        return 3; // call disp8(%reg)
    if (p[-7] == 0xff && (p[-6] & 070) == 020)
        return 7; // call disp32(%reg,%reg)
    if (p[-6] == 0xff && (p[-5] & 070) == 020)
        return 6; // call disp32(%reg)
    return 0;
}

//                         abashed the devil stood
//                      and felt how awful goodness is
char *describe_crash(char *buf, size_t len, int sig, siginfo_t *si, void *arg) {
    char *p = buf;

    // check minimum length
    if (len < 64)
        return p;

    // describe crash
    char signame[21];
    p = stpcpy(p, strsignal_r(sig, signame));
    if (si && //
        (sig == SIGFPE || //
         sig == SIGILL || //
         sig == SIGBUS || //
         sig == SIGSEGV || //
         sig == SIGTRAP)) {
        p = stpcpy(p, " at ");
        p = hexcpy(p, (long)si->si_addr);
    }

    // get stack frame daisy chain
    struct StackFrame pc;
    struct StackFrame *sf;
    ucontext_t *ctx;
    if ((ctx = (ucontext_t *)arg)) {
        pc.addr = ctx->uc_mcontext.PC;
        pc.next = (struct StackFrame *)ctx->uc_mcontext.BP;
        sf = &pc;
    } else {
        sf = (struct StackFrame *)__builtin_frame_address(0);
    }

    // describe backtrace
    p = stpcpy(p, " bt ");
    p = describe_backtrace(p, len - (p - buf), sf);

    return p;
}

char *describe_backtrace(char *p, size_t len, const struct StackFrame *sf) {
    char *pe = p + len;
    bool gotsome = false;

    // show address of each function
    while (sf) {
        if (kisdangerous(sf)) {
            if (p + 1 + 9 + 1 < pe) {
                if (gotsome)
                    *p++ = ' ';
                p = stpcpy(p, "DANGEROUS");
                if (p + 16 + 1 < pe) {
                    *p++ = ' ';
                    p = hexcpy(p, (long)sf);
                }
            }
            break;
        }
        if (p + 16 + 1 < pe) {
            unsigned char *ip = (unsigned char *)sf->addr;
#ifdef __x86_64__
            // x86 advances the progrem counter before an instruction
            // begins executing. return addresses in backtraces shall
            // point to code after the call, which means addr2line is
            // going to print unrelated code unless we fixup the addr
            if (!kisdangerous(ip))
                ip -= is_call(ip);
#endif
            if (gotsome)
                *p++ = ' ';
            else
                gotsome = true;
            p = hexcpy(p, (long)ip);
        } else {
            break;
        }
        sf = sf->next;
    }

    // terminate string
    if (p < pe)
        *p = '\0';
    return p;
}
