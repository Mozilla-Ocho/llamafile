// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#include "llamafile.h"

#include <cosmo.h>
#include <fenv.h>
#include <signal.h>
#include <stdatomic.h>
#include <termios.h>
#include <unistd.h>

#include "llama.cpp/ggml.h"

#define CONFIG FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW // | FE_UNDERFLOW

bool FLAG_trap;
thread_local int llamafile_debug_op_index;
const struct ggml_cgraph *llamafile_debug_graph;

static struct TerminalBuddy {
    TerminalBuddy() {
        active = !tcgetattr(0, &state);
    }
    void restore() {
        if (active)
            tcsetattr(0, TCSANOW, &state);
    }
    bool active;
    struct termios state;
} g_terminal_buddy;

static const char *describe_vertex(struct ggml_tensor *t) {
    if (t->op == GGML_OP_UNARY)
        return ggml_unary_op_name((enum ggml_unary_op)t->op_params[0]);
    if (t->op)
        return ggml_op_name(t->op);
    return t->name;
}

static void print_graph(FILE *f, const struct ggml_cgraph *g) {
    for (int i = 0; i < g->n_nodes; ++i) {
        fprintf(f, "%5d %p %s:%s(", i, g->nodes[i], ggml_type_name(g->nodes[i]->type),
                describe_vertex(g->nodes[i]));
        for (int j = 0; j < GGML_MAX_SRC && g->nodes[i]->src[j]; ++j) {
            if (j)
                fprintf(f, ", ");
            fprintf(f, "%s:%s[%p]", ggml_type_name(g->nodes[i]->src[j]->type),
                    describe_vertex(g->nodes[i]->src[j]), g->nodes[i]->src[j]);
        }
        fprintf(f, ")\n");
    }
}

// TODO: put this in cosmo libc
int feenableexcept(int excepts) {
    excepts &= FE_ALL_EXCEPT;
#ifdef __x86_64__
    unsigned short neux;
    asm("fstcw\t%0" : "=m"(neux));
    unsigned short res = ~neux & FE_ALL_EXCEPT;
    neux &= ~excepts;
    asm("fldcw\t%0" : /* no inputs */ : "m"(neux));
    unsigned neu;
    asm("stmxcsr\t%0" : "=m"(neu));
    neu &= ~(excepts << 7);
    asm("ldmxcsr\t%0" : /* no inputs */ : "m"(neu));
    return res;
#else
    unsigned fpcr;
    unsigned fpcr2;
    unsigned updated_fpcr;
    fpcr = __builtin_aarch64_get_fpcr();
    fpcr2 = fpcr | (excepts << 8);
    if (fpcr != fpcr2) {
        __builtin_aarch64_set_fpcr(fpcr2);
        // floating point exception trapping is optional in aarch64
        updated_fpcr = __builtin_aarch64_get_fpsr();
        if (fpcr2 & ~updated_fpcr)
            return -1;
    }
    return (fpcr >> 8) & FE_ALL_EXCEPT;
#endif
}

int fedisableexcept(int excepts) {
    excepts &= FE_ALL_EXCEPT;
#ifdef __x86_64__
    unsigned neu;
    unsigned short neux;
    asm("fstcw\t%0" : "=m"(neux));
    unsigned short res = ~neux & FE_ALL_EXCEPT;
    neux |= excepts;
    asm("fldcw\t%0" : /* no inputs */ : "m"(neux));
    asm("stmxcsr\t%0" : "=m"(neu));
    neu |= excepts << 7;
    asm("ldmxcsr\t%0" : /* no inputs */ : "m"(neu));
    return res;
#else
    unsigned fpcr;
    unsigned fpcr2;
    fpcr = __builtin_aarch64_get_fpcr();
    fpcr2 = fpcr & ~(excepts << 8);
    if (fpcr != fpcr2)
        __builtin_aarch64_set_fpcr(fpcr2);
    return (fpcr >> 8) & FE_ALL_EXCEPT;
#endif
}

static inline void spinlock(atomic_uint *lock) {
    int x;
    for (;;) {
        x = atomic_exchange_explicit(lock, 1, memory_order_acquire);
        if (!x)
            break;
    }
}

static void on_sigfpe(int sig, siginfo_t *si, void *ctx) {
    static atomic_uint lock;
    spinlock(&lock); // first thread to crash wins
    g_terminal_buddy.restore();
    const char *s = "\nerror: ";
    if (si->si_code == FPE_INTDIV)
        tinyprint(2, s, "integer divide by zero\n", NULL);
    else if (si->si_code == FPE_INTOVF)
        tinyprint(2, s, "integer overflow\n", NULL);
    else if (si->si_code == FPE_FLTDIV)
        tinyprint(2, s, "floating point divide by zero\n", NULL);
    else if (si->si_code == FPE_FLTOVF)
        tinyprint(2, s, "floating point overflow\n", NULL);
    else if (si->si_code == FPE_FLTUND)
        tinyprint(2, s, "floating point underflow\n", NULL);
    else if (si->si_code == FPE_FLTRES)
        tinyprint(2, s, "floating point inexact\n", NULL);
    else if (si->si_code == FPE_FLTINV)
        tinyprint(2, s, "invalid floating point operation\n", NULL);
    else if (si->si_code == FPE_FLTSUB)
        tinyprint(2, s, "subscript out of range\n", NULL);
    else
        tinyprint(2, s, "caught sigfpe\n", NULL);
    ShowBacktrace(2, 0);
#ifdef LLAMAFILE_DEBUG
    const struct ggml_cgraph *g;
    if ((g = llamafile_debug_graph)) {
        int idx = llamafile_debug_op_index;
        const char *path = "/tmp/cgraph.txt";
        fprintf(stderr, "while executing %s at index #%d of %s\n", //
                ggml_op_name(g->nodes[idx]->op), idx, path);
        FILE *f = fopen(path, "w");
        print_graph(f, g);
        fclose(f);
    }
#endif
    _exit(128 + SIGFPE);
}

static void setup_sigfpe(void) {
    struct sigaction sa;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = on_sigfpe;
    if (sigaction(SIGFPE, &sa, 0)) {
        perror("sigaction(SIGFPE)");
        exit(1);
    }
}

int llamafile_trapping_enabled(int delta) {
    static atomic_uint once;
    static thread_local int g_enabled;
    bool was_enabled = g_enabled > 0;
    bool is_enabled = (g_enabled += delta) > 0;
    feclearexcept(FE_ALL_EXCEPT);
    if (is_enabled && !was_enabled) {
        cosmo_once(&once, setup_sigfpe);
        feenableexcept(CONFIG);
    }
    if (!is_enabled && was_enabled) {
        fedisableexcept(CONFIG);
    }
    return g_enabled;
}
