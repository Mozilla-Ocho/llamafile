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

#include "trace.h"

#include <cosmo.h>
#include <pthread.h>
#include <stdatomic.h>
#include <threads.h>

#include "llamafile.h"
#include "log.h"

struct TraceEvent {
    unsigned long long ts;
    int pid;
    int tid;
    const char *name;
    const char *cat;
    char ph;
};

static int g_pid;
static atomic_bool g_oom;
static atomic_int g_count;
static thread_local int g_id;
static thread_local int g_ids;
static thread_local int g_tid;
static struct TraceEvent g_events[1000000];

static int llamafile_trace_oom(void) {
    if (atomic_load_explicit(&g_oom, memory_order_relaxed))
        return -1;
    if (atomic_exchange_explicit(&g_oom, true, memory_order_acq_rel))
        return -1;
    tinylog("warning: ran out of trace event memory\n", NULL);
    return -1;
}

static int llamafile_trace_reserve(int count) {
    int id = atomic_load_explicit(&g_count, memory_order_relaxed);
    if (id + count > sizeof(g_events) / sizeof(*g_events))
        return llamafile_trace_oom();
    id = atomic_fetch_add_explicit(&g_count, count, memory_order_acq_rel);
    if (id + count > sizeof(g_events) / sizeof(*g_events))
        return llamafile_trace_oom();
    return id;
}

static void llamafile_trace_event(int id, const char *name, const char *cat, char ph) {
    g_events[id].ts = rdtsc();
    g_events[id].pid = g_pid ? g_pid - 1 : getpid();
    g_events[id].tid = g_tid ? g_tid - 1 : gettid();
    g_events[id].name = name;
    g_events[id].cat = cat;
    g_events[id].ph = ph;
}

void llamafile_trace_set_pid(int pid) {
    g_pid = pid + 1;
}

void llamafile_trace_set_tid(int tid) {
    g_tid = tid + 1;
}

void llamafile_trace_begin(const char *name) {
    if (!FLAG_trace)
        return;
    if (g_ids < 2) {
        g_ids = 20;
        g_id = llamafile_trace_reserve(g_ids);
        if (g_id == -1) {
            g_ids = 0;
            return;
        }
    }
    llamafile_trace_event(g_id++, name, "category", 'B');
    --g_ids;
}

void llamafile_trace_end(const char *name) {
    if (!FLAG_trace)
        return;
    if (g_ids < 1)
        return;
    llamafile_trace_event(g_id++, name, "category", 'E');
    --g_ids;
}

static void llamafile_trace_save(const char *filename) {
    int count = atomic_load_explicit(&g_count, memory_order_relaxed);
    if (!count)
        return;
    tinylog("saving trace to ", filename, "...\n", NULL);
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror(filename);
        return;
    }
    fprintf(file, "[\n");
    bool once = false;
    for (int i = 0; i < count; i++) {
        if (!g_events[i].name)
            continue;
        if (!once) {
            once = true;
        } else {
            fputs(",\n", file);
        }
        fprintf(file,
                "{\"name\": \"%s\", \"cat\": \"%s\", \"ph\": \"%c\", "
                "\"ts\": %.3f, \"pid\": %d, \"tid\": %d}",
                g_events[i].name, g_events[i].cat, g_events[i].ph,
                (g_events[i].ts - kStartTsc) / 3000., g_events[i].pid, g_events[i].tid);
    }
    fprintf(file, "\n]\n");
    fclose(file);
}

__attribute__((__destructor__)) static void trace_shutdown(void) {
    llamafile_trace_save("trace.json"); // see chrome://tracing/
}
