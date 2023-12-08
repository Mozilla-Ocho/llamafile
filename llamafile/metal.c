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

#include <time.h>
#include <cosmo.h>
#include <dlfcn.h>
#include <errno.h>
#include <spawn.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <limits.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <stdatomic.h>
#include "llama.cpp/ggml-metal.h"

__static_yoink("llama.cpp/ggml.h");
__static_yoink("llamafile/llamafile.h");
__static_yoink("llama.cpp/ggml-impl.h");
__static_yoink("llama.cpp/ggml-metal.m");
__static_yoink("llama.cpp/ggml-metal.h");
__static_yoink("llama.cpp/ggml-quants.h");
__static_yoink("llama.cpp/ggml-metal.metal");

static const struct Source {
    const char *zip;
    const char *name;
} srcs[] = {
    {"/zip/llama.cpp/ggml.h", "ggml.h"},
    {"/zip/llamafile/llamafile.h", "llamafile.h"},
    {"/zip/llama.cpp/ggml-impl.h", "ggml-impl.h"},
    {"/zip/llama.cpp/ggml-metal.h", "ggml-metal.h"},
    {"/zip/llama.cpp/ggml-quants.h", "ggml-quants.h"},
    {"/zip/llama.cpp/ggml-metal.metal", "ggml-metal.metal"},
    {"/zip/llama.cpp/ggml-metal.m", "ggml-metal.m"}, // must come last
};

static struct Metal {
    bool supported;
    atomic_uint once;
    typeof(ggml_metal_add_buffer) *add_buffer;
    typeof(ggml_metal_free) *free;
    typeof(ggml_metal_get_concur_list) *get_concur_list;
    typeof(ggml_metal_graph_compute) *graph_compute;
    typeof(ggml_metal_graph_find_concurrency) *graph_find_concurrency;
    typeof(ggml_metal_host_free) *host_free;
    typeof(ggml_metal_host_malloc) *host_malloc;
    typeof(ggml_metal_if_optimized) *if_optimized;
    typeof(ggml_metal_init) *init;
    typeof(ggml_metal_set_n_cb) *set_n_cb;
} ggml_metal;

static const char *Dlerror(void) {
    const char *msg;
    msg = cosmo_dlerror();
    if (!msg) msg = "null dlopen error";
    return msg;
}

static bool ImportMetalImpl(void) {

    // extract source code
    char src[PATH_MAX];
    bool needs_rebuild = false;
    for (int i = 0; i < sizeof(srcs) / sizeof(*srcs); ++i) {
        llamafile_get_app_dir(src, PATH_MAX);
        if (!i && mkdir(src, 0755) && errno != EEXIST) {
            perror(src);
            return false;
        }
        strlcat(src, srcs[i].name, sizeof(src));
        switch (llamafile_is_file_newer_than(srcs[i].zip, src)) {
            case -1:
                return false;
            case 0:
                break;
            case 1:
                needs_rebuild = true;
                if (!llamafile_extract(srcs[i].zip, src)) {
                    return false;
                }
                break;
            default:
                __builtin_unreachable();
        }
    }

    // determine if we need to build
    char dso[PATH_MAX];
    llamafile_get_app_dir(dso, PATH_MAX);
    strlcat(dso, "ggml-metal.dylib", sizeof(dso));
    if (!needs_rebuild) {
        switch (llamafile_is_file_newer_than(src, dso)) {
            case -1:
                return false;
            case 0:
                break;
            case 1:
                needs_rebuild = true;
                break;
            default:
                __builtin_unreachable();
        }
    }

    // compile dynamic shared object
    if (needs_rebuild) {
        tinyprint(2, "building ggml-metal.dylib with xcode...\n", NULL);
        int fd;
        char tmpdso[PATH_MAX];
        strlcpy(tmpdso, dso, sizeof(tmpdso));
        strlcat(tmpdso, ".XXXXXX", sizeof(tmpdso));
        if ((fd = mkostemp(tmpdso, O_CLOEXEC)) != -1) {
            close(fd);
        } else {
            perror(tmpdso);
            return false;
        }
        char *args[] = {
            "cc",
            "-shared",
            "-O3",
            "-I.",
            "-DTARGET_OS_OSX",
            "-DNDEBUG",
            "-fPIC",
            "-pthread",
            src,
            "-o", tmpdso,
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalKit",
            NULL,
        };
        int pid, ws;
        errno_t err = posix_spawnp(&pid, "cc", NULL, NULL, args, environ);
        if (err) {
            perror("cc");
            if (err == ENOENT) {
                tinyprint(2, "PLEASE RUN: xcode-select --install\n", NULL);
            }
            return false;
        }
        while (waitpid(pid, &ws, 0) == -1) {
            if (errno != EINTR) {
                perror("cc");
                return false;
            }
        }
        if (ws) {
            tinyprint(2, "compiler returned nonzero exit status\n", NULL);
            return false;
        }
        if (rename(tmpdso, dso)) {
            perror(dso);
            return false;
        }
    }

    // runtime link dynamic shared object
    void *lib;
    lib = cosmo_dlopen(dso, RTLD_LAZY);
    if (!lib) {
        tinyprint(2, Dlerror(), ": failed to load library\n", NULL);
        return false;
    }

    // import functions
    bool ok = true;
    ok &= !!(ggml_metal.add_buffer = cosmo_dlsym(lib, "ggml_metal_add_buffer"));
    ok &= !!(ggml_metal.free = cosmo_dlsym(lib, "ggml_metal_free"));
    ok &= !!(ggml_metal.get_concur_list = cosmo_dlsym(lib, "ggml_metal_get_concur_list"));
    ok &= !!(ggml_metal.graph_compute = cosmo_dlsym(lib, "ggml_metal_graph_compute"));
    ok &= !!(ggml_metal.graph_find_concurrency = cosmo_dlsym(lib, "ggml_metal_graph_find_concurrency"));
    ok &= !!(ggml_metal.host_free = cosmo_dlsym(lib, "ggml_metal_host_free"));
    ok &= !!(ggml_metal.host_malloc = cosmo_dlsym(lib, "ggml_metal_host_malloc"));
    ok &= !!(ggml_metal.if_optimized = cosmo_dlsym(lib, "ggml_metal_if_optimized"));
    ok &= !!(ggml_metal.init = cosmo_dlsym(lib, "ggml_metal_init"));
    ok &= !!(ggml_metal.set_n_cb = cosmo_dlsym(lib, "ggml_metal_set_n_cb"));
    if (!ok) {
        tinyprint(2, Dlerror(), ": not all symbols could be imported\n", NULL);
        return false;
    }

    // we're good
    return true;
}

static void ImportMetal(void) {
    if (IsXnuSilicon() && ImportMetalImpl()) {
        ggml_metal.supported = true;
        tinyprint(2, "Apple Metal GPU support successfully loaded\n", NULL);
    }
}

bool ggml_metal_supported(void) {
    cosmo_once(&ggml_metal.once, ImportMetal);
    return ggml_metal.supported;
}

void *ggml_metal_host_malloc(size_t n) {
    if (!ggml_metal_supported()) return NULL;
    return ggml_metal.host_malloc(n);
}

void ggml_metal_host_free(void *data) {
    if (!ggml_metal_supported()) return;
    return ggml_metal.host_free(data);
}

struct ggml_metal_context *ggml_metal_init(int n_cb, const char *metalPath) {
    char path[PATH_MAX];
    if (!ggml_metal_supported()) return NULL;
    if (!metalPath) {
        llamafile_get_app_dir(path, PATH_MAX);
        strlcat(path, "ggml-metal.metal", sizeof(path));
        metalPath = path;
    }
    return ggml_metal.init(n_cb, metalPath);
}

bool ggml_metal_add_buffer(struct ggml_metal_context *ctx,
                           const char *name, void *data,
                           size_t size, size_t max_size) {
    if (!ggml_metal_supported()) return false;
    return ggml_metal.add_buffer(ctx, name, data, size, max_size);
}

void ggml_metal_free(struct ggml_metal_context *ctx) {
    if (!ggml_metal_supported()) return;
    return ggml_metal.free(ctx);
}

int *ggml_metal_get_concur_list(struct ggml_metal_context *ctx) {
    if (!ggml_metal_supported()) return NULL;
    return ggml_metal.get_concur_list(ctx);
}

void ggml_metal_graph_compute(struct ggml_metal_context *ctx,
                              struct ggml_cgraph *gf) {
    if (!ggml_metal_supported()) return;
    return ggml_metal.graph_compute(ctx, gf);
}

void ggml_metal_graph_find_concurrency(struct ggml_metal_context *ctx,
                                       struct ggml_cgraph *gf,
                                       bool check_mem) {
    if (!ggml_metal_supported()) return;
    return ggml_metal.graph_find_concurrency(ctx, gf, check_mem);
}

int ggml_metal_if_optimized(struct ggml_metal_context *ctx) {
    if (!ggml_metal_supported()) return 0;
    return ggml_metal.if_optimized(ctx);
}

void ggml_metal_set_n_cb(struct ggml_metal_context * ctx, int n_cb) {
    if (!ggml_metal_supported()) return;
    return ggml_metal.set_n_cb(ctx, n_cb);
}
