// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set net ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
#define _COSMO_SOURCE
#include <cosmo.h>
#include <time.h>
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
#include "ggml-metal.h"

__static_yoink("llama.cpp/ggml.h");
__static_yoink("llama.cpp/ggml-impl.h");
__static_yoink("llama.cpp/ggml-metal.m");
__static_yoink("llama.cpp/ggml-metal.h");
__static_yoink("llama.cpp/ggml-quants.h");
__static_yoink("llama.cpp/ggml-metal.metal");

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

static const char *GetTmpDir(void) {
    const char *tmpdir;
    if (!(tmpdir = getenv("TMPDIR")) || !*tmpdir) {
        if (!(tmpdir = getenv("HOME")) || !*tmpdir) {
            tmpdir = ".";
        }
    }
    return tmpdir;
}

static void GetAppDir(char path[PATH_MAX]) {
    strlcpy(path, GetTmpDir(), PATH_MAX);
    strlcat(path, "/.llamafile/", PATH_MAX);
}

static bool IsNewerThan(const char *path, const char *other) {
    struct stat st1, st2;
    if (stat(path, &st1) == -1) {
        // PATH should always exist when calling this function
        perror(path);
        return false;
    }
    if (stat(other, &st2) == -1) {
        if (errno == ENOENT) {
            // PATH should replace OTHER because OTHER doesn't exist yet
            return true;
        } else {
            // some other error happened, so we can't do anything
            perror(path);
            return false;
        }
    }
    // PATH should replace OTHER if PATH was modified more recently
    return timespec_cmp(st1.st_mtim, st2.st_mtim) > 0;
}

static const struct Source {
    const char *zip;
    const char *name;
} srcs[] = {
    {"/zip/llama.cpp/ggml.h", "ggml.h"},
    {"/zip/llama.cpp/ggml-impl.h", "ggml-impl.h"},
    {"/zip/llama.cpp/ggml-metal.h", "ggml-metal.h"},
    {"/zip/llama.cpp/ggml-quants.h", "ggml-quants.h"},
    {"/zip/llama.cpp/ggml-metal.metal", "ggml-metal.metal"},
    {"/zip/llama.cpp/ggml-metal.m", "ggml-metal.m"}, // must come last
};

static bool ImportMetalImpl(void) {

    // extract source code
    char src[PATH_MAX];
    bool needs_rebuild = false;
    for (int i = 0; i < sizeof(srcs) / sizeof(*srcs); ++i) {
        GetAppDir(src);
        if (!i && mkdir(src, 0755) && errno != EEXIST) {
            perror(src);
            return false;
        }
        strlcat(src, srcs[i].name, sizeof(src));
        if (IsNewerThan(srcs[i].zip, src)) {
            needs_rebuild = true;
            if (!ggml_extract(srcs[i].zip, src)) {
                return false;
            }
        }
    }

    // compile dynamic shared object
    char dso[PATH_MAX];
    GetAppDir(dso);
    strlcat(dso, "ggml-metal.dylib", sizeof(dso));
    if (needs_rebuild || IsNewerThan(src, dso)) {
        tinyprint(2, "building ggml-metal.dylib with xcode...\n", NULL);
        int fd;
        char tmpdso[PATH_MAX];
        strlcpy(tmpdso, dso, sizeof(tmpdso));
        strlcat(tmpdso, ".XXXXXX", sizeof(tmpdso));
        if ((fd = mkstemp(tmpdso)) != -1) {
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
    if (ImportMetalImpl()) {
        ggml_metal.supported = true;
    } else {
        tinyprint(2, "couldn't import metal gpu support\n", NULL);
    }
}

bool ggml_metal_supported(void) {
    if (!IsXnuSilicon()) return false;
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
        GetAppDir(path);
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
