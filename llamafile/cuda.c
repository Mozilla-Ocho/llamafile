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
#include <time.h>
#include <dlfcn.h>
#include <errno.h>
#include <spawn.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <limits.h>
#include <signal.h>
#include <libgen.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <stdatomic.h>
#include "llamafile/log.h"
#include "llama.cpp/ggml-cuda.h"
#include "llama.cpp/ggml-backend-impl.h"

__static_yoink("llama.cpp/ggml.h");
__static_yoink("llamafile/compcap.cu");
__static_yoink("llamafile/tinyblas.h");
__static_yoink("llamafile/tinyblas.cu");
__static_yoink("llama.cpp/ggml-impl.h");
__static_yoink("llamafile/llamafile.h");
__static_yoink("llama.cpp/ggml-cuda.h");
__static_yoink("llama.cpp/ggml-alloc.h");
__static_yoink("llama.cpp/ggml-cuda.cu");
__static_yoink("llama.cpp/ggml-backend.h");
__static_yoink("llama.cpp/ggml-backend-impl.h");

#define THESTRING(x) #x
#define STRINGIFY(x) THESTRING(x)
#define WIND_ONLY(x) (!IsWindows() ? "-DIGNORE" STRINGIFY(__COUNTER__) : x)
#define ARMS_ONLY(x) (!IsAarch64() ? "-DIGNORE" STRINGIFY(__COUNTER__) : x)
#define BLAS_ONLY(x) (FLAG_tinyblas ? "-DIGNORE" STRINGIFY(__COUNTER__) : x)

#define NVCC_LIBS BLAS_ONLY("-lcublas"), "-lcuda"

#define NVCC_FLAGS "--shared",                                          \
        "--forward-unknown-to-host-compiler",                           \
        "-use_fast_math",                                               \
        "--compiler-options",                                           \
        (!IsWindows()                                                   \
         ? (!IsAarch64()                                                \
            ? "-fPIC -O3 -march=native -mtune=native"                   \
            : "-fPIC -O3 -march=native -mtune=native -ffixed-x28")      \
         : "/nologo /EHsc /O2 /GR /MT"),                                \
        "-DNDEBUG",                                                     \
        "-DGGML_BUILD=1",                                               \
        "-DGGML_SHARED=1",                                              \
        "-DGGML_CUDA_DMMV_X=32",                                        \
        "-DGGML_CUDA_MMV_Y=1",                                          \
        "-DK_QUANTS_PER_ITERATION=2",                                   \
        "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128",                          \
        (FLAG_tinyblas                                                  \
         ? "-DGGML_USE_TINYBLAS"                                        \
         : "-DGGML_USE_CUBLAS")

int ggml_backend_cuda_reg_devices(void);

static const struct Source {
    const char *zip;
    const char *name;
} srcs[] = {
    {"/zip/llama.cpp/ggml.h", "ggml.h"},
    {"/zip/llamafile/compcap.cu", "compcap.cu"},
    {"/zip/llamafile/llamafile.h", "llamafile.h"},
    {"/zip/llamafile/tinyblas.h", "tinyblas.h"},
    {"/zip/llamafile/tinyblas.cu", "tinyblas.cu"},
    {"/zip/llama.cpp/ggml-impl.h", "ggml-impl.h"},
    {"/zip/llama.cpp/ggml-cuda.h", "ggml-cuda.h"},
    {"/zip/llama.cpp/ggml-alloc.h", "ggml-alloc.h"},
    {"/zip/llama.cpp/ggml-backend.h", "ggml-backend.h"},
    {"/zip/llama.cpp/ggml-backend-impl.h", "ggml-backend-impl.h"},
    {"/zip/llama.cpp/ggml-cuda.cu", "ggml-cuda.cu"}, // must come last
};

static struct Cuda {
    bool supported;
    atomic_uint once;
    typeof(ggml_cuda_link) *ggml_cuda_link;
    typeof(ggml_init_cublas) *ggml_init_cublas;
    typeof(ggml_cublas_loaded) *ggml_cublas_loaded;
    typeof(ggml_cuda_host_free) *ggml_cuda_host_free;
    typeof(ggml_cuda_host_malloc) *ggml_cuda_host_malloc;
    typeof(ggml_cuda_can_mul_mat) *can_mul_mat;
    typeof(ggml_cuda_set_tensor_split) *set_tensor_split;
    typeof(ggml_cuda_transform_tensor) *ggml_cuda_transform_tensor;
    typeof(ggml_cuda_free_data) *ggml_cuda_free_data;
    typeof(ggml_cuda_assign_buffers) *assign_buffers;
    typeof(ggml_cuda_assign_buffers_no_scratch) *assign_buffers_no_scratch;
    typeof(ggml_cuda_assign_buffers_force_inplace) *assign_buffers_force_inplace;
    typeof(ggml_cuda_assign_buffers_no_alloc) *assign_buffers_no_alloc;
    typeof(ggml_cuda_assign_scratch_offset) *assign_scratch_offset;
    typeof(ggml_cuda_copy_to_device) *copy_to_device;
    typeof(ggml_cuda_set_main_device) *set_main_device;
    typeof(ggml_cuda_set_scratch_size) *set_scratch_size;
    typeof(ggml_cuda_free_scratch) *free_scratch;
    typeof(ggml_cuda_compute_forward) *compute_forward;
    typeof(ggml_cuda_get_device_count) *get_device_count;
    typeof(ggml_cuda_get_device_description) *get_device_description;
    typeof(ggml_backend_cuda_buffer_type) *ggml_backend_cuda_buffer_type;
    typeof(ggml_backend_reg_cuda_init) *backend_reg_init;
    typeof(ggml_backend_cuda_host_buffer_type) *ggml_backend_cuda_host_buffer_type;
    typeof(ggml_backend_cuda_init) *ggml_backend_cuda_init;
} ggml_cuda;

static const char *Dlerror(void) {
    const char *msg;
    msg = cosmo_dlerror();
    if (!msg) msg = "null dlopen error";
    return msg;
}

static const char *GetDsoExtension(void) {
    if (IsWindows()) {
        return "dll";
    } else if (IsXnu()) {
        return "dylib";
    } else {
        return "so";
    }
}

static bool FileExists(const char *path) {
    struct stat st;
    return !stat(path, &st);
}

static bool IsExecutable(const char *path) {
    struct stat st;
    return !stat(path, &st) &&
            (st.st_mode & 0111) &&
            !S_ISDIR(st.st_mode);
}

static bool CreateTempPath(const char *path, char tmp[static PATH_MAX]) {
    int fd;
    strlcpy(tmp, path, PATH_MAX);
    strlcat(tmp, ".XXXXXX", PATH_MAX);
    if ((fd = mkostemp(tmp, O_CLOEXEC)) != -1) {
        close(fd);
        return true;
    } else {
        perror(tmp);
        return false;
    }
}

static void LogCommand(char *args[]) {
    for (int i = 0; args[i]; ++i) {
        if (i) {
            tinylog(" ", NULL);
        }
        // this quoting should be close enough to correct to be
        // copy/pastable on both unix and windows command terms
        bool need_quotes = !!strchr(args[i], ' ');
        if (need_quotes) {
            tinylog("\"", NULL);
        }
        tinylog(args[i], NULL);
        if (need_quotes) {
            tinylog("\"", NULL);
        }
    }
    tinylog("\n", NULL);
}

static bool Compile(const char *src,
                    const char *tmp,
                    const char *out,
                    char *args[]) {
    int pid, ws;
    LogCommand(args);
    errno_t err = posix_spawnp(&pid, args[0], NULL, NULL, args, environ);
    if (err) {
        perror(args[0]);
        unlink(tmp);
        return false;
    }
    while (waitpid(pid, &ws, 0) == -1) {
        if (errno != EINTR) {
            perror(args[0]);
            unlink(tmp);
            return false;
        }
    }
    if (ws) {
        tinylog(args[0], ": returned nonzero exit status\n", NULL);
        unlink(tmp);
        return false;
    }
    if (rename(tmp, out)) {
        perror(out);
        unlink(tmp);
        return false;
    }
    return true;
}

static bool GetRocmBinPath(char path[static PATH_MAX], const char *bin) {
    const char *hip_path = getenv("HIP_PATH");
    if (!hip_path) return false;
    strlcpy(path, hip_path, PATH_MAX);
    strlcat(path, "/bin/", PATH_MAX);
    strlcat(path, bin, PATH_MAX);
    if (IsWindows()) {
        strlcat(path, ".exe", PATH_MAX);
    }
    return FileExists(path);
}

// Returns word-encoded array of 16-bit gfxXXXX gcnArchName numbers.
static bool GetAmdOffloadArchFlag(char out[static 64]) {

    // Get hipInfo executable path.
    char hip_info_path[PATH_MAX];
    if (!GetRocmBinPath(hip_info_path, "hipInfo")) {
        return false;
    }

    // Create pipe.
    int pipefds[2];
    if (pipe2(pipefds, O_CLOEXEC)) {
        perror("pipe2");
        return false;
    }

    // Run HIP info program.
    int pid;
    char *args[] = {hip_info_path, 0};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, pipefds[1], 1);
    errno_t err = posix_spawn(&pid, args[0], &fa, NULL, args, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(pipefds[1]);
    if (err) {
        errno = err;
        perror(args[0]);
        close(pipefds[0]);
        return false;
    }

    // Parse program output to word-encoded array.
    int rc;
    int a = 0;
    int t = 0;
    char buf[512];
    unsigned long archs = 0;
    while ((rc = read(pipefds[0], buf, sizeof(buf))) > 0) {
        for (int i = 0; i < rc; ++i) {
            switch (t) {
                case 0:
                    if (buf[i] == 'g') {
                        t = 1;
                    }
                    break;
                case 1:
                    if (buf[i] == 'f') {
                        t = 2;
                    } else {
                        t = 0;
                    }
                    break;
                case 2:
                    if (buf[i] == 'x') {
                        t = 3;
                        a = 0;
                    } else {
                        t = 0;
                    }
                    break;
                case 3:
                    if (isdigit(buf[i])) {
                        a *= 10;
                        a += buf[i] - '0';
                    } else {
                        t = 0;
                        if ((a & 0xffff) && (a & 0xffff) == a) {
                            a &= 0xffff;
                            bool dupe = false;
                            for (int j = 0; j < 4; ++j) {
                                if (((archs >> (j * 16)) & 0xffff) == a) {
                                    dupe = true;
                                }
                            }
                            if (!dupe) {
                                archs <<= 16;
                                archs |= a;
                            }
                        }
                    }
                    break;
                default:
                    __builtin_unreachable();
            }
        }
    }
    close(pipefds[0]);

    // Wait for program to exit.
    int ws;
    while (waitpid(pid, &ws, 0) == -1) {
        if (errno != EINTR) {
            perror(args[0]);
            return false;
        }
    }
    if (ws) {
        tinylog("error: hipInfo returned non-zero exit status\n", NULL);
        return false;
    }

    // Serialize value for --offload-arch=LIST flag.
    if (!archs) {
        tinylog("warning: hipInfo output didn't list any graphics cards\n", NULL);
        return false;
    }
    bool gotsome = false;
    char *p = stpcpy(out, "--offload-arch=");
    do {
        if (gotsome) *p++ = ',';
        p += sprintf(p, "gfx%d", archs & 0xffff);
        gotsome = true;
    } while ((archs >>= 16));

    // woot
    return true;
}

// finds nvidia compiler
//
//   1. nvcc on $PATH environ
//   2. $CUDA_PATH/bin/nvcc
//   3. /opt/cuda/bin/nvcc
//   4. /usr/local/cuda/bin/nvcc
//
// set $CUDA_PATH to empty string to disable cuda
static bool GetNvccPath(char path[static PATH_MAX]) {
    const char *cuda_path;
    if (commandv(IsWindows() ? "nvcc.exe" : "nvcc", path, PATH_MAX)) {
        return true;
    } else if ((cuda_path = getenv("CUDA_PATH"))) {
        if (!*cuda_path) return false;
        strlcpy(path, cuda_path, PATH_MAX);
        strlcat(path, "/bin/", PATH_MAX);
    } else if (FileExists("/opt/cuda")) {
        strlcpy(path, "/opt/cuda/bin/", PATH_MAX);
    } else {
        strlcpy(path, "/usr/local/cuda/bin/", PATH_MAX);
    }
    strlcat(path, "nvcc", PATH_MAX);
    if (IsWindows()) {
        strlcat(path, ".exe", PATH_MAX);
    }
    return IsExecutable(path);
}

static dontinline bool GetNvccArchFlag(const char *nvcc, char flag[static 32]) {

    // create path of exe
    char exe[PATH_MAX];
    llamafile_get_app_dir(exe, sizeof(exe));
    strlcat(exe, "compcap", PATH_MAX);

    // get path of sauce
    char src[PATH_MAX];
    strlcpy(src, exe, PATH_MAX);
    strlcat(src, ".cu", PATH_MAX);

    // create temporary path for output
    char tmp[PATH_MAX];
    if (!CreateTempPath(exe, tmp)) {
        return false;
    }

    // build nvidia compute capability detector
    // https://stackoverflow.com/a/40695640/1653720
    //
    // Ideally we would just say `nvcc -arch=native`, but that flag is
    // not available on devices like Jetson. If we omit the flag, then
    // llm inference prints gibberish. In order to run nvcc we need to
    // detect the microarchitecture version of the host gpu.
    //
    // nvidia-smi in cuda 11.5+ can do this but (1) it takes longer to
    // run than compiling / running this script and (2) the nvidia-smi
    // command isn't available on Jetson devices.
    tinylog("building nvidia compute capability detector...\n", NULL);
    if (!Compile(src, tmp, exe, (char *[]){(char *)nvcc, "-o", tmp, src, 0})) {
        return false;
    }

    // create pipe
    int pipefds[2];
    if (pipe2(pipefds, O_CLOEXEC)) {
        perror("pipe2");
        return false;
    }

    // run nvidia compute capability detector
    int pid;
    char *args[] = {exe, 0};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, pipefds[1], 1);
    errno_t err = posix_spawn(&pid, args[0], &fa, NULL, args, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(pipefds[1]);
    if (err) {
        errno = err;
        perror(args[0]);
        close(pipefds[0]);
        return false;
    }
    char ibuf[12] = {0};
    read(pipefds[0], ibuf, 11);
    close(pipefds[0]);
    int ws;
    while (waitpid(pid, &ws, 0) == -1) {
        if (errno != EINTR) {
            perror(args[0]);
            return false;
        }
    }
    if (ws) {
        tinylog("error: compute capability detector returned nonzero exit status\n", NULL);
        return false;
    }

    // parse output of detector
    char *endptr;
    if (!*ibuf || !strtol(ibuf, &endptr, 10) || *endptr) {
        tinylog("error: bad compute capability detector output\n", NULL);
        return false;
    }

    // return resulting flag
    stpcpy(stpcpy(flag, "-arch=compute_"), ibuf);
    return true;
}

static bool CompileAmd(const char *clangxx, const char *dso, const char *src) {
    const char *lib = IsWindows() ? "lib" : GetDsoExtension();
    const char *hip_path = getenv("HIP_PATH");

    // get set of microarchitectures for all installed graphics cards
    char offload_arch[64];
    if (!GetAmdOffloadArchFlag(offload_arch)) {
        return false;
    }

    // create temporary output path for atomicity
    char tmpdso[PATH_MAX];
    if (!CreateTempPath(dso, tmpdso)) {
        return false;
    }

    // run the compiler to create a native build
    //
    // there's a higher level program called hipcc, but we can't use it,
    // since it's a perl script and rocm doesn't bundle perl on windows.
    //
    // TODO(jart): test this on linux computer
    if (Compile(src, tmpdso, dso,
                (char *[]){
                    (char *)clangxx,
                    "-fuse-ld=lld",
                    "-shared",
                    "-nostartfiles",
                    "-nostdlib",
                    "-DGGML_BUILD=1",
                    "-DGGML_SHARED=1",
                    "-Wno-ignored-attributes",
                    "-DGGML_CUDA_DMMV_X=32",
                    "-DGGML_CUDA_MMV_Y=1",
                    "-DGGML_USE_HIPBLAS",
                    (FLAG_tinyblas
                     ? "-DGGML_USE_TINYBLAS"
                     : "-DIGNORE"),
                    "-DK_QUANTS_PER_ITERATION=2",
                    "-D_CRT_SECURE_NO_WARNINGS",
                    "-D_XOPEN_SOURCE=600",
                    "-D__HIP_PLATFORM_AMD__=1",
                    "-D__HIP_PLATFORM_HCC__=1",
                    "-isystem", _gc(xasprintf("%s/include", hip_path)),
                    "-O3",
                    "-DNDEBUG",
                    "-D_DLL",
                    "-D_MT",
                    WIND_ONLY("-Xclang"), WIND_ONLY("--dependent-lib=msvcrt"),
                    ARMS_ONLY("-ffixed-x28"),
                    "-std=gnu++14",
                    "-mllvm", "-amdgpu-early-inline-all=true",
                    "-mllvm", "-amdgpu-function-calls=false",
                    "-x", "hip",
                    "--hip-link",
                    (char *)offload_arch,
                    "-o", tmpdso,
                    (char *)src,
                    BLAS_ONLY("-l"), BLAS_ONLY(_gc(xasprintf("%s/lib/hipblas.%s", hip_path, lib))),
                    BLAS_ONLY("-l"), BLAS_ONLY(_gc(xasprintf("%s/lib/rocblas.%s", hip_path, lib))),
                    "-l", _gc(xasprintf("%s/lib/amdhip64.%s", hip_path, lib)),
                    WIND_ONLY("-lkernel32"),
                    0})) {
        return true;
    }

    // oh no
    return false;
}

static bool CompileNvidia(const char *nvcc, const char *dso, const char *src) {

    // create temporary output path for atomicity
    char tmpdso[PATH_MAX];
    if (!CreateTempPath(dso, tmpdso)) {
        return false;
    }

    // try building dso with host nvidia microarchitecture
    tinylog("building ggml-cuda with nvcc -arch=native...\n", NULL);
    if (Compile(src, tmpdso, dso, (char *[]){
                (char *)nvcc, "-arch=native", NVCC_FLAGS, "-o", tmpdso,
                (char *)src, NVCC_LIBS, NULL})) {
        return true;
    }

    // try again with different arch flag
    char archflag[32];
    if (!GetNvccArchFlag(nvcc, archflag)) {
        return false;
    }
    tinylog("building ggml-cuda with nvcc ", archflag, "...\n", NULL);
    if (Compile(src, tmpdso, dso, (char *[]){
                (char *)nvcc, archflag, NVCC_FLAGS, "-o", tmpdso,
                (char *)src, NVCC_LIBS, NULL})) {
        return true;
    }

    // oh no
    return false;
}

static bool ExtractCudaDso(const char *dso, const char *name) {

    // see if prebuilt dso is bundled in zip assets
    char zip[80];
    strlcpy(zip, "/zip/", sizeof(zip));
    strlcat(zip, name, sizeof(zip));
    strlcat(zip, ".", sizeof(zip));
    strlcat(zip, GetDsoExtension(), sizeof(zip));
    if (!FileExists(zip)) {
        tinylog("prebuilt binary ", zip, " not found\n", NULL);
        return false;
    }

    // extract prebuilt dso
    return llamafile_extract(zip, dso);
}

static bool LinkCudaDso(const char *dso, const char *dir) {

    // Change directory so BLAS library is more likely to be linked.
    char cwd[PATH_MAX];
    if (dir) {
        getcwd(cwd, sizeof(cwd));
        chdir(dir);
    }

    // runtime link dynamic shared object
    void *lib;
    tinylog("dynamically linking ", dso, "\n", NULL);
    lib = cosmo_dlopen(dso, RTLD_LAZY);
    if (dir) {
        chdir(cwd);
    }
    if (!lib) {
        char cc[PATH_MAX];
        tinylog(Dlerror(), ": failed to load library\n", NULL);
        if ((IsLinux() || IsBsd()) && !commandv("cc", cc, PATH_MAX)) {
            tinylog("you need to install cc for gpu support\n", NULL);
        }
        return false;
    }

    // import functions
    bool ok = true;
    ok &= !!(ggml_cuda.ggml_cuda_link = cosmo_dlsym(lib, "ggml_cuda_link"));
    ok &= !!(ggml_cuda.ggml_init_cublas = cosmo_dlsym(lib, "ggml_init_cublas"));
    ok &= !!(ggml_cuda.ggml_cublas_loaded = cosmo_dlsym(lib, "ggml_cublas_loaded"));
    ok &= !!(ggml_cuda.ggml_cuda_host_free = cosmo_dlsym(lib, "ggml_cuda_host_free"));
    ok &= !!(ggml_cuda.ggml_cuda_host_malloc = cosmo_dlsym(lib, "ggml_cuda_host_malloc"));
    ok &= !!(ggml_cuda.can_mul_mat = cosmo_dlsym(lib, "ggml_cuda_can_mul_mat"));
    ok &= !!(ggml_cuda.set_tensor_split = cosmo_dlsym(lib, "ggml_cuda_set_tensor_split"));
    ok &= !!(ggml_cuda.ggml_cuda_transform_tensor = cosmo_dlsym(lib, "ggml_cuda_transform_tensor"));
    ok &= !!(ggml_cuda.ggml_cuda_free_data = cosmo_dlsym(lib, "ggml_cuda_free_data"));
    ok &= !!(ggml_cuda.assign_buffers = cosmo_dlsym(lib, "ggml_cuda_assign_buffers"));
    ok &= !!(ggml_cuda.assign_buffers_no_scratch = cosmo_dlsym(lib, "ggml_cuda_assign_buffers_no_scratch"));
    ok &= !!(ggml_cuda.assign_buffers_force_inplace = cosmo_dlsym(lib, "ggml_cuda_assign_buffers_force_inplace"));
    ok &= !!(ggml_cuda.assign_buffers_no_alloc = cosmo_dlsym(lib, "ggml_cuda_assign_buffers_no_alloc"));
    ok &= !!(ggml_cuda.assign_scratch_offset = cosmo_dlsym(lib, "ggml_cuda_assign_scratch_offset"));
    ok &= !!(ggml_cuda.copy_to_device = cosmo_dlsym(lib, "ggml_cuda_copy_to_device"));
    ok &= !!(ggml_cuda.set_main_device = cosmo_dlsym(lib, "ggml_cuda_set_main_device"));
    ok &= !!(ggml_cuda.set_scratch_size = cosmo_dlsym(lib, "ggml_cuda_set_scratch_size"));
    ok &= !!(ggml_cuda.free_scratch = cosmo_dlsym(lib, "ggml_cuda_free_scratch"));
    ok &= !!(ggml_cuda.compute_forward = cosmo_dlsym(lib, "ggml_cuda_compute_forward"));
    ok &= !!(ggml_cuda.get_device_count = cosmo_dlsym(lib, "ggml_cuda_get_device_count"));
    ok &= !!(ggml_cuda.get_device_description = cosmo_dlsym(lib, "ggml_cuda_get_device_description"));
    ok &= !!(ggml_cuda.backend_reg_init = cosmo_dlsym(lib, "ggml_backend_reg_cuda_init"));
    ok &= !!(ggml_cuda.ggml_backend_cuda_host_buffer_type = cosmo_dlsym(lib, "ggml_backend_cuda_host_buffer_type"));
    ok &= !!(ggml_cuda.ggml_backend_cuda_buffer_type = cosmo_dlsym(lib, "ggml_backend_cuda_buffer_type"));
    ok &= !!(ggml_cuda.ggml_backend_cuda_init = cosmo_dlsym(lib, "ggml_backend_cuda_init"));
    if (!ok) {
        tinylog("error: not all cuda symbols could be imported\n", NULL);
        return false;
    }

    // ask the library if actual gpu devices exist
    ggml_cuda.ggml_cuda_link(ggml_backend_api());
    return true;
}

static bool ImportCudaImpl(void) {

    // No dynamic linking support on OpenBSD yet.
    if (IsOpenbsd()) {
        return false;
    }

    // Check if we're allowed to even try.
    switch (FLAG_gpu) {
        case LLAMAFILE_GPU_AUTO:
        case LLAMAFILE_GPU_AMD:
        case LLAMAFILE_GPU_NVIDIA:
            break;
        default:
            return false;
    }
    tinylog("initializing gpu module...\n", NULL);

    // extract source code
    char src[PATH_MAX];
    bool needs_rebuild = FLAG_recompile;
    for (int i = 0; i < sizeof(srcs) / sizeof(*srcs); ++i) {
        llamafile_get_app_dir(src, sizeof(src));
        if (!i && mkdir(src, 0755) && errno != EEXIST) {
            perror(src);
            return false;
        }
        strlcat(src, srcs[i].name, sizeof(src));
        switch (llamafile_is_file_newer_than(srcs[i].zip, src)) {
            case -1:
                return false;
            case false:
                break;
            case true:
                needs_rebuild = true;
                if (!llamafile_extract(srcs[i].zip, src)) {
                    return false;
                }
                break;
            default:
                __builtin_unreachable();
        }
    }

    char dso[PATH_MAX];
    char bindir[PATH_MAX];
    const char *compiler_path;
    char compiler_path_buf[PATH_MAX];
    const char *library_path;
    char library_path_buf[PATH_MAX];

    // Attempt to load AMD GPU support.
    // We favor the underdog on AMD + NVIDIA chimeras.
    switch (FLAG_gpu) {
        case LLAMAFILE_GPU_AMD:
        case LLAMAFILE_GPU_AUTO:

            // Get some essential paths.
            // ROCm SDK puts BLAS DLLs in same folder as clang++
            if (GetRocmBinPath(compiler_path_buf, "clang++")) {
                strcpy(library_path_buf, compiler_path_buf);
                dirname(library_path_buf);
                compiler_path = compiler_path_buf;
                library_path = library_path_buf;
            } else {
                compiler_path = 0;
                library_path = 0;
            }

            // Get path of GGML DSO for AMD.
            llamafile_get_app_dir(dso, PATH_MAX);
            strlcat(dso, "ggml-rocm.", PATH_MAX);
            strlcat(dso, GetDsoExtension(), PATH_MAX);
            if (FLAG_nocompile) {
                if ((FileExists(dso) ||
                     ExtractCudaDso(dso, "ggml-rocm")) &&
                    LinkCudaDso(dso, library_path)) {
                    return true;
                } else {
                    goto TryNvidia;
                }
            }

            // Check if DSO is already compiled.
            if (!needs_rebuild && !FLAG_recompile) {
                switch (llamafile_is_file_newer_than(src, dso)) {
                    case -1:
                        return false;
                    case false:
                        if (LinkCudaDso(dso, library_path)) {
                            return true;
                        } else {
                            goto TryNvidia;
                        }
                    case true:
                        break;
                    default:
                        __builtin_unreachable();
                }
            }

            // Try building CUDA with ROCm SDK.
            if (compiler_path) {
                if (CompileAmd(compiler_path, dso, src)) {
                    if (LinkCudaDso(dso, library_path)) {
                        return true;
                    } else {
                        goto TryNvidia;
                    }
                }
            } else {
                tinylog("note: won't compile AMD GPU support because $HIP_PATH/bin/clang++ is missing\n", NULL);
            }

            // Try extracting prebuilt tinyBLAS DSO from PKZIP.
            if (ExtractCudaDso(dso, "ggml-rocm")) {
                if (LinkCudaDso(dso, library_path)) {
                    return true;
                } else {
                    goto TryNvidia;
                }
            }

            break;
        default:
            break;
    }

TryNvidia:
    // Attempt to load NVIDIA GPU support.
    switch (FLAG_gpu) {
        case LLAMAFILE_GPU_AUTO:
        case LLAMAFILE_GPU_NVIDIA:

            // Get some essential paths.
            // CUDA SDK puts cuBLAS DLL in same folder as NVCC
            if (GetNvccPath(compiler_path_buf)) {
                strcpy(library_path_buf, compiler_path_buf);
                dirname(library_path_buf);
                compiler_path = compiler_path_buf;
                library_path = library_path_buf;
            } else {
                compiler_path = 0;
                library_path = 0;
            }

            // Get path of GGML DSO for NVIDIA.
            llamafile_get_app_dir(dso, PATH_MAX);
            strlcat(dso, "ggml-cuda.", PATH_MAX);
            strlcat(dso, GetDsoExtension(), PATH_MAX);
            if (FLAG_nocompile) {
                return ((FileExists(dso) ||
                         ExtractCudaDso(dso, "ggml-cuda")) &&
                        LinkCudaDso(dso, library_path));
            }

            // Check if DSO is already compiled.
            if (!needs_rebuild && !FLAG_recompile) {
                switch (llamafile_is_file_newer_than(src, dso)) {
                    case -1:
                        return false;
                    case false:
                        return LinkCudaDso(dso, library_path);
                    case true:
                        break;
                    default:
                        __builtin_unreachable();
                }
            }

            // Try building CUDA from source with mighty cuBLAS.
            if (compiler_path && CompileNvidia(compiler_path, dso, src)) {
                return LinkCudaDso(dso, library_path);
            }

            // Try extracting prebuilt tinyBLAS DSO from PKZIP.
            if (ExtractCudaDso(dso, "ggml-cuda")) {
                return LinkCudaDso(dso, library_path);
            }

            break;
        default:
            break;
    }

    // too bad
    return false;
}

static void ImportCuda(void) {
    if (ImportCudaImpl()) {
        tinylog("GPU support successfully linked and loaded\n", NULL);
        ggml_cuda.supported = true;
    } else if (FLAG_gpu == LLAMAFILE_GPU_AMD ||
               FLAG_gpu == LLAMAFILE_GPU_NVIDIA) {
        tinyprint(2, "fatal error: support for --gpu ",
                  llamafile_describe_gpu(), FLAG_tinyblas ? " --tinyblas" : "",
                  " was explicitly requested, but it wasn't available\n", NULL);
        exit(1);
    }
}

bool ggml_cuda_supported(void) {
    cosmo_once(&ggml_cuda.once, ImportCuda);
    return ggml_cuda.supported;
}

void ggml_init_cublas(void) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.ggml_init_cublas();
}

bool ggml_cublas_loaded(void) {
    if (!ggml_cuda_supported()) return false;
    return ggml_cuda.ggml_cublas_loaded();
}

void *ggml_cuda_host_malloc(size_t n) {
    if (!ggml_cuda_supported()) return NULL;
    return ggml_cuda.ggml_cuda_host_malloc(n);
}

void ggml_cuda_host_free(void *data) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.ggml_cuda_host_free(data);
}

bool ggml_cuda_can_mul_mat(const struct ggml_tensor *src0,
                           const struct ggml_tensor *src1,
                           struct ggml_tensor *dst) {
    if (!ggml_cuda_supported()) return false;
    return ggml_cuda.can_mul_mat(src0, src1, dst);
}

void ggml_cuda_set_tensor_split(const float *tensor_split) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.set_tensor_split(tensor_split);
}

void ggml_cuda_transform_tensor(void *data, struct ggml_tensor *tensor) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.ggml_cuda_transform_tensor(data, tensor);
}

void ggml_cuda_free_data(struct ggml_tensor *tensor) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.ggml_cuda_free_data(tensor);
}

void ggml_cuda_assign_buffers(struct ggml_tensor *tensor) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.assign_buffers(tensor);
}

void ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor *tensor) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.assign_buffers_no_scratch(tensor);
}

void ggml_cuda_assign_buffers_force_inplace(struct ggml_tensor *tensor) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.assign_buffers_force_inplace(tensor);
}

void ggml_cuda_assign_buffers_no_alloc(struct ggml_tensor *tensor) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.assign_buffers_no_alloc(tensor);
}

void ggml_cuda_assign_scratch_offset(struct ggml_tensor *tensor, size_t offset) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.assign_scratch_offset(tensor, offset);
}

void ggml_cuda_copy_to_device(struct ggml_tensor *tensor) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.copy_to_device(tensor);
}

void ggml_cuda_set_main_device(int main_device) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.set_main_device(main_device);
}

void ggml_cuda_set_scratch_size(size_t scratch_size) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.set_scratch_size(scratch_size);
}

void ggml_cuda_free_scratch(void) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.free_scratch();
}

bool ggml_cuda_compute_forward(struct ggml_compute_params *params,
                               struct ggml_tensor *tensor) {
    if (!ggml_cuda_supported()) return false;
    return ggml_cuda.compute_forward(params, tensor);
}

int ggml_cuda_get_device_count(void) {
    if (!ggml_cuda_supported()) return 0;
    return ggml_cuda.get_device_count();
}

void ggml_cuda_get_device_description(int device,
                                      char *description,
                                      size_t description_size) {
    if (!ggml_cuda_supported()) return;
    return ggml_cuda.get_device_description(device, description,
                                            description_size);
}

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    if (!ggml_cuda_supported()) return 0;
    return ggml_cuda.ggml_backend_cuda_buffer_type(device);
}

ggml_backend_t ggml_backend_reg_cuda_init(const char * params, void * user_data) {
    if (!ggml_cuda_supported()) return 0;
    ggml_cuda.backend_reg_init(params, user_data);
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    if (!ggml_cuda_supported()) return 0;
    ggml_cuda.ggml_backend_cuda_host_buffer_type();
}

int ggml_backend_cuda_reg_devices(void) {
    int device_count = ggml_cuda_get_device_count();
    //int device_count = 1; // DEBUG: some tools require delaying CUDA initialization
    for (int i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_CUDA_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_cuda_init, ggml_backend_cuda_buffer_type(i), (void *) (intptr_t) i);
    }
    return device_count;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (!ggml_cuda_supported()) return 0;
    return ggml_cuda.ggml_backend_cuda_init(device);
}
