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

#include "llama.cpp/ggml-backend-impl.h"
#include "llama.cpp/ggml-cuda.h"
#include "llama.cpp/ggml-metal.h"
#include "llamafile/llamafile.h"
#include "llamafile/log.h"
#include "llamafile/x.h"
#include <assert.h>
#include <cosmo.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

__static_yoink("llama.cpp/ggml.h");
__static_yoink("llamafile/compcap.cu");
__static_yoink("llamafile/tinyblas.h");
__static_yoink("llamafile/tinyblas.cu");
__static_yoink("llama.cpp/ggml-impl.h");
__static_yoink("llamafile/llamafile.h");
__static_yoink("llama.cpp/ggml-cuda.h");
__static_yoink("llama.cpp/ggml-alloc.h");
__static_yoink("llama.cpp/ggml-cuda.cu");
__static_yoink("llama.cpp/ggml-common.h");
__static_yoink("llama.cpp/ggml-backend.h");
__static_yoink("llama.cpp/ggml-backend-impl.h");

// yoink the fastest zlib deflate impl from cosmo libc
__static_yoink("_Cz_inflateInit2");
__static_yoink("_Cz_inflate");
__static_yoink("_Cz_inflateEnd");

#define THESTRING(x) #x
#define STRINGIFY(x) THESTRING(x)
#define ARMS_ONLY(x) (!IsAarch64() ? "-DIGNORE" STRINGIFY(__COUNTER__) : x)
#define BLAS_ONLY(x) (FLAG_tinyblas ? "-DIGNORE" STRINGIFY(__COUNTER__) : x)

#define NVCC_LIBS BLAS_ONLY("-lcublas"), "-lcuda"

#define COMMON_FLAGS \
    "-DNDEBUG", "-DGGML_BUILD=1", "-DGGML_SHARED=1", "-DGGML_MULTIPLATFORM", \
        "-DGGML_CUDA_DMMV_X=32", "-DK_QUANTS_PER_ITERATION=2", \
        "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128", "-DGGML_CUDA_MMV_Y=1", \
        (FLAG_tinyblas ? "-DGGML_USE_TINYBLAS" : "-DGGML_USE_CUBLAS"), \
        (FLAG_iq || FLAG_flash_attn ? "-DTEHFLASH" : "-DGGML_MINIMIZE_CODE_SIZE")

#define NVCC_FLAGS \
    (!IsWindows() ? "-std=c++11" : "-DIGNORE123"), "-O3", "--shared", "--use_fast_math", \
        "-Xcudafe", "--diag_suppress=177", "-Xcudafe", "--diag_suppress=940", "-Xcudafe", \
        "--diag_suppress=1305", "--forward-unknown-to-host-compiler", "--compiler-options", \
        (!IsWindows() \
             ? (!IsAarch64() \
                    ? "-fPIC -O3 -march=native -mtune=native -std=c++11 -Wno-unused-function " \
                      "-Wno-unused-result -Wno-return-type -Wno-pedantic" \
                    : "-fPIC -O3 -march=native -mtune=native -std=c++11 -Wno-unused-function " \
                      "-Wno-unused-result -Wno-return-type -Wno-pedantic -ffixed-x28") \
             : "/nologo /EHsc /O2 /GR /MT"), \
        COMMON_FLAGS

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
    {"/zip/llama.cpp/ggml-common.h", "ggml-common.h"},
    {"/zip/llama.cpp/ggml-backend.h", "ggml-backend.h"},
    {"/zip/llama.cpp/ggml-backend-impl.h", "ggml-backend-impl.h"},
    {"/zip/llama.cpp/ggml-cuda.cu", "ggml-cuda.cu"}, // must come last
};

static struct Cuda {
    bool supported;
    bool has_amd_gpu;
    atomic_uint once;
    typeof(ggml_cuda_link) *GGML_CALL link;
    typeof(ggml_backend_cuda_buffer_type) *GGML_CALL buffer_type;
    typeof(ggml_backend_cuda_host_buffer_type) *GGML_CALL host_buffer_type;
    typeof(ggml_backend_cuda_init) *GGML_CALL backend_init;
    typeof(ggml_backend_cuda_split_buffer_type) *GGML_CALL split_buffer_type;
    typeof(ggml_backend_cuda_reg_devices) *GGML_CALL reg_devices;
    typeof(ggml_backend_cuda_get_device_properties) *GGML_CALL get_device_properties;
    typeof(ggml_backend_cuda_get_device_memory) *GGML_CALL get_device_memory;
    typeof(ggml_backend_cuda_get_device_count) *GGML_CALL get_device_count;
    typeof(ggml_backend_cuda_unregister_host_buffer) *GGML_CALL unreg_host_buf;
    typeof(ggml_backend_cuda_register_host_buffer) *GGML_CALL register_host_buffer;
    typeof(ggml_backend_cuda_get_device_description) *GGML_CALL get_description;
} ggml_cuda;

static const char *Dlerror(void) {
    const char *msg;
    msg = cosmo_dlerror();
    if (!msg)
        msg = "null dlopen error";
    return msg;
}

static const char *GetDsoExtension(void) {
    if (IsWindows())
        return "dll";
    else if (IsXnu())
        return "dylib";
    else
        return "so";
}

static bool FileExists(const char *path) {
    struct stat st;
    return !stat(path, &st);
}

static bool IsExecutable(const char *path) {
    struct stat st;
    return !stat(path, &st) && (st.st_mode & 0111) && !S_ISDIR(st.st_mode);
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

static bool Compile(const char *src, const char *tmp, const char *out, char *args[]) {
    int pid, ws;
    llamafile_log_command(args);
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
        tinylog(__func__, ": warning: ", args[0], " returned nonzero exit status\n", NULL);
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

static bool get_rocm_bin_path(char path[static PATH_MAX], const char *bin) {

    // create filename of executable
    char name[NAME_MAX];
    strlcpy(name, bin, PATH_MAX);
    if (IsWindows())
        strlcat(name, ".exe", PATH_MAX);

    // search for it on $PATH
    if (commandv(name, path, PATH_MAX))
        return path;
    else
        tinylog(__func__, ": note: ", name, " not found on $PATH\n", NULL);

    // 1. use $HIP_PATH/bin/$name if it exists
    // 2. use /opt/rocm/bin/$name if it exists
    const char *hip_path = getenv("HIP_PATH");
    if (!hip_path) {
        tinylog(__func__, ": note: $HIP_PATH/bin/", name, " does not exist\n", NULL);
        if (!FileExists((hip_path = "/opt/rocm"))) {
            tinylog(__func__, ": note: /opt/rocm/bin/", name, " does not exist\n", NULL);
            return false;
        }
    }
    strlcpy(path, hip_path, PATH_MAX);
    strlcat(path, "/bin/", PATH_MAX);
    strlcat(path, name, PATH_MAX);
    if (FileExists(path)) {
        return true;
    } else {
        tinylog(__func__, ": note: ", path, " does not exist\n", NULL);
        return false;
    }
}

struct StringListEntry {
    char *string;
    struct StringListEntry *next;
};

struct StringList {
    struct StringListEntry *head;
    struct StringListEntry *tail;
    size_t length;
};

static int StringCompare(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

static void AddStringToStringList(struct StringList *string_list, char *string,
                                  size_t string_length) {
    struct StringListEntry *new_entry = malloc(sizeof(struct StringListEntry));
    new_entry->string = malloc(sizeof(char) * (string_length + 1));
    strncpy(new_entry->string, string, string_length);
    new_entry->string[string_length] = '\0';
    new_entry->next = NULL;
    if (string_list->head == NULL) {
        string_list->head = new_entry;
    } else {
        string_list->tail->next = new_entry;
    }
    string_list->tail = new_entry;
    ++string_list->length;
}

static void CopyStringListToStringArray(struct StringList *string_list, char ***string_array) {
    *string_array = malloc(sizeof(char *) * string_list->length);
    struct StringListEntry *current = string_list->head;
    int i = 0;
    while (current != NULL) {
        (*string_array)[i] = current->string;
        current = current->next;
        ++i;
    }
}

static int RemoveDuplicatesFromSortedStringArray(char **strings, int num_strings) {
    if (num_strings == 1)
        return 1;
    int tail = 0;
    for (int current = 1; current < num_strings; ++current) {
        if (strcmp(strings[tail], strings[current]) != 0) {
            strings[tail + 1] = strings[current];
            ++tail;
        } else {
            strings[current] = NULL;
        }
    }
    return tail + 1;
}

static void FreeStringList(struct StringList *string_list) {
    struct StringListEntry *current = string_list->head;
    while (current != NULL) {
        struct StringListEntry *next = current->next;
        free(current->string);
        free(current);
        current = next;
    }
}

// Returns word-encoded array of 16-bit gfxXXXX gcnArchName numbers.
static bool get_amd_offload_arch_flag(char out[static 64]) {

    // Get hipInfo executable path.
    char hip_info_path[PATH_MAX];
    if (!get_rocm_bin_path(hip_info_path, "hipInfo") &&
        !get_rocm_bin_path(hip_info_path, "rocminfo")) {
        tinylog(__func__,
                ": warning: can't find hipInfo/rocminfo commands for AMD GPU "
                "detection\n",
                NULL);
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
    llamafile_log_command(args);
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
    int t = 0;
    char buf[512];
    char name[64] = {'g', 'f', 'x'};
    int j = 3;
    struct StringList name_list = {NULL, NULL, 0};
    while ((rc = read(pipefds[0], buf, sizeof(buf))) > 0) {
        for (int i = 0; i < rc; ++i) {
            switch (t) {
            case 0:
                if (buf[i] == 'g')
                    t = 1;
                break;
            case 1:
                if (buf[i] == 'f')
                    t = 2;
                else
                    t = 0;
                break;
            case 2:
                if (buf[i] == 'x') {
                    t = 3;
                } else {
                    t = 0;
                }
                break;
            case 3:
                if (isalnum(buf[i])) {
                    name[j] = buf[i];
                    ++j;
                } else {
                    AddStringToStringList(&name_list, name, j);
                    t = 0;
                    j = 3;
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
        tinylog(__func__, ": error: hipInfo returned non-zero exit status\n", NULL);
        return false;
    }

    // Serialize value for --offload-arch=LIST flag.
    if (name_list.length == 0) {
        tinylog(__func__, ": warning: hipInfo output didn't list any graphics cards\n", NULL);
        return false;
    }
    char *p = stpcpy(out, "--offload-arch=");
    char **names = NULL;

    CopyStringListToStringArray(&name_list, &names);
    qsort(names, name_list.length, sizeof(char *), StringCompare);

    int num_names = RemoveDuplicatesFromSortedStringArray(names, name_list.length);

    for (int i = 0; i < num_names; ++i) {
        if (i > 0)
            *p++ = ',';
        p += sprintf(p, "%s", names[i]);
    }

    FreeStringList(&name_list);
    free(names);

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
static bool get_nvcc_path(char path[static PATH_MAX]) {
    const char *name = IsWindows() ? "nvcc.exe" : "nvcc";
    if (commandv(name, path, PATH_MAX))
        return true;
    else
        tinylog(__func__, ": note: ", name, " not found on $PATH\n", NULL);
    const char *cuda_path;
    if ((cuda_path = getenv("CUDA_PATH"))) {
        if (!*cuda_path)
            return false;
        strlcpy(path, cuda_path, PATH_MAX);
        strlcat(path, "/bin/", PATH_MAX);
    } else {
        tinylog(__func__, ": note: $CUDA_PATH/bin/", name, " does not exist\n", NULL);
        if (FileExists("/opt/cuda")) {
            strlcpy(path, "/opt/cuda/bin/", PATH_MAX);
        } else {
            tinylog(__func__, ": note: /opt/cuda/bin/", name, " does not exist\n", NULL);
            strlcpy(path, "/usr/local/cuda/bin/", PATH_MAX);
        }
    }
    strlcat(path, name, PATH_MAX);
    if (IsExecutable(path)) {
        return true;
    } else {
        tinylog(__func__, ": note: ", path, " does not exist\n", NULL);
        return false;
    }
}

static dontinline bool get_nvcc_arch_flag(const char *nvcc, char flag[static 32]) {

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
    if (!CreateTempPath(exe, tmp))
        return false;

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
    tinylog(__func__, ": note: building nvidia compute capability detector...\n", NULL);
    if (!Compile(src, tmp, exe, (char *[]){(char *)nvcc, "-o", tmp, src, 0}))
        return false;

    // create pipe
    int pipefds[2];
    if (pipe2(pipefds, O_CLOEXEC)) {
        perror("pipe2");
        return false;
    }

    // run nvidia compute capability detector
    int pid;
    char *args[] = {exe, 0};
    llamafile_log_command(args);
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
        tinylog(__func__,
                ": error: compute capability detector returned nonzero exit "
                "status\n",
                NULL);
        return false;
    }

    // parse output of detector
    char *endptr;
    if (!*ibuf || !strtol(ibuf, &endptr, 10) || *endptr) {
        tinylog(__func__, ": error: bad compute capability detector output\n", NULL);
        return false;
    }

    // return resulting flag
    stpcpy(stpcpy(flag, "-arch=compute_"), ibuf);
    return true;
}

static bool compile_amd_windows(const char *clangxx, const char *dso, const char *src,
                                const char *tmpdso) {
    const char *lib = IsWindows() ? "lib" : GetDsoExtension();
    const char *hip_path = getenv("HIP_PATH");

    // get set of microarchitectures for all installed graphics cards
    char offload_arch[64];
    if (!get_amd_offload_arch_flag(offload_arch)) {
        unlink(tmpdso);
        return false;
    }

    // run the compiler to create a native build
    //
    // there's a higher level program called hipcc, but we can't use it,
    // since it's a perl script and rocm doesn't bundle perl on windows.
    char *args[] = {
        (char *)clangxx,
        "-O3",
        "-shared",
        "-x",
        "hip",
        "--hip-link",
        "-std=gnu++14",
        "-fuse-ld=lld",
        "-DGGML_USE_HIPBLAS",
        "-Wno-return-type",
        "-Wno-unused-result",
        "-Wno-unused-function",
        "-Wno-expansion-to-defined",
        (char *)offload_arch,
        "-Wno-ignored-attributes",
        "-D_CRT_SECURE_NO_WARNINGS",
        "-DGGML_BUILD=1",
        "-DGGML_SHARED=1",
        "-DGGML_MULTIPLATFORM",
        "-DGGML_CUDA_DMMV_X=32",
        "-DK_QUANTS_PER_ITERATION=2",
        "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128",
        "-DGGML_CUDA_MMV_Y=1",
        "-DGGML_USE_TINYBLAS",
        FLAG_iq || FLAG_flash_attn ? "-DTEHFLASH" : "-DGGML_MINIMIZE_CODE_SIZE",
        "-o",
        (char *)tmpdso,
        (char *)src,
        "-Xclang",
        "--dependent-lib=msvcrt",
        "-mllvm",
        "-amdgpu-function-calls=false",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-isystem",
        gc(xasprintf("%s/include", hip_path)),
        /* BLAS_ONLY("-l"), */
        /* BLAS_ONLY(gc(xasprintf("%s/lib/hipblas.%s", hip_path, lib))), */
        /* BLAS_ONLY("-l"), */
        /* BLAS_ONLY(gc(xasprintf("%s/lib/rocblas.%s", hip_path, lib))), */
        "-l",
        gc(xasprintf("%s/lib/amdhip64.%s", hip_path, lib)),
        "-lkernel32",
        NULL,
    };
    return Compile(src, tmpdso, dso, args);
}

static bool compile_amd_unix(const char *dso, const char *src, const char *tmpdso) {

    // get set of microarchitectures for all installed graphics cards
    // it's possible to safe --offload-arch=native but we do it ourself
    // the amdgpu-arch that hipcc runs fails to link libhsa-runtime64.so
    char offload_arch[64];
    if (!get_amd_offload_arch_flag(offload_arch))
        strcpy(offload_arch, "--offload-arch=native");

    char *args[] = {
        "hipcc",
        "-O3",
        "-fPIC",
        "-shared",
        offload_arch,
        "-march=native",
        "-mtune=native",
        "-DGGML_USE_HIPBLAS",
        "-Wno-return-type",
        "-Wno-unused-result",
        "-Wno-unused-function",
        "-Wno-expansion-to-defined",
        ARMS_ONLY("-ffixed-x28"),
        COMMON_FLAGS,
        "-o",
        (char *)tmpdso,
        (char *)src,
        BLAS_ONLY("-lhipblas"),
        BLAS_ONLY("-lrocblas"),
        NULL,
    };
    return Compile(src, tmpdso, dso, args);
}

static bool compile_amd(const char *clangxx, const char *dso, const char *src) {

    // create temporary output path for atomicity
    char tmpdso[PATH_MAX];
    if (!CreateTempPath(dso, tmpdso))
        return false;

    if (!IsWindows())
        return compile_amd_unix(dso, src, tmpdso);
    else
        return compile_amd_windows(clangxx, dso, src, tmpdso);
}

static bool compile_nvidia(const char *nvcc, const char *dso, const char *src) {

    // create temporary output path for atomicity
    char tmpdso[PATH_MAX];
    if (!CreateTempPath(dso, tmpdso))
        return false;

    // try building dso with host nvidia microarchitecture
    tinylog(__func__, ": note: building ggml-cuda with nvcc -arch=native...\n", NULL);
    if (Compile(src, tmpdso, dso,
                (char *[]){(char *)nvcc, "-arch=native", NVCC_FLAGS, "-o", tmpdso, (char *)src,
                           NVCC_LIBS, NULL}))
        return true;

    // try again with different arch flag
    char archflag[32];
    if (!get_nvcc_arch_flag(nvcc, archflag))
        return false;
    tinylog(__func__, ": note: building ggml-cuda with nvcc ", archflag, "...\n", NULL);
    if (Compile(src, tmpdso, dso,
                (char *[]){(char *)nvcc, archflag, NVCC_FLAGS, "-o", tmpdso, (char *)src, NVCC_LIBS,
                           NULL}))
        return true;

    // oh no
    return false;
}

static bool extract_cuda_dso(const char *dso, const char *name) {

    // see if prebuilt dso is bundled in zip assets
    char zip[80];
    strlcpy(zip, "/zip/", sizeof(zip));
    strlcat(zip, name, sizeof(zip));
    strlcat(zip, ".", sizeof(zip));
    strlcat(zip, GetDsoExtension(), sizeof(zip));
    if (!FileExists(zip)) {
        tinylog(__func__, ": note: prebuilt binary ", zip, " not found\n", NULL);
        return false;
    }

    // extract prebuilt dso
    return llamafile_extract(zip, dso);
}

static void *imp(void *lib, const char *sym) {
    void *fun = cosmo_dlsym(lib, sym);
    if (!fun)
        tinylog(__func__, ": error: failed to import symbol: ", sym, "\n", NULL);
    return fun;
}

static bool link_cuda_dso(const char *dso, const char *dir) {

    // Change directory so BLAS library is more likely to be linked.
    char cwd[PATH_MAX];
    if (dir) {
        getcwd(cwd, sizeof(cwd));
        chdir(dir);
    }

    // runtime link dynamic shared object
    void *lib;
    tinylog(__func__, ": note: dynamically linking ", dso, "\n", NULL);
    lib = cosmo_dlopen(dso, RTLD_LAZY);
    if (dir)
        chdir(cwd);
    if (!lib) {
        char cc[PATH_MAX];
        tinylog(__func__, ": warning: ", Dlerror(), ": failed to load library\n", NULL);
        if ((IsLinux() || IsBsd()) && !commandv("cc", cc, PATH_MAX))
            tinylog(__func__, ": note: you need to install cc for gpu support\n", NULL);
        return false;
    }

    // import functions
    bool ok = true;
    ok &= !!(ggml_cuda.link = imp(lib, "ggml_cuda_link"));
    ok &= !!(ggml_cuda.host_buffer_type = imp(lib, "ggml_backend_cuda_host_buffer_type"));
    ok &= !!(ggml_cuda.buffer_type = imp(lib, "ggml_backend_cuda_buffer_type"));
    ok &= !!(ggml_cuda.backend_init = imp(lib, "ggml_backend_cuda_init"));
    ok &= !!(ggml_cuda.split_buffer_type = imp(lib, "ggml_backend_cuda_split_buffer_type"));
    ok &= !!(ggml_cuda.reg_devices = imp(lib, "ggml_backend_cuda_reg_devices"));
    ok &= !!(ggml_cuda.get_device_properties= imp(lib, "ggml_backend_cuda_get_device_properties"));
    ok &= !!(ggml_cuda.get_device_memory = imp(lib, "ggml_backend_cuda_get_device_memory"));
    ok &= !!(ggml_cuda.get_device_count = imp(lib, "ggml_backend_cuda_get_device_count"));
    ok &= !!(ggml_cuda.unreg_host_buf = imp(lib, "ggml_backend_cuda_unregister_host_buffer"));
    ok &= !!(ggml_cuda.register_host_buffer = imp(lib, "ggml_backend_cuda_register_host_buffer"));
    ok &= !!(ggml_cuda.get_description = imp(lib, "ggml_backend_cuda_get_device_description"));
    if (!ok) {
        tinylog(__func__, ": error: not all cuda symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }

    // ask the library if actual gpu devices exist
    if (ggml_cuda.link(ggml_backend_api())) {
        tinylog(__func__, ": GPU support loaded\n", NULL);
        return true;
    } else {
        tinylog(__func__, ": No GPU devices found\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }
}

static bool import_cuda_impl(void) {

    // No dynamic linking support on OpenBSD yet.
    if (IsOpenbsd())
        return false;

    // Check if we're allowed to even try.
    switch (FLAG_gpu) {
    case LLAMAFILE_GPU_AUTO:
    case LLAMAFILE_GPU_AMD:
    case LLAMAFILE_GPU_NVIDIA:
        break;
    default:
        return false;
    }

    tinylog(__func__, ": initializing gpu module...\n", NULL);

    npassert(FLAGS_READY);

    // extract source code
    char src[PATH_MAX];
    bool needs_rebuild = FLAG_recompile;
    for (int i = 0; i < sizeof(srcs) / sizeof(*srcs); ++i) {
        llamafile_get_app_dir(src, sizeof(src));
        if (!i && makedirs(src, 0755)) {
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
            if (!llamafile_extract(srcs[i].zip, src))
                return false;
            break;
        default:
            __builtin_unreachable();
        }
    }

    char dso[PATH_MAX];
    char bindir[PATH_MAX];
    const char *compiler_path = NULL;
    char compiler_path_buf[PATH_MAX];
    const char *library_path = NULL;
    char library_path_buf[PATH_MAX];

    // Attempt to load AMD GPU support.
    // We favor the underdog on AMD + NVIDIA chimeras.
    switch (FLAG_gpu) {
    case LLAMAFILE_GPU_AMD:
    case LLAMAFILE_GPU_AUTO:

        // Get some essential paths.
        // ROCm SDK puts BLAS DLLs in same folder as clang++
        if (!IsWindows()) {
            if (get_rocm_bin_path(compiler_path_buf, "hipcc")) {
                strcpy(library_path_buf, compiler_path_buf);
                dirname(library_path_buf);
                compiler_path = compiler_path_buf;
                library_path = library_path_buf;
            }
        } else {
            if (get_rocm_bin_path(compiler_path_buf, "amdclang++") ||
                get_rocm_bin_path(compiler_path_buf, "clang++")) {
                strcpy(library_path_buf, compiler_path_buf);
                dirname(library_path_buf);
                compiler_path = compiler_path_buf;
                library_path = library_path_buf;
            }
        }

        // Get path of GGML DSO for AMD.
        llamafile_get_app_dir(dso, PATH_MAX);
        strlcat(dso, "ggml-rocm.", PATH_MAX);
        strlcat(dso, GetDsoExtension(), PATH_MAX);
        if (FLAG_nocompile || !FLAG_recompile) {
            if ((FileExists(dso) || extract_cuda_dso(dso, "ggml-rocm")) &&
                link_cuda_dso(dso, library_path)) {
                ggml_cuda.has_amd_gpu = true;
                return true;
            } else if (FLAG_nocompile) {
                goto TryNvidia;
            }
        }

        // Check if DSO is already compiled.
        if (!needs_rebuild && !FLAG_recompile) {
            switch (llamafile_is_file_newer_than(src, dso)) {
            case -1:
                return false;
            case false:
                if (link_cuda_dso(dso, library_path)) {
                    ggml_cuda.has_amd_gpu = true;
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
            if (compile_amd(compiler_path, dso, src)) {
                if (link_cuda_dso(dso, library_path)) {
                    ggml_cuda.has_amd_gpu = true;
                    return true;
                } else {
                    goto TryNvidia;
                }
            }
        } else {
            tinylog(__func__,
                    ": won't compile AMD GPU support because "
                    "$HIP_PATH/bin/clang++ is missing\n",
                    NULL);
        }

        // Try extracting prebuilt tinyBLAS DSO from PKZIP.
        if (extract_cuda_dso(dso, "ggml-rocm")) {
            if (link_cuda_dso(dso, library_path)) {
                ggml_cuda.has_amd_gpu = true;
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
        if (get_nvcc_path(compiler_path_buf)) {
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
        if (FLAG_nocompile || !FLAG_recompile) {
            if ((FileExists(dso) || extract_cuda_dso(dso, "ggml-cuda")) &&
                link_cuda_dso(dso, library_path)) {
                return true;
            } else if (FLAG_nocompile) {
                return false;
            }
        }

        // Check if DSO is already compiled.
        if (!needs_rebuild && !FLAG_recompile) {
            switch (llamafile_is_file_newer_than(src, dso)) {
            case -1:
                return false;
            case false:
                return link_cuda_dso(dso, library_path);
            case true:
                break;
            default:
                __builtin_unreachable();
            }
        }

        // Try building CUDA from source with mighty cuBLAS.
        if (compiler_path && compile_nvidia(compiler_path, dso, src)) {
            return link_cuda_dso(dso, library_path);
        }

        // Try extracting prebuilt tinyBLAS DSO from PKZIP.
        if (extract_cuda_dso(dso, "ggml-cuda"))
            return link_cuda_dso(dso, library_path);

        break;
    default:
        break;
    }

    // too bad
    return false;
}

static void import_cuda(void) {
    if (llamafile_has_metal())
        return;
    if (import_cuda_impl()) {
        ggml_cuda.supported = true;
    } else if (FLAG_gpu == LLAMAFILE_GPU_AMD || FLAG_gpu == LLAMAFILE_GPU_NVIDIA) {
        tinyprint(2, "fatal error: support for --gpu ", llamafile_describe_gpu(),
                  FLAG_tinyblas ? " --tinyblas" : "",
                  " was explicitly requested, but it wasn't available\n", NULL);
        exit(1);
    }
}

bool llamafile_has_cuda(void) {
    cosmo_once(&ggml_cuda.once, import_cuda);
    return ggml_cuda.supported;
}

bool llamafile_has_amd_gpu(void) {
    cosmo_once(&ggml_cuda.once, import_cuda);
    return ggml_cuda.has_amd_gpu;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    if (!llamafile_has_cuda())
        return 0;
    return ggml_cuda.buffer_type(device);
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    if (!llamafile_has_cuda())
        return 0;
    return ggml_cuda.host_buffer_type();
}

GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device) {
    if (!llamafile_has_cuda())
        return 0;
    return ggml_cuda.backend_init(device);
}

GGML_CALL ggml_backend_buffer_type_t
ggml_backend_cuda_split_buffer_type(const float *tensor_split) {
    if (!llamafile_has_cuda())
        return 0;
    return ggml_cuda.split_buffer_type(tensor_split);
}

GGML_CALL int ggml_backend_cuda_reg_devices(void) {
    if (!llamafile_has_cuda())
        return 0;
    return ggml_cuda.reg_devices();
}

GGML_CALL void ggml_backend_cuda_get_device_properties(int device, struct ggml_cuda_device_properties * properties) {
    if (!llamafile_has_cuda())
        return;
    return ggml_cuda.get_device_properties(device, properties);
}

GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t *free, size_t *total) {
    if (!llamafile_has_cuda())
        return;
    return ggml_cuda.get_device_memory(device, free, total);
}

GGML_CALL int ggml_backend_cuda_get_device_count(void) {
    if (!llamafile_has_cuda())
        return 0;
    return ggml_cuda.get_device_count();
}

GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void *buffer) {
    if (!llamafile_has_cuda())
        return;
    return ggml_cuda.unreg_host_buf(buffer);
}

GGML_CALL bool ggml_backend_cuda_register_host_buffer(void *buffer, size_t size) {
    if (!llamafile_has_cuda())
        return false;
    return ggml_cuda.register_host_buffer(buffer, size);
}

GGML_CALL void ggml_backend_cuda_get_device_description(int device, char *description,
                                                        size_t description_size) {
    if (!llamafile_has_cuda())
        return;
    return ggml_cuda.get_description(device, description, description_size);
}
