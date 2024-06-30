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

#include "debug.h"
#include "llamafile.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "llama.cpp/llama.h"

bool FLAG_log_disable = false;
bool FLAG_mlock = false;
bool FLAG_mmap = true;
bool FLAG_nocompile = false;
bool FLAG_precise = false;
bool FLAG_recompile = false;
bool FLAG_tinyblas = false;
bool FLAG_unsecure = false;
const char *FLAG_file = nullptr;
const char *FLAG_listen = "0.0.0.0:8080";
const char *FLAG_model = nullptr;
const char *FLAG_prompt = nullptr;
float FLAG_temp = 0.8;
int FLAG_batch = 2048;
int FLAG_ctx = 512;
int FLAG_flash_attn = false;
int FLAG_gpu = 0;
int FLAG_http_ibuf_size = 1024 * 1024;
int FLAG_http_obuf_size = 1024 * 1024;
int FLAG_keepalive = 5;
int FLAG_main_gpu = 0;
int FLAG_n_gpu_layers = -1;
int FLAG_seed = LLAMA_DEFAULT_SEED;
int FLAG_split_mode = LLAMA_SPLIT_MODE_LAYER;
int FLAG_threads;
int FLAG_ubatch = 512;
int FLAG_verbose = 0;
int FLAG_workers;

int cpu_get_num_math();

static wontreturn void usage(int rc, int fd) {
    tinyprint(fd, "usage: ", program_invocation_name, " -m MODEL -l [HOST:]PORT\n", NULL);
    exit(rc);
}

static wontreturn void error(const char *message) {
    tinyprint(2, program_invocation_name, ": ", message, "\n", NULL);
    exit(1);
}

static wontreturn void bad(const char *flag) {
    tinyprint(2, program_invocation_name, ": bad value for ", flag, "\n", NULL);
    exit(1);
}

static wontreturn void missing(const char *flag) {
    tinyprint(2, program_invocation_name, ": ", flag, " missing argument\n", NULL);
    exit(1);
}

static wontreturn void required(const char *flag) {
    tinyprint(2, program_invocation_name, ": ", flag, " is required\n", NULL);
    exit(1);
}

static wontreturn void unknown(const char *flag) {
    tinyprint(2, program_invocation_name, ": ", flag, " unknown argument\n", NULL);
    exit(1);
}

void llamafile_get_flags(int argc, char **argv) {
    FLAG_threads = cpu_get_num_math();
    for (int i = 1; i < argc;) {
        const char *flag = argv[i++];

        if (*flag != '-')
            break;

        //////////////////////////////////////////////////////////////////////
        // logging flags

        if (!strcmp(flag, "--log-disable")) {
            FLAG_log_disable = true;
            continue;
        }

        if (!strcmp(flag, "-v") || !strcmp(flag, "--verbose")) {
            FLAG_verbose++;
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // server flags

        if (!strcmp(flag, "-l") || !strcmp(flag, "--listen")) {
            if (i == argc)
                missing("--listen");
            FLAG_listen = argv[i++];
            continue;
        }

        if (!strcmp(flag, "-k") || !strcmp(flag, "--keepalive")) {
            if (i == argc)
                missing("--keepalive");
            FLAG_keepalive = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-w") || !strcmp(flag, "--workers")) {
            if (i == argc)
                missing("--workers");
            FLAG_workers = atoi(argv[i++]);
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // http server flags

        if (!strcmp(flag, "--http-ibuf-size")) {
            if (i == argc)
                missing("--http-ibuf-size");
            FLAG_http_ibuf_size = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "--http-obuf-size")) {
            if (i == argc)
                missing("--http-obuf-size");
            FLAG_http_obuf_size = atoi(argv[i++]);
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // model flags

        if (!strcmp(flag, "-m") || !strcmp(flag, "--model")) {
            if (i == argc)
                missing("--model");
            FLAG_model = argv[i++];
            continue;
        }

        if (!strcmp(flag, "-f") || !strcmp(flag, "--file")) {
            if (i == argc)
                missing("--file");
            FLAG_file = argv[i++];
            continue;
        }

        if (!strcmp(flag, "--seed")) {
            if (i == argc)
                missing("--seed");
            FLAG_seed = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "--temp")) {
            if (i == argc)
                missing("--temp");
            FLAG_temp = atof(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-t") || !strcmp(flag, "--threads")) {
            if (i == argc)
                missing("--threads");
            FLAG_threads = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-b") || !strcmp(flag, "--batch-size")) {
            if (i == argc)
                missing("--batch-size");
            FLAG_batch = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-ub") || !strcmp(flag, "--ubatch-size")) {
            if (i == argc)
                missing("--ubatch-size");
            FLAG_ubatch = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-fa") || !strcmp(flag, "--flash-attn")) {
            if (i == argc)
                missing("--flash-attn");
            FLAG_flash_attn = true;
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // cpu flags

        if (!strcmp(flag, "--fast")) {
            FLAG_precise = false;
            continue;
        }

        if (!strcmp(flag, "--precise")) {
            FLAG_precise = true;
            continue;
        }

        if (!strcmp(flag, "--trap")) {
            FLAG_trap = true;
            FLAG_unsecure = true;
            llamafile_trapping_enabled(+1);
            continue;
        }

        if (!strcmp(flag, "--mlock")) {
            FLAG_mlock = true;
            continue;
        }

        if (!strcmp(flag, "--no-mmap")) {
            FLAG_mmap = false;
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // gpu flags

        if (!strcmp(flag, "--tinyblas")) {
            FLAG_tinyblas = true;
            continue;
        }

        if (!strcmp(flag, "--nocompile")) {
            FLAG_nocompile = true;
            continue;
        }

        if (!strcmp(flag, "--recompile")) {
            FLAG_recompile = true;
            continue;
        }

        if (!strcmp(flag, "--gpu")) {
            if (i == argc)
                missing("--gpu");
            FLAG_gpu = llamafile_gpu_parse(argv[i++]);
            if (FLAG_gpu == LLAMAFILE_GPU_ERROR)
                bad("--gpu");
            continue;
        }

        if (!strcmp(flag, "-ngl") || //
            !strcmp(flag, "--gpu-layers") || //
            !strcmp(flag, "--n-gpu-layers")) {
            if (i == argc)
                missing("--n-gpu-layers");
            FLAG_n_gpu_layers = atoi(argv[i++]);
            if (FLAG_n_gpu_layers <= 0)
                FLAG_gpu = LLAMAFILE_GPU_DISABLE;
            continue;
        }

        if (!strcmp(flag, "-mg") || !strcmp(flag, "--main-gpu")) {
            if (i == argc)
                missing("--main-gpu");
            FLAG_main_gpu = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-sm") || !strcmp(flag, "--split-mode")) {
            if (i == argc)
                missing("--split-mode");
            const char *value = argv[i];
            if (!strcmp(value, "none"))
                FLAG_split_mode = LLAMA_SPLIT_MODE_NONE;
            else if (!strcmp(value, "layer"))
                FLAG_split_mode = LLAMA_SPLIT_MODE_LAYER;
            else if (!strcmp(value, "row"))
                FLAG_split_mode = LLAMA_SPLIT_MODE_ROW;
            else
                bad("--split-mode");
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // security flags

        if (!strcmp(flag, "--unsecure")) {
            FLAG_unsecure = true;
            continue;
        }

        unknown(flag);
    }

    if (!FLAG_model)
        required("--model");

    FLAG_n_gpu_layers = llamafile_gpu_layers(FLAG_n_gpu_layers);
}
