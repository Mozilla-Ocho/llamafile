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

#include "llamafile.h"
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <limits.h>
#include "llama.cpp/ggml-cuda.h"
#include "llama.cpp/ggml-metal.h"

int FLAG_gpu;
bool FLAG_nogpu;
bool FLAG_tinyblas;
bool FLAG_nocompile;
bool FLAG_recompile;

const char *llamafile_describe_gpu(void) {
    switch (FLAG_gpu) {
        case LLAMAFILE_GPU_AUTO:
            return "auto";
        case LLAMAFILE_GPU_AMD:
            return "amd";
        case LLAMAFILE_GPU_APPLE:
            return "apple";
        case LLAMAFILE_GPU_NVIDIA:
            return "nvidia";
        case LLAMAFILE_GPU_DISABLE:
            return "disabled";
        default:
            __builtin_unreachable();
    }
}

/**
 * Returns true if GPU support is available.
 */
int llamafile_gpu_supported(void) {
    return ggml_metal_supported() || ggml_cuda_supported();
}

/**
 * Parses `--gpu` flag.
 * @return GPU configuration, or -1 if `s` is a bad value
 */
int llamafile_gpu_parse(const char *s) {

    // Parse canonical names for GPUs.
    if (!strcasecmp(s, "disable")) return LLAMAFILE_GPU_DISABLE;
    if (!strcasecmp(s, "auto")) return LLAMAFILE_GPU_AUTO;
    if (!strcasecmp(s, "amd")) return LLAMAFILE_GPU_AMD;
    if (!strcasecmp(s, "apple")) return LLAMAFILE_GPU_APPLE;
    if (!strcasecmp(s, "nvidia")) return LLAMAFILE_GPU_NVIDIA;

    // Parse aliases.
    if (!strcasecmp(s, "disabled")) return LLAMAFILE_GPU_DISABLE;
    if (!strcasecmp(s, "metal")) return LLAMAFILE_GPU_APPLE;
    if (!strcasecmp(s, "cublas")) return LLAMAFILE_GPU_NVIDIA;
    if (!strcasecmp(s, "rocblas")) return LLAMAFILE_GPU_AMD;
    if (!strcasecmp(s, "rocm")) return LLAMAFILE_GPU_AMD;
    if (!strcasecmp(s, "hip")) return LLAMAFILE_GPU_AMD;

    return INT_MIN;
}
