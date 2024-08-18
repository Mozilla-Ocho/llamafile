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

#include "llama.cpp/ggml-cuda.h"
#include "llama.cpp/ggml-metal.h"
#include "llamafile.h"
#include "llamafile/log.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
        return "error";
    }
}

/**
 * Returns true if GPU support is available.
 */
bool llamafile_has_gpu(void) {
    return llamafile_has_metal() || llamafile_has_cuda();
}

/**
 * Figures out the GPU story after config is loaded.
 */
int llamafile_gpu_layers(int n_gpu_layers) {

    if (FLAG_gpu == LLAMAFILE_GPU_DISABLE)
        return 0;

    // if user explicitly passed `--gpu KIND` but didn't specify `-ngl
    // LAYERS` then assume the user wants their model fully offloaded.
    if (n_gpu_layers < 0 && FLAG_gpu > 0)
        n_gpu_layers = INT_MAX;

    // Apple Metal is safe enough to enable by default.
    if (n_gpu_layers <= 0 && llamafile_has_metal())
        n_gpu_layers = INT_MAX;

    // make the -ngl flag not break example code when it's impossible
    // if you want it to be an error just pass --gpu apple/amd/nvidia
    if (n_gpu_layers > 0 && !llamafile_has_gpu()) {
        tinylogf("warning: --n-gpu-layers %d was passed but no GPUs were found;"
                 " falling back to CPU inference\n",
                 n_gpu_layers);
        FLAG_gpu = LLAMAFILE_GPU_DISABLE;
        n_gpu_layers = 0;
    }

    // don't bother linking gpu modules if zero layers are offloaded
    if (n_gpu_layers <= 0) {
        if (n_gpu_layers == -1)
            tinylogf("note: if you have an AMD or NVIDIA GPU then you need to pass -ngl 9999 to "
                     "enable GPU offloading\n");
        FLAG_gpu = LLAMAFILE_GPU_DISABLE;
    }

    return n_gpu_layers;
}

/**
 * Parses `--gpu` flag.
 * @return GPU configuration, or -1 if `s` is a bad value
 */
int llamafile_gpu_parse(const char *s) {

    // Parse canonical names for GPUs.
    if (!strcasecmp(s, "disable"))
        return LLAMAFILE_GPU_DISABLE;
    if (!strcasecmp(s, "auto"))
        return LLAMAFILE_GPU_AUTO;
    if (!strcasecmp(s, "amd"))
        return LLAMAFILE_GPU_AMD;
    if (!strcasecmp(s, "apple"))
        return LLAMAFILE_GPU_APPLE;
    if (!strcasecmp(s, "nvidia"))
        return LLAMAFILE_GPU_NVIDIA;

    // Parse aliases.
    if (!strcasecmp(s, "disabled"))
        return LLAMAFILE_GPU_DISABLE;
    if (!strcasecmp(s, "metal"))
        return LLAMAFILE_GPU_APPLE;
    if (!strcasecmp(s, "cublas"))
        return LLAMAFILE_GPU_NVIDIA;
    if (!strcasecmp(s, "rocblas"))
        return LLAMAFILE_GPU_AMD;
    if (!strcasecmp(s, "rocm"))
        return LLAMAFILE_GPU_AMD;
    if (!strcasecmp(s, "hip"))
        return LLAMAFILE_GPU_AMD;

    return LLAMAFILE_GPU_ERROR;
}
