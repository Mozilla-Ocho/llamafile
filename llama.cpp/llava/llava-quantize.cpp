// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include "clip.h"
#include "llama.cpp/ggml.h"
#include "llamafile/version.h"
#include <stdio.h>
#include <stdlib.h>
#include "llamafile/llamafile.h"
#include <string.h>

int main(int argc, char *argv[]) {

    FLAG_gpu = LLAMAFILE_GPU_DISABLE; // [jart]

    if (llamafile_has(argv, "--version")) {
        puts("llava-quantize v" LLAMAFILE_VERSION_STRING);
        return 0;
    }

    if (llamafile_has(argv, "-h") ||
        llamafile_has(argv, "-help") ||
        llamafile_has(argv, "--help")) {
        llamafile_help("/zip/llama.cpp/llava/llava-quantize.1.asc");
        __builtin_unreachable();
    }

    llamafile_check_cpu();

    if (argc != 4) {
        fprintf(stderr, "%s: missing argument\n", argv[0]);
        return 1;
    }

    if (!clip_model_quantize(argv[1], argv[2], atoi(argv[3]))) {
        exit(1);
    }
}
