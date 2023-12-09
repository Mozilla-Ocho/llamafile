// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include "clip.h"
#include "llama.cpp/ggml.h"
#include "llamafile/version.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc == 2 && !strcmp(argv[1], "--version")) {
        printf("llamafile v" LLAMAFILE_VERSION_STRING " llava-quantize\n");
        exit(0);
    }
    llamafile_check_cpu();
    if (argc != 4) {
        fprintf(stderr,
                "Usage: %s INPUT OUTPUT FORMAT\n"
                "  - 2 is Q4_0\n"
                "  - 3 is Q4_1\n"
                "  - 6 is Q5_0\n"
                "  - 7 is Q5_1\n"
                "  - 8 is Q8_0\n",
                argv[0]);
        return 1;
    }
    if (!clip_model_quantize(argv[1], argv[2], atoi(argv[3]))) {
        exit(1);
    }
}
