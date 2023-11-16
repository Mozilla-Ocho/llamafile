// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set net ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include <stdio.h>
#include <stdlib.h>
#include "llama.cpp/ggml.h"
#include "clip.h"

int main(int argc, char *argv[]) {
    llamafile_check_cpu();
    if (argc != 4) {
        fprintf(stderr, "Usage: %s INPUT OUTPUT FORMAT\n");
        exit(1);
    }
    if (!clip_model_quantize(argv[1], argv[2], atoi(argv[3]))) {
        exit(1);
    }
}
