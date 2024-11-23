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

#include "llamafile.h"
#include "version.h"

#include <cosmo.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.cpp/llama.h"

int main(int argc, char **argv) {
    llamafile_check_cpu();

    if (llamafile_has(argv, "--version")) {
        puts("llamafile-tokenize v" LLAMAFILE_VERSION_STRING);
        return 0;
    }

    FLAG_log_disable = true;

    argc = cosmo_args("/zip/.args", &argv);
    llamafile_get_flags(argc, argv);

    llama_model_params mparams = {
        .n_gpu_layers = 0,
        .split_mode = (enum llama_split_mode)FLAG_split_mode,
        .main_gpu = 0,
        .tensor_split = nullptr,
        .rpc_servers = nullptr,
        .progress_callback = nullptr,
        .progress_callback_user_data = nullptr,
        .kv_overrides = nullptr,
        .vocab_only = true,
        .use_mmap = true,
        .use_mlock = false,
        .check_tensors = false,
    };
    llama_model *model = llama_load_model_from_file(FLAG_model, mparams);
    if (model == NULL)
        return 3;

    FILE *input;
    if (FLAG_prompt) {
        input = fmemopen((void *)FLAG_prompt, strlen(FLAG_prompt), "rb");
    } else if (FLAG_file) {
        if (!(input = fopen(FLAG_file, "rb"))) {
            perror(FLAG_file);
            exit(1);
        }
    } else {
        input = stdin;
    }

    for (;;) {
        char *text;
        size_t textlen;
        if (!(text = fgetln(input, &textlen)))
            break;

        static llama_token toks[4096];
        int count = llama_tokenize(model, text, textlen, toks, 4096, false, false);
        if (count < 0) {
            fprintf(stderr, "%s: failed to tokenize line\n", argv[0]);
            exit(1);
        }

        for (int i = 0; i < count; ++i) {

            char s[256];
            int n = llama_token_to_piece(model, toks[i], s, sizeof(s), false, false);
            if (n < 0) {
                fprintf(stderr, "%s: failed to convert token %d to string\n", argv[0], toks[i]);
                exit(1);
            }

            for (int i = 0; i < n; ++i) {
                int c = s[i] & 255;
                switch (c) {
                case '\\':
                    fputc('\\', stdout);
                    fputc('\\', stdout);
                    break;
                case '\a':
                    fputc('\\', stdout);
                    fputc('b', stdout);
                    break;
                case '\e':
                    fputc('\\', stdout);
                    fputc('e', stdout);
                    break;
                case '\v':
                    fputc('\\', stdout);
                    fputc('v', stdout);
                    break;
                case '\t':
                    fputc('\\', stdout);
                    fputc('t', stdout);
                    break;
                case '\r':
                    fputc('\\', stdout);
                    fputc('r', stdout);
                    break;
                case '\n':
                    fputc('\\', stdout);
                    fputc('n', stdout);
                    break;
                default:
                    if (isascii(c) && iscntrl(c)) {
                        fputc('\\', stdout);
                        fputc('0' + ((c & 0300) >> 6), stdout);
                        fputc('0' + ((c & 0070) >> 3), stdout);
                        fputc('0' + ((c & 0007) >> 0), stdout);
                    } else {
                        fputc(c, stdout);
                    }
                    break;
                }
            }
            fputc('\n', stdout);
        }
    }

    llama_free_model(model);
}
