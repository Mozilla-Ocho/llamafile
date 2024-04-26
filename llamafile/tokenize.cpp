// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#include <cerrno>
#include <cmath>
#include <cosmo.h>
#include <cstdio>
#include <cstring>
#include <vector>

#include "llama.cpp/common.h"
#include "llama.cpp/llama.h"
#include "llamafile/llamafile.h"

int main(int argc, char **argv) {

    if (llamafile_has(argv, "--version")) {
        puts("llamafile-tokenize v" LLAMAFILE_VERSION_STRING);
        return 0;
    }

    llamafile_check_cpu();
    log_disable();

    gpt_params params;
    params.n_ctx = 0;

    if (!gpt_params_parse(argc, argv, params))
        return 1;

    llama_model_params model_params = llama_model_default_params();
    llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL)
        return 3;

    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    llama_context *ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL)
        return 4;

    bool should_read_stdin = params.prompt.empty();

    for (;;) {
        ssize_t n;
        char buf[4097];
        const char *input;
        if (should_read_stdin) {
            n = read(0, buf, 4096);
            if (n == -1) {
                fprintf(stderr, "/dev/stdin: %s\n", strerror(errno));
                exit(1);
            }
            if (!n)
                break;
            buf[n] = 0;
            input = buf;
        } else {
            input = params.prompt.c_str();
        }

        std::vector<llama_token> toks = ::llama_tokenize(ctx, input, false);
        for (llama_token tok : toks) {
            std::string str = llama_token_to_piece(ctx, tok, true);
            const char *s = str.c_str();
            for (int i = 0; s[i]; ++i) {
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

    llama_free(ctx);
    llama_free_model(model);
}
