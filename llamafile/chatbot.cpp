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

#include <assert.h>
#include <cosmo.h>
#include <ctype.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "llama.cpp/common.h"
#include "llama.cpp/llama.h"
#include "llamafile/bestline.h"
#include "llamafile/llamafile.h"

static sig_atomic_t g_got_sigint;

static void on_sigint(int sig) {
    g_got_sigint = 1;
}

static bool is_empty(const char *s) {
    int c;
    while ((c = *s++))
        if (!isspace(c))
            return false;
    return true;
}

static std::string basename(const std::string_view path) {
    size_t i, e;
    if ((e = path.size())) {
        while (e > 1 && path[e - 1] == '/')
            --e;
        i = e - 1;
        while (i && path[i - 1] != '/')
            --i;
        return std::string(path.substr(i, e - i));
    } else {
        return ".";
    }
}

static bool eval_tokens(struct llama_context *ctx_llama, std::vector<llama_token> tokens,
                        int n_batch, int *n_past) {
    int N = (int)tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch)
            n_eval = n_batch;
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0)))
            return false; // probably ran out of context
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context *ctx_llama, int id, int *n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context *ctx_llama, const char *str, int n_batch, int *n_past,
                        bool add_special, bool parse_special) {
    std::string str2 = str;
    std::vector<llama_token> embd_inp =
        ::llama_tokenize(ctx_llama, str2, add_special, parse_special);
    return eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
}

static void print_ephemeral(const char *description) {
    fprintf(stderr, " \e[90m%s\e[39m\r", description);
}

static void clear_ephemeral(void) {
    fprintf(stderr, "\e[K");
}

int main(int argc, char **argv) {
    llamafile_check_cpu();
    ShowCrashReports();
    log_disable();

    gpt_params params;
    params.n_ctx = 8192; // make default context more reasonable

    if (!gpt_params_parse(argc, argv, params))
        return 1;

    printf("\n\e[32m\
██╗     ██╗      █████╗ ███╗   ███╗ █████╗ ███████╗██╗██╗     ███████╗\n\
██║     ██║     ██╔══██╗████╗ ████║██╔══██╗██╔════╝██║██║     ██╔════╝\n\
██║     ██║     ███████║██╔████╔██║███████║█████╗  ██║██║     █████╗\n\
██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══╝  ██║██║     ██╔══╝\n\
███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║     ██║███████╗███████╗\n\
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝\e[39m\n\
\e[1msoftware\e[22m: llamafile " LLAMAFILE_VERSION_STRING "\n\
\e[1mmodel\e[22m:    %s\n\n",
           basename(params.model).c_str());

    print_ephemeral("initializing backend...");
    llama_backend_init();
    clear_ephemeral();

    print_ephemeral("initializing model...");
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = llamafile_gpu_layers(35);
    llama_model *model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL)
        return 2;
    clear_ephemeral();

    print_ephemeral("initializing context...");
    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    llama_context *ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL)
        return 3;
    clear_ephemeral();

    if (params.prompt.empty())
        params.prompt =
            "A chat between a curious human and an artificial intelligence assistant. The "
            "assistant gives helpful, detailed, and polite answers to the human's questions.";

    print_ephemeral("loading prompt...");
    int n_past = 0;
    bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));
    std::vector<llama_chat_msg> chat = {{"system", params.prompt}};
    std::string msg = llama_chat_apply_template(model, params.chat_template, chat, false);
    eval_string(ctx, msg.c_str(), params.n_batch, &n_past, add_bos, true);
    clear_ephemeral();
    printf("%s\n", params.special ? msg.c_str() : params.prompt.c_str());

    // perform important setup
    struct llama_sampling_context *ctx_sampling = llama_sampling_init(params.sparams);
    signal(SIGINT, on_sigint);

    // run chatbot
    char *line;
    bestlineLlamaMode(true);
    while ((line = bestlineWithHistory(">>> ", "llamafile"))) {
        if (is_empty(line)) {
            free(line);
            continue;
        }
        std::vector<llama_chat_msg> chat = {{"user", line}};
        std::string msg = llama_chat_apply_template(model, params.chat_template, chat, true);
        eval_string(ctx, msg.c_str(), params.n_batch, &n_past, false, true);
        while (!g_got_sigint) {
            llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL);
            llama_sampling_accept(ctx_sampling, ctx, id, true);
            printf("%s", llama_token_to_piece(ctx, id, params.special).c_str());
            if (llama_token_is_eog(model, id))
                break;
            fflush(stdout);
            if (!eval_id(ctx, id, &n_past)) {
                fprintf(stderr, "[out of context]\n");
                exit(1);
            }
        }
        g_got_sigint = 0;
        printf("\n");
        free(line);
    }

    llama_sampling_free(ctx_sampling);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}
