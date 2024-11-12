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

#include "chatbot.h"
#include "llama.cpp/base64.h"
#include "llama.cpp/common.h"
#include "llama.cpp/llama.h"
#include "llama.cpp/llava/llava.h"
#include "llamafile/datauri.h"
#include "llamafile/image.h"
#include "llamafile/llama.h"
#include "llamafile/string.h"
#include <cassert>
#include <string>
#include <vector>

namespace lf {
namespace chatbot {

bool eval_tokens(std::vector<llama_token> tokens) {
    int N = (int)tokens.size();
    if (tokens_used() + N > llama_n_ctx(g_ctx))
        return out_of_context(N);
    for (int i = 0; i < N; i += g_params.n_batch) {
        if (g_got_sigint) {
            g_got_sigint = false;
            clear_ephemeral();
            return false;
        }
        if (N > g_params.n_batch)
            print_ephemeral(format("loading prompt %d%%...", (int)((double)i / N * 100)));
        int n_eval = (int)tokens.size() - i;
        if (n_eval > g_params.n_batch)
            n_eval = g_params.n_batch;
        if (llama_decode(g_ctx, llama_batch_get_one(&tokens[i], n_eval, tokens_used(), 0)))
            return out_of_context(n_eval);
        g_history.insert(g_history.end(), tokens.begin() + i, tokens.begin() + i + n_eval);
    }
    clear_ephemeral();
    // this function is what computes /stats. we need to call it now
    // since llama_decode() kicks the can down the road to functions
    // like llama_sampling_sample(). that is bad because the chatbot
    // returns control to the repl rather than sampling when loading
    // system and image prompts.
    llama_synchronize(g_ctx);
    return true;
}

bool eval_image_embed(const llava_image_embed *image_embed) {
    int N = image_embed->n_image_pos;
    if (tokens_used() + N > llama_n_ctx(g_ctx))
        return out_of_context(N);
    int n_embd = llama_n_embd(llama_get_model(g_ctx));
    for (int i = 0; i < N; i += g_params.n_batch) {
        if (g_got_sigint) {
            g_got_sigint = false;
            clear_ephemeral();
            return false;
        }
        if (N > g_params.n_batch)
            print_ephemeral(format("loading image %d%%...", (int)((double)i / N * 100)));
        int n_eval = N - i;
        if (n_eval > g_params.n_batch)
            n_eval = g_params.n_batch;
        llama_batch batch = {
            .n_tokens = n_eval,
            .embd = image_embed->embed + i * n_embd,
            .all_pos_0 = tokens_used(),
            .all_pos_1 = 1,
            .all_seq_id = 0,
        };
        if (llama_decode(g_ctx, batch))
            return out_of_context(n_eval);
        for (int i = 0; i < n_eval; ++i)
            g_history.push_back(IMAGE_PLACEHOLDER_TOKEN);
    }
    clear_ephemeral();
    llama_synchronize(g_ctx);
    return true;
}

bool eval_image(const std::string_view binary) {
    unassert(g_clip);
    llava_image_embed *image_embed;
    print_ephemeral("analyzing image...");
    image_embed = llava_image_embed_make_with_bytes(
        g_clip, FLAG_threads_batch, (const unsigned char *)binary.data(), binary.size());
    clear_ephemeral();
    if (!image_embed) {
        err("failed to load image");
        return false;
    }
    bool ok = eval_image_embed(image_embed);
    llava_image_embed_free(image_embed);
    return ok;
}

bool eval_token(int id) {
    return eval_tokens({id});
}

bool eval_plain_text(const std::string &str, bool add_special, bool parse_special) {
    return eval_tokens(llamafile_tokenize(g_model, str, add_special, parse_special));
}

bool eval_string(std::string_view s, bool add_special, bool parse_special) {
    size_t i = 0;
    for (;;) {
        size_t pos = s.find("data:", i);
        if (pos == std::string_view::npos)
            return eval_plain_text(std::string(s), add_special, parse_special);
        i = pos + 5;
        DataUri uri;
        size_t end = uri.parse(s.substr(pos + 5));
        if (end == std::string_view::npos)
            continue;
        if (!lf::startscasewith(uri.mime, "image/"))
            continue;
        std::string image;
        try {
            image = uri.decode();
        } catch (const base64_error &e) {
            continue;
        }
        if (!is_image(image))
            continue;
        if (!eval_plain_text(std::string(s.substr(0, pos)), add_special, parse_special))
            return false;
        if (!eval_image(image))
            return false;
        s = s.substr(i + end);
        i = 0;
    }
}

} // namespace chatbot
} // namespace lf
