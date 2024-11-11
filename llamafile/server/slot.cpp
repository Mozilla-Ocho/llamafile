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

#include "slot.h"
#include "llama.cpp/llava/clip.h"
#include "llama.cpp/llava/llava.h"
#include "llamafile/image.h"
#include "llamafile/llama.h"
#include "llamafile/llamafile.h"
#include "llamafile/macros.h"
#include "llamafile/server/atom.h"
#include "llamafile/server/image.h"
#include "llamafile/server/log.h"
#include "llamafile/vector.h"
#include "llamafile/version.h"
#include <algorithm>
#include <cassert>
#include <cosmo.h>

namespace lf {
namespace server {

static int
choose_ctx_size(llama_model* model)
{
    int n_ctx_train = llama_n_ctx_train(model);
    if (FLAG_ctx_size <= 0 || FLAG_ctx_size > n_ctx_train)
        return n_ctx_train;
    return FLAG_ctx_size;
}

static std::string
generate_system_fingerprint(const llama_context_params* cparams)
{
    uint64_t h = 0;
    h ^= __fnv(LLAMAFILE_VERSION_STRING, sizeof(LLAMAFILE_VERSION_STRING));
    h ^= __fnv(cparams, sizeof(*cparams));
    std::string b = "fp_";
    for (int j = 0; j < 64 / 5; ++j) {
        b += "abcdefghijklmnopqrstuvwxyz012345"[h & 31];
        h >>= 5;
    }
    return b;
}

const char*
Slot::describe_error(int err)
{
    switch (err) {
        case uninitialized:
            return "uninitialized";
        case out_of_context:
            return "out_of_context";
        case no_vision_model:
            return "no_vision_model";
        case decode_token_failed:
            return "decode_token_failed";
        case decode_image_failed:
            return "decode_image_failed";
        case encode_image_failed:
            return "encode_image_failed";
        default:
            return "bad_error_code";
    }
}

Slot::Slot(llama_model* model) : model_(model)
{
}

Slot::~Slot()
{
    if (ctx_)
        llama_free(ctx_);
    if (clip_ctx_)
        clip_free(clip_ctx_);
}

bool
Slot::start()
{
    unassert(!ctx_);
    llama_context_params cparams = {};
    cparams.embeddings = false;
    cparams.embeddings_only = false;
    cparams.logits_all = false;
    cparams.seed = 12345;
    cparams.n_ctx = choose_ctx_size(model_);
    cparams.n_batch = FLAG_batch;
    cparams.n_ubatch = FLAG_ubatch;
    cparams.n_seq_max = 1;
    cparams.n_threads = MIN(FLAG_threads, 20);
    cparams.n_threads_batch = FLAG_threads;
    cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
    cparams.attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
    cparams.rope_freq_base = 0;
    cparams.yarn_ext_factor = -1;
    cparams.yarn_attn_factor = 1;
    cparams.yarn_beta_fast = 32;
    cparams.yarn_beta_slow = 1;
    cparams.yarn_orig_ctx = 0;
    cparams.defrag_thold = -1;
    cparams.offload_kqv = true;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    cparams.flash_attn = FLAG_flash_attn;
    system_fingerprint_ = generate_system_fingerprint(&cparams);
    if (!(ctx_ = llama_new_context_with_model(model_, cparams)))
        return false;
    if (FLAG_mmproj)
        if (!(clip_ctx_ = clip_model_load(FLAG_mmproj, FLAG_verbose)))
            return false;
    return true;
}

int
Slot::ctx_size() const
{
    return llama_n_ctx(ctx_);
}

int
Slot::ctx_used() const
{
    int tokens = 0;
    for (size_t i = 0; i < history_.size(); ++i)
        tokens += history_[i].ctx_used();
    return tokens;
}

int
Slot::eval_token(int token)
{
    return eval_tokens({ token });
}

int
Slot::eval_tokens(const std::vector<int>& tokens)
{
    if (!ctx_)
        return uninitialized;
    if (tokens.empty())
        return 0;
    int N = tokens.size();
    int used = ctx_used();
    if (used + N > ctx_size())
        return out_of_context;
    std::vector<int> toks(tokens); // TODO(jart): is copying really needed?
    for (int i = 0; i < N; i += FLAG_batch) {
        int n_eval = N - i;
        if (n_eval > FLAG_batch)
            n_eval = FLAG_batch;
        if (llama_decode(ctx_,
                         { .n_tokens = n_eval,
                           .token = &toks[i],
                           .all_pos_0 = used,
                           .all_pos_1 = 1 }))
            return decode_token_failed;
        for (int j = 0; j < n_eval; ++j)
            history_.emplace_back(toks[i + j]);
        used += n_eval;
    }
    last_used_ = timespec_real();
    return N;
}

int
Slot::eval_image(const std::string_view& bytes)
{
    if (!ctx_)
        return uninitialized;
    if (!clip_ctx_)
        return no_vision_model;
    llava_image_embed* image_embed =
      llava_image_embed_make_with_bytes(clip_ctx_,
                                        FLAG_threads_batch,
                                        (const unsigned char*)bytes.data(),
                                        bytes.size());
    if (!image_embed)
        return encode_image_failed;
    int used = ctx_used();
    int N = image_embed->n_image_pos;
    if (used + N > ctx_size()) {
        llava_image_embed_free(image_embed);
        return out_of_context;
    }
    int n_embd = llama_n_embd(llama_get_model(ctx_));
    for (int i = 0; i < N; i += FLAG_batch) {
        int n_eval = N - i;
        if (n_eval > FLAG_batch)
            n_eval = FLAG_batch;
        if (llama_decode(ctx_,
                         { .n_tokens = n_eval,
                           .embd = image_embed->embed + i * n_embd,
                           .all_pos_0 = used,
                           .all_pos_1 = 1 })) {
            llava_image_embed_free(image_embed);
            return decode_image_failed;
        }
        used += n_eval;
    }
    llava_image_embed_free(image_embed);
    history_.emplace_back(new Image(bytes, N));
    last_used_ = timespec_real();
    return N;
}

int
Slot::eval_atoms(const std::vector<Atom>& atoms)
{
    int rc;
    int token_count = 0;
    std::vector<int> tokens;
    for (const Atom& atom : atoms) {
        if (atom.is_token()) {
            tokens.emplace_back(atom.token());
        } else if (atom.is_image()) {
            if ((rc = eval_tokens(tokens)) < 0)
                return rc;
            token_count += rc;
            tokens.clear();
            if ((rc = eval_image(atom.image().bytes())) < 0)
                return rc;
            token_count += rc;
        }
    }
    if ((rc = eval_tokens(tokens)) < 0)
        return rc;
    token_count += rc;
    return token_count;
}

int
Slot::prefill(const std::vector<Atom>& atoms)
{
    if (!ctx_)
        return uninitialized;
    int used_tokens = ctx_used();
    int reuse_atoms = 0;
    int reuse_tokens = 0;
    int erase_tokens = 0;
    int n = std::min(atoms.size(), history_.size());
    for (int i = 0; i < n && atoms[i] == history_[i]; ++i) {
        reuse_tokens += history_[i].ctx_used();
        reuse_atoms += 1;
    }
    // xxx: ensure we prefill at least one token (prevents badness)
    if (reuse_tokens >= 1) {
        reuse_atoms -= 1;
        reuse_tokens -= history_[reuse_atoms].ctx_used();
    }
    if (used_tokens > reuse_tokens) {
        erase_tokens = used_tokens - reuse_tokens;
        if (llama_kv_cache_seq_rm(ctx_, 0, reuse_tokens, -1)) {
            history_.resize(reuse_atoms);
        } else {
            SLOG("failed to remove tokens from KV cache");
            llama_kv_cache_clear(ctx_);
            reuse_atoms = 0;
            reuse_tokens = 0;
            erase_tokens = used_tokens;
            history_.clear();
        }
    }
    std::vector<Atom> new_atoms(atoms.begin() + reuse_atoms, atoms.end());
    int rc;
    if ((rc = eval_atoms(new_atoms)) < 0)
        return rc;
    int token_count = reuse_tokens + rc;
    SLOG("prefilled %zu tokens (after removing %zu and reusing %zu)",
         token_count,
         erase_tokens,
         reuse_tokens);
    return token_count;
}

void
Slot::dump(std::string* result)
{
    for (size_t i = 0; i < history_.size(); ++i) {
        if (history_[i].is_token()) {
            llama_token token = history_[i].token();
            *result +=
              llamafile_token_to_piece(ctx_, token, RENDER_SPECIAL_TOKENS);
        } else if (history_[i].is_image()) {
            convert_image_to_uri(result, history_[i].image().bytes());
        }
    }
}

} // namespace server
} // namespace lf
