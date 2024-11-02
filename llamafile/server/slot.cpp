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
#include "llamafile/llama.h"
#include "llamafile/llamafile.h"
#include "llamafile/macros.h"
#include "llamafile/vector.h"
#include "llamafile/version.h"
#include "log.h"
#include <cassert>
#include <cosmo.h>

static int
ctx_size(llama_model* model)
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

Slot::Slot(llama_model* model) : model_(model)
{
}

Slot::~Slot()
{
    if (ctx_)
        llama_free(ctx_);
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
    cparams.n_ctx = ctx_size(model_);
    cparams.n_batch = FLAG_batch;
    cparams.n_ubatch = FLAG_ubatch;
    cparams.n_seq_max = 1;
    cparams.n_threads = FLAG_threads;
    cparams.n_threads_batch = MIN(FLAG_threads, 20);
    cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.rope_freq_base = 0;
    cparams.yarn_ext_factor = -1;
    cparams.yarn_attn_factor = 1;
    cparams.yarn_beta_fast = 32;
    cparams.yarn_beta_slow = 1;
    cparams.yarn_orig_ctx = 0;
    cparams.defrag_thold = -1;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    cparams.flash_attn = FLAG_flash_attn;
    system_fingerprint_ = generate_system_fingerprint(&cparams);
    if (!(ctx_ = llama_new_context_with_model(model_, cparams)))
        return false;
    return true;
}

int
Slot::n_ctx()
{
    return llama_n_ctx(ctx_);
}

bool
Slot::eval_token(int id)
{
    std::vector<int> tokens;
    tokens.push_back(id);
    return eval_tokens(tokens);
}

bool
Slot::eval_tokens(std::vector<int> tokens)
{
    unassert(ctx_);
    if (history_.size() + tokens.size() > n_ctx())
        return false;
    for (int i = 0; i < tokens.size(); i += FLAG_batch) {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > FLAG_batch)
            n_eval = FLAG_batch;
        if (llama_decode(
              ctx_,
              llama_batch_get_one(&tokens[i], n_eval, history_.size(), 0)))
            return false;
        history_.insert(
          history_.end(), tokens.begin() + i, tokens.begin() + i + n_eval);
    }
    last_used_ = timespec_real();
    return true;
}

bool
Slot::prefill(const std::vector<int>& tokens)
{
    unassert(ctx_);
    size_t erase_count = 0;
    size_t reuse_count = lf::vector_common_prefix_length(tokens, history_);
    if (history_.size() > reuse_count) {
        erase_count = history_.size() - reuse_count;
        if (llama_kv_cache_seq_rm(ctx_, 0, reuse_count, -1)) {
            history_.resize(reuse_count);
        } else {
            SLOG("failed to remove tokens from KV cache");
            llama_kv_cache_clear(ctx_);
            reuse_count = 0;
            erase_count = history_.size();
            history_.clear();
        }
    }
    std::vector<int> new_tokens(tokens.begin() + history_.size(), tokens.end());
    SLOG("prefilling %zu tokens (after removing %zu and reusing %zu)",
         new_tokens.size(),
         erase_count,
         reuse_count);
    return eval_tokens(new_tokens);
}
