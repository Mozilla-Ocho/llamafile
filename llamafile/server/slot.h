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

#pragma once
#include "llama.cpp/llama.h"
#include <cosmo.h>
#include <stdatomic.h>
#include <time.h>
#include <vector>

#define SLOT(e) DLL_CONTAINER(Slot, elem, e)

struct Slot
{
    Dll elem_;
    atomic_int refs_ = 0;
    llama_context* ctx_ = nullptr;
    timespec last_used_ = timespec_zero;
    std::vector<llama_token> history_;

    Slot();
    ~Slot();
    int n_ctx();
    bool start();
    bool eval_token(llama_token);
    bool eval_tokens(std::vector<llama_token>);
    bool can_use_slot(const std::vector<llama_token>&);
    bool prefill(const std::vector<llama_token>&);
};
