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
#include <string>
#include <time.h>
#include <utility>
#include <vector>

struct llama_model;
struct llama_context;

struct Slot
{
    llama_model* model_;
    llama_context* ctx_ = nullptr;
    timespec last_used_ = timespec_zero;
    std::vector<int> history_;
    std::string system_fingerprint_;

    Slot(llama_model*);
    ~Slot();
    int n_ctx();
    bool start();
    bool eval_token(int);
    bool eval_tokens(std::vector<int>);
    bool prefill(const std::vector<int>&);
    std::string dump();
};
