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
#include <ctime>
#include <string>
#include <vector>

struct llama_context;
struct llama_model;
struct clip_ctx;

namespace lf {
namespace server {

struct Atom;
struct Image;

struct Slot
{
    enum
    {
        uninitialized = -4096,
        out_of_context,
        no_vision_model,
        decode_token_failed,
        decode_image_failed,
        encode_image_failed,
    };

    static const char* describe_error(int);

    llama_model* model_;
    clip_ctx* clip_ctx_ = nullptr;
    llama_context* ctx_ = nullptr;
    timespec last_used_ = timespec_zero;
    std::vector<Atom> history_;
    std::string system_fingerprint_;

    ~Slot();
    explicit Slot(llama_model*);
    int ctx_size() const;
    int ctx_used() const;
    bool start();
    int eval_token(int);
    int eval_image(const std::string_view&);
    int eval_tokens(const std::vector<int>&);
    int eval_atoms(const std::vector<Atom>&);
    int prefill(const std::vector<Atom>&);
    void tokenize(std::vector<Atom>*, std::string_view, bool);
    void dump(std::string*);
};

} // namespace server
} // namespace lf
