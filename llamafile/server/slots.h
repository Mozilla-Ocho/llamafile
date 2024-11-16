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
#include <memory>
#include <pthread.h>
#include <vector>

struct llama_model;
struct Dll;

namespace lf {
namespace server {

class Atom;
class SlotEntry;
struct Slot;

struct Slots
{
    llama_model* model_;
    pthread_cond_t cond_;
    pthread_mutex_t lock_;
    std::vector<std::unique_ptr<Slot>> slots_;

    // first elements are most recently used
    // last elements are least recently used
    Dll* free_slots_ = nullptr;

    explicit Slots(llama_model*);
    ~Slots();
    size_t size();
    int start(int);
    void tokenize(std::vector<Atom>*, std::string_view, bool);
    Slot* take(const std::vector<Atom>&);
    void give(Slot*);
};

} // namespace server
} // namespace lf
