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
#include <pthread.h>
#include <set>
#include <vector>

struct llama_model;

namespace lf {
namespace server {

class Atom;
struct Slot;
class SlotEntry;

struct Slots
{
    llama_model* model_;
    std::multiset<SlotEntry> slots_;
    std::vector<Slot*> all_slots_;
    pthread_mutex_t lock_;
    pthread_cond_t cond_;

    explicit Slots(llama_model*);
    ~Slots();
    int start(int);
    void tokenize(std::vector<Atom>*, std::string_view, bool);
    Slot* take(const std::vector<Atom>&);
    void give(Slot*);
};

} // namespace server
} // namespace lf
