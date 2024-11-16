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

#include "slots.h"
#include "llamafile/server/atom.h"
#include "llamafile/server/log.h"
#include "llamafile/server/slot.h"
#include "llamafile/server/slot_entry.h"
#include "llamafile/vector.h"
#include <cassert>

namespace lf {
namespace server {

Slots::Slots(llama_model* model) : model_(model)
{
    pthread_cond_init(&cond_, 0);
    pthread_mutex_init(&lock_, 0);
}

Slots::~Slots()
{
    pthread_mutex_destroy(&lock_);
    pthread_cond_destroy(&cond_);
}

size_t
Slots::size()
{
    return slots_.size();
}

int
Slots::start(int count)
{
    int made = 0;
    pthread_mutex_lock(&lock_);
    for (int i = 0; i < count; ++i) {
        Slot* slot = new Slot(model_);
        if (slot->start()) {
            ++made;
            slots_.emplace_back(slot);
            dll_make_last(&free_slots_, &slot->elem_);
        } else {
            delete slot;
        }
    }
    if (made)
        pthread_cond_broadcast(&cond_);
    pthread_mutex_unlock(&lock_);
    if (made < count)
        SLOG("could only make %d out of %d slots", made);
    return made;
}

Slot*
Slots::take(const std::vector<Atom>& prefix)
{
    pthread_mutex_lock(&lock_);
    for (;;) {

        // find slot with longest matching prefix
        // favoring least recently used if multiple ones
        int best_cpl = 0;
        Dll* best_slot = nullptr;
        for (Dll* e = dll_first(free_slots_); e; e = dll_next(free_slots_, e)) {
            int cpl = vector_common_prefix_length(SLOT(e)->history_, prefix);
            if (cpl >= best_cpl) {
                best_cpl = cpl;
                best_slot = e;
            }
        }

        // return borrowed pointer to best slot
        if (best_slot) {
            dll_remove(&free_slots_, best_slot);
            pthread_mutex_unlock(&lock_);
            return SLOT(best_slot);
        }

        // all slots are being used
        SLOG("waiting for slot to be relinquished...");
        pthread_cond_wait(&cond_, &lock_);
    }
}

void
Slots::give(Slot* slot)
{
    SLOG("relinquishing slot");
    unassert(slot);
    pthread_mutex_lock(&lock_);
    dll_make_first(&free_slots_, &slot->elem_);
    pthread_cond_signal(&cond_);
    pthread_mutex_unlock(&lock_);
}

} // namespace server
} // namespace lf
