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
    pthread_mutex_init(&lock_, 0);
    pthread_cond_init(&cond_, 0);
}

Slots::~Slots()
{
    for (const auto& e : slots_)
        if (e.slot())
            delete e.slot();
    pthread_cond_destroy(&cond_);
    pthread_mutex_destroy(&lock_);
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
            slots_.emplace(slot);
            all_slots_.push_back(slot);
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
        SlotEntry search_key{ &prefix };
        auto i = slots_.upper_bound(search_key);

        // handle special case
        if (i == slots_.end() && i != slots_.begin()) {
            --i;
            Slot* slot = i->slot();
            unassert(slot);
            slots_.erase(i);
            pthread_mutex_unlock(&lock_);
            return slot;
        }

        // avoid slots with non-matching suffix
        // they probably belong to another client
        if (i != slots_.begin()) {
            --i;
            Slot* slot = i->slot();
            unassert(slot);
            int cpl = vector_common_prefix_length(slot->history_, prefix);
            if (cpl == slot->history_.size()) {
                slots_.erase(i);
                pthread_mutex_unlock(&lock_);
                return slot;
            }
            ++i;
        }

        // otherwise return result of search
        if (i != slots_.end()) {
            Slot* slot = i->slot();
            unassert(slot);
            slots_.erase(i);
            pthread_mutex_unlock(&lock_);
            return slot;
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
    slots_.emplace(slot);
    pthread_cond_signal(&cond_);
    pthread_mutex_unlock(&lock_);
}

} // namespace server
} // namespace lf
