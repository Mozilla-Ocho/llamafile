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
#include "llamafile/llamafile.h"
#include "llamafile/macros.h"
#include "llamafile/server/atom.h"
#include "llamafile/server/log.h"
#include "llamafile/server/slot.h"
#include "llamafile/server/slot_entry.h"
#include "llamafile/vector.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>

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
        Slot* slot = new Slot(i, model_);
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
Slots::take(const std::vector<Atom>& atoms)
{
    pthread_mutex_lock(&lock_);
    for (;;) {

        // find best slot
        // iteration order favors lru
        time_t now = time(0);
        Dll* best_slot = nullptr;
        double best_score = INT_MIN;
        for (Dll* e = dll_first(free_slots_); e; e = dll_next(free_slots_, e)) {

            // least recently used is good
            int age = now - SLOT(e)->last_used_;
            double decay =
              age + exp(FLAG_decay_growth * (age - FLAG_decay_delay));

            // common prefix length is good
            int cpl = vector_common_prefix_length(SLOT(e)->history_, atoms);

            // common suffix length is good
            int csl = 0;
            int size = SLOT(e)->history_.size();
            for (int i = cpl + 1; i < size; ++i) {
                if (size - i > atoms.size() - cpl)
                    continue;
                if (std::equal(SLOT(e)->history_.begin() + i,
                               SLOT(e)->history_.end(),
                               atoms.begin() + cpl)) {
                    csl = size - i;
                    break;
                }
            }

            // discarded atoms is bad
            int discard;
            if (csl) {
                discard = 0;
            } else {
                discard = size - cpl;
            }

            // tally up score to determine best
            double score = cpl + csl + decay - discard;
            if (score >= best_score) {
                best_score = score;
                best_slot = e;
            }
        }

        // return borrowed pointer to best slot
        if (best_slot) {
            dll_remove(&free_slots_, best_slot);
            pthread_mutex_unlock(&lock_);
            SLOG("acquired slot #%d with score %d",
                 SLOT(best_slot)->id_,
                 (int)MIN(INT_MAX, best_score));
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
    unassert(slot);
    SLOG("relinquishing slot #%d", slot->id_);
    slot->last_used_ = time(0);
    pthread_mutex_lock(&lock_);
    dll_make_first(&free_slots_, &slot->elem_);
    pthread_cond_signal(&cond_);
    pthread_mutex_unlock(&lock_);
}

} // namespace server
} // namespace lf
