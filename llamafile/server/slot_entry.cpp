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

#include "slot_entry.h"
#include "llamafile/server/atom.h"
#include "llamafile/server/slot.h"
#include <cassert>

namespace lf {
namespace server {

SlotEntry::SlotEntry(Slot* slot) : slot_(slot), key_(nullptr)
{
    unassert(slot);
}

SlotEntry::SlotEntry(const std::vector<Atom>* key) : slot_(nullptr), key_(key)
{
    unassert(key);
}

SlotEntry::~SlotEntry()
{
}

Slot*
SlotEntry::slot() const
{
    return slot_;
}

const std::vector<Atom>*
SlotEntry::key() const
{
    return key_ ? key_ : &slot_->history_;
}

bool
operator<(const SlotEntry& lhs, const SlotEntry& rhs)
{
    return *lhs.key() < *rhs.key();
}

} // namespace server
} // namespace lf
