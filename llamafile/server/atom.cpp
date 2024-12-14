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

#include "atom.h"
#include "image.h"
#include <cassert>

namespace lf {
namespace server {

/**
 * @fileoverview Context window content container.
 *
 * The `Atom` class is used for tracking what's inside a context window.
 * Each individual atom can hold one of two values. The first is a token
 * which is a short piece of text, subjective to each `llama_model`. The
 * second is an `Image` which uses a separate evaluation process.
 */

Atom::Atom(int token) : word_(1ull << 56 | (unsigned)token)
{
}

Atom::Atom(Image* image) : word_(2ull << 56 | (uintptr_t)image)
{
    unassert(image != nullptr);
}

Atom::Atom(Atom&& other)
{
    word_ = other.word_;
    other.word_ = 0;
}

Atom::Atom(const Atom& other)
{
    if (!other.is_image()) {
        word_ = other.word_;
        return;
    }
    Image* image = new Image(other.image());
    word_ = 2ull << 56 | (uintptr_t)image;
}

Atom&
Atom::operator=(const Atom& other)
{
    if (this != &other) {
        if (is_image())
            delete (Image*)(word_ & 0x00ffffffffffffff);
        if (!other.is_image()) {
            word_ = other.word_;
        } else {
            Image* image = new Image(other.image());
            word_ = 2ull << 56 | (uintptr_t)image;
        }
    }
    return *this;
}

Atom::~Atom()
{
    if (is_image())
        delete (Image*)(word_ & 0x00ffffffffffffff);
}

bool
Atom::empty() const
{
    return !word_;
}

bool
Atom::is_token() const
{
    return (word_ >> 56) == 1;
}

bool
Atom::is_image() const
{
    return (word_ >> 56) == 2;
}

int
Atom::token() const
{
    unassert(is_token());
    return word_;
}

const Image&
Atom::image() const
{
    unassert(is_image());
    return *(Image*)(word_ & 0x00ffffffffffffff);
}

int
Atom::ctx_used() const
{
    switch (word_ >> 56) {
        case 1:
            return 1;
        case 2:
            return image().ctx_used();
        default:
            return 0;
    }
}

bool
operator<(const Atom& lhs, const Atom& rhs)
{
    if (lhs.empty())
        return !rhs.empty();
    if (rhs.empty())
        return false;
    if (!lhs.is_image()) {
        if (!rhs.is_image())
            return lhs.token() < rhs.token();
        return true;
    } else {
        if (rhs.is_image())
            return lhs.image() < rhs.image();
        return false;
    }
}

bool
operator==(const Atom& lhs, const Atom& rhs)
{
    if (lhs.empty())
        return rhs.empty();
    if (rhs.empty())
        return false;
    if (lhs.is_token()) {
        if (!rhs.is_token())
            return false;
        return lhs.token() == rhs.token();
    } else {
        if (!rhs.is_image())
            return false;
        return lhs.image() == rhs.image();
    }
}

} // namespace server
} // namespace lf
