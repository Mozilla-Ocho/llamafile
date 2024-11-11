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

#include "image.h"
#include "llamafile/llamafile.h"
#include <cassert>
#include <utility>

namespace lf {
namespace server {

Image::~Image() = default;

Image::Image(const Image& old) : Image(old.bytes_, old.ctx_used_)
{
}

Image::Image(const std::string_view& bytes, int ctx_used)
  : bytes_(bytes), ctx_used_(ctx_used)
{
}

const std::string&
Image::bytes() const
{
    return bytes_;
}

int
Image::ctx_used() const
{
    // there are two use cases for image
    //
    // 1. when the http client asks for their chat messages to be
    //    tokenized, ctx_used should be set to -1, because we don't want
    //    to pay the price of the expensive clip image encode operation.
    //    since ctx_used isn't considered by comparators, this is a fine
    //    object state that's useful for searching data structures.
    //
    // 2. when an atom is actually evaluated by the model, if it's an
    //    image atom it'll be recreated with ctx_used specified. this is
    //    what slot does when maintaining its history, since it's the
    //    only thing that cares right now about context usage.
    //
    unassert(ctx_used_ > 0);
    return ctx_used_;
}

bool
operator<(const Image& lhs, const Image& rhs)
{
    return lhs.bytes() < rhs.bytes();
}

bool
operator==(const Image& lhs, const Image& rhs)
{
    return lhs.bytes() == rhs.bytes();
}

} // namespace server
} // namespace lf
