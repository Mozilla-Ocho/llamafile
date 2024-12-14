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

namespace lf {
namespace server {

class Image;

class Atom
{
  public:
    Atom() = default;
    explicit Atom(int);
    explicit Atom(Image*);
    Atom(const Atom&);
    Atom(Atom&&);
    ~Atom();
    Atom& operator=(const Atom&);
    int token() const;
    bool empty() const;
    int ctx_used() const;
    bool is_token() const;
    bool is_image() const;
    const Image& image() const;

  private:
    uint64_t word_ = 0;
};

bool
operator<(const Atom&, const Atom&);

bool
operator==(const Atom&, const Atom&);

} // namespace server
} // namespace lf
