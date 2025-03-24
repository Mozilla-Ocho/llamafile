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
#include <__fwd/string.h>
#include <__fwd/string_view.h>
#include <__fwd/vector.h>
#include <optional>
#include <sys/uio.h>

struct llama_model;

namespace lf {
namespace server {

class Atom;

ssize_t
safe_writev(int, const iovec*, int);

bool
atob(std::string_view, bool);

std::string_view
or_empty(std::optional<std::string_view> x);

void
atomize(const llama_model* model,
        std::vector<Atom>* result,
        std::string_view s,
        bool parse_special);

std::vector<Atom>
remove_old_image_atoms(const std::vector<Atom>&);

int
count_tokens(const std::vector<Atom>&);

bool
ends_with_incomplete_utf8(const std::string& str);

} // namespace server
} // namespace lf
