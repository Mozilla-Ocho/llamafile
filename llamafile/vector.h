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
#include <__fwd/vector.h>
#include <algorithm>

namespace lf {

template <typename T>
bool vector_starts_with(const std::vector<T> &sequence, const std::vector<T> &prefix) {
    if (prefix.size() > sequence.size())
        return false;
    return std::equal(prefix.begin(), prefix.end(), sequence.begin());
}

template <typename T>
bool vector_ends_with(const std::vector<T> &sequence, const std::vector<T> &suffix) {
    if (suffix.size() > sequence.size())
        return false;
    return std::equal(sequence.rbegin(), sequence.rend(), suffix.rbegin());
}

template <typename T>
size_t vector_common_prefix_length(const std::vector<T> &a, const std::vector<T> &b) {
    size_t i = 0;
    size_t n = std::min(a.size(), b.size());
    const T *ai = a.data();
    const T *bi = b.data();
    while (i < n && ai[i] == bi[i])
        ++i;
    return i;
}

} // namespace lf
