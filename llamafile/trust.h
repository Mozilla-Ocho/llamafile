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
#include <__fwd/string_view.h>
#include <__fwd/vector.h>

struct cidr {
    unsigned ip;
    unsigned bits;

    constexpr bool matches(unsigned x) const noexcept {
        unsigned mask = -1u << (32 - bits);
        return (x & mask) == (ip & mask);
    }
};

extern std::vector<cidr> FLAG_trust;

bool is_trusted_ip(unsigned) noexcept;
bool is_loopback_ip(unsigned) noexcept;
long parse_ip(const std::string_view &) noexcept;
bool parse_cidr(const std::string_view &, cidr *) noexcept;
