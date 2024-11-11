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
#include <string_view>
#include <vector>

struct DataUri {
    std::string_view mime;
    std::string_view data;
    std::vector<std::pair<std::string_view, std::string_view>> params;

    DataUri();
    ~DataUri();
    size_t parse(std::string_view);
    bool has_param(std::string_view);
    std::string_view get_param(std::string_view);
    std::string decode();
};
