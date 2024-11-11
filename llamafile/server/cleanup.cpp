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

#include "cleanup.h"
#include "llama.cpp/llama.h"
#include <unistd.h>
#include <vector>

namespace lf {
namespace server {

void
cleanup_fildes(void* arg)
{
    close((intptr_t)arg);
}

void
cleanup_float_vector(void* arg)
{
    delete (std::vector<float>*)arg;
}

void
cleanup_token_vector(void* arg)
{
    delete (std::vector<int>*)arg;
}

void
cleanup_llama_batch(void* arg)
{
    llama_batch* batch = (llama_batch*)arg;
    llama_batch_free(*batch);
    delete batch;
}

void
cleanup_llama_context(void* arg)
{
    llama_free((llama_context*)arg);
}

} // namespace server
} // namespace lf
