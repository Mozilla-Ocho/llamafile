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

#include "buffer.h"
#include <sys/mman.h>
#include <unistd.h>

namespace lf {
namespace server {

static int pagesz = getpagesize();

Buffer::Buffer(size_t capacity) noexcept
  : i(0)
  , n(0)
  , c(capacity - pagesz)
  , p((char*)mmap(nullptr,
                  capacity,
                  PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS,
                  -1,
                  0))
{
    if (p == MAP_FAILED)
        __builtin_trap();
    if (c & (pagesz - 1))
        __builtin_trap();
    if (mprotect(p + c, pagesz, PROT_NONE))
        __builtin_trap();
}

Buffer::~Buffer() noexcept
{
    if (munmap(p, c + pagesz))
        __builtin_trap();
}

} // namespace server
} // namespace lf
