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

#include "llamafile/llamafile.h"
#include "llamafile/server/tokenbucket.h"
#include <climits>
#include <cosmo.h>
#include <cstdlib>

namespace lf {
namespace server {
namespace {

void
tokenbucket_test()
{
    FLAG_token_rate = .0001;
    FLAG_token_cidr = 24;
    tokenbucket_init();

    if (tokenbucket_acquire(0x7f000001) != 0)
        exit(1);
    if (tokenbucket_acquire(0x7f000002) != 1)
        exit(1);
    if (tokenbucket_acquire(0x7f000003) != 2)
        exit(1);
    if (tokenbucket_acquire(0x7f000100) != 0)
        exit(1);
    if (tokenbucket_acquire(0x7f000101) != 1)
        exit(1);

    tokenbucket_replenish();
    tokenbucket_replenish();

    if (tokenbucket_acquire(0x7f000003) != 1)
        exit(1);
    if (tokenbucket_acquire(0x7f000003) != 2)
        exit(1);
    if (tokenbucket_acquire(0x7f000101) != 0)
        exit(1);

    tokenbucket_replenish();
    tokenbucket_replenish();
    tokenbucket_replenish();

    if (tokenbucket_acquire(0x7f000003) != 0)
        exit(1);
    if (tokenbucket_acquire(0x7f000101) != 0)
        exit(1);

    tokenbucket_destroy();
    CheckForMemoryLeaks();
}

} // namespace
} // namespace server
} // namespace lf

int
main()
{
    lf::server::tokenbucket_test();
}
