// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#include "llamafile.h"
#include "log.h"
#include <cassert>

namespace {

struct SgemmAccountant {

    long long wont;

    long long ggml_ops;
    long long llamafile_ops;

    long long ggml_flops;
    long long llamafile_flops;

    SgemmAccountant() = default;

    ~SgemmAccountant() {
        if (!ggml_ops && !llamafile_ops)
            return;
        double total_ops = ggml_ops + llamafile_ops;
        double total_flops = ggml_flops + llamafile_flops;
        tinylogf("llamafile_sgemm: successfully accelerated %g%% of flops and %g%% of ops on cpu"
                 " (%lld cant %lld wont)\n",
                 llamafile_ops / total_ops * 100, llamafile_flops / total_flops * 100,
                 ggml_ops - wont, wont);
    }
};

struct SgemmAccountant g_sgemm_accountant;

} // namespace

void(llamafile_sgemm_was_used)(long long flops) {
    g_sgemm_accountant.llamafile_ops++;
    g_sgemm_accountant.llamafile_flops += flops;
}

void(llamafile_sgemm_was_not_used)(long long flops, bool wont) {
    g_sgemm_accountant.ggml_ops++;
    g_sgemm_accountant.ggml_flops += flops;
    g_sgemm_accountant.wont += wont;
}
