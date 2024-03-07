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

#include "tester.h"

void checkTinyblasWorksHHHH() {
    double TOLERANCE = .5;

    {
        half α = 1;
        half β = 0;
        int m = 577;
        int n = 577;
        int k = 64;
        int lda = k;
        int ldb = k;
        int ldc = m;
        cuda_memory<half> A{lda * m};
        cuda_memory<half> B{ldb * n};
        cuda_memory<half> C{ldc * n};
        cuda_memory<half> G{ldc * n};
        randomize(k, m, A.p, lda);
        randomize(k, n, B.p, ldb);
        cublas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        broadcast(m, n, C.p, ldc, TOMBSTONE);
        broadcast(m, n, G.p, ldc, TOMBSTONE);
        gemmref<float>(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        tinyblas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
        CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);
    }

    {
        half α = 1;
        half β = 0;
        int m = 4096;
        int n = 1024;
        int k = 2048;
        int lda = k;
        int ldb = k;
        int ldc = m;
        cuda_memory<half> A{lda * m};
        cuda_memory<half> B{ldb * n};
        cuda_memory<half> C{ldc * n};
        cuda_memory<half> G{ldc * n};
        randomize(k, m, A.p, lda);
        randomize(k, n, B.p, ldb);
        cublas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        broadcast(m, n, C.p, ldc, TOMBSTONE);
        broadcast(m, n, G.p, ldc, TOMBSTONE);
        gemmref<float>(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        tinyblas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
        CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);
    }

    {
        half α = 1;
        half β = 0;
        int m = 4096;
        int n = 1024;
        int k = 577;
        int lda = k;
        int ldb = k;
        int ldc = m;
        cuda_memory<half> A{lda * m};
        cuda_memory<half> B{ldb * n};
        cuda_memory<half> C{ldc * n};
        cuda_memory<half> G{ldc * n};
        randomize(k, m, A.p, lda);
        randomize(k, n, B.p, ldb);
        cublas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        broadcast(m, n, C.p, ldc, TOMBSTONE);
        broadcast(m, n, G.p, ldc, TOMBSTONE);
        gemmref<float>(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        tinyblas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
        CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);
    }

    test_matmul([&](int m, int n, int k, int l, half α, half β) {
        int lda = m + l;
        int ldb = k + l;
        int ldc = m + l;
        cuda_memory<half> A{lda * k};
        cuda_memory<half> B{ldb * n};
        cuda_memory<half> C{ldc * n};
        cuda_memory<half> G{ldc * n};
        randomize(m, k, A.p, lda);
        randomize(k, n, B.p, ldb);
        broadcast(m, n, G.p, ldc, TOMBSTONE);
        broadcast(m, n, C.p, ldc, TOMBSTONE);
        gemmref<float>(false, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        tinyblas(false, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
        CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);
    });
}

void checkTinyblasWorksHHHS() {
    double TOLERANCE = .5;

    float α = 1;
    float β = 0;
    int m = 4096;
    int n = 1024;
    int k = 577;
    int lda = k;
    int ldb = k;
    int ldc = m;
    cuda_memory<half> A{lda * m};
    cuda_memory<half> B{ldb * n};
    cuda_memory<half> C{ldc * n};
    cuda_memory<half> G{ldc * n};
    randomize(k, m, A.p, lda);
    randomize(k, n, B.p, ldb);
    cublas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
    broadcast(m, n, C.p, ldc, TOMBSTONE);
    broadcast(m, n, G.p, ldc, TOMBSTONE);
    gemmref(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
    tinyblas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
    CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);

    test_matmul([&](int m, int n, int k, int l, float α, float β) {
        int lda = m + l;
        int ldb = k + l;
        int ldc = m + l;
        cuda_memory<half> A{lda * k};
        cuda_memory<half> B{ldb * n};
        cuda_memory<half> C{ldc * n};
        cuda_memory<half> G{ldc * n};
        randomize(m, k, A.p, lda);
        randomize(k, n, B.p, ldb);
        broadcast(m, n, G.p, ldc, TOMBSTONE);
        broadcast(m, n, C.p, ldc, TOMBSTONE);
        gemmref(false, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        tinyblas(false, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
        CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);
    });
}

void checkTinyblasWorksHHSS() {
    double TOLERANCE = .5;

    float α = 1;
    float β = 0;
    int m = 4096;
    int n = 1024;
    int k = 1536;
    int lda = k;
    int ldb = k;
    int ldc = m;
    cuda_memory<half> A{lda * m};
    cuda_memory<half> B{ldb * n};
    cuda_memory<float> C{ldc * n};
    cuda_memory<float> G{ldc * n};
    randomize(k, m, A.p, lda);
    randomize(k, n, B.p, ldb);
    cublas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
    broadcast(m, n, G.p, ldc, TOMBSTONE);
    broadcast(m, n, C.p, ldc, TOMBSTONE);
    gemmref(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
    tinyblas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
    CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);

    test_matmul([&](int m, int n, int k, int l, float α, float β) {
        int lda = k + l;
        int ldb = k + l;
        int ldc = m + l;
        cuda_memory<half> A{lda * m};
        cuda_memory<half> B{ldb * n};
        cuda_memory<float> C{ldc * n};
        cuda_memory<float> G{ldc * n};
        randomize(k, m, A.p, lda);
        randomize(k, n, B.p, ldb);
        broadcast(m, n, G.p, ldc, TOMBSTONE);
        broadcast(m, n, C.p, ldc, TOMBSTONE);
        gemmref(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        tinyblas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
        CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);
    });
}

void checkTinyblasWorksSSSS() {
    double TOLERANCE = .5;

    float α = 1;
    float β = 0;
    int m = 4096;
    int n = 1024;
    int k = 1020;
    int lda = k;
    int ldb = k;
    int ldc = m;
    cuda_memory<float> A{lda * m};
    cuda_memory<float> B{ldb * n};
    cuda_memory<float> C{ldc * n};
    cuda_memory<float> G{ldc * n};
    randomize(k, m, A.p, lda);
    randomize(k, n, B.p, ldb);
    cublas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
    broadcast(m, n, C.p, ldc, TOMBSTONE);
    broadcast(m, n, G.p, ldc, TOMBSTONE);
    gemmref(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
    tinyblas(true, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
    CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);

    test_matmul([&](int m, int n, int k, int l, float α, float β) {
        int lda = m + l;
        int ldb = k + l;
        int ldc = m + l;
        cuda_memory<float> A{lda * k};
        cuda_memory<float> B{ldb * n};
        cuda_memory<float> C{ldc * n};
        cuda_memory<float> G{ldc * n};
        randomize(m, k, A.p, lda);
        randomize(k, n, B.p, ldb);
        broadcast(m, n, G.p, ldc, TOMBSTONE);
        broadcast(m, n, C.p, ldc, TOMBSTONE);
        gemmref(false, false, m, n, k, α, A.p, lda, B.p, ldb, β, G.p, ldc);
        tinyblas(false, false, m, n, k, α, A.p, lda, B.p, ldb, β, C.p, ldc);
        CHECK(TOLERANCE, m, n, k, G.p, ldc, C.p, ldc);
    });
}

int main(int argc, char *argv[]) {
    RUN(checkTinyblasWorksHHHH());
    RUN(checkTinyblasWorksSSSS());
    RUN(checkTinyblasWorksHHSS());
    RUN(checkTinyblasWorksHHHS());
}
