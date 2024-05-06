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

#include <stdio.h>

// constexpr int TT = 1024;
// constexpr int BN = 256;
// constexpr int BM = 128;
// constexpr int VE = 4;
// constexpr int BK = 32;
// constexpr int WN = 8;
// constexpr int WM = 128;
// constexpr int WNI = 1;
// constexpr int TN = 4;
// constexpr int TM = 8;

constexpr int TT = 256;
constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int VE = 16;
constexpr int WM = 32;
constexpr int WN = 32;
constexpr int WNI = 1;
constexpr int TM = 8;
constexpr int TN = 4;

#define SRC 4 // half

// run o//llamafile/cudaprops to get these values for your gpu
#define regsPerBlock 65536
#define sharedMemPerBlock 65536 // 49152
#define warpSize 32
#define cudaSharedOverhead 1024
#define maxThreadsPerBlock 1024
#define regsPerMultiprocessor 65536
#define sharedMemPerMultiprocessor 65536
#define maxThreadsPerMultiProcessor 2048

#define MIN(X, Y) ((Y) > (X) ? (X) : (Y))
#define ROUNDUP(X, K) (((X) + (K) - 1) & -(K))

int main(int argc, char *argv[]) {

    if (TT > maxThreadsPerBlock) {
        printf("TT[%d] > maxThreadsPerBlock[%d]\n", TT, maxThreadsPerBlock);
        return 1;
    }

    if (BM % WM) {
        printf("BM[%d] %% WM[%d] was %d (should be 0)\n", BM, WM, BM % WM);
        return 2;
    }

    if (BN % WN) {
        printf("BN[%d] %% WN[%d] was %d (should be 0)\n", BN, WN, BN % WN);
        return 3;
    }

    int WARPS = TT / warpSize;
    if ((BN / WN) * (BM / WM) != WARPS) {
        printf("TT[%d] / warpSize[%d] = WARPS[%d]\n", TT, warpSize, WARPS);
        printf("(BN[%d] / WN[%d]) * (BM[%d] / WM[%d]) = WARPS[%d]\n", BN, WN, BM, WM,
               (BN / WN) * (BM / WM));
        printf("try using TT = %d\n", (BN / WN) * (BM / WM) * warpSize);
        return 4;
    }

    if ((WM * WN) % (warpSize * TM * TN * WNI))
        return 6;

    int WMI = (WM * WN) / (warpSize * TM * TN * WNI);
    if ((WM % WMI) || (WN % WNI))
        return 7;

    if (BK % VE)
        return 8;

    if (BN % VE)
        return 9;

    if ((BM * BK) % (VE * TT)) {
        printf("(BM[%d] * BK[%d])[%d] %% (VE[%d] * TT[%d])[%d] was non-zero\n", BM, BK, BM * BK, VE,
               TT, VE * TT);
        return 10;
    }

    if ((BN * BK) % (VE * TT)) {
        printf("(BN[%d] * BK[%d])[%d] %% (VE[%d] * TT[%d])[%d] was non-zero\n", BN, BK, BN * BK, VE,
               TT, VE * TT);
        return 11;
    }

    int SMEM = (BK * BM) + (BK * BN); // in words
    if (SMEM * 4 + cudaSharedOverhead > sharedMemPerBlock) {
        printf("SMEM[%d] * 4 + cudaSharedOverhead[%d] > sharedMemPerBlock[%d]\n", SMEM,
               cudaSharedOverhead, sharedMemPerBlock);
        return 12;
    }
    SMEM *= SRC; // now in bytes

    int REGS = ROUNDUP(((WMI * TM) + (WNI * TN) + (WMI * TM * WNI * TN)) * warpSize, 256) *
               (TT / warpSize);
    if (REGS > regsPerBlock)
        return 10;

    int occupants = MIN(regsPerMultiprocessor / REGS,
                        MIN(maxThreadsPerMultiProcessor / TT,
                            sharedMemPerMultiprocessor / (SMEM + cudaSharedOverhead)));

    int occupancy =
        ((TT * occupants / (double)maxThreadsPerMultiProcessor) + //
         (REGS * occupants / (double)regsPerMultiprocessor) +
         ((SMEM + cudaSharedOverhead) * occupants / (double)sharedMemPerMultiprocessor)) /
            3 * 100 +
        .5;

    printf("\n");
    printf("constexpr int TT = %d;\n", TT);
    printf("constexpr int BN = %d;\n", BN);
    printf("constexpr int BM = %d;\n", BM);
    printf("constexpr int VE = %d;\n", VE);
    printf("constexpr int BK = %d;\n", BK);
    printf("constexpr int WN = %d;\n", WN);
    printf("constexpr int WM = %d;\n", WM);
    printf("constexpr int WNI = %d;\n", WNI);
    printf("constexpr int TN = %d;\n", TN);
    printf("constexpr int TM = %d;\n", TM);
    printf("constexpr int SMEM = %d * sizeof(SRC) + %d;\n", (SMEM / SRC), cudaSharedOverhead);
    printf("constexpr int REGS = %d;\n", REGS);
    printf("\n");

    printf("occupants = %d\n", occupants);
    printf("occupancy = %d\n", occupancy);
    printf("register occupancy = %d\n",
           (int)(REGS * occupants / (double)regsPerMultiprocessor * 100 + .5));
    printf("shared occupancy = %d\n", (int)((SMEM + cudaSharedOverhead) * occupants /
                                                (double)sharedMemPerMultiprocessor * 100 +
                                            .5));
    printf("thread occupancy = %d\n",
           (int)(TT * occupants / (double)maxThreadsPerMultiProcessor * 100 + .5));

    //
}
