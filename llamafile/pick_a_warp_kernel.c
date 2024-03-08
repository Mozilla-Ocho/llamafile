// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;coding:utf-8 -*-
// vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8 :vi
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
// clang-format off

#include <stdio.h>
#include <stdlib.h>

// this program prints legal template args for tinyBLAS matmul_warp2d()
//
//     make o//llamafile/pick_a_warp_kernel
//     o//llamafile/pick_a_warp_kernel | sort
//
// note that the sizes generated are in row major order, as used by the
// warp kernel implementation which is the opposite of the tinyBLAS API

const int VECTOR = 16;

// run o//llamafile/cudaprops to get these values for your gpu
const int warpSize = 32;
const int regsPerBlock = 65536;
const int maxThreadsPerBlock = 1024;
const int sharedMemPerBlock = 49152;
const int warpsPerMultiprocessor = 48;
const int regsPerMultiprocessor = 65536;
const int sharedMemPerMultiprocessor = 65536;
const int maxThreadsPerMultiProcessor = 2048;

#define MIN(X, Y) ((Y) > (X) ? (X) : (Y))

#define INTEGER(VAR, START, STOP) \
  int VAR = START;                \
  VAR <= STOP;                    \
  VAR++

#define TWOPOW(VAR, START, STOP) \
  int VAR = START;               \
  VAR <= STOP;                   \
  VAR <<= 1

int main(int argc, char *argv[]) {
  for (TWOPOW(BM, 1, 256))
    for (TWOPOW(BN, 1, 128))
      for (TWOPOW(BK, 1, 64))
        for (TWOPOW(WM, 1, 256))
          for (TWOPOW(WN, 1, 256))
            for (INTEGER(WNI, 1, 8))
              for (TWOPOW(TM, 2, 16))
                for (TWOPOW(TN, 2, 16))
                  for (TWOPOW(TT, warpSize, maxThreadsPerBlock)) {
                    int warps, VE, WMI, SRC;

                    // make sure there's a solution for each word size
                    // for resource calculation we assume fp16 because
                    // that's the data type we use most but since it's
                    // possible to have f32 output type in practice we
                    // should only choose a kernel with >= 2 occupancy
                    for (SRC = 4; SRC >= 2; SRC >>= 1) {
                      warps = TT / warpSize;
                      WMI = (WM * WN) / (warpSize * TM * TN * WNI);
                      VE = VECTOR / SRC;
                      if (warps > warpsPerMultiprocessor) goto nope;
                      if (BN % WN || BM % WM) goto nope;
                      if ((BN / WN) * (BM / WM) != warps) goto nope;
                      if (BN % (VECTOR * TN)) goto nope;
                      if (BM % (VECTOR * TM)) goto nope;
                      if ((BM * BK) % (VE * TT)) goto nope;
                      if ((BN * BK) % (VE * TT)) goto nope;
                      if ((WM % WMI) || (WN % WNI)) goto nope;
                      if ((WM * WN) % (warpSize * TM * TN * WNI)) goto nope;
                      if (!((WM * WN) / (warpSize * TM * TN * WNI))) goto nope;

                      // with LLMs the m dimension is the most wild of
                      // the three, since its size is often determined
                      // by user input. as a result, it has a habit of
                      // being small and also not aligned on a 2 power
                      if (BM > BN) goto nope;

                      // cargo culting boehm
                      if (WM < WN) goto nope;

                      // seems slightly better
                      if (TN > TM) goto nope;
                    }

                    int regs = ((WMI * TM) + (WNI * TN) + (WMI * TM * WNI * TN)) * TT; // in 32-bit words
                    int shared = ((BK * BM) + (BK * BN)) * SRC; // in bytes
                    if (shared > sharedMemPerBlock) goto nope;
                    if (regs > regsPerBlock) goto nope;

                    int occupants = MIN(regsPerMultiprocessor / regs,
                                        MIN(warpsPerMultiprocessor / warps,
                                            sharedMemPerMultiprocessor / shared));
                    if (occupants < 2) goto nope;  // so there's room for sgemm float
                    int occupancy = ((regs * occupants / (double)regsPerMultiprocessor) +
                                     (shared * occupants / (double)sharedMemPerMultiprocessor)) / 2 * 100 + .5;
                    if (occupancy < 85) goto nope; // not worth printing

                    long long score = (long long)      // our goal in scoring is as follows (bigger is better)
                                      BK * TM * TN *   // - maximize outer product's algorithmic advantage
                                      occupants *      // - minimize resource usage
                                      TT *             // - maximize concurrency
                                      WM * WN;         // - maximum warp engage

                    printf("%012lld constexpr int TT=%4d, BM=%3d, BN=%3d, BK=%3d, WM=%3d, "
                           "WN=%3d, WNI=%d, TM=%2d, TN=%3d, REGS=%7d, SHARED=%7d; // %2d occupants (%d%%)\n",
                           score, TT, BM, BN, BK, WM, WN, WNI, TM, TN, regs, shared, occupants, occupancy);

                 nope:
                    (void)0;
                  }
}
