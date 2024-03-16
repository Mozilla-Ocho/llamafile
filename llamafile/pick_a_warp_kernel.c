// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
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
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>

// this program prints legal template args for tinyBLAS matmul_warp2d()
//
//     make o//llamafile/pick_a_warp_kernel
//     o//llamafile/pick_a_warp_kernel
//
// note that the sizes generated are in row major order, as used by the
// warp kernel implementation which is the opposite of the tinyBLAS API

#define SRC 2

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
#define ARRAYLEN(A) (sizeof(A) / sizeof(*(A)))

#define INTEGER(VAR, START, STOP)               \
    int VAR = START;                            \
    VAR <= STOP;                                \
    VAR++

#define TWOPOW(VAR, START, STOP)                \
    int VAR = START;                            \
    VAR <= STOP;                                \
    VAR *= 2

#define STRINGS 100000
char *strings[STRINGS];
atomic_int lock;
int results;

static const int kNumbers[] = {
      1,   2,   4,   8,  12,  14,  16,
     18,  20,  22,  24,  26,  28,  30,
     32,  64, 128, 256,
};

int cmp(const void *x, const void *y) {
  return strcmp(*(const char *const *)x,
                *(const char *const *)y);
}

int main(int argc, char *argv[]) {
#pragma omp parallel for collapse(4)
  for (int iBM = 0; iBM < ARRAYLEN(kNumbers); ++iBM)
  for (int iBN = 0; iBN < ARRAYLEN(kNumbers); ++iBN)
  for (int iBK = 0; iBK < ARRAYLEN(kNumbers); ++iBK)
  for (int iWM = 0; iWM < ARRAYLEN(kNumbers); ++iWM)
  for (int iWN = 0; iWN < ARRAYLEN(kNumbers); ++iWN)
  for (int BK = 4; BK <= 128; BK<<=1)
  for (int VE = 4; VE <= 32; VE<<=1)
  for (int TM = 1; TM <= 8; TM<<=1)
  for (int TN = 1; TN <= 8; TN<<=1)
  for (int WNI = 1; WNI <= 8; ++WNI)
  for (int TT = warpSize; TT <= maxThreadsPerBlock; TT += warpSize) {
      int BM = kNumbers[iBM];
      int BN = kNumbers[iBN];
      int BK = kNumbers[iBK];
      int WN = kNumbers[iWN];
      int WM = kNumbers[iWM];

      // with LLMs the m dimension is the most wild of
      // the three, since its size is often determined
      // by user input. as a result, it has a habit of
      // being small and also not aligned on a 2 power
      if (BM > BN) continue;

      // cargo culting boehm
      if (WM < WN) continue;

      // seems slightly better
      if (TN > TM) continue;

      // make warp tiles fit evenly in blocks
      if (BN % WN) continue;
      if (BM % WM) continue;

      // not worth bothering
      if (TM * TN < 24) continue;

      // banking possible
      if (BK % VE) continue;
      if (BN % VE) continue;
      if ((BM * BK) % (VE * TT)) continue;
      if ((BN * BK) % (VE * TT)) continue;

      // plan out warp tiles for each thread
      int WARPS = TT / warpSize;
      if ((BN / WN) * (BM / WM) != WARPS) continue;
      if ((WM * WN) % (warpSize * TM * TN * WNI)) continue;
      int WMI = (WM * WN) / (warpSize * TM * TN * WNI);
      if ((WM % WMI) || (WN % WNI)) continue;
      if (WM % WMI) continue;
      if (WN % WNI) continue;
      int WSUBN = WN / WNI;
      if (WSUBN % TN) continue;

      // compute number of bytes of shared memory required per block
      int SMEM = (BK * BM) + (BK * BN); // in words
      if (SMEM * SRC > sharedMemPerBlock) continue;
      SMEM *= SRC; // now in bytes

      // compute number of words used for thread local storage
      int REGS = ROUNDUP(((WMI * TM) + (WNI * TN) + (WMI * TM * WNI * TN)) * warpSize, 256) * (TT / warpSize);
      if (REGS > regsPerBlock) continue;

      int occupants = MIN(regsPerMultiprocessor / REGS,
                          MIN(maxThreadsPerMultiProcessor / TT,
                              sharedMemPerMultiprocessor / (SMEM + cudaSharedOverhead)));

      int register_occupancy = REGS * occupants / (double)regsPerMultiprocessor * 100 + .5;
      int shared_occupancy = (SMEM + cudaSharedOverhead) * occupants / (double)sharedMemPerMultiprocessor * 100 + .5;
      int thread_occupancy = TT * occupants / (double)maxThreadsPerMultiProcessor * 100 + .5;

      /* int score = register_occupancy + shared_occupancy + thread_occupancy; */

      int score =
              (TT > 256) +
              (BM == 128) +
              (BN == 64) +
              (BK == 64) +
              (VE == 16) +
              (WM == 32) +
              (WN == 32) +
              (WNI == 1) +
              (TM == 8) +
              (TN == 4);

      int n = 256;
      char *s = malloc(256);
      snprintf(s, n, "%07d constexpr int TT = %3d, BM = %3d, BN = %3d, BK = %3d, VE = %3d, WM = %3d, "
               "WN = %3d, WNI = %d, TM = %2d, TN = %3d; // %5d REGS, %5d SMEM, %2d occupants (%2d%% / %2d%% / %2d%%)",
               score, TT, BM, BN, BK, VE, WM, WN, WNI, TM, TN, REGS, SMEM, occupants,
               register_occupancy, shared_occupancy, thread_occupancy);

      while (atomic_exchange_explicit(&lock, 1, memory_order_acquire));
      if (results == STRINGS) exit(7);
      strings[results++] = s;
      atomic_store_explicit(&lock, 0, memory_order_release);
  }

  int printed = MIN(results, 500);
  qsort(strings, results, sizeof(char *), cmp);
  for (int i = results - printed; i < results; ++i)
      puts(strings[i]);

  fprintf(stderr, "note: printed %d of the %d legal kernels that exist\n", printed, results);
}
