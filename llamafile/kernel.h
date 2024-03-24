#pragma once

#define BEGIN_KERNEL(RM, RN) \
    int ytiles = (m - m0) / RM; \
    int xtiles = (n - n0) / RN; \
    int tiles = ytiles * xtiles; \
    double duty = (double)tiles / nth; \
    if (duty < 1) \
        duty = 1; \
    double spot = duty * ith + .5; \
    int end = spot + duty; \
    int start = spot; \
    if (end > tiles) \
        end = tiles; \
    for (int job = start; job < end; ++job) { \
        int i = m0 + job / xtiles * RM; \
        int j = n0 + job % xtiles * RN;

#define END_KERNEL() }
