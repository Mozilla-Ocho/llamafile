#pragma once

inline int rand32(void) {
    static unsigned long long lcg = 1;
    lcg *= 6364136223846793005;
    lcg += 1442695040888963407;
    return lcg >> 32;
}

inline int popcount(unsigned x) {
    x = x - ((x >> 1) & 0x55555555);
    x = ((x >> 2) & 0x33333333) + (x & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = (x + (x >> 16));
    return (x + (x >> 8)) & 0x0000003F;
}

inline int hamming(int x, int y) {
    return popcount(x ^ y);
}

inline float float01(unsigned x) { // (0,1)
    return 1.f / 8388608 * ((x >> 9) + .5f);
}

inline float numba(void) { // (-1,1)
    return float01(rand32()) * 2.f - 1.f;
}

template <typename T>
void randomize(T *A, int n) {
    for (int i = 0; i < n; ++i)
        A[i] = numba();
}

template <typename T>
void randomize(int m, int n, T *A, int lda) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[lda * j + i] = numba();
}

template <typename T, typename U>
void broadcast(T *A, int n, U x) {
    for (int i = 0; i < n; ++i)
        A[i] = x;
}

template <typename T, typename U>
void broadcast(int m, int n, T *A, int lda, U x) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[lda * j + i] = x;
}
