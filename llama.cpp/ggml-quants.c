// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include "ggml-quants.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <cosmo.h>

#ifdef __ARM_NEON

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>

#if !defined(__aarch64__)
inline static int32_t vaddvq_s16(int16x8_t v) {
    return
        (int32_t)vgetq_lane_s16(v, 0) + (int32_t)vgetq_lane_s16(v, 1) +
        (int32_t)vgetq_lane_s16(v, 2) + (int32_t)vgetq_lane_s16(v, 3) +
        (int32_t)vgetq_lane_s16(v, 4) + (int32_t)vgetq_lane_s16(v, 5) +
        (int32_t)vgetq_lane_s16(v, 6) + (int32_t)vgetq_lane_s16(v, 7);
}

inline static int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
    int16x4_t b0 = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
    return vcombine_s16(a0, b0);
}

inline static int32_t vaddvq_s32(int32x4_t v) {
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}
#endif

#else

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#else
#ifdef __POWER9_VECTOR__
#include <altivec.h>
#undef bool
#define bool _Bool
#else
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#if !defined(__riscv) && !defined(__s390__)
#include <immintrin.h>
#endif
#endif
#endif
#endif
#endif

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// horizontally add 8 floats
#define hsum_float_8(x) ({ \
    __m128 res = _mm256_extractf128_ps(x, 1); \
    res = _mm_add_ps(res, _mm256_castps256_ps128(x)); \
    res = _mm_add_ps(res, _mm_movehl_ps(res, res)); \
    res = _mm_add_ss(res, _mm_movehdup_ps(res)); \
    _mm_cvtss_f32(res); \
})

#ifdef __x86_64__

// add int16_t pairwise and return as float vector
__attribute__((__target__("avx")))
static inline __m256 sum_i16_pairs_float_avx(const __m128i xh, const __m128i xl) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
    const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
    const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
    return _mm256_cvtepi32_ps(summed_pairs);
}

// add int16_t pairwise and return as float vector
__attribute__((__target__("avx2,fma")))
static inline __m256 sum_i16_pairs_float_avx2(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

__attribute__((__target__("avx")))
static inline __m256 mul_sum_us8_pairs_float_avx(const __m256i ax, const __m256i sy) {
    const __m128i axl = _mm256_castsi256_si128(ax);
    const __m128i axh = _mm256_extractf128_si256(ax, 1);
    const __m128i syl = _mm256_castsi256_si128(sy);
    const __m128i syh = _mm256_extractf128_si256(sy, 1);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float_avx(doth, dotl);
}

__attribute__((__target__("avx2,fma")))
static inline __m256 mul_sum_us8_pairs_float_avx2(const __m256i ax, const __m256i sy) {
#if __AVXVNNI__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float_avx2(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
__attribute__((__target__("avx")))
static inline __m256 mul_sum_i8_pairs_float_avx(const __m256i x, const __m256i y) {
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extractf128_si256(x, 1);
    const __m128i yl = _mm256_castsi256_si128(y);
    const __m128i yh = _mm256_extractf128_si256(y, 1);
    // Get absolute values of x vectors
    const __m128i axl = _mm_sign_epi8(xl, xl);
    const __m128i axh = _mm_sign_epi8(xh, xh);
    // Sign the values of the y vectors
    const __m128i syl = _mm_sign_epi8(yl, xl);
    const __m128i syh = _mm_sign_epi8(yh, xh);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float_avx(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
__attribute__((__target__("avx2,fma")))
static inline __m256 mul_sum_i8_pairs_float_avx2(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float_avx2(ax, sy);
#endif
}

#endif /* __x86_64__ */

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

#if __AVX__ || __AVX2__ || __AVX512F__

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    const __m128i hi64 = _mm_unpackhi_epi64(a, a);
    const __m128i sum64 = _mm_add_epi32(hi64, a);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX2__) || defined(__AVX512F__)
// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);   // 0000_0000_abcd_0000
    bytes = _mm256_or_si256(bytes, bytes_srli_4);               // 0000_abcd_abcd_efgh
    return _mm256_cvtepi16_epi8(bytes);                         // abcd_efgh
#else
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
#endif
}
#elif defined(__AVX__)
// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
    __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytesl = _mm_or_si128(bytesl, bit_mask);
    bytesh = _mm_or_si128(bytesh, bit_mask);
    bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
    bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
    return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    const __m128i lowMask = _mm_set1_epi8(0xF);
    tmpl = _mm_and_si128(lowMask, tmpl);
    tmph = _mm_and_si128(lowMask, tmph);
    return MM256_SET_M128I(tmph, tmpl);
}

static inline __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m128i lowByte = _mm_set1_epi16( 0xFF );
    __m128i high = _mm_andnot_si128( lowByte, bytes1 );
    __m128i low = _mm_and_si128( lowByte, bytes1 );
    high = _mm_srli_epi16( high, 4 );
    bytes1 = _mm_or_si128( low, high );
    high = _mm_andnot_si128( lowByte, bytes2 );
    low = _mm_and_si128( lowByte, bytes2 );
    high = _mm_srli_epi16( high, 4 );
    bytes2 = _mm_or_si128( low, high );

    return _mm_packus_epi16( bytes1, bytes2);
}
#endif
#elif defined(__SSSE3__)
// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    __m128 res_0 =_mm_hadd_ps(a, b);
    __m128 res_1 =_mm_hadd_ps(c, d);
    __m128 res =_mm_hadd_ps(res_0, res_1);
    res =_mm_hadd_ps(res, res);
    res =_mm_hadd_ps(res, res);

    return _mm_cvtss_f32(res);
}
#endif // __AVX__ || __AVX2__ || __AVX512F__
#endif // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)

#if defined(__ARM_NEON)

#if !defined(__aarch64__)

inline static int32_t vaddvq_s32(int32x4_t v) {
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

inline static float vaddvq_f32(float32x4_t v) {
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

inline static float vmaxvq_f32(float32x4_t v) {
    return
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

inline static int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    return res;
}

#endif
#endif

#if defined(__ARM_NEON) || defined(__wasm_simd128__)
#define B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#define B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#define B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#define B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#define B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#define B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#define B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#define B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
static const uint64_t table_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
static const uint64_t table_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#endif

// reference implementation for deterministic creation of model files
void quantize_row_q4_0_reference(const float * restrict x, block_q4_0 * restrict y, int k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q4_0(const float * restrict x, void * restrict y, int k) {
    quantize_row_q4_0_reference(x, y, k);
}

void quantize_row_q4_1_reference(const float * restrict x, block_q4_1 * restrict y, int k) {
    const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q4_1(const float * restrict x, void * restrict y, int k) {
    quantize_row_q4_1_reference(x, y, k);
}

void quantize_row_q5_0_reference(const float * restrict x, block_q5_0 * restrict y, int k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}

void quantize_row_q5_0(const float * restrict x, void * restrict y, int k) {
    quantize_row_q5_0_reference(x, y, k);
}

void quantize_row_q5_1_reference(const float * restrict x, block_q5_1 * restrict y, int k) {
    const int qk = QK5_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 5) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
    }
}

void quantize_row_q5_1(const float * restrict x, void * restrict y, int k) {
    quantize_row_q5_1_reference(x, y, k);
}

// reference implementation for deterministic creation of model files
void quantize_row_q8_0_reference(const float * restrict x, block_q8_0 * restrict y, int k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

void quantize_row_q8_0(const float * restrict x, void * restrict vy, int k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
        }
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = wasm_v128_load(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
                                   wasm_f32x4_extract_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                                   wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_extract_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_extract_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_extract_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_extract_lane(vi, 3);
        }
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#elif defined(__riscv_v_intrinsic)

    size_t vl = __riscv_vsetvl_e32m4(QK8_0);

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(x+i*QK8_0, vl);

        vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m4_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);

        // convert to integer
        vint16m2_t   vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
        vint8m1_t    vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);

        // store result
        __riscv_vse8_v_i8m1(y[i].qs , vs, vl);
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_reference(x, y, k);
#endif
}

// reference implementation for deterministic creation of model files
void quantize_row_q8_1_reference(const float * restrict x, block_q8_1 * restrict y, int k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_1; j++) {
            const float v = x[i*QK8_1 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        int sum = 0;

        for (int j = 0; j < QK8_1/2; ++j) {
            const float v0 = x[i*QK8_1           + j]*id;
            const float v1 = x[i*QK8_1 + QK8_1/2 + j]*id;

            y[i].qs[          j] = roundf(v0);
            y[i].qs[QK8_1/2 + j] = roundf(v1);

            sum += y[i].qs[          j];
            sum += y[i].qs[QK8_1/2 + j];
        }

        y[i].s = sum*d;
    }
}

void quantize_row_q8_1(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);

            accv = vaddq_s32(accv, vi);
        }

        y[i].s = d * vaddvq_s32(accv);
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = wasm_v128_load(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
                                   wasm_f32x4_extract_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                                   wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        v128_t accv = wasm_i32x4_splat(0);

        for (int j = 0; j < 8; j++) {
            const v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_extract_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_extract_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_extract_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_extract_lane(vi, 3);

            accv = wasm_i32x4_add(accv, vi);
        }

        y[i].s = d * (wasm_i32x4_extract_lane(accv, 0) +
                      wasm_i32x4_extract_lane(accv, 1) +
                      wasm_i32x4_extract_lane(accv, 2) +
                      wasm_i32x4_extract_lane(accv, 3));
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = d;
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Compute the sum of the quants and set y[i].s
        y[i].s = d * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = _mm_add_epi32(_mm_add_epi32(ni0, ni1), _mm_add_epi32(ni2, ni3));
        const __m128i s1 = _mm_add_epi32(_mm_add_epi32(ni4, ni5), _mm_add_epi32(ni6, ni7));
        y[i].s = d * hsum_i32_4(_mm_add_epi32(s0, s1));

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#elif defined(__riscv_v_intrinsic)

    size_t vl = __riscv_vsetvl_e32m4(QK8_1);

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(x+i*QK8_1, vl);

        vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m4_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d  = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);

        // convert to integer
        vint16m2_t   vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
        vint8m1_t    vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);

        // store result
        __riscv_vse8_v_i8m1(y[i].qs , vs, vl);

        // compute sum for y[i].s
        vint16m1_t tmp2 = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t vwrs = __riscv_vwredsum_vs_i8m1_i16m1(vs, tmp2, vl);

        // set y[i].s
        int sum = __riscv_vmv_x_s_i16m1_i16(vwrs);
        y[i].s = sum*d;
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_reference(x, y, k);
#endif
}

void dequantize_row_q4_0(const block_q4_0 * restrict x, float * restrict y, int k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q4_1(const block_q4_1 * restrict x, float * restrict y, int k) {
    static const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

void dequantize_row_q5_0(const block_q5_0 * restrict x, float * restrict y, int k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q5_1(const block_q5_1 * restrict x, float * restrict y, int k) {
    static const int qk = QK5_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

void dequantize_row_q8_0(const block_q8_0 * restrict x, float * restrict y, int k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

//
// 2-6 bit quantization in super-blocks
//

//
// ===================== Helper functions
//
static inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, int rmse_type) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-30f) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    int weight_type = rmse_type%2;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = weight_type == 1 ? x[i] * x[i] : 1;
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = sumlx/suml2;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = weight_type == 1 ? x[i] * x[i] : 1;
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static float make_q3_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return sumlx / suml2;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

static float make_qkx1_quants(int n, int nmax, const float * restrict x, uint8_t * restrict L, float * restrict the_min,
        int ntry, float alpha) {
    float min = x[0];
    float max = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
    }
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = 0;
        return 0.f;
    }
    if (min > 0) min = 0;
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    for (int itry = 0; itry < ntry; ++itry) {
        float sumlx = 0; int suml2 = 0;
        bool did_change = false;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            if (l != L[i]) {
                L[i] = l;
                did_change = true;
            }
            sumlx += (x[i] - min)*l;
            suml2 += l*l;
        }
        scale = sumlx/suml2;
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += x[i] - scale*L[i];
        }
        min = alpha*min + (1 - alpha)*sum/n;
        if (min > 0) min = 0;
        iscale = 1/scale;
        if (!did_change) break;
    }
    *the_min = -min;
    return scale;
}

static float make_qkx2_quants(int n, int nmax, const float * restrict x, const float * restrict weights,
        uint8_t * restrict L, float * restrict the_min, uint8_t * restrict Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

#if QK_K == 256
static inline void get_scale_min_k4(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

//========================- 2-bit (de)-quantization

void quantize_row_q2_K_reference(const float * restrict x, block_q2_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
        } else {
            y[i].dmin = GGML_FP32_TO_FP16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[16*j + ii] = l;
            }
        }

#if QK_K == 256
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }
#else
        for (int l = 0; l < 16; ++l) {
            y[i].qs[l] = L[l] | (L[l + 16] << 2) | (L[l + 32] << 4) | (L[l + 48] << 6);
        }
#endif

        x += QK_K;

    }
}

void dequantize_row_q2_K(const block_q2_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * q = x[i].qs;

#if QK_K == 256
        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
#else
        float dl1 = d * (x[i].scales[0] & 0xF), ml1 = min * (x[i].scales[0] >> 4);
        float dl2 = d * (x[i].scales[1] & 0xF), ml2 = min * (x[i].scales[1] >> 4);
        float dl3 = d * (x[i].scales[2] & 0xF), ml3 = min * (x[i].scales[2] >> 4);
        float dl4 = d * (x[i].scales[3] & 0xF), ml4 = min * (x[i].scales[3] >> 4);
        for (int l = 0; l < 16; ++l) {
            y[l+ 0] = dl1 * ((int8_t)((q[l] >> 0) & 3)) - ml1;
            y[l+16] = dl2 * ((int8_t)((q[l] >> 2) & 3)) - ml2;
            y[l+32] = dl3 * ((int8_t)((q[l] >> 4) & 3)) - ml3;
            y[l+48] = dl4 * ((int8_t)((q[l] >> 6) & 3)) - ml4;
        }
        y += QK_K;
#endif
    }
}

void quantize_row_q2_K(const float * restrict x, void * restrict vy, int k) {
    quantize_row_q2_K_reference(x, vy, k);
}

size_t ggml_quantize_q2_K(const float * restrict src, void * restrict dst, int n, int k, int64_t * restrict hist) {
    (void)hist; // TODO: collect histograms

    for (int j = 0; j < n; j += k) {
        block_q2_K * restrict y = (block_q2_K *)dst + j/QK_K;
        quantize_row_q2_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q2_K));
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K_reference(const float * restrict x, block_q3_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float scales[QK_K / 16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);
            float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

#if QK_K == 256
        memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int8_t l = nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } else {
                    y[i].scales[j-8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } else {
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }

        int8_t sc;
        for (int j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[16*j + ii] = l + 4;
            }
        }
#else
        if (max_scale) {
            float iscale = -8.f/max_scale;
            for (int j = 0; j < QK_K/16; j+=2) {
                int l1 = nearest_int(iscale*scales[j]);
                l1 = 8 + MAX(-8, MIN(7, l1));
                int l2 = nearest_int(iscale*scales[j+1]);
                l2 = 8 + MAX(-8, MIN(7, l2));
                y[i].scales[j/2] = l1 | (l2 << 4);
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } else {
            for (int j = 0; j < QK_K/16; j+=2) {
                y[i].scales[j/2] = 0;
            }
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int s = j%2 == 0 ? y[i].scales[j/2] & 0xF : y[i].scales[j/2] >> 4;
            float d = GGML_FP16_TO_FP32(y[i].d) * (s - 8);
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[16*j + ii] = l + 4;
            }
        }
#endif

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
#if QK_K == 256
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }
#else
        for (int l = 0; l < 16; ++l) {
            y[i].qs[l] = L[l] | (L[l + 16] << 2) | (L[l + 32] << 4) | (L[l + 48] << 6);
        }
#endif

        x += QK_K;
    }
}

#if QK_K == 256
void dequantize_row_q3_K(const block_q3_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}
#else
void dequantize_row_q3_K(const block_q3_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    assert(QK_K == 64);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d_all = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;

        const float d1 = d_all * ((x[i].scales[0] & 0xF) - 8);
        const float d2 = d_all * ((x[i].scales[0] >>  4) - 8);
        const float d3 = d_all * ((x[i].scales[1] & 0xF) - 8);
        const float d4 = d_all * ((x[i].scales[1] >>  4) - 8);

        for (int l=0; l<8; ++l) {
            uint8_t h = hm[l];
            y[l+ 0] = d1 * ((int8_t)((q[l+0] >> 0) & 3) - ((h & 0x01) ? 0 : 4));
            y[l+ 8] = d1 * ((int8_t)((q[l+8] >> 0) & 3) - ((h & 0x02) ? 0 : 4));
            y[l+16] = d2 * ((int8_t)((q[l+0] >> 2) & 3) - ((h & 0x04) ? 0 : 4));
            y[l+24] = d2 * ((int8_t)((q[l+8] >> 2) & 3) - ((h & 0x08) ? 0 : 4));
            y[l+32] = d3 * ((int8_t)((q[l+0] >> 4) & 3) - ((h & 0x10) ? 0 : 4));
            y[l+40] = d3 * ((int8_t)((q[l+8] >> 4) & 3) - ((h & 0x20) ? 0 : 4));
            y[l+48] = d4 * ((int8_t)((q[l+0] >> 6) & 3) - ((h & 0x40) ? 0 : 4));
            y[l+56] = d4 * ((int8_t)((q[l+8] >> 6) & 3) - ((h & 0x80) ? 0 : 4));
        }
        y += QK_K;
    }
}
#endif

void quantize_row_q3_K(const float * restrict x, void * restrict vy, int k) {
    quantize_row_q3_K_reference(x, vy, k);
}

size_t ggml_quantize_q3_K(const float * restrict src, void * restrict dst, int n, int k, int64_t * restrict hist) {
    (void)hist; // TODO: collect histograms

    for (int j = 0; j < n; j += k) {
        block_q3_K * restrict y = (block_q3_K *)dst + j/QK_K;
        quantize_row_q3_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q3_K));
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K_reference(const float * restrict x, block_q4_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

#if QK_K == 256
        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }
#else
        const float s_factor = 15.f;
        float inv_scale = max_scale > 0 ? s_factor/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? s_factor/max_min   : 0.f;
        int d1 = nearest_int(inv_scale*scales[0]);
        int m1 = nearest_int(inv_min*mins[0]);
        int d2 = nearest_int(inv_scale*scales[1]);
        int m2 = nearest_int(inv_min*mins[1]);
        y[i].scales[0] = d1 | (m1 << 4);
        y[i].scales[1] = d2 | (m2 << 4);
        y[i].d[0] = GGML_FP32_TO_FP16(max_scale/s_factor);
        y[i].d[1] = GGML_FP32_TO_FP16(max_min/s_factor);

        float sumlx = 0;
        int   suml2 = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            const uint8_t sd = y[i].scales[j] & 0xF;
            const uint8_t sm = y[i].scales[j] >>  4;
            const float d = GGML_FP16_TO_FP32(y[i].d[0]) * sd;
            if (!d) continue;
            const float m = GGML_FP16_TO_FP32(y[i].d[1]) * sm;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + m)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
                sumlx += (x[32*j + ii] + m)*l*sd;
                suml2 += l*l*sd*sd;
            }
        }
        if (suml2) {
            y[i].d[0] = GGML_FP32_TO_FP16(sumlx/suml2);
        }
#endif
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;

    }
}

void dequantize_row_q4_K(const block_q4_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * q = x[i].qs;

#if QK_K == 256

        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
#else
        const float dall = GGML_FP16_TO_FP32(x[i].d[0]);
        const float mall = GGML_FP16_TO_FP32(x[i].d[1]);
        const float d1 = dall * (x[i].scales[0] & 0xF), m1 = mall * (x[i].scales[0] >> 4);
        const float d2 = dall * (x[i].scales[1] & 0xF), m2 = mall * (x[i].scales[1] >> 4);
        for (int l = 0; l < 32; ++l) {
            y[l+ 0] = d1 * (q[l] & 0xF) - m1;
            y[l+32] = d2 * (q[l] >>  4) - m2;
        }
        y += QK_K;
#endif

    }
}

void quantize_row_q4_K(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK_K == 0);
    block_q4_K * restrict y = vy;
    quantize_row_q4_K_reference(x, y, k);
}

size_t ggml_quantize_q4_K(const float * restrict src, void * restrict dst, int n, int k, int64_t * restrict hist) {
    assert(k % QK_K == 0);
    (void)hist; // TODO: collect histograms

    for (int j = 0; j < n; j += k) {
        block_q4_K * restrict y = (block_q4_K *)dst + j/QK_K;
        quantize_row_q4_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q4_K));
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K_reference(const float * restrict x, block_q5_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

#if QK_K == 256
    uint8_t L[QK_K];
    float mins[QK_K/32];
    float scales[QK_K/32];
    float weights[32];
    uint8_t Laux[32];
#else
    int8_t L[QK_K];
    float scales[QK_K/16];
#endif

    for (int i = 0; i < nb; i++) {

#if QK_K == 256

        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * restrict qh = y[i].qh;
        uint8_t * restrict ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }
#else
        float max_scale = 0, amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_qx_quants(16, 16, x + 16*j, L + 16*j, 1);
            float abs_scale = fabsf(scales[j]);
            if (abs_scale > amax) {
                amax = abs_scale;
                max_scale = scales[j];
            }
        }

        float iscale = -128.f/max_scale;
        for (int j = 0; j < QK_K/16; ++j) {
            int l = nearest_int(iscale*scales[j]);
            y[i].scales[j] = MAX(-128, MIN(127, l));
        }
        y[i].d = GGML_FP32_TO_FP16(1/iscale);

        for (int j = 0; j < QK_K/16; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-16, MIN(15, l));
                L[16*j + ii] = l + 16;
            }
        }

        uint8_t * restrict qh = y[i].qh;
        uint8_t * restrict ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        for (int j = 0; j < 32; ++j) {
            int jm = j%8;
            int is = j/8;
            int l1 = L[j];
            if (l1 > 15) {
                l1 -= 16; qh[jm] |= (1 << is);
            }
            int l2 = L[j + 32];
            if (l2 > 15) {
                l2 -= 16; qh[jm] |= (1 << (4 + is));
            }
            ql[j] = l1 | (l2 << 4);
        }
#endif

        x += QK_K;

    }
}

void dequantize_row_q5_K(const block_q5_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

#if QK_K == 256

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
#else
        float d = GGML_FP16_TO_FP32(x[i].d);
        const int8_t * restrict s = x[i].scales;
        for (int l = 0; l < 8; ++l) {
            y[l+ 0] = d * s[0] * ((ql[l+ 0] & 0xF) - (qh[l] & 0x01 ? 0 : 16));
            y[l+ 8] = d * s[0] * ((ql[l+ 8] & 0xF) - (qh[l] & 0x02 ? 0 : 16));
            y[l+16] = d * s[1] * ((ql[l+16] & 0xF) - (qh[l] & 0x04 ? 0 : 16));
            y[l+24] = d * s[1] * ((ql[l+24] & 0xF) - (qh[l] & 0x08 ? 0 : 16));
            y[l+32] = d * s[2] * ((ql[l+ 0] >>  4) - (qh[l] & 0x10 ? 0 : 16));
            y[l+40] = d * s[2] * ((ql[l+ 8] >>  4) - (qh[l] & 0x20 ? 0 : 16));
            y[l+48] = d * s[3] * ((ql[l+16] >>  4) - (qh[l] & 0x40 ? 0 : 16));
            y[l+56] = d * s[3] * ((ql[l+24] >>  4) - (qh[l] & 0x80 ? 0 : 16));
        }
        y += QK_K;
#endif
    }
}

void quantize_row_q5_K(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK_K == 0);
    block_q5_K * restrict y = vy;
    quantize_row_q5_K_reference(x, y, k);
}

size_t ggml_quantize_q5_K(const float * restrict src, void * restrict dst, int n, int k, int64_t * restrict hist) {
    assert(k % QK_K == 0);
    (void)hist; // TODO: collect histograms

    for (int j = 0; j < n; j += k) {
        block_q5_K * restrict y = (block_q5_K *)dst + j/QK_K;
        quantize_row_q5_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q5_K));
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K_reference(const float * restrict x, block_q6_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (!max_abs_scale) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * restrict ql = y[i].ql;
        uint8_t * restrict qh = y[i].qh;
#if QK_K == 256
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }
#else
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = L[l +  0] & 0xF;
            const uint8_t q2 = L[l + 32] & 0xF;
            ql[l] = q1 | (q2 << 4);
        }
        for (int l = 0; l < 16; ++l) {
            qh[l] = (L[l] >> 4) | ((L[l + 16] >> 4) << 2) | ((L[l + 32] >> 4) << 4) | ((L[l + 48] >> 4) << 6);
        }
#endif

        x += QK_K;

    }
}

void dequantize_row_q6_K(const block_q6_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict ql = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict sc = x[i].scales;

#if QK_K == 256
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
#else
        for (int l = 0; l < 16; ++l) {
            const int8_t q1 = (int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l+16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l+16]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l+ 0] = d * sc[0] * q1;
            y[l+16] = d * sc[1] * q2;
            y[l+32] = d * sc[2] * q3;
            y[l+48] = d * sc[3] * q4;
        }
        y  += 64;
#endif

    }
}

void quantize_row_q6_K(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK_K == 0);
    block_q6_K * restrict y = vy;
    quantize_row_q6_K_reference(x, y, k);
}

size_t ggml_quantize_q6_K(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK_K == 0);
    (void)hist; // TODO: collect histograms

    for (int j = 0; j < n; j += k) {
        block_q6_K * restrict y = (block_q6_K *)dst + j/QK_K;
        quantize_row_q6_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q6_K));
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K_reference(const float * restrict x, block_q8_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        const float iscale = -128.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

void dequantize_row_q8_K(const block_q8_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

void quantize_row_q8_K(const float * restrict x, void * restrict y, int k) {
    quantize_row_q8_K_reference(x, y, k);
}

//===================================== Dot ptoducts =================================

//
// Helper functions
//
#if __x86_64__
// shuffles to pick the required scales in dot products
__attribute__((__target__("avx")))
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
__attribute__((__target__("avx")))
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
__attribute__((__target__("avx")))
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#endif

void ggml_vec_dot_q4_0_q8_0(int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);

    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    assert(nb % 2 == 0); // TODO: handle odd nb

    for (int i = 0; i < nb; i += 2) {
        const block_q4_0 * restrict x0 = &x[i + 0];
        const block_q4_0 * restrict x1 = &x[i + 1];
        const block_q8_0 * restrict y0 = &y[i + 0];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1ls), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hs), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps( GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d) );

        __m256i bx = bytes_from_nibbles_32(x[i].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        bx = _mm256_sub_epi8( bx, off );

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float_avx2(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    *s = hsum_float_8(acc);
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps( GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d) );

        const __m128i lowMask = _mm_set1_epi8(0xF);
        const __m128i off = _mm_set1_epi8(8);

        const __m128i tmp = _mm_loadu_si128((const __m128i *)x[i].qs);

        __m128i bx = _mm_and_si128(lowMask, tmp);
        __m128i by = _mm_loadu_si128((const __m128i *)y[i].qs);
        bx = _mm_sub_epi8(bx, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx, by);

        bx = _mm_and_si128(lowMask, _mm_srli_epi64(tmp, 4));
        by = _mm_loadu_si128((const __m128i *)(y[i].qs + 16));
        bx = _mm_sub_epi8(bx, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx, by);

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps(MM256_SET_M128I(i32_0, i32_1));

        // Apply the scale, and accumulate
        acc = _mm256_add_ps(_mm256_mul_ps( d, p ), acc);
    }

    *s = hsum_float_8(acc);
#elif defined(__SSSE3__)
    // set constants
    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    // First round without accumulation
    {
        _mm_prefetch(&x[0] + sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[0] + sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = _mm_set1_ps( GGML_FP16_TO_FP32(x[0].d) * GGML_FP16_TO_FP32(y[0].d) );

        const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *)x[0].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
        __m128i by_0 = _mm_loadu_si128((const __m128i *)y[0].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
        __m128i by_1 = _mm_loadu_si128((const __m128i *)(y[0].qs + 16));
        bx_1 = _mm_sub_epi8(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        _mm_prefetch(&x[1] + sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[1] + sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = _mm_set1_ps( GGML_FP16_TO_FP32(x[1].d) * GGML_FP16_TO_FP32(y[1].d) );

        const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i *)x[1].qs);

        __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
        __m128i by_2 = _mm_loadu_si128((const __m128i *)y[1].qs);
        bx_2 = _mm_sub_epi8(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
        __m128i by_3 = _mm_loadu_si128((const __m128i *)(y[1].qs + 16));
        bx_3 = _mm_sub_epi8(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = _mm_cvtepi32_ps(i32_0);
        __m128 p1 = _mm_cvtepi32_ps(i32_1);
        __m128 p2 = _mm_cvtepi32_ps(i32_2);
        __m128 p3 = _mm_cvtepi32_ps(i32_3);

        // Apply the scale
        acc_0 = _mm_mul_ps( d_0_1, p0 );
        acc_1 = _mm_mul_ps( d_0_1, p1 );
        acc_2 = _mm_mul_ps( d_2_3, p2 );
        acc_3 = _mm_mul_ps( d_2_3, p3 );
    }

    assert(nb % 2 == 0); // TODO: handle odd nb

    // Main loop
    for (int i = 2; i < nb; i+=2) {
        _mm_prefetch(&x[i] + sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[i] + sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = _mm_set1_ps( GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d) );

        const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *)x[i].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
        __m128i by_0 = _mm_loadu_si128((const __m128i *)y[i].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
        __m128i by_1 = _mm_loadu_si128((const __m128i *)(y[i].qs + 16));
        bx_1 = _mm_sub_epi8(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        _mm_prefetch(&x[i] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[i] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = _mm_set1_ps( GGML_FP16_TO_FP32(x[i + 1].d) * GGML_FP16_TO_FP32(y[i + 1].d) );

        const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i *)x[i + 1].qs);

        __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
        __m128i by_2 = _mm_loadu_si128((const __m128i *)y[i + 1].qs);
        bx_2 = _mm_sub_epi8(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
        __m128i by_3 = _mm_loadu_si128((const __m128i *)(y[i + 1].qs + 16));
        bx_3 = _mm_sub_epi8(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = _mm_cvtepi32_ps(i32_0);
        __m128 p1 = _mm_cvtepi32_ps(i32_1);
        __m128 p2 = _mm_cvtepi32_ps(i32_2);
        __m128 p3 = _mm_cvtepi32_ps(i32_3);

        // Apply the scale
        __m128 p0_d = _mm_mul_ps( d_0_1, p0 );
        __m128 p1_d = _mm_mul_ps( d_0_1, p1 );
        __m128 p2_d = _mm_mul_ps( d_2_3, p2 );
        __m128 p3_d = _mm_mul_ps( d_2_3, p3 );

        // Acummulate
        acc_0 = _mm_add_ps(p0_d, acc_0);
        acc_1 = _mm_add_ps(p1_d, acc_1);
        acc_2 = _mm_add_ps(p2_d, acc_2);
        acc_3 = _mm_add_ps(p3_d, acc_3);
    }

    *s = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#elif defined(__riscv_v_intrinsic)
    float sumf = 0.0;

    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    for (int i = 0; i < nb; i++) {
        // load elements
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[i].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[i].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[i].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8mf2_t x_a = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_l = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vint8mf2_t x_ai = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t x_li = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        // subtract offset
        vint8mf2_t v0 = __riscv_vsub_vx_i8mf2(x_ai, 8, vl);
        vint8mf2_t v1 = __riscv_vsub_vx_i8mf2(x_li, 8, vl);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += sumi*GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d);
    }

    *s = sumf;
#else
    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[i].qs[j] & 0x0F) - 8;
            const int v1 = (x[i].qs[j] >>   4) - 8;

            sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk/2]);
        }

        sumf += sumi*GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d);
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_q4_1_q8_1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);

    const block_q4_1 * restrict x = vx;
    const block_q8_1 * restrict y = vy;

    // TODO: add WASM SIMD
#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs = 0;

    assert(nb % 2 == 0); // TODO: handle odd nb

    for (int i = 0; i < nb; i += 2) {
        const block_q4_1 * restrict x0 = &x[i + 0];
        const block_q4_1 * restrict x1 = &x[i + 1];
        const block_q8_1 * restrict y0 = &y[i + 0];
        const block_q8_1 * restrict y1 = &y[i + 1];

        summs += GGML_FP16_TO_FP32(x0->m) * y0->s + GGML_FP16_TO_FP32(x1->m) * y1->s;

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0l, v1_0l), v0_0h, v1_0h);
        const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1l, v1_1l), v0_1h, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d)*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d)*y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0l), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0l), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0h), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0h), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1l), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1l), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1h), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1h), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), GGML_FP16_TO_FP32(x0->d)*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), GGML_FP16_TO_FP32(x1->d)*y1->d);
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;
#elif defined(__AVX2__) || defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    // Main loop
    for (int i = 0; i < nb; ++i) {
        const float d0 = GGML_FP16_TO_FP32(x[i].d);
        const float d1 = y[i].d;

        summs += GGML_FP16_TO_FP32(x[i].m) * y[i].s;

        const __m256 d0v = _mm256_set1_ps( d0 );
        const __m256 d1v = _mm256_set1_ps( d1 );

        // Compute combined scales
        const __m256 d0d1 = _mm256_mul_ps( d0v, d1v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        const __m256i bx = bytes_from_nibbles_32(x[i].qs);
        const __m256i by = _mm256_loadu_si256( (const __m256i *)y[i].qs );

        const __m256 xy = mul_sum_us8_pairs_float_avx2(bx, by);

        // Accumulate d0*d1*x*y
#if defined(__AVX2__)
        acc = _mm256_fmadd_ps( d0d1, xy, acc );
#else
        acc = _mm256_add_ps( _mm256_mul_ps( d0d1, xy ), acc );
#endif
    }

    *s = hsum_float_8(acc) + summs;
#elif defined(__riscv_v_intrinsic)
    float sumf = 0.0;

    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    for (int i = 0; i < nb; i++) {
        // load elements
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[i].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[i].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[i].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8mf2_t x_a = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_l = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vint8mf2_t v0 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t v1 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_FP16_TO_FP32(x[i].d)*y[i].d)*sumi + GGML_FP16_TO_FP32(x[i].m)*y[i].s;
    }

    *s = sumf;
#else
    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[i].qs[j] & 0x0F);
            const int v1 = (x[i].qs[j] >>   4);

            sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk/2]);
        }

        sumf += (GGML_FP16_TO_FP32(x[i].d)*y[i].d)*sumi + GGML_FP16_TO_FP32(x[i].m)*y[i].s;
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_q5_0_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(qk == QK5_0);

    const block_q5_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    assert(nb % 2 == 0); // TODO: handle odd nb

    for (int i = 0; i < nb; i += 2) {
        const block_q5_0 * restrict x0 = &x[i];
        const block_q5_0 * restrict x1 = &x[i + 1];
        const block_q8_0 * restrict y0 = &y[i];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        // extract the 5th bit via lookup table ((!b) << 4)
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        tmp0[0] = table_b2b_1[(qh0 >>  0) & 0xFF];
        tmp0[1] = table_b2b_1[(qh0 >>  8) & 0xFF];
        tmp0[2] = table_b2b_1[(qh0 >> 16) & 0xFF];
        tmp0[3] = table_b2b_1[(qh0 >> 24)       ];

        tmp1[0] = table_b2b_1[(qh1 >>  0) & 0xFF];
        tmp1[1] = table_b2b_1[(qh1 >>  8) & 0xFF];
        tmp1[2] = table_b2b_1[(qh1 >> 16) & 0xFF];
        tmp1[3] = table_b2b_1[(qh1 >> 24)       ];

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        const int8x16_t v0_0lf = vsubq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vsubq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vsubq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vsubq_s8(v0_1h, qhh1);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0lf), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lf), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hf), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hf), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1lf), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1lf), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hf), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hf), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__wasm_simd128__)
    v128_t sumv = wasm_f32x4_splat(0.0f);

    uint32_t qh;
    uint64_t tmp[4];

    // TODO: check if unrolling this is better
    for (int i = 0; i < nb; ++i) {
        const block_q5_0 * restrict x0 = &x[i];
        const block_q8_0 * restrict y0 = &y[i];

        const v128_t m4b  = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        memcpy(&qh, x0->qh, sizeof(qh));

        tmp[0] = table_b2b_1[(qh >>  0) & 0xFF];
        tmp[1] = table_b2b_1[(qh >>  8) & 0xFF];
        tmp[2] = table_b2b_1[(qh >> 16) & 0xFF];
        tmp[3] = table_b2b_1[(qh >> 24)       ];

        const v128_t qhl = wasm_v128_load(tmp + 0);
        const v128_t qhh = wasm_v128_load(tmp + 2);

        const v128_t v0 = wasm_v128_load(x0->qs);

        // 4-bit -> 8-bit
        const v128_t v0l = wasm_v128_and (v0, m4b);
        const v128_t v0h = wasm_u8x16_shr(v0, 4);

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        const v128_t v0lf = wasm_i8x16_sub(v0l, qhl);
        const v128_t v0hf = wasm_i8x16_sub(v0h, qhh);

        // load y
        const v128_t v1l = wasm_v128_load(y0->qs);
        const v128_t v1h = wasm_v128_load(y0->qs + 16);

        // int8x16 -> int16x8
        const v128_t v0lfl = wasm_i16x8_extend_low_i8x16 (v0lf);
        const v128_t v0lfh = wasm_i16x8_extend_high_i8x16(v0lf);
        const v128_t v0hfl = wasm_i16x8_extend_low_i8x16 (v0hf);
        const v128_t v0hfh = wasm_i16x8_extend_high_i8x16(v0hf);

        const v128_t v1ll = wasm_i16x8_extend_low_i8x16 (v1l);
        const v128_t v1lh = wasm_i16x8_extend_high_i8x16(v1l);
        const v128_t v1hl = wasm_i16x8_extend_low_i8x16 (v1h);
        const v128_t v1hh = wasm_i16x8_extend_high_i8x16(v1h);

        // dot product
        sumv = wasm_f32x4_add(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(
                        wasm_i32x4_add(
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))),
                    wasm_f32x4_splat(GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d))));
    }

    *s = wasm_f32x4_extract_lane(sumv, 0) + wasm_f32x4_extract_lane(sumv, 1) +
         wasm_f32x4_extract_lane(sumv, 2) + wasm_f32x4_extract_lane(sumv, 3);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; i++) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));

        __m256i bx = bytes_from_nibbles_32(x[i].qs);
        __m256i bxhi = bytes_from_bits_32(x[i].qh);
        bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char)0xF0));
        bx = _mm256_or_si256(bx, bxhi);

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float_avx2(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *s = hsum_float_8(acc);
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    __m128i mask = _mm_set1_epi8((char)0xF0);

    // Main loop
    for (int i = 0; i < nb; i++) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));

        __m256i bx = bytes_from_nibbles_32(x[i].qs);
        const __m256i bxhi = bytes_from_bits_32(x[i].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
        bxhil = _mm_andnot_si128(bxhil, mask);
        bxhih = _mm_andnot_si128(bxhih, mask);
        __m128i bxl = _mm256_castsi256_si128(bx);
        __m128i bxh = _mm256_extractf128_si256(bx, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx = MM256_SET_M128I(bxh, bxl);

        const __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float_avx(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
    }

    *s = hsum_float_8(acc);
#elif defined(__riscv_v_intrinsic)
    float sumf = 0.0;

    uint32_t qh;

    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    // These tempory registers are for masking and shift operations
    vuint32m2_t vt_1 = __riscv_vid_v_u32m2(vl);
    vuint32m2_t vt_2 = __riscv_vsll_vv_u32m2(__riscv_vmv_v_x_u32m2(1, vl), vt_1, vl);

    vuint32m2_t vt_3 = __riscv_vsll_vx_u32m2(vt_2, 16, vl);
    vuint32m2_t vt_4 = __riscv_vadd_vx_u32m2(vt_1, 12, vl);

    for (int i = 0; i < nb; i++) {
        memcpy(&qh, x[i].qh, sizeof(uint32_t));

        // ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
        vuint32m2_t xha_0 = __riscv_vand_vx_u32m2(vt_2, qh, vl);
        vuint32m2_t xhr_0 = __riscv_vsrl_vv_u32m2(xha_0, vt_1, vl);
        vuint32m2_t xhl_0 = __riscv_vsll_vx_u32m2(xhr_0, 4, vl);

        // ((qh & (1u << (j + 16))) >> (j + 12));
        vuint32m2_t xha_1 = __riscv_vand_vx_u32m2(vt_3, qh, vl);
        vuint32m2_t xhl_1 = __riscv_vsrl_vv_u32m2(xha_1, vt_4, vl);

        // narrowing
        vuint16m1_t xhc_0 = __riscv_vncvt_x_x_w_u16m1(xhl_0, vl);
        vuint8mf2_t xh_0 = __riscv_vncvt_x_x_w_u8mf2(xhc_0, vl);

        vuint16m1_t xhc_1 = __riscv_vncvt_x_x_w_u16m1(xhl_1, vl);
        vuint8mf2_t xh_1 = __riscv_vncvt_x_x_w_u8mf2(xhc_1, vl);

        // load
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[i].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[i].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[i].qs+16, vl);

        vuint8mf2_t x_at = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_lt = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vuint8mf2_t x_a = __riscv_vor_vv_u8mf2(x_at, xh_0, vl);
        vuint8mf2_t x_l = __riscv_vor_vv_u8mf2(x_lt, xh_1, vl);

        vint8mf2_t x_ai = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t x_li = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        vint8mf2_t v0 = __riscv_vsub_vx_i8mf2(x_ai, 16, vl);
        vint8mf2_t v1 = __riscv_vsub_vx_i8mf2(x_li, 16, vl);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d)) * sumi;
    }

    *s = sumf;
#else
    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk/2]);
        }

        sumf += (GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d)) * sumi;
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_q5_1_q8_1(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(qk == QK5_1);

    const block_q5_1 * restrict x = vx;
    const block_q8_1 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs0 = 0.0f;
    float summs1 = 0.0f;

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    assert(nb % 2 == 0); // TODO: handle odd nb

    for (int i = 0; i < nb; i += 2) {
        const block_q5_1 * restrict x0 = &x[i];
        const block_q5_1 * restrict x1 = &x[i + 1];
        const block_q8_1 * restrict y0 = &y[i];
        const block_q8_1 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        summs0 += GGML_FP16_TO_FP32(x0->m) * y0->s;
        summs1 += GGML_FP16_TO_FP32(x1->m) * y1->s;

        // extract the 5th bit via lookup table ((b) << 4)
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        tmp0[0] = table_b2b_0[(qh0 >>  0) & 0xFF];
        tmp0[1] = table_b2b_0[(qh0 >>  8) & 0xFF];
        tmp0[2] = table_b2b_0[(qh0 >> 16) & 0xFF];
        tmp0[3] = table_b2b_0[(qh0 >> 24)       ];

        tmp1[0] = table_b2b_0[(qh1 >>  0) & 0xFF];
        tmp1[1] = table_b2b_0[(qh1 >>  8) & 0xFF];
        tmp1[2] = table_b2b_0[(qh1 >> 16) & 0xFF];
        tmp1[3] = table_b2b_0[(qh1 >> 24)       ];

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // add high bit
        const int8x16_t v0_0lf = vorrq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vorrq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vorrq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vorrq_s8(v0_1h, qhh1);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))), GGML_FP16_TO_FP32(x0->d)*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))), GGML_FP16_TO_FP32(x1->d)*y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0lf), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lf), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hf), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hf), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1lf), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1lf), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hf), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hf), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), GGML_FP16_TO_FP32(x0->d)*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), GGML_FP16_TO_FP32(x1->d)*y1->d);
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs0 + summs1;
#elif defined(__wasm_simd128__)
    v128_t sumv = wasm_f32x4_splat(0.0f);

    float summs = 0.0f;

    uint32_t qh;
    uint64_t tmp[4];

    // TODO: check if unrolling this is better
    for (int i = 0; i < nb; ++i) {
        const block_q5_1 * restrict x0 = &x[i];
        const block_q8_1 * restrict y0 = &y[i];

        summs += GGML_FP16_TO_FP32(x0->m) * y0->s;

        const v128_t m4b = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        memcpy(&qh, x0->qh, sizeof(qh));

        tmp[0] = table_b2b_0[(qh >>  0) & 0xFF];
        tmp[1] = table_b2b_0[(qh >>  8) & 0xFF];
        tmp[2] = table_b2b_0[(qh >> 16) & 0xFF];
        tmp[3] = table_b2b_0[(qh >> 24)       ];

        const v128_t qhl = wasm_v128_load(tmp + 0);
        const v128_t qhh = wasm_v128_load(tmp + 2);

        const v128_t v0 = wasm_v128_load(x0->qs);

        // 4-bit -> 8-bit
        const v128_t v0l = wasm_v128_and (v0, m4b);
        const v128_t v0h = wasm_u8x16_shr(v0, 4);

        // add high bit
        const v128_t v0lf = wasm_v128_or(v0l, qhl);
        const v128_t v0hf = wasm_v128_or(v0h, qhh);

        // load y
        const v128_t v1l = wasm_v128_load(y0->qs);
        const v128_t v1h = wasm_v128_load(y0->qs + 16);

        // int8x16 -> int16x8
        const v128_t v0lfl = wasm_i16x8_extend_low_i8x16 (v0lf);
        const v128_t v0lfh = wasm_i16x8_extend_high_i8x16(v0lf);
        const v128_t v0hfl = wasm_i16x8_extend_low_i8x16 (v0hf);
        const v128_t v0hfh = wasm_i16x8_extend_high_i8x16(v0hf);

        const v128_t v1ll = wasm_i16x8_extend_low_i8x16 (v1l);
        const v128_t v1lh = wasm_i16x8_extend_high_i8x16(v1l);
        const v128_t v1hl = wasm_i16x8_extend_low_i8x16 (v1h);
        const v128_t v1hh = wasm_i16x8_extend_high_i8x16(v1h);

        // dot product
        sumv = wasm_f32x4_add(sumv,
                wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_add(
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))),
                    wasm_f32x4_splat(GGML_FP16_TO_FP32(x0->d) * y0->d)));
    }

    *s = wasm_f32x4_extract_lane(sumv, 0) + wasm_f32x4_extract_lane(sumv, 1) +
         wasm_f32x4_extract_lane(sumv, 2) + wasm_f32x4_extract_lane(sumv, 3) + summs;
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    float summs = 0.0f;

    // Main loop
    for (int i = 0; i < nb; i++) {
        const __m256 dx = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d));

        summs += GGML_FP16_TO_FP32(x[i].m) * y[i].s;

        __m256i bx = bytes_from_nibbles_32(x[i].qs);
        __m256i bxhi = bytes_from_bits_32(x[i].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        bx = _mm256_or_si256(bx, bxhi);

        const __m256 dy = _mm256_set1_ps(y[i].d);
        const __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_us8_pairs_float_avx2(bx, by);

        acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
    }

    *s = hsum_float_8(acc) + summs;
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    __m128i mask = _mm_set1_epi8(0x10);

    float summs = 0.0f;

    // Main loop
    for (int i = 0; i < nb; i++) {
        const __m256 dx = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d));

        summs += GGML_FP16_TO_FP32(x[i].m) * y[i].s;

        __m256i bx = bytes_from_nibbles_32(x[i].qs);
        const __m256i bxhi = bytes_from_bits_32(x[i].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
        bxhil = _mm_and_si128(bxhil, mask);
        bxhih = _mm_and_si128(bxhih, mask);
        __m128i bxl = _mm256_castsi256_si128(bx);
        __m128i bxh = _mm256_extractf128_si256(bx, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx = MM256_SET_M128I(bxh, bxl);

        const __m256 dy = _mm256_set1_ps(y[i].d);
        const __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_us8_pairs_float_avx(bx, by);

        acc = _mm256_add_ps(_mm256_mul_ps(q, _mm256_mul_ps(dx, dy)), acc);
    }

    *s = hsum_float_8(acc) + summs;
#elif defined(__riscv_v_intrinsic)
    float sumf = 0.0;

    uint32_t qh;

    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    // temporary registers for shift operations
    vuint32m2_t vt_1 = __riscv_vid_v_u32m2(vl);
    vuint32m2_t vt_2 = __riscv_vadd_vx_u32m2(vt_1, 12, vl);

    for (int i = 0; i < nb; i++) {
        memcpy(&qh, x[i].qh, sizeof(uint32_t));

        // load qh
        vuint32m2_t vqh = __riscv_vmv_v_x_u32m2(qh, vl);

        // ((qh >> (j +  0)) << 4) & 0x10;
        vuint32m2_t xhr_0 = __riscv_vsrl_vv_u32m2(vqh, vt_1, vl);
        vuint32m2_t xhl_0 = __riscv_vsll_vx_u32m2(xhr_0, 4, vl);
        vuint32m2_t xha_0 = __riscv_vand_vx_u32m2(xhl_0, 0x10, vl);

        // ((qh >> (j + 12))     ) & 0x10;
        vuint32m2_t xhr_1 = __riscv_vsrl_vv_u32m2(vqh, vt_2, vl);
        vuint32m2_t xha_1 = __riscv_vand_vx_u32m2(xhr_1, 0x10, vl);

        // narrowing
        vuint16m1_t xhc_0 = __riscv_vncvt_x_x_w_u16m1(xha_0, vl);
        vuint8mf2_t xh_0 = __riscv_vncvt_x_x_w_u8mf2(xhc_0, vl);

        vuint16m1_t xhc_1 = __riscv_vncvt_x_x_w_u16m1(xha_1, vl);
        vuint8mf2_t xh_1 = __riscv_vncvt_x_x_w_u8mf2(xhc_1, vl);

        // load
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[i].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[i].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[i].qs+16, vl);

        vuint8mf2_t x_at = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_lt = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vuint8mf2_t x_a = __riscv_vor_vv_u8mf2(x_at, xh_0, vl);
        vuint8mf2_t x_l = __riscv_vor_vv_u8mf2(x_lt, xh_1, vl);

        vint8mf2_t v0 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t v1 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_FP16_TO_FP32(x[i].d)*y[i].d)*sumi + GGML_FP16_TO_FP32(x[i].m)*y[i].s;
    }

    *s = sumf;
#else
    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[i].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[i].qs[j] >>  4) | xh_1;

            sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk/2]);
        }

        sumf += (GGML_FP16_TO_FP32(x[i].d)*y[i].d)*sumi + GGML_FP16_TO_FP32(x[i].m)*y[i].s;
    }

    *s = sumf;
#endif
}

#ifdef __x86_64__

#define ggml_vec_dot_q8_0_q8_0 ggml_vec_dot_q8_0_q8_0_default
#include "ggml_vec_dot_q8_0_q8_0.inc"
#undef ggml_vec_dot_q8_0_q8_0

#define ggml_vec_dot_q8_0_q8_0 __attribute__((__target__("avx"))) ggml_vec_dot_q8_0_q8_0_avx
#ifndef __AVX__
#define __UNDEF_AVX__
#define __AVX__
#endif
#include "ggml_vec_dot_q8_0_q8_0.inc"
#ifdef __UNDEF_AVX__
#undef __UNDEF_AVX__
#undef __AVX__
#endif
#undef ggml_vec_dot_q8_0_q8_0

#define ggml_vec_dot_q8_0_q8_0 __attribute__((__target__("avx2,fma"))) ggml_vec_dot_q8_0_q8_0_avx2
#ifndef __AVX2__
#define __UNDEF_AVX2__
#define __AVX2__
#endif
#include "ggml_vec_dot_q8_0_q8_0.inc"
#ifdef __UNDEF_AVX2__
#undef __UNDEF_AVX2__
#undef __AVX2__
#endif
#undef ggml_vec_dot_q8_0_q8_0

void ggml_vec_dot_q8_0_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    if (X86_HAVE(AVX2) && X86_HAVE(FMA)) {
        ggml_vec_dot_q8_0_q8_0_avx2(n, s, vx, vy);
    } else if (X86_HAVE(AVX)) {
        ggml_vec_dot_q8_0_q8_0_avx(n, s, vx, vy);
    } else {
        ggml_vec_dot_q8_0_q8_0_default(n, s, vx, vy);
    }
}

#else
#define static
#include "ggml_vec_dot_q8_0_q8_0.inc"
#undef static
#endif


#if QK_K == 256

#ifdef __x86_64__

#define ggml_vec_dot_q2_K_q8_K ggml_vec_dot_q2_K_q8_K_default
#include "ggml_vec_dot_q2_K_q8_K_256.inc"
#undef ggml_vec_dot_q2_K_q8_K

#define ggml_vec_dot_q2_K_q8_K __attribute__((__target__("avx"))) ggml_vec_dot_q2_K_q8_K_avx
#ifndef __AVX__
#define __UNDEF_AVX__
#define __AVX__
#endif
#include "ggml_vec_dot_q2_K_q8_K_256.inc"
#ifdef __UNDEF_AVX__
#undef __UNDEF_AVX__
#undef __AVX__
#endif
#undef ggml_vec_dot_q2_K_q8_K

#define ggml_vec_dot_q2_K_q8_K __attribute__((__target__("avx2,fma,f16c"))) ggml_vec_dot_q2_K_q8_K_avx2
#ifndef __AVX2__
#define __UNDEF_AVX2__
#define __AVX2__
#endif
#include "ggml_vec_dot_q2_K_q8_K_256.inc"
#ifdef __UNDEF_AVX2__
#undef __UNDEF_AVX2__
#undef __AVX2__
#endif
#undef ggml_vec_dot_q2_K_q8_K

void ggml_vec_dot_q2_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    if (X86_HAVE(AVX2) && X86_HAVE(FMA)) {
        ggml_vec_dot_q2_K_q8_K_avx2(n, s, vx, vy);
    } else if (X86_HAVE(AVX)) {
        ggml_vec_dot_q2_K_q8_K_avx(n, s, vx, vy);
    } else {
        ggml_vec_dot_q2_K_q8_K_default(n, s, vx, vy);
    }
}

#else
#define static
#include "ggml_vec_dot_q2_K_q8_K_256.inc"
#undef static
#endif /* __x86_64__ */

#else

void ggml_vec_dot_q2_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {

    const block_q2_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    const uint8x16_t m3 = vdupq_n_u8(0x3);
#if defined(__ARM_FEATURE_DOTPROD)
    const int32x4_t  vzero = vdupq_n_s32(0);
#endif

    int8x16x4_t q2bytes;

    uint32_t aux32[2];
    const uint8_t * scales = (const uint8_t *)aux32;

    float sum = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * (float)x[i].d;
        const float dmin = -y[i].d * (float)x[i].dmin;

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;
        const uint32_t * restrict sc = (const uint32_t *)x[i].scales;

        aux32[0] = sc[0] & 0x0f0f0f0f;
        aux32[1] = (sc[0] >> 4) & 0x0f0f0f0f;

        sum += dmin * (scales[4] * y[i].bsums[0] + scales[5] * y[i].bsums[1] + scales[6] * y[i].bsums[2] + scales[7] * y[i].bsums[3]);

        int isum1 = 0, isum2 = 0;

        const uint8x16_t q2bits = vld1q_u8(q2);

        const int8x16x4_t q8bytes = vld1q_s8_x4(q8);

        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits, m3));
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits, 2), m3));
        q2bytes.val[2] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits, 4), m3));
        q2bytes.val[3] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits, 6), m3));

#if defined(__ARM_FEATURE_DOTPROD)
        isum1 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * scales[0];
        isum2 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * scales[1];
        isum1 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[2], q8bytes.val[2])) * scales[2];
        isum2 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[3], q8bytes.val[3])) * scales[3];
#else
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q2bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q2bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        isum1 += vaddvq_s16(p1) * scales[0];
        isum2 += vaddvq_s16(p2) * scales[1];

        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q2bytes.val[2]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p4 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                       vmull_s8(vget_high_s8(q2bytes.val[3]), vget_high_s8(q8bytes.val[3])));
        isum1 += vaddvq_s16(p3) * scales[2];
        isum2 += vaddvq_s16(p4) * scales[3];
#endif
        sum += d * (isum1 + isum2);

    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);

    __m256 acc = _mm256_setzero_ps();

    uint32_t ud, um;
    const uint8_t * restrict db = (const uint8_t *)&ud;
    const uint8_t * restrict mb = (const uint8_t *)&um;

    float summs = 0;

    // TODO: optimize this

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint32_t * restrict sc = (const uint32_t *)x[i].scales;
        ud = (sc[0] >> 0) & 0x0f0f0f0f;
        um = (sc[0] >> 4) & 0x0f0f0f0f;

        int32_t smin = mb[0] * y[i].bsums[0] + mb[1] * y[i].bsums[1] + mb[2] * y[i].bsums[2] + mb[3] * y[i].bsums[3];
        summs += dmin * smin;

        const __m128i q2bits = _mm_loadu_si128((const __m128i*)q2);
        const __m256i q2_0 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q2bits, 2), q2bits), m3);
        const __m256i q2_1 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q2bits, 6), _mm_srli_epi16(q2bits, 4)), m3);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
        const __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);

        const __m256i p_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p0, 0));
        const __m256i p_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p0, 1));
        const __m256i p_2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p1, 0));
        const __m256i p_3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p1, 1));

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * db[0]), _mm256_cvtepi32_ps(p_0), acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * db[1]), _mm256_cvtepi32_ps(p_1), acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * db[2]), _mm256_cvtepi32_ps(p_2), acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * db[3]), _mm256_cvtepi32_ps(p_3), acc);
    }

    *s = hsum_float_8(acc) + summs;

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);

    __m256 acc = _mm256_setzero_ps();

    uint32_t ud, um;
    const uint8_t * restrict db = (const uint8_t *)&ud;
    const uint8_t * restrict mb = (const uint8_t *)&um;

    float summs = 0;

    // TODO: optimize this

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint32_t * restrict sc = (const uint32_t *)x[i].scales;
        ud = (sc[0] >> 0) & 0x0f0f0f0f;
        um = (sc[0] >> 4) & 0x0f0f0f0f;

        int32_t smin = mb[0] * y[i].bsums[0] + mb[1] * y[i].bsums[1] + mb[2] * y[i].bsums[2] + mb[3] * y[i].bsums[3];
        summs += dmin * smin;

        const __m128i q2bits = _mm_loadu_si128((const __m128i*)q2);
        const __m128i q2_0 = _mm_and_si128(q2bits, m3);
        const __m128i q2_1 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
        const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
        const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m128i p0 = _mm_maddubs_epi16(q2_0, _mm256_extractf128_si256(q8_0, 0));
        const __m128i p1 = _mm_maddubs_epi16(q2_1, _mm256_extractf128_si256(q8_0, 1));
        const __m128i p2 = _mm_maddubs_epi16(q2_2, _mm256_extractf128_si256(q8_1, 0));
        const __m128i p3 = _mm_maddubs_epi16(q2_3, _mm256_extractf128_si256(q8_1, 1));

        const __m256i p_0 = MM256_SET_M128I(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(p0, p0)), _mm_cvtepi16_epi32(p0));
        const __m256i p_1 = MM256_SET_M128I(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(p1, p1)), _mm_cvtepi16_epi32(p1));
        const __m256i p_2 = MM256_SET_M128I(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(p2, p2)), _mm_cvtepi16_epi32(p2));
        const __m256i p_3 = MM256_SET_M128I(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(p3, p3)), _mm_cvtepi16_epi32(p3));

        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d * db[0]), _mm256_cvtepi32_ps(p_0)), acc);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d * db[1]), _mm256_cvtepi32_ps(p_1)), acc);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d * db[2]), _mm256_cvtepi32_ps(p_2)), acc);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d * db[3]), _mm256_cvtepi32_ps(p_3)), acc);
    }

    *s = hsum_float_8(acc) + summs;

#elif defined __riscv_v_intrinsic

    uint32_t aux32[2];
    const uint8_t * scales = (const uint8_t *)aux32;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * (float)x[i].d;
        const float dmin = -y[i].d * (float)x[i].dmin;

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;
        const uint32_t * restrict sc = (const uint32_t *)x[i].scales;

        aux32[0] = sc[0] & 0x0f0f0f0f;
        aux32[1] = (sc[0] >> 4) & 0x0f0f0f0f;

        sumf += dmin * (scales[4] * y[i].bsums[0] + scales[5] * y[i].bsums[1] + scales[6] * y[i].bsums[2] + scales[7] * y[i].bsums[3]);

        int isum1 = 0;
        int isum2 = 0;

        size_t vl = 16;

        vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

        // load Q2
        vuint8mf2_t q2_x = __riscv_vle8_v_u8mf2(q2, vl);

        vint8mf2_t q2_0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(q2_x, 0x03, vl));
        vint8mf2_t q2_1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(q2_x, 0x2, vl), 0x03 , vl));
        vint8mf2_t q2_2 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(q2_x, 0x4, vl), 0x03 , vl));
        vint8mf2_t q2_3 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(q2_x, 0x6, vl), 0x03 , vl));

        // load Q8, and take product with Q2
        vint16m1_t p0 = __riscv_vwmul_vv_i16m1(q2_0, __riscv_vle8_v_i8mf2(q8, vl), vl);
        vint16m1_t p1 = __riscv_vwmul_vv_i16m1(q2_1, __riscv_vle8_v_i8mf2(q8+16, vl), vl);
        vint16m1_t p2 = __riscv_vwmul_vv_i16m1(q2_2, __riscv_vle8_v_i8mf2(q8+32, vl), vl);
        vint16m1_t p3 = __riscv_vwmul_vv_i16m1(q2_3, __riscv_vle8_v_i8mf2(q8+48, vl), vl);

        vint16m1_t vs_0 = __riscv_vredsum_vs_i16m1_i16m1(p0, vzero, vl);
        vint16m1_t vs_1 = __riscv_vredsum_vs_i16m1_i16m1(p1, vzero, vl);
        vint16m1_t vs_2 = __riscv_vredsum_vs_i16m1_i16m1(p2, vzero, vl);
        vint16m1_t vs_3 = __riscv_vredsum_vs_i16m1_i16m1(p3, vzero, vl);

        isum1 += __riscv_vmv_x_s_i16m1_i16(vs_0) * scales[0];
        isum2 += __riscv_vmv_x_s_i16m1_i16(vs_1) * scales[1];
        isum1 += __riscv_vmv_x_s_i16m1_i16(vs_2) * scales[2];
        isum2 += __riscv_vmv_x_s_i16m1_i16(vs_3) * scales[3];

        sumf += d * (isum1 + isum2);

    }

    *s = sumf;

#else

    float sumf = 0;

    int isum[4];

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        isum[0] = isum[1] = isum[2] = isum[3] = 0;
        for (int l =  0; l < 16; ++l) {
            isum[0] += q8[l+ 0] * ((q2[l] >> 0) & 3);
            isum[1] += q8[l+16] * ((q2[l] >> 2) & 3);
            isum[2] += q8[l+32] * ((q2[l] >> 4) & 3);
            isum[3] += q8[l+48] * ((q2[l] >> 6) & 3);
        }
        for (int l = 0; l < 4; ++l) {
            isum[l] *= (sc[l] & 0xF);
        }
        sumf += dall * (isum[0] + isum[1] + isum[2] + isum[3]) - dmin * summs;
    }
    *s = sumf;
#endif
}
#endif

#if QK_K == 256

#ifdef __x86_64__

#define ggml_vec_dot_q3_K_q8_K ggml_vec_dot_q3_K_q8_K_default
#include "ggml_vec_dot_q3_K_q8_K_256.inc"
#undef ggml_vec_dot_q3_K_q8_K

#define ggml_vec_dot_q3_K_q8_K __attribute__((__target__("avx"))) ggml_vec_dot_q3_K_q8_K_avx
#ifndef __AVX__
#define __UNDEF_AVX__
#define __AVX__
#endif
#include "ggml_vec_dot_q3_K_q8_K_256.inc"
#ifdef __UNDEF_AVX__
#undef __UNDEF_AVX__
#undef __AVX__
#endif
#undef ggml_vec_dot_q3_K_q8_K

#define ggml_vec_dot_q3_K_q8_K __attribute__((__target__("avx2,fma,f16c"))) ggml_vec_dot_q3_K_q8_K_avx2
#ifndef __AVX2__
#define __UNDEF_AVX2__
#define __AVX2__
#endif
#include "ggml_vec_dot_q3_K_q8_K_256.inc"
#ifdef __UNDEF_AVX2__
#undef __UNDEF_AVX2__
#undef __AVX2__
#endif
#undef ggml_vec_dot_q3_K_q8_K

void ggml_vec_dot_q3_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    if (X86_HAVE(AVX2) && X86_HAVE(FMA) && X86_HAVE(F16C)) {
        ggml_vec_dot_q3_K_q8_K_avx2(n, s, vx, vy);
    } else if (X86_HAVE(AVX)) {
        ggml_vec_dot_q3_K_q8_K_avx(n, s, vx, vy);
    } else {
        ggml_vec_dot_q3_K_q8_K_default(n, s, vx, vy);
    }
}

#else
#define static
#include "ggml_vec_dot_q3_K_q8_K_256.inc"
#undef static
#endif

#else

void ggml_vec_dot_q3_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q3_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

#ifdef __ARM_FEATURE_DOTPROD
    const int32x4_t  vzero = vdupq_n_s32(0);
#endif

    const uint8x16_t m3b = vdupq_n_u8(0x3);
    const uint8x16_t mh  = vdupq_n_u8(4);

    int8x16x4_t q3bytes;

    uint16_t aux16[2];
    int8_t * scales = (int8_t *)aux16;

    float sum = 0;

    for (int i = 0; i < nb; ++i) {

        uint8x16x4_t q3h;

        const uint8x8_t  hbits    = vld1_u8(x[i].hmask);
        const uint8x16_t q3bits   = vld1q_u8(x[i].qs);
        const int8x16x4_t q8bytes = vld1q_s8_x4(y[i].qs);

        const uint16_t a = *(const uint16_t *)x[i].scales;
        aux16[0] = a & 0x0f0f;
        aux16[1] = (a >> 4) & 0x0f0f;

        for (int j = 0; j < 4; ++j) scales[j] -= 8;

        int32_t isum = -4*(scales[0] * y[i].bsums[0] + scales[2] * y[i].bsums[1] + scales[1] * y[i].bsums[2] + scales[3] * y[i].bsums[3]);

        const float d = y[i].d * (float)x[i].d;

        const uint8x16_t htmp = vcombine_u8(hbits, vshr_n_u8(hbits, 1));
        q3h.val[0] = vandq_u8(mh, vshlq_n_u8(htmp, 2));
        q3h.val[1] = vandq_u8(mh, htmp);
        q3h.val[2] = vandq_u8(mh, vshrq_n_u8(htmp, 2));
        q3h.val[3] = vandq_u8(mh, vshrq_n_u8(htmp, 4));

        q3bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q3bits, m3b),                q3h.val[0]));
        q3bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vshrq_n_u8(q3bits, 2), m3b), q3h.val[1]));
        q3bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vshrq_n_u8(q3bits, 4), m3b), q3h.val[2]));
        q3bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q3bits, 6),                q3h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)
        isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[0], q8bytes.val[0])) * scales[0];
        isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[1], q8bytes.val[1])) * scales[2];
        isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[2], q8bytes.val[2])) * scales[1];
        isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[3], q8bytes.val[3])) * scales[3];
#else
        const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q3bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q3bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q3bytes.val[2]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                       vmull_s8(vget_high_s8(q3bytes.val[3]), vget_high_s8(q8bytes.val[3])));
        isum += vaddvq_s16(p0) * scales[0] + vaddvq_s16(p1) * scales[2] + vaddvq_s16(p2) * scales[1] + vaddvq_s16(p3) * scales[3];
#endif

        sum += d * isum;

    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m1 = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    uint64_t aux64;

    uint16_t aux16[2];
    const int8_t * aux8 = (const int8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint16_t a = *(const uint16_t *)x[i].scales;
        aux16[0] = a & 0x0f0f;
        aux16[1] = (a >> 4) & 0x0f0f;

        const __m256i scale_0 = MM256_SET_M128I(_mm_set1_epi16(aux8[2] - 8), _mm_set1_epi16(aux8[0] - 8));
        const __m256i scale_1 = MM256_SET_M128I(_mm_set1_epi16(aux8[3] - 8), _mm_set1_epi16(aux8[1] - 8));

        memcpy(&aux64, x[i].hmask, 8);

        const __m128i haux = _mm_set_epi64x(aux64 >> 1, aux64 >> 0);
        __m256i q3h_0 = MM256_SET_M128I(_mm_srli_epi16(haux, 2), haux);
        __m256i q3h_1 = _mm256_srli_epi16(q3h_0, 4);
        q3h_0 = _mm256_slli_epi16(_mm256_andnot_si256(q3h_0, m1), 2);
        q3h_1 = _mm256_slli_epi16(_mm256_andnot_si256(q3h_1, m1), 2);

        // load low 2 bits
        const __m128i q3bits = _mm_loadu_si128((const __m128i*)q3);

        // prepare low and high bits
        const __m256i q3aux  = MM256_SET_M128I(_mm_srli_epi16(q3bits, 2), q3bits);
        const __m256i q3l_0 = _mm256_and_si256(q3aux, m3);
        const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3aux, 4), m3);

        // load Q8 quants
        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
        // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
        // and 2 if the high bit was set)
        const __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
        const __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);

        __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
        __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);

        p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm256_sub_epi16(p16_1, q8s_1);

        // multiply with scales
        p16_0 = _mm256_madd_epi16(scale_0, p16_0);
        p16_1 = _mm256_madd_epi16(scale_1, p16_1);

        p16_0 = _mm256_add_epi32(p16_0, p16_1);

        // multiply with block scale and accumulate
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(p16_0), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i m1 = _mm_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    uint64_t aux64;

    uint16_t aux16[2];
    const int8_t * aux8 = (const int8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint16_t a = *(const uint16_t *)x[i].scales;
        aux16[0] = a & 0x0f0f;
        aux16[1] = (a >> 4) & 0x0f0f;

        const __m128i scale_0 = _mm_set1_epi16(aux8[0] - 8);
        const __m128i scale_1 = _mm_set1_epi16(aux8[2] - 8);
        const __m128i scale_2 = _mm_set1_epi16(aux8[1] - 8);
        const __m128i scale_3 = _mm_set1_epi16(aux8[3] - 8);

        memcpy(&aux64, x[i].hmask, 8);

        __m128i q3h_0 = _mm_set_epi64x(aux64 >> 1, aux64 >> 0);
        __m128i q3h_1 = _mm_srli_epi16(q3h_0, 2);
        __m128i q3h_2 = _mm_srli_epi16(q3h_0, 4);
        __m128i q3h_3 = _mm_srli_epi16(q3h_0, 6);
        q3h_0 = _mm_slli_epi16(_mm_andnot_si128(q3h_0, m1), 2);
        q3h_1 = _mm_slli_epi16(_mm_andnot_si128(q3h_1, m1), 2);
        q3h_2 = _mm_slli_epi16(_mm_andnot_si128(q3h_2, m1), 2);
        q3h_3 = _mm_slli_epi16(_mm_andnot_si128(q3h_3, m1), 2);

        // load low 2 bits
        const __m128i q3bits = _mm_loadu_si128((const __m128i*)q3);

        // prepare low and high bits
        const __m128i q3l_0 = _mm_and_si128(q3bits, m3);
        const __m128i q3l_1 = _mm_and_si128(_mm_srli_epi16(q3bits, 2), m3);
        const __m128i q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits, 4), m3);
        const __m128i q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits, 6), m3);

        // load Q8 quants
        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm_maddubs_epi16,
        // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
        // and 2 if the high bit was set)
        const __m128i q8s_0 = _mm_maddubs_epi16(q3h_0, _mm256_extractf128_si256(q8_0, 0));
        const __m128i q8s_1 = _mm_maddubs_epi16(q3h_1, _mm256_extractf128_si256(q8_0, 1));
        const __m128i q8s_2 = _mm_maddubs_epi16(q3h_2, _mm256_extractf128_si256(q8_1, 0));
        const __m128i q8s_3 = _mm_maddubs_epi16(q3h_3, _mm256_extractf128_si256(q8_1, 1));

        __m128i p16_0 = _mm_maddubs_epi16(q3l_0, _mm256_extractf128_si256(q8_0, 0));
        __m128i p16_1 = _mm_maddubs_epi16(q3l_1, _mm256_extractf128_si256(q8_0, 1));
        __m128i p16_2 = _mm_maddubs_epi16(q3l_2, _mm256_extractf128_si256(q8_1, 0));
        __m128i p16_3 = _mm_maddubs_epi16(q3l_3, _mm256_extractf128_si256(q8_1, 1));

        p16_0 = _mm_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm_sub_epi16(p16_1, q8s_1);
        p16_2 = _mm_sub_epi16(p16_2, q8s_2);
        p16_3 = _mm_sub_epi16(p16_3, q8s_3);

        // multiply with scales
        p16_0 = _mm_madd_epi16(scale_0, p16_0);
        p16_1 = _mm_madd_epi16(scale_1, p16_1);
        p16_2 = _mm_madd_epi16(scale_2, p16_2);
        p16_3 = _mm_madd_epi16(scale_3, p16_3);

        p16_0 = _mm_add_epi32(p16_0, p16_2);
        p16_1 = _mm_add_epi32(p16_1, p16_3);
        __m256i p16 = MM256_SET_M128I(p16_1, p16_0);

        // multiply with block scale and accumulate
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(p16)), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __riscv_v_intrinsic

    uint16_t aux16[2];
    int8_t * scales = (int8_t *)aux16;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint16_t a = *(const uint16_t *)x[i].scales;
        aux16[0] = a & 0x0f0f;
        aux16[1] = (a >> 4) & 0x0f0f;

        for (int j = 0; j < 4; ++j) scales[j] -= 8;

        int32_t isum = -4*(scales[0] * y[i].bsums[0] + scales[2] * y[i].bsums[1] + scales[1] * y[i].bsums[2] + scales[3] * y[i].bsums[3]);

        const float d = y[i].d * (float)x[i].d;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

        // load qh
        vuint8mf4_t qh_x1   = __riscv_vle8_v_u8mf4(x[i].hmask, 8);
        vuint8mf2_t qh_x2   = __riscv_vlmul_ext_v_u8mf4_u8mf2(__riscv_vsrl_vx_u8mf4(qh_x1, 1, 8));

        size_t vl = 16;

        // extend and combine both qh_x1 and qh_x2
        vuint8mf2_t qh_x = __riscv_vslideup_vx_u8mf2(__riscv_vlmul_ext_v_u8mf4_u8mf2(qh_x1), qh_x2, vl/2, vl);

        vuint8mf2_t qh_0 = __riscv_vand_vx_u8mf2(__riscv_vsll_vx_u8mf2(qh_x, 0x2, vl), 0x4, vl);
        vuint8mf2_t qh_1 = __riscv_vand_vx_u8mf2(qh_x, 0x4, vl);
        vuint8mf2_t qh_2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(qh_x, 0x2, vl), 0x4, vl);
        vuint8mf2_t qh_3 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(qh_x, 0x4, vl), 0x4, vl);

        // load Q3
        vuint8mf2_t q3_x  = __riscv_vle8_v_u8mf2(q3, vl);

        vuint8mf2_t q3h_0 = __riscv_vor_vv_u8mf2(__riscv_vand_vx_u8mf2(q3_x, 0x3, vl), qh_0, vl);
        vuint8mf2_t q3h_1 = __riscv_vor_vv_u8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(q3_x, 2, vl), 0x3, vl), qh_1, vl);
        vuint8mf2_t q3h_2 = __riscv_vor_vv_u8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(q3_x, 4, vl), 0x3, vl), qh_2, vl);
        vuint8mf2_t q3h_3 = __riscv_vor_vv_u8mf2(__riscv_vsrl_vx_u8mf2(q3_x, 0x6, vl), qh_3, vl);

        vint8mf2_t q3_0 = __riscv_vreinterpret_v_u8mf2_i8mf2(q3h_0);
        vint8mf2_t q3_1 = __riscv_vreinterpret_v_u8mf2_i8mf2(q3h_1);
        vint8mf2_t q3_2 = __riscv_vreinterpret_v_u8mf2_i8mf2(q3h_2);
        vint8mf2_t q3_3 = __riscv_vreinterpret_v_u8mf2_i8mf2(q3h_3);

        // load Q8 and take product with Q3
        vint16m1_t p0 = __riscv_vwmul_vv_i16m1(q3_0, __riscv_vle8_v_i8mf2(q8, vl), vl);
        vint16m1_t p1 = __riscv_vwmul_vv_i16m1(q3_1, __riscv_vle8_v_i8mf2(q8+16, vl), vl);
        vint16m1_t p2 = __riscv_vwmul_vv_i16m1(q3_2, __riscv_vle8_v_i8mf2(q8+32, vl), vl);
        vint16m1_t p3 = __riscv_vwmul_vv_i16m1(q3_3, __riscv_vle8_v_i8mf2(q8+48, vl), vl);

        vint32m1_t vs_0 = __riscv_vwredsum_vs_i16m1_i32m1(p0, vzero, vl);
        vint32m1_t vs_1 = __riscv_vwredsum_vs_i16m1_i32m1(p1, vzero, vl);
        vint32m1_t vs_2 = __riscv_vwredsum_vs_i16m1_i32m1(p2, vzero, vl);
        vint32m1_t vs_3 = __riscv_vwredsum_vs_i16m1_i32m1(p3, vzero, vl);

        isum += __riscv_vmv_x_s_i32m1_i32(vs_0) * scales[0];
        isum += __riscv_vmv_x_s_i32m1_i32(vs_1) * scales[2];
        isum += __riscv_vmv_x_s_i32m1_i32(vs_2) * scales[1];
        isum += __riscv_vmv_x_s_i32m1_i32(vs_3) * scales[3];

        sumf += d * isum;

    }

    *s = sumf;

#else

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    int32_t scales[4];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;
        int8_t * restrict a = aux8;
        for (int l = 0; l < 8; ++l) {
            a[l+ 0] = (int8_t)((q3[l+0] >> 0) & 3) - (hm[l] & 0x01 ? 0 : 4);
            a[l+ 8] = (int8_t)((q3[l+8] >> 0) & 3) - (hm[l] & 0x02 ? 0 : 4);
            a[l+16] = (int8_t)((q3[l+0] >> 2) & 3) - (hm[l] & 0x04 ? 0 : 4);
            a[l+24] = (int8_t)((q3[l+8] >> 2) & 3) - (hm[l] & 0x08 ? 0 : 4);
            a[l+32] = (int8_t)((q3[l+0] >> 4) & 3) - (hm[l] & 0x10 ? 0 : 4);
            a[l+40] = (int8_t)((q3[l+8] >> 4) & 3) - (hm[l] & 0x20 ? 0 : 4);
            a[l+48] = (int8_t)((q3[l+0] >> 6) & 3) - (hm[l] & 0x40 ? 0 : 4);
            a[l+56] = (int8_t)((q3[l+8] >> 6) & 3) - (hm[l] & 0x80 ? 0 : 4);
        }

        scales[0] = (x[i].scales[0] & 0xF) - 8;
        scales[1] = (x[i].scales[0] >>  4) - 8;
        scales[2] = (x[i].scales[1] & 0xF) - 8;
        scales[3] = (x[i].scales[1] >>  4) - 8;

        memset(aux32, 0, 8*sizeof(int32_t));
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] += q8[l] * a[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux32[l] += scales[j] * aux16[l];
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif

}
#endif

#if QK_K == 256

#ifdef __x86_64__

#define ggml_vec_dot_q4_K_q8_K ggml_vec_dot_q4_K_q8_K_default
#include "ggml_vec_dot_q4_K_q8_K_256.inc"
#undef ggml_vec_dot_q4_K_q8_K

#define ggml_vec_dot_q4_K_q8_K __attribute__((__target__("avx"))) ggml_vec_dot_q4_K_q8_K_avx
#ifndef __AVX__
#define __UNDEF_AVX__
#define __AVX__
#endif
#include "ggml_vec_dot_q4_K_q8_K_256.inc"
#ifdef __UNDEF_AVX__
#undef __UNDEF_AVX__
#undef __AVX__
#endif
#undef ggml_vec_dot_q4_K_q8_K

#define ggml_vec_dot_q4_K_q8_K __attribute__((__target__("avx2,fma,f16c"))) ggml_vec_dot_q4_K_q8_K_avx2
#ifndef __AVX2__
#define __UNDEF_AVX2__
#define __AVX2__
#endif
#include "ggml_vec_dot_q4_K_q8_K_256.inc"
#ifdef __UNDEF_AVX2__
#undef __UNDEF_AVX2__
#undef __AVX2__
#endif
#undef ggml_vec_dot_q4_K_q8_K

void ggml_vec_dot_q4_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    if (X86_HAVE(AVX2) && X86_HAVE(FMA) && X86_HAVE(F16C)) {
        ggml_vec_dot_q4_K_q8_K_avx2(n, s, vx, vy);
    } else if (X86_HAVE(AVX)) {
        ggml_vec_dot_q4_K_q8_K_avx(n, s, vx, vy);
    } else {
        ggml_vec_dot_q4_K_q8_K_default(n, s, vx, vy);
    }
}

#else
#define static
#include "ggml_vec_dot_q4_K_q8_K_256.inc"
#undef static
#endif

#else
void ggml_vec_dot_q4_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    const uint8x16_t m4b = vdupq_n_u8(0xf);

#ifdef __ARM_FEATURE_DOTPROD
    const int32x4_t mzero = vdupq_n_s32(0);
#endif

    float sumf = 0;

    int8x16x2_t q4bytes;
    int8x16x4_t q8bytes;

    float sum_mins = 0.f;

    uint16_t aux16[2];
    const uint8_t * restrict scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint16_t * restrict a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        const int32_t summi = scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]);
        sum_mins += y[i].d * (float)x[i].d[1] * summi;

        const float d = y[i].d * (float)x[i].d[0];

        const uint8x16x2_t q4bits = vld1q_u8_x2(q4);

#ifdef __ARM_FEATURE_DOTPROD
        q8bytes = vld1q_s8_x4(q8);
        q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
        q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

        const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
        const int32_t sumi1 = vaddvq_s32(p1) * scales[0];

        q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
        q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

        const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[2]), q4bytes.val[1], q8bytes.val[3]);
        const int32_t sumi2 = vaddvq_s32(p2) * scales[1];

#else
        q8bytes = vld1q_s8_x4(q8);
        q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
        q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));
        const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        int32_t sumi1 = vaddvq_s16(vaddq_s16(p0, p1)) * scales[0];

        q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
        q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[3])),
                                       vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[3])));
        int32_t sumi2 = vaddvq_s16(vaddq_s16(p2, p3)) * scales[1];

#endif
        sumf += d * (sumi1 + sumi2);

    }

    *s = sumf - sum_mins;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    uint16_t aux16[2];
    const uint8_t * scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d[0]) * y[i].d;
        const float m = GGML_FP16_TO_FP32(x[i].d[1]) * y[i].d;
        const __m256 vd = _mm256_set1_ps(d);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        summs += m * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
        const __m256i q4l = _mm256_and_si256(q4bits, m4);
        const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

        const __m256i q8l = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8h = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
        const __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);

        const __m256i p32l = _mm256_madd_epi16(_mm256_set1_epi16(scales[0]), p16l);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32l), acc);

        const __m256i p32h = _mm256_madd_epi16(_mm256_set1_epi16(scales[1]), p16h);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32h), acc);

    }

    *s = hsum_float_8(acc) - summs;

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    uint16_t aux16[2];
    const uint8_t * scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d[0]) * y[i].d;
        const float m = GGML_FP16_TO_FP32(x[i].d[1]) * y[i].d;
        const __m256 vd = _mm256_set1_ps(d);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        summs += m * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
        const __m128i q4bits_0 = _mm256_extractf128_si256(q4bits, 0);
        const __m128i q4bits_1 = _mm256_extractf128_si256(q4bits, 1);
        const __m128i q4_0 = _mm_and_si128(q4bits_0, m4);
        const __m128i q4_1 = _mm_and_si128(q4bits_1, m4);
        const __m128i q4_2 = _mm_and_si128(_mm_srli_epi16(q4bits_0, 4), m4);
        const __m128i q4_3 = _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m128i p16_0 = _mm_maddubs_epi16(q4_0, _mm256_extractf128_si256(q8_0, 0));
        const __m128i p16_1 = _mm_maddubs_epi16(q4_1, _mm256_extractf128_si256(q8_0, 1));
        const __m128i p16_2 = _mm_maddubs_epi16(q4_2, _mm256_extractf128_si256(q8_1, 0));
        const __m128i p16_3 = _mm_maddubs_epi16(q4_3, _mm256_extractf128_si256(q8_1, 1));

        const __m128i p32_0 = _mm_madd_epi16(_mm_set1_epi16(scales[0]), p16_0);
        const __m128i p32_1 = _mm_madd_epi16(_mm_set1_epi16(scales[0]), p16_1);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(MM256_SET_M128I(p32_1, p32_0))), acc);

        const __m128i p32_2 = _mm_madd_epi16(_mm_set1_epi16(scales[1]), p16_2);
        const __m128i p32_3 = _mm_madd_epi16(_mm_set1_epi16(scales[1]), p16_3);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(MM256_SET_M128I(p32_3, p32_2))), acc);

    }

    *s = hsum_float_8(acc) - summs;

#elif defined __riscv_v_intrinsic

    uint16_t s16[2];
    const uint8_t * restrict scales = (const uint8_t *)s16;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;

        const uint16_t * restrict b = (const uint16_t *)x[i].scales;
        s16[0] = b[0] & 0x0f0f;
        s16[1] = (b[0] >> 4) & 0x0f0f;

        sumf -= y[i].d * GGML_FP16_TO_FP32(x[i].d[1]) * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));
        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d[0]);

        size_t vl = 32;

        vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

        // load Q4
        vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);

        // load Q8 and multiply it with lower Q4 nibble
        vint8m1_t  q4_a = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
        vint16m2_t va_0 = __riscv_vwmul_vv_i16m2(q4_a, __riscv_vle8_v_i8m1(q8, vl), vl);
        vint16m1_t aux1 = __riscv_vredsum_vs_i16m2_i16m1(va_0, vzero, vl);

        sumf += d*scales[0]*__riscv_vmv_x_s_i16m1_i16(aux1);

        // load Q8 and multiply it with upper Q4 nibble
        vint8m1_t  q4_s = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
        vint16m2_t va_1 = __riscv_vwmul_vv_i16m2(q4_s, __riscv_vle8_v_i8m1(q8+32, vl), vl);
        vint16m1_t aux2 = __riscv_vredsum_vs_i16m2_i16m1(va_1, vzero, vl);

        sumf += d*scales[1]*__riscv_vmv_x_s_i16m1_i16(aux2);

    }

    *s = sumf;

#else

    uint8_t aux8[QK_K];
    int16_t aux16[16];
    float   sums [8];
    memset(sums, 0, 8*sizeof(float));

    uint16_t s16[2];
    const uint8_t * restrict scales = (const uint8_t *)s16;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        uint8_t * restrict a = aux8;
        for (int l = 0; l < 32; ++l) a[l+ 0] = q4[l] & 0xF;
        for (int l = 0; l < 32; ++l) a[l+32] = q4[l]  >> 4;

        const uint16_t * restrict b = (const uint16_t *)x[i].scales;
        s16[0] = b[0] & 0x0f0f;
        s16[1] = (b[0] >> 4) & 0x0f0f;

        sumf -= y[i].d * GGML_FP16_TO_FP32(x[i].d[1]) * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d[0]);

        for (int j = 0; j < QK_K/32; ++j) {
            for (int l = 0; l < 16; ++l) aux16[l] = q8[l] * a[l];
            q8 += 16; a += 16;
            for (int l = 0; l < 16; ++l) aux16[l] += q8[l] * a[l];
            q8 += 16; a += 16;
            const float dl = d * scales[j];
            for (int l = 0; l < 8; ++l) sums[l] += dl * (aux16[l] + aux16[l+8]);
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
#endif

#if QK_K == 256

#ifdef __x86_64__

#define ggml_vec_dot_q5_K_q8_K ggml_vec_dot_q5_K_q8_K_default
#include "ggml_vec_dot_q5_K_q8_K_256.inc"
#undef ggml_vec_dot_q5_K_q8_K

#define ggml_vec_dot_q5_K_q8_K __attribute__((__target__("avx"))) ggml_vec_dot_q5_K_q8_K_avx
#ifndef __AVX__
#define __UNDEF_AVX__
#define __AVX__
#endif
#include "ggml_vec_dot_q5_K_q8_K_256.inc"
#ifdef __UNDEF_AVX__
#undef __UNDEF_AVX__
#undef __AVX__
#endif
#undef ggml_vec_dot_q5_K_q8_K

#define ggml_vec_dot_q5_K_q8_K __attribute__((__target__("avx2,fma,f16c"))) ggml_vec_dot_q5_K_q8_K_avx2
#ifndef __AVX2__
#define __UNDEF_AVX2__
#define __AVX2__
#endif
#include "ggml_vec_dot_q5_K_q8_K_256.inc"
#ifdef __UNDEF_AVX2__
#undef __UNDEF_AVX2__
#undef __AVX2__
#endif
#undef ggml_vec_dot_q5_K_q8_K

void ggml_vec_dot_q5_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    if (X86_HAVE(AVX2) && X86_HAVE(FMA) && X86_HAVE(F16C)) {
        ggml_vec_dot_q5_K_q8_K_avx2(n, s, vx, vy);
    } else if (X86_HAVE(AVX)) {
        ggml_vec_dot_q5_K_q8_K_avx(n, s, vx, vy);
    } else {
        ggml_vec_dot_q5_K_q8_K_default(n, s, vx, vy);
    }
}

#else
#define static
#include "ggml_vec_dot_q5_K_q8_K_256.inc"
#undef static
#endif

#else

void ggml_vec_dot_q5_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q5_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const uint8x16_t mh = vdupq_n_u8(16);
#if defined(__ARM_FEATURE_DOTPROD)
    const int32x4_t mzero = vdupq_n_s32(0);
#endif

    int8x16x4_t q5bytes;
    uint8x16x4_t q5h;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * (float)x[i].d;
        const int8_t * sc = x[i].scales;

        const uint8_t * restrict q5 = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const uint8x8_t qhbits = vld1_u8(qh);

        const uint8x16x2_t q5bits = vld1q_u8_x2(q5);
        const int8x16x4_t q8bytes = vld1q_s8_x4(q8);

        const uint8x16_t htmp = vcombine_u8(qhbits, vshr_n_u8(qhbits, 1));
        q5h.val[0] = vbicq_u8(mh, vshlq_n_u8(htmp, 4));
        q5h.val[1] = vbicq_u8(mh, vshlq_n_u8(htmp, 2));
        q5h.val[2] = vbicq_u8(mh, htmp);
        q5h.val[3] = vbicq_u8(mh, vshrq_n_u8(htmp, 2));

        q5bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q5bits.val[0], m4b)), vreinterpretq_s8_u8(q5h.val[0]));
        q5bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q5bits.val[1], m4b)), vreinterpretq_s8_u8(q5h.val[1]));
        q5bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q5bits.val[0], 4)), vreinterpretq_s8_u8(q5h.val[2]));
        q5bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q5bits.val[1], 4)), vreinterpretq_s8_u8(q5h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

        int32_t sumi1 = sc[0] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[0], q8bytes.val[0]));
        int32_t sumi2 = sc[1] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[1], q8bytes.val[1]));
        int32_t sumi3 = sc[2] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[2], q8bytes.val[2]));
        int32_t sumi4 = sc[3] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[3], q8bytes.val[3]));

        sumf += d * (sumi1 + sumi2 + sumi3 + sumi4);

#else

        const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q5bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q5bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        int32_t sumi = sc[0] * vaddvq_s16(p0) + sc[1] * vaddvq_s16(p1);

        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q5bytes.val[2]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                       vmull_s8(vget_high_s8(q5bytes.val[3]), vget_high_s8(q8bytes.val[3])));
        sumi += sc[2] * vaddvq_s16(p2) + sc[3] * vaddvq_s16(p3);

        sumf += d*sumi;
#endif

    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5);

        const __m256i scale_l = MM256_SET_M128I(_mm_set1_epi16(x[i].scales[1]), _mm_set1_epi16(x[i].scales[0]));
        const __m256i scale_h = MM256_SET_M128I(_mm_set1_epi16(x[i].scales[3]), _mm_set1_epi16(x[i].scales[2]));

        int64_t aux64;
        memcpy(&aux64, x[i].qh, 8);
        const __m128i haux128 = _mm_set_epi64x(aux64 >> 1, aux64);
        const __m256i haux256 = MM256_SET_M128I(_mm_srli_epi16(haux128, 2), haux128);

        const __m256i q5h_0 = _mm256_slli_epi16(_mm256_andnot_si256(haux256, mone), 4);
        const __m256i q5h_1 = _mm256_slli_epi16(_mm256_andnot_si256(_mm256_srli_epi16(haux256, 4), mone), 4);

        const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
        const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m256i p16_0 = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q5l_0, q8_0));
        const __m256i p16_1 = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q5l_1, q8_1));
        const __m256i s16_0 = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q5h_0, q8_0));
        const __m256i s16_1 = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q5h_1, q8_1));

        const __m256i dot = _mm256_sub_epi32(_mm256_add_epi32(p16_0, p16_1), _mm256_add_epi32(s16_0, s16_1));

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(dot), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i mone  = _mm_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5);

        const __m128i scale_0 = _mm_set1_epi16(x[i].scales[0]);
        const __m128i scale_1 = _mm_set1_epi16(x[i].scales[1]);
        const __m128i scale_2 = _mm_set1_epi16(x[i].scales[2]);
        const __m128i scale_3 = _mm_set1_epi16(x[i].scales[3]);

        int64_t aux64;
        memcpy(&aux64, x[i].qh, 8);
        const __m128i haux128_0 = _mm_set_epi64x(aux64 >> 1, aux64);
        const __m128i haux128_1 = _mm_srli_epi16(haux128_0, 2);

        const __m128i q5h_0 = _mm_slli_epi16(_mm_andnot_si128(haux128_0, mone), 4);
        const __m128i q5h_1 = _mm_slli_epi16(_mm_andnot_si128(haux128_1, mone), 4);
        const __m128i q5h_2 = _mm_slli_epi16(_mm_andnot_si128(_mm_srli_epi16(haux128_0, 4), mone), 4);
        const __m128i q5h_3 = _mm_slli_epi16(_mm_andnot_si128(_mm_srli_epi16(haux128_1, 4), mone), 4);

        const __m128i q5l_0 = _mm_and_si128(_mm256_extractf128_si256(q5bits, 0), m4);
        const __m128i q5l_1 = _mm_and_si128(_mm256_extractf128_si256(q5bits, 1), m4);
        const __m128i q5l_2 = _mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q5bits, 0), 4), m4);
        const __m128i q5l_3 = _mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q5bits, 1), 4), m4);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m128i p16_0 = _mm_madd_epi16(scale_0, _mm_maddubs_epi16(q5l_0, _mm256_extractf128_si256(q8_0, 0)));
        const __m128i p16_1 = _mm_madd_epi16(scale_1, _mm_maddubs_epi16(q5l_1, _mm256_extractf128_si256(q8_0, 1)));
        const __m128i p16_2 = _mm_madd_epi16(scale_2, _mm_maddubs_epi16(q5l_2, _mm256_extractf128_si256(q8_1, 0)));
        const __m128i p16_3 = _mm_madd_epi16(scale_3, _mm_maddubs_epi16(q5l_3, _mm256_extractf128_si256(q8_1, 1)));
        const __m128i s16_0 = _mm_madd_epi16(scale_0, _mm_maddubs_epi16(q5h_0, _mm256_extractf128_si256(q8_0, 0)));
        const __m128i s16_1 = _mm_madd_epi16(scale_1, _mm_maddubs_epi16(q5h_1, _mm256_extractf128_si256(q8_0, 1)));
        const __m128i s16_2 = _mm_madd_epi16(scale_2, _mm_maddubs_epi16(q5h_2, _mm256_extractf128_si256(q8_1, 0)));
        const __m128i s16_3 = _mm_madd_epi16(scale_3, _mm_maddubs_epi16(q5h_3, _mm256_extractf128_si256(q8_1, 1)));

        const __m128i dot_0 = _mm_sub_epi32(_mm_add_epi32(p16_0, p16_2), _mm_add_epi32(s16_0, s16_2));
        const __m128i dot_1 = _mm_sub_epi32(_mm_add_epi32(p16_1, p16_3), _mm_add_epi32(s16_1, s16_3));

        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(dot_1, dot_0))), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __riscv_v_intrinsic

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * (float)x[i].d;
        const int8_t * sc = x[i].scales;

        const uint8_t * restrict q5 = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

        // load qh
        vuint8mf4_t qh_x1   = __riscv_vle8_v_u8mf4(qh, 8);
        vuint8mf2_t qh_x2   = __riscv_vlmul_ext_v_u8mf4_u8mf2(__riscv_vsrl_vx_u8mf4(qh_x1, 1, 8));

        size_t vl = 16;

        // combine both qh_1 and qh_2
        vuint8mf2_t qh_x = __riscv_vslideup_vx_u8mf2(__riscv_vlmul_ext_v_u8mf4_u8mf2(qh_x1), qh_x2, vl/2, vl);

        vuint8mf2_t qh_h0 = __riscv_vand_vx_u8mf2(__riscv_vnot_v_u8mf2(__riscv_vsll_vx_u8mf2(qh_x, 0x4, vl), vl), 16, vl);
        vuint8mf2_t qh_h1 = __riscv_vand_vx_u8mf2(__riscv_vnot_v_u8mf2(__riscv_vsll_vx_u8mf2(qh_x, 0x2, vl), vl), 16, vl);
        vuint8mf2_t qh_h2 = __riscv_vand_vx_u8mf2(__riscv_vnot_v_u8mf2(qh_x, vl), 16, vl);
        vuint8mf2_t qh_h3 = __riscv_vand_vx_u8mf2(__riscv_vnot_v_u8mf2(__riscv_vsrl_vx_u8mf2(qh_x, 0x4, vl), vl), 16, vl);

        vint8mf2_t qh_0 = __riscv_vreinterpret_v_u8mf2_i8mf2(qh_h0);
        vint8mf2_t qh_1 = __riscv_vreinterpret_v_u8mf2_i8mf2(qh_h1);
        vint8mf2_t qh_2 = __riscv_vreinterpret_v_u8mf2_i8mf2(qh_h2);
        vint8mf2_t qh_3 = __riscv_vreinterpret_v_u8mf2_i8mf2(qh_h3);

        // load q5
        vuint8mf2_t q5_x1  = __riscv_vle8_v_u8mf2(q5, vl);
        vuint8mf2_t q5_x2  = __riscv_vle8_v_u8mf2(q5+16, vl);

        vint8mf2_t q5s_0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(q5_x1, 0xF, vl));
        vint8mf2_t q5s_1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(q5_x2, 0xF, vl));
        vint8mf2_t q5s_2 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(q5_x1, 0x4, vl));
        vint8mf2_t q5s_3 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(q5_x2, 0x4, vl));

        vint8mf2_t q5_0 = __riscv_vsub_vv_i8mf2(q5s_0, qh_0, vl);
        vint8mf2_t q5_1 = __riscv_vsub_vv_i8mf2(q5s_1, qh_1, vl);
        vint8mf2_t q5_2 = __riscv_vsub_vv_i8mf2(q5s_2, qh_2, vl);
        vint8mf2_t q5_3 = __riscv_vsub_vv_i8mf2(q5s_3, qh_3, vl);

        // load Q8 and multiply it with Q5
        vint16m1_t p0 = __riscv_vwmul_vv_i16m1(q5_0, __riscv_vle8_v_i8mf2(q8, vl), vl);
        vint16m1_t p1 = __riscv_vwmul_vv_i16m1(q5_1, __riscv_vle8_v_i8mf2(q8+16, vl), vl);
        vint16m1_t p2 = __riscv_vwmul_vv_i16m1(q5_2, __riscv_vle8_v_i8mf2(q8+32, vl), vl);
        vint16m1_t p3 = __riscv_vwmul_vv_i16m1(q5_3, __riscv_vle8_v_i8mf2(q8+48, vl), vl);

        vint32m1_t vs_0 = __riscv_vwredsum_vs_i16m1_i32m1(p0, vzero, vl);
        vint32m1_t vs_1 = __riscv_vwredsum_vs_i16m1_i32m1(p1, vzero, vl);
        vint32m1_t vs_2 = __riscv_vwredsum_vs_i16m1_i32m1(p2, vzero, vl);
        vint32m1_t vs_3 = __riscv_vwredsum_vs_i16m1_i32m1(p3, vzero, vl);

        int32_t sumi1 = sc[0] * __riscv_vmv_x_s_i32m1_i32(vs_0);
        int32_t sumi2 = sc[1] * __riscv_vmv_x_s_i32m1_i32(vs_1);
        int32_t sumi3 = sc[2] * __riscv_vmv_x_s_i32m1_i32(vs_2);
        int32_t sumi4 = sc[3] * __riscv_vmv_x_s_i32m1_i32(vs_3);

        sumf += d * (sumi1 + sumi2 + sumi3 + sumi4);

    }

    *s = sumf;

#else

    int8_t aux8[QK_K];
    int16_t aux16[16];
    float   sums [8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const uint8_t * restrict hm = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        int8_t * restrict a = aux8;
        for (int l = 0; l < 32; ++l) {
            a[l+ 0] = q4[l] & 0xF;
            a[l+32] = q4[l]  >> 4;
        }
        for (int is = 0; is < 8; ++is) {
            uint8_t m = 1 << is;
            for (int l = 0; l < 8; ++l) a[8*is + l] -= (hm[l] & m ? 0 : 16);
        }

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const int8_t * restrict sc = x[i].scales;

        for (int j = 0; j < QK_K/16; ++j) {
            const float dl = d * sc[j];
            for (int l = 0; l < 16; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l <  8; ++l) sums[l] += dl * (aux16[l] + aux16[8+l]);
            q8 += 16; a += 16;
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
#endif


#if QK_K == 256

#ifdef __x86_64__

#define ggml_vec_dot_q6_K_q8_K ggml_vec_dot_q6_K_q8_K_default
#include "ggml_vec_dot_q6_K_q8_K_256.inc"
#undef ggml_vec_dot_q6_K_q8_K

#define ggml_vec_dot_q6_K_q8_K __attribute__((__target__("avx"))) ggml_vec_dot_q6_K_q8_K_avx
#ifndef __AVX__
#define __UNDEF_AVX__
#define __AVX__
#endif
#include "ggml_vec_dot_q6_K_q8_K_256.inc"
#ifdef __UNDEF_AVX__
#undef __UNDEF_AVX__
#undef __AVX__
#endif
#undef ggml_vec_dot_q6_K_q8_K

#define ggml_vec_dot_q6_K_q8_K __attribute__((__target__("avx2,fma"))) ggml_vec_dot_q6_K_q8_K_avx2
#ifndef __AVX2__
#define __UNDEF_AVX2__
#define __AVX2__
#endif
#include "ggml_vec_dot_q6_K_q8_K_256.inc"
#ifdef __UNDEF_AVX2__
#undef __UNDEF_AVX2__
#undef __AVX2__
#endif
#undef ggml_vec_dot_q6_K_q8_K

void ggml_vec_dot_q6_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    if (X86_HAVE(AVX2) && X86_HAVE(FMA)) {
        ggml_vec_dot_q6_K_q8_K_avx2(n, s, vx, vy);
    } else if (X86_HAVE(AVX)) {
        ggml_vec_dot_q6_K_q8_K_avx(n, s, vx, vy);
    } else {
        ggml_vec_dot_q6_K_q8_K_default(n, s, vx, vy);
    }
}

#else
#define static
#include "ggml_vec_dot_q6_K_q8_K_256.inc"
#undef static
#endif

#else

void ggml_vec_dot_q6_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q6_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    float sum = 0;

    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const int8x16_t  m32s = vdupq_n_s8(32);
#if defined(__ARM_FEATURE_DOTPROD)
    const int32x4_t  vzero = vdupq_n_s32(0);
#endif

    const uint8x16_t mone = vdupq_n_u8(3);

    int8x16x4_t q6bytes;
    uint8x16x4_t q6h;

    for (int i = 0; i < nb; ++i) {

        const float d_all = (float)x[i].d;

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const int8_t * restrict scale = x[i].scales;

        int32_t isum = 0;

        uint8x16_t   qhbits = vld1q_u8(qh);
        uint8x16x2_t q6bits = vld1q_u8_x2(q6);
        int8x16x4_t q8bytes = vld1q_s8_x4(q8);

        q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits), 4);
        uint8x16_t shifted = vshrq_n_u8(qhbits, 2);
        q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        shifted = vshrq_n_u8(qhbits, 4);
        q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        shifted = vshrq_n_u8(qhbits, 6);
        q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

        q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
        q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
        q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[2])), m32s);
        q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[3])), m32s);

#if defined(__ARM_FEATURE_DOTPROD)

        isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
#else

        int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                 vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                 vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];

        int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                 vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
        int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                 vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
        isum += vaddvq_s16(p2) * scale[2] + vaddvq_s16(p3) * scale[3];
#endif

        sum += isum * d_all * y[i].d;

    }
    *s = sum;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
        const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
        const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
        const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

        __m256i sumi = _mm256_setzero_si256();

        const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
        const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

        const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
        const __m128i q4bitsH = _mm_loadu_si128((const __m128i*)qh);

        const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 2), q4bitsH), m2), 4);
        const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 6), _mm_srli_epi16(q4bitsH, 4)), m2), 4);

        const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
        const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_1);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
        __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);

        __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
        __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);

        p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm256_sub_epi16(p16_1, q8s_1);

        p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
        p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);

        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(3);
    const __m128i m32s = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
        const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
        const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
        const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
        const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

        const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
        const __m128i q4bitsH = _mm_loadu_si128((const __m128i*)qh);

        const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH, m2), 4);
        const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 2), m2), 4);
        const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 4), m2), 4);
        const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 6), m2), 4);

        const __m128i q4_0 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 0), m4), q4h_0);
        const __m128i q4_1 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 1), m4), q4h_1);
        const __m128i q4_2 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 0), 4), m4), q4h_2);
        const __m128i q4_3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 1), 4), m4), q4h_3);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        __m128i q8s_0 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 0));
        __m128i q8s_1 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 1));
        __m128i q8s_2 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 0));
        __m128i q8s_3 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 1));

        __m128i p16_0 = _mm_maddubs_epi16(q4_0, _mm256_extractf128_si256(q8_0, 0));
        __m128i p16_1 = _mm_maddubs_epi16(q4_1, _mm256_extractf128_si256(q8_0, 1));
        __m128i p16_2 = _mm_maddubs_epi16(q4_2, _mm256_extractf128_si256(q8_1, 0));
        __m128i p16_3 = _mm_maddubs_epi16(q4_3, _mm256_extractf128_si256(q8_1, 1));

        p16_0 = _mm_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm_sub_epi16(p16_1, q8s_1);
        p16_2 = _mm_sub_epi16(p16_2, q8s_2);
        p16_3 = _mm_sub_epi16(p16_3, q8s_3);

        p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
        p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_0, scale_0)), p16_1);
        p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
        p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_1, scale_1)), p16_3);

        sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
        sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));

        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi_1, sumi_0))), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __riscv_v_intrinsic

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d_all = (float)x[i].d;

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const int8_t * restrict scale = x[i].scales;

        int32_t isum = 0;

        size_t vl = 16;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

        // load Q6
        vuint8mf2_t q6_0 = __riscv_vle8_v_u8mf2(q6, vl);
        vuint8mf2_t q6_1 = __riscv_vle8_v_u8mf2(q6+16, vl);

        // load qh
        vuint8mf2_t qh_x = __riscv_vle8_v_u8mf2(qh, vl);

        vuint8mf2_t qh0 = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(qh_x, 0x3, vl), 0x4, vl);
        qh_x = __riscv_vsrl_vx_u8mf2(qh_x, 0x2, vl);
        vuint8mf2_t qh1 = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(qh_x, 0x3, vl), 0x4, vl);
        qh_x = __riscv_vsrl_vx_u8mf2(qh_x, 0x2, vl);
        vuint8mf2_t qh2 = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(qh_x, 0x3, vl), 0x4, vl);
        qh_x = __riscv_vsrl_vx_u8mf2(qh_x, 0x2, vl);
        vuint8mf2_t qh3 = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(qh_x, 0x3, vl), 0x4, vl);

        vuint8mf2_t q6h_0 = __riscv_vor_vv_u8mf2(__riscv_vand_vx_u8mf2(q6_0, 0xF, vl), qh0, vl);
        vuint8mf2_t q6h_1 = __riscv_vor_vv_u8mf2(__riscv_vand_vx_u8mf2(q6_1, 0xF, vl), qh1, vl);
        vuint8mf2_t q6h_2 = __riscv_vor_vv_u8mf2(__riscv_vsrl_vx_u8mf2(q6_0, 0x4, vl), qh2, vl);
        vuint8mf2_t q6h_3 = __riscv_vor_vv_u8mf2(__riscv_vsrl_vx_u8mf2(q6_1, 0x4, vl), qh3, vl);

        vint8mf2_t q6v_0 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(q6h_0), 32, vl);
        vint8mf2_t q6v_1 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(q6h_1), 32, vl);
        vint8mf2_t q6v_2 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(q6h_2), 32, vl);
        vint8mf2_t q6v_3 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(q6h_3), 32, vl);

        // load Q8 and take product
        vint16m1_t p0 = __riscv_vwmul_vv_i16m1(q6v_0, __riscv_vle8_v_i8mf2(q8, vl), vl);
        vint16m1_t p1 = __riscv_vwmul_vv_i16m1(q6v_1, __riscv_vle8_v_i8mf2(q8+16, vl), vl);
        vint16m1_t p2 = __riscv_vwmul_vv_i16m1(q6v_2, __riscv_vle8_v_i8mf2(q8+32, vl), vl);
        vint16m1_t p3 = __riscv_vwmul_vv_i16m1(q6v_3, __riscv_vle8_v_i8mf2(q8+48, vl), vl);

        vint32m1_t vs_0 = __riscv_vwredsum_vs_i16m1_i32m1(p0, vzero, vl);
        vint32m1_t vs_1 = __riscv_vwredsum_vs_i16m1_i32m1(p1, vzero, vl);
        vint32m1_t vs_2 = __riscv_vwredsum_vs_i16m1_i32m1(p2, vzero, vl);
        vint32m1_t vs_3 = __riscv_vwredsum_vs_i16m1_i32m1(p3, vzero, vl);

        isum += __riscv_vmv_x_s_i32m1_i32(vs_0) * scale[0];
        isum += __riscv_vmv_x_s_i32m1_i32(vs_1) * scale[1];
        isum += __riscv_vmv_x_s_i32m1_i32(vs_2) * scale[2];
        isum += __riscv_vmv_x_s_i32m1_i32(vs_3) * scale[3];

        sumf += isum * d_all * y[i].d;

    }

    *s = sumf;

#else

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int l = 0; l < 16; ++l) {
            a[l+ 0] = (int8_t)((q4[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            a[l+16] = (int8_t)((q4[l+16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            a[l+32] = (int8_t)((q4[l+ 0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            a[l+48] = (int8_t)((q4[l+16] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        }
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

#endif
