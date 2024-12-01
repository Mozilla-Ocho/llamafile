/*
 * Tiny arbitrary precision floating point library
 *
 * Copyright (c) 2017-2021 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef LIBBF_H
#define LIBBF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if INTPTR_MAX >= INT64_MAX && !defined(_WIN32) && !defined(__TINYC__)
#define LIMB_LOG2_BITS 6
#else
#define LIMB_LOG2_BITS 5
#endif

#define LIMB_BITS (1 << LIMB_LOG2_BITS)

#if LIMB_BITS == 64
#ifndef INT128_MAX
__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;
#endif
typedef int64_t slimb_t;
typedef uint64_t limb_t;
typedef uint128_t dlimb_t;
#define BF_RAW_EXP_MIN INT64_MIN
#define BF_RAW_EXP_MAX INT64_MAX

#define LIMB_DIGITS 19
#define BF_DEC_BASE UINT64_C(10000000000000000000)

#else

typedef int32_t slimb_t;
typedef uint32_t limb_t;
typedef uint64_t dlimb_t;
#define BF_RAW_EXP_MIN INT32_MIN
#define BF_RAW_EXP_MAX INT32_MAX

#define LIMB_DIGITS 9
#define BF_DEC_BASE 1000000000U

#endif

/* in bits */
/* minimum number of bits for the exponent */
#define BF_EXP_BITS_MIN 3
/* maximum number of bits for the exponent */
#define BF_EXP_BITS_MAX (LIMB_BITS - 3)
/* extended range for exponent, used internally */
#define BF_EXT_EXP_BITS_MAX (BF_EXP_BITS_MAX + 1)
/* minimum possible precision */
#define BF_PREC_MIN 2
/* minimum possible precision */
#define BF_PREC_MAX (((limb_t)1 << (LIMB_BITS - 2)) - 2)
/* some operations support infinite precision */
#define BF_PREC_INF (BF_PREC_MAX + 1) /* infinite precision */

#if LIMB_BITS == 64
#define BF_CHKSUM_MOD (UINT64_C(975620677) * UINT64_C(9795002197))
#else
#define BF_CHKSUM_MOD 975620677U
#endif

#define BF_EXP_ZERO BF_RAW_EXP_MIN
#define BF_EXP_INF (BF_RAW_EXP_MAX - 1)
#define BF_EXP_NAN BF_RAW_EXP_MAX

/* +/-zero is represented with expn = BF_EXP_ZERO and len = 0,
   +/-infinity is represented with expn = BF_EXP_INF and len = 0,
   NaN is represented with expn = BF_EXP_NAN and len = 0 (sign is ignored)
 */
typedef struct {
    struct bf_context_t *ctx;
    int sign;
    slimb_t expn;
    limb_t len;
    limb_t *tab;
} bf_t;

typedef struct {
    /* must be kept identical to bf_t */
    struct bf_context_t *ctx;
    int sign;
    slimb_t expn;
    limb_t len;
    limb_t *tab;
} bfdec_t;

typedef enum {
    BF_RNDN, /* round to nearest, ties to even */
    BF_RNDZ, /* round to zero */
    BF_RNDD, /* round to -inf (the code relies on (BF_RNDD xor BF_RNDU) = 1) */
    BF_RNDU, /* round to +inf */
    BF_RNDNA, /* round to nearest, ties away from zero */
    BF_RNDA, /* round away from zero */
    BF_RNDF, /* faithful rounding (nondeterministic, either RNDD or RNDU,
                inexact flag is always set)  */
} bf_rnd_t;

/* allow subnormal numbers. Only available if the number of exponent
   bits is <= BF_EXP_BITS_USER_MAX and prec != BF_PREC_INF. */
#define BF_FLAG_SUBNORMAL (1 << 3)
/* 'prec' is the precision after the radix point instead of the whole
   mantissa. Can only be used with bf_round() and
   bfdec_[add|sub|mul|div|sqrt|round](). */
#define BF_FLAG_RADPNT_PREC (1 << 4)

#define BF_RND_MASK 0x7
#define BF_EXP_BITS_SHIFT 5
#define BF_EXP_BITS_MASK 0x3f

/* shortcut for bf_set_exp_bits(BF_EXT_EXP_BITS_MAX) */
#define BF_FLAG_EXT_EXP (BF_EXP_BITS_MASK << BF_EXP_BITS_SHIFT)

/* contains the rounding mode and number of exponents bits */
typedef uint32_t bf_flags_t;

typedef void *bf_realloc_func_t(void *opaque, void *ptr, size_t size);

typedef struct {
    bf_t val;
    limb_t prec;
} BFConstCache;

typedef struct bf_context_t {
    void *realloc_opaque;
    bf_realloc_func_t *realloc_func;
    BFConstCache log2_cache;
    BFConstCache pi_cache;
    struct BFNTTState *ntt_state;
} bf_context_t;

static inline int bf_get_exp_bits(bf_flags_t flags)
{
    int e;
    e = (flags >> BF_EXP_BITS_SHIFT) & BF_EXP_BITS_MASK;
    if (e == BF_EXP_BITS_MASK)
        return BF_EXP_BITS_MAX + 1;
    else
        return BF_EXP_BITS_MAX - e;
}

static inline bf_flags_t bf_set_exp_bits(int n)
{
    return ((BF_EXP_BITS_MAX - n) & BF_EXP_BITS_MASK) << BF_EXP_BITS_SHIFT;
}

/* returned status */
#define BF_ST_INVALID_OP  (1 << 0)
#define BF_ST_DIVIDE_ZERO (1 << 1)
#define BF_ST_OVERFLOW    (1 << 2)
#define BF_ST_UNDERFLOW   (1 << 3)
#define BF_ST_INEXACT     (1 << 4)
/* indicate that a memory allocation error occured. NaN is returned */
#define BF_ST_MEM_ERROR   (1 << 5)

#define BF_RADIX_MAX 36 /* maximum radix for bf_atof() and bf_ftoa() */

static inline slimb_t bf_max(slimb_t a, slimb_t b)
{
    if (a > b)
        return a;
    else
        return b;
}

static inline slimb_t bf_min(slimb_t a, slimb_t b)
{
    if (a < b)
        return a;
    else
        return b;
}

void bf_context_init(bf_context_t *s, bf_realloc_func_t *realloc_func,
                     void *realloc_opaque);
void bf_context_end(bf_context_t *s);
/* free memory allocated for the bf cache data */
void bf_clear_cache(bf_context_t *s);

static inline void *bf_realloc(bf_context_t *s, void *ptr, size_t size)
{
    return s->realloc_func(s->realloc_opaque, ptr, size);
}

/* 'size' must be != 0 */
static inline void *bf_malloc(bf_context_t *s, size_t size)
{
    return bf_realloc(s, NULL, size);
}

static inline void bf_free(bf_context_t *s, void *ptr)
{
    /* must test ptr otherwise equivalent to malloc(0) */
    if (ptr)
        bf_realloc(s, ptr, 0);
}

void bf_init(bf_context_t *s, bf_t *r);

static inline void bf_delete(bf_t *r)
{
    bf_context_t *s = r->ctx;
    /* we accept to delete a zeroed bf_t structure */
    if (s && r->tab) {
        bf_realloc(s, r->tab, 0);
    }
}

static inline void bf_neg(bf_t *r)
{
    r->sign ^= 1;
}

static inline int bf_is_finite(const bf_t *a)
{
    return (a->expn < BF_EXP_INF);
}

static inline int bf_is_nan(const bf_t *a)
{
    return (a->expn == BF_EXP_NAN);
}

static inline int bf_is_zero(const bf_t *a)
{
    return (a->expn == BF_EXP_ZERO);
}

static inline void bf_memcpy(bf_t *r, const bf_t *a)
{
    *r = *a;
}

int bf_set_ui(bf_t *r, uint64_t a);
int bf_set_si(bf_t *r, int64_t a);
void bf_set_nan(bf_t *r);
void bf_set_zero(bf_t *r, int is_neg);
void bf_set_inf(bf_t *r, int is_neg);
int bf_set(bf_t *r, const bf_t *a);
void bf_move(bf_t *r, bf_t *a);
int bf_get_float64(const bf_t *a, double *pres, bf_rnd_t rnd_mode);
int bf_set_float64(bf_t *a, double d);

int bf_cmpu(const bf_t *a, const bf_t *b);
int bf_cmp_full(const bf_t *a, const bf_t *b);
int bf_cmp(const bf_t *a, const bf_t *b);
static inline int bf_cmp_eq(const bf_t *a, const bf_t *b)
{
    return bf_cmp(a, b) == 0;
}

static inline int bf_cmp_le(const bf_t *a, const bf_t *b)
{
    return bf_cmp(a, b) <= 0;
}

static inline int bf_cmp_lt(const bf_t *a, const bf_t *b)
{
    return bf_cmp(a, b) < 0;
}

int bf_add(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec, bf_flags_t flags);
int bf_sub(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec, bf_flags_t flags);
int bf_add_si(bf_t *r, const bf_t *a, int64_t b1, limb_t prec, bf_flags_t flags);
int bf_mul(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec, bf_flags_t flags);
int bf_mul_ui(bf_t *r, const bf_t *a, uint64_t b1, limb_t prec, bf_flags_t flags);
int bf_mul_si(bf_t *r, const bf_t *a, int64_t b1, limb_t prec,
              bf_flags_t flags);
int bf_mul_2exp(bf_t *r, slimb_t e, limb_t prec, bf_flags_t flags);
int bf_div(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec, bf_flags_t flags);
#define BF_DIVREM_EUCLIDIAN BF_RNDF
int bf_divrem(bf_t *q, bf_t *r, const bf_t *a, const bf_t *b,
              limb_t prec, bf_flags_t flags, int rnd_mode);
int bf_rem(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
           bf_flags_t flags, int rnd_mode);
int bf_remquo(slimb_t *pq, bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
              bf_flags_t flags, int rnd_mode);
/* round to integer with infinite precision */
int bf_rint(bf_t *r, int rnd_mode);
int bf_round(bf_t *r, limb_t prec, bf_flags_t flags);
int bf_sqrtrem(bf_t *r, bf_t *rem1, const bf_t *a);
int bf_sqrt(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags);
slimb_t bf_get_exp_min(const bf_t *a);
int bf_logic_or(bf_t *r, const bf_t *a, const bf_t *b);
int bf_logic_xor(bf_t *r, const bf_t *a, const bf_t *b);
int bf_logic_and(bf_t *r, const bf_t *a, const bf_t *b);

/* additional flags for bf_atof */
/* do not accept hex radix prefix (0x or 0X) if radix = 0 or radix = 16 */
#define BF_ATOF_NO_HEX       (1 << 16)
/* accept binary (0b or 0B) or octal (0o or 0O) radix prefix if radix = 0 */
#define BF_ATOF_BIN_OCT      (1 << 17)
/* Do not parse NaN or Inf */
#define BF_ATOF_NO_NAN_INF   (1 << 18)
/* return the exponent separately */
#define BF_ATOF_EXPONENT       (1 << 19)

int bf_atof(bf_t *a, const char *str, const char **pnext, int radix,
            limb_t prec, bf_flags_t flags);
/* this version accepts prec = BF_PREC_INF and returns the radix
   exponent */
int bf_atof2(bf_t *r, slimb_t *pexponent,
             const char *str, const char **pnext, int radix,
             limb_t prec, bf_flags_t flags);
int bf_mul_pow_radix(bf_t *r, const bf_t *T, limb_t radix,
                     slimb_t expn, limb_t prec, bf_flags_t flags);


/* Conversion of floating point number to string. Return a null
   terminated string or NULL if memory error. *plen contains its
   length if plen != NULL.  The exponent letter is "e" for base 10,
   "p" for bases 2, 8, 16 with a binary exponent and "@" for the other
   bases. */

#define BF_FTOA_FORMAT_MASK (3 << 16)

/* fixed format: prec significant digits rounded with (flags &
   BF_RND_MASK). Exponential notation is used if too many zeros are
   needed.*/
#define BF_FTOA_FORMAT_FIXED (0 << 16)
/* fractional format: prec digits after the decimal point rounded with
   (flags & BF_RND_MASK) */
#define BF_FTOA_FORMAT_FRAC  (1 << 16)
/* free format:

   For binary radices with bf_ftoa() and for bfdec_ftoa(): use the minimum
   number of digits to represent 'a'. The precision and the rounding
   mode are ignored.

   For the non binary radices with bf_ftoa(): use as many digits as
   necessary so that bf_atof() return the same number when using
   precision 'prec', rounding to nearest and the subnormal
   configuration of 'flags'. The result is meaningful only if 'a' is
   already rounded to 'prec' bits. If the subnormal flag is set, the
   exponent in 'flags' must also be set to the desired exponent range.
*/
#define BF_FTOA_FORMAT_FREE  (2 << 16)
/* same as BF_FTOA_FORMAT_FREE but uses the minimum number of digits
   (takes more computation time). Identical to BF_FTOA_FORMAT_FREE for
   binary radices with bf_ftoa() and for bfdec_ftoa(). */
#define BF_FTOA_FORMAT_FREE_MIN (3 << 16)

/* force exponential notation for fixed or free format */
#define BF_FTOA_FORCE_EXP    (1 << 20)
/* add 0x prefix for base 16, 0o prefix for base 8 or 0b prefix for
   base 2 if non zero value */
#define BF_FTOA_ADD_PREFIX   (1 << 21)
/* return "Infinity" instead of "Inf" and add a "+" for positive
   exponents */
#define BF_FTOA_JS_QUIRKS    (1 << 22)

char *bf_ftoa(size_t *plen, const bf_t *a, int radix, limb_t prec,
              bf_flags_t flags);

/* modulo 2^n instead of saturation. NaN and infinity return 0 */
#define BF_GET_INT_MOD (1 << 0)
int bf_get_int32(int *pres, const bf_t *a, int flags);
int bf_get_int64(int64_t *pres, const bf_t *a, int flags);
int bf_get_uint64(uint64_t *pres, const bf_t *a);

/* the following functions are exported for testing only. */
void mp_print_str(const char *str, const limb_t *tab, limb_t n);
void bf_print_str(const char *str, const bf_t *a);
int bf_resize(bf_t *r, limb_t len);
int bf_get_fft_size(int *pdpl, int *pnb_mods, limb_t len);
int bf_normalize_and_round(bf_t *r, limb_t prec1, bf_flags_t flags);
int bf_can_round(const bf_t *a, slimb_t prec, bf_rnd_t rnd_mode, slimb_t k);
slimb_t bf_mul_log2_radix(slimb_t a1, unsigned int radix, int is_inv,
                          int is_ceil1);
int mp_mul(bf_context_t *s, limb_t *result,
           const limb_t *op1, limb_t op1_size,
           const limb_t *op2, limb_t op2_size);
limb_t mp_add(limb_t *res, const limb_t *op1, const limb_t *op2,
              limb_t n, limb_t carry);
limb_t mp_add_ui(limb_t *tab, limb_t b, size_t n);
int mp_sqrtrem(bf_context_t *s, limb_t *tabs, limb_t *taba, limb_t n);
int mp_recip(bf_context_t *s, limb_t *tabr, const limb_t *taba, limb_t n);
limb_t bf_isqrt(limb_t a);

/* transcendental functions */
int bf_const_log2(bf_t *T, limb_t prec, bf_flags_t flags);
int bf_const_pi(bf_t *T, limb_t prec, bf_flags_t flags);
int bf_exp(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags);
int bf_log(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags);
#define BF_POW_JS_QUIRKS (1 << 16) /* (+/-1)^(+/-Inf) = NaN, 1^NaN = NaN */
int bf_pow(bf_t *r, const bf_t *x, const bf_t *y, limb_t prec, bf_flags_t flags);
int bf_cos(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags);
int bf_sin(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags);
int bf_tan(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags);
int bf_atan(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags);
int bf_atan2(bf_t *r, const bf_t *y, const bf_t *x,
             limb_t prec, bf_flags_t flags);
int bf_asin(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags);
int bf_acos(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags);

/* decimal floating point */

static inline void bfdec_init(bf_context_t *s, bfdec_t *r)
{
    bf_init(s, (bf_t *)r);
}
static inline void bfdec_delete(bfdec_t *r)
{
    bf_delete((bf_t *)r);
}

static inline void bfdec_neg(bfdec_t *r)
{
    r->sign ^= 1;
}

static inline int bfdec_is_finite(const bfdec_t *a)
{
    return (a->expn < BF_EXP_INF);
}

static inline int bfdec_is_nan(const bfdec_t *a)
{
    return (a->expn == BF_EXP_NAN);
}

static inline int bfdec_is_zero(const bfdec_t *a)
{
    return (a->expn == BF_EXP_ZERO);
}

static inline void bfdec_memcpy(bfdec_t *r, const bfdec_t *a)
{
    bf_memcpy((bf_t *)r, (const bf_t *)a);
}

int bfdec_set_ui(bfdec_t *r, uint64_t a);
int bfdec_set_si(bfdec_t *r, int64_t a);

static inline void bfdec_set_nan(bfdec_t *r)
{
    bf_set_nan((bf_t *)r);
}
static inline void bfdec_set_zero(bfdec_t *r, int is_neg)
{
    bf_set_zero((bf_t *)r, is_neg);
}
static inline void bfdec_set_inf(bfdec_t *r, int is_neg)
{
    bf_set_inf((bf_t *)r, is_neg);
}
static inline int bfdec_set(bfdec_t *r, const bfdec_t *a)
{
    return bf_set((bf_t *)r, (bf_t *)a);
}
static inline void bfdec_move(bfdec_t *r, bfdec_t *a)
{
    bf_move((bf_t *)r, (bf_t *)a);
}
static inline int bfdec_cmpu(const bfdec_t *a, const bfdec_t *b)
{
    return bf_cmpu((const bf_t *)a, (const bf_t *)b);
}
static inline int bfdec_cmp_full(const bfdec_t *a, const bfdec_t *b)
{
    return bf_cmp_full((const bf_t *)a, (const bf_t *)b);
}
static inline int bfdec_cmp(const bfdec_t *a, const bfdec_t *b)
{
    return bf_cmp((const bf_t *)a, (const bf_t *)b);
}
static inline int bfdec_cmp_eq(const bfdec_t *a, const bfdec_t *b)
{
    return bfdec_cmp(a, b) == 0;
}
static inline int bfdec_cmp_le(const bfdec_t *a, const bfdec_t *b)
{
    return bfdec_cmp(a, b) <= 0;
}
static inline int bfdec_cmp_lt(const bfdec_t *a, const bfdec_t *b)
{
    return bfdec_cmp(a, b) < 0;
}

int bfdec_add(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags);
int bfdec_sub(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags);
int bfdec_add_si(bfdec_t *r, const bfdec_t *a, int64_t b1, limb_t prec,
                 bf_flags_t flags);
int bfdec_mul(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags);
int bfdec_mul_si(bfdec_t *r, const bfdec_t *a, int64_t b1, limb_t prec,
                 bf_flags_t flags);
int bfdec_div(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags);
int bfdec_divrem(bfdec_t *q, bfdec_t *r, const bfdec_t *a, const bfdec_t *b,
                 limb_t prec, bf_flags_t flags, int rnd_mode);
int bfdec_rem(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags, int rnd_mode);
int bfdec_rint(bfdec_t *r, int rnd_mode);
int bfdec_sqrt(bfdec_t *r, const bfdec_t *a, limb_t prec, bf_flags_t flags);
int bfdec_round(bfdec_t *r, limb_t prec, bf_flags_t flags);
int bfdec_get_int32(int *pres, const bfdec_t *a);
int bfdec_pow_ui(bfdec_t *r, const bfdec_t *a, limb_t b);

char *bfdec_ftoa(size_t *plen, const bfdec_t *a, limb_t prec, bf_flags_t flags);
int bfdec_atof(bfdec_t *r, const char *str, const char **pnext,
               limb_t prec, bf_flags_t flags);

/* the following functions are exported for testing only. */
extern const limb_t mp_pow_dec[LIMB_DIGITS + 1];
void bfdec_print_str(const char *str, const bfdec_t *a);
static inline int bfdec_resize(bfdec_t *r, limb_t len)
{
    return bf_resize((bf_t *)r, len);
}
int bfdec_normalize_and_round(bfdec_t *r, limb_t prec1, bf_flags_t flags);

#ifdef __cplusplus
} /* extern "C" { */
#endif

#endif /* LIBBF_H */
