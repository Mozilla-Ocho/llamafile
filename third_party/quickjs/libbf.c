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
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "cutils.h"
#include "libbf.h"

/* enable it to check the multiplication result */
//#define USE_MUL_CHECK
/* enable it to use FFT/NTT multiplication */
#define USE_FFT_MUL
/* enable decimal floating point support */
#define USE_BF_DEC

//#define inline __attribute__((always_inline))

#ifdef __AVX2__
#define FFT_MUL_THRESHOLD 100 /* in limbs of the smallest factor */
#else
#define FFT_MUL_THRESHOLD 100 /* in limbs of the smallest factor */
#endif

/* XXX: adjust */
#define DIVNORM_LARGE_THRESHOLD 50
#define UDIV1NORM_THRESHOLD 3

#if LIMB_BITS == 64
#define FMT_LIMB1 "%" PRIx64
#define FMT_LIMB "%016" PRIx64
#define PRId_LIMB PRId64
#define PRIu_LIMB PRIu64

#else

#define FMT_LIMB1 "%x"
#define FMT_LIMB "%08x"
#define PRId_LIMB "d"
#define PRIu_LIMB "u"

#endif

typedef intptr_t mp_size_t;

typedef int bf_op2_func_t(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
                          bf_flags_t flags);

#ifdef USE_FFT_MUL

#define FFT_MUL_R_OVERLAP_A (1 << 0)
#define FFT_MUL_R_OVERLAP_B (1 << 1)
#define FFT_MUL_R_NORESIZE  (1 << 2)

static no_inline int fft_mul(bf_context_t *s,
                             bf_t *res, limb_t *a_tab, limb_t a_len,
                             limb_t *b_tab, limb_t b_len, int mul_flags);
static void fft_clear_cache(bf_context_t *s);
#endif
#ifdef USE_BF_DEC
static limb_t get_digit(const limb_t *tab, limb_t len, slimb_t pos);
#endif


/* could leading zeros */
static inline int clz(limb_t a)
{
    if (a == 0) {
        return LIMB_BITS;
    } else {
#if LIMB_BITS == 64
        return clz64(a);
#else
        return clz32(a);
#endif
    }
}

static inline int ctz(limb_t a)
{
    if (a == 0) {
        return LIMB_BITS;
    } else {
#if LIMB_BITS == 64
        return ctz64(a);
#else
        return ctz32(a);
#endif
    }
}

static inline int ceil_log2(limb_t a)
{
    if (a <= 1)
        return 0;
    else
        return LIMB_BITS - clz(a - 1);
}

/* b must be >= 1 */
static inline slimb_t ceil_div(slimb_t a, slimb_t b)
{
    if (a >= 0)
        return (a + b - 1) / b;
    else
        return a / b;
}

/* b must be >= 1 */
static inline slimb_t floor_div(slimb_t a, slimb_t b)
{
    if (a >= 0) {
        return a / b;
    } else {
        return (a - b + 1) / b;
    }
}

/* return r = a modulo b (0 <= r <= b - 1. b must be >= 1 */
static inline limb_t smod(slimb_t a, slimb_t b)
{
    a = a % (slimb_t)b;
    if (a < 0)
        a += b;
    return a;
}

/* signed addition with saturation */
static inline slimb_t sat_add(slimb_t a, slimb_t b)
{
    slimb_t r;
    r = a + b;
    /* overflow ? */
    if (((a ^ r) & (b ^ r)) < 0)
        r = (a >> (LIMB_BITS - 1)) ^ (((limb_t)1 << (LIMB_BITS - 1)) - 1);
    return r;
}

static inline __maybe_unused limb_t shrd(limb_t low, limb_t high, long shift)
{
    if (shift != 0)
        low = (low >> shift) | (high << (LIMB_BITS - shift));
    return low;
}

static inline __maybe_unused limb_t shld(limb_t a1, limb_t a0, long shift)
{
    if (shift != 0)
        return (a1 << shift) | (a0 >> (LIMB_BITS - shift));
    else
        return a1;
}

#define malloc(s) malloc_is_forbidden(s)
#define free(p) free_is_forbidden(p)
#define realloc(p, s) realloc_is_forbidden(p, s)

void bf_context_init(bf_context_t *s, bf_realloc_func_t *realloc_func,
                     void *realloc_opaque)
{
    memset(s, 0, sizeof(*s));
    s->realloc_func = realloc_func;
    s->realloc_opaque = realloc_opaque;
}

void bf_context_end(bf_context_t *s)
{
    bf_clear_cache(s);
}

void bf_init(bf_context_t *s, bf_t *r)
{
    r->ctx = s;
    r->sign = 0;
    r->expn = BF_EXP_ZERO;
    r->len = 0;
    r->tab = NULL;
}

/* return 0 if OK, -1 if alloc error */
int bf_resize(bf_t *r, limb_t len)
{
    limb_t *tab;

    if (len != r->len) {
        tab = bf_realloc(r->ctx, r->tab, len * sizeof(limb_t));
        if (!tab && len != 0)
            return -1;
        r->tab = tab;
        r->len = len;
    }
    return 0;
}

/* return 0 or BF_ST_MEM_ERROR */
int bf_set_ui(bf_t *r, uint64_t a)
{
    r->sign = 0;
    if (a == 0) {
        r->expn = BF_EXP_ZERO;
        bf_resize(r, 0); /* cannot fail */
    }
#if LIMB_BITS == 32
    else if (a <= 0xffffffff)
#else
    else
#endif
    {
        int shift;
        if (bf_resize(r, 1))
            goto fail;
        shift = clz(a);
        r->tab[0] = a << shift;
        r->expn = LIMB_BITS - shift;
    }
#if LIMB_BITS == 32
    else {
        uint32_t a1, a0;
        int shift;
        if (bf_resize(r, 2))
            goto fail;
        a0 = a;
        a1 = a >> 32;
        shift = clz(a1);
        r->tab[0] = a0 << shift;
        r->tab[1] = shld(a1, a0, shift);
        r->expn = 2 * LIMB_BITS - shift;
    }
#endif
    return 0;
 fail:
    bf_set_nan(r);
    return BF_ST_MEM_ERROR;
}

/* return 0 or BF_ST_MEM_ERROR */
int bf_set_si(bf_t *r, int64_t a)
{
    int ret;

    if (a < 0) {
        ret = bf_set_ui(r, -a);
        r->sign = 1;
    } else {
        ret = bf_set_ui(r, a);
    }
    return ret;
}

void bf_set_nan(bf_t *r)
{
    bf_resize(r, 0); /* cannot fail */
    r->expn = BF_EXP_NAN;
    r->sign = 0;
}

void bf_set_zero(bf_t *r, int is_neg)
{
    bf_resize(r, 0); /* cannot fail */
    r->expn = BF_EXP_ZERO;
    r->sign = is_neg;
}

void bf_set_inf(bf_t *r, int is_neg)
{
    bf_resize(r, 0); /* cannot fail */
    r->expn = BF_EXP_INF;
    r->sign = is_neg;
}

/* return 0 or BF_ST_MEM_ERROR */
int bf_set(bf_t *r, const bf_t *a)
{
    if (r == a)
        return 0;
    if (bf_resize(r, a->len)) {
        bf_set_nan(r);
        return BF_ST_MEM_ERROR;
    }
    r->sign = a->sign;
    r->expn = a->expn;
    if (a->len > 0)
        memcpy(r->tab, a->tab, a->len * sizeof(limb_t));
    return 0;
}

/* equivalent to bf_set(r, a); bf_delete(a) */
void bf_move(bf_t *r, bf_t *a)
{
    bf_context_t *s = r->ctx;
    if (r == a)
        return;
    bf_free(s, r->tab);
    *r = *a;
}

static limb_t get_limbz(const bf_t *a, limb_t idx)
{
    if (idx >= a->len)
        return 0;
    else
        return a->tab[idx];
}

/* get LIMB_BITS at bit position 'pos' in tab */
static inline limb_t get_bits(const limb_t *tab, limb_t len, slimb_t pos)
{
    limb_t i, a0, a1;
    int p;

    i = pos >> LIMB_LOG2_BITS;
    p = pos & (LIMB_BITS - 1);
    if (i < len)
        a0 = tab[i];
    else
        a0 = 0;
    if (p == 0) {
        return a0;
    } else {
        i++;
        if (i < len)
            a1 = tab[i];
        else
            a1 = 0;
        return (a0 >> p) | (a1 << (LIMB_BITS - p));
    }
}

static inline limb_t get_bit(const limb_t *tab, limb_t len, slimb_t pos)
{
    slimb_t i;
    i = pos >> LIMB_LOG2_BITS;
    if (i < 0 || i >= len)
        return 0;
    return (tab[i] >> (pos & (LIMB_BITS - 1))) & 1;
}

static inline limb_t limb_mask(int start, int last)
{
    limb_t v;
    int n;
    n = last - start + 1;
    if (n == LIMB_BITS)
        v = -1;
    else
        v = (((limb_t)1 << n) - 1) << start;
    return v;
}

static limb_t mp_scan_nz(const limb_t *tab, mp_size_t n)
{
    mp_size_t i;
    for(i = 0; i < n; i++) {
        if (tab[i] != 0)
            return 1;
    }
    return 0;
}

/* return != 0 if one bit between 0 and bit_pos inclusive is not zero. */
static inline limb_t scan_bit_nz(const bf_t *r, slimb_t bit_pos)
{
    slimb_t pos;
    limb_t v;

    pos = bit_pos >> LIMB_LOG2_BITS;
    if (pos < 0)
        return 0;
    v = r->tab[pos] & limb_mask(0, bit_pos & (LIMB_BITS - 1));
    if (v != 0)
        return 1;
    pos--;
    while (pos >= 0) {
        if (r->tab[pos] != 0)
            return 1;
        pos--;
    }
    return 0;
}

/* return the addend for rounding. Note that prec can be <= 0 (for
   BF_FLAG_RADPNT_PREC) */
static int bf_get_rnd_add(int *pret, const bf_t *r, limb_t l,
                          slimb_t prec, int rnd_mode)
{
    int add_one, inexact;
    limb_t bit1, bit0;

    if (rnd_mode == BF_RNDF) {
        bit0 = 1; /* faithful rounding does not honor the INEXACT flag */
    } else {
        /* starting limb for bit 'prec + 1' */
        bit0 = scan_bit_nz(r, l * LIMB_BITS - 1 - bf_max(0, prec + 1));
    }

    /* get the bit at 'prec' */
    bit1 = get_bit(r->tab, l, l * LIMB_BITS - 1 - prec);
    inexact = (bit1 | bit0) != 0;

    add_one = 0;
    switch(rnd_mode) {
    case BF_RNDZ:
        break;
    case BF_RNDN:
        if (bit1) {
            if (bit0) {
                add_one = 1;
            } else {
                /* round to even */
                add_one =
                    get_bit(r->tab, l, l * LIMB_BITS - 1 - (prec - 1));
            }
        }
        break;
    case BF_RNDD:
    case BF_RNDU:
        if (r->sign == (rnd_mode == BF_RNDD))
            add_one = inexact;
        break;
    case BF_RNDA:
        add_one = inexact;
        break;
    case BF_RNDNA:
    case BF_RNDF:
        add_one = bit1;
        break;
    default:
        abort();
    }

    if (inexact)
        *pret |= BF_ST_INEXACT;
    return add_one;
}

static int bf_set_overflow(bf_t *r, int sign, limb_t prec, bf_flags_t flags)
{
    slimb_t i, l, e_max;
    int rnd_mode;

    rnd_mode = flags & BF_RND_MASK;
    if (prec == BF_PREC_INF ||
        rnd_mode == BF_RNDN ||
        rnd_mode == BF_RNDNA ||
        rnd_mode == BF_RNDA ||
        (rnd_mode == BF_RNDD && sign == 1) ||
        (rnd_mode == BF_RNDU && sign == 0)) {
        bf_set_inf(r, sign);
    } else {
        /* set to maximum finite number */
        l = (prec + LIMB_BITS - 1) / LIMB_BITS;
        if (bf_resize(r, l)) {
            bf_set_nan(r);
            return BF_ST_MEM_ERROR;
        }
        r->tab[0] = limb_mask((-prec) & (LIMB_BITS - 1),
                              LIMB_BITS - 1);
        for(i = 1; i < l; i++)
            r->tab[i] = (limb_t)-1;
        e_max = (limb_t)1 << (bf_get_exp_bits(flags) - 1);
        r->expn = e_max;
        r->sign = sign;
    }
    return BF_ST_OVERFLOW | BF_ST_INEXACT;
}

/* round to prec1 bits assuming 'r' is non zero and finite. 'r' is
   assumed to have length 'l' (1 <= l <= r->len). Note: 'prec1' can be
   infinite (BF_PREC_INF). 'ret' is 0 or BF_ST_INEXACT if the result
   is known to be inexact. Can fail with BF_ST_MEM_ERROR in case of
   overflow not returning infinity. */
static int __bf_round(bf_t *r, limb_t prec1, bf_flags_t flags, limb_t l,
                      int ret)
{
    limb_t v, a;
    int shift, add_one, rnd_mode;
    slimb_t i, bit_pos, pos, e_min, e_max, e_range, prec;

    /* e_min and e_max are computed to match the IEEE 754 conventions */
    e_range = (limb_t)1 << (bf_get_exp_bits(flags) - 1);
    e_min = -e_range + 3;
    e_max = e_range;

    if (flags & BF_FLAG_RADPNT_PREC) {
        /* 'prec' is the precision after the radix point */
        if (prec1 != BF_PREC_INF)
            prec = r->expn + prec1;
        else
            prec = prec1;
    } else if (unlikely(r->expn < e_min) && (flags & BF_FLAG_SUBNORMAL)) {
        /* restrict the precision in case of potentially subnormal
           result */
        assert(prec1 != BF_PREC_INF);
        prec = prec1 - (e_min - r->expn);
    } else {
        prec = prec1;
    }

    /* round to prec bits */
    rnd_mode = flags & BF_RND_MASK;
    add_one = bf_get_rnd_add(&ret, r, l, prec, rnd_mode);

    if (prec <= 0) {
        if (add_one) {
            bf_resize(r, 1); /* cannot fail */
            r->tab[0] = (limb_t)1 << (LIMB_BITS - 1);
            r->expn += 1 - prec;
            ret |= BF_ST_UNDERFLOW | BF_ST_INEXACT;
            return ret;
        } else {
            goto underflow;
        }
    } else if (add_one) {
        limb_t carry;

        /* add one starting at digit 'prec - 1' */
        bit_pos = l * LIMB_BITS - 1 - (prec - 1);
        pos = bit_pos >> LIMB_LOG2_BITS;
        carry = (limb_t)1 << (bit_pos & (LIMB_BITS - 1));

        for(i = pos; i < l; i++) {
            v = r->tab[i] + carry;
            carry = (v < carry);
            r->tab[i] = v;
            if (carry == 0)
                break;
        }
        if (carry) {
            /* shift right by one digit */
            v = 1;
            for(i = l - 1; i >= pos; i--) {
                a = r->tab[i];
                r->tab[i] = (a >> 1) | (v << (LIMB_BITS - 1));
                v = a;
            }
            r->expn++;
        }
    }

    /* check underflow */
    if (unlikely(r->expn < e_min)) {
        if (flags & BF_FLAG_SUBNORMAL) {
            /* if inexact, also set the underflow flag */
            if (ret & BF_ST_INEXACT)
                ret |= BF_ST_UNDERFLOW;
        } else {
        underflow:
            ret |= BF_ST_UNDERFLOW | BF_ST_INEXACT;
            bf_set_zero(r, r->sign);
            return ret;
        }
    }

    /* check overflow */
    if (unlikely(r->expn > e_max))
        return bf_set_overflow(r, r->sign, prec1, flags);

    /* keep the bits starting at 'prec - 1' */
    bit_pos = l * LIMB_BITS - 1 - (prec - 1);
    i = bit_pos >> LIMB_LOG2_BITS;
    if (i >= 0) {
        shift = bit_pos & (LIMB_BITS - 1);
        if (shift != 0)
            r->tab[i] &= limb_mask(shift, LIMB_BITS - 1);
    } else {
        i = 0;
    }
    /* remove trailing zeros */
    while (r->tab[i] == 0)
        i++;
    if (i > 0) {
        l -= i;
        memmove(r->tab, r->tab + i, l * sizeof(limb_t));
    }
    bf_resize(r, l); /* cannot fail */
    return ret;
}

/* 'r' must be a finite number. */
int bf_normalize_and_round(bf_t *r, limb_t prec1, bf_flags_t flags)
{
    limb_t l, v, a;
    int shift, ret;
    slimb_t i;

    //    bf_print_str("bf_renorm", r);
    l = r->len;
    while (l > 0 && r->tab[l - 1] == 0)
        l--;
    if (l == 0) {
        /* zero */
        r->expn = BF_EXP_ZERO;
        bf_resize(r, 0); /* cannot fail */
        ret = 0;
    } else {
        r->expn -= (r->len - l) * LIMB_BITS;
        /* shift to have the MSB set to '1' */
        v = r->tab[l - 1];
        shift = clz(v);
        if (shift != 0) {
            v = 0;
            for(i = 0; i < l; i++) {
                a = r->tab[i];
                r->tab[i] = (a << shift) | (v >> (LIMB_BITS - shift));
                v = a;
            }
            r->expn -= shift;
        }
        ret = __bf_round(r, prec1, flags, l, 0);
    }
    //    bf_print_str("r_final", r);
    return ret;
}

/* return true if rounding can be done at precision 'prec' assuming
   the exact result r is such that |r-a| <= 2^(EXP(a)-k). */
/* XXX: check the case where the exponent would be incremented by the
   rounding */
int bf_can_round(const bf_t *a, slimb_t prec, bf_rnd_t rnd_mode, slimb_t k)
{
    BOOL is_rndn;
    slimb_t bit_pos, n;
    limb_t bit;

    if (a->expn == BF_EXP_INF || a->expn == BF_EXP_NAN)
        return FALSE;
    if (rnd_mode == BF_RNDF) {
        return (k >= (prec + 1));
    }
    if (a->expn == BF_EXP_ZERO)
        return FALSE;
    is_rndn = (rnd_mode == BF_RNDN || rnd_mode == BF_RNDNA);
    if (k < (prec + 2))
        return FALSE;
    bit_pos = a->len * LIMB_BITS - 1 - prec;
    n = k - prec;
    /* bit pattern for RNDN or RNDNA: 0111.. or 1000...
       for other rounding modes: 000... or 111...
    */
    bit = get_bit(a->tab, a->len, bit_pos);
    bit_pos--;
    n--;
    bit ^= is_rndn;
    /* XXX: slow, but a few iterations on average */
    while (n != 0) {
        if (get_bit(a->tab, a->len, bit_pos) != bit)
            return TRUE;
        bit_pos--;
        n--;
    }
    return FALSE;
}

/* Cannot fail with BF_ST_MEM_ERROR. */
int bf_round(bf_t *r, limb_t prec, bf_flags_t flags)
{
    if (r->len == 0)
        return 0;
    return __bf_round(r, prec, flags, r->len, 0);
}

/* for debugging */
static __maybe_unused void dump_limbs(const char *str, const limb_t *tab, limb_t n)
{
    limb_t i;
    printf("%s: len=%" PRId_LIMB "\n", str, n);
    for(i = 0; i < n; i++) {
        printf("%" PRId_LIMB ": " FMT_LIMB "\n",
               i, tab[i]);
    }
}

void mp_print_str(const char *str, const limb_t *tab, limb_t n)
{
    slimb_t i;
    printf("%s= 0x", str);
    for(i = n - 1; i >= 0; i--) {
        if (i != (n - 1))
            printf("_");
        printf(FMT_LIMB, tab[i]);
    }
    printf("\n");
}

static __maybe_unused void mp_print_str_h(const char *str,
                                          const limb_t *tab, limb_t n,
                                          limb_t high)
{
    slimb_t i;
    printf("%s= 0x", str);
    printf(FMT_LIMB, high);
    for(i = n - 1; i >= 0; i--) {
        printf("_");
        printf(FMT_LIMB, tab[i]);
    }
    printf("\n");
}

/* for debugging */
void bf_print_str(const char *str, const bf_t *a)
{
    slimb_t i;
    printf("%s=", str);

    if (a->expn == BF_EXP_NAN) {
        printf("NaN");
    } else {
        if (a->sign)
            putchar('-');
        if (a->expn == BF_EXP_ZERO) {
            putchar('0');
        } else if (a->expn == BF_EXP_INF) {
            printf("Inf");
        } else {
            printf("0x0.");
            for(i = a->len - 1; i >= 0; i--)
                printf(FMT_LIMB, a->tab[i]);
            printf("p%" PRId_LIMB, a->expn);
        }
    }
    printf("\n");
}

/* compare the absolute value of 'a' and 'b'. Return < 0 if a < b, 0
   if a = b and > 0 otherwise. */
int bf_cmpu(const bf_t *a, const bf_t *b)
{
    slimb_t i;
    limb_t len, v1, v2;

    if (a->expn != b->expn) {
        if (a->expn < b->expn)
            return -1;
        else
            return 1;
    }
    len = bf_max(a->len, b->len);
    for(i = len - 1; i >= 0; i--) {
        v1 = get_limbz(a, a->len - len + i);
        v2 = get_limbz(b, b->len - len + i);
        if (v1 != v2) {
            if (v1 < v2)
                return -1;
            else
                return 1;
        }
    }
    return 0;
}

/* Full order: -0 < 0, NaN == NaN and NaN is larger than all other numbers */
int bf_cmp_full(const bf_t *a, const bf_t *b)
{
    int res;

    if (a->expn == BF_EXP_NAN || b->expn == BF_EXP_NAN) {
        if (a->expn == b->expn)
            res = 0;
        else if (a->expn == BF_EXP_NAN)
            res = 1;
        else
            res = -1;
    } else if (a->sign != b->sign) {
        res = 1 - 2 * a->sign;
    } else {
        res = bf_cmpu(a, b);
        if (a->sign)
            res = -res;
    }
    return res;
}

/* Standard floating point comparison: return 2 if one of the operands
   is NaN (unordered) or -1, 0, 1 depending on the ordering assuming
   -0 == +0 */
int bf_cmp(const bf_t *a, const bf_t *b)
{
    int res;

    if (a->expn == BF_EXP_NAN || b->expn == BF_EXP_NAN) {
        res = 2;
    } else if (a->sign != b->sign) {
        if (a->expn == BF_EXP_ZERO && b->expn == BF_EXP_ZERO)
            res = 0;
        else
            res = 1 - 2 * a->sign;
    } else {
        res = bf_cmpu(a, b);
        if (a->sign)
            res = -res;
    }
    return res;
}

/* Compute the number of bits 'n' matching the pattern:
   a= X1000..0
   b= X0111..1

   When computing a-b, the result will have at least n leading zero
   bits.

   Precondition: a > b and a.expn - b.expn = 0 or 1
*/
static limb_t count_cancelled_bits(const bf_t *a, const bf_t *b)
{
    slimb_t bit_offset, b_offset, n;
    int p, p1;
    limb_t v1, v2, mask;

    bit_offset = a->len * LIMB_BITS - 1;
    b_offset = (b->len - a->len) * LIMB_BITS - (LIMB_BITS - 1) +
        a->expn - b->expn;
    n = 0;

    /* first search the equals bits */
    for(;;) {
        v1 = get_limbz(a, bit_offset >> LIMB_LOG2_BITS);
        v2 = get_bits(b->tab, b->len, bit_offset + b_offset);
        //        printf("v1=" FMT_LIMB " v2=" FMT_LIMB "\n", v1, v2);
        if (v1 != v2)
            break;
        n += LIMB_BITS;
        bit_offset -= LIMB_BITS;
    }
    /* find the position of the first different bit */
    p = clz(v1 ^ v2) + 1;
    n += p;
    /* then search for '0' in a and '1' in b */
    p = LIMB_BITS - p;
    if (p > 0) {
        /* search in the trailing p bits of v1 and v2 */
        mask = limb_mask(0, p - 1);
        p1 = bf_min(clz(v1 & mask), clz((~v2) & mask)) - (LIMB_BITS - p);
        n += p1;
        if (p1 != p)
            goto done;
    }
    bit_offset -= LIMB_BITS;
    for(;;) {
        v1 = get_limbz(a, bit_offset >> LIMB_LOG2_BITS);
        v2 = get_bits(b->tab, b->len, bit_offset + b_offset);
        //        printf("v1=" FMT_LIMB " v2=" FMT_LIMB "\n", v1, v2);
        if (v1 != 0 || v2 != -1) {
            /* different: count the matching bits */
            p1 = bf_min(clz(v1), clz(~v2));
            n += p1;
            break;
        }
        n += LIMB_BITS;
        bit_offset -= LIMB_BITS;
    }
 done:
    return n;
}

static int bf_add_internal(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
                           bf_flags_t flags, int b_neg)
{
    const bf_t *tmp;
    int is_sub, ret, cmp_res, a_sign, b_sign;

    a_sign = a->sign;
    b_sign = b->sign ^ b_neg;
    is_sub = a_sign ^ b_sign;
    cmp_res = bf_cmpu(a, b);
    if (cmp_res < 0) {
        tmp = a;
        a = b;
        b = tmp;
        a_sign = b_sign; /* b_sign is never used later */
    }
    /* abs(a) >= abs(b) */
    if (cmp_res == 0 && is_sub && a->expn < BF_EXP_INF) {
        /* zero result */
        bf_set_zero(r, (flags & BF_RND_MASK) == BF_RNDD);
        ret = 0;
    } else if (a->len == 0 || b->len == 0) {
        ret = 0;
        if (a->expn >= BF_EXP_INF) {
            if (a->expn == BF_EXP_NAN) {
                /* at least one operand is NaN */
                bf_set_nan(r);
            } else if (b->expn == BF_EXP_INF && is_sub) {
                /* infinities with different signs */
                bf_set_nan(r);
                ret = BF_ST_INVALID_OP;
            } else {
                bf_set_inf(r, a_sign);
            }
        } else {
            /* at least one zero and not subtract */
            bf_set(r, a);
            r->sign = a_sign;
            goto renorm;
        }
    } else {
        slimb_t d, a_offset, b_bit_offset, i, cancelled_bits;
        limb_t carry, v1, v2, u, r_len, carry1, precl, tot_len, z, sub_mask;

        r->sign = a_sign;
        r->expn = a->expn;
        d = a->expn - b->expn;
        /* must add more precision for the leading cancelled bits in
           subtraction */
        if (is_sub) {
            if (d <= 1)
                cancelled_bits = count_cancelled_bits(a, b);
            else
                cancelled_bits = 1;
        } else {
            cancelled_bits = 0;
        }

        /* add two extra bits for rounding */
        precl = (cancelled_bits + prec + 2 + LIMB_BITS - 1) / LIMB_BITS;
        tot_len = bf_max(a->len, b->len + (d + LIMB_BITS - 1) / LIMB_BITS);
        r_len = bf_min(precl, tot_len);
        if (bf_resize(r, r_len))
            goto fail;
        a_offset = a->len - r_len;
        b_bit_offset = (b->len - r_len) * LIMB_BITS + d;

        /* compute the bits before for the rounding */
        carry = is_sub;
        z = 0;
        sub_mask = -is_sub;
        i = r_len - tot_len;
        while (i < 0) {
            slimb_t ap, bp;
            BOOL inflag;

            ap = a_offset + i;
            bp = b_bit_offset + i * LIMB_BITS;
            inflag = FALSE;
            if (ap >= 0 && ap < a->len) {
                v1 = a->tab[ap];
                inflag = TRUE;
            } else {
                v1 = 0;
            }
            if (bp + LIMB_BITS > 0 && bp < (slimb_t)(b->len * LIMB_BITS)) {
                v2 = get_bits(b->tab, b->len, bp);
                inflag = TRUE;
            } else {
                v2 = 0;
            }
            if (!inflag) {
                /* outside 'a' and 'b': go directly to the next value
                   inside a or b so that the running time does not
                   depend on the exponent difference */
                i = 0;
                if (ap < 0)
                    i = bf_min(i, -a_offset);
                /* b_bit_offset + i * LIMB_BITS + LIMB_BITS >= 1
                   equivalent to
                   i >= ceil(-b_bit_offset + 1 - LIMB_BITS) / LIMB_BITS)
                */
                if (bp + LIMB_BITS <= 0)
                    i = bf_min(i, (-b_bit_offset) >> LIMB_LOG2_BITS);
            } else {
                i++;
            }
            v2 ^= sub_mask;
            u = v1 + v2;
            carry1 = u < v1;
            u += carry;
            carry = (u < carry) | carry1;
            z |= u;
        }
        /* and the result */
        for(i = 0; i < r_len; i++) {
            v1 = get_limbz(a, a_offset + i);
            v2 = get_bits(b->tab, b->len, b_bit_offset + i * LIMB_BITS);
            v2 ^= sub_mask;
            u = v1 + v2;
            carry1 = u < v1;
            u += carry;
            carry = (u < carry) | carry1;
            r->tab[i] = u;
        }
        /* set the extra bits for the rounding */
        r->tab[0] |= (z != 0);

        /* carry is only possible in add case */
        if (!is_sub && carry) {
            if (bf_resize(r, r_len + 1))
                goto fail;
            r->tab[r_len] = 1;
            r->expn += LIMB_BITS;
        }
    renorm:
        ret = bf_normalize_and_round(r, prec, flags);
    }
    return ret;
 fail:
    bf_set_nan(r);
    return BF_ST_MEM_ERROR;
}

static int __bf_add(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
                     bf_flags_t flags)
{
    return bf_add_internal(r, a, b, prec, flags, 0);
}

static int __bf_sub(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
                     bf_flags_t flags)
{
    return bf_add_internal(r, a, b, prec, flags, 1);
}

limb_t mp_add(limb_t *res, const limb_t *op1, const limb_t *op2,
              limb_t n, limb_t carry)
{
    slimb_t i;
    limb_t k, a, v, k1;

    k = carry;
    for(i=0;i<n;i++) {
        v = op1[i];
        a = v + op2[i];
        k1 = a < v;
        a = a + k;
        k = (a < k) | k1;
        res[i] = a;
    }
    return k;
}

limb_t mp_add_ui(limb_t *tab, limb_t b, size_t n)
{
    size_t i;
    limb_t k, a;

    k=b;
    for(i=0;i<n;i++) {
        if (k == 0)
            break;
        a = tab[i] + k;
        k = (a < k);
        tab[i] = a;
    }
    return k;
}

limb_t mp_sub(limb_t *res, const limb_t *op1, const limb_t *op2,
              mp_size_t n, limb_t carry)
{
    int i;
    limb_t k, a, v, k1;

    k = carry;
    for(i=0;i<n;i++) {
        v = op1[i];
        a = v - op2[i];
        k1 = a > v;
        v = a - k;
        k = (v > a) | k1;
        res[i] = v;
    }
    return k;
}

/* compute 0 - op2 */
static limb_t mp_neg(limb_t *res, const limb_t *op2, mp_size_t n, limb_t carry)
{
    int i;
    limb_t k, a, v, k1;

    k = carry;
    for(i=0;i<n;i++) {
        v = 0;
        a = v - op2[i];
        k1 = a > v;
        v = a - k;
        k = (v > a) | k1;
        res[i] = v;
    }
    return k;
}

limb_t mp_sub_ui(limb_t *tab, limb_t b, mp_size_t n)
{
    mp_size_t i;
    limb_t k, a, v;

    k=b;
    for(i=0;i<n;i++) {
        v = tab[i];
        a = v - k;
        k = a > v;
        tab[i] = a;
        if (k == 0)
            break;
    }
    return k;
}

/* r = (a + high*B^n) >> shift. Return the remainder r (0 <= r < 2^shift).
   1 <= shift <= LIMB_BITS - 1 */
static limb_t mp_shr(limb_t *tab_r, const limb_t *tab, mp_size_t n,
                     int shift, limb_t high)
{
    mp_size_t i;
    limb_t l, a;

    assert(shift >= 1 && shift < LIMB_BITS);
    l = high;
    for(i = n - 1; i >= 0; i--) {
        a = tab[i];
        tab_r[i] = (a >> shift) | (l << (LIMB_BITS - shift));
        l = a;
    }
    return l & (((limb_t)1 << shift) - 1);
}

/* tabr[] = taba[] * b + l. Return the high carry */
static limb_t mp_mul1(limb_t *tabr, const limb_t *taba, limb_t n,
                      limb_t b, limb_t l)
{
    limb_t i;
    dlimb_t t;

    for(i = 0; i < n; i++) {
        t = (dlimb_t)taba[i] * (dlimb_t)b + l;
        tabr[i] = t;
        l = t >> LIMB_BITS;
    }
    return l;
}

/* tabr[] += taba[] * b, return the high word. */
static limb_t mp_add_mul1(limb_t *tabr, const limb_t *taba, limb_t n,
                          limb_t b)
{
    limb_t i, l;
    dlimb_t t;

    l = 0;
    for(i = 0; i < n; i++) {
        t = (dlimb_t)taba[i] * (dlimb_t)b + l + tabr[i];
        tabr[i] = t;
        l = t >> LIMB_BITS;
    }
    return l;
}

/* size of the result : op1_size + op2_size. */
static void mp_mul_basecase(limb_t *result,
                            const limb_t *op1, limb_t op1_size,
                            const limb_t *op2, limb_t op2_size)
{
    limb_t i, r;

    result[op1_size] = mp_mul1(result, op1, op1_size, op2[0], 0);
    for(i=1;i<op2_size;i++) {
        r = mp_add_mul1(result + i, op1, op1_size, op2[i]);
        result[i + op1_size] = r;
    }
}

/* return 0 if OK, -1 if memory error */
/* XXX: change API so that result can be allocated */
int mp_mul(bf_context_t *s, limb_t *result,
           const limb_t *op1, limb_t op1_size,
           const limb_t *op2, limb_t op2_size)
{
#ifdef USE_FFT_MUL
    if (unlikely(bf_min(op1_size, op2_size) >= FFT_MUL_THRESHOLD)) {
        bf_t r_s, *r = &r_s;
        r->tab = result;
        /* XXX: optimize memory usage in API */
        if (fft_mul(s, r, (limb_t *)op1, op1_size,
                    (limb_t *)op2, op2_size, FFT_MUL_R_NORESIZE))
            return -1;
    } else
#endif
    {
        mp_mul_basecase(result, op1, op1_size, op2, op2_size);
    }
    return 0;
}

/* tabr[] -= taba[] * b. Return the value to substract to the high
   word. */
static limb_t mp_sub_mul1(limb_t *tabr, const limb_t *taba, limb_t n,
                          limb_t b)
{
    limb_t i, l;
    dlimb_t t;

    l = 0;
    for(i = 0; i < n; i++) {
        t = tabr[i] - (dlimb_t)taba[i] * (dlimb_t)b - l;
        tabr[i] = t;
        l = -(t >> LIMB_BITS);
    }
    return l;
}

/* WARNING: d must be >= 2^(LIMB_BITS-1) */
static inline limb_t udiv1norm_init(limb_t d)
{
    limb_t a0, a1;
    a1 = -d - 1;
    a0 = -1;
    return (((dlimb_t)a1 << LIMB_BITS) | a0) / d;
}

/* return the quotient and the remainder in '*pr'of 'a1*2^LIMB_BITS+a0
   / d' with 0 <= a1 < d. */
static inline limb_t udiv1norm(limb_t *pr, limb_t a1, limb_t a0,
                                limb_t d, limb_t d_inv)
{
    limb_t n1m, n_adj, q, r, ah;
    dlimb_t a;
    n1m = ((slimb_t)a0 >> (LIMB_BITS - 1));
    n_adj = a0 + (n1m & d);
    a = (dlimb_t)d_inv * (a1 - n1m) + n_adj;
    q = (a >> LIMB_BITS) + a1;
    /* compute a - q * r and update q so that the remainder is\
       between 0 and d - 1 */
    a = ((dlimb_t)a1 << LIMB_BITS) | a0;
    a = a - (dlimb_t)q * d - d;
    ah = a >> LIMB_BITS;
    q += 1 + ah;
    r = (limb_t)a + (ah & d);
    *pr = r;
    return q;
}

/* b must be >= 1 << (LIMB_BITS - 1) */
static limb_t mp_div1norm(limb_t *tabr, const limb_t *taba, limb_t n,
                          limb_t b, limb_t r)
{
    slimb_t i;

    if (n >= UDIV1NORM_THRESHOLD) {
        limb_t b_inv;
        b_inv = udiv1norm_init(b);
        for(i = n - 1; i >= 0; i--) {
            tabr[i] = udiv1norm(&r, r, taba[i], b, b_inv);
        }
    } else {
        dlimb_t a1;
        for(i = n - 1; i >= 0; i--) {
            a1 = ((dlimb_t)r << LIMB_BITS) | taba[i];
            tabr[i] = a1 / b;
            r = a1 % b;
        }
    }
    return r;
}

static int mp_divnorm_large(bf_context_t *s,
                            limb_t *tabq, limb_t *taba, limb_t na,
                            const limb_t *tabb, limb_t nb);

/* base case division: divides taba[0..na-1] by tabb[0..nb-1]. tabb[nb
   - 1] must be >= 1 << (LIMB_BITS - 1). na - nb must be >= 0. 'taba'
   is modified and contains the remainder (nb limbs). tabq[0..na-nb]
   contains the quotient with tabq[na - nb] <= 1. */
static int mp_divnorm(bf_context_t *s, limb_t *tabq, limb_t *taba, limb_t na,
                      const limb_t *tabb, limb_t nb)
{
    limb_t r, a, c, q, v, b1, b1_inv, n, dummy_r;
    slimb_t i, j;

    b1 = tabb[nb - 1];
    if (nb == 1) {
        taba[0] = mp_div1norm(tabq, taba, na, b1, 0);
        return 0;
    }
    n = na - nb;
    if (bf_min(n, nb) >= DIVNORM_LARGE_THRESHOLD) {
        return mp_divnorm_large(s, tabq, taba, na, tabb, nb);
    }

    if (n >= UDIV1NORM_THRESHOLD)
        b1_inv = udiv1norm_init(b1);
    else
        b1_inv = 0;

    /* first iteration: the quotient is only 0 or 1 */
    q = 1;
    for(j = nb - 1; j >= 0; j--) {
        if (taba[n + j] != tabb[j]) {
            if (taba[n + j] < tabb[j])
                q = 0;
            break;
        }
    }
    tabq[n] = q;
    if (q) {
        mp_sub(taba + n, taba + n, tabb, nb, 0);
    }

    for(i = n - 1; i >= 0; i--) {
        if (unlikely(taba[i + nb] >= b1)) {
            q = -1;
        } else if (b1_inv) {
            q = udiv1norm(&dummy_r, taba[i + nb], taba[i + nb - 1], b1, b1_inv);
        } else {
            dlimb_t al;
            al = ((dlimb_t)taba[i + nb] << LIMB_BITS) | taba[i + nb - 1];
            q = al / b1;
            r = al % b1;
        }
        r = mp_sub_mul1(taba + i, tabb, nb, q);

        v = taba[i + nb];
        a = v - r;
        c = (a > v);
        taba[i + nb] = a;

        if (c != 0) {
            /* negative result */
            for(;;) {
                q--;
                c = mp_add(taba + i, taba + i, tabb, nb, 0);
                /* propagate carry and test if positive result */
                if (c != 0) {
                    if (++taba[i + nb] == 0) {
                        break;
                    }
                }
            }
        }
        tabq[i] = q;
    }
    return 0;
}

/* compute r=B^(2*n)/a such as a*r < B^(2*n) < a*r + 2 with n >= 1. 'a'
   has n limbs with a[n-1] >= B/2 and 'r' has n+1 limbs with r[n] = 1.

   See Modern Computer Arithmetic by Richard P. Brent and Paul
   Zimmermann, algorithm 3.5 */
int mp_recip(bf_context_t *s, limb_t *tabr, const limb_t *taba, limb_t n)
{
    mp_size_t l, h, k, i;
    limb_t *tabxh, *tabt, c, *tabu;

    if (n <= 2) {
        /* return ceil(B^(2*n)/a) - 1 */
        /* XXX: could avoid allocation */
        tabu = bf_malloc(s, sizeof(limb_t) * (2 * n + 1));
        tabt = bf_malloc(s, sizeof(limb_t) * (n + 2));
        if (!tabt || !tabu)
            goto fail;
        for(i = 0; i < 2 * n; i++)
            tabu[i] = 0;
        tabu[2 * n] = 1;
        if (mp_divnorm(s, tabt, tabu, 2 * n + 1, taba, n))
            goto fail;
        for(i = 0; i < n + 1; i++)
            tabr[i] = tabt[i];
        if (mp_scan_nz(tabu, n) == 0) {
            /* only happens for a=B^n/2 */
            mp_sub_ui(tabr, 1, n + 1);
        }
    } else {
        l = (n - 1) / 2;
        h = n - l;
        /* n=2p  -> l=p-1, h = p + 1, k = p + 3
           n=2p+1-> l=p,  h = p + 1; k = p + 2
        */
        tabt = bf_malloc(s, sizeof(limb_t) * (n + h + 1));
        tabu = bf_malloc(s, sizeof(limb_t) * (n + 2 * h - l + 2));
        if (!tabt || !tabu)
            goto fail;
        tabxh = tabr + l;
        if (mp_recip(s, tabxh, taba + l, h))
            goto fail;
        if (mp_mul(s, tabt, taba, n, tabxh, h + 1)) /* n + h + 1 limbs */
            goto fail;
        while (tabt[n + h] != 0) {
            mp_sub_ui(tabxh, 1, h + 1);
            c = mp_sub(tabt, tabt, taba, n, 0);
            mp_sub_ui(tabt + n, c, h + 1);
        }
        /* T = B^(n+h) - T */
        mp_neg(tabt, tabt, n + h + 1, 0);
        tabt[n + h]++;
        if (mp_mul(s, tabu, tabt + l, n + h + 1 - l, tabxh, h + 1))
            goto fail;
        /* n + 2*h - l + 2 limbs */
        k = 2 * h - l;
        for(i = 0; i < l; i++)
            tabr[i] = tabu[i + k];
        mp_add(tabr + l, tabr + l, tabu + 2 * h, h, 0);
    }
    bf_free(s, tabt);
    bf_free(s, tabu);
    return 0;
 fail:
    bf_free(s, tabt);
    bf_free(s, tabu);
    return -1;
}

/* return -1, 0 or 1 */
static int mp_cmp(const limb_t *taba, const limb_t *tabb, mp_size_t n)
{
    mp_size_t i;
    for(i = n - 1; i >= 0; i--) {
        if (taba[i] != tabb[i]) {
            if (taba[i] < tabb[i])
                return -1;
            else
                return 1;
        }
    }
    return 0;
}

//#define DEBUG_DIVNORM_LARGE
//#define DEBUG_DIVNORM_LARGE2

/* subquadratic divnorm */
static int mp_divnorm_large(bf_context_t *s,
                            limb_t *tabq, limb_t *taba, limb_t na,
                            const limb_t *tabb, limb_t nb)
{
    limb_t *tabb_inv, nq, *tabt, i, n;
    nq = na - nb;
#ifdef DEBUG_DIVNORM_LARGE
    printf("na=%d nb=%d nq=%d\n", (int)na, (int)nb, (int)nq);
    mp_print_str("a", taba, na);
    mp_print_str("b", tabb, nb);
#endif
    assert(nq >= 1);
    n = nq;
    if (nq < nb)
        n++;
    tabb_inv = bf_malloc(s, sizeof(limb_t) * (n + 1));
    tabt = bf_malloc(s, sizeof(limb_t) * 2 * (n + 1));
    if (!tabb_inv || !tabt)
        goto fail;

    if (n >= nb) {
        for(i = 0; i < n - nb; i++)
            tabt[i] = 0;
        for(i = 0; i < nb; i++)
            tabt[i + n - nb] = tabb[i];
    } else {
        /* truncate B: need to increment it so that the approximate
           inverse is smaller that the exact inverse */
        for(i = 0; i < n; i++)
            tabt[i] = tabb[i + nb - n];
        if (mp_add_ui(tabt, 1, n)) {
            /* tabt = B^n : tabb_inv = B^n */
            memset(tabb_inv, 0, n * sizeof(limb_t));
            tabb_inv[n] = 1;
            goto recip_done;
        }
    }
    if (mp_recip(s, tabb_inv, tabt, n))
        goto fail;
 recip_done:
    /* Q=A*B^-1 */
    if (mp_mul(s, tabt, tabb_inv, n + 1, taba + na - (n + 1), n + 1))
        goto fail;

    for(i = 0; i < nq + 1; i++)
        tabq[i] = tabt[i + 2 * (n + 1) - (nq + 1)];
#ifdef DEBUG_DIVNORM_LARGE
    mp_print_str("q", tabq, nq + 1);
#endif

    bf_free(s, tabt);
    bf_free(s, tabb_inv);
    tabb_inv = NULL;

    /* R=A-B*Q */
    tabt = bf_malloc(s, sizeof(limb_t) * (na + 1));
    if (!tabt)
        goto fail;
    if (mp_mul(s, tabt, tabq, nq + 1, tabb, nb))
        goto fail;
    /* we add one more limb for the result */
    mp_sub(taba, taba, tabt, nb + 1, 0);
    bf_free(s, tabt);
    /* the approximated quotient is smaller than than the exact one,
       hence we may have to increment it */
#ifdef DEBUG_DIVNORM_LARGE2
    int cnt = 0;
    static int cnt_max;
#endif
    for(;;) {
        if (taba[nb] == 0 && mp_cmp(taba, tabb, nb) < 0)
            break;
        taba[nb] -= mp_sub(taba, taba, tabb, nb, 0);
        mp_add_ui(tabq, 1, nq + 1);
#ifdef DEBUG_DIVNORM_LARGE2
        cnt++;
#endif
    }
#ifdef DEBUG_DIVNORM_LARGE2
    if (cnt > cnt_max) {
        cnt_max = cnt;
        printf("\ncnt=%d nq=%d nb=%d\n", cnt_max, (int)nq, (int)nb);
    }
#endif
    return 0;
 fail:
    bf_free(s, tabb_inv);
    bf_free(s, tabt);
    return -1;
}

int bf_mul(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
           bf_flags_t flags)
{
    int ret, r_sign;

    if (a->len < b->len) {
        const bf_t *tmp = a;
        a = b;
        b = tmp;
    }
    r_sign = a->sign ^ b->sign;
    /* here b->len <= a->len */
    if (b->len == 0) {
        if (a->expn == BF_EXP_NAN || b->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            ret = 0;
        } else if (a->expn == BF_EXP_INF || b->expn == BF_EXP_INF) {
            if ((a->expn == BF_EXP_INF && b->expn == BF_EXP_ZERO) ||
                (a->expn == BF_EXP_ZERO && b->expn == BF_EXP_INF)) {
                bf_set_nan(r);
                ret = BF_ST_INVALID_OP;
            } else {
                bf_set_inf(r, r_sign);
                ret = 0;
            }
        } else {
            bf_set_zero(r, r_sign);
            ret = 0;
        }
    } else {
        bf_t tmp, *r1 = NULL;
        limb_t a_len, b_len, precl;
        limb_t *a_tab, *b_tab;

        a_len = a->len;
        b_len = b->len;

        if ((flags & BF_RND_MASK) == BF_RNDF) {
            /* faithful rounding does not require using the full inputs */
            precl = (prec + 2 + LIMB_BITS - 1) / LIMB_BITS;
            a_len = bf_min(a_len, precl);
            b_len = bf_min(b_len, precl);
        }
        a_tab = a->tab + a->len - a_len;
        b_tab = b->tab + b->len - b_len;

#ifdef USE_FFT_MUL
        if (b_len >= FFT_MUL_THRESHOLD) {
            int mul_flags = 0;
            if (r == a)
                mul_flags |= FFT_MUL_R_OVERLAP_A;
            if (r == b)
                mul_flags |= FFT_MUL_R_OVERLAP_B;
            if (fft_mul(r->ctx, r, a_tab, a_len, b_tab, b_len, mul_flags))
                goto fail;
        } else
#endif
        {
            if (r == a || r == b) {
                bf_init(r->ctx, &tmp);
                r1 = r;
                r = &tmp;
            }
            if (bf_resize(r, a_len + b_len)) {
            fail:
                bf_set_nan(r);
                ret = BF_ST_MEM_ERROR;
                goto done;
            }
            mp_mul_basecase(r->tab, a_tab, a_len, b_tab, b_len);
        }
        r->sign = r_sign;
        r->expn = a->expn + b->expn;
        ret = bf_normalize_and_round(r, prec, flags);
    done:
        if (r == &tmp)
            bf_move(r1, &tmp);
    }
    return ret;
}

/* multiply 'r' by 2^e */
int bf_mul_2exp(bf_t *r, slimb_t e, limb_t prec, bf_flags_t flags)
{
    slimb_t e_max;
    if (r->len == 0)
        return 0;
    e_max = ((limb_t)1 << BF_EXT_EXP_BITS_MAX) - 1;
    e = bf_max(e, -e_max);
    e = bf_min(e, e_max);
    r->expn += e;
    return __bf_round(r, prec, flags, r->len, 0);
}

/* Return e such as a=m*2^e with m odd integer. return 0 if a is zero,
   Infinite or Nan. */
slimb_t bf_get_exp_min(const bf_t *a)
{
    slimb_t i;
    limb_t v;
    int k;

    for(i = 0; i < a->len; i++) {
        v = a->tab[i];
        if (v != 0) {
            k = ctz(v);
            return a->expn - (a->len - i) * LIMB_BITS + k;
        }
    }
    return 0;
}

/* a and b must be finite numbers with a >= 0 and b > 0. 'q' is the
   integer defined as floor(a/b) and r = a - q * b. */
static void bf_tdivremu(bf_t *q, bf_t *r,
                        const bf_t *a, const bf_t *b)
{
    if (bf_cmpu(a, b) < 0) {
        bf_set_ui(q, 0);
        bf_set(r, a);
    } else {
        bf_div(q, a, b, bf_max(a->expn - b->expn + 1, 2), BF_RNDZ);
        bf_rint(q, BF_RNDZ);
        bf_mul(r, q, b, BF_PREC_INF, BF_RNDZ);
        bf_sub(r, a, r, BF_PREC_INF, BF_RNDZ);
    }
}

static int __bf_div(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
                    bf_flags_t flags)
{
    bf_context_t *s = r->ctx;
    int ret, r_sign;
    limb_t n, nb, precl;

    r_sign = a->sign ^ b->sign;
    if (a->expn >= BF_EXP_INF || b->expn >= BF_EXP_INF) {
        if (a->expn == BF_EXP_NAN || b->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF && b->expn == BF_EXP_INF) {
            bf_set_nan(r);
            return BF_ST_INVALID_OP;
        } else if (a->expn == BF_EXP_INF) {
            bf_set_inf(r, r_sign);
            return 0;
        } else {
            bf_set_zero(r, r_sign);
            return 0;
        }
    } else if (a->expn == BF_EXP_ZERO) {
        if (b->expn == BF_EXP_ZERO) {
            bf_set_nan(r);
            return BF_ST_INVALID_OP;
        } else {
            bf_set_zero(r, r_sign);
            return 0;
        }
    } else if (b->expn == BF_EXP_ZERO) {
        bf_set_inf(r, r_sign);
        return BF_ST_DIVIDE_ZERO;
    }

    /* number of limbs of the quotient (2 extra bits for rounding) */
    precl = (prec + 2 + LIMB_BITS - 1) / LIMB_BITS;
    nb = b->len;
    n = bf_max(a->len, precl);

    {
        limb_t *taba, na;
        slimb_t d;

        na = n + nb;

#if LIMB_LOG2_BITS == 6
        if (na >= (SIZE_MAX / sizeof(limb_t)) - 1) {
            return BF_ST_MEM_ERROR;  /* Return memory error status */
        }
#endif

        taba = bf_malloc(s, (na + 1) * sizeof(limb_t));
        if (!taba)
            goto fail;
        d = na - a->len;
        memset(taba, 0, d * sizeof(limb_t));
        memcpy(taba + d, a->tab, a->len * sizeof(limb_t));
        if (bf_resize(r, n + 1))
            goto fail1;
        if (mp_divnorm(s, r->tab, taba, na, b->tab, nb)) {
        fail1:
            bf_free(s, taba);
            goto fail;
        }
        /* see if non zero remainder */
        if (mp_scan_nz(taba, nb))
            r->tab[0] |= 1;
        bf_free(r->ctx, taba);
        r->expn = a->expn - b->expn + LIMB_BITS;
        r->sign = r_sign;
        ret = bf_normalize_and_round(r, prec, flags);
    }
    return ret;
 fail:
    bf_set_nan(r);
    return BF_ST_MEM_ERROR;
}

/* division and remainder.

   rnd_mode is the rounding mode for the quotient. The additional
   rounding mode BF_RND_EUCLIDIAN is supported.

   'q' is an integer. 'r' is rounded with prec and flags (prec can be
   BF_PREC_INF).
*/
int bf_divrem(bf_t *q, bf_t *r, const bf_t *a, const bf_t *b,
              limb_t prec, bf_flags_t flags, int rnd_mode)
{
    bf_t a1_s, *a1 = &a1_s;
    bf_t b1_s, *b1 = &b1_s;
    int q_sign, ret;
    BOOL is_ceil, is_rndn;

    assert(q != a && q != b);
    assert(r != a && r != b);
    assert(q != r);

    if (a->len == 0 || b->len == 0) {
        bf_set_zero(q, 0);
        if (a->expn == BF_EXP_NAN || b->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF || b->expn == BF_EXP_ZERO) {
            bf_set_nan(r);
            return BF_ST_INVALID_OP;
        } else {
            bf_set(r, a);
            return bf_round(r, prec, flags);
        }
    }

    q_sign = a->sign ^ b->sign;
    is_rndn = (rnd_mode == BF_RNDN || rnd_mode == BF_RNDNA);
    switch(rnd_mode) {
    default:
    case BF_RNDZ:
    case BF_RNDN:
    case BF_RNDNA:
        is_ceil = FALSE;
        break;
    case BF_RNDD:
        is_ceil = q_sign;
        break;
    case BF_RNDU:
        is_ceil = q_sign ^ 1;
        break;
    case BF_RNDA:
        is_ceil = TRUE;
        break;
    case BF_DIVREM_EUCLIDIAN:
        is_ceil = a->sign;
        break;
    }

    a1->expn = a->expn;
    a1->tab = a->tab;
    a1->len = a->len;
    a1->sign = 0;

    b1->expn = b->expn;
    b1->tab = b->tab;
    b1->len = b->len;
    b1->sign = 0;

    /* XXX: could improve to avoid having a large 'q' */
    bf_tdivremu(q, r, a1, b1);
    if (bf_is_nan(q) || bf_is_nan(r))
        goto fail;

    if (r->len != 0) {
        if (is_rndn) {
            int res;
            b1->expn--;
            res = bf_cmpu(r, b1);
            b1->expn++;
            if (res > 0 ||
                (res == 0 &&
                 (rnd_mode == BF_RNDNA ||
                  get_bit(q->tab, q->len, q->len * LIMB_BITS - q->expn)))) {
                goto do_sub_r;
            }
        } else if (is_ceil) {
        do_sub_r:
            ret = bf_add_si(q, q, 1, BF_PREC_INF, BF_RNDZ);
            ret |= bf_sub(r, r, b1, BF_PREC_INF, BF_RNDZ);
            if (ret & BF_ST_MEM_ERROR)
                goto fail;
        }
    }

    r->sign ^= a->sign;
    q->sign = q_sign;
    return bf_round(r, prec, flags);
 fail:
    bf_set_nan(q);
    bf_set_nan(r);
    return BF_ST_MEM_ERROR;
}

int bf_rem(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
           bf_flags_t flags, int rnd_mode)
{
    bf_t q_s, *q = &q_s;
    int ret;

    bf_init(r->ctx, q);
    ret = bf_divrem(q, r, a, b, prec, flags, rnd_mode);
    bf_delete(q);
    return ret;
}

static inline int bf_get_limb(slimb_t *pres, const bf_t *a, int flags)
{
#if LIMB_BITS == 32
    return bf_get_int32(pres, a, flags);
#else
    return bf_get_int64(pres, a, flags);
#endif
}

int bf_remquo(slimb_t *pq, bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
              bf_flags_t flags, int rnd_mode)
{
    bf_t q_s, *q = &q_s;
    int ret;

    bf_init(r->ctx, q);
    ret = bf_divrem(q, r, a, b, prec, flags, rnd_mode);
    bf_get_limb(pq, q, BF_GET_INT_MOD);
    bf_delete(q);
    return ret;
}

static __maybe_unused inline limb_t mul_mod(limb_t a, limb_t b, limb_t m)
{
    dlimb_t t;
    t = (dlimb_t)a * (dlimb_t)b;
    return t % m;
}

#if defined(USE_MUL_CHECK)
static limb_t mp_mod1(const limb_t *tab, limb_t n, limb_t m, limb_t r)
{
    slimb_t i;
    dlimb_t t;

    for(i = n - 1; i >= 0; i--) {
        t = ((dlimb_t)r << LIMB_BITS) | tab[i];
        r = t % m;
    }
    return r;
}
#endif

static const uint16_t sqrt_table[192] = {
128,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,144,145,146,147,148,149,150,150,151,152,153,154,155,155,156,157,158,159,160,160,161,162,163,163,164,165,166,167,167,168,169,170,170,171,172,173,173,174,175,176,176,177,178,178,179,180,181,181,182,183,183,184,185,185,186,187,187,188,189,189,190,191,192,192,193,193,194,195,195,196,197,197,198,199,199,200,201,201,202,203,203,204,204,205,206,206,207,208,208,209,209,210,211,211,212,212,213,214,214,215,215,216,217,217,218,218,219,219,220,221,221,222,222,223,224,224,225,225,226,226,227,227,228,229,229,230,230,231,231,232,232,233,234,234,235,235,236,236,237,237,238,238,239,240,240,241,241,242,242,243,243,244,244,245,245,246,246,247,247,248,248,249,249,250,250,251,251,252,252,253,253,254,254,255,
};

/* a >= 2^(LIMB_BITS - 2).  Return (s, r) with s=floor(sqrt(a)) and
   r=a-s^2. 0 <= r <= 2 * s */
static limb_t mp_sqrtrem1(limb_t *pr, limb_t a)
{
    limb_t s1, r1, s, r, q, u, num;

    /* use a table for the 16 -> 8 bit sqrt */
    s1 = sqrt_table[(a >> (LIMB_BITS - 8)) - 64];
    r1 = (a >> (LIMB_BITS - 16)) - s1 * s1;
    if (r1 > 2 * s1) {
        r1 -= 2 * s1 + 1;
        s1++;
    }

    /* one iteration to get a 32 -> 16 bit sqrt */
    num = (r1 << 8) | ((a >> (LIMB_BITS - 32 + 8)) & 0xff);
    q = num / (2 * s1); /* q <= 2^8 */
    u = num % (2 * s1);
    s = (s1 << 8) + q;
    r = (u << 8) | ((a >> (LIMB_BITS - 32)) & 0xff);
    r -= q * q;
    if ((slimb_t)r < 0) {
        s--;
        r += 2 * s + 1;
    }

#if LIMB_BITS == 64
    s1 = s;
    r1 = r;
    /* one more iteration for 64 -> 32 bit sqrt */
    num = (r1 << 16) | ((a >> (LIMB_BITS - 64 + 16)) & 0xffff);
    q = num / (2 * s1); /* q <= 2^16 */
    u = num % (2 * s1);
    s = (s1 << 16) + q;
    r = (u << 16) | ((a >> (LIMB_BITS - 64)) & 0xffff);
    r -= q * q;
    if ((slimb_t)r < 0) {
        s--;
        r += 2 * s + 1;
    }
#endif
    *pr = r;
    return s;
}

/* return floor(sqrt(a)) */
limb_t bf_isqrt(limb_t a)
{
    limb_t s, r;
    int k;

    if (a == 0)
        return 0;
    k = clz(a) & ~1;
    s = mp_sqrtrem1(&r, a << k);
    s >>= (k >> 1);
    return s;
}

static limb_t mp_sqrtrem2(limb_t *tabs, limb_t *taba)
{
    limb_t s1, r1, s, q, u, a0, a1;
    dlimb_t r, num;
    int l;

    a0 = taba[0];
    a1 = taba[1];
    s1 = mp_sqrtrem1(&r1, a1);
    l = LIMB_BITS / 2;
    num = ((dlimb_t)r1 << l) | (a0 >> l);
    q = num / (2 * s1);
    u = num % (2 * s1);
    s = (s1 << l) + q;
    r = ((dlimb_t)u << l) | (a0 & (((limb_t)1 << l) - 1));
    if (unlikely((q >> l) != 0))
        r -= (dlimb_t)1 << LIMB_BITS; /* special case when q=2^l */
    else
        r -= q * q;
    if ((slimb_t)(r >> LIMB_BITS) < 0) {
        s--;
        r += 2 * (dlimb_t)s + 1;
    }
    tabs[0] = s;
    taba[0] = r;
    return r >> LIMB_BITS;
}

//#define DEBUG_SQRTREM

/* tmp_buf must contain (n / 2 + 1 limbs). *prh contains the highest
   limb of the remainder. */
static int mp_sqrtrem_rec(bf_context_t *s, limb_t *tabs, limb_t *taba, limb_t n,
                          limb_t *tmp_buf, limb_t *prh)
{
    limb_t l, h, rh, ql, qh, c, i;

    if (n == 1) {
        *prh = mp_sqrtrem2(tabs, taba);
        return 0;
    }
#ifdef DEBUG_SQRTREM
    mp_print_str("a", taba, 2 * n);
#endif
    l = n / 2;
    h = n - l;
    if (mp_sqrtrem_rec(s, tabs + l, taba + 2 * l, h, tmp_buf, &qh))
        return -1;
#ifdef DEBUG_SQRTREM
    mp_print_str("s1", tabs + l, h);
    mp_print_str_h("r1", taba + 2 * l, h, qh);
    mp_print_str_h("r2", taba + l, n, qh);
#endif

    /* the remainder is in taba + 2 * l. Its high bit is in qh */
    if (qh) {
        mp_sub(taba + 2 * l, taba + 2 * l, tabs + l, h, 0);
    }
    /* instead of dividing by 2*s, divide by s (which is normalized)
       and update q and r */
    if (mp_divnorm(s, tmp_buf, taba + l, n, tabs + l, h))
        return -1;
    qh += tmp_buf[l];
    for(i = 0; i < l; i++)
        tabs[i] = tmp_buf[i];
    ql = mp_shr(tabs, tabs, l, 1, qh & 1);
    qh = qh >> 1; /* 0 or 1 */
    if (ql)
        rh = mp_add(taba + l, taba + l, tabs + l, h, 0);
    else
        rh = 0;
#ifdef DEBUG_SQRTREM
    mp_print_str_h("q", tabs, l, qh);
    mp_print_str_h("u", taba + l, h, rh);
#endif

    mp_add_ui(tabs + l, qh, h);
#ifdef DEBUG_SQRTREM
    mp_print_str_h("s2", tabs, n, sh);
#endif

    /* q = qh, tabs[l - 1 ... 0], r = taba[n - 1 ... l] */
    /* subtract q^2. if qh = 1 then q = B^l, so we can take shortcuts */
    if (qh) {
        c = qh;
    } else {
        if (mp_mul(s, taba + n, tabs, l, tabs, l))
            return -1;
        c = mp_sub(taba, taba, taba + n, 2 * l, 0);
    }
    rh -= mp_sub_ui(taba + 2 * l, c, n - 2 * l);
    if ((slimb_t)rh < 0) {
        mp_sub_ui(tabs, 1, n);
        rh += mp_add_mul1(taba, tabs, n, 2);
        rh += mp_add_ui(taba, 1, n);
    }
    *prh = rh;
    return 0;
}

/* 'taba' has 2*n limbs with n >= 1 and taba[2*n-1] >= 2 ^ (LIMB_BITS
   - 2). Return (s, r) with s=floor(sqrt(a)) and r=a-s^2. 0 <= r <= 2
   * s. tabs has n limbs. r is returned in the lower n limbs of
   taba. Its r[n] is the returned value of the function. */
/* Algorithm from the article "Karatsuba Square Root" by Paul Zimmermann and
   inspirated from its GMP implementation */
int mp_sqrtrem(bf_context_t *s, limb_t *tabs, limb_t *taba, limb_t n)
{
    limb_t tmp_buf1[8];
    limb_t *tmp_buf;
    mp_size_t n2;
    int ret;
    n2 = n / 2 + 1;
    if (n2 <= countof(tmp_buf1)) {
        tmp_buf = tmp_buf1;
    } else {
        tmp_buf = bf_malloc(s, sizeof(limb_t) * n2);
        if (!tmp_buf)
            return -1;
    }
    ret = mp_sqrtrem_rec(s, tabs, taba, n, tmp_buf, taba + n);
    if (tmp_buf != tmp_buf1)
        bf_free(s, tmp_buf);
    return ret;
}

/* Integer square root with remainder. 'a' must be an integer. r =
   floor(sqrt(a)) and rem = a - r^2.  BF_ST_INEXACT is set if the result
   is inexact. 'rem' can be NULL if the remainder is not needed. */
int bf_sqrtrem(bf_t *r, bf_t *rem1, const bf_t *a)
{
    int ret;

    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
        } else if (a->expn == BF_EXP_INF && a->sign) {
            goto invalid_op;
        } else {
            bf_set(r, a);
        }
        if (rem1)
            bf_set_ui(rem1, 0);
        ret = 0;
    } else if (a->sign) {
 invalid_op:
        bf_set_nan(r);
        if (rem1)
            bf_set_ui(rem1, 0);
        ret = BF_ST_INVALID_OP;
    } else {
        bf_t rem_s, *rem;

        bf_sqrt(r, a, (a->expn + 1) / 2, BF_RNDZ);
        bf_rint(r, BF_RNDZ);
        /* see if the result is exact by computing the remainder */
        if (rem1) {
            rem = rem1;
        } else {
            rem = &rem_s;
            bf_init(r->ctx, rem);
        }
        /* XXX: could avoid recomputing the remainder */
        bf_mul(rem, r, r, BF_PREC_INF, BF_RNDZ);
        bf_neg(rem);
        bf_add(rem, rem, a, BF_PREC_INF, BF_RNDZ);
        if (bf_is_nan(rem)) {
            ret = BF_ST_MEM_ERROR;
            goto done;
        }
        if (rem->len != 0) {
            ret = BF_ST_INEXACT;
        } else {
            ret = 0;
        }
    done:
        if (!rem1)
            bf_delete(rem);
    }
    return ret;
}

int bf_sqrt(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = a->ctx;
    int ret;

    assert(r != a);

    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
        } else if (a->expn == BF_EXP_INF && a->sign) {
            goto invalid_op;
        } else {
            bf_set(r, a);
        }
        ret = 0;
    } else if (a->sign) {
 invalid_op:
        bf_set_nan(r);
        ret = BF_ST_INVALID_OP;
    } else {
        limb_t *a1;
        slimb_t n, n1;
        limb_t res;

        /* convert the mantissa to an integer with at least 2 *
           prec + 4 bits */
        n = (2 * (prec + 2) + 2 * LIMB_BITS - 1) / (2 * LIMB_BITS);
        if (bf_resize(r, n))
            goto fail;
        a1 = bf_malloc(s, sizeof(limb_t) * 2 * n);
        if (!a1)
            goto fail;
        n1 = bf_min(2 * n, a->len);
        memset(a1, 0, (2 * n - n1) * sizeof(limb_t));
        memcpy(a1 + 2 * n - n1, a->tab + a->len - n1, n1 * sizeof(limb_t));
        if (a->expn & 1) {
            res = mp_shr(a1, a1, 2 * n, 1, 0);
        } else {
            res = 0;
        }
        if (mp_sqrtrem(s, r->tab, a1, n)) {
            bf_free(s, a1);
            goto fail;
        }
        if (!res) {
            res = mp_scan_nz(a1, n + 1);
        }
        bf_free(s, a1);
        if (!res) {
            res = mp_scan_nz(a->tab, a->len - n1);
        }
        if (res != 0)
            r->tab[0] |= 1;
        r->sign = 0;
        r->expn = (a->expn + 1) >> 1;
        ret = bf_round(r, prec, flags);
    }
    return ret;
 fail:
    bf_set_nan(r);
    return BF_ST_MEM_ERROR;
}

static no_inline int bf_op2(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
                            bf_flags_t flags, bf_op2_func_t *func)
{
    bf_t tmp;
    int ret;

    if (r == a || r == b) {
        bf_init(r->ctx, &tmp);
        ret = func(&tmp, a, b, prec, flags);
        bf_move(r, &tmp);
    } else {
        ret = func(r, a, b, prec, flags);
    }
    return ret;
}

int bf_add(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
            bf_flags_t flags)
{
    return bf_op2(r, a, b, prec, flags, __bf_add);
}

int bf_sub(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
            bf_flags_t flags)
{
    return bf_op2(r, a, b, prec, flags, __bf_sub);
}

int bf_div(bf_t *r, const bf_t *a, const bf_t *b, limb_t prec,
           bf_flags_t flags)
{
    return bf_op2(r, a, b, prec, flags, __bf_div);
}

int bf_mul_ui(bf_t *r, const bf_t *a, uint64_t b1, limb_t prec,
               bf_flags_t flags)
{
    bf_t b;
    int ret;
    bf_init(r->ctx, &b);
    ret = bf_set_ui(&b, b1);
    ret |= bf_mul(r, a, &b, prec, flags);
    bf_delete(&b);
    return ret;
}

int bf_mul_si(bf_t *r, const bf_t *a, int64_t b1, limb_t prec,
               bf_flags_t flags)
{
    bf_t b;
    int ret;
    bf_init(r->ctx, &b);
    ret = bf_set_si(&b, b1);
    ret |= bf_mul(r, a, &b, prec, flags);
    bf_delete(&b);
    return ret;
}

int bf_add_si(bf_t *r, const bf_t *a, int64_t b1, limb_t prec,
              bf_flags_t flags)
{
    bf_t b;
    int ret;

    bf_init(r->ctx, &b);
    ret = bf_set_si(&b, b1);
    ret |= bf_add(r, a, &b, prec, flags);
    bf_delete(&b);
    return ret;
}

static int bf_pow_ui(bf_t *r, const bf_t *a, limb_t b, limb_t prec,
                     bf_flags_t flags)
{
    int ret, n_bits, i;

    assert(r != a);
    if (b == 0)
        return bf_set_ui(r, 1);
    ret = bf_set(r, a);
    n_bits = LIMB_BITS - clz(b);
    for(i = n_bits - 2; i >= 0; i--) {
        ret |= bf_mul(r, r, r, prec, flags);
        if ((b >> i) & 1)
            ret |= bf_mul(r, r, a, prec, flags);
    }
    return ret;
}

static int bf_pow_ui_ui(bf_t *r, limb_t a1, limb_t b,
                        limb_t prec, bf_flags_t flags)
{
    bf_t a;
    int ret;

    if (a1 == 10 && b <= LIMB_DIGITS) {
        /* use precomputed powers. We do not round at this point
           because we expect the caller to do it */
        ret = bf_set_ui(r, mp_pow_dec[b]);
    } else {
        bf_init(r->ctx, &a);
        ret = bf_set_ui(&a, a1);
        ret |= bf_pow_ui(r, &a, b, prec, flags);
        bf_delete(&a);
    }
    return ret;
}

/* convert to integer (infinite precision) */
int bf_rint(bf_t *r, int rnd_mode)
{
    return bf_round(r, 0, rnd_mode | BF_FLAG_RADPNT_PREC);
}

/* logical operations */
#define BF_LOGIC_OR  0
#define BF_LOGIC_XOR 1
#define BF_LOGIC_AND 2

static inline limb_t bf_logic_op1(limb_t a, limb_t b, int op)
{
    switch(op) {
    case BF_LOGIC_OR:
        return a | b;
    case BF_LOGIC_XOR:
        return a ^ b;
    default:
    case BF_LOGIC_AND:
        return a & b;
    }
}

static int bf_logic_op(bf_t *r, const bf_t *a1, const bf_t *b1, int op)
{
    bf_t b1_s, a1_s, *a, *b;
    limb_t a_sign, b_sign, r_sign;
    slimb_t l, i, a_bit_offset, b_bit_offset;
    limb_t v1, v2, v1_mask, v2_mask, r_mask;
    int ret;

    assert(r != a1 && r != b1);

    if (a1->expn <= 0)
        a_sign = 0; /* minus zero is considered as positive */
    else
        a_sign = a1->sign;

    if (b1->expn <= 0)
        b_sign = 0; /* minus zero is considered as positive */
    else
        b_sign = b1->sign;

    if (a_sign) {
        a = &a1_s;
        bf_init(r->ctx, a);
        if (bf_add_si(a, a1, 1, BF_PREC_INF, BF_RNDZ)) {
            b = NULL;
            goto fail;
        }
    } else {
        a = (bf_t *)a1;
    }

    if (b_sign) {
        b = &b1_s;
        bf_init(r->ctx, b);
        if (bf_add_si(b, b1, 1, BF_PREC_INF, BF_RNDZ))
            goto fail;
    } else {
        b = (bf_t *)b1;
    }

    r_sign = bf_logic_op1(a_sign, b_sign, op);
    if (op == BF_LOGIC_AND && r_sign == 0) {
        /* no need to compute extra zeros for and */
        if (a_sign == 0 && b_sign == 0)
            l = bf_min(a->expn, b->expn);
        else if (a_sign == 0)
            l = a->expn;
        else
            l = b->expn;
    } else {
        l = bf_max(a->expn, b->expn);
    }
    /* Note: a or b can be zero */
    l = (bf_max(l, 1) + LIMB_BITS - 1) / LIMB_BITS;
    if (bf_resize(r, l))
        goto fail;
    a_bit_offset = a->len * LIMB_BITS - a->expn;
    b_bit_offset = b->len * LIMB_BITS - b->expn;
    v1_mask = -a_sign;
    v2_mask = -b_sign;
    r_mask = -r_sign;
    for(i = 0; i < l; i++) {
        v1 = get_bits(a->tab, a->len, a_bit_offset + i * LIMB_BITS) ^ v1_mask;
        v2 = get_bits(b->tab, b->len, b_bit_offset + i * LIMB_BITS) ^ v2_mask;
        r->tab[i] = bf_logic_op1(v1, v2, op) ^ r_mask;
    }
    r->expn = l * LIMB_BITS;
    r->sign = r_sign;
    bf_normalize_and_round(r, BF_PREC_INF, BF_RNDZ); /* cannot fail */
    if (r_sign) {
        if (bf_add_si(r, r, -1, BF_PREC_INF, BF_RNDZ))
            goto fail;
    }
    ret = 0;
 done:
    if (a == &a1_s)
        bf_delete(a);
    if (b == &b1_s)
        bf_delete(b);
    return ret;
 fail:
    bf_set_nan(r);
    ret = BF_ST_MEM_ERROR;
    goto done;
}

/* 'a' and 'b' must be integers. Return 0 or BF_ST_MEM_ERROR. */
int bf_logic_or(bf_t *r, const bf_t *a, const bf_t *b)
{
    return bf_logic_op(r, a, b, BF_LOGIC_OR);
}

/* 'a' and 'b' must be integers. Return 0 or BF_ST_MEM_ERROR. */
int bf_logic_xor(bf_t *r, const bf_t *a, const bf_t *b)
{
    return bf_logic_op(r, a, b, BF_LOGIC_XOR);
}

/* 'a' and 'b' must be integers. Return 0 or BF_ST_MEM_ERROR. */
int bf_logic_and(bf_t *r, const bf_t *a, const bf_t *b)
{
    return bf_logic_op(r, a, b, BF_LOGIC_AND);
}

/* conversion between fixed size types */

typedef union {
    double d;
    uint64_t u;
} Float64Union;

int bf_get_float64(const bf_t *a, double *pres, bf_rnd_t rnd_mode)
{
    Float64Union u;
    int e, ret;
    uint64_t m;

    ret = 0;
    if (a->expn == BF_EXP_NAN) {
        u.u = 0x7ff8000000000000; /* quiet nan */
    } else {
        bf_t b_s, *b = &b_s;

        bf_init(a->ctx, b);
        bf_set(b, a);
        if (bf_is_finite(b)) {
            ret = bf_round(b, 53, rnd_mode | BF_FLAG_SUBNORMAL | bf_set_exp_bits(11));
        }
        if (b->expn == BF_EXP_INF) {
            e = (1 << 11) - 1;
            m = 0;
        } else if (b->expn == BF_EXP_ZERO) {
            e = 0;
            m = 0;
        } else {
            e = b->expn + 1023 - 1;
#if LIMB_BITS == 32
            if (b->len == 2) {
                m = ((uint64_t)b->tab[1] << 32) | b->tab[0];
            } else {
                m = ((uint64_t)b->tab[0] << 32);
            }
#else
            m = b->tab[0];
#endif
            if (e <= 0) {
                /* subnormal */
                m = m >> (12 - e);
                e = 0;
            } else {
                m = (m << 1) >> 12;
            }
        }
        u.u = m | ((uint64_t)e << 52) | ((uint64_t)b->sign << 63);
        bf_delete(b);
    }
    *pres = u.d;
    return ret;
}

int bf_set_float64(bf_t *a, double d)
{
    Float64Union u;
    uint64_t m;
    int shift, e, sgn;

    u.d = d;
    sgn = u.u >> 63;
    e = (u.u >> 52) & ((1 << 11) - 1);
    m = u.u & (((uint64_t)1 << 52) - 1);
    if (e == ((1 << 11) - 1)) {
        if (m != 0) {
            bf_set_nan(a);
        } else {
            bf_set_inf(a, sgn);
        }
    } else if (e == 0) {
        if (m == 0) {
            bf_set_zero(a, sgn);
        } else {
            /* subnormal number */
            m <<= 12;
            shift = clz64(m);
            m <<= shift;
            e = -shift;
            goto norm;
        }
    } else {
        m = (m << 11) | ((uint64_t)1 << 63);
    norm:
        a->expn = e - 1023 + 1;
#if LIMB_BITS == 32
        if (bf_resize(a, 2))
            goto fail;
        a->tab[0] = m;
        a->tab[1] = m >> 32;
#else
        if (bf_resize(a, 1))
            goto fail;
        a->tab[0] = m;
#endif
        a->sign = sgn;
    }
    return 0;
fail:
    bf_set_nan(a);
    return BF_ST_MEM_ERROR;
}

/* The rounding mode is always BF_RNDZ. Return BF_ST_INVALID_OP if there
   is an overflow and 0 otherwise. */
int bf_get_int32(int *pres, const bf_t *a, int flags)
{
    uint32_t v;
    int ret;
    if (a->expn >= BF_EXP_INF) {
        ret = BF_ST_INVALID_OP;
        if (flags & BF_GET_INT_MOD) {
            v = 0;
        } else if (a->expn == BF_EXP_INF) {
            v = (uint32_t)INT32_MAX + a->sign;
        } else {
            v = INT32_MAX;
        }
    } else if (a->expn <= 0) {
        v = 0;
        ret = 0;
    } else if (a->expn <= 31) {
        v = a->tab[a->len - 1] >> (LIMB_BITS - a->expn);
        if (a->sign)
            v = -v;
        ret = 0;
    } else if (!(flags & BF_GET_INT_MOD)) {
        ret = BF_ST_INVALID_OP;
        if (a->sign) {
            v = (uint32_t)INT32_MAX + 1;
            if (a->expn == 32 &&
                (a->tab[a->len - 1] >> (LIMB_BITS - 32)) == v) {
                ret = 0;
            }
        } else {
            v = INT32_MAX;
        }
    } else {
        v = get_bits(a->tab, a->len, a->len * LIMB_BITS - a->expn);
        if (a->sign)
            v = -v;
        ret = 0;
    }
    *pres = v;
    return ret;
}

/* The rounding mode is always BF_RNDZ. Return BF_ST_INVALID_OP if there
   is an overflow and 0 otherwise. */
int bf_get_int64(int64_t *pres, const bf_t *a, int flags)
{
    uint64_t v;
    int ret;
    if (a->expn >= BF_EXP_INF) {
        ret = BF_ST_INVALID_OP;
        if (flags & BF_GET_INT_MOD) {
            v = 0;
        } else if (a->expn == BF_EXP_INF) {
            v = (uint64_t)INT64_MAX + a->sign;
        } else {
            v = INT64_MAX;
        }
    } else if (a->expn <= 0) {
        v = 0;
        ret = 0;
    } else if (a->expn <= 63) {
#if LIMB_BITS == 32
        if (a->expn <= 32)
            v = a->tab[a->len - 1] >> (LIMB_BITS - a->expn);
        else
            v = (((uint64_t)a->tab[a->len - 1] << 32) |
                 get_limbz(a, a->len - 2)) >> (64 - a->expn);
#else
        v = a->tab[a->len - 1] >> (LIMB_BITS - a->expn);
#endif
        if (a->sign)
            v = -v;
        ret = 0;
    } else if (!(flags & BF_GET_INT_MOD)) {
        ret = BF_ST_INVALID_OP;
        if (a->sign) {
            uint64_t v1;
            v = (uint64_t)INT64_MAX + 1;
            if (a->expn == 64) {
                v1 = a->tab[a->len - 1];
#if LIMB_BITS == 32
                v1 = (v1 << 32) | get_limbz(a, a->len - 2);
#endif
                if (v1 == v)
                    ret = 0;
            }
        } else {
            v = INT64_MAX;
        }
    } else {
        slimb_t bit_pos = a->len * LIMB_BITS - a->expn;
        v = get_bits(a->tab, a->len, bit_pos);
#if LIMB_BITS == 32
        v |= (uint64_t)get_bits(a->tab, a->len, bit_pos + 32) << 32;
#endif
        if (a->sign)
            v = -v;
        ret = 0;
    }
    *pres = v;
    return ret;
}

/* The rounding mode is always BF_RNDZ. Return BF_ST_INVALID_OP if there
   is an overflow and 0 otherwise. */
int bf_get_uint64(uint64_t *pres, const bf_t *a)
{
    uint64_t v;
    int ret;
    if (a->expn == BF_EXP_NAN) {
        goto overflow;
    } else if (a->expn <= 0) {
        v = 0;
        ret = 0;
    } else if (a->sign) {
        v = 0;
        ret = BF_ST_INVALID_OP;
    } else if (a->expn <= 64) {
#if LIMB_BITS == 32
        if (a->expn <= 32)
            v = a->tab[a->len - 1] >> (LIMB_BITS - a->expn);
        else
            v = (((uint64_t)a->tab[a->len - 1] << 32) |
                 get_limbz(a, a->len - 2)) >> (64 - a->expn);
#else
        v = a->tab[a->len - 1] >> (LIMB_BITS - a->expn);
#endif
        ret = 0;
    } else {
    overflow:
        v = UINT64_MAX;
        ret = BF_ST_INVALID_OP;
    }
    *pres = v;
    return ret;
}

/* base conversion from radix */

static const uint8_t digits_per_limb_table[BF_RADIX_MAX - 1] = {
#if LIMB_BITS == 32
32,20,16,13,12,11,10,10, 9, 9, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#else
64,40,32,27,24,22,21,20,19,18,17,17,16,16,16,15,15,15,14,14,14,14,13,13,13,13,13,13,13,12,12,12,12,12,12,
#endif
};

static limb_t get_limb_radix(int radix)
{
    int i, k;
    limb_t radixl;

    k = digits_per_limb_table[radix - 2];
    radixl = radix;
    for(i = 1; i < k; i++)
        radixl *= radix;
    return radixl;
}

/* return != 0 if error */
static int bf_integer_from_radix_rec(bf_t *r, const limb_t *tab,
                                     limb_t n, int level, limb_t n0,
                                     limb_t radix, bf_t *pow_tab)
{
    int ret;
    if (n == 1) {
        ret = bf_set_ui(r, tab[0]);
    } else {
        bf_t T_s, *T = &T_s, *B;
        limb_t n1, n2;

        n2 = (((n0 * 2) >> (level + 1)) + 1) / 2;
        n1 = n - n2;
        //        printf("level=%d n0=%ld n1=%ld n2=%ld\n", level, n0, n1, n2);
        B = &pow_tab[level];
        if (B->len == 0) {
            ret = bf_pow_ui_ui(B, radix, n2, BF_PREC_INF, BF_RNDZ);
            if (ret)
                return ret;
        }
        ret = bf_integer_from_radix_rec(r, tab + n2, n1, level + 1, n0,
                                        radix, pow_tab);
        if (ret)
            return ret;
        ret = bf_mul(r, r, B, BF_PREC_INF, BF_RNDZ);
        if (ret)
            return ret;
        bf_init(r->ctx, T);
        ret = bf_integer_from_radix_rec(T, tab, n2, level + 1, n0,
                                        radix, pow_tab);
        if (!ret)
            ret = bf_add(r, r, T, BF_PREC_INF, BF_RNDZ);
        bf_delete(T);
    }
    return ret;
    //    bf_print_str("  r=", r);
}

/* return 0 if OK != 0 if memory error */
static int bf_integer_from_radix(bf_t *r, const limb_t *tab,
                                 limb_t n, limb_t radix)
{
    bf_context_t *s = r->ctx;
    int pow_tab_len, i, ret;
    limb_t radixl;
    bf_t *pow_tab;

    radixl = get_limb_radix(radix);
    pow_tab_len = ceil_log2(n) + 2; /* XXX: check */
    pow_tab = bf_malloc(s, sizeof(pow_tab[0]) * pow_tab_len);
    if (!pow_tab)
        return -1;
    for(i = 0; i < pow_tab_len; i++)
        bf_init(r->ctx, &pow_tab[i]);
    ret = bf_integer_from_radix_rec(r, tab, n, 0, n, radixl, pow_tab);
    for(i = 0; i < pow_tab_len; i++) {
        bf_delete(&pow_tab[i]);
    }
    bf_free(s, pow_tab);
    return ret;
}

/* compute and round T * radix^expn. */
int bf_mul_pow_radix(bf_t *r, const bf_t *T, limb_t radix,
                     slimb_t expn, limb_t prec, bf_flags_t flags)
{
    int ret, expn_sign, overflow;
    slimb_t e, extra_bits, prec1, ziv_extra_bits;
    bf_t B_s, *B = &B_s;

    if (T->len == 0) {
        return bf_set(r, T);
    } else if (expn == 0) {
        ret = bf_set(r, T);
        ret |= bf_round(r, prec, flags);
        return ret;
    }

    e = expn;
    expn_sign = 0;
    if (e < 0) {
        e = -e;
        expn_sign = 1;
    }
    bf_init(r->ctx, B);
    if (prec == BF_PREC_INF) {
        /* infinite precision: only used if the result is known to be exact */
        ret = bf_pow_ui_ui(B, radix, e, BF_PREC_INF, BF_RNDN);
        if (expn_sign) {
            ret |= bf_div(r, T, B, T->len * LIMB_BITS, BF_RNDN);
        } else {
            ret |= bf_mul(r, T, B, BF_PREC_INF, BF_RNDN);
        }
    } else {
        ziv_extra_bits = 16;
        for(;;) {
            prec1 = prec + ziv_extra_bits;
            /* XXX: correct overflow/underflow handling */
            /* XXX: rigorous error analysis needed */
            extra_bits = ceil_log2(e) * 2 + 1;
            ret = bf_pow_ui_ui(B, radix, e, prec1 + extra_bits, BF_RNDN | BF_FLAG_EXT_EXP);
            overflow = !bf_is_finite(B);
            /* XXX: if bf_pow_ui_ui returns an exact result, can stop
               after the next operation */
            if (expn_sign)
                ret |= bf_div(r, T, B, prec1 + extra_bits, BF_RNDN | BF_FLAG_EXT_EXP);
            else
                ret |= bf_mul(r, T, B, prec1 + extra_bits, BF_RNDN | BF_FLAG_EXT_EXP);
            if (ret & BF_ST_MEM_ERROR)
                break;
            if ((ret & BF_ST_INEXACT) &&
                !bf_can_round(r, prec, flags & BF_RND_MASK, prec1) &&
                !overflow) {
                /* and more precision and retry */
                ziv_extra_bits = ziv_extra_bits  + (ziv_extra_bits / 2);
            } else {
                /* XXX: need to use __bf_round() to pass the inexact
                   flag for the subnormal case */
                ret = bf_round(r, prec, flags) | (ret & BF_ST_INEXACT);
                break;
            }
        }
    }
    bf_delete(B);
    return ret;
}

static inline int bf_to_digit(int c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    else if (c >= 'A' && c <= 'Z')
        return c - 'A' + 10;
    else if (c >= 'a' && c <= 'z')
        return c - 'a' + 10;
    else
        return 36;
}

/* add a limb at 'pos' and decrement pos. new space is created if
   needed. Return 0 if OK, -1 if memory error */
static int bf_add_limb(bf_t *a, slimb_t *ppos, limb_t v)
{
    slimb_t pos;
    pos = *ppos;
    if (unlikely(pos < 0)) {
        limb_t new_size, d, *new_tab;
        new_size = bf_max(a->len + 1, a->len * 3 / 2);
        new_tab = bf_realloc(a->ctx, a->tab, sizeof(limb_t) * new_size);
        if (!new_tab)
            return -1;
        a->tab = new_tab;
        d = new_size - a->len;
        memmove(a->tab + d, a->tab, a->len * sizeof(limb_t));
        a->len = new_size;
        pos += d;
    }
    a->tab[pos--] = v;
    *ppos = pos;
    return 0;
}

static int bf_tolower(int c)
{
    if (c >= 'A' && c <= 'Z')
        c = c - 'A' + 'a';
    return c;
}

static int strcasestart(const char *str, const char *val, const char **ptr)
{
    const char *p, *q;
    p = str;
    q = val;
    while (*q != '\0') {
        if (bf_tolower(*p) != *q)
            return 0;
        p++;
        q++;
    }
    if (ptr)
        *ptr = p;
    return 1;
}

static int bf_atof_internal(bf_t *r, slimb_t *pexponent,
                            const char *str, const char **pnext, int radix,
                            limb_t prec, bf_flags_t flags, BOOL is_dec)
{
    const char *p, *p_start;
    int is_neg, radix_bits, exp_is_neg, ret, digits_per_limb, shift;
    limb_t cur_limb;
    slimb_t pos, expn, int_len, digit_count;
    BOOL has_decpt, is_bin_exp;
    bf_t a_s, *a;

    *pexponent = 0;
    p = str;
    if (!(flags & BF_ATOF_NO_NAN_INF) && radix <= 16 &&
        strcasestart(p, "nan", &p)) {
        bf_set_nan(r);
        ret = 0;
        goto done;
    }
    is_neg = 0;

    if (p[0] == '+') {
        p++;
        p_start = p;
    } else if (p[0] == '-') {
        is_neg = 1;
        p++;
        p_start = p;
    } else {
        p_start = p;
    }
    if (p[0] == '0') {
        if ((p[1] == 'x' || p[1] == 'X') &&
            (radix == 0 || radix == 16) &&
            !(flags & BF_ATOF_NO_HEX)) {
            radix = 16;
            p += 2;
        } else if ((p[1] == 'o' || p[1] == 'O') &&
                   radix == 0 && (flags & BF_ATOF_BIN_OCT)) {
            p += 2;
            radix = 8;
        } else if ((p[1] == 'b' || p[1] == 'B') &&
                   radix == 0 && (flags & BF_ATOF_BIN_OCT)) {
            p += 2;
            radix = 2;
        } else {
            goto no_prefix;
        }
        /* there must be a digit after the prefix */
        if (bf_to_digit((uint8_t)*p) >= radix) {
            bf_set_nan(r);
            ret = 0;
            goto done;
        }
    no_prefix: ;
    } else {
        if (!(flags & BF_ATOF_NO_NAN_INF) && radix <= 16 &&
            strcasestart(p, "inf", &p)) {
            bf_set_inf(r, is_neg);
            ret = 0;
            goto done;
        }
    }

    if (radix == 0)
        radix = 10;
    if (is_dec) {
        assert(radix == 10);
        radix_bits = 0;
        a = r;
    } else if ((radix & (radix - 1)) != 0) {
        radix_bits = 0; /* base is not a power of two */
        a = &a_s;
        bf_init(r->ctx, a);
    } else {
        radix_bits = ceil_log2(radix);
        a = r;
    }

    /* skip leading zeros */
    /* XXX: could also skip zeros after the decimal point */
    while (*p == '0')
        p++;

    if (radix_bits) {
        shift = digits_per_limb = LIMB_BITS;
    } else {
        radix_bits = 0;
        shift = digits_per_limb = digits_per_limb_table[radix - 2];
    }
    cur_limb = 0;
    bf_resize(a, 1);
    pos = 0;
    has_decpt = FALSE;
    int_len = digit_count = 0;
    for(;;) {
        limb_t c;
        if (*p == '.' && (p > p_start || bf_to_digit(p[1]) < radix)) {
            if (has_decpt)
                break;
            has_decpt = TRUE;
            int_len = digit_count;
            p++;
        }
        c = bf_to_digit(*p);
        if (c >= radix)
            break;
        digit_count++;
        p++;
        if (radix_bits) {
            shift -= radix_bits;
            if (shift <= 0) {
                cur_limb |= c >> (-shift);
                if (bf_add_limb(a, &pos, cur_limb))
                    goto mem_error;
                if (shift < 0)
                    cur_limb = c << (LIMB_BITS + shift);
                else
                    cur_limb = 0;
                shift += LIMB_BITS;
            } else {
                cur_limb |= c << shift;
            }
        } else {
            cur_limb = cur_limb * radix + c;
            shift--;
            if (shift == 0) {
                if (bf_add_limb(a, &pos, cur_limb))
                    goto mem_error;
                shift = digits_per_limb;
                cur_limb = 0;
            }
        }
    }
    if (!has_decpt)
        int_len = digit_count;

    /* add the last limb and pad with zeros */
    if (shift != digits_per_limb) {
        if (radix_bits == 0) {
            while (shift != 0) {
                cur_limb *= radix;
                shift--;
            }
        }
        if (bf_add_limb(a, &pos, cur_limb)) {
        mem_error:
            ret = BF_ST_MEM_ERROR;
            if (!radix_bits)
                bf_delete(a);
            bf_set_nan(r);
            goto done;
        }
    }

    /* reset the next limbs to zero (we prefer to reallocate in the
       renormalization) */
    memset(a->tab, 0, (pos + 1) * sizeof(limb_t));

    if (p == p_start) {
        ret = 0;
        if (!radix_bits)
            bf_delete(a);
        bf_set_nan(r);
        goto done;
    }

    /* parse the exponent, if any */
    expn = 0;
    is_bin_exp = FALSE;
    if (((radix == 10 && (*p == 'e' || *p == 'E')) ||
         (radix != 10 && (*p == '@' ||
                          (radix_bits && (*p == 'p' || *p == 'P'))))) &&
        p > p_start) {
        is_bin_exp = (*p == 'p' || *p == 'P');
        p++;
        exp_is_neg = 0;
        if (*p == '+') {
            p++;
        } else if (*p == '-') {
            exp_is_neg = 1;
            p++;
        }
        for(;;) {
            int c;
            c = bf_to_digit(*p);
            if (c >= 10)
                break;
            if (unlikely(expn > ((BF_RAW_EXP_MAX - 2 - 9) / 10))) {
                /* exponent overflow */
                if (exp_is_neg) {
                    bf_set_zero(r, is_neg);
                    ret = BF_ST_UNDERFLOW | BF_ST_INEXACT;
                } else {
                    bf_set_inf(r, is_neg);
                    ret = BF_ST_OVERFLOW | BF_ST_INEXACT;
                }
                goto done;
            }
            p++;
            expn = expn * 10 + c;
        }
        if (exp_is_neg)
            expn = -expn;
    }
    if (is_dec) {
        a->expn = expn + int_len;
        a->sign = is_neg;
        ret = bfdec_normalize_and_round((bfdec_t *)a, prec, flags);
    } else if (radix_bits) {
        /* XXX: may overflow */
        if (!is_bin_exp)
            expn *= radix_bits;
        a->expn = expn + (int_len * radix_bits);
        a->sign = is_neg;
        ret = bf_normalize_and_round(a, prec, flags);
    } else {
        limb_t l;
        pos++;
        l = a->len - pos; /* number of limbs */
        if (l == 0) {
            bf_set_zero(r, is_neg);
            ret = 0;
        } else {
            bf_t T_s, *T = &T_s;

            expn -= l * digits_per_limb - int_len;
            bf_init(r->ctx, T);
            if (bf_integer_from_radix(T, a->tab + pos, l, radix)) {
                bf_set_nan(r);
                ret = BF_ST_MEM_ERROR;
            } else {
                T->sign = is_neg;
                if (flags & BF_ATOF_EXPONENT) {
                    /* return the exponent */
                    *pexponent = expn;
                    ret = bf_set(r, T);
                } else {
                    ret = bf_mul_pow_radix(r, T, radix, expn, prec, flags);
                }
            }
            bf_delete(T);
        }
        bf_delete(a);
    }
 done:
    if (pnext)
        *pnext = p;
    return ret;
}

/*
   Return (status, n, exp). 'status' is the floating point status. 'n'
   is the parsed number.

   If (flags & BF_ATOF_EXPONENT) and if the radix is not a power of
   two, the parsed number is equal to r *
   (*pexponent)^radix. Otherwise *pexponent = 0.
*/
int bf_atof2(bf_t *r, slimb_t *pexponent,
             const char *str, const char **pnext, int radix,
             limb_t prec, bf_flags_t flags)
{
    return bf_atof_internal(r, pexponent, str, pnext, radix, prec, flags,
                            FALSE);
}

int bf_atof(bf_t *r, const char *str, const char **pnext, int radix,
            limb_t prec, bf_flags_t flags)
{
    slimb_t dummy_exp;
    return bf_atof_internal(r, &dummy_exp, str, pnext, radix, prec, flags, FALSE);
}

/* base conversion to radix */

#if LIMB_BITS == 64
#define RADIXL_10 UINT64_C(10000000000000000000)
#else
#define RADIXL_10 UINT64_C(1000000000)
#endif

static const uint32_t inv_log2_radix[BF_RADIX_MAX - 1][LIMB_BITS / 32 + 1] = {
#if LIMB_BITS == 32
{ 0x80000000, 0x00000000,},
{ 0x50c24e60, 0xd4d4f4a7,},
{ 0x40000000, 0x00000000,},
{ 0x372068d2, 0x0a1ee5ca,},
{ 0x3184648d, 0xb8153e7a,},
{ 0x2d983275, 0x9d5369c4,},
{ 0x2aaaaaaa, 0xaaaaaaab,},
{ 0x28612730, 0x6a6a7a54,},
{ 0x268826a1, 0x3ef3fde6,},
{ 0x25001383, 0xbac8a744,},
{ 0x23b46706, 0x82c0c709,},
{ 0x229729f1, 0xb2c83ded,},
{ 0x219e7ffd, 0xa5ad572b,},
{ 0x20c33b88, 0xda7c29ab,},
{ 0x20000000, 0x00000000,},
{ 0x1f50b57e, 0xac5884b3,},
{ 0x1eb22cc6, 0x8aa6e26f,},
{ 0x1e21e118, 0x0c5daab2,},
{ 0x1d9dcd21, 0x439834e4,},
{ 0x1d244c78, 0x367a0d65,},
{ 0x1cb40589, 0xac173e0c,},
{ 0x1c4bd95b, 0xa8d72b0d,},
{ 0x1bead768, 0x98f8ce4c,},
{ 0x1b903469, 0x050f72e5,},
{ 0x1b3b433f, 0x2eb06f15,},
{ 0x1aeb6f75, 0x9c46fc38,},
{ 0x1aa038eb, 0x0e3bfd17,},
{ 0x1a593062, 0xb38d8c56,},
{ 0x1a15f4c3, 0x2b95a2e6,},
{ 0x19d630dc, 0xcc7ddef9,},
{ 0x19999999, 0x9999999a,},
{ 0x195fec80, 0x8a609431,},
{ 0x1928ee7b, 0x0b4f22f9,},
{ 0x18f46acf, 0x8c06e318,},
{ 0x18c23246, 0xdc0a9f3d,},
#else
{ 0x80000000, 0x00000000, 0x00000000,},
{ 0x50c24e60, 0xd4d4f4a7, 0x021f57bc,},
{ 0x40000000, 0x00000000, 0x00000000,},
{ 0x372068d2, 0x0a1ee5ca, 0x19ea911b,},
{ 0x3184648d, 0xb8153e7a, 0x7fc2d2e1,},
{ 0x2d983275, 0x9d5369c4, 0x4dec1661,},
{ 0x2aaaaaaa, 0xaaaaaaaa, 0xaaaaaaab,},
{ 0x28612730, 0x6a6a7a53, 0x810fabde,},
{ 0x268826a1, 0x3ef3fde6, 0x23e2566b,},
{ 0x25001383, 0xbac8a744, 0x385a3349,},
{ 0x23b46706, 0x82c0c709, 0x3f891718,},
{ 0x229729f1, 0xb2c83ded, 0x15fba800,},
{ 0x219e7ffd, 0xa5ad572a, 0xe169744b,},
{ 0x20c33b88, 0xda7c29aa, 0x9bddee52,},
{ 0x20000000, 0x00000000, 0x00000000,},
{ 0x1f50b57e, 0xac5884b3, 0x70e28eee,},
{ 0x1eb22cc6, 0x8aa6e26f, 0x06d1a2a2,},
{ 0x1e21e118, 0x0c5daab1, 0x81b4f4bf,},
{ 0x1d9dcd21, 0x439834e3, 0x81667575,},
{ 0x1d244c78, 0x367a0d64, 0xc8204d6d,},
{ 0x1cb40589, 0xac173e0c, 0x3b7b16ba,},
{ 0x1c4bd95b, 0xa8d72b0d, 0x5879f25a,},
{ 0x1bead768, 0x98f8ce4c, 0x66cc2858,},
{ 0x1b903469, 0x050f72e5, 0x0cf5488e,},
{ 0x1b3b433f, 0x2eb06f14, 0x8c89719c,},
{ 0x1aeb6f75, 0x9c46fc37, 0xab5fc7e9,},
{ 0x1aa038eb, 0x0e3bfd17, 0x1bd62080,},
{ 0x1a593062, 0xb38d8c56, 0x7998ab45,},
{ 0x1a15f4c3, 0x2b95a2e6, 0x46aed6a0,},
{ 0x19d630dc, 0xcc7ddef9, 0x5aadd61b,},
{ 0x19999999, 0x99999999, 0x9999999a,},
{ 0x195fec80, 0x8a609430, 0xe1106014,},
{ 0x1928ee7b, 0x0b4f22f9, 0x5f69791d,},
{ 0x18f46acf, 0x8c06e318, 0x4d2aeb2c,},
{ 0x18c23246, 0xdc0a9f3d, 0x3fe16970,},
#endif
};

static const limb_t log2_radix[BF_RADIX_MAX - 1] = {
#if LIMB_BITS == 32
0x20000000,
0x32b80347,
0x40000000,
0x4a4d3c26,
0x52b80347,
0x59d5d9fd,
0x60000000,
0x6570068e,
0x6a4d3c26,
0x6eb3a9f0,
0x72b80347,
0x766a008e,
0x79d5d9fd,
0x7d053f6d,
0x80000000,
0x82cc7edf,
0x8570068e,
0x87ef05ae,
0x8a4d3c26,
0x8c8ddd45,
0x8eb3a9f0,
0x90c10501,
0x92b80347,
0x949a784c,
0x966a008e,
0x982809d6,
0x99d5d9fd,
0x9b74948f,
0x9d053f6d,
0x9e88c6b3,
0xa0000000,
0xa16bad37,
0xa2cc7edf,
0xa4231623,
0xa570068e,
#else
0x2000000000000000,
0x32b803473f7ad0f4,
0x4000000000000000,
0x4a4d3c25e68dc57f,
0x52b803473f7ad0f4,
0x59d5d9fd5010b366,
0x6000000000000000,
0x6570068e7ef5a1e8,
0x6a4d3c25e68dc57f,
0x6eb3a9f01975077f,
0x72b803473f7ad0f4,
0x766a008e4788cbcd,
0x79d5d9fd5010b366,
0x7d053f6d26089673,
0x8000000000000000,
0x82cc7edf592262d0,
0x8570068e7ef5a1e8,
0x87ef05ae409a0289,
0x8a4d3c25e68dc57f,
0x8c8ddd448f8b845a,
0x8eb3a9f01975077f,
0x90c10500d63aa659,
0x92b803473f7ad0f4,
0x949a784bcd1b8afe,
0x966a008e4788cbcd,
0x982809d5be7072dc,
0x99d5d9fd5010b366,
0x9b74948f5532da4b,
0x9d053f6d26089673,
0x9e88c6b3626a72aa,
0xa000000000000000,
0xa16bad3758efd873,
0xa2cc7edf592262d0,
0xa4231623369e78e6,
0xa570068e7ef5a1e8,
#endif
};

/* compute floor(a*b) or ceil(a*b) with b = log2(radix) or
   b=1/log2(radix). For is_inv = 0, strict accuracy is not guaranteed
   when radix is not a power of two. */
slimb_t bf_mul_log2_radix(slimb_t a1, unsigned int radix, int is_inv,
                          int is_ceil1)
{
    int is_neg;
    limb_t a;
    BOOL is_ceil;

    is_ceil = is_ceil1;
    a = a1;
    if (a1 < 0) {
        a = -a;
        is_neg = 1;
    } else {
        is_neg = 0;
    }
    is_ceil ^= is_neg;
    if ((radix & (radix - 1)) == 0) {
        int radix_bits;
        /* radix is a power of two */
        radix_bits = ceil_log2(radix);
        if (is_inv) {
            if (is_ceil)
                a += radix_bits - 1;
            a = a / radix_bits;
        } else {
            a = a * radix_bits;
        }
    } else {
        const uint32_t *tab;
        limb_t b0, b1;
        dlimb_t t;

        if (is_inv) {
            tab = inv_log2_radix[radix - 2];
#if LIMB_BITS == 32
            b1 = tab[0];
            b0 = tab[1];
#else
            b1 = ((limb_t)tab[0] << 32) | tab[1];
            b0 = (limb_t)tab[2] << 32;
#endif
            t = (dlimb_t)b0 * (dlimb_t)a;
            t = (dlimb_t)b1 * (dlimb_t)a + (t >> LIMB_BITS);
            a = t >> (LIMB_BITS - 1);
        } else {
            b0 = log2_radix[radix - 2];
            t = (dlimb_t)b0 * (dlimb_t)a;
            a = t >> (LIMB_BITS - 3);
        }
        /* a = floor(result) and 'result' cannot be an integer */
        a += is_ceil;
    }
    if (is_neg)
        a = -a;
    return a;
}

/* 'n' is the number of output limbs */
static int bf_integer_to_radix_rec(bf_t *pow_tab,
                                   limb_t *out, const bf_t *a, limb_t n,
                                   int level, limb_t n0, limb_t radixl,
                                   unsigned int radixl_bits)
{
    limb_t n1, n2, q_prec;
    int ret;

    assert(n >= 1);
    if (n == 1) {
        out[0] = get_bits(a->tab, a->len, a->len * LIMB_BITS - a->expn);
    } else if (n == 2) {
        dlimb_t t;
        slimb_t pos;
        pos = a->len * LIMB_BITS - a->expn;
        t = ((dlimb_t)get_bits(a->tab, a->len, pos + LIMB_BITS) << LIMB_BITS) |
            get_bits(a->tab, a->len, pos);
        if (likely(radixl == RADIXL_10)) {
            /* use division by a constant when possible */
            out[0] = t % RADIXL_10;
            out[1] = t / RADIXL_10;
        } else {
            out[0] = t % radixl;
            out[1] = t / radixl;
        }
    } else {
        bf_t Q, R, *B, *B_inv;
        int q_add;
        bf_init(a->ctx, &Q);
        bf_init(a->ctx, &R);
        n2 = (((n0 * 2) >> (level + 1)) + 1) / 2;
        n1 = n - n2;
        B = &pow_tab[2 * level];
        B_inv = &pow_tab[2 * level + 1];
        ret = 0;
        if (B->len == 0) {
            /* compute BASE^n2 */
            ret |= bf_pow_ui_ui(B, radixl, n2, BF_PREC_INF, BF_RNDZ);
            /* we use enough bits for the maximum possible 'n1' value,
               i.e. n2 + 1 */
            ret |= bf_set_ui(&R, 1);
            ret |= bf_div(B_inv, &R, B, (n2 + 1) * radixl_bits + 2, BF_RNDN);
        }
        //        printf("%d: n1=% " PRId64 " n2=%" PRId64 "\n", level, n1, n2);
        q_prec = n1 * radixl_bits;
        ret |= bf_mul(&Q, a, B_inv, q_prec, BF_RNDN);
        ret |= bf_rint(&Q, BF_RNDZ);

        ret |= bf_mul(&R, &Q, B, BF_PREC_INF, BF_RNDZ);
        ret |= bf_sub(&R, a, &R, BF_PREC_INF, BF_RNDZ);

        if (ret & BF_ST_MEM_ERROR)
            goto fail;
        /* adjust if necessary */
        q_add = 0;
        while (R.sign && R.len != 0) {
            if (bf_add(&R, &R, B, BF_PREC_INF, BF_RNDZ))
                goto fail;
            q_add--;
        }
        while (bf_cmpu(&R, B) >= 0) {
            if (bf_sub(&R, &R, B, BF_PREC_INF, BF_RNDZ))
                goto fail;
            q_add++;
        }
        if (q_add != 0) {
            if (bf_add_si(&Q, &Q, q_add, BF_PREC_INF, BF_RNDZ))
                goto fail;
        }
        if (bf_integer_to_radix_rec(pow_tab, out + n2, &Q, n1, level + 1, n0,
                                    radixl, radixl_bits))
            goto fail;
        if (bf_integer_to_radix_rec(pow_tab, out, &R, n2, level + 1, n0,
                                    radixl, radixl_bits)) {
        fail:
            bf_delete(&Q);
            bf_delete(&R);
            return -1;
        }
        bf_delete(&Q);
        bf_delete(&R);
    }
    return 0;
}

/* return 0 if OK != 0 if memory error */
static int bf_integer_to_radix(bf_t *r, const bf_t *a, limb_t radixl)
{
    bf_context_t *s = r->ctx;
    limb_t r_len;
    bf_t *pow_tab;
    int i, pow_tab_len, ret;

    r_len = r->len;
    pow_tab_len = (ceil_log2(r_len) + 2) * 2; /* XXX: check */
    pow_tab = bf_malloc(s, sizeof(pow_tab[0]) * pow_tab_len);
    if (!pow_tab)
        return -1;
    for(i = 0; i < pow_tab_len; i++)
        bf_init(r->ctx, &pow_tab[i]);

    ret = bf_integer_to_radix_rec(pow_tab, r->tab, a, r_len, 0, r_len, radixl,
                                  ceil_log2(radixl));

    for(i = 0; i < pow_tab_len; i++) {
        bf_delete(&pow_tab[i]);
    }
    bf_free(s, pow_tab);
    return ret;
}

/* a must be >= 0. 'P' is the wanted number of digits in radix
   'radix'. 'r' is the mantissa represented as an integer. *pE
   contains the exponent. Return != 0 if memory error. */
static int bf_convert_to_radix(bf_t *r, slimb_t *pE,
                               const bf_t *a, int radix,
                               limb_t P, bf_rnd_t rnd_mode,
                               BOOL is_fixed_exponent)
{
    slimb_t E, e, prec, extra_bits, ziv_extra_bits, prec0;
    bf_t B_s, *B = &B_s;
    int e_sign, ret, res;

    if (a->len == 0) {
        /* zero case */
        *pE = 0;
        return bf_set(r, a);
    }

    if (is_fixed_exponent) {
        E = *pE;
    } else {
        /* compute the new exponent */
        E = 1 + bf_mul_log2_radix(a->expn - 1, radix, TRUE, FALSE);
    }
    //    bf_print_str("a", a);
    //    printf("E=%ld P=%ld radix=%d\n", E, P, radix);

    for(;;) {
        e = P - E;
        e_sign = 0;
        if (e < 0) {
            e = -e;
            e_sign = 1;
        }
        /* Note: precision for log2(radix) is not critical here */
        prec0 = bf_mul_log2_radix(P, radix, FALSE, TRUE);
        ziv_extra_bits = 16;
        for(;;) {
            prec = prec0 + ziv_extra_bits;
            /* XXX: rigorous error analysis needed */
            extra_bits = ceil_log2(e) * 2 + 1;
            ret = bf_pow_ui_ui(r, radix, e, prec + extra_bits,
                               BF_RNDN | BF_FLAG_EXT_EXP);
            if (!e_sign)
                ret |= bf_mul(r, r, a, prec + extra_bits,
                              BF_RNDN | BF_FLAG_EXT_EXP);
            else
                ret |= bf_div(r, a, r, prec + extra_bits,
                              BF_RNDN | BF_FLAG_EXT_EXP);
            if (ret & BF_ST_MEM_ERROR)
                return BF_ST_MEM_ERROR;
            /* if the result is not exact, check that it can be safely
               rounded to an integer */
            if ((ret & BF_ST_INEXACT) &&
                !bf_can_round(r, r->expn, rnd_mode, prec)) {
                /* and more precision and retry */
                ziv_extra_bits = ziv_extra_bits  + (ziv_extra_bits / 2);
                continue;
            } else {
                ret = bf_rint(r, rnd_mode);
                if (ret & BF_ST_MEM_ERROR)
                    return BF_ST_MEM_ERROR;
                break;
            }
        }
        if (is_fixed_exponent)
            break;
        /* check that the result is < B^P */
        /* XXX: do a fast approximate test first ? */
        bf_init(r->ctx, B);
        ret = bf_pow_ui_ui(B, radix, P, BF_PREC_INF, BF_RNDZ);
        if (ret) {
            bf_delete(B);
            return ret;
        }
        res = bf_cmpu(r, B);
        bf_delete(B);
        if (res < 0)
            break;
        /* try a larger exponent */
        E++;
    }
    *pE = E;
    return 0;
}

static void limb_to_a(char *buf, limb_t n, unsigned int radix, int len)
{
    int digit, i;

    if (radix == 10) {
        /* specific case with constant divisor */
        for(i = len - 1; i >= 0; i--) {
            digit = (limb_t)n % 10;
            n = (limb_t)n / 10;
            buf[i] = digit + '0';
        }
    } else {
        for(i = len - 1; i >= 0; i--) {
            digit = (limb_t)n % radix;
            n = (limb_t)n / radix;
            if (digit < 10)
                digit += '0';
            else
                digit += 'a' - 10;
            buf[i] = digit;
        }
    }
}

/* for power of 2 radixes */
static void limb_to_a2(char *buf, limb_t n, unsigned int radix_bits, int len)
{
    int digit, i;
    unsigned int mask;

    mask = (1 << radix_bits) - 1;
    for(i = len - 1; i >= 0; i--) {
        digit = n & mask;
        n >>= radix_bits;
        if (digit < 10)
            digit += '0';
        else
            digit += 'a' - 10;
        buf[i] = digit;
    }
}

/* 'a' must be an integer if the is_dec = FALSE or if the radix is not
   a power of two. A dot is added before the 'dot_pos' digit. dot_pos
   = n_digits does not display the dot. 0 <= dot_pos <=
   n_digits. n_digits >= 1. */
static void output_digits(DynBuf *s, const bf_t *a1, int radix, limb_t n_digits,
                          limb_t dot_pos, BOOL is_dec)
{
    limb_t i, v, l;
    slimb_t pos, pos_incr;
    int digits_per_limb, buf_pos, radix_bits, first_buf_pos;
    char buf[65];
    bf_t a_s, *a;

    if (is_dec) {
        digits_per_limb = LIMB_DIGITS;
        a = (bf_t *)a1;
        radix_bits = 0;
        pos = a->len;
        pos_incr = 1;
        first_buf_pos = 0;
    } else if ((radix & (radix - 1)) == 0) {
        a = (bf_t *)a1;
        radix_bits = ceil_log2(radix);
        digits_per_limb = LIMB_BITS / radix_bits;
        pos_incr = digits_per_limb * radix_bits;
        /* digits are aligned relative to the radix point */
        pos = a->len * LIMB_BITS + smod(-a->expn, radix_bits);
        first_buf_pos = 0;
    } else {
        limb_t n, radixl;

        digits_per_limb = digits_per_limb_table[radix - 2];
        radixl = get_limb_radix(radix);
        a = &a_s;
        bf_init(a1->ctx, a);
        n = (n_digits + digits_per_limb - 1) / digits_per_limb;
        if (bf_resize(a, n)) {
            dbuf_set_error(s);
            goto done;
        }
        if (bf_integer_to_radix(a, a1, radixl)) {
            dbuf_set_error(s);
            goto done;
        }
        radix_bits = 0;
        pos = n;
        pos_incr = 1;
        first_buf_pos = pos * digits_per_limb - n_digits;
    }
    buf_pos = digits_per_limb;
    i = 0;
    while (i < n_digits) {
        if (buf_pos == digits_per_limb) {
            pos -= pos_incr;
            if (radix_bits == 0) {
                v = get_limbz(a, pos);
                limb_to_a(buf, v, radix, digits_per_limb);
            } else {
                v = get_bits(a->tab, a->len, pos);
                limb_to_a2(buf, v, radix_bits, digits_per_limb);
            }
            buf_pos = first_buf_pos;
            first_buf_pos = 0;
        }
        if (i < dot_pos) {
            l = dot_pos;
        } else {
            if (i == dot_pos)
                dbuf_putc(s, '.');
            l = n_digits;
        }
        l = bf_min(digits_per_limb - buf_pos, l - i);
        dbuf_put(s, (uint8_t *)(buf + buf_pos), l);
        buf_pos += l;
        i += l;
    }
 done:
    if (a != a1)
        bf_delete(a);
}

static void *bf_dbuf_realloc(void *opaque, void *ptr, size_t size)
{
    bf_context_t *s = opaque;
    return bf_realloc(s, ptr, size);
}

/* return the length in bytes. A trailing '\0' is added */
static char *bf_ftoa_internal(size_t *plen, const bf_t *a2, int radix,
                              limb_t prec, bf_flags_t flags, BOOL is_dec)
{
    bf_context_t *ctx = a2->ctx;
    DynBuf s_s, *s = &s_s;
    int radix_bits;

    //    bf_print_str("ftoa", a2);
    //    printf("radix=%d\n", radix);
    dbuf_init2(s, ctx, bf_dbuf_realloc);
    if (a2->expn == BF_EXP_NAN) {
        dbuf_putstr(s, "NaN");
    } else {
        if (a2->sign)
            dbuf_putc(s, '-');
        if (a2->expn == BF_EXP_INF) {
            if (flags & BF_FTOA_JS_QUIRKS)
                dbuf_putstr(s, "Infinity");
            else
                dbuf_putstr(s, "Inf");
        } else {
            int fmt, ret;
            slimb_t n_digits, n, i, n_max, n1;
            bf_t a1_s, *a1 = &a1_s;

            if ((radix & (radix - 1)) != 0)
                radix_bits = 0;
            else
                radix_bits = ceil_log2(radix);

            fmt = flags & BF_FTOA_FORMAT_MASK;
            bf_init(ctx, a1);
            if (fmt == BF_FTOA_FORMAT_FRAC) {
                if (is_dec || radix_bits != 0) {
                    if (bf_set(a1, a2))
                        goto fail1;
#ifdef USE_BF_DEC
                    if (is_dec) {
                        if (bfdec_round((bfdec_t *)a1, prec, (flags & BF_RND_MASK) | BF_FLAG_RADPNT_PREC) & BF_ST_MEM_ERROR)
                            goto fail1;
                        n = a1->expn;
                    } else
#endif
                    {
                        if (bf_round(a1, prec * radix_bits, (flags & BF_RND_MASK) | BF_FLAG_RADPNT_PREC) & BF_ST_MEM_ERROR)
                            goto fail1;
                        n = ceil_div(a1->expn, radix_bits);
                    }
                    if (flags & BF_FTOA_ADD_PREFIX) {
                        if (radix == 16)
                            dbuf_putstr(s, "0x");
                        else if (radix == 8)
                            dbuf_putstr(s, "0o");
                        else if (radix == 2)
                            dbuf_putstr(s, "0b");
                    }
                    if (a1->expn == BF_EXP_ZERO) {
                        dbuf_putstr(s, "0");
                        if (prec > 0) {
                            dbuf_putstr(s, ".");
                            for(i = 0; i < prec; i++) {
                                dbuf_putc(s, '0');
                            }
                        }
                    } else {
                        n_digits = prec + n;
                        if (n <= 0) {
                            /* 0.x */
                            dbuf_putstr(s, "0.");
                            for(i = 0; i < -n; i++) {
                                dbuf_putc(s, '0');
                            }
                            if (n_digits > 0) {
                                output_digits(s, a1, radix, n_digits, n_digits, is_dec);
                            }
                        } else {
                            output_digits(s, a1, radix, n_digits, n, is_dec);
                        }
                    }
                } else {
                    size_t pos, start;
                    bf_t a_s, *a = &a_s;

                    /* make a positive number */
                    a->tab = a2->tab;
                    a->len = a2->len;
                    a->expn = a2->expn;
                    a->sign = 0;

                    /* one more digit for the rounding */
                    n = 1 + bf_mul_log2_radix(bf_max(a->expn, 0), radix, TRUE, TRUE);
                    n_digits = n + prec;
                    n1 = n;
                    if (bf_convert_to_radix(a1, &n1, a, radix, n_digits,
                                            flags & BF_RND_MASK, TRUE))
                        goto fail1;
                    start = s->size;
                    output_digits(s, a1, radix, n_digits, n, is_dec);
                    /* remove leading zeros because we allocated one more digit */
                    pos = start;
                    while ((pos + 1) < s->size && s->buf[pos] == '0' &&
                           s->buf[pos + 1] != '.')
                        pos++;
                    if (pos > start) {
                        memmove(s->buf + start, s->buf + pos, s->size - pos);
                        s->size -= (pos - start);
                    }
                }
            } else {
#ifdef USE_BF_DEC
                if (is_dec) {
                    if (bf_set(a1, a2))
                        goto fail1;
                    if (fmt == BF_FTOA_FORMAT_FIXED) {
                        n_digits = prec;
                        n_max = n_digits;
                        if (bfdec_round((bfdec_t *)a1, prec, (flags & BF_RND_MASK)) & BF_ST_MEM_ERROR)
                            goto fail1;
                    } else {
                        /* prec is ignored */
                        prec = n_digits = a1->len * LIMB_DIGITS;
                        /* remove the trailing zero digits */
                        while (n_digits > 1 &&
                               get_digit(a1->tab, a1->len, prec - n_digits) == 0) {
                            n_digits--;
                        }
                        n_max = n_digits + 4;
                    }
                    n = a1->expn;
                } else
#endif
                if (radix_bits != 0) {
                    if (bf_set(a1, a2))
                        goto fail1;
                    if (fmt == BF_FTOA_FORMAT_FIXED) {
                        slimb_t prec_bits;
                        n_digits = prec;
                        n_max = n_digits;
                        /* align to the radix point */
                        prec_bits = prec * radix_bits -
                            smod(-a1->expn, radix_bits);
                        if (bf_round(a1, prec_bits,
                                     (flags & BF_RND_MASK)) & BF_ST_MEM_ERROR)
                            goto fail1;
                    } else {
                        limb_t digit_mask;
                        slimb_t pos;
                        /* position of the digit before the most
                           significant digit in bits */
                        pos = a1->len * LIMB_BITS +
                            smod(-a1->expn, radix_bits);
                        n_digits = ceil_div(pos, radix_bits);
                        /* remove the trailing zero digits */
                        digit_mask = ((limb_t)1 << radix_bits) - 1;
                        while (n_digits > 1 &&
                               (get_bits(a1->tab, a1->len, pos - n_digits * radix_bits) & digit_mask) == 0) {
                            n_digits--;
                        }
                        n_max = n_digits + 4;
                    }
                    n = ceil_div(a1->expn, radix_bits);
                } else {
                    bf_t a_s, *a = &a_s;

                    /* make a positive number */
                    a->tab = a2->tab;
                    a->len = a2->len;
                    a->expn = a2->expn;
                    a->sign = 0;

                    if (fmt == BF_FTOA_FORMAT_FIXED) {
                        n_digits = prec;
                        n_max = n_digits;
                    } else {
                        slimb_t n_digits_max, n_digits_min;

                        assert(prec != BF_PREC_INF);
                        n_digits = 1 + bf_mul_log2_radix(prec, radix, TRUE, TRUE);
                        /* max number of digits for non exponential
                           notation. The rational is to have the same rule
                           as JS i.e. n_max = 21 for 64 bit float in base 10. */
                        n_max = n_digits + 4;
                        if (fmt == BF_FTOA_FORMAT_FREE_MIN) {
                            bf_t b_s, *b = &b_s;

                            /* find the minimum number of digits by
                               dichotomy. */
                            /* XXX: inefficient */
                            n_digits_max = n_digits;
                            n_digits_min = 1;
                            bf_init(ctx, b);
                            while (n_digits_min < n_digits_max) {
                                n_digits = (n_digits_min + n_digits_max) / 2;
                                if (bf_convert_to_radix(a1, &n, a, radix, n_digits,
                                                        flags & BF_RND_MASK, FALSE)) {
                                    bf_delete(b);
                                    goto fail1;
                                }
                                /* convert back to a number and compare */
                                ret = bf_mul_pow_radix(b, a1, radix, n - n_digits,
                                                       prec,
                                                       (flags & ~BF_RND_MASK) |
                                                       BF_RNDN);
                                if (ret & BF_ST_MEM_ERROR) {
                                    bf_delete(b);
                                    goto fail1;
                                }
                                if (bf_cmpu(b, a) == 0) {
                                    n_digits_max = n_digits;
                                } else {
                                    n_digits_min = n_digits + 1;
                                }
                            }
                            bf_delete(b);
                            n_digits = n_digits_max;
                        }
                    }
                    if (bf_convert_to_radix(a1, &n, a, radix, n_digits,
                                            flags & BF_RND_MASK, FALSE)) {
                    fail1:
                        bf_delete(a1);
                        goto fail;
                    }
                }
                if (a1->expn == BF_EXP_ZERO &&
                    fmt != BF_FTOA_FORMAT_FIXED &&
                    !(flags & BF_FTOA_FORCE_EXP)) {
                    /* just output zero */
                    dbuf_putstr(s, "0");
                } else {
                    if (flags & BF_FTOA_ADD_PREFIX) {
                        if (radix == 16)
                            dbuf_putstr(s, "0x");
                        else if (radix == 8)
                            dbuf_putstr(s, "0o");
                        else if (radix == 2)
                            dbuf_putstr(s, "0b");
                    }
                    if (a1->expn == BF_EXP_ZERO)
                        n = 1;
                    if ((flags & BF_FTOA_FORCE_EXP) ||
                        n <= -6 || n > n_max) {
                        const char *fmt;
                        /* exponential notation */
                        output_digits(s, a1, radix, n_digits, 1, is_dec);
                        if (radix_bits != 0 && radix <= 16) {
                            if (flags & BF_FTOA_JS_QUIRKS)
                                fmt = "p%+" PRId_LIMB;
                            else
                                fmt = "p%" PRId_LIMB;
                            dbuf_printf(s, fmt, (n - 1) * radix_bits);
                        } else {
                            if (flags & BF_FTOA_JS_QUIRKS)
                                fmt = "%c%+" PRId_LIMB;
                            else
                                fmt = "%c%" PRId_LIMB;
                            dbuf_printf(s, fmt,
                                        radix <= 10 ? 'e' : '@', n - 1);
                        }
                    } else if (n <= 0) {
                        /* 0.x */
                        dbuf_putstr(s, "0.");
                        for(i = 0; i < -n; i++) {
                            dbuf_putc(s, '0');
                        }
                        output_digits(s, a1, radix, n_digits, n_digits, is_dec);
                    } else {
                        if (n_digits <= n) {
                            /* no dot */
                            output_digits(s, a1, radix, n_digits, n_digits, is_dec);
                            for(i = 0; i < (n - n_digits); i++)
                                dbuf_putc(s, '0');
                        } else {
                            output_digits(s, a1, radix, n_digits, n, is_dec);
                        }
                    }
                }
            }
            bf_delete(a1);
        }
    }
    dbuf_putc(s, '\0');
    if (dbuf_error(s))
        goto fail;
    if (plen)
        *plen = s->size - 1;
    return (char *)s->buf;
 fail:
    bf_free(ctx, s->buf);
    if (plen)
        *plen = 0;
    return NULL;
}

char *bf_ftoa(size_t *plen, const bf_t *a, int radix, limb_t prec,
              bf_flags_t flags)
{
    return bf_ftoa_internal(plen, a, radix, prec, flags, FALSE);
}

/***************************************************************/
/* transcendental functions */

/* Note: the algorithm is from MPFR */
static void bf_const_log2_rec(bf_t *T, bf_t *P, bf_t *Q, limb_t n1,
                              limb_t n2, BOOL need_P)
{
    bf_context_t *s = T->ctx;
    if ((n2 - n1) == 1) {
        if (n1 == 0) {
            bf_set_ui(P, 3);
        } else {
            bf_set_ui(P, n1);
            P->sign = 1;
        }
        bf_set_ui(Q, 2 * n1 + 1);
        Q->expn += 2;
        bf_set(T, P);
    } else {
        limb_t m;
        bf_t T1_s, *T1 = &T1_s;
        bf_t P1_s, *P1 = &P1_s;
        bf_t Q1_s, *Q1 = &Q1_s;

        m = n1 + ((n2 - n1) >> 1);
        bf_const_log2_rec(T, P, Q, n1, m, TRUE);
        bf_init(s, T1);
        bf_init(s, P1);
        bf_init(s, Q1);
        bf_const_log2_rec(T1, P1, Q1, m, n2, need_P);
        bf_mul(T, T, Q1, BF_PREC_INF, BF_RNDZ);
        bf_mul(T1, T1, P, BF_PREC_INF, BF_RNDZ);
        bf_add(T, T, T1, BF_PREC_INF, BF_RNDZ);
        if (need_P)
            bf_mul(P, P, P1, BF_PREC_INF, BF_RNDZ);
        bf_mul(Q, Q, Q1, BF_PREC_INF, BF_RNDZ);
        bf_delete(T1);
        bf_delete(P1);
        bf_delete(Q1);
    }
}

/* compute log(2) with faithful rounding at precision 'prec' */
static void bf_const_log2_internal(bf_t *T, limb_t prec)
{
    limb_t w, N;
    bf_t P_s, *P = &P_s;
    bf_t Q_s, *Q = &Q_s;

    w = prec + 15;
    N = w / 3 + 1;
    bf_init(T->ctx, P);
    bf_init(T->ctx, Q);
    bf_const_log2_rec(T, P, Q, 0, N, FALSE);
    bf_div(T, T, Q, prec, BF_RNDN);
    bf_delete(P);
    bf_delete(Q);
}

/* PI constant */

#define CHUD_A 13591409
#define CHUD_B 545140134
#define CHUD_C 640320
#define CHUD_BITS_PER_TERM 47

static void chud_bs(bf_t *P, bf_t *Q, bf_t *G, int64_t a, int64_t b, int need_g,
                    limb_t prec)
{
    bf_context_t *s = P->ctx;
    int64_t c;

    if (a == (b - 1)) {
        bf_t T0, T1;

        bf_init(s, &T0);
        bf_init(s, &T1);
        bf_set_ui(G, 2 * b - 1);
        bf_mul_ui(G, G, 6 * b - 1, prec, BF_RNDN);
        bf_mul_ui(G, G, 6 * b - 5, prec, BF_RNDN);
        bf_set_ui(&T0, CHUD_B);
        bf_mul_ui(&T0, &T0, b, prec, BF_RNDN);
        bf_set_ui(&T1, CHUD_A);
        bf_add(&T0, &T0, &T1, prec, BF_RNDN);
        bf_mul(P, G, &T0, prec, BF_RNDN);
        P->sign = b & 1;

        bf_set_ui(Q, b);
        bf_mul_ui(Q, Q, b, prec, BF_RNDN);
        bf_mul_ui(Q, Q, b, prec, BF_RNDN);
        bf_mul_ui(Q, Q, (uint64_t)CHUD_C * CHUD_C * CHUD_C / 24, prec, BF_RNDN);
        bf_delete(&T0);
        bf_delete(&T1);
    } else {
        bf_t P2, Q2, G2;

        bf_init(s, &P2);
        bf_init(s, &Q2);
        bf_init(s, &G2);

        c = (a + b) / 2;
        chud_bs(P, Q, G, a, c, 1, prec);
        chud_bs(&P2, &Q2, &G2, c, b, need_g, prec);

        /* Q = Q1 * Q2 */
        /* G = G1 * G2 */
        /* P = P1 * Q2 + P2 * G1 */
        bf_mul(&P2, &P2, G, prec, BF_RNDN);
        if (!need_g)
            bf_set_ui(G, 0);
        bf_mul(P, P, &Q2, prec, BF_RNDN);
        bf_add(P, P, &P2, prec, BF_RNDN);
        bf_delete(&P2);

        bf_mul(Q, Q, &Q2, prec, BF_RNDN);
        bf_delete(&Q2);
        if (need_g)
            bf_mul(G, G, &G2, prec, BF_RNDN);
        bf_delete(&G2);
    }
}

/* compute Pi with faithful rounding at precision 'prec' using the
   Chudnovsky formula */
static void bf_const_pi_internal(bf_t *Q, limb_t prec)
{
    bf_context_t *s = Q->ctx;
    int64_t n, prec1;
    bf_t P, G;

    /* number of serie terms */
    n = prec / CHUD_BITS_PER_TERM + 1;
    /* XXX: precision analysis */
    prec1 = prec + 32;

    bf_init(s, &P);
    bf_init(s, &G);

    chud_bs(&P, Q, &G, 0, n, 0, BF_PREC_INF);

    bf_mul_ui(&G, Q, CHUD_A, prec1, BF_RNDN);
    bf_add(&P, &G, &P, prec1, BF_RNDN);
    bf_div(Q, Q, &P, prec1, BF_RNDF);

    bf_set_ui(&P, CHUD_C);
    bf_sqrt(&G, &P, prec1, BF_RNDF);
    bf_mul_ui(&G, &G, (uint64_t)CHUD_C / 12, prec1, BF_RNDF);
    bf_mul(Q, Q, &G, prec, BF_RNDN);
    bf_delete(&P);
    bf_delete(&G);
}

static int bf_const_get(bf_t *T, limb_t prec, bf_flags_t flags,
                        BFConstCache *c,
                        void (*func)(bf_t *res, limb_t prec), int sign)
{
    limb_t ziv_extra_bits, prec1;

    ziv_extra_bits = 32;
    for(;;) {
        prec1 = prec + ziv_extra_bits;
        if (c->prec < prec1) {
            if (c->val.len == 0)
                bf_init(T->ctx, &c->val);
            func(&c->val, prec1);
            c->prec = prec1;
        } else {
            prec1 = c->prec;
        }
        bf_set(T, &c->val);
        T->sign = sign;
        if (!bf_can_round(T, prec, flags & BF_RND_MASK, prec1)) {
            /* and more precision and retry */
            ziv_extra_bits = ziv_extra_bits  + (ziv_extra_bits / 2);
        } else {
            break;
        }
    }
    return bf_round(T, prec, flags);
}

static void bf_const_free(BFConstCache *c)
{
    bf_delete(&c->val);
    memset(c, 0, sizeof(*c));
}

int bf_const_log2(bf_t *T, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = T->ctx;
    return bf_const_get(T, prec, flags, &s->log2_cache, bf_const_log2_internal, 0);
}

/* return rounded pi * (1 - 2 * sign) */
static int bf_const_pi_signed(bf_t *T, int sign, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = T->ctx;
    return bf_const_get(T, prec, flags, &s->pi_cache, bf_const_pi_internal,
                        sign);
}

int bf_const_pi(bf_t *T, limb_t prec, bf_flags_t flags)
{
    return bf_const_pi_signed(T, 0, prec, flags);
}

void bf_clear_cache(bf_context_t *s)
{
#ifdef USE_FFT_MUL
    fft_clear_cache(s);
#endif
    bf_const_free(&s->log2_cache);
    bf_const_free(&s->pi_cache);
}

/* ZivFunc should compute the result 'r' with faithful rounding at
   precision 'prec'. For efficiency purposes, the final bf_round()
   does not need to be done in the function. */
typedef int ZivFunc(bf_t *r, const bf_t *a, limb_t prec, void *opaque);

static int bf_ziv_rounding(bf_t *r, const bf_t *a,
                           limb_t prec, bf_flags_t flags,
                           ZivFunc *f, void *opaque)
{
    int rnd_mode, ret;
    slimb_t prec1, ziv_extra_bits;

    rnd_mode = flags & BF_RND_MASK;
    if (rnd_mode == BF_RNDF) {
        /* no need to iterate */
        f(r, a, prec, opaque);
        ret = 0;
    } else {
        ziv_extra_bits = 32;
        for(;;) {
            prec1 = prec + ziv_extra_bits;
            ret = f(r, a, prec1, opaque);
            if (ret & (BF_ST_OVERFLOW | BF_ST_UNDERFLOW | BF_ST_MEM_ERROR)) {
                /* overflow or underflow should never happen because
                   it indicates the rounding cannot be done correctly,
                   but we do not catch all the cases */
                return ret;
            }
            /* if the result is exact, we can stop */
            if (!(ret & BF_ST_INEXACT)) {
                ret = 0;
                break;
            }
            if (bf_can_round(r, prec, rnd_mode, prec1)) {
                ret = BF_ST_INEXACT;
                break;
            }
            ziv_extra_bits = ziv_extra_bits * 2;
            //            printf("ziv_extra_bits=%" PRId64 "\n", (int64_t)ziv_extra_bits);
        }
    }
    if (r->len == 0)
        return ret;
    else
        return __bf_round(r, prec, flags, r->len, ret);
}

/* add (1 - 2*e_sign) * 2^e */
static int bf_add_epsilon(bf_t *r, const bf_t *a, slimb_t e, int e_sign,
                          limb_t prec, int flags)
{
    bf_t T_s, *T = &T_s;
    int ret;
    /* small argument case: result = 1 + epsilon * sign(x) */
    bf_init(a->ctx, T);
    bf_set_ui(T, 1);
    T->sign = e_sign;
    T->expn += e;
    ret = bf_add(r, r, T, prec, flags);
    bf_delete(T);
    return ret;
}

/* Compute the exponential using faithful rounding at precision 'prec'.
   Note: the algorithm is from MPFR */
static int bf_exp_internal(bf_t *r, const bf_t *a, limb_t prec, void *opaque)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;
    slimb_t n, K, l, i, prec1;

    assert(r != a);

    /* argument reduction:
       T = a - n*log(2) with 0 <= T < log(2) and n integer.
    */
    bf_init(s, T);
    if (a->expn <= -1) {
        /* 0 <= abs(a) <= 0.5 */
        if (a->sign)
            n = -1;
        else
            n = 0;
    } else {
        bf_const_log2(T, LIMB_BITS, BF_RNDZ);
        bf_div(T, a, T, LIMB_BITS, BF_RNDD);
        bf_get_limb(&n, T, 0);
    }

    K = bf_isqrt((prec + 1) / 2);
    l = (prec - 1) / K + 1;
    /* XXX: precision analysis ? */
    prec1 = prec + (K + 2 * l + 18) + K + 8;
    if (a->expn > 0)
        prec1 += a->expn;
    //    printf("n=%ld K=%ld prec1=%ld\n", n, K, prec1);

    bf_const_log2(T, prec1, BF_RNDF);
    bf_mul_si(T, T, n, prec1, BF_RNDN);
    bf_sub(T, a, T, prec1, BF_RNDN);

    /* reduce the range of T */
    bf_mul_2exp(T, -K, BF_PREC_INF, BF_RNDZ);

    /* Taylor expansion around zero :
     1 + x + x^2/2 + ... + x^n/n!
     = (1 + x * (1 + x/2 * (1 + ... (x/n))))
    */
    {
        bf_t U_s, *U = &U_s;

        bf_init(s, U);
        bf_set_ui(r, 1);
        for(i = l ; i >= 1; i--) {
            bf_set_ui(U, i);
            bf_div(U, T, U, prec1, BF_RNDN);
            bf_mul(r, r, U, prec1, BF_RNDN);
            bf_add_si(r, r, 1, prec1, BF_RNDN);
        }
        bf_delete(U);
    }
    bf_delete(T);

    /* undo the range reduction */
    for(i = 0; i < K; i++) {
        bf_mul(r, r, r, prec1, BF_RNDN | BF_FLAG_EXT_EXP);
    }

    /* undo the argument reduction */
    bf_mul_2exp(r, n, BF_PREC_INF, BF_RNDZ | BF_FLAG_EXT_EXP);

    return BF_ST_INEXACT;
}

/* crude overflow and underflow tests for exp(a). a_low <= a <= a_high */
static int check_exp_underflow_overflow(bf_context_t *s, bf_t *r,
                                        const bf_t *a_low, const bf_t *a_high,
                                        limb_t prec, bf_flags_t flags)
{
    bf_t T_s, *T = &T_s;
    bf_t log2_s, *log2 = &log2_s;
    slimb_t e_min, e_max;

    if (a_high->expn <= 0)
        return 0;

    e_max = (limb_t)1 << (bf_get_exp_bits(flags) - 1);
    e_min = -e_max + 3;
    if (flags & BF_FLAG_SUBNORMAL)
        e_min -= (prec - 1);

    bf_init(s, T);
    bf_init(s, log2);
    bf_const_log2(log2, LIMB_BITS, BF_RNDU);
    bf_mul_ui(T, log2, e_max, LIMB_BITS, BF_RNDU);
    /* a_low > e_max * log(2) implies exp(a) > e_max */
    if (bf_cmp_lt(T, a_low) > 0) {
        /* overflow */
        bf_delete(T);
        bf_delete(log2);
        return bf_set_overflow(r, 0, prec, flags);
    }
    /* a_high < (e_min - 2) * log(2) implies exp(a) < (e_min - 2) */
    bf_const_log2(log2, LIMB_BITS, BF_RNDD);
    bf_mul_si(T, log2, e_min - 2, LIMB_BITS, BF_RNDD);
    if (bf_cmp_lt(a_high, T)) {
        int rnd_mode = flags & BF_RND_MASK;

        /* underflow */
        bf_delete(T);
        bf_delete(log2);
        if (rnd_mode == BF_RNDU) {
            /* set the smallest value */
            bf_set_ui(r, 1);
            r->expn = e_min;
        } else {
            bf_set_zero(r, 0);
        }
        return BF_ST_UNDERFLOW | BF_ST_INEXACT;
    }
    bf_delete(log2);
    bf_delete(T);
    return 0;
}

int bf_exp(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = r->ctx;
    int ret;
    assert(r != a);
    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
        } else if (a->expn == BF_EXP_INF) {
            if (a->sign)
                bf_set_zero(r, 0);
            else
                bf_set_inf(r, 0);
        } else {
            bf_set_ui(r, 1);
        }
        return 0;
    }

    ret = check_exp_underflow_overflow(s, r, a, a, prec, flags);
    if (ret)
        return ret;
    if (a->expn < 0 && (-a->expn) >= (prec + 2)) {
        /* small argument case: result = 1 + epsilon * sign(x) */
        bf_set_ui(r, 1);
        return bf_add_epsilon(r, r, -(prec + 2), a->sign, prec, flags);
    }

    return bf_ziv_rounding(r, a, prec, flags, bf_exp_internal, NULL);
}

static int bf_log_internal(bf_t *r, const bf_t *a, limb_t prec, void *opaque)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;
    bf_t U_s, *U = &U_s;
    bf_t V_s, *V = &V_s;
    slimb_t n, prec1, l, i, K;

    assert(r != a);

    bf_init(s, T);
    /* argument reduction 1 */
    /* T=a*2^n with 2/3 <= T <= 4/3 */
    {
        bf_t U_s, *U = &U_s;
        bf_set(T, a);
        n = T->expn;
        T->expn = 0;
        /* U= ~ 2/3 */
        bf_init(s, U);
        bf_set_ui(U, 0xaaaaaaaa);
        U->expn = 0;
        if (bf_cmp_lt(T, U)) {
            T->expn++;
            n--;
        }
        bf_delete(U);
    }
    //    printf("n=%ld\n", n);
    //    bf_print_str("T", T);

    /* XXX: precision analysis */
    /* number of iterations for argument reduction 2 */
    K = bf_isqrt((prec + 1) / 2);
    /* order of Taylor expansion */
    l = prec / (2 * K) + 1;
    /* precision of the intermediate computations */
    prec1 = prec + K + 2 * l + 32;

    bf_init(s, U);
    bf_init(s, V);

    /* Note: cancellation occurs here, so we use more precision (XXX:
       reduce the precision by computing the exact cancellation) */
    bf_add_si(T, T, -1, BF_PREC_INF, BF_RNDN);

    /* argument reduction 2 */
    for(i = 0; i < K; i++) {
        /* T = T / (1 + sqrt(1 + T)) */
        bf_add_si(U, T, 1, prec1, BF_RNDN);
        bf_sqrt(V, U, prec1, BF_RNDF);
        bf_add_si(U, V, 1, prec1, BF_RNDN);
        bf_div(T, T, U, prec1, BF_RNDN);
    }

    {
        bf_t Y_s, *Y = &Y_s;
        bf_t Y2_s, *Y2 = &Y2_s;
        bf_init(s, Y);
        bf_init(s, Y2);

        /* compute ln(1+x) = ln((1+y)/(1-y)) with y=x/(2+x)
           = y + y^3/3 + ... + y^(2*l + 1) / (2*l+1)
           with Y=Y^2
           = y*(1+Y/3+Y^2/5+...) = y*(1+Y*(1/3+Y*(1/5 + ...)))
        */
        bf_add_si(Y, T, 2, prec1, BF_RNDN);
        bf_div(Y, T, Y, prec1, BF_RNDN);

        bf_mul(Y2, Y, Y, prec1, BF_RNDN);
        bf_set_ui(r, 0);
        for(i = l; i >= 1; i--) {
            bf_set_ui(U, 1);
            bf_set_ui(V, 2 * i + 1);
            bf_div(U, U, V, prec1, BF_RNDN);
            bf_add(r, r, U, prec1, BF_RNDN);
            bf_mul(r, r, Y2, prec1, BF_RNDN);
        }
        bf_add_si(r, r, 1, prec1, BF_RNDN);
        bf_mul(r, r, Y, prec1, BF_RNDN);
        bf_delete(Y);
        bf_delete(Y2);
    }
    bf_delete(V);
    bf_delete(U);

    /* multiplication by 2 for the Taylor expansion and undo the
       argument reduction 2*/
    bf_mul_2exp(r, K + 1, BF_PREC_INF, BF_RNDZ);

    /* undo the argument reduction 1 */
    bf_const_log2(T, prec1, BF_RNDF);
    bf_mul_si(T, T, n, prec1, BF_RNDN);
    bf_add(r, r, T, prec1, BF_RNDN);

    bf_delete(T);
    return BF_ST_INEXACT;
}

int bf_log(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;

    assert(r != a);
    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF) {
            if (a->sign) {
                bf_set_nan(r);
                return BF_ST_INVALID_OP;
            } else {
                bf_set_inf(r, 0);
                return 0;
            }
        } else {
            bf_set_inf(r, 1);
            return 0;
        }
    }
    if (a->sign) {
        bf_set_nan(r);
        return BF_ST_INVALID_OP;
    }
    bf_init(s, T);
    bf_set_ui(T, 1);
    if (bf_cmp_eq(a, T)) {
        bf_set_zero(r, 0);
        bf_delete(T);
        return 0;
    }
    bf_delete(T);

    return bf_ziv_rounding(r, a, prec, flags, bf_log_internal, NULL);
}

/* x and y finite and x > 0 */
static int bf_pow_generic(bf_t *r, const bf_t *x, limb_t prec, void *opaque)
{
    bf_context_t *s = r->ctx;
    const bf_t *y = opaque;
    bf_t T_s, *T = &T_s;
    limb_t prec1;

    bf_init(s, T);
    /* XXX: proof for the added precision */
    prec1 = prec + 32;
    bf_log(T, x, prec1, BF_RNDF | BF_FLAG_EXT_EXP);
    bf_mul(T, T, y, prec1, BF_RNDF | BF_FLAG_EXT_EXP);
    if (bf_is_nan(T))
        bf_set_nan(r);
    else
        bf_exp_internal(r, T, prec1, NULL); /* no overflow/underlow test needed */
    bf_delete(T);
    return BF_ST_INEXACT;
}

/* x and y finite, x > 0, y integer and y fits on one limb */
static int bf_pow_int(bf_t *r, const bf_t *x, limb_t prec, void *opaque)
{
    bf_context_t *s = r->ctx;
    const bf_t *y = opaque;
    bf_t T_s, *T = &T_s;
    limb_t prec1;
    int ret;
    slimb_t y1;

    bf_get_limb(&y1, y, 0);
    if (y1 < 0)
        y1 = -y1;
    /* XXX: proof for the added precision */
    prec1 = prec + ceil_log2(y1) * 2 + 8;
    ret = bf_pow_ui(r, x, y1 < 0 ? -y1 : y1, prec1, BF_RNDN | BF_FLAG_EXT_EXP);
    if (y->sign) {
        bf_init(s, T);
        bf_set_ui(T, 1);
        ret |= bf_div(r, T, r, prec1, BF_RNDN | BF_FLAG_EXT_EXP);
        bf_delete(T);
    }
    return ret;
}

/* x must be a finite non zero float. Return TRUE if there is a
   floating point number r such as x=r^(2^n) and return this floating
   point number 'r'. Otherwise return FALSE and r is undefined. */
static BOOL check_exact_power2n(bf_t *r, const bf_t *x, slimb_t n)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;
    slimb_t e, i, er;
    limb_t v;

    /* x = m*2^e with m odd integer */
    e = bf_get_exp_min(x);
    /* fast check on the exponent */
    if (n > (LIMB_BITS - 1)) {
        if (e != 0)
            return FALSE;
        er = 0;
    } else {
        if ((e & (((limb_t)1 << n) - 1)) != 0)
            return FALSE;
        er = e >> n;
    }
    /* every perfect odd square = 1 modulo 8 */
    v = get_bits(x->tab, x->len, x->len * LIMB_BITS - x->expn + e);
    if ((v & 7) != 1)
        return FALSE;

    bf_init(s, T);
    bf_set(T, x);
    T->expn -= e;
    for(i = 0; i < n; i++) {
        if (i != 0)
            bf_set(T, r);
        if (bf_sqrtrem(r, NULL, T) != 0)
            return FALSE;
    }
    r->expn += er;
    return TRUE;
}

/* prec = BF_PREC_INF is accepted for x and y integers and y >= 0 */
int bf_pow(bf_t *r, const bf_t *x, const bf_t *y, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;
    bf_t ytmp_s;
    BOOL y_is_int, y_is_odd;
    int r_sign, ret, rnd_mode;
    slimb_t y_emin;

    if (x->len == 0 || y->len == 0) {
        if (y->expn == BF_EXP_ZERO) {
            /* pow(x, 0) = 1 */
            bf_set_ui(r, 1);
        } else if (x->expn == BF_EXP_NAN) {
            bf_set_nan(r);
        } else {
            int cmp_x_abs_1;
            bf_set_ui(r, 1);
            cmp_x_abs_1 = bf_cmpu(x, r);
            if (cmp_x_abs_1 == 0 && (flags & BF_POW_JS_QUIRKS) &&
                (y->expn >= BF_EXP_INF)) {
                bf_set_nan(r);
            } else if (cmp_x_abs_1 == 0 &&
                       (!x->sign || y->expn != BF_EXP_NAN)) {
                /* pow(1, y) = 1 even if y = NaN */
                /* pow(-1, +/-inf) = 1 */
            } else if (y->expn == BF_EXP_NAN) {
                bf_set_nan(r);
            } else if (y->expn == BF_EXP_INF) {
                if (y->sign == (cmp_x_abs_1 > 0)) {
                    bf_set_zero(r, 0);
                } else {
                    bf_set_inf(r, 0);
                }
            } else {
                y_emin = bf_get_exp_min(y);
                y_is_odd = (y_emin == 0);
                if (y->sign == (x->expn == BF_EXP_ZERO)) {
                    bf_set_inf(r, y_is_odd & x->sign);
                    if (y->sign) {
                        /* pow(0, y) with y < 0 */
                        return BF_ST_DIVIDE_ZERO;
                    }
                } else {
                    bf_set_zero(r, y_is_odd & x->sign);
                }
            }
        }
        return 0;
    }
    bf_init(s, T);
    bf_set(T, x);
    y_emin = bf_get_exp_min(y);
    y_is_int = (y_emin >= 0);
    rnd_mode = flags & BF_RND_MASK;
    if (x->sign) {
        if (!y_is_int) {
            bf_set_nan(r);
            bf_delete(T);
            return BF_ST_INVALID_OP;
        }
        y_is_odd = (y_emin == 0);
        r_sign = y_is_odd;
        /* change the directed rounding mode if the sign of the result
           is changed */
        if (r_sign && (rnd_mode == BF_RNDD || rnd_mode == BF_RNDU))
            flags ^= 1;
        bf_neg(T);
    } else {
        r_sign = 0;
    }

    bf_set_ui(r, 1);
    if (bf_cmp_eq(T, r)) {
        /* abs(x) = 1: nothing more to do */
        ret = 0;
    } else {
        /* check the overflow/underflow cases */
        {
            bf_t al_s, *al = &al_s;
            bf_t ah_s, *ah = &ah_s;
            limb_t precl = LIMB_BITS;

            bf_init(s, al);
            bf_init(s, ah);
            /* compute bounds of log(abs(x)) * y with a low precision */
            /* XXX: compute bf_log() once */
            /* XXX: add a fast test before this slow test */
            bf_log(al, T, precl, BF_RNDD);
            bf_log(ah, T, precl, BF_RNDU);
            bf_mul(al, al, y, precl, BF_RNDD ^ y->sign);
            bf_mul(ah, ah, y, precl, BF_RNDU ^ y->sign);
            ret = check_exp_underflow_overflow(s, r, al, ah, prec, flags);
            bf_delete(al);
            bf_delete(ah);
            if (ret)
                goto done;
        }

        if (y_is_int) {
            slimb_t T_bits, e;
        int_pow:
            T_bits = T->expn - bf_get_exp_min(T);
            if (T_bits == 1) {
                /* pow(2^b, y) = 2^(b*y) */
                bf_mul_si(T, y, T->expn - 1, LIMB_BITS, BF_RNDZ);
                bf_get_limb(&e, T, 0);
                bf_set_ui(r, 1);
                ret = bf_mul_2exp(r, e, prec, flags);
            } else if (prec == BF_PREC_INF) {
                slimb_t y1;
                /* specific case for infinite precision (integer case) */
                bf_get_limb(&y1, y, 0);
                assert(!y->sign);
                /* x must be an integer, so abs(x) >= 2 */
                if (y1 >= ((slimb_t)1 << BF_EXP_BITS_MAX)) {
                    bf_delete(T);
                    return bf_set_overflow(r, 0, BF_PREC_INF, flags);
                }
                ret = bf_pow_ui(r, T, y1, BF_PREC_INF, BF_RNDZ);
            } else {
                if (y->expn <= 31) {
                    /* small enough power: use exponentiation in all cases */
                } else if (y->sign) {
                    /* cannot be exact */
                    goto general_case;
                } else {
                    if (rnd_mode == BF_RNDF)
                        goto general_case; /* no need to track exact results */
                    /* see if the result has a chance to be exact:
                       if x=a*2^b (a odd), x^y=a^y*2^(b*y)
                       x^y needs a precision of at least floor_log2(a)*y bits
                    */
                    bf_mul_si(r, y, T_bits - 1, LIMB_BITS, BF_RNDZ);
                    bf_get_limb(&e, r, 0);
                    if (prec < e)
                        goto general_case;
                }
                ret = bf_ziv_rounding(r, T, prec, flags, bf_pow_int, (void *)y);
            }
        } else {
            if (rnd_mode != BF_RNDF) {
                bf_t *y1;
                if (y_emin < 0 && check_exact_power2n(r, T, -y_emin)) {
                    /* the problem is reduced to a power to an integer */
                    bf_set(T, r);
                    y1 = &ytmp_s;
                    y1->tab = y->tab;
                    y1->len = y->len;
                    y1->sign = y->sign;
                    y1->expn = y->expn - y_emin;
                    y = y1;
                    goto int_pow;
                }
            }
        general_case:
            ret = bf_ziv_rounding(r, T, prec, flags, bf_pow_generic, (void *)y);
        }
    }
 done:
    bf_delete(T);
    r->sign = r_sign;
    return ret;
}

/* compute sqrt(-2*x-x^2) to get |sin(x)| from cos(x) - 1. */
static void bf_sqrt_sin(bf_t *r, const bf_t *x, limb_t prec1)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;
    bf_init(s, T);
    bf_set(T, x);
    bf_mul(r, T, T, prec1, BF_RNDN);
    bf_mul_2exp(T, 1, BF_PREC_INF, BF_RNDZ);
    bf_add(T, T, r, prec1, BF_RNDN);
    bf_neg(T);
    bf_sqrt(r, T, prec1, BF_RNDF);
    bf_delete(T);
}

static int bf_sincos(bf_t *s, bf_t *c, const bf_t *a, limb_t prec)
{
    bf_context_t *s1 = a->ctx;
    bf_t T_s, *T = &T_s;
    bf_t U_s, *U = &U_s;
    bf_t r_s, *r = &r_s;
    slimb_t K, prec1, i, l, mod, prec2;
    int is_neg;

    assert(c != a && s != a);

    bf_init(s1, T);
    bf_init(s1, U);
    bf_init(s1, r);

    /* XXX: precision analysis */
    K = bf_isqrt(prec / 2);
    l = prec / (2 * K) + 1;
    prec1 = prec + 2 * K + l + 8;

    /* after the modulo reduction, -pi/4 <= T <= pi/4 */
    if (a->expn <= -1) {
        /* abs(a) <= 0.25: no modulo reduction needed */
        bf_set(T, a);
        mod = 0;
    } else {
        slimb_t cancel;
        cancel = 0;
        for(;;) {
            prec2 = prec1 + a->expn + cancel;
            bf_const_pi(U, prec2, BF_RNDF);
            bf_mul_2exp(U, -1, BF_PREC_INF, BF_RNDZ);
            bf_remquo(&mod, T, a, U, prec2, BF_RNDN, BF_RNDN);
            //            printf("T.expn=%ld prec2=%ld\n", T->expn, prec2);
            if (mod == 0 || (T->expn != BF_EXP_ZERO &&
                             (T->expn + prec2) >= (prec1 - 1)))
                break;
            /* increase the number of bits until the precision is good enough */
            cancel = bf_max(-T->expn, (cancel + 1) * 3 / 2);
        }
        mod &= 3;
    }

    is_neg = T->sign;

    /* compute cosm1(x) = cos(x) - 1 */
    bf_mul(T, T, T, prec1, BF_RNDN);
    bf_mul_2exp(T, -2 * K, BF_PREC_INF, BF_RNDZ);

    /* Taylor expansion:
       -x^2/2 + x^4/4! - x^6/6! + ...
    */
    bf_set_ui(r, 1);
    for(i = l ; i >= 1; i--) {
        bf_set_ui(U, 2 * i - 1);
        bf_mul_ui(U, U, 2 * i, BF_PREC_INF, BF_RNDZ);
        bf_div(U, T, U, prec1, BF_RNDN);
        bf_mul(r, r, U, prec1, BF_RNDN);
        bf_neg(r);
        if (i != 1)
            bf_add_si(r, r, 1, prec1, BF_RNDN);
    }
    bf_delete(U);

    /* undo argument reduction:
       cosm1(2*x)= 2*(2*cosm1(x)+cosm1(x)^2)
    */
    for(i = 0; i < K; i++) {
        bf_mul(T, r, r, prec1, BF_RNDN);
        bf_mul_2exp(r, 1, BF_PREC_INF, BF_RNDZ);
        bf_add(r, r, T, prec1, BF_RNDN);
        bf_mul_2exp(r, 1, BF_PREC_INF, BF_RNDZ);
    }
    bf_delete(T);

    if (c) {
        if ((mod & 1) == 0) {
            bf_add_si(c, r, 1, prec1, BF_RNDN);
        } else {
            bf_sqrt_sin(c, r, prec1);
            c->sign = is_neg ^ 1;
        }
        c->sign ^= mod >> 1;
    }
    if (s) {
        if ((mod & 1) == 0) {
            bf_sqrt_sin(s, r, prec1);
            s->sign = is_neg;
        } else {
            bf_add_si(s, r, 1, prec1, BF_RNDN);
        }
        s->sign ^= mod >> 1;
    }
    bf_delete(r);
    return BF_ST_INEXACT;
}

static int bf_cos_internal(bf_t *r, const bf_t *a, limb_t prec, void *opaque)
{
    return bf_sincos(NULL, r, a, prec);
}

int bf_cos(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags)
{
    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF) {
            bf_set_nan(r);
            return BF_ST_INVALID_OP;
        } else {
            bf_set_ui(r, 1);
            return 0;
        }
    }

    /* small argument case: result = 1+r(x) with r(x) = -x^2/2 +
       O(X^4). We assume r(x) < 2^(2*EXP(x) - 1). */
    if (a->expn < 0) {
        slimb_t e;
        e = 2 * a->expn - 1;
        if (e < -(prec + 2)) {
            bf_set_ui(r, 1);
            return bf_add_epsilon(r, r, e, 1, prec, flags);
        }
    }

    return bf_ziv_rounding(r, a, prec, flags, bf_cos_internal, NULL);
}

static int bf_sin_internal(bf_t *r, const bf_t *a, limb_t prec, void *opaque)
{
    return bf_sincos(r, NULL, a, prec);
}

int bf_sin(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags)
{
    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF) {
            bf_set_nan(r);
            return BF_ST_INVALID_OP;
        } else {
            bf_set_zero(r, a->sign);
            return 0;
        }
    }

    /* small argument case: result = x+r(x) with r(x) = -x^3/6 +
       O(X^5). We assume r(x) < 2^(3*EXP(x) - 2). */
    if (a->expn < 0) {
        slimb_t e;
        e = sat_add(2 * a->expn, a->expn - 2);
        if (e < a->expn - bf_max(prec + 2, a->len * LIMB_BITS + 2)) {
            bf_set(r, a);
            return bf_add_epsilon(r, r, e, 1 - a->sign, prec, flags);
        }
    }

    return bf_ziv_rounding(r, a, prec, flags, bf_sin_internal, NULL);
}

static int bf_tan_internal(bf_t *r, const bf_t *a, limb_t prec, void *opaque)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;
    limb_t prec1;

    /* XXX: precision analysis */
    prec1 = prec + 8;
    bf_init(s, T);
    bf_sincos(r, T, a, prec1);
    bf_div(r, r, T, prec1, BF_RNDF);
    bf_delete(T);
    return BF_ST_INEXACT;
}

int bf_tan(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags)
{
    assert(r != a);
    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF) {
            bf_set_nan(r);
            return BF_ST_INVALID_OP;
        } else {
            bf_set_zero(r, a->sign);
            return 0;
        }
    }

    /* small argument case: result = x+r(x) with r(x) = x^3/3 +
       O(X^5). We assume r(x) < 2^(3*EXP(x) - 1). */
    if (a->expn < 0) {
        slimb_t e;
        e = sat_add(2 * a->expn, a->expn - 1);
        if (e < a->expn - bf_max(prec + 2, a->len * LIMB_BITS + 2)) {
            bf_set(r, a);
            return bf_add_epsilon(r, r, e, a->sign, prec, flags);
        }
    }

    return bf_ziv_rounding(r, a, prec, flags, bf_tan_internal, NULL);
}

/* if add_pi2 is true, add pi/2 to the result (used for acos(x) to
   avoid cancellation) */
static int bf_atan_internal(bf_t *r, const bf_t *a, limb_t prec,
                            void *opaque)
{
    bf_context_t *s = r->ctx;
    BOOL add_pi2 = (BOOL)(intptr_t)opaque;
    bf_t T_s, *T = &T_s;
    bf_t U_s, *U = &U_s;
    bf_t V_s, *V = &V_s;
    bf_t X2_s, *X2 = &X2_s;
    int cmp_1;
    slimb_t prec1, i, K, l;

    /* XXX: precision analysis */
    K = bf_isqrt((prec + 1) / 2);
    l = prec / (2 * K) + 1;
    prec1 = prec + K + 2 * l + 32;
    //    printf("prec=%d K=%d l=%d prec1=%d\n", (int)prec, (int)K, (int)l, (int)prec1);

    bf_init(s, T);
    cmp_1 = (a->expn >= 1); /* a >= 1 */
    if (cmp_1) {
        bf_set_ui(T, 1);
        bf_div(T, T, a, prec1, BF_RNDN);
    } else {
        bf_set(T, a);
    }

    /* abs(T) <= 1 */

    /* argument reduction */

    bf_init(s, U);
    bf_init(s, V);
    bf_init(s, X2);
    for(i = 0; i < K; i++) {
        /* T = T / (1 + sqrt(1 + T^2)) */
        bf_mul(U, T, T, prec1, BF_RNDN);
        bf_add_si(U, U, 1, prec1, BF_RNDN);
        bf_sqrt(V, U, prec1, BF_RNDN);
        bf_add_si(V, V, 1, prec1, BF_RNDN);
        bf_div(T, T, V, prec1, BF_RNDN);
    }

    /* Taylor series:
       x - x^3/3 + ... + (-1)^ l * y^(2*l + 1) / (2*l+1)
    */
    bf_mul(X2, T, T, prec1, BF_RNDN);
    bf_set_ui(r, 0);
    for(i = l; i >= 1; i--) {
        bf_set_si(U, 1);
        bf_set_ui(V, 2 * i + 1);
        bf_div(U, U, V, prec1, BF_RNDN);
        bf_neg(r);
        bf_add(r, r, U, prec1, BF_RNDN);
        bf_mul(r, r, X2, prec1, BF_RNDN);
    }
    bf_neg(r);
    bf_add_si(r, r, 1, prec1, BF_RNDN);
    bf_mul(r, r, T, prec1, BF_RNDN);

    /* undo the argument reduction */
    bf_mul_2exp(r, K, BF_PREC_INF, BF_RNDZ);

    bf_delete(U);
    bf_delete(V);
    bf_delete(X2);

    i = add_pi2;
    if (cmp_1 > 0) {
        /* undo the inversion : r = sign(a)*PI/2 - r */
        bf_neg(r);
        i += 1 - 2 * a->sign;
    }
    /* add i*(pi/2) with -1 <= i <= 2 */
    if (i != 0) {
        bf_const_pi(T, prec1, BF_RNDF);
        if (i != 2)
            bf_mul_2exp(T, -1, BF_PREC_INF, BF_RNDZ);
        T->sign = (i < 0);
        bf_add(r, T, r, prec1, BF_RNDN);
    }

    bf_delete(T);
    return BF_ST_INEXACT;
}

int bf_atan(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;
    int res;

    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF)  {
            /* -PI/2 or PI/2 */
            bf_const_pi_signed(r, a->sign, prec, flags);
            bf_mul_2exp(r, -1, BF_PREC_INF, BF_RNDZ);
            return BF_ST_INEXACT;
        } else {
            bf_set_zero(r, a->sign);
            return 0;
        }
    }

    bf_init(s, T);
    bf_set_ui(T, 1);
    res = bf_cmpu(a, T);
    bf_delete(T);
    if (res == 0) {
        /* short cut: abs(a) == 1 -> +/-pi/4 */
        bf_const_pi_signed(r, a->sign, prec, flags);
        bf_mul_2exp(r, -2, BF_PREC_INF, BF_RNDZ);
        return BF_ST_INEXACT;
    }

    /* small argument case: result = x+r(x) with r(x) = -x^3/3 +
       O(X^5). We assume r(x) < 2^(3*EXP(x) - 1). */
    if (a->expn < 0) {
        slimb_t e;
        e = sat_add(2 * a->expn, a->expn - 1);
        if (e < a->expn - bf_max(prec + 2, a->len * LIMB_BITS + 2)) {
            bf_set(r, a);
            return bf_add_epsilon(r, r, e, 1 - a->sign, prec, flags);
        }
    }

    return bf_ziv_rounding(r, a, prec, flags, bf_atan_internal, (void *)FALSE);
}

static int bf_atan2_internal(bf_t *r, const bf_t *y, limb_t prec, void *opaque)
{
    bf_context_t *s = r->ctx;
    const bf_t *x = opaque;
    bf_t T_s, *T = &T_s;
    limb_t prec1;
    int ret;

    if (y->expn == BF_EXP_NAN || x->expn == BF_EXP_NAN) {
        bf_set_nan(r);
        return 0;
    }

    /* compute atan(y/x) assumming inf/inf = 1 and 0/0 = 0 */
    bf_init(s, T);
    prec1 = prec + 32;
    if (y->expn == BF_EXP_INF && x->expn == BF_EXP_INF) {
        bf_set_ui(T, 1);
        T->sign = y->sign ^ x->sign;
    } else if (y->expn == BF_EXP_ZERO && x->expn == BF_EXP_ZERO) {
        bf_set_zero(T, y->sign ^ x->sign);
    } else {
        bf_div(T, y, x, prec1, BF_RNDF);
    }
    ret = bf_atan(r, T, prec1, BF_RNDF);

    if (x->sign) {
        /* if x < 0 (it includes -0), return sign(y)*pi + atan(y/x) */
        bf_const_pi(T, prec1, BF_RNDF);
        T->sign = y->sign;
        bf_add(r, r, T, prec1, BF_RNDN);
        ret |= BF_ST_INEXACT;
    }

    bf_delete(T);
    return ret;
}

int bf_atan2(bf_t *r, const bf_t *y, const bf_t *x,
             limb_t prec, bf_flags_t flags)
{
    return bf_ziv_rounding(r, y, prec, flags, bf_atan2_internal, (void *)x);
}

static int bf_asin_internal(bf_t *r, const bf_t *a, limb_t prec, void *opaque)
{
    bf_context_t *s = r->ctx;
    BOOL is_acos = (BOOL)(intptr_t)opaque;
    bf_t T_s, *T = &T_s;
    limb_t prec1, prec2;

    /* asin(x) = atan(x/sqrt(1-x^2))
       acos(x) = pi/2 - asin(x) */
    prec1 = prec + 8;
    /* increase the precision in x^2 to compensate the cancellation in
       (1-x^2) if x is close to 1 */
    /* XXX: use less precision when possible */
    if (a->expn >= 0)
        prec2 = BF_PREC_INF;
    else
        prec2 = prec1;
    bf_init(s, T);
    bf_mul(T, a, a, prec2, BF_RNDN);
    bf_neg(T);
    bf_add_si(T, T, 1, prec2, BF_RNDN);

    bf_sqrt(r, T, prec1, BF_RNDN);
    bf_div(T, a, r, prec1, BF_RNDN);
    if (is_acos)
        bf_neg(T);
    bf_atan_internal(r, T, prec1, (void *)(intptr_t)is_acos);
    bf_delete(T);
    return BF_ST_INEXACT;
}

int bf_asin(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;
    int res;

    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF) {
            bf_set_nan(r);
            return BF_ST_INVALID_OP;
        } else {
            bf_set_zero(r, a->sign);
            return 0;
        }
    }
    bf_init(s, T);
    bf_set_ui(T, 1);
    res = bf_cmpu(a, T);
    bf_delete(T);
    if (res > 0) {
        bf_set_nan(r);
        return BF_ST_INVALID_OP;
    }

    /* small argument case: result = x+r(x) with r(x) = x^3/6 +
       O(X^5). We assume r(x) < 2^(3*EXP(x) - 2). */
    if (a->expn < 0) {
        slimb_t e;
        e = sat_add(2 * a->expn, a->expn - 2);
        if (e < a->expn - bf_max(prec + 2, a->len * LIMB_BITS + 2)) {
            bf_set(r, a);
            return bf_add_epsilon(r, r, e, a->sign, prec, flags);
        }
    }

    return bf_ziv_rounding(r, a, prec, flags, bf_asin_internal, (void *)FALSE);
}

int bf_acos(bf_t *r, const bf_t *a, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = r->ctx;
    bf_t T_s, *T = &T_s;
    int res;

    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bf_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF) {
            bf_set_nan(r);
            return BF_ST_INVALID_OP;
        } else {
            bf_const_pi(r, prec, flags);
            bf_mul_2exp(r, -1, BF_PREC_INF, BF_RNDZ);
            return BF_ST_INEXACT;
        }
    }
    bf_init(s, T);
    bf_set_ui(T, 1);
    res = bf_cmpu(a, T);
    bf_delete(T);
    if (res > 0) {
        bf_set_nan(r);
        return BF_ST_INVALID_OP;
    } else if (res == 0 && a->sign == 0) {
        bf_set_zero(r, 0);
        return 0;
    }

    return bf_ziv_rounding(r, a, prec, flags, bf_asin_internal, (void *)TRUE);
}

/***************************************************************/
/* decimal floating point numbers */

#ifdef USE_BF_DEC

#define adddq(r1, r0, a1, a0)                   \
    do {                                        \
        limb_t __t = r0;                        \
        r0 += (a0);                             \
        r1 += (a1) + (r0 < __t);                \
    } while (0)

#define subdq(r1, r0, a1, a0)                   \
    do {                                        \
        limb_t __t = r0;                        \
        r0 -= (a0);                             \
        r1 -= (a1) + (r0 > __t);                \
    } while (0)

#if LIMB_BITS == 64

/* Note: we assume __int128 is available */
/* uint128_t defined in libbf.h          */
#define muldq(r1, r0, a, b)                     \
    do {                                        \
        uint128_t __t;                          \
        __t = (uint128_t)(a) * (uint128_t)(b);  \
        r0 = __t;                               \
        r1 = __t >> 64;                         \
    } while (0)

#define divdq(q, r, a1, a0, b)                  \
    do {                                        \
        uint128_t __t;                  \
        limb_t __b = (b);                       \
        __t = ((uint128_t)(a1) << 64) | (a0);   \
        q = __t / __b;                                  \
        r = __t % __b;                                  \
    } while (0)

#else

#define muldq(r1, r0, a, b)                     \
    do {                                        \
        uint64_t __t;                          \
        __t = (uint64_t)(a) * (uint64_t)(b);  \
        r0 = __t;                               \
        r1 = __t >> 32;                         \
    } while (0)

#define divdq(q, r, a1, a0, b)                  \
    do {                                        \
        uint64_t __t;                  \
        limb_t __b = (b);                       \
        __t = ((uint64_t)(a1) << 32) | (a0);   \
        q = __t / __b;                                  \
        r = __t % __b;                                  \
    } while (0)

#endif /* LIMB_BITS != 64 */

#if LIMB_DIGITS == 19

/* WARNING: hardcoded for b = 1e19. It is assumed that:
   0 <= a1 < 2^63 */
#define divdq_base(q, r, a1, a0)\
do {\
    uint64_t __a0, __a1, __t0, __t1, __b = BF_DEC_BASE; \
    __a0 = a0;\
    __a1 = a1;\
    __t0 = __a1;\
    __t0 = shld(__t0, __a0, 1);\
    muldq(q, __t1, __t0, UINT64_C(17014118346046923173)); \
    muldq(__t1, __t0, q, __b);\
    subdq(__a1, __a0, __t1, __t0);\
    subdq(__a1, __a0, 1, __b * 2);    \
    __t0 = (slimb_t)__a1 >> 1; \
    q += 2 + __t0;\
    adddq(__a1, __a0, 0, __b & __t0);\
    q += __a1;                  \
    __a0 += __b & __a1;           \
    r = __a0;\
} while(0)

#elif LIMB_DIGITS == 9

/* WARNING: hardcoded for b = 1e9. It is assumed that:
   0 <= a1 < 2^29 */
#define divdq_base(q, r, a1, a0)\
do {\
    uint32_t __t0, __t1, __b = BF_DEC_BASE; \
    __t0 = a1;\
    __t1 = a0;\
    __t0 = (__t0 << 3) | (__t1 >> (32 - 3));    \
    muldq(q, __t1, __t0, 2305843009U);\
    r = a0 - q * __b;\
    __t1 = (r >= __b);\
    q += __t1;\
    if (__t1)\
        r -= __b;\
} while(0)

#endif

/* fast integer division by a fixed constant */

typedef struct FastDivData {
    limb_t m1; /* multiplier */
    int8_t shift1;
    int8_t shift2;
} FastDivData;

/* From "Division by Invariant Integers using Multiplication" by
   Torborn Granlund and Peter L. Montgomery */
/* d must be != 0 */
static inline __maybe_unused void fast_udiv_init(FastDivData *s, limb_t d)
{
    int l;
    limb_t q, r, m1;
    if (d == 1)
        l = 0;
    else
        l = 64 - clz64(d - 1);
    divdq(q, r, ((limb_t)1 << l) - d, 0, d);
    (void)r;
    m1 = q + 1;
    //    printf("d=%lu l=%d m1=0x%016lx\n", d, l, m1);
    s->m1 = m1;
    s->shift1 = l;
    if (s->shift1 > 1)
        s->shift1 = 1;
    s->shift2 = l - 1;
    if (s->shift2 < 0)
        s->shift2 = 0;
}

static inline limb_t fast_udiv(limb_t a, const FastDivData *s)
{
    limb_t t0, t1;
    muldq(t1, t0, s->m1, a);
    t0 = (a - t1) >> s->shift1;
    return (t1 + t0) >> s->shift2;
}

/* contains 10^i */
const limb_t mp_pow_dec[LIMB_DIGITS + 1] = {
    1U,
    10U,
    100U,
    1000U,
    10000U,
    100000U,
    1000000U,
    10000000U,
    100000000U,
    1000000000U,
#if LIMB_BITS == 64
    10000000000U,
    100000000000U,
    1000000000000U,
    10000000000000U,
    100000000000000U,
    1000000000000000U,
    10000000000000000U,
    100000000000000000U,
    1000000000000000000U,
    10000000000000000000U,
#endif
};

/* precomputed from fast_udiv_init(10^i) */
static const FastDivData mp_pow_div[LIMB_DIGITS + 1] = {
#if LIMB_BITS == 32
    { 0x00000001, 0, 0 },
    { 0x9999999a, 1, 3 },
    { 0x47ae147b, 1, 6 },
    { 0x0624dd30, 1, 9 },
    { 0xa36e2eb2, 1, 13 },
    { 0x4f8b588f, 1, 16 },
    { 0x0c6f7a0c, 1, 19 },
    { 0xad7f29ac, 1, 23 },
    { 0x5798ee24, 1, 26 },
    { 0x12e0be83, 1, 29 },
#else
    { 0x0000000000000001, 0, 0 },
    { 0x999999999999999a, 1, 3 },
    { 0x47ae147ae147ae15, 1, 6 },
    { 0x0624dd2f1a9fbe77, 1, 9 },
    { 0xa36e2eb1c432ca58, 1, 13 },
    { 0x4f8b588e368f0847, 1, 16 },
    { 0x0c6f7a0b5ed8d36c, 1, 19 },
    { 0xad7f29abcaf48579, 1, 23 },
    { 0x5798ee2308c39dfa, 1, 26 },
    { 0x12e0be826d694b2f, 1, 29 },
    { 0xb7cdfd9d7bdbab7e, 1, 33 },
    { 0x5fd7fe17964955fe, 1, 36 },
    { 0x19799812dea11198, 1, 39 },
    { 0xc25c268497681c27, 1, 43 },
    { 0x6849b86a12b9b01f, 1, 46 },
    { 0x203af9ee756159b3, 1, 49 },
    { 0xcd2b297d889bc2b7, 1, 53 },
    { 0x70ef54646d496893, 1, 56 },
    { 0x2725dd1d243aba0f, 1, 59 },
    { 0xd83c94fb6d2ac34d, 1, 63 },
#endif
};

/* divide by 10^shift with 0 <= shift <= LIMB_DIGITS */
static inline limb_t fast_shr_dec(limb_t a, int shift)
{
    return fast_udiv(a, &mp_pow_div[shift]);
}

/* division and remainder by 10^shift */
#define fast_shr_rem_dec(q, r, a, shift) q = fast_shr_dec(a, shift), r = a - q * mp_pow_dec[shift]

limb_t mp_add_dec(limb_t *res, const limb_t *op1, const limb_t *op2,
                  mp_size_t n, limb_t carry)
{
    limb_t base = BF_DEC_BASE;
    mp_size_t i;
    limb_t k, a, v;

    k=carry;
    for(i=0;i<n;i++) {
        /* XXX: reuse the trick in add_mod */
        v = op1[i];
        a = v + op2[i] + k - base;
        k = a <= v;
        if (!k)
            a += base;
        res[i]=a;
    }
    return k;
}

limb_t mp_add_ui_dec(limb_t *tab, limb_t b, mp_size_t n)
{
    limb_t base = BF_DEC_BASE;
    mp_size_t i;
    limb_t k, a, v;

    k=b;
    for(i=0;i<n;i++) {
        v = tab[i];
        a = v + k - base;
        k = a <= v;
        if (!k)
            a += base;
        tab[i] = a;
        if (k == 0)
            break;
    }
    return k;
}

limb_t mp_sub_dec(limb_t *res, const limb_t *op1, const limb_t *op2,
                  mp_size_t n, limb_t carry)
{
    limb_t base = BF_DEC_BASE;
    mp_size_t i;
    limb_t k, v, a;

    k=carry;
    for(i=0;i<n;i++) {
        v = op1[i];
        a = v - op2[i] - k;
        k = a > v;
        if (k)
            a += base;
        res[i] = a;
    }
    return k;
}

limb_t mp_sub_ui_dec(limb_t *tab, limb_t b, mp_size_t n)
{
    limb_t base = BF_DEC_BASE;
    mp_size_t i;
    limb_t k, v, a;

    k=b;
    for(i=0;i<n;i++) {
        v = tab[i];
        a = v - k;
        k = a > v;
        if (k)
            a += base;
        tab[i]=a;
        if (k == 0)
            break;
    }
    return k;
}

/* taba[] = taba[] * b + l. 0 <= b, l <= base - 1. Return the high carry */
limb_t mp_mul1_dec(limb_t *tabr, const limb_t *taba, mp_size_t n,
                   limb_t b, limb_t l)
{
    mp_size_t i;
    limb_t t0, t1, r;

    for(i = 0; i < n; i++) {
        muldq(t1, t0, taba[i], b);
        adddq(t1, t0, 0, l);
        divdq_base(l, r, t1, t0);
        tabr[i] = r;
    }
    return l;
}

/* tabr[] += taba[] * b. 0 <= b <= base - 1. Return the value to add
   to the high word */
limb_t mp_add_mul1_dec(limb_t *tabr, const limb_t *taba, mp_size_t n,
                       limb_t b)
{
    mp_size_t i;
    limb_t l, t0, t1, r;

    l = 0;
    for(i = 0; i < n; i++) {
        muldq(t1, t0, taba[i], b);
        adddq(t1, t0, 0, l);
        adddq(t1, t0, 0, tabr[i]);
        divdq_base(l, r, t1, t0);
        tabr[i] = r;
    }
    return l;
}

/* tabr[] -= taba[] * b. 0 <= b <= base - 1. Return the value to
   substract to the high word. */
limb_t mp_sub_mul1_dec(limb_t *tabr, const limb_t *taba, mp_size_t n,
                       limb_t b)
{
    limb_t base = BF_DEC_BASE;
    mp_size_t i;
    limb_t l, t0, t1, r, a, v, c;

    /* XXX: optimize */
    l = 0;
    for(i = 0; i < n; i++) {
        muldq(t1, t0, taba[i], b);
        adddq(t1, t0, 0, l);
        divdq_base(l, r, t1, t0);
        v = tabr[i];
        a = v - r;
        c = a > v;
        if (c)
            a += base;
        /* never bigger than base because r = 0 when l = base - 1 */
        l += c;
        tabr[i] = a;
    }
    return l;
}

/* size of the result : op1_size + op2_size. */
void mp_mul_basecase_dec(limb_t *result,
                         const limb_t *op1, mp_size_t op1_size,
                         const limb_t *op2, mp_size_t op2_size)
{
    mp_size_t i;
    limb_t r;

    result[op1_size] = mp_mul1_dec(result, op1, op1_size, op2[0], 0);

    for(i=1;i<op2_size;i++) {
        r = mp_add_mul1_dec(result + i, op1, op1_size, op2[i]);
        result[i + op1_size] = r;
    }
}

/* taba[] = (taba[] + r*base^na) / b. 0 <= b < base. 0 <= r <
   b. Return the remainder. */
limb_t mp_div1_dec(limb_t *tabr, const limb_t *taba, mp_size_t na,
                   limb_t b, limb_t r)
{
    limb_t base = BF_DEC_BASE;
    mp_size_t i;
    limb_t t0, t1, q;
    int shift;

#if (BF_DEC_BASE % 2) == 0
    if (b == 2) {
        limb_t base_div2;
        /* Note: only works if base is even */
        base_div2 = base >> 1;
        if (r)
            r = base_div2;
        for(i = na - 1; i >= 0; i--) {
            t0 = taba[i];
            tabr[i] = (t0 >> 1) + r;
            r = 0;
            if (t0 & 1)
                r = base_div2;
        }
        if (r)
            r = 1;
    } else
#endif
    if (na >= UDIV1NORM_THRESHOLD) {
        shift = clz(b);
        if (shift == 0) {
            /* normalized case: b >= 2^(LIMB_BITS-1) */
            limb_t b_inv;
            b_inv = udiv1norm_init(b);
            for(i = na - 1; i >= 0; i--) {
                muldq(t1, t0, r, base);
                adddq(t1, t0, 0, taba[i]);
                q = udiv1norm(&r, t1, t0, b, b_inv);
                tabr[i] = q;
            }
        } else {
            limb_t b_inv;
            b <<= shift;
            b_inv = udiv1norm_init(b);
            for(i = na - 1; i >= 0; i--) {
                muldq(t1, t0, r, base);
                adddq(t1, t0, 0, taba[i]);
                t1 = (t1 << shift) | (t0 >> (LIMB_BITS - shift));
                t0 <<= shift;
                q = udiv1norm(&r, t1, t0, b, b_inv);
                r >>= shift;
                tabr[i] = q;
            }
        }
    } else {
        for(i = na - 1; i >= 0; i--) {
            muldq(t1, t0, r, base);
            adddq(t1, t0, 0, taba[i]);
            divdq(q, r, t1, t0, b);
            tabr[i] = q;
        }
    }
    return r;
}

static __maybe_unused void mp_print_str_dec(const char *str,
                                       const limb_t *tab, slimb_t n)
{
    slimb_t i;
    printf("%s=", str);
    for(i = n - 1; i >= 0; i--) {
        if (i != n - 1)
            printf("_");
        printf("%0*" PRIu_LIMB, LIMB_DIGITS, tab[i]);
    }
    printf("\n");
}

static __maybe_unused void mp_print_str_h_dec(const char *str,
                                              const limb_t *tab, slimb_t n,
                                              limb_t high)
{
    slimb_t i;
    printf("%s=", str);
    printf("%0*" PRIu_LIMB, LIMB_DIGITS, high);
    for(i = n - 1; i >= 0; i--) {
        printf("_");
        printf("%0*" PRIu_LIMB, LIMB_DIGITS, tab[i]);
    }
    printf("\n");
}

//#define DEBUG_DIV_SLOW

#define DIV_STATIC_ALLOC_LEN 16

/* return q = a / b and r = a % b.

   taba[na] must be allocated if tabb1[nb - 1] < B / 2.  tabb1[nb - 1]
   must be != zero. na must be >= nb. 's' can be NULL if tabb1[nb - 1]
   >= B / 2.

   The remainder is is returned in taba and contains nb libms. tabq
   contains na - nb + 1 limbs. No overlap is permitted.

   Running time of the standard method: (na - nb + 1) * nb
   Return 0 if OK, -1 if memory alloc error
*/
/* XXX: optimize */
static int mp_div_dec(bf_context_t *s, limb_t *tabq,
                      limb_t *taba, mp_size_t na,
                      const limb_t *tabb1, mp_size_t nb)
{
    limb_t base = BF_DEC_BASE;
    limb_t r, mult, t0, t1, a, c, q, v, *tabb;
    mp_size_t i, j;
    limb_t static_tabb[DIV_STATIC_ALLOC_LEN];

#ifdef DEBUG_DIV_SLOW
    mp_print_str_dec("a", taba, na);
    mp_print_str_dec("b", tabb1, nb);
#endif

    /* normalize tabb */
    r = tabb1[nb - 1];
    assert(r != 0);
    i = na - nb;
    if (r >= BF_DEC_BASE / 2) {
        mult = 1;
        tabb = (limb_t *)tabb1;
        q = 1;
        for(j = nb - 1; j >= 0; j--) {
            if (taba[i + j] != tabb[j]) {
                if (taba[i + j] < tabb[j])
                    q = 0;
                break;
            }
        }
        tabq[i] = q;
        if (q) {
            mp_sub_dec(taba + i, taba + i, tabb, nb, 0);
        }
        i--;
    } else {
        mult = base / (r + 1);
        if (likely(nb <= DIV_STATIC_ALLOC_LEN)) {
            tabb = static_tabb;
        } else {
            tabb = bf_malloc(s, sizeof(limb_t) * nb);
            if (!tabb)
                return -1;
        }
        mp_mul1_dec(tabb, tabb1, nb, mult, 0);
        taba[na] = mp_mul1_dec(taba, taba, na, mult, 0);
    }

#ifdef DEBUG_DIV_SLOW
    printf("mult=" FMT_LIMB "\n", mult);
    mp_print_str_dec("a_norm", taba, na + 1);
    mp_print_str_dec("b_norm", tabb, nb);
#endif

    for(; i >= 0; i--) {
        if (unlikely(taba[i + nb] >= tabb[nb - 1])) {
            /* XXX: check if it is really possible */
            q = base - 1;
        } else {
            muldq(t1, t0, taba[i + nb], base);
            adddq(t1, t0, 0, taba[i + nb - 1]);
            divdq(q, r, t1, t0, tabb[nb - 1]);
        }
        //        printf("i=%d q1=%ld\n", i, q);

        r = mp_sub_mul1_dec(taba + i, tabb, nb, q);
        //        mp_dump("r1", taba + i, nb, bd);
        //        printf("r2=%ld\n", r);

        v = taba[i + nb];
        a = v - r;
        c = a > v;
        if (c)
            a += base;
        taba[i + nb] = a;

        if (c != 0) {
            /* negative result */
            for(;;) {
                q--;
                c = mp_add_dec(taba + i, taba + i, tabb, nb, 0);
                /* propagate carry and test if positive result */
                if (c != 0) {
                    if (++taba[i + nb] == base) {
                        break;
                    }
                }
            }
        }
        tabq[i] = q;
    }

#ifdef DEBUG_DIV_SLOW
    mp_print_str_dec("q", tabq, na - nb + 1);
    mp_print_str_dec("r", taba, nb);
#endif

    /* remove the normalization */
    if (mult != 1) {
        mp_div1_dec(taba, taba, nb, mult, 0);
        if (unlikely(tabb != static_tabb))
            bf_free(s, tabb);
    }
    return 0;
}

/* divide by 10^shift */
static limb_t mp_shr_dec(limb_t *tab_r, const limb_t *tab, mp_size_t n,
                         limb_t shift, limb_t high)
{
    mp_size_t i;
    limb_t l, a, q, r;

    assert(shift >= 1 && shift < LIMB_DIGITS);
    l = high;
    for(i = n - 1; i >= 0; i--) {
        a = tab[i];
        fast_shr_rem_dec(q, r, a, shift);
        tab_r[i] = q + l * mp_pow_dec[LIMB_DIGITS - shift];
        l = r;
    }
    return l;
}

/* multiply by 10^shift */
static limb_t mp_shl_dec(limb_t *tab_r, const limb_t *tab, mp_size_t n,
                         limb_t shift, limb_t low)
{
    mp_size_t i;
    limb_t l, a, q, r;

    assert(shift >= 1 && shift < LIMB_DIGITS);
    l = low;
    for(i = 0; i < n; i++) {
        a = tab[i];
        fast_shr_rem_dec(q, r, a, LIMB_DIGITS - shift);
        tab_r[i] = r * mp_pow_dec[shift] + l;
        l = q;
    }
    return l;
}

static limb_t mp_sqrtrem2_dec(limb_t *tabs, limb_t *taba)
{
    int k;
    dlimb_t a, b, r;
    limb_t taba1[2], s, r0, r1;

    /* convert to binary and normalize */
    a = (dlimb_t)taba[1] * BF_DEC_BASE + taba[0];
    k = clz(a >> LIMB_BITS) & ~1;
    b = a << k;
    taba1[0] = b;
    taba1[1] = b >> LIMB_BITS;
    mp_sqrtrem2(&s, taba1);
    s >>= (k >> 1);
    /* convert the remainder back to decimal */
    r = a - (dlimb_t)s * (dlimb_t)s;
    divdq_base(r1, r0, r >> LIMB_BITS, r);
    taba[0] = r0;
    tabs[0] = s;
    return r1;
}

//#define DEBUG_SQRTREM_DEC

/* tmp_buf must contain (n / 2 + 1 limbs) */
static limb_t mp_sqrtrem_rec_dec(limb_t *tabs, limb_t *taba, limb_t n,
                                 limb_t *tmp_buf)
{
    limb_t l, h, rh, ql, qh, c, i;

    if (n == 1)
        return mp_sqrtrem2_dec(tabs, taba);
#ifdef DEBUG_SQRTREM_DEC
    mp_print_str_dec("a", taba, 2 * n);
#endif
    l = n / 2;
    h = n - l;
    qh = mp_sqrtrem_rec_dec(tabs + l, taba + 2 * l, h, tmp_buf);
#ifdef DEBUG_SQRTREM_DEC
    mp_print_str_dec("s1", tabs + l, h);
    mp_print_str_h_dec("r1", taba + 2 * l, h, qh);
    mp_print_str_h_dec("r2", taba + l, n, qh);
#endif

    /* the remainder is in taba + 2 * l. Its high bit is in qh */
    if (qh) {
        mp_sub_dec(taba + 2 * l, taba + 2 * l, tabs + l, h, 0);
    }
    /* instead of dividing by 2*s, divide by s (which is normalized)
       and update q and r */
    mp_div_dec(NULL, tmp_buf, taba + l, n, tabs + l, h);
    qh += tmp_buf[l];
    for(i = 0; i < l; i++)
        tabs[i] = tmp_buf[i];
    ql = mp_div1_dec(tabs, tabs, l, 2, qh & 1);
    qh = qh >> 1; /* 0 or 1 */
    if (ql)
        rh = mp_add_dec(taba + l, taba + l, tabs + l, h, 0);
    else
        rh = 0;
#ifdef DEBUG_SQRTREM_DEC
    mp_print_str_h_dec("q", tabs, l, qh);
    mp_print_str_h_dec("u", taba + l, h, rh);
#endif

    mp_add_ui_dec(tabs + l, qh, h);
#ifdef DEBUG_SQRTREM_DEC
    mp_print_str_dec("s2", tabs, n);
#endif

    /* q = qh, tabs[l - 1 ... 0], r = taba[n - 1 ... l] */
    /* subtract q^2. if qh = 1 then q = B^l, so we can take shortcuts */
    if (qh) {
        c = qh;
    } else {
        mp_mul_basecase_dec(taba + n, tabs, l, tabs, l);
        c = mp_sub_dec(taba, taba, taba + n, 2 * l, 0);
    }
    rh -= mp_sub_ui_dec(taba + 2 * l, c, n - 2 * l);
    if ((slimb_t)rh < 0) {
        mp_sub_ui_dec(tabs, 1, n);
        rh += mp_add_mul1_dec(taba, tabs, n, 2);
        rh += mp_add_ui_dec(taba, 1, n);
    }
    return rh;
}

/* 'taba' has 2*n limbs with n >= 1 and taba[2*n-1] >= B/4. Return (s,
   r) with s=floor(sqrt(a)) and r=a-s^2. 0 <= r <= 2 * s. tabs has n
   limbs. r is returned in the lower n limbs of taba. Its r[n] is the
   returned value of the function. */
int mp_sqrtrem_dec(bf_context_t *s, limb_t *tabs, limb_t *taba, limb_t n)
{
    limb_t tmp_buf1[8];
    limb_t *tmp_buf;
    mp_size_t n2;
    n2 = n / 2 + 1;
    if (n2 <= countof(tmp_buf1)) {
        tmp_buf = tmp_buf1;
    } else {
        tmp_buf = bf_malloc(s, sizeof(limb_t) * n2);
        if (!tmp_buf)
            return -1;
    }
    taba[n] = mp_sqrtrem_rec_dec(tabs, taba, n, tmp_buf);
    if (tmp_buf != tmp_buf1)
        bf_free(s, tmp_buf);
    return 0;
}

/* return the number of leading zero digits, from 0 to LIMB_DIGITS */
static int clz_dec(limb_t a)
{
    if (a == 0)
        return LIMB_DIGITS;
    switch(LIMB_BITS - 1 - clz(a)) {
    case 0: /* 1-1 */
        return LIMB_DIGITS - 1;
    case 1: /* 2-3 */
        return LIMB_DIGITS - 1;
    case 2: /* 4-7 */
        return LIMB_DIGITS - 1;
    case 3: /* 8-15 */
        if (a < 10)
            return LIMB_DIGITS - 1;
        else
            return LIMB_DIGITS - 2;
    case 4: /* 16-31 */
        return LIMB_DIGITS - 2;
    case 5: /* 32-63 */
        return LIMB_DIGITS - 2;
    case 6: /* 64-127 */
        if (a < 100)
            return LIMB_DIGITS - 2;
        else
            return LIMB_DIGITS - 3;
    case 7: /* 128-255 */
        return LIMB_DIGITS - 3;
    case 8: /* 256-511 */
        return LIMB_DIGITS - 3;
    case 9: /* 512-1023 */
        if (a < 1000)
            return LIMB_DIGITS - 3;
        else
            return LIMB_DIGITS - 4;
    case 10: /* 1024-2047 */
        return LIMB_DIGITS - 4;
    case 11: /* 2048-4095 */
        return LIMB_DIGITS - 4;
    case 12: /* 4096-8191 */
        return LIMB_DIGITS - 4;
    case 13: /* 8192-16383 */
        if (a < 10000)
            return LIMB_DIGITS - 4;
        else
            return LIMB_DIGITS - 5;
    case 14: /* 16384-32767 */
        return LIMB_DIGITS - 5;
    case 15: /* 32768-65535 */
        return LIMB_DIGITS - 5;
    case 16: /* 65536-131071 */
        if (a < 100000)
            return LIMB_DIGITS - 5;
        else
            return LIMB_DIGITS - 6;
    case 17: /* 131072-262143 */
        return LIMB_DIGITS - 6;
    case 18: /* 262144-524287 */
        return LIMB_DIGITS - 6;
    case 19: /* 524288-1048575 */
        if (a < 1000000)
            return LIMB_DIGITS - 6;
        else
            return LIMB_DIGITS - 7;
    case 20: /* 1048576-2097151 */
        return LIMB_DIGITS - 7;
    case 21: /* 2097152-4194303 */
        return LIMB_DIGITS - 7;
    case 22: /* 4194304-8388607 */
        return LIMB_DIGITS - 7;
    case 23: /* 8388608-16777215 */
        if (a < 10000000)
            return LIMB_DIGITS - 7;
        else
            return LIMB_DIGITS - 8;
    case 24: /* 16777216-33554431 */
        return LIMB_DIGITS - 8;
    case 25: /* 33554432-67108863 */
        return LIMB_DIGITS - 8;
    case 26: /* 67108864-134217727 */
        if (a < 100000000)
            return LIMB_DIGITS - 8;
        else
            return LIMB_DIGITS - 9;
#if LIMB_BITS == 64
    case 27: /* 134217728-268435455 */
        return LIMB_DIGITS - 9;
    case 28: /* 268435456-536870911 */
        return LIMB_DIGITS - 9;
    case 29: /* 536870912-1073741823 */
        if (a < 1000000000)
            return LIMB_DIGITS - 9;
        else
            return LIMB_DIGITS - 10;
    case 30: /* 1073741824-2147483647 */
        return LIMB_DIGITS - 10;
    case 31: /* 2147483648-4294967295 */
        return LIMB_DIGITS - 10;
    case 32: /* 4294967296-8589934591 */
        return LIMB_DIGITS - 10;
    case 33: /* 8589934592-17179869183 */
        if (a < 10000000000)
            return LIMB_DIGITS - 10;
        else
            return LIMB_DIGITS - 11;
    case 34: /* 17179869184-34359738367 */
        return LIMB_DIGITS - 11;
    case 35: /* 34359738368-68719476735 */
        return LIMB_DIGITS - 11;
    case 36: /* 68719476736-137438953471 */
        if (a < 100000000000)
            return LIMB_DIGITS - 11;
        else
            return LIMB_DIGITS - 12;
    case 37: /* 137438953472-274877906943 */
        return LIMB_DIGITS - 12;
    case 38: /* 274877906944-549755813887 */
        return LIMB_DIGITS - 12;
    case 39: /* 549755813888-1099511627775 */
        if (a < 1000000000000)
            return LIMB_DIGITS - 12;
        else
            return LIMB_DIGITS - 13;
    case 40: /* 1099511627776-2199023255551 */
        return LIMB_DIGITS - 13;
    case 41: /* 2199023255552-4398046511103 */
        return LIMB_DIGITS - 13;
    case 42: /* 4398046511104-8796093022207 */
        return LIMB_DIGITS - 13;
    case 43: /* 8796093022208-17592186044415 */
        if (a < 10000000000000)
            return LIMB_DIGITS - 13;
        else
            return LIMB_DIGITS - 14;
    case 44: /* 17592186044416-35184372088831 */
        return LIMB_DIGITS - 14;
    case 45: /* 35184372088832-70368744177663 */
        return LIMB_DIGITS - 14;
    case 46: /* 70368744177664-140737488355327 */
        if (a < 100000000000000)
            return LIMB_DIGITS - 14;
        else
            return LIMB_DIGITS - 15;
    case 47: /* 140737488355328-281474976710655 */
        return LIMB_DIGITS - 15;
    case 48: /* 281474976710656-562949953421311 */
        return LIMB_DIGITS - 15;
    case 49: /* 562949953421312-1125899906842623 */
        if (a < 1000000000000000)
            return LIMB_DIGITS - 15;
        else
            return LIMB_DIGITS - 16;
    case 50: /* 1125899906842624-2251799813685247 */
        return LIMB_DIGITS - 16;
    case 51: /* 2251799813685248-4503599627370495 */
        return LIMB_DIGITS - 16;
    case 52: /* 4503599627370496-9007199254740991 */
        return LIMB_DIGITS - 16;
    case 53: /* 9007199254740992-18014398509481983 */
        if (a < 10000000000000000)
            return LIMB_DIGITS - 16;
        else
            return LIMB_DIGITS - 17;
    case 54: /* 18014398509481984-36028797018963967 */
        return LIMB_DIGITS - 17;
    case 55: /* 36028797018963968-72057594037927935 */
        return LIMB_DIGITS - 17;
    case 56: /* 72057594037927936-144115188075855871 */
        if (a < 100000000000000000)
            return LIMB_DIGITS - 17;
        else
            return LIMB_DIGITS - 18;
    case 57: /* 144115188075855872-288230376151711743 */
        return LIMB_DIGITS - 18;
    case 58: /* 288230376151711744-576460752303423487 */
        return LIMB_DIGITS - 18;
    case 59: /* 576460752303423488-1152921504606846975 */
        if (a < 1000000000000000000)
            return LIMB_DIGITS - 18;
        else
            return LIMB_DIGITS - 19;
#endif
    default:
        return 0;
    }
}

/* for debugging */
void bfdec_print_str(const char *str, const bfdec_t *a)
{
    slimb_t i;
    printf("%s=", str);

    if (a->expn == BF_EXP_NAN) {
        printf("NaN");
    } else {
        if (a->sign)
            putchar('-');
        if (a->expn == BF_EXP_ZERO) {
            putchar('0');
        } else if (a->expn == BF_EXP_INF) {
            printf("Inf");
        } else {
            printf("0.");
            for(i = a->len - 1; i >= 0; i--)
                printf("%0*" PRIu_LIMB, LIMB_DIGITS, a->tab[i]);
            printf("e%" PRId_LIMB, a->expn);
        }
    }
    printf("\n");
}

/* return != 0 if one digit between 0 and bit_pos inclusive is not zero. */
static inline limb_t scan_digit_nz(const bfdec_t *r, slimb_t bit_pos)
{
    slimb_t pos;
    limb_t v, q;
    int shift;

    if (bit_pos < 0)
        return 0;
    pos = (limb_t)bit_pos / LIMB_DIGITS;
    shift = (limb_t)bit_pos % LIMB_DIGITS;
    fast_shr_rem_dec(q, v, r->tab[pos], shift + 1);
    (void)q;
    if (v != 0)
        return 1;
    pos--;
    while (pos >= 0) {
        if (r->tab[pos] != 0)
            return 1;
        pos--;
    }
    return 0;
}

static limb_t get_digit(const limb_t *tab, limb_t len, slimb_t pos)
{
    slimb_t i;
    int shift;
    i = floor_div(pos, LIMB_DIGITS);
    if (i < 0 || i >= len)
        return 0;
    shift = pos - i * LIMB_DIGITS;
    return fast_shr_dec(tab[i], shift) % 10;
}

/* return the addend for rounding. Note that prec can be <= 0 for bf_rint() */
static int bfdec_get_rnd_add(int *pret, const bfdec_t *r, limb_t l,
                             slimb_t prec, int rnd_mode)
{
    int add_one, inexact;
    limb_t digit1, digit0;

    //    bfdec_print_str("get_rnd_add", r);
    if (rnd_mode == BF_RNDF) {
        digit0 = 1; /* faithful rounding does not honor the INEXACT flag */
    } else {
        /* starting limb for bit 'prec + 1' */
        digit0 = scan_digit_nz(r, l * LIMB_DIGITS - 1 - bf_max(0, prec + 1));
    }

    /* get the digit at 'prec' */
    digit1 = get_digit(r->tab, l, l * LIMB_DIGITS - 1 - prec);
    inexact = (digit1 | digit0) != 0;

    add_one = 0;
    switch(rnd_mode) {
    case BF_RNDZ:
        break;
    case BF_RNDN:
        if (digit1 == 5) {
            if (digit0) {
                add_one = 1;
            } else {
                /* round to even */
                add_one =
                    get_digit(r->tab, l, l * LIMB_DIGITS - 1 - (prec - 1)) & 1;
            }
        } else if (digit1 > 5) {
            add_one = 1;
        }
        break;
    case BF_RNDD:
    case BF_RNDU:
        if (r->sign == (rnd_mode == BF_RNDD))
            add_one = inexact;
        break;
    case BF_RNDNA:
    case BF_RNDF:
        add_one = (digit1 >= 5);
        break;
    case BF_RNDA:
        add_one = inexact;
        break;
    default:
        abort();
    }

    if (inexact)
        *pret |= BF_ST_INEXACT;
    return add_one;
}

/* round to prec1 bits assuming 'r' is non zero and finite. 'r' is
   assumed to have length 'l' (1 <= l <= r->len). prec1 can be
   BF_PREC_INF. BF_FLAG_SUBNORMAL is not supported. Cannot fail with
   BF_ST_MEM_ERROR.
 */
static int __bfdec_round(bfdec_t *r, limb_t prec1, bf_flags_t flags, limb_t l)
{
    int shift, add_one, rnd_mode, ret;
    slimb_t i, bit_pos, pos, e_min, e_max, e_range, prec;

    /* XXX: align to IEEE 754 2008 for decimal numbers ? */
    e_range = (limb_t)1 << (bf_get_exp_bits(flags) - 1);
    e_min = -e_range + 3;
    e_max = e_range;

    if (flags & BF_FLAG_RADPNT_PREC) {
        /* 'prec' is the precision after the decimal point */
        if (prec1 != BF_PREC_INF)
            prec = r->expn + prec1;
        else
            prec = prec1;
    } else if (unlikely(r->expn < e_min) && (flags & BF_FLAG_SUBNORMAL)) {
        /* restrict the precision in case of potentially subnormal
           result */
        assert(prec1 != BF_PREC_INF);
        prec = prec1 - (e_min - r->expn);
    } else {
        prec = prec1;
    }

    /* round to prec bits */
    rnd_mode = flags & BF_RND_MASK;
    ret = 0;
    add_one = bfdec_get_rnd_add(&ret, r, l, prec, rnd_mode);

    if (prec <= 0) {
        if (add_one) {
            bfdec_resize(r, 1); /* cannot fail because r is non zero */
            r->tab[0] = BF_DEC_BASE / 10;
            r->expn += 1 - prec;
            ret |= BF_ST_UNDERFLOW | BF_ST_INEXACT;
            return ret;
        } else {
            goto underflow;
        }
    } else if (add_one) {
        limb_t carry;

        /* add one starting at digit 'prec - 1' */
        bit_pos = l * LIMB_DIGITS - 1 - (prec - 1);
        pos = bit_pos / LIMB_DIGITS;
        carry = mp_pow_dec[bit_pos % LIMB_DIGITS];
        carry = mp_add_ui_dec(r->tab + pos, carry, l - pos);
        if (carry) {
            /* shift right by one digit */
            mp_shr_dec(r->tab + pos, r->tab + pos, l - pos, 1, 1);
            r->expn++;
        }
    }

    /* check underflow */
    if (unlikely(r->expn < e_min)) {
        if (flags & BF_FLAG_SUBNORMAL) {
            /* if inexact, also set the underflow flag */
            if (ret & BF_ST_INEXACT)
                ret |= BF_ST_UNDERFLOW;
        } else {
        underflow:
            bfdec_set_zero(r, r->sign);
            ret |= BF_ST_UNDERFLOW | BF_ST_INEXACT;
            return ret;
        }
    }

    /* check overflow */
    if (unlikely(r->expn > e_max)) {
        bfdec_set_inf(r, r->sign);
        ret |= BF_ST_OVERFLOW | BF_ST_INEXACT;
        return ret;
    }

    /* keep the bits starting at 'prec - 1' */
    bit_pos = l * LIMB_DIGITS - 1 - (prec - 1);
    i = floor_div(bit_pos, LIMB_DIGITS);
    if (i >= 0) {
        shift = smod(bit_pos, LIMB_DIGITS);
        if (shift != 0) {
            r->tab[i] = fast_shr_dec(r->tab[i], shift) *
                mp_pow_dec[shift];
        }
    } else {
        i = 0;
    }
    /* remove trailing zeros */
    while (r->tab[i] == 0)
        i++;
    if (i > 0) {
        l -= i;
        memmove(r->tab, r->tab + i, l * sizeof(limb_t));
    }
    bfdec_resize(r, l); /* cannot fail */
    return ret;
}

/* Cannot fail with BF_ST_MEM_ERROR. */
int bfdec_round(bfdec_t *r, limb_t prec, bf_flags_t flags)
{
    if (r->len == 0)
        return 0;
    return __bfdec_round(r, prec, flags, r->len);
}

/* 'r' must be a finite number. Cannot fail with BF_ST_MEM_ERROR.  */
int bfdec_normalize_and_round(bfdec_t *r, limb_t prec1, bf_flags_t flags)
{
    limb_t l, v;
    int shift, ret;

    //    bfdec_print_str("bf_renorm", r);
    l = r->len;
    while (l > 0 && r->tab[l - 1] == 0)
        l--;
    if (l == 0) {
        /* zero */
        r->expn = BF_EXP_ZERO;
        bfdec_resize(r, 0); /* cannot fail */
        ret = 0;
    } else {
        r->expn -= (r->len - l) * LIMB_DIGITS;
        /* shift to have the MSB set to '1' */
        v = r->tab[l - 1];
        shift = clz_dec(v);
        if (shift != 0) {
            mp_shl_dec(r->tab, r->tab, l, shift, 0);
            r->expn -= shift;
        }
        ret = __bfdec_round(r, prec1, flags, l);
    }
    //    bf_print_str("r_final", r);
    return ret;
}

int bfdec_set_ui(bfdec_t *r, uint64_t v)
{
#if LIMB_BITS == 32
    if (v >= BF_DEC_BASE * BF_DEC_BASE) {
        if (bfdec_resize(r, 3))
            goto fail;
        r->tab[0] = v % BF_DEC_BASE;
        v /= BF_DEC_BASE;
        r->tab[1] = v % BF_DEC_BASE;
        r->tab[2] = v / BF_DEC_BASE;
        r->expn = 3 * LIMB_DIGITS;
    } else
#endif
    if (v >= BF_DEC_BASE) {
        if (bfdec_resize(r, 2))
            goto fail;
        r->tab[0] = v % BF_DEC_BASE;
        r->tab[1] = v / BF_DEC_BASE;
        r->expn = 2 * LIMB_DIGITS;
    } else {
        if (bfdec_resize(r, 1))
            goto fail;
        r->tab[0] = v;
        r->expn = LIMB_DIGITS;
    }
    r->sign = 0;
    return bfdec_normalize_and_round(r, BF_PREC_INF, 0);
 fail:
    bfdec_set_nan(r);
    return BF_ST_MEM_ERROR;
}

int bfdec_set_si(bfdec_t *r, int64_t v)
{
    int ret;
    if (v < 0) {
        ret = bfdec_set_ui(r, -v);
        r->sign = 1;
    } else {
        ret = bfdec_set_ui(r, v);
    }
    return ret;
}

static int bfdec_add_internal(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec, bf_flags_t flags, int b_neg)
{
    bf_context_t *s = r->ctx;
    int is_sub, cmp_res, a_sign, b_sign, ret;

    a_sign = a->sign;
    b_sign = b->sign ^ b_neg;
    is_sub = a_sign ^ b_sign;
    cmp_res = bfdec_cmpu(a, b);
    if (cmp_res < 0) {
        const bfdec_t *tmp;
        tmp = a;
        a = b;
        b = tmp;
        a_sign = b_sign; /* b_sign is never used later */
    }
    /* abs(a) >= abs(b) */
    if (cmp_res == 0 && is_sub && a->expn < BF_EXP_INF) {
        /* zero result */
        bfdec_set_zero(r, (flags & BF_RND_MASK) == BF_RNDD);
        ret = 0;
    } else if (a->len == 0 || b->len == 0) {
        ret = 0;
        if (a->expn >= BF_EXP_INF) {
            if (a->expn == BF_EXP_NAN) {
                /* at least one operand is NaN */
                bfdec_set_nan(r);
                ret = 0;
            } else if (b->expn == BF_EXP_INF && is_sub) {
                /* infinities with different signs */
                bfdec_set_nan(r);
                ret = BF_ST_INVALID_OP;
            } else {
                bfdec_set_inf(r, a_sign);
            }
        } else {
            /* at least one zero and not subtract */
            if (bfdec_set(r, a))
                return BF_ST_MEM_ERROR;
            r->sign = a_sign;
            goto renorm;
        }
    } else {
        slimb_t d, a_offset, b_offset, i, r_len;
        limb_t carry;
        limb_t *b1_tab;
        int b_shift;
        mp_size_t b1_len;

        d = a->expn - b->expn;

        /* XXX: not efficient in time and memory if the precision is
           not infinite */
        r_len = bf_max(a->len, b->len + (d + LIMB_DIGITS - 1) / LIMB_DIGITS);
        if (bfdec_resize(r, r_len))
            goto fail;
        r->sign = a_sign;
        r->expn = a->expn;

        a_offset = r_len - a->len;
        for(i = 0; i < a_offset; i++)
            r->tab[i] = 0;
        for(i = 0; i < a->len; i++)
            r->tab[a_offset + i] = a->tab[i];

        b_shift = d % LIMB_DIGITS;
        if (b_shift == 0) {
            b1_len = b->len;
            b1_tab = (limb_t *)b->tab;
        } else {
            b1_len = b->len + 1;
            b1_tab = bf_malloc(s, sizeof(limb_t) * b1_len);
            if (!b1_tab)
                goto fail;
            b1_tab[0] = mp_shr_dec(b1_tab + 1, b->tab, b->len, b_shift, 0) *
                mp_pow_dec[LIMB_DIGITS - b_shift];
        }
        b_offset = r_len - (b->len + (d + LIMB_DIGITS - 1) / LIMB_DIGITS);

        if (is_sub) {
            carry = mp_sub_dec(r->tab + b_offset, r->tab + b_offset,
                               b1_tab, b1_len, 0);
            if (carry != 0) {
                carry = mp_sub_ui_dec(r->tab + b_offset + b1_len, carry,
                                      r_len - (b_offset + b1_len));
                assert(carry == 0);
            }
        } else {
            carry = mp_add_dec(r->tab + b_offset, r->tab + b_offset,
                               b1_tab, b1_len, 0);
            if (carry != 0) {
                carry = mp_add_ui_dec(r->tab + b_offset + b1_len, carry,
                                      r_len - (b_offset + b1_len));
            }
            if (carry != 0) {
                if (bfdec_resize(r, r_len + 1)) {
                    if (b_shift != 0)
                        bf_free(s, b1_tab);
                    goto fail;
                }
                r->tab[r_len] = 1;
                r->expn += LIMB_DIGITS;
            }
        }
        if (b_shift != 0)
            bf_free(s, b1_tab);
    renorm:
        ret = bfdec_normalize_and_round(r, prec, flags);
    }
    return ret;
 fail:
    bfdec_set_nan(r);
    return BF_ST_MEM_ERROR;
}

static int __bfdec_add(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
                     bf_flags_t flags)
{
    return bfdec_add_internal(r, a, b, prec, flags, 0);
}

static int __bfdec_sub(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
                     bf_flags_t flags)
{
    return bfdec_add_internal(r, a, b, prec, flags, 1);
}

int bfdec_add(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags)
{
    return bf_op2((bf_t *)r, (bf_t *)a, (bf_t *)b, prec, flags,
                  (bf_op2_func_t *)__bfdec_add);
}

int bfdec_sub(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags)
{
    return bf_op2((bf_t *)r, (bf_t *)a, (bf_t *)b, prec, flags,
                  (bf_op2_func_t *)__bfdec_sub);
}

int bfdec_mul(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags)
{
    int ret, r_sign;

    if (a->len < b->len) {
        const bfdec_t *tmp = a;
        a = b;
        b = tmp;
    }
    r_sign = a->sign ^ b->sign;
    /* here b->len <= a->len */
    if (b->len == 0) {
        if (a->expn == BF_EXP_NAN || b->expn == BF_EXP_NAN) {
            bfdec_set_nan(r);
            ret = 0;
        } else if (a->expn == BF_EXP_INF || b->expn == BF_EXP_INF) {
            if ((a->expn == BF_EXP_INF && b->expn == BF_EXP_ZERO) ||
                (a->expn == BF_EXP_ZERO && b->expn == BF_EXP_INF)) {
                bfdec_set_nan(r);
                ret = BF_ST_INVALID_OP;
            } else {
                bfdec_set_inf(r, r_sign);
                ret = 0;
            }
        } else {
            bfdec_set_zero(r, r_sign);
            ret = 0;
        }
    } else {
        bfdec_t tmp, *r1 = NULL;
        limb_t a_len, b_len;
        limb_t *a_tab, *b_tab;

        a_len = a->len;
        b_len = b->len;
        a_tab = a->tab;
        b_tab = b->tab;

        if (r == a || r == b) {
            bfdec_init(r->ctx, &tmp);
            r1 = r;
            r = &tmp;
        }
        if (bfdec_resize(r, a_len + b_len)) {
            bfdec_set_nan(r);
            ret = BF_ST_MEM_ERROR;
            goto done;
        }
        mp_mul_basecase_dec(r->tab, a_tab, a_len, b_tab, b_len);
        r->sign = r_sign;
        r->expn = a->expn + b->expn;
        ret = bfdec_normalize_and_round(r, prec, flags);
    done:
        if (r == &tmp)
            bfdec_move(r1, &tmp);
    }
    return ret;
}

int bfdec_mul_si(bfdec_t *r, const bfdec_t *a, int64_t b1, limb_t prec,
                 bf_flags_t flags)
{
    bfdec_t b;
    int ret;
    bfdec_init(r->ctx, &b);
    ret = bfdec_set_si(&b, b1);
    ret |= bfdec_mul(r, a, &b, prec, flags);
    bfdec_delete(&b);
    return ret;
}

int bfdec_add_si(bfdec_t *r, const bfdec_t *a, int64_t b1, limb_t prec,
                 bf_flags_t flags)
{
    bfdec_t b;
    int ret;

    bfdec_init(r->ctx, &b);
    ret = bfdec_set_si(&b, b1);
    ret |= bfdec_add(r, a, &b, prec, flags);
    bfdec_delete(&b);
    return ret;
}

static int __bfdec_div(bfdec_t *r, const bfdec_t *a, const bfdec_t *b,
                       limb_t prec, bf_flags_t flags)
{
    int ret, r_sign;
    limb_t n, nb, precl;

    r_sign = a->sign ^ b->sign;
    if (a->expn >= BF_EXP_INF || b->expn >= BF_EXP_INF) {
        if (a->expn == BF_EXP_NAN || b->expn == BF_EXP_NAN) {
            bfdec_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF && b->expn == BF_EXP_INF) {
            bfdec_set_nan(r);
            return BF_ST_INVALID_OP;
        } else if (a->expn == BF_EXP_INF) {
            bfdec_set_inf(r, r_sign);
            return 0;
        } else {
            bfdec_set_zero(r, r_sign);
            return 0;
        }
    } else if (a->expn == BF_EXP_ZERO) {
        if (b->expn == BF_EXP_ZERO) {
            bfdec_set_nan(r);
            return BF_ST_INVALID_OP;
        } else {
            bfdec_set_zero(r, r_sign);
            return 0;
        }
    } else if (b->expn == BF_EXP_ZERO) {
        bfdec_set_inf(r, r_sign);
        return BF_ST_DIVIDE_ZERO;
    }

    nb = b->len;
    if (prec == BF_PREC_INF) {
        /* infinite precision: return BF_ST_INVALID_OP if not an exact
           result */
        /* XXX: check */
        precl = nb + 1;
    } else if (flags & BF_FLAG_RADPNT_PREC) {
        /* number of digits after the decimal point */
        /* XXX: check (2 extra digits for rounding + 2 digits) */
        precl = (bf_max(a->expn - b->expn, 0) + 2 +
                 prec + 2 + LIMB_DIGITS - 1) / LIMB_DIGITS;
    } else {
        /* number of limbs of the quotient (2 extra digits for rounding) */
        precl = (prec + 2 + LIMB_DIGITS - 1) / LIMB_DIGITS;
    }
    n = bf_max(a->len, precl);

    {
        limb_t *taba, na, i;
        slimb_t d;

        na = n + nb;
        taba = bf_malloc(r->ctx, (na + 1) * sizeof(limb_t));
        if (!taba)
            goto fail;
        d = na - a->len;
        memset(taba, 0, d * sizeof(limb_t));
        memcpy(taba + d, a->tab, a->len * sizeof(limb_t));
        if (bfdec_resize(r, n + 1))
            goto fail1;
        if (mp_div_dec(r->ctx, r->tab, taba, na, b->tab, nb)) {
        fail1:
            bf_free(r->ctx, taba);
            goto fail;
        }
        /* see if non zero remainder */
        for(i = 0; i < nb; i++) {
            if (taba[i] != 0)
                break;
        }
        bf_free(r->ctx, taba);
        if (i != nb) {
            if (prec == BF_PREC_INF) {
                bfdec_set_nan(r);
                return BF_ST_INVALID_OP;
            } else {
                r->tab[0] |= 1;
            }
        }
        r->expn = a->expn - b->expn + LIMB_DIGITS;
        r->sign = r_sign;
        ret = bfdec_normalize_and_round(r, prec, flags);
    }
    return ret;
 fail:
    bfdec_set_nan(r);
    return BF_ST_MEM_ERROR;
}

int bfdec_div(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags)
{
    return bf_op2((bf_t *)r, (bf_t *)a, (bf_t *)b, prec, flags,
                  (bf_op2_func_t *)__bfdec_div);
}

/* a and b must be finite numbers with a >= 0 and b > 0. 'q' is the
   integer defined as floor(a/b) and r = a - q * b. */
static void bfdec_tdivremu(bf_context_t *s, bfdec_t *q, bfdec_t *r,
                           const bfdec_t *a, const bfdec_t *b)
{
    if (bfdec_cmpu(a, b) < 0) {
        bfdec_set_ui(q, 0);
        bfdec_set(r, a);
    } else {
        bfdec_div(q, a, b, 0, BF_RNDZ | BF_FLAG_RADPNT_PREC);
        bfdec_mul(r, q, b, BF_PREC_INF, BF_RNDZ);
        bfdec_sub(r, a, r, BF_PREC_INF, BF_RNDZ);
    }
}

/* division and remainder.

   rnd_mode is the rounding mode for the quotient. The additional
   rounding mode BF_RND_EUCLIDIAN is supported.

   'q' is an integer. 'r' is rounded with prec and flags (prec can be
   BF_PREC_INF).
*/
int bfdec_divrem(bfdec_t *q, bfdec_t *r, const bfdec_t *a, const bfdec_t *b,
                 limb_t prec, bf_flags_t flags, int rnd_mode)
{
    bf_context_t *s = q->ctx;
    bfdec_t a1_s, *a1 = &a1_s;
    bfdec_t b1_s, *b1 = &b1_s;
    bfdec_t r1_s, *r1 = &r1_s;
    int q_sign, res;
    BOOL is_ceil, is_rndn;

    assert(q != a && q != b);
    assert(r != a && r != b);
    assert(q != r);

    if (a->len == 0 || b->len == 0) {
        bfdec_set_zero(q, 0);
        if (a->expn == BF_EXP_NAN || b->expn == BF_EXP_NAN) {
            bfdec_set_nan(r);
            return 0;
        } else if (a->expn == BF_EXP_INF || b->expn == BF_EXP_ZERO) {
            bfdec_set_nan(r);
            return BF_ST_INVALID_OP;
        } else {
            bfdec_set(r, a);
            return bfdec_round(r, prec, flags);
        }
    }

    q_sign = a->sign ^ b->sign;
    is_rndn = (rnd_mode == BF_RNDN || rnd_mode == BF_RNDNA);
    switch(rnd_mode) {
    default:
    case BF_RNDZ:
    case BF_RNDN:
    case BF_RNDNA:
        is_ceil = FALSE;
        break;
    case BF_RNDD:
        is_ceil = q_sign;
        break;
    case BF_RNDU:
        is_ceil = q_sign ^ 1;
        break;
    case BF_RNDA:
        is_ceil = TRUE;
        break;
    case BF_DIVREM_EUCLIDIAN:
        is_ceil = a->sign;
        break;
    }

    a1->expn = a->expn;
    a1->tab = a->tab;
    a1->len = a->len;
    a1->sign = 0;

    b1->expn = b->expn;
    b1->tab = b->tab;
    b1->len = b->len;
    b1->sign = 0;

    //    bfdec_print_str("a1", a1);
    //    bfdec_print_str("b1", b1);
    /* XXX: could improve to avoid having a large 'q' */
    bfdec_tdivremu(s, q, r, a1, b1);
    if (bfdec_is_nan(q) || bfdec_is_nan(r))
        goto fail;
    //    bfdec_print_str("q", q);
    //    bfdec_print_str("r", r);

    if (r->len != 0) {
        if (is_rndn) {
            bfdec_init(s, r1);
            if (bfdec_set(r1, r))
                goto fail;
            if (bfdec_mul_si(r1, r1, 2, BF_PREC_INF, BF_RNDZ)) {
                bfdec_delete(r1);
                goto fail;
            }
            res = bfdec_cmpu(r1, b);
            bfdec_delete(r1);
            if (res > 0 ||
                (res == 0 &&
                 (rnd_mode == BF_RNDNA ||
                  (get_digit(q->tab, q->len, q->len * LIMB_DIGITS - q->expn) & 1) != 0))) {
                goto do_sub_r;
            }
        } else if (is_ceil) {
        do_sub_r:
            res = bfdec_add_si(q, q, 1, BF_PREC_INF, BF_RNDZ);
            res |= bfdec_sub(r, r, b1, BF_PREC_INF, BF_RNDZ);
            if (res & BF_ST_MEM_ERROR)
                goto fail;
        }
    }

    r->sign ^= a->sign;
    q->sign = q_sign;
    return bfdec_round(r, prec, flags);
 fail:
    bfdec_set_nan(q);
    bfdec_set_nan(r);
    return BF_ST_MEM_ERROR;
}

int bfdec_rem(bfdec_t *r, const bfdec_t *a, const bfdec_t *b, limb_t prec,
              bf_flags_t flags, int rnd_mode)
{
    bfdec_t q_s, *q = &q_s;
    int ret;

    bfdec_init(r->ctx, q);
    ret = bfdec_divrem(q, r, a, b, prec, flags, rnd_mode);
    bfdec_delete(q);
    return ret;
}

/* convert to integer (infinite precision) */
int bfdec_rint(bfdec_t *r, int rnd_mode)
{
    return bfdec_round(r, 0, rnd_mode | BF_FLAG_RADPNT_PREC);
}

int bfdec_sqrt(bfdec_t *r, const bfdec_t *a, limb_t prec, bf_flags_t flags)
{
    bf_context_t *s = a->ctx;
    int ret, k;
    limb_t *a1, v;
    slimb_t n, n1, prec1;
    limb_t res;

    assert(r != a);

    if (a->len == 0) {
        if (a->expn == BF_EXP_NAN) {
            bfdec_set_nan(r);
        } else if (a->expn == BF_EXP_INF && a->sign) {
            goto invalid_op;
        } else {
            bfdec_set(r, a);
        }
        ret = 0;
    } else if (a->sign || prec == BF_PREC_INF) {
 invalid_op:
        bfdec_set_nan(r);
        ret = BF_ST_INVALID_OP;
    } else {
        if (flags & BF_FLAG_RADPNT_PREC) {
            prec1 = bf_max(floor_div(a->expn + 1, 2) + prec, 1);
        } else {
            prec1 = prec;
        }
        /* convert the mantissa to an integer with at least 2 *
           prec + 4 digits */
        n = (2 * (prec1 + 2) + 2 * LIMB_DIGITS - 1) / (2 * LIMB_DIGITS);
        if (bfdec_resize(r, n))
            goto fail;
        a1 = bf_malloc(s, sizeof(limb_t) * 2 * n);
        if (!a1)
            goto fail;
        n1 = bf_min(2 * n, a->len);
        memset(a1, 0, (2 * n - n1) * sizeof(limb_t));
        memcpy(a1 + 2 * n - n1, a->tab + a->len - n1, n1 * sizeof(limb_t));
        if (a->expn & 1) {
            res = mp_shr_dec(a1, a1, 2 * n, 1, 0);
        } else {
            res = 0;
        }
        /* normalize so that a1 >= B^(2*n)/4. Not need for n = 1
           because mp_sqrtrem2_dec already does it */
        k = 0;
        if (n > 1) {
            v = a1[2 * n - 1];
            while (v < BF_DEC_BASE / 4) {
                k++;
                v *= 4;
            }
            if (k != 0)
                mp_mul1_dec(a1, a1, 2 * n, 1 << (2 * k), 0);
        }
        if (mp_sqrtrem_dec(s, r->tab, a1, n)) {
            bf_free(s, a1);
            goto fail;
        }
        if (k != 0)
            mp_div1_dec(r->tab, r->tab, n, 1 << k, 0);
        if (!res) {
            res = mp_scan_nz(a1, n + 1);
        }
        bf_free(s, a1);
        if (!res) {
            res = mp_scan_nz(a->tab, a->len - n1);
        }
        if (res != 0)
            r->tab[0] |= 1;
        r->sign = 0;
        r->expn = (a->expn + 1) >> 1;
        ret = bfdec_round(r, prec, flags);
    }
    return ret;
 fail:
    bfdec_set_nan(r);
    return BF_ST_MEM_ERROR;
}

/* The rounding mode is always BF_RNDZ. Return BF_ST_OVERFLOW if there
   is an overflow and 0 otherwise. No memory error is possible. */
int bfdec_get_int32(int *pres, const bfdec_t *a)
{
    uint32_t v;
    int ret;
    if (a->expn >= BF_EXP_INF) {
        ret = 0;
        if (a->expn == BF_EXP_INF) {
            v = (uint32_t)INT32_MAX + a->sign;
             /* XXX: return overflow ? */
        } else {
            v = INT32_MAX;
        }
    } else if (a->expn <= 0) {
        v = 0;
        ret = 0;
    } else if (a->expn <= 9) {
        v = fast_shr_dec(a->tab[a->len - 1], LIMB_DIGITS - a->expn);
        if (a->sign)
            v = -v;
        ret = 0;
    } else if (a->expn == 10) {
        uint64_t v1;
        uint32_t v_max;
#if LIMB_BITS == 64
        v1 = fast_shr_dec(a->tab[a->len - 1], LIMB_DIGITS - a->expn);
#else
        v1 = (uint64_t)a->tab[a->len - 1] * 10 +
            get_digit(a->tab, a->len, (a->len - 1) * LIMB_DIGITS - 1);
#endif
        v_max = (uint32_t)INT32_MAX + a->sign;
        if (v1 > v_max) {
            v = v_max;
            ret = BF_ST_OVERFLOW;
        } else {
            v = v1;
            if (a->sign)
                v = -v;
            ret = 0;
        }
    } else {
        v = (uint32_t)INT32_MAX + a->sign;
        ret = BF_ST_OVERFLOW;
    }
    *pres = v;
    return ret;
}

/* power to an integer with infinite precision */
int bfdec_pow_ui(bfdec_t *r, const bfdec_t *a, limb_t b)
{
    int ret, n_bits, i;

    assert(r != a);
    if (b == 0)
        return bfdec_set_ui(r, 1);
    ret = bfdec_set(r, a);
    n_bits = LIMB_BITS - clz(b);
    for(i = n_bits - 2; i >= 0; i--) {
        ret |= bfdec_mul(r, r, r, BF_PREC_INF, BF_RNDZ);
        if ((b >> i) & 1)
            ret |= bfdec_mul(r, r, a, BF_PREC_INF, BF_RNDZ);
    }
    return ret;
}

char *bfdec_ftoa(size_t *plen, const bfdec_t *a, limb_t prec, bf_flags_t flags)
{
    return bf_ftoa_internal(plen, (const bf_t *)a, 10, prec, flags, TRUE);
}

int bfdec_atof(bfdec_t *r, const char *str, const char **pnext,
               limb_t prec, bf_flags_t flags)
{
    slimb_t dummy_exp;
    return bf_atof_internal((bf_t *)r, &dummy_exp, str, pnext, 10, prec,
                            flags, TRUE);
}

#endif /* USE_BF_DEC */

#ifdef USE_FFT_MUL
/***************************************************************/
/* Integer multiplication with FFT */

/* or LIMB_BITS at bit position 'pos' in tab */
static inline void put_bits(limb_t *tab, limb_t len, slimb_t pos, limb_t val)
{
    limb_t i;
    int p;

    i = pos >> LIMB_LOG2_BITS;
    p = pos & (LIMB_BITS - 1);
    if (i < len)
        tab[i] |= val << p;
    if (p != 0) {
        i++;
        if (i < len) {
            tab[i] |= val >> (LIMB_BITS - p);
        }
    }
}

#if defined(__AVX2__)

typedef double NTTLimb;

/* we must have: modulo >= 1 << NTT_MOD_LOG2_MIN */
#define NTT_MOD_LOG2_MIN 50
#define NTT_MOD_LOG2_MAX 51
#define NB_MODS 5
#define NTT_PROOT_2EXP 39
static const int ntt_int_bits[NB_MODS] = { 254, 203, 152, 101, 50, };

static const limb_t ntt_mods[NB_MODS] = { 0x00073a8000000001, 0x0007858000000001, 0x0007a38000000001, 0x0007a68000000001, 0x0007fd8000000001,
};

static const limb_t ntt_proot[2][NB_MODS] = {
    { 0x00056198d44332c8, 0x0002eb5d640aad39, 0x00047e31eaa35fd0, 0x0005271ac118a150, 0x00075e0ce8442bd5, },
    { 0x000461169761bcc5, 0x0002dac3cb2da688, 0x0004abc97751e3bf, 0x000656778fc8c485, 0x0000dc6469c269fa, },
};

static const limb_t ntt_mods_cr[NB_MODS * (NB_MODS - 1) / 2] = {
 0x00020e4da740da8e, 0x0004c3dc09c09c1d, 0x000063bd097b4271, 0x000799d8f18f18fd,
 0x0005384222222264, 0x000572b07c1f07fe, 0x00035cd08888889a,
 0x00066015555557e3, 0x000725960b60b623,
 0x0002fc1fa1d6ce12,
};

#else

typedef limb_t NTTLimb;

#if LIMB_BITS == 64

#define NTT_MOD_LOG2_MIN 61
#define NTT_MOD_LOG2_MAX 62
#define NB_MODS 5
#define NTT_PROOT_2EXP 51
static const int ntt_int_bits[NB_MODS] = { 307, 246, 185, 123, 61, };

static const limb_t ntt_mods[NB_MODS] = { 0x28d8000000000001, 0x2a88000000000001, 0x2ed8000000000001, 0x3508000000000001, 0x3aa8000000000001,
};

static const limb_t ntt_proot[2][NB_MODS] = {
    { 0x1b8ea61034a2bea7, 0x21a9762de58206fb, 0x02ca782f0756a8ea, 0x278384537a3e50a1, 0x106e13fee74ce0ab, },
    { 0x233513af133e13b8, 0x1d13140d1c6f75f1, 0x12cde57f97e3eeda, 0x0d6149e23cbe654f, 0x36cd204f522a1379, },
};

static const limb_t ntt_mods_cr[NB_MODS * (NB_MODS - 1) / 2] = {
 0x08a9ed097b425eea, 0x18a44aaaaaaaaab3, 0x2493f57f57f57f5d, 0x126b8d0649a7f8d4,
 0x09d80ed7303b5ccc, 0x25b8bcf3cf3cf3d5, 0x2ce6ce63398ce638,
 0x0e31fad40a57eb59, 0x02a3529fd4a7f52f,
 0x3a5493e93e93e94a,
};

#elif LIMB_BITS == 32

/* we must have: modulo >= 1 << NTT_MOD_LOG2_MIN */
#define NTT_MOD_LOG2_MIN 29
#define NTT_MOD_LOG2_MAX 30
#define NB_MODS 5
#define NTT_PROOT_2EXP 20
static const int ntt_int_bits[NB_MODS] = { 148, 119, 89, 59, 29, };

static const limb_t ntt_mods[NB_MODS] = { 0x0000000032b00001, 0x0000000033700001, 0x0000000036d00001, 0x0000000037300001, 0x000000003e500001,
};

static const limb_t ntt_proot[2][NB_MODS] = {
    { 0x0000000032525f31, 0x0000000005eb3b37, 0x00000000246eda9f, 0x0000000035f25901, 0x00000000022f5768, },
    { 0x00000000051eba1a, 0x00000000107be10e, 0x000000001cd574e0, 0x00000000053806e6, 0x000000002cd6bf98, },
};

static const limb_t ntt_mods_cr[NB_MODS * (NB_MODS - 1) / 2] = {
 0x000000000449559a, 0x000000001eba6ca9, 0x000000002ec18e46, 0x000000000860160b,
 0x000000000d321307, 0x000000000bf51120, 0x000000000f662938,
 0x000000000932ab3e, 0x000000002f40eef8,
 0x000000002e760905,
};

#endif /* LIMB_BITS */

#endif /* !AVX2 */

#if defined(__AVX2__)
#define NTT_TRIG_K_MAX 18
#else
#define NTT_TRIG_K_MAX 19
#endif

typedef struct BFNTTState {
    bf_context_t *ctx;

    /* used for mul_mod_fast() */
    limb_t ntt_mods_div[NB_MODS];

    limb_t ntt_proot_pow[NB_MODS][2][NTT_PROOT_2EXP + 1];
    limb_t ntt_proot_pow_inv[NB_MODS][2][NTT_PROOT_2EXP + 1];
    NTTLimb *ntt_trig[NB_MODS][2][NTT_TRIG_K_MAX + 1];
    /* 1/2^n mod m */
    limb_t ntt_len_inv[NB_MODS][NTT_PROOT_2EXP + 1][2];
#if defined(__AVX2__)
    __m256d ntt_mods_cr_vec[NB_MODS * (NB_MODS - 1) / 2];
    __m256d ntt_mods_vec[NB_MODS];
    __m256d ntt_mods_inv_vec[NB_MODS];
#else
    limb_t ntt_mods_cr_inv[NB_MODS * (NB_MODS - 1) / 2];
#endif
} BFNTTState;

static NTTLimb *get_trig(BFNTTState *s, int k, int inverse, int m_idx);

/* add modulo with up to (LIMB_BITS-1) bit modulo */
static inline limb_t add_mod(limb_t a, limb_t b, limb_t m)
{
    limb_t r;
    r = a + b;
    if (r >= m)
        r -= m;
    return r;
}

/* sub modulo with up to LIMB_BITS bit modulo */
static inline limb_t sub_mod(limb_t a, limb_t b, limb_t m)
{
    limb_t r;
    r = a - b;
    if (r > a)
        r += m;
    return r;
}

/* return (r0+r1*B) mod m
   precondition: 0 <= r0+r1*B < 2^(64+NTT_MOD_LOG2_MIN)
*/
static inline limb_t mod_fast(dlimb_t r,
                                limb_t m, limb_t m_inv)
{
    limb_t a1, q, t0, r1, r0;

    a1 = r >> NTT_MOD_LOG2_MIN;

    q = ((dlimb_t)a1 * m_inv) >> LIMB_BITS;
    r = r - (dlimb_t)q * m - m * 2;
    r1 = r >> LIMB_BITS;
    t0 = (slimb_t)r1 >> 1;
    r += m & t0;
    r0 = r;
    r1 = r >> LIMB_BITS;
    r0 += m & r1;
    return r0;
}

/* faster version using precomputed modulo inverse.
   precondition: 0 <= a * b < 2^(64+NTT_MOD_LOG2_MIN) */
static inline limb_t mul_mod_fast(limb_t a, limb_t b,
                                    limb_t m, limb_t m_inv)
{
    dlimb_t r;
    r = (dlimb_t)a * (dlimb_t)b;
    return mod_fast(r, m, m_inv);
}

static inline limb_t init_mul_mod_fast(limb_t m)
{
    dlimb_t t;
    assert(m < (limb_t)1 << NTT_MOD_LOG2_MAX);
    assert(m >= (limb_t)1 << NTT_MOD_LOG2_MIN);
    t = (dlimb_t)1 << (LIMB_BITS + NTT_MOD_LOG2_MIN);
    return t / m;
}

/* Faster version used when the multiplier is constant. 0 <= a < 2^64,
   0 <= b < m. */
static inline limb_t mul_mod_fast2(limb_t a, limb_t b,
                                     limb_t m, limb_t b_inv)
{
    limb_t r, q;

    q = ((dlimb_t)a * (dlimb_t)b_inv) >> LIMB_BITS;
    r = a * b - q * m;
    if (r >= m)
        r -= m;
    return r;
}

/* Faster version used when the multiplier is constant. 0 <= a < 2^64,
   0 <= b < m. Let r = a * b mod m. The return value is 'r' or 'r +
   m'. */
static inline limb_t mul_mod_fast3(limb_t a, limb_t b,
                                     limb_t m, limb_t b_inv)
{
    limb_t r, q;

    q = ((dlimb_t)a * (dlimb_t)b_inv) >> LIMB_BITS;
    r = a * b - q * m;
    return r;
}

static inline limb_t init_mul_mod_fast2(limb_t b, limb_t m)
{
    return ((dlimb_t)b << LIMB_BITS) / m;
}

#ifdef __AVX2__

static inline limb_t ntt_limb_to_int(NTTLimb a, limb_t m)
{
    slimb_t v;
    v = a;
    if (v < 0)
        v += m;
    if (v >= m)
        v -= m;
    return v;
}

static inline NTTLimb int_to_ntt_limb(limb_t a, limb_t m)
{
    return (slimb_t)a;
}

static inline NTTLimb int_to_ntt_limb2(limb_t a, limb_t m)
{
    if (a >= (m / 2))
        a -= m;
    return (slimb_t)a;
}

/* return r + m if r < 0 otherwise r. */
static inline __m256d ntt_mod1(__m256d r, __m256d m)
{
    return _mm256_blendv_pd(r, r + m, r);
}

/* input: abs(r) < 2 * m. Output: abs(r) < m */
static inline __m256d ntt_mod(__m256d r, __m256d mf, __m256d m2f)
{
    return _mm256_blendv_pd(r, r + m2f, r) - mf;
}

/* input: abs(a*b) < 2 * m^2, output: abs(r) < m */
static inline __m256d ntt_mul_mod(__m256d a, __m256d b, __m256d mf,
                                  __m256d m_inv)
{
    __m256d r, q, ab1, ab0, qm0, qm1;
    ab1 = a * b;
    q = _mm256_round_pd(ab1 * m_inv, 0); /* round to nearest */
    qm1 = q * mf;
    qm0 = _mm256_fmsub_pd(q, mf, qm1); /* low part */
    ab0 = _mm256_fmsub_pd(a, b, ab1); /* low part */
    r = (ab1 - qm1) + (ab0 - qm0);
    return r;
}

static void *bf_aligned_malloc(bf_context_t *s, size_t size, size_t align)
{
    void *ptr;
    void **ptr1;
    ptr = bf_malloc(s, size + sizeof(void *) + align - 1);
    if (!ptr)
        return NULL;
    ptr1 = (void **)(((uintptr_t)ptr + sizeof(void *) + align - 1) &
                     ~(align - 1));
    ptr1[-1] = ptr;
    return ptr1;
}

static void bf_aligned_free(bf_context_t *s, void *ptr)
{
    if (!ptr)
        return;
    bf_free(s, ((void **)ptr)[-1]);
}

static void *ntt_malloc(BFNTTState *s, size_t size)
{
    return bf_aligned_malloc(s->ctx, size, 64);
}

static void ntt_free(BFNTTState *s, void *ptr)
{
    bf_aligned_free(s->ctx, ptr);
}

static no_inline int ntt_fft(BFNTTState *s,
                             NTTLimb *out_buf, NTTLimb *in_buf,
                             NTTLimb *tmp_buf, int fft_len_log2,
                             int inverse, int m_idx)
{
    limb_t nb_blocks, fft_per_block, p, k, n, stride_in, i, j;
    NTTLimb *tab_in, *tab_out, *tmp, *trig;
    __m256d m_inv, mf, m2f, c, a0, a1, b0, b1;
    limb_t m;
    int l;

    m = ntt_mods[m_idx];

    m_inv = _mm256_set1_pd(1.0 / (double)m);
    mf = _mm256_set1_pd(m);
    m2f = _mm256_set1_pd(m * 2);

    n = (limb_t)1 << fft_len_log2;
    assert(n >= 8);
    stride_in = n / 2;

    tab_in = in_buf;
    tab_out = tmp_buf;
    trig = get_trig(s, fft_len_log2, inverse, m_idx);
    if (!trig)
        return -1;
    p = 0;
    for(k = 0; k < stride_in; k += 4) {
        a0 = _mm256_load_pd(&tab_in[k]);
        a1 = _mm256_load_pd(&tab_in[k + stride_in]);
        c = _mm256_load_pd(trig);
        trig += 4;
        b0 = ntt_mod(a0 + a1, mf, m2f);
        b1 = ntt_mul_mod(a0 - a1, c, mf, m_inv);
        a0 = _mm256_permute2f128_pd(b0, b1, 0x20);
        a1 = _mm256_permute2f128_pd(b0, b1, 0x31);
        a0 = _mm256_permute4x64_pd(a0, 0xd8);
        a1 = _mm256_permute4x64_pd(a1, 0xd8);
        _mm256_store_pd(&tab_out[p], a0);
        _mm256_store_pd(&tab_out[p + 4], a1);
        p += 2 * 4;
    }
    tmp = tab_in;
    tab_in = tab_out;
    tab_out = tmp;

    trig = get_trig(s, fft_len_log2 - 1, inverse, m_idx);
    if (!trig)
        return -1;
    p = 0;
    for(k = 0; k < stride_in; k += 4) {
        a0 = _mm256_load_pd(&tab_in[k]);
        a1 = _mm256_load_pd(&tab_in[k + stride_in]);
        c = _mm256_setr_pd(trig[0], trig[0], trig[1], trig[1]);
        trig += 2;
        b0 = ntt_mod(a0 + a1, mf, m2f);
        b1 = ntt_mul_mod(a0 - a1, c, mf, m_inv);
        a0 = _mm256_permute2f128_pd(b0, b1, 0x20);
        a1 = _mm256_permute2f128_pd(b0, b1, 0x31);
        _mm256_store_pd(&tab_out[p], a0);
        _mm256_store_pd(&tab_out[p + 4], a1);
        p += 2 * 4;
    }
    tmp = tab_in;
    tab_in = tab_out;
    tab_out = tmp;

    nb_blocks = n / 4;
    fft_per_block = 4;

    l = fft_len_log2 - 2;
    while (nb_blocks != 2) {
        nb_blocks >>= 1;
        p = 0;
        k = 0;
        trig = get_trig(s, l, inverse, m_idx);
        if (!trig)
            return -1;
        for(i = 0; i < nb_blocks; i++) {
            c = _mm256_set1_pd(trig[0]);
            trig++;
            for(j = 0; j < fft_per_block; j += 4) {
                a0 = _mm256_load_pd(&tab_in[k + j]);
                a1 = _mm256_load_pd(&tab_in[k + j + stride_in]);
                b0 = ntt_mod(a0 + a1, mf, m2f);
                b1 = ntt_mul_mod(a0 - a1, c, mf, m_inv);
                _mm256_store_pd(&tab_out[p + j], b0);
                _mm256_store_pd(&tab_out[p + j + fft_per_block], b1);
            }
            k += fft_per_block;
            p += 2 * fft_per_block;
        }
        fft_per_block <<= 1;
        l--;
        tmp = tab_in;
        tab_in = tab_out;
        tab_out = tmp;
    }

    tab_out = out_buf;
    for(k = 0; k < stride_in; k += 4) {
        a0 = _mm256_load_pd(&tab_in[k]);
        a1 = _mm256_load_pd(&tab_in[k + stride_in]);
        b0 = ntt_mod(a0 + a1, mf, m2f);
        b1 = ntt_mod(a0 - a1, mf, m2f);
        _mm256_store_pd(&tab_out[k], b0);
        _mm256_store_pd(&tab_out[k + stride_in], b1);
    }
    return 0;
}

static void ntt_vec_mul(BFNTTState *s,
                        NTTLimb *tab1, NTTLimb *tab2, limb_t fft_len_log2,
                        int k_tot, int m_idx)
{
    limb_t i, c_inv, n, m;
    __m256d m_inv, mf, a, b, c;

    m = ntt_mods[m_idx];
    c_inv = s->ntt_len_inv[m_idx][k_tot][0];
    m_inv = _mm256_set1_pd(1.0 / (double)m);
    mf = _mm256_set1_pd(m);
    c = _mm256_set1_pd(int_to_ntt_limb(c_inv, m));
    n = (limb_t)1 << fft_len_log2;
    for(i = 0; i < n; i += 4) {
        a = _mm256_load_pd(&tab1[i]);
        b = _mm256_load_pd(&tab2[i]);
        a = ntt_mul_mod(a, b, mf, m_inv);
        a = ntt_mul_mod(a, c, mf, m_inv);
        _mm256_store_pd(&tab1[i], a);
    }
}

static no_inline void mul_trig(NTTLimb *buf,
                               limb_t n, limb_t c1, limb_t m, limb_t m_inv1)
{
    limb_t i, c2, c3, c4;
    __m256d c, c_mul, a0, mf, m_inv;
    assert(n >= 2);

    mf = _mm256_set1_pd(m);
    m_inv = _mm256_set1_pd(1.0 / (double)m);

    c2 = mul_mod_fast(c1, c1, m, m_inv1);
    c3 = mul_mod_fast(c2, c1, m, m_inv1);
    c4 = mul_mod_fast(c2, c2, m, m_inv1);
    c = _mm256_setr_pd(1, int_to_ntt_limb(c1, m),
                       int_to_ntt_limb(c2, m), int_to_ntt_limb(c3, m));
    c_mul = _mm256_set1_pd(int_to_ntt_limb(c4, m));
    for(i = 0; i < n; i += 4) {
        a0 = _mm256_load_pd(&buf[i]);
        a0 = ntt_mul_mod(a0, c, mf, m_inv);
        _mm256_store_pd(&buf[i], a0);
        c = ntt_mul_mod(c, c_mul, mf, m_inv);
    }
}

#else

static void *ntt_malloc(BFNTTState *s, size_t size)
{
    return bf_malloc(s->ctx, size);
}

static void ntt_free(BFNTTState *s, void *ptr)
{
    bf_free(s->ctx, ptr);
}

static inline limb_t ntt_limb_to_int(NTTLimb a, limb_t m)
{
    if (a >= m)
        a -= m;
    return a;
}

static inline NTTLimb int_to_ntt_limb(slimb_t a, limb_t m)
{
    return a;
}

static no_inline int ntt_fft(BFNTTState *s, NTTLimb *out_buf, NTTLimb *in_buf,
                             NTTLimb *tmp_buf, int fft_len_log2,
                             int inverse, int m_idx)
{
    limb_t nb_blocks, fft_per_block, p, k, n, stride_in, i, j, m, m2;
    NTTLimb *tab_in, *tab_out, *tmp, a0, a1, b0, b1, c, *trig, c_inv;
    int l;

    m = ntt_mods[m_idx];
    m2 = 2 * m;
    n = (limb_t)1 << fft_len_log2;
    nb_blocks = n;
    fft_per_block = 1;
    stride_in = n / 2;
    tab_in = in_buf;
    tab_out = tmp_buf;
    l = fft_len_log2;
    while (nb_blocks != 2) {
        nb_blocks >>= 1;
        p = 0;
        k = 0;
        trig = get_trig(s, l, inverse, m_idx);
        if (!trig)
            return -1;
        for(i = 0; i < nb_blocks; i++) {
            c = trig[0];
            c_inv = trig[1];
            trig += 2;
            for(j = 0; j < fft_per_block; j++) {
                a0 = tab_in[k + j];
                a1 = tab_in[k + j + stride_in];
                b0 = add_mod(a0, a1, m2);
                b1 = a0 - a1 + m2;
                b1 = mul_mod_fast3(b1, c, m, c_inv);
                tab_out[p + j] = b0;
                tab_out[p + j + fft_per_block] = b1;
            }
            k += fft_per_block;
            p += 2 * fft_per_block;
        }
        fft_per_block <<= 1;
        l--;
        tmp = tab_in;
        tab_in = tab_out;
        tab_out = tmp;
    }
    /* no twiddle in last step */
    tab_out = out_buf;
    for(k = 0; k < stride_in; k++) {
        a0 = tab_in[k];
        a1 = tab_in[k + stride_in];
        b0 = add_mod(a0, a1, m2);
        b1 = sub_mod(a0, a1, m2);
        tab_out[k] = b0;
        tab_out[k + stride_in] = b1;
    }
    return 0;
}

static void ntt_vec_mul(BFNTTState *s,
                        NTTLimb *tab1, NTTLimb *tab2, int fft_len_log2,
                        int k_tot, int m_idx)
{
    limb_t i, norm, norm_inv, a, n, m, m_inv;

    m = ntt_mods[m_idx];
    m_inv = s->ntt_mods_div[m_idx];
    norm = s->ntt_len_inv[m_idx][k_tot][0];
    norm_inv = s->ntt_len_inv[m_idx][k_tot][1];
    n = (limb_t)1 << fft_len_log2;
    for(i = 0; i < n; i++) {
        a = tab1[i];
        /* need to reduce the range so that the product is <
           2^(LIMB_BITS+NTT_MOD_LOG2_MIN) */
        if (a >= m)
            a -= m;
        a = mul_mod_fast(a, tab2[i], m, m_inv);
        a = mul_mod_fast3(a, norm, m, norm_inv);
        tab1[i] = a;
    }
}

static no_inline void mul_trig(NTTLimb *buf,
                               limb_t n, limb_t c_mul, limb_t m, limb_t m_inv)
{
    limb_t i, c0, c_mul_inv;

    c0 = 1;
    c_mul_inv = init_mul_mod_fast2(c_mul, m);
    for(i = 0; i < n; i++) {
        buf[i] = mul_mod_fast(buf[i], c0, m, m_inv);
        c0 = mul_mod_fast2(c0, c_mul, m, c_mul_inv);
    }
}

#endif /* !AVX2 */

static no_inline NTTLimb *get_trig(BFNTTState *s,
                                   int k, int inverse, int m_idx)
{
    NTTLimb *tab;
    limb_t i, n2, c, c_mul, m, c_mul_inv;

    if (k > NTT_TRIG_K_MAX)
        return NULL;

    tab = s->ntt_trig[m_idx][inverse][k];
    if (tab)
        return tab;
    n2 = (limb_t)1 << (k - 1);
    m = ntt_mods[m_idx];
#ifdef __AVX2__
    tab = ntt_malloc(s, sizeof(NTTLimb) * n2);
#else
    tab = ntt_malloc(s, sizeof(NTTLimb) * n2 * 2);
#endif
    if (!tab)
        return NULL;
    c = 1;
    c_mul = s->ntt_proot_pow[m_idx][inverse][k];
    c_mul_inv = s->ntt_proot_pow_inv[m_idx][inverse][k];
    for(i = 0; i < n2; i++) {
#ifdef __AVX2__
        tab[i] = int_to_ntt_limb2(c, m);
#else
        tab[2 * i] = int_to_ntt_limb(c, m);
        tab[2 * i + 1] = init_mul_mod_fast2(c, m);
#endif
        c = mul_mod_fast2(c, c_mul, m, c_mul_inv);
    }
    s->ntt_trig[m_idx][inverse][k] = tab;
    return tab;
}

void fft_clear_cache(bf_context_t *s1)
{
    int m_idx, inverse, k;
    BFNTTState *s = s1->ntt_state;
    if (s) {
        for(m_idx = 0; m_idx < NB_MODS; m_idx++) {
            for(inverse = 0; inverse < 2; inverse++) {
                for(k = 0; k < NTT_TRIG_K_MAX + 1; k++) {
                    if (s->ntt_trig[m_idx][inverse][k]) {
                        ntt_free(s, s->ntt_trig[m_idx][inverse][k]);
                        s->ntt_trig[m_idx][inverse][k] = NULL;
                    }
                }
            }
        }
#if defined(__AVX2__)
        bf_aligned_free(s1, s);
#else
        bf_free(s1, s);
#endif
        s1->ntt_state = NULL;
    }
}

#define STRIP_LEN 16

/* dst = buf1, src = buf2 */
static int ntt_fft_partial(BFNTTState *s, NTTLimb *buf1,
                           int k1, int k2, limb_t n1, limb_t n2, int inverse,
                           limb_t m_idx)
{
    limb_t i, j, c_mul, c0, m, m_inv, strip_len, l;
    NTTLimb *buf2, *buf3;

    buf2 = NULL;
    buf3 = ntt_malloc(s, sizeof(NTTLimb) * n1);
    if (!buf3)
        goto fail;
    if (k2 == 0) {
        if (ntt_fft(s, buf1, buf1, buf3, k1, inverse, m_idx))
            goto fail;
    } else {
        strip_len = STRIP_LEN;
        buf2 = ntt_malloc(s, sizeof(NTTLimb) * n1 * strip_len);
        if (!buf2)
            goto fail;
        m = ntt_mods[m_idx];
        m_inv = s->ntt_mods_div[m_idx];
        c0 = s->ntt_proot_pow[m_idx][inverse][k1 + k2];
        c_mul = 1;
        assert((n2 % strip_len) == 0);
        for(j = 0; j < n2; j += strip_len) {
            for(i = 0; i < n1; i++) {
                for(l = 0; l < strip_len; l++) {
                    buf2[i + l * n1] = buf1[i * n2 + (j + l)];
                }
            }
            for(l = 0; l < strip_len; l++) {
                if (inverse)
                    mul_trig(buf2 + l * n1, n1, c_mul, m, m_inv);
                if (ntt_fft(s, buf2 + l * n1, buf2 + l * n1, buf3, k1, inverse, m_idx))
                    goto fail;
                if (!inverse)
                    mul_trig(buf2 + l * n1, n1, c_mul, m, m_inv);
                c_mul = mul_mod_fast(c_mul, c0, m, m_inv);
            }

            for(i = 0; i < n1; i++) {
                for(l = 0; l < strip_len; l++) {
                    buf1[i * n2 + (j + l)] = buf2[i + l *n1];
                }
            }
        }
        ntt_free(s, buf2);
    }
    ntt_free(s, buf3);
    return 0;
 fail:
    ntt_free(s, buf2);
    ntt_free(s, buf3);
    return -1;
}


/* dst = buf1, src = buf2, tmp = buf3 */
static int ntt_conv(BFNTTState *s, NTTLimb *buf1, NTTLimb *buf2,
                    int k, int k_tot, limb_t m_idx)
{
    limb_t n1, n2, i;
    int k1, k2;

    if (k <= NTT_TRIG_K_MAX) {
        k1 = k;
    } else {
        /* recursive split of the FFT */
        k1 = bf_min(k / 2, NTT_TRIG_K_MAX);
    }
    k2 = k - k1;
    n1 = (limb_t)1 << k1;
    n2 = (limb_t)1 << k2;

    if (ntt_fft_partial(s, buf1, k1, k2, n1, n2, 0, m_idx))
        return -1;
    if (ntt_fft_partial(s, buf2, k1, k2, n1, n2, 0, m_idx))
        return -1;
    if (k2 == 0) {
        ntt_vec_mul(s, buf1, buf2, k, k_tot, m_idx);
    } else {
        for(i = 0; i < n1; i++) {
            ntt_conv(s, buf1 + i * n2, buf2 + i * n2, k2, k_tot, m_idx);
        }
    }
    if (ntt_fft_partial(s, buf1, k1, k2, n1, n2, 1, m_idx))
        return -1;
    return 0;
}


static no_inline void limb_to_ntt(BFNTTState *s,
                                  NTTLimb *tabr, limb_t fft_len,
                                  const limb_t *taba, limb_t a_len, int dpl,
                                  int first_m_idx, int nb_mods)
{
    slimb_t i, n;
    dlimb_t a, b;
    int j, shift;
    limb_t base_mask1, a0, a1, a2, r, m, m_inv;

    memset(tabr, 0, sizeof(NTTLimb) * fft_len * nb_mods);
    shift = dpl & (LIMB_BITS - 1);
    if (shift == 0)
        base_mask1 = -1;
    else
        base_mask1 = ((limb_t)1 << shift) - 1;
    n = bf_min(fft_len, (a_len * LIMB_BITS + dpl - 1) / dpl);
    for(i = 0; i < n; i++) {
        a0 = get_bits(taba, a_len, i * dpl);
        if (dpl <= LIMB_BITS) {
            a0 &= base_mask1;
            a = a0;
        } else {
            a1 = get_bits(taba, a_len, i * dpl + LIMB_BITS);
            if (dpl <= (LIMB_BITS + NTT_MOD_LOG2_MIN)) {
                a = a0 | ((dlimb_t)(a1 & base_mask1) << LIMB_BITS);
            } else {
                if (dpl > 2 * LIMB_BITS) {
                    a2 = get_bits(taba, a_len, i * dpl + LIMB_BITS * 2) &
                        base_mask1;
                } else {
                    a1 &= base_mask1;
                    a2 = 0;
                }
                //            printf("a=0x%016lx%016lx%016lx\n", a2, a1, a0);
                a = (a0 >> (LIMB_BITS - NTT_MOD_LOG2_MAX + NTT_MOD_LOG2_MIN)) |
                    ((dlimb_t)a1 << (NTT_MOD_LOG2_MAX - NTT_MOD_LOG2_MIN)) |
                    ((dlimb_t)a2 << (LIMB_BITS + NTT_MOD_LOG2_MAX - NTT_MOD_LOG2_MIN));
                a0 &= ((limb_t)1 << (LIMB_BITS - NTT_MOD_LOG2_MAX + NTT_MOD_LOG2_MIN)) - 1;
            }
        }
        for(j = 0; j < nb_mods; j++) {
            m = ntt_mods[first_m_idx + j];
            m_inv = s->ntt_mods_div[first_m_idx + j];
            r = mod_fast(a, m, m_inv);
            if (dpl > (LIMB_BITS + NTT_MOD_LOG2_MIN)) {
                b = ((dlimb_t)r << (LIMB_BITS - NTT_MOD_LOG2_MAX + NTT_MOD_LOG2_MIN)) | a0;
                r = mod_fast(b, m, m_inv);
            }
            tabr[i + j * fft_len] = int_to_ntt_limb(r, m);
        }
    }
}

#if defined(__AVX2__)

#define VEC_LEN 4

typedef union {
    __m256d v;
    double d[4];
} VecUnion;

static no_inline void ntt_to_limb(BFNTTState *s, limb_t *tabr, limb_t r_len,
                                  const NTTLimb *buf, int fft_len_log2, int dpl,
                                  int nb_mods)
{
    const limb_t *mods = ntt_mods + NB_MODS - nb_mods;
    const __m256d *mods_cr_vec, *mf, *m_inv;
    VecUnion y[NB_MODS];
    limb_t u[NB_MODS], carry[NB_MODS], fft_len, base_mask1, r;
    slimb_t i, len, pos;
    int j, k, l, shift, n_limb1, p;
    dlimb_t t;

    j = NB_MODS * (NB_MODS - 1) / 2 - nb_mods * (nb_mods - 1) / 2;
    mods_cr_vec = s->ntt_mods_cr_vec + j;
    mf = s->ntt_mods_vec + NB_MODS - nb_mods;
    m_inv = s->ntt_mods_inv_vec + NB_MODS - nb_mods;

    shift = dpl & (LIMB_BITS - 1);
    if (shift == 0)
        base_mask1 = -1;
    else
        base_mask1 = ((limb_t)1 << shift) - 1;
    n_limb1 = ((unsigned)dpl - 1) / LIMB_BITS;
    for(j = 0; j < NB_MODS; j++)
        carry[j] = 0;
    for(j = 0; j < NB_MODS; j++)
        u[j] = 0; /* avoid warnings */
    memset(tabr, 0, sizeof(limb_t) * r_len);
    fft_len = (limb_t)1 << fft_len_log2;
    len = bf_min(fft_len, (r_len * LIMB_BITS + dpl - 1) / dpl);
    len = (len + VEC_LEN - 1) & ~(VEC_LEN - 1);
    i = 0;
    while (i < len) {
        for(j = 0; j < nb_mods; j++)
            y[j].v = *(__m256d *)&buf[i + fft_len * j];

        /* Chinese remainder to get mixed radix representation */
        l = 0;
        for(j = 0; j < nb_mods - 1; j++) {
            y[j].v = ntt_mod1(y[j].v, mf[j]);
            for(k = j + 1; k < nb_mods; k++) {
                y[k].v = ntt_mul_mod(y[k].v - y[j].v,
                                     mods_cr_vec[l], mf[k], m_inv[k]);
                l++;
            }
        }
        y[j].v = ntt_mod1(y[j].v, mf[j]);

        for(p = 0; p < VEC_LEN; p++) {
            /* back to normal representation */
            u[0] = (int64_t)y[nb_mods - 1].d[p];
            l = 1;
            for(j = nb_mods - 2; j >= 1; j--) {
                r = (int64_t)y[j].d[p];
                for(k = 0; k < l; k++) {
                    t = (dlimb_t)u[k] * mods[j] + r;
                    r = t >> LIMB_BITS;
                    u[k] = t;
                }
                u[l] = r;
                l++;
            }
            /* XXX: for nb_mods = 5, l should be 4 */

            /* last step adds the carry */
            r = (int64_t)y[0].d[p];
            for(k = 0; k < l; k++) {
                t = (dlimb_t)u[k] * mods[j] + r + carry[k];
                r = t >> LIMB_BITS;
                u[k] = t;
            }
            u[l] = r + carry[l];

            /* write the digits */
            pos = i * dpl;
            for(j = 0; j < n_limb1; j++) {
                put_bits(tabr, r_len, pos, u[j]);
                pos += LIMB_BITS;
            }
            put_bits(tabr, r_len, pos, u[n_limb1] & base_mask1);
            /* shift by dpl digits and set the carry */
            if (shift == 0) {
                for(j = n_limb1 + 1; j < nb_mods; j++)
                    carry[j - (n_limb1 + 1)] = u[j];
            } else {
                for(j = n_limb1; j < nb_mods - 1; j++) {
                    carry[j - n_limb1] = (u[j] >> shift) |
                        (u[j + 1] << (LIMB_BITS - shift));
                }
                carry[nb_mods - 1 - n_limb1] = u[nb_mods - 1] >> shift;
            }
            i++;
        }
    }
}
#else
static no_inline void ntt_to_limb(BFNTTState *s, limb_t *tabr, limb_t r_len,
                                  const NTTLimb *buf, int fft_len_log2, int dpl,
                                  int nb_mods)
{
    const limb_t *mods = ntt_mods + NB_MODS - nb_mods;
    const limb_t *mods_cr, *mods_cr_inv;
    limb_t y[NB_MODS], u[NB_MODS], carry[NB_MODS], fft_len, base_mask1, r;
    slimb_t i, len, pos;
    int j, k, l, shift, n_limb1;
    dlimb_t t;

    j = NB_MODS * (NB_MODS - 1) / 2 - nb_mods * (nb_mods - 1) / 2;
    mods_cr = ntt_mods_cr + j;
    mods_cr_inv = s->ntt_mods_cr_inv + j;

    shift = dpl & (LIMB_BITS - 1);
    if (shift == 0)
        base_mask1 = -1;
    else
        base_mask1 = ((limb_t)1 << shift) - 1;
    n_limb1 = ((unsigned)dpl - 1) / LIMB_BITS;
    for(j = 0; j < NB_MODS; j++)
        carry[j] = 0;
    for(j = 0; j < NB_MODS; j++)
        u[j] = 0; /* avoid warnings */
    memset(tabr, 0, sizeof(limb_t) * r_len);
    fft_len = (limb_t)1 << fft_len_log2;
    len = bf_min(fft_len, (r_len * LIMB_BITS + dpl - 1) / dpl);
    for(i = 0; i < len; i++) {
        for(j = 0; j < nb_mods; j++)  {
            y[j] = ntt_limb_to_int(buf[i + fft_len * j], mods[j]);
        }

        /* Chinese remainder to get mixed radix representation */
        l = 0;
        for(j = 0; j < nb_mods - 1; j++) {
            for(k = j + 1; k < nb_mods; k++) {
                limb_t m;
                m = mods[k];
                /* Note: there is no overflow in the sub_mod() because
                   the modulos are sorted by increasing order */
                y[k] = mul_mod_fast2(y[k] - y[j] + m,
                                     mods_cr[l], m, mods_cr_inv[l]);
                l++;
            }
        }

        /* back to normal representation */
        u[0] = y[nb_mods - 1];
        l = 1;
        for(j = nb_mods - 2; j >= 1; j--) {
            r = y[j];
            for(k = 0; k < l; k++) {
                t = (dlimb_t)u[k] * mods[j] + r;
                r = t >> LIMB_BITS;
                u[k] = t;
            }
            u[l] = r;
            l++;
        }

        /* last step adds the carry */
        r = y[0];
        for(k = 0; k < l; k++) {
            t = (dlimb_t)u[k] * mods[j] + r + carry[k];
            r = t >> LIMB_BITS;
            u[k] = t;
        }
        u[l] = r + carry[l];

        /* write the digits */
        pos = i * dpl;
        for(j = 0; j < n_limb1; j++) {
            put_bits(tabr, r_len, pos, u[j]);
            pos += LIMB_BITS;
        }
        put_bits(tabr, r_len, pos, u[n_limb1] & base_mask1);
        /* shift by dpl digits and set the carry */
        if (shift == 0) {
            for(j = n_limb1 + 1; j < nb_mods; j++)
                carry[j - (n_limb1 + 1)] = u[j];
        } else {
            for(j = n_limb1; j < nb_mods - 1; j++) {
                carry[j - n_limb1] = (u[j] >> shift) |
                    (u[j + 1] << (LIMB_BITS - shift));
            }
            carry[nb_mods - 1 - n_limb1] = u[nb_mods - 1] >> shift;
        }
    }
}
#endif

static int ntt_static_init(bf_context_t *s1)
{
    BFNTTState *s;
    int inverse, i, j, k, l;
    limb_t c, c_inv, c_inv2, m, m_inv;

    if (s1->ntt_state)
        return 0;
#if defined(__AVX2__)
    s = bf_aligned_malloc(s1, sizeof(*s), 64);
#else
    s = bf_malloc(s1, sizeof(*s));
#endif
    if (!s)
        return -1;
    memset(s, 0, sizeof(*s));
    s1->ntt_state = s;
    s->ctx = s1;

    for(j = 0; j < NB_MODS; j++) {
        m = ntt_mods[j];
        m_inv = init_mul_mod_fast(m);
        s->ntt_mods_div[j] = m_inv;
#if defined(__AVX2__)
        s->ntt_mods_vec[j] = _mm256_set1_pd(m);
        s->ntt_mods_inv_vec[j] = _mm256_set1_pd(1.0 / (double)m);
#endif
        c_inv2 = (m + 1) / 2; /* 1/2 */
        c_inv = 1;
        for(i = 0; i <= NTT_PROOT_2EXP; i++) {
            s->ntt_len_inv[j][i][0] = c_inv;
            s->ntt_len_inv[j][i][1] = init_mul_mod_fast2(c_inv, m);
            c_inv = mul_mod_fast(c_inv, c_inv2, m, m_inv);
        }

        for(inverse = 0; inverse < 2; inverse++) {
            c = ntt_proot[inverse][j];
            for(i = 0; i < NTT_PROOT_2EXP; i++) {
                s->ntt_proot_pow[j][inverse][NTT_PROOT_2EXP - i] = c;
                s->ntt_proot_pow_inv[j][inverse][NTT_PROOT_2EXP - i] =
                    init_mul_mod_fast2(c, m);
                c = mul_mod_fast(c, c, m, m_inv);
            }
        }
    }

    l = 0;
    for(j = 0; j < NB_MODS - 1; j++) {
        for(k = j + 1; k < NB_MODS; k++) {
#if defined(__AVX2__)
            s->ntt_mods_cr_vec[l] = _mm256_set1_pd(int_to_ntt_limb2(ntt_mods_cr[l],
                                                                    ntt_mods[k]));
#else
            s->ntt_mods_cr_inv[l] = init_mul_mod_fast2(ntt_mods_cr[l],
                                                       ntt_mods[k]);
#endif
            l++;
        }
    }
    return 0;
}

int bf_get_fft_size(int *pdpl, int *pnb_mods, limb_t len)
{
    int dpl, fft_len_log2, n_bits, nb_mods, dpl_found, fft_len_log2_found;
    int int_bits, nb_mods_found;
    limb_t cost, min_cost;

    min_cost = -1;
    dpl_found = 0;
    nb_mods_found = 4;
    fft_len_log2_found = 0;
    for(nb_mods = 3; nb_mods <= NB_MODS; nb_mods++) {
        int_bits = ntt_int_bits[NB_MODS - nb_mods];
        dpl = bf_min((int_bits - 4) / 2,
                     2 * LIMB_BITS + 2 * NTT_MOD_LOG2_MIN - NTT_MOD_LOG2_MAX);
        for(;;) {
            fft_len_log2 = ceil_log2((len * LIMB_BITS + dpl - 1) / dpl);
            if (fft_len_log2 > NTT_PROOT_2EXP)
                goto next;
            n_bits = fft_len_log2 + 2 * dpl;
            if (n_bits <= int_bits) {
                cost = ((limb_t)(fft_len_log2 + 1) << fft_len_log2) * nb_mods;
                //                printf("n=%d dpl=%d: cost=%" PRId64 "\n", nb_mods, dpl, (int64_t)cost);
                if (cost < min_cost) {
                    min_cost = cost;
                    dpl_found = dpl;
                    nb_mods_found = nb_mods;
                    fft_len_log2_found = fft_len_log2;
                }
                break;
            }
            dpl--;
            if (dpl == 0)
                break;
        }
    next: ;
    }
    if (!dpl_found)
        abort();
    /* limit dpl if possible to reduce fixed cost of limb/NTT conversion */
    if (dpl_found > (LIMB_BITS + NTT_MOD_LOG2_MIN) &&
        ((limb_t)(LIMB_BITS + NTT_MOD_LOG2_MIN) << fft_len_log2_found) >=
        len * LIMB_BITS) {
        dpl_found = LIMB_BITS + NTT_MOD_LOG2_MIN;
    }
    *pnb_mods = nb_mods_found;
    *pdpl = dpl_found;
    return fft_len_log2_found;
}

/* return 0 if OK, -1 if memory error */
static no_inline int fft_mul(bf_context_t *s1,
                             bf_t *res, limb_t *a_tab, limb_t a_len,
                             limb_t *b_tab, limb_t b_len, int mul_flags)
{
    BFNTTState *s;
    int dpl, fft_len_log2, j, nb_mods, reduced_mem;
    slimb_t len, fft_len;
    NTTLimb *buf1, *buf2, *ptr;
#if defined(USE_MUL_CHECK)
    limb_t ha, hb, hr, h_ref;
#endif

    if (ntt_static_init(s1))
        return -1;
    s = s1->ntt_state;

    /* find the optimal number of digits per limb (dpl) */
    len = a_len + b_len;
    fft_len_log2 = bf_get_fft_size(&dpl, &nb_mods, len);
    fft_len = (uint64_t)1 << fft_len_log2;
    //    printf("len=%" PRId64 " fft_len_log2=%d dpl=%d\n", len, fft_len_log2, dpl);
#if defined(USE_MUL_CHECK)
    ha = mp_mod1(a_tab, a_len, BF_CHKSUM_MOD, 0);
    hb = mp_mod1(b_tab, b_len, BF_CHKSUM_MOD, 0);
#endif
    if ((mul_flags & (FFT_MUL_R_OVERLAP_A | FFT_MUL_R_OVERLAP_B)) == 0) {
        if (!(mul_flags & FFT_MUL_R_NORESIZE))
            bf_resize(res, 0);
    } else if (mul_flags & FFT_MUL_R_OVERLAP_B) {
        limb_t *tmp_tab, tmp_len;
        /* it is better to free 'b' first */
        tmp_tab = a_tab;
        a_tab = b_tab;
        b_tab = tmp_tab;
        tmp_len = a_len;
        a_len = b_len;
        b_len = tmp_len;
    }
    buf1 = ntt_malloc(s, sizeof(NTTLimb) * fft_len * nb_mods);
    if (!buf1)
        return -1;
    limb_to_ntt(s, buf1, fft_len, a_tab, a_len, dpl,
                NB_MODS - nb_mods, nb_mods);
    if ((mul_flags & (FFT_MUL_R_OVERLAP_A | FFT_MUL_R_OVERLAP_B)) ==
        FFT_MUL_R_OVERLAP_A) {
        if (!(mul_flags & FFT_MUL_R_NORESIZE))
            bf_resize(res, 0);
    }
    reduced_mem = (fft_len_log2 >= 14);
    if (!reduced_mem) {
        buf2 = ntt_malloc(s, sizeof(NTTLimb) * fft_len * nb_mods);
        if (!buf2)
            goto fail;
        limb_to_ntt(s, buf2, fft_len, b_tab, b_len, dpl,
                    NB_MODS - nb_mods, nb_mods);
        if (!(mul_flags & FFT_MUL_R_NORESIZE))
            bf_resize(res, 0); /* in case res == b */
    } else {
        buf2 = ntt_malloc(s, sizeof(NTTLimb) * fft_len);
        if (!buf2)
            goto fail;
    }
    for(j = 0; j < nb_mods; j++) {
        if (reduced_mem) {
            limb_to_ntt(s, buf2, fft_len, b_tab, b_len, dpl,
                        NB_MODS - nb_mods + j, 1);
            ptr = buf2;
        } else {
            ptr = buf2 + fft_len * j;
        }
        if (ntt_conv(s, buf1 + fft_len * j, ptr,
                     fft_len_log2, fft_len_log2, j + NB_MODS - nb_mods))
            goto fail;
    }
    if (!(mul_flags & FFT_MUL_R_NORESIZE))
        bf_resize(res, 0); /* in case res == b and reduced mem */
    ntt_free(s, buf2);
    buf2 = NULL;
    if (!(mul_flags & FFT_MUL_R_NORESIZE)) {
        if (bf_resize(res, len))
            goto fail;
    }
    ntt_to_limb(s, res->tab, len, buf1, fft_len_log2, dpl, nb_mods);
    ntt_free(s, buf1);
#if defined(USE_MUL_CHECK)
    hr = mp_mod1(res->tab, len, BF_CHKSUM_MOD, 0);
    h_ref = mul_mod(ha, hb, BF_CHKSUM_MOD);
    if (hr != h_ref) {
        printf("ntt_mul_error: len=%" PRId_LIMB " fft_len_log2=%d dpl=%d nb_mods=%d\n",
               len, fft_len_log2, dpl, nb_mods);
        //        printf("ha=0x" FMT_LIMB" hb=0x" FMT_LIMB " hr=0x" FMT_LIMB " expected=0x" FMT_LIMB "\n", ha, hb, hr, h_ref);
        exit(1);
    }
#endif
    return 0;
 fail:
    ntt_free(s, buf1);
    ntt_free(s, buf2);
    return -1;
}

#else /* USE_FFT_MUL */

int bf_get_fft_size(int *pdpl, int *pnb_mods, limb_t len)
{
    return 0;
}

#endif /* !USE_FFT_MUL */

#undef malloc
#undef free
#undef realloc
