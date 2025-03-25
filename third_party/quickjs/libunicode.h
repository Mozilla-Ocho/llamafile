/*
 * Unicode utilities
 *
 * Copyright (c) 2017-2018 Fabrice Bellard
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
#ifndef LIBUNICODE_H
#define LIBUNICODE_H

#include <stddef.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LRE_BOOL  int       /* for documentation purposes */

#define LRE_CC_RES_LEN_MAX 3

typedef enum {
    UNICODE_NFC,
    UNICODE_NFD,
    UNICODE_NFKC,
    UNICODE_NFKD,
} UnicodeNormalizationEnum;

int lre_case_conv(uint32_t *res, uint32_t c, int conv_type);
int lre_canonicalize(uint32_t c, BOOL is_unicode);
LRE_BOOL lre_is_cased(uint32_t c);
LRE_BOOL lre_is_case_ignorable(uint32_t c);

/* char ranges */

typedef struct {
    int len; /* in points, always even */
    int size;
    uint32_t *points; /* points sorted by increasing value */
    void *mem_opaque;
    void *(*realloc_func)(void *opaque, void *ptr, size_t size);
} CharRange;

typedef enum {
    CR_OP_UNION,
    CR_OP_INTER,
    CR_OP_XOR,
} CharRangeOpEnum;

void cr_init(CharRange *cr, void *mem_opaque, void *(*realloc_func)(void *opaque, void *ptr, size_t size));
void cr_free(CharRange *cr);
int cr_realloc(CharRange *cr, int size);
int cr_copy(CharRange *cr, const CharRange *cr1);

static inline int cr_add_point(CharRange *cr, uint32_t v)
{
    if (cr->len >= cr->size) {
        if (cr_realloc(cr, cr->len + 1))
            return -1;
    }
    cr->points[cr->len++] = v;
    return 0;
}

static inline int cr_add_interval(CharRange *cr, uint32_t c1, uint32_t c2)
{
    if ((cr->len + 2) > cr->size) {
        if (cr_realloc(cr, cr->len + 2))
            return -1;
    }
    cr->points[cr->len++] = c1;
    cr->points[cr->len++] = c2;
    return 0;
}

int cr_union1(CharRange *cr, const uint32_t *b_pt, int b_len);

static inline int cr_union_interval(CharRange *cr, uint32_t c1, uint32_t c2)
{
    uint32_t b_pt[2];
    b_pt[0] = c1;
    b_pt[1] = c2 + 1;
    return cr_union1(cr, b_pt, 2);
}

int cr_op(CharRange *cr, const uint32_t *a_pt, int a_len,
          const uint32_t *b_pt, int b_len, int op);

int cr_invert(CharRange *cr);
int cr_regexp_canonicalize(CharRange *cr, BOOL is_unicode);

LRE_BOOL lre_is_id_start(uint32_t c);
LRE_BOOL lre_is_id_continue(uint32_t c);
LRE_BOOL lre_is_white_space(uint32_t c);

int unicode_normalize(uint32_t **pdst, const uint32_t *src, int src_len,
                      UnicodeNormalizationEnum n_type,
                      void *opaque, void *(*realloc_func)(void *opaque, void *ptr, size_t size));

/* Unicode character range functions */

int unicode_script(CharRange *cr,
                   const char *script_name, LRE_BOOL is_ext);
int unicode_general_category(CharRange *cr, const char *gc_name);
int unicode_prop(CharRange *cr, const char *prop_name);

#undef LRE_BOOL

#ifdef __cplusplus
} /* extern "C" { */
#endif

#endif /* LIBUNICODE_H */
