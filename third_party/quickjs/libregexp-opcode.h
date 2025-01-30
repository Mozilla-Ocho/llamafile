/*
 * Regular Expression Engine
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

#ifdef DEF

DEF(invalid, 1) /* never used */
DEF(char8, 2) /* 7 bits in fact */
DEF(char16, 3)
DEF(char32, 5)
DEF(dot, 1)
DEF(any, 1) /* same as dot but match any character including line terminator */
DEF(line_start, 1)
DEF(line_end, 1)
DEF(goto, 5)
DEF(split_goto_first, 5)
DEF(split_next_first, 5)
DEF(match, 1)
DEF(save_start, 2) /* save start position */
DEF(save_end, 2) /* save end position, must come after saved_start */
DEF(save_reset, 3) /* reset save positions */
DEF(loop, 5) /* decrement the top the stack and goto if != 0 */
DEF(push_i32, 5) /* push integer on the stack */
DEF(drop, 1)
DEF(word_boundary, 1)
DEF(not_word_boundary, 1)
DEF(back_reference, 2)
DEF(backward_back_reference, 2) /* must come after back_reference */
DEF(range, 3) /* variable length */
DEF(range32, 3) /* variable length */
DEF(lookahead, 5)
DEF(negative_lookahead, 5)
DEF(push_char_pos, 1) /* push the character position on the stack */
DEF(check_advance, 1) /* pop one stack element and check that it is different from the character position */
DEF(prev, 1) /* go to the previous char */
DEF(simple_greedy_quant, 17)

#endif /* DEF */
