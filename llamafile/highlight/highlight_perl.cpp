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

#include "highlight.h"
#include "util.h"
#include <cosmo.h>
#include <ctype.h>

enum
{
    NORMAL,
    WORD,
    QUOTE,
    QUOTE_BACKSLASH,
    DQUOTE,
    DQUOTE_BACKSLASH,
    TICK,
    TICK_BACKSLASH,
    VAR,
    VAR2,
    COMMENT,
    LT,
    LT_LT,
    LT_LT_NAME,
    LT_LT_QNAME,
    HEREDOC_BOL,
    HEREDOC,
    REGEX,
    REGEX_BACKSLASH,
    S_REGEX,
    S_REGEX_BACKSLASH,
    S_REGEX_S,
    S_REGEX_S_BACKSLASH,
    EQUAL,
    BACKSLASH,
};

enum
{
    EXPECT_VALUE,
    EXPECT_OPERATOR,
};

static int
mirror(int c)
{
    switch (c) {
    case '(':
        return ')';
    case '{':
        return '}';
    case '[':
        return ']';
    case '<':
        return '>';
    default:
        return c;
    }
}

static bool
is_magic_var(int c)
{
    switch (c) {
    case '!':
    case '"':
    case '#':
    case '&':
    case '-':
    case '/':
    case '<':
    case '=':
    case '>':
    case '?':
    case '@':
    case '\'':
    case '\\':
    case '^':
    case '_':
    case '`':
        return true;
    default:
        return false;
    }
}

static bool
is_regex_punct(int c)
{
    switch (c) {
    case '!':
    case '"':
    case '#':
    case '%':
    case '&':
    case '(':
    case '*':
    case ',':
    case '-':
    case '.':
    case '/':
    case ':':
    case ';':
    case '<':
    case '=':
    case '@':
    case '[':
    case '\'':
    case '^':
    case '`':
    case '{':
    case '|':
    case '~':
        return true;
    default:
        return false;
    }
}

static bool
is_regex_prefix(const std::string_view& s)
{
    return s == "m" || //
           s == "s" || //
           s == "y" || //
           s == "q" || //
           s == "tr" || //
           s == "qq" || //
           s == "qw" || //
           s == "qx" || //
           s == "qr";
}

static bool
is_double_regex(const std::string_view& s)
{
    return s == "s" || //
           s == "y" || //
           s == "tr";
}

HighlightPerl::HighlightPerl()
{
}

HighlightPerl::~HighlightPerl()
{
}

void
HighlightPerl::feed(std::string* r, std::string_view input)
{
    for (size_t i = 0; i < input.size(); ++i) {
        wchar_t c;
        int b = input[i] & 255;
        last_ = c_;
        if (!u_) {
            if (b < 0300) {
                c_ = c = b;
            } else {
                c_ = ThomPikeByte(b);
                u_ = ThomPikeLen(b) - 1;
                continue;
            }
        } else if (ThomPikeCont(b)) {
            c_ = c = ThomPikeMerge(c_, b);
            if (--u_)
                continue;
        } else {
            u_ = 0;
            c_ = c = b;
        }
        if (c == 0xFEFF)
            continue; // utf-8 bom
        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                lf::append_wchar(&word_, c);
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += '\'';
                expect_ = EXPECT_OPERATOR;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += '"';
                expect_ = EXPECT_OPERATOR;
            } else if (c == '=' && (!last_ || last_ == '\n')) {
                t_ = EQUAL;
            } else if (c == '\\') {
                t_ = BACKSLASH;
                *r += HI_ESCAPE;
                *r += '\\';
            } else if (c == '`') {
                t_ = TICK;
                *r += HI_STRING;
                *r += '`';
                expect_ = EXPECT_OPERATOR;
            } else if (c == '$') {
                *r += '$';
                t_ = VAR;
                expect_ = EXPECT_OPERATOR;
            } else if (c == '@' || c == '%') {
                lf::append_wchar(r, c);
                *r += HI_VAR;
                t_ = VAR2;
                expect_ = EXPECT_OPERATOR;
            } else if (c == '#') {
                *r += HI_COMMENT;
                *r += '#';
                t_ = COMMENT;
            } else if (c == '<') {
                *r += '<';
                t_ = LT;
                expect_ = EXPECT_VALUE;
            } else if (c == '/' && expect_ == EXPECT_VALUE && last_ != '/') {
                opener_ = '/';
                closer_ = '/';
                expect_ = EXPECT_OPERATOR;
                *r += HI_STRING;
                lf::append_wchar(r, c);
                t_ = REGEX;
            } else if (c == '\n') {
                *r += '\n';
                if (pending_heredoc_) {
                    *r += HI_STRING;
                    pending_heredoc_ = false;
                    t_ = HEREDOC_BOL;
                    i_ = 0;
                }
            } else if (c == ')' || c == '}' || c == ']') {
                expect_ = EXPECT_OPERATOR;
                lf::append_wchar(r, c);
            } else if (ispunct(c)) {
                expect_ = EXPECT_VALUE;
                lf::append_wchar(r, c);
            } else if (isdigit(c) || c == '.') {
                expect_ = EXPECT_OPERATOR;
                lf::append_wchar(r, c);
            } else {
                lf::append_wchar(r, c);
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                lf::append_wchar(&word_, c);
            } else {
                if (is_keyword_perl(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                    if (word_ == "shift") {
                        expect_ = EXPECT_OPERATOR;
                    } else {
                        expect_ = EXPECT_VALUE;
                    }
                } else {
                    *r += word_;
                    expect_ = EXPECT_VALUE;
                    if (is_regex_punct(c) && is_regex_prefix(word_)) {
                        opener_ = c;
                        closer_ = mirror(c);
                        *r += HI_STRING;
                        lf::append_wchar(r, c);
                        if (is_double_regex(word_)) {
                            t_ = S_REGEX;
                        } else {
                            t_ = REGEX;
                        }
                        word_.clear();
                        break;
                    }
                }
                word_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case BACKSLASH:
            lf::append_wchar(r, c);
            *r += HI_RESET;
            t_ = NORMAL;
            break;

        case VAR:
            if (isdigit(c) || is_magic_var(c)) {
                *r += HI_VAR;
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
                break;
            } else if (c == '{') {
                t_ = VAR2;
                lf::append_wchar(r, c);
                *r += HI_VAR;
                break;
            } else {
                *r += HI_VAR;
                t_ = VAR2;
            }
            // fallthrough

        case VAR2:
            if (!isascii(c) || isalnum(c) || c == '_') {
                lf::append_wchar(r, c);
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COMMENT:
            lf::append_wchar(r, c);
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case REGEX:
            lf::append_wchar(r, c);
            if (c == closer_) {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = REGEX_BACKSLASH;
            }
            break;

        case REGEX_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = REGEX;
            break;

        case S_REGEX:
            lf::append_wchar(r, c);
            if (c == opener_) {
                t_ = S_REGEX_S;
            } else if (c == '\\') {
                t_ = S_REGEX_BACKSLASH;
            }
            break;

        case S_REGEX_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = S_REGEX;
            break;

        case S_REGEX_S:
            lf::append_wchar(r, c);
            if (c == closer_) {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = S_REGEX_S_BACKSLASH;
            }
            break;

        case S_REGEX_S_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = S_REGEX_S;
            break;

        case QUOTE:
            lf::append_wchar(r, c);
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = QUOTE_BACKSLASH;
            }
            break;

        case QUOTE_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = QUOTE;
            break;

        case DQUOTE:
            lf::append_wchar(r, c);
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = DQUOTE_BACKSLASH;
            }
            break;

        case DQUOTE_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = DQUOTE;
            break;

        case TICK:
            lf::append_wchar(r, c);
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = TICK_BACKSLASH;
            }
            break;

        case TICK_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = TICK;
            break;

        case EQUAL:
            if (isalpha(c)) {
                *r += HI_COMMENT;
                *r += '=';
                lf::append_wchar(r, c);
                heredoc_ = "=cut";
                t_ = HEREDOC;
                i_ = 0;
            } else {
                *r += '=';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT:
            if (c == '<') {
                lf::append_wchar(r, c);
                t_ = LT_LT;
                heredoc_.clear();
                pending_heredoc_ = false;
                indented_heredoc_ = false;
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT:
            if (c == '-') {
                indented_heredoc_ = true;
                lf::append_wchar(r, c);
            } else if (c == '"' || c == '\'') {
                closer_ = c;
                t_ = LT_LT_QNAME;
                *r += HI_STRING;
                lf::append_wchar(r, c);
            } else if (isalpha(c) || c == '_') {
                t_ = LT_LT_NAME;
                lf::append_wchar(&heredoc_, c);
                lf::append_wchar(r, c);
            } else if (isascii(c) && isblank(c)) {
                *r += c;
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT_NAME:
            if (isalnum(c) || c == '_') {
                t_ = LT_LT_NAME;
                lf::append_wchar(&heredoc_, c);
                lf::append_wchar(r, c);
            } else if (c == '\n') {
                lf::append_wchar(r, c);
                *r += HI_STRING;
                t_ = HEREDOC_BOL;
            } else {
                pending_heredoc_ = true;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT_QNAME:
            lf::append_wchar(r, c);
            if (c == closer_) {
                *r += HI_RESET;
                t_ = HEREDOC_BOL;
                pending_heredoc_ = true;
                t_ = NORMAL;
            } else {
                lf::append_wchar(&heredoc_, c);
            }
            break;

        case HEREDOC_BOL:
            lf::append_wchar(r, c);
            if (c == '\n') {
                if (i_ == heredoc_.size()) {
                    t_ = NORMAL;
                    *r += HI_RESET;
                }
                i_ = 0;
            } else if (c == '\t' && indented_heredoc_) {
                // do nothing
            } else if (i_ < heredoc_.size() && (heredoc_[i_] & 255) == c) {
                i_++;
            } else {
                t_ = HEREDOC;
                i_ = 0;
            }
            break;

        case HEREDOC:
            lf::append_wchar(r, c);
            if (c == '\n') {
                t_ = HEREDOC_BOL;
                i_ = 0;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightPerl::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_perl(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case EQUAL:
        *r += '=';
        break;
    case VAR2:
    case TICK:
    case TICK_BACKSLASH:
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case COMMENT:
    case HEREDOC_BOL:
    case HEREDOC:
    case LT_LT_QNAME:
    case REGEX:
    case REGEX_BACKSLASH:
    case S_REGEX:
    case S_REGEX_BACKSLASH:
    case S_REGEX_S:
    case S_REGEX_S_BACKSLASH:
    case BACKSLASH:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
}
