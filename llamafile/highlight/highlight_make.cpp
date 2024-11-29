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
    COMMENT,
    COMMENT_BACKSLASH,
    DOLLAR,
    DOLLAR2,
    VARIABLE,
    BACKSLASH,
};

static bool
is_automatic_variable(int c)
{
    switch (c) {
    case '@':
    case '%':
    case '<':
    case '?':
    case '^':
    case '+':
    case '|':
    case '*':
        return true;
    default:
        return false;
    }
}

HighlightMake::HighlightMake()
{
}

HighlightMake::~HighlightMake()
{
}

void
HighlightMake::feed(std::string* r, std::string_view input)
{
    for (size_t i = 0; i < input.size(); ++i) {
        wchar_t c;
        int b = input[i] & 255;
        if (!u_) {
            if (b < 0300) {
                c = b;
            } else {
                c_ = ThomPikeByte(b);
                u_ = ThomPikeLen(b) - 1;
                continue;
            }
        } else if (ThomPikeCont(b)) {
            c = c_ = ThomPikeMerge(c_, b);
            if (--u_)
                continue;
        } else {
            u_ = 0;
            c = b;
        }
        if (c == 0xFEFF)
            continue; // utf-8 bom
        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c) || c == '_' || c == '-' || c == '.') {
                t_ = WORD;
                lf::append_wchar(&word_, c);
                break;
            } else if (c == '#') {
                t_ = COMMENT;
                *r += HI_COMMENT;
                *r += '#';
            } else if (c == '$') {
                t_ = DOLLAR;
                *r += '$';
            } else if (c == '\\') {
                t_ = BACKSLASH;
                *r += HI_ESCAPE;
                *r += '\\';
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case BACKSLASH:
            lf::append_wchar(r, c);
            *r += HI_RESET;
            t_ = NORMAL;
            break;

        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_' || c == '-' || c == '.') {
                lf::append_wchar(&word_, c);
            } else {
                if (is_keyword_make(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else {
                    *r += word_;
                }
                word_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case DOLLAR:
            if (isdigit(c) || is_automatic_variable(c)) {
                *r += HI_VAR;
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '$') {
                t_ = DOLLAR2;
                *r += '$';
            } else if (c == '(') {
                t_ = VARIABLE;
                *r += '(';
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case DOLLAR2:
            if (c == '(') {
                t_ = VARIABLE;
                *r += '(';
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case VARIABLE:
            if (isalnum(c) || //
                c == '%' || //
                c == '*' || //
                c == '+' || //
                c == '-' || //
                c == '.' || //
                c == '<' || //
                c == '?' || //
                c == '@' || //
                c == '_') {
                lf::append_wchar(&word_, c);
            } else if (c == '$' && word_.empty()) {
                t_ = DOLLAR;
                *r += '$';
            } else if (c == ')' || //
                       c == ':') {
                *r += HI_VAR;
                *r += word_;
                *r += HI_RESET;
                word_.clear();
                lf::append_wchar(r, c);
                t_ = NORMAL;
            } else {
                if (is_keyword_make_builtin(word_.data(), word_.size())) {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                } else {
                    *r += word_;
                }
                word_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COMMENT:
            lf::append_wchar(r, c);
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = COMMENT_BACKSLASH;
            }
            break;

        case COMMENT_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = COMMENT;
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightMake::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_make(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case VARIABLE:
        if (is_keyword_make_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case COMMENT:
    case COMMENT_BACKSLASH:
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
