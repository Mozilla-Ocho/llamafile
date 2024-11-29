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

#include <ctype.h>

enum
{
    NORMAL,
    SYMBOL,
    DQUOTE,
    DQUOTE_BACKSLASH,
    COMMENT,
};

HighlightLisp::HighlightLisp()
{
}

HighlightLisp::~HighlightLisp()
{
}

void
HighlightLisp::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;
        switch (t_) {

        Normal:
        case NORMAL:
            if (c == '(') {
                *r += c;
                is_first_ = true;
            } else if (c == ';') {
                *r += HI_COMMENT;
                *r += c;
                t_ = COMMENT;
            } else if (c == '[') {
                *r += c;
                t_ = SYMBOL;
                is_first_ = false;
            } else if (c == ')' || c == ']') {
                *r += c;
                is_first_ = false;
            } else if (c == '\'' || c == '#' || c == '`' || c == ',') {
                *r += c;
                is_first_ = false;
            } else if (c == '"') {
                *r += HI_STRING;
                *r += c;
                t_ = DQUOTE;
                is_first_ = false;
            } else if (isspace(c)) {
                *r += c;
            } else {
                symbol_ += c;
                t_ = SYMBOL;
            }
            break;

        case SYMBOL:
            if (isspace(c) || //
                c == '(' || //
                c == ')' || //
                c == '[' || //
                c == ']' || //
                c == ',' || //
                c == '#' || //
                c == '`' || //
                c == '"' || //
                c == '\'') {
                if (is_first_ &&
                    is_keyword_lisp(symbol_.data(), symbol_.size())) {
                    *r += HI_KEYWORD;
                    *r += symbol_;
                    *r += HI_RESET;
                } else if (symbol_.size() > 1 && symbol_[0] == ':') {
                    *r += HI_LISPKW;
                    *r += symbol_;
                    *r += HI_RESET;
                } else {
                    *r += symbol_;
                }
                is_first_ = false;
                symbol_.clear();
                t_ = NORMAL;
                goto Normal;
            } else {
                symbol_ += c;
            }
            break;

        case DQUOTE:
            *r += c;
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = DQUOTE_BACKSLASH;
            }
            break;

        case DQUOTE_BACKSLASH:
            *r += c;
            t_ = DQUOTE;
            break;

        case COMMENT:
            *r += c;
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightLisp::flush(std::string* r)
{
    switch (t_) {
    case SYMBOL:
        if (is_first_ && is_keyword_lisp(symbol_.data(), symbol_.size())) {
            *r += HI_KEYWORD;
            *r += symbol_;
            *r += HI_RESET;
        } else if (symbol_.size() > 1 && symbol_[0] == ':') {
            *r += HI_LISPKW;
            *r += symbol_;
            *r += HI_RESET;
        } else {
            *r += symbol_;
        }
        symbol_.clear();
        break;
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case COMMENT:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    is_first_ = false;
    t_ = NORMAL;
}
