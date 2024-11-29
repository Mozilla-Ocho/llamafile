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
    QUOTE,
    DQUOTE,
    HYPHEN,
    COMMENT,
};

HighlightAda::HighlightAda()
{
}

HighlightAda::~HighlightAda()
{
}

void
HighlightAda::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        last_ = c_;
        c_ = c = input[i] & 255;

        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c)) {
                t_ = SYMBOL;
                goto Symbol;
            } else if (c == '-') {
                t_ = HYPHEN;
            } else if (c == '\'' && last_ != ')') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else {
                *r += c;
            }
            break;

        Symbol:
        case SYMBOL:
            if (!isascii(c) || isalnum(c) || c == '_' || c == '\'') {
                symbol_ += c;
            } else {
                if (is_keyword_ada(symbol_.data(), symbol_.size())) {
                    *r += HI_KEYWORD;
                    *r += symbol_;
                    *r += HI_RESET;
                } else if (is_keyword_ada_constant(symbol_.data(),
                                                   symbol_.size())) {
                    *r += HI_CONSTANT;
                    *r += symbol_;
                    *r += HI_RESET;
                } else {
                    *r += symbol_;
                }
                symbol_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case HYPHEN:
            if (c == '-') {
                *r += HI_COMMENT;
                *r += "--";
                t_ = COMMENT;
            } else {
                *r += '-';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COMMENT:
            *r += c;
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case QUOTE:
            *r += c;
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case DQUOTE:
            *r += c;
            if (c == '"') {
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
HighlightAda::flush(std::string* r)
{
    switch (t_) {
    case SYMBOL:
        if (is_keyword_ada(symbol_.data(), symbol_.size())) {
            *r += HI_KEYWORD;
            *r += symbol_;
            *r += HI_RESET;
        } else if (is_keyword_ada_constant(symbol_.data(), symbol_.size())) {
            *r += HI_CONSTANT;
            *r += symbol_;
            *r += HI_RESET;
        } else {
            *r += symbol_;
        }
        symbol_.clear();
        break;
    case HYPHEN:
        *r += '-';
        break;
    case QUOTE:
    case DQUOTE:
    case COMMENT:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
