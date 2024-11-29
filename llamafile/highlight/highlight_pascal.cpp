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

// syntax highlighting for pascal (european c)
//
// this syntax highlighter aims to support a blended dialect of
//
// - pascal
// - delphi
//
// doing that requires extra keywords

enum
{
    NORMAL,
    WORD,
    QUOTE,
    DQUOTE,
    SLASH,
    SLASH_SLASH,
    CURLY,
    PAREN,
    PAREN_STAR,
    PAREN_STAR_STAR,
};

HighlightPascal::HighlightPascal()
{
}

HighlightPascal::~HighlightPascal()
{
}

void
HighlightPascal::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;

        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                goto Word;
            } else if (c == '/') {
                t_ = SLASH;
            } else if (c == '(') {
                t_ = PAREN;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '{') {
                t_ = CURLY;
                *r += HI_COMMENT;
                *r += c;
            } else {
                *r += c;
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                word_ += c;
            } else {
                if (is_keyword_pascal(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_pascal_type(word_.data(), word_.size())) {
                    *r += HI_TYPE;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_pascal_builtin(word_.data(),
                                                     word_.size())) {
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

        case SLASH:
            if (c == '/') {
                *r += HI_COMMENT;
                *r += "//";
                t_ = SLASH_SLASH;
            } else {
                *r += '/';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case SLASH_SLASH:
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

        case CURLY:
            *r += c;
            if (c == '}') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case PAREN:
            if (c == '*') {
                *r += HI_COMMENT;
                *r += "(*";
                t_ = PAREN_STAR;
            } else {
                *r += '(';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case PAREN_STAR:
            *r += c;
            if (c == '*')
                t_ = PAREN_STAR_STAR;
            break;

        case PAREN_STAR_STAR:
            *r += c;
            if (c == ')') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c != '*') {
                t_ = PAREN_STAR;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightPascal::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_pascal(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_pascal_type(word_.data(), word_.size())) {
            *r += HI_TYPE;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_pascal_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case SLASH:
        *r += '/';
        break;
    case PAREN:
        *r += '(';
        break;
    case QUOTE:
    case DQUOTE:
    case SLASH_SLASH:
    case CURLY:
    case PAREN_STAR:
    case PAREN_STAR_STAR:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
