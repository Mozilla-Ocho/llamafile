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
    WORD,
    COMMENT,
    DOLLAR,
};

HighlightM4::HighlightM4()
{
}

HighlightM4::~HighlightM4()
{
}

void
HighlightM4::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;
        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                word_ += c;
            } else if (c == '#') {
                t_ = COMMENT;
                *r += HI_COMMENT;
                *r += '#';
            } else if (c == '$') {
                t_ = DOLLAR;
            } else {
                *r += c;
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                word_ += c;
            } else {
                if (is_keyword_m4(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (word_ == "dnl" || word_ == "m4_dnl") {
                    *r += HI_COMMENT;
                    *r += word_;
                    word_.clear();
                    t_ = COMMENT;
                    goto Comment;
                } else {
                    *r += word_;
                }
                word_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case DOLLAR:
            if (isdigit(c) || c == '*' || c == '#' || c == '@') {
                *r += '$';
                *r += HI_VAR;
                *r += c;
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += '$';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        Comment:
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
HighlightM4::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_m4(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case DOLLAR:
        *r += '$';
        break;
    case COMMENT:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
