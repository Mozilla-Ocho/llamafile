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
#include <cctype>

// beginner's all-purpose symbolic instruction code
//
// this syntax highlighter aims to support a blended dialect of
//
// - basic
// - visual basic
// - visual basic .net
//
// doing that requires extra keywords

enum
{
    NORMAL,
    WORD,
    LINENO,
    DQUOTE,
    COMMENT,
    DIRECTIVE,
};

static std::string
ToLower(const std::string_view& s)
{
    std::string b;
    for (char c : s)
        b += std::tolower(c);
    return b;
}

HighlightBasic::HighlightBasic()
{
}

HighlightBasic::~HighlightBasic()
{
}

void
HighlightBasic::feed(std::string* r, std::string_view input)
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
            } else if (c == '\'') {
                t_ = COMMENT;
                *r += HI_COMMENT;
                *r += '\'';
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += '"';
            } else if (c == '#' && is_bol_) {
                t_ = DIRECTIVE;
                *r += HI_DIRECTIVE;
                *r += '#';
            } else if (isdigit(c) && is_bol_) {
                *r += HI_LINENO;
                *r += c;
                t_ = LINENO;
            } else {
                *r += c;
            }
            break;

        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                word_ += c;
            } else {
                if (is_keyword_basic(word_.data(), word_.size())) {
                    if (ToLower(word_) == "rem") {
                        *r += HI_COMMENT;
                        *r += word_;
                        t_ = COMMENT;
                        word_.clear();
                        goto Comment;
                    } else {
                        *r += HI_KEYWORD;
                        *r += word_;
                        *r += HI_RESET;
                    }
                } else if (is_keyword_basic_type(word_.data(), word_.size())) {
                    *r += HI_TYPE;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_basic_builtin(word_.data(),
                                                    word_.size())) {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_basic_constant(word_.data(),
                                                     word_.size())) {
                    *r += HI_CONSTANT;
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

        case DQUOTE:
            *r += c;
            if (c == '"' || c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case LINENO:
            if (isdigit(c)) {
                *r += c;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        Comment:
        case COMMENT:
        case DIRECTIVE:
            *r += c;
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
            }
            break;

        default:
            __builtin_unreachable();
        }
        if (is_bol_) {
            if (!isspace(c))
                is_bol_ = false;
        } else {
            if (c == '\n')
                is_bol_ = true;
        }
    }
}

void
HighlightBasic::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_basic(word_.data(), word_.size())) {
            if (ToLower(word_) == "rem") {
                *r += HI_KEYWORD;
                *r += word_;
                *r += HI_RESET;
            } else {
                *r += HI_KEYWORD;
                *r += word_;
                *r += HI_RESET;
            }
        } else if (is_keyword_basic_type(word_.data(), word_.size())) {
            *r += HI_TYPE;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_basic_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_basic_constant(word_.data(), word_.size())) {
            *r += HI_CONSTANT;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case LINENO:
    case DQUOTE:
    case COMMENT:
    case DIRECTIVE:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    is_bol_ = true;
    t_ = NORMAL;
}
