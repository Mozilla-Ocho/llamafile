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
    QUOTE,
    QUOTE_BACKSLASH,
    DQUOTE,
    DQUOTE_BACKSLASH,
    HYPHEN,
    HYPHEN_HYPHEN,
    HYPHEN_HYPHEN_LSB,
    COMMENT,
    TICK,
    LSB,
    LITERAL,
    LITERAL_RSB,
};

HighlightLua::HighlightLua()
{
}

HighlightLua::~HighlightLua()
{
}

void
HighlightLua::feed(std::string* r, std::string_view input)
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
            } else if (c == '-') {
                t_ = HYPHEN;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '[') {
                t_ = LSB;
                level1_ = 0;
            } else {
                *r += c;
            }
            break;

        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                word_ += c;
            } else {
                if (is_keyword_lua(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_lua_constant(word_.data(),
                                                   word_.size())) {
                    *r += HI_CONSTANT;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_lua_builtin(word_.data(), word_.size())) {
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

        case HYPHEN:
            if (c == '-') {
                *r += HI_COMMENT;
                *r += "--";
                t_ = HYPHEN_HYPHEN;
            } else {
                *r += '-';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case HYPHEN_HYPHEN:
            if (c == '[') {
                *r += '[';
                t_ = HYPHEN_HYPHEN_LSB;
                level1_ = 0;
            } else {
                t_ = COMMENT;
                goto Comment;
            }
            break;

        case HYPHEN_HYPHEN_LSB:
            if (c == '=') {
                *r += '=';
                ++level1_;
            } else if (c == '[') {
                *r += '[';
                t_ = LITERAL;
            } else {
                t_ = COMMENT;
                goto Comment;
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

        case QUOTE:
            *r += c;
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = QUOTE_BACKSLASH;
            }
            break;

        case QUOTE_BACKSLASH:
            *r += c;
            t_ = QUOTE;
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

        case LSB:
            if (c == '=') {
                ++level1_;
            } else if (c == '[') {
                *r += HI_STRING;
                *r += '[';
                for (int i = 0; i < level1_; ++i)
                    *r += '=';
                *r += '[';
                t_ = LITERAL;
            } else {
                *r += '[';
                for (int i = 0; i < level1_; ++i)
                    *r += '=';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LITERAL:
            *r += c;
            if (c == ']') {
                t_ = LITERAL_RSB;
                level2_ = 0;
            }
            break;

        case LITERAL_RSB:
            *r += c;
            if (c == '=') {
                ++level2_;
            } else if (c == ']') {
                if (level2_ == level1_) {
                    *r += HI_RESET;
                    t_ = NORMAL;
                } else {
                    level2_ = 0;
                }
            } else {
                t_ = LITERAL;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightLua::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_lua(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_lua_constant(word_.data(), word_.size())) {
            *r += HI_CONSTANT;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_lua_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case LSB:
        *r += '[';
        for (int i = 0; i < level1_; ++i)
            *r += '=';
        break;
    case HYPHEN:
        *r += '-';
        break;
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case COMMENT:
    case LITERAL:
    case LITERAL_RSB:
    case HYPHEN_HYPHEN:
    case HYPHEN_HYPHEN_LSB:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
