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

#define NORMAL 0
#define WORD 1
#define QUOTE 2
#define DQUOTE 3
#define HYPHEN 4
#define COMMENT 5
#define TICK 6
#define LSB 7
#define LITERAL 8
#define LITERAL_RSB 9
#define BACKSLASH 64

HighlightLua::HighlightLua() {
}

HighlightLua::~HighlightLua() {
}

void HighlightLua::feed(std::string *r, std::string_view input) {
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;

        if (t_ & BACKSLASH) {
            t_ &= ~BACKSLASH;
            *r += c;
            continue;
        } else if (c == '\\') {
            *r += c;
            t_ |= BACKSLASH;
            continue;
        }

        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                goto Word;
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

        Word:
        case WORD:
            if (!isascii(c) || isalpha(c) || isdigit(c) || c == '_') {
                word_ += c;
            } else {
                if (is_keyword_lua(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
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
                t_ = COMMENT;
            } else {
                *r += '/';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COMMENT:
            if (c == '\n') {
                *r += HI_RESET;
                *r += c;
                t_ = NORMAL;
            } else {
                *r += c;
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
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void HighlightLua::flush(std::string *r) {
    t_ &= ~BACKSLASH;
    switch (t_) {
    case WORD:
        if (is_keyword_lua(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
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
    case HYPHEN:
        *r += '-';
        break;
    case QUOTE:
    case DQUOTE:
    case COMMENT:
    case LITERAL:
    case LITERAL_RSB:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
