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
    COM,
    SQUOTE,
    SQUOTESTR,
    SQUOTESTR_BACKSLASH,
    SQUOTE2,
    SQUOTE3,
    SQUOTE3_BACKSLASH,
    SQUOTE31,
    SQUOTE32,
    DQUOTE,
    DQUOTESTR,
    DQUOTESTR_BACKSLASH,
    DQUOTE2,
    DQUOTE3,
    DQUOTE3_BACKSLASH,
    DQUOTE31,
    DQUOTE32,
};

HighlightPython::HighlightPython()
{
}

HighlightPython::~HighlightPython()
{
}

void
HighlightPython::feed(std::string* r, std::string_view input)
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
            } else if (c == '#') {
                t_ = COM;
                *r += HI_COMMENT;
                *r += c;
            } else if (c == '\'') {
                t_ = SQUOTE;
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

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                word_ += c;
            } else {
                if (is_keyword_python(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_python_builtin(word_.data(),
                                                     word_.size())) {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_python_constant(word_.data(),
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

        case COM:
            *r += c;
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        // handle 'string'
        case SQUOTE:
            *r += c;
            if (c == '\'') {
                t_ = SQUOTE2;
            } else if (c == '\\') {
                t_ = SQUOTESTR_BACKSLASH;
            } else {
                t_ = SQUOTESTR;
            }
            break;
        case SQUOTESTR:
            *r += c;
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = SQUOTESTR_BACKSLASH;
            }
            break;
        case SQUOTESTR_BACKSLASH:
            *r += c;
            t_ = SQUOTESTR;
            break;

        // handle '''string'''
        case SQUOTE2:
            if (c == '\'') {
                *r += c;
                t_ = SQUOTE3;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case SQUOTE3:
            *r += c;
            if (c == '\'') {
                t_ = SQUOTE31;
            } else if (c == '\\') {
                t_ = SQUOTE3_BACKSLASH;
            }
            break;

        case SQUOTE31:
            *r += c;
            if (c == '\'') {
                t_ = SQUOTE32;
            } else if (c == '\\') {
                t_ = SQUOTE3_BACKSLASH;
            } else {
                t_ = SQUOTE3;
            }
            break;

        case SQUOTE32:
            *r += c;
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = SQUOTE3_BACKSLASH;
            } else {
                t_ = SQUOTE3;
            }
            break;

        case SQUOTE3_BACKSLASH:
            *r += c;
            t_ = SQUOTE3;
            break;

        // handle "string"
        case DQUOTE:
            *r += c;
            if (c == '"') {
                t_ = DQUOTE2;
            } else if (c == '\\') {
                t_ = DQUOTESTR_BACKSLASH;
            } else {
                t_ = DQUOTESTR;
            }
            break;

        case DQUOTESTR:
            *r += c;
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = DQUOTESTR_BACKSLASH;
            }
            break;

        case DQUOTESTR_BACKSLASH:
            *r += c;
            t_ = DQUOTESTR;
            break;

        // handle """string"""
        case DQUOTE2:
            if (c == '"') {
                *r += '"';
                t_ = DQUOTE3;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case DQUOTE3:
            *r += c;
            if (c == '"') {
                t_ = DQUOTE31;
            } else if (c == '\\') {
                t_ = DQUOTE3_BACKSLASH;
            }
            break;

        case DQUOTE31:
            *r += c;
            if (c == '"') {
                t_ = DQUOTE32;
            } else if (c == '\\') {
                t_ = DQUOTE3_BACKSLASH;
            } else {
                t_ = DQUOTE3;
            }
            break;

        case DQUOTE32:
            *r += c;
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = DQUOTE3_BACKSLASH;
            } else {
                t_ = DQUOTE3;
            }
            break;

        case DQUOTE3_BACKSLASH:
            *r += c;
            t_ = DQUOTE3;
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightPython::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_python(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_python_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_python_constant(word_.data(), word_.size())) {
            *r += HI_CONSTANT;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case COM:
    case SQUOTE:
    case SQUOTESTR:
    case SQUOTESTR_BACKSLASH:
    case SQUOTE2:
    case SQUOTE3:
    case SQUOTE3_BACKSLASH:
    case SQUOTE31:
    case SQUOTE32:
    case DQUOTE:
    case DQUOTESTR:
    case DQUOTESTR_BACKSLASH:
    case DQUOTE2:
    case DQUOTE3:
    case DQUOTE3_BACKSLASH:
    case DQUOTE31:
    case DQUOTE32:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
