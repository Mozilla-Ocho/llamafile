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
#define COM 2

#define SQUOTE 3 // '
#define SQUOTESTR 4 // '...
#define SQUOTE2 5 // ''
#define SQUOTE3 6 // '''...
#define SQUOTE31 7 // '''...'
#define SQUOTE32 8 // '''...''

#define DQUOTE 9 // "
#define DQUOTESTR 10 // "...
#define DQUOTE2 11 // ""
#define DQUOTE3 12 // """...
#define DQUOTE31 13 // """..."
#define DQUOTE32 14 // """...""

#define BACKSLASH 64

HighlightPython::HighlightPython() {
}

HighlightPython::~HighlightPython() {
}

void HighlightPython::feed(std::string *r, std::string_view input) {
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
            if (!isascii(c) || isalpha(c) || isdigit(c) || c == '_') {
                word_ += c;
            } else {
                if (is_keyword_python(word_.data(), word_.size())) {
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

        case COM:
            if (c == '\n') {
                *r += HI_RESET;
                *r += c;
                t_ = NORMAL;
            } else {
                *r += c;
            }
            break;

        // handle 'string'
        case SQUOTE:
            *r += c;
            if (c == '\'') {
                t_ = SQUOTE2;
            } else {
                t_ = SQUOTESTR;
            }
            break;
        case SQUOTESTR:
            *r += c;
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
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
            if (c == '\'')
                t_ = SQUOTE31;
            break;
        case SQUOTE31:
            *r += c;
            if (c == '\'') {
                t_ = SQUOTE32;
            } else {
                t_ = SQUOTE3;
            }
            break;
        case SQUOTE32:
            *r += c;
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                t_ = SQUOTE3;
            }
            break;

        // handle "string"
        case DQUOTE:
            *r += c;
            if (c == '"') {
                t_ = DQUOTE2;
            } else {
                t_ = DQUOTESTR;
            }
            break;
        case DQUOTESTR:
            *r += c;
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        // handle """string"""
        case DQUOTE2:
            if (c == '"') {
                *r += c;
                t_ = DQUOTE3;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;
        case DQUOTE3:
            *r += c;
            if (c == '"')
                t_ = DQUOTE31;
            break;
        case DQUOTE31:
            *r += c;
            if (c == '"') {
                t_ = DQUOTE32;
            } else {
                t_ = DQUOTE3;
            }
            break;
        case DQUOTE32:
            *r += c;
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                t_ = DQUOTE3;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void HighlightPython::flush(std::string *r) {
    t_ &= ~BACKSLASH;
    switch (t_) {
    case WORD:
        if (is_keyword_python(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
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
    case SQUOTE2:
    case SQUOTE3:
    case SQUOTE31:
    case SQUOTE32:
    case DQUOTE:
    case DQUOTESTR:
    case DQUOTE2:
    case DQUOTE3:
    case DQUOTE31:
    case DQUOTE32:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
