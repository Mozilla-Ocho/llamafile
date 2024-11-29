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
    ANNOTATION,
    ANNOTATION2,
    SLASH,
    SLASH_SLASH,
    SLASH_STAR,
    SLASH_STAR_STAR,
    DQUOTE, // "
    DQUOTESTR, // "...
    DQUOTESTR_BACKSLASH, // "...
    DQUOTE2, // ""
    DQUOTE3, // """...
    DQUOTE3_BACKSLASH,
    DQUOTE31, // """..."
    DQUOTE32, // """...""
};

HighlightJava::HighlightJava()
{
}

HighlightJava::~HighlightJava()
{
}

void
HighlightJava::feed(std::string* r, std::string_view input)
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
            } else if (c == '/') {
                t_ = SLASH;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += '\'';
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += '"';
            } else if (c == '@') {
                t_ = ANNOTATION;
            } else {
                *r += c;
            }
            break;

        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                word_ += c;
            } else {
                if (is_keyword_java(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_java_constant(word_.data(),
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

        case SLASH:
            if (c == '/') {
                *r += HI_COMMENT;
                *r += "//";
                t_ = SLASH_SLASH;
            } else if (c == '*') {
                *r += HI_COMMENT;
                *r += "/*";
                t_ = SLASH_STAR;
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

        case SLASH_STAR:
            *r += c;
            if (c == '*')
                t_ = SLASH_STAR_STAR;
            break;

        case SLASH_STAR_STAR:
            *r += c;
            if (c == '/') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c != '*') {
                t_ = SLASH_STAR;
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

        // handle """string""" from java 15+
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

        case ANNOTATION:
            if (!isascii(c) || isalpha(c) || c == '_') {
                *r += HI_ATTRIB;
                *r += '@';
                *r += c;
                t_ = ANNOTATION2;
            } else {
                *r += '@';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case ANNOTATION2:
            if (!isascii(c) || isalnum(c) || c == '_') {
                *r += c;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightJava::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_java(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_java_constant(word_.data(), word_.size())) {
            *r += HI_CONSTANT;
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
    case ANNOTATION:
        *r += '@';
        break;
    case QUOTE:
    case QUOTE_BACKSLASH:
    case SLASH_SLASH:
    case SLASH_STAR:
    case SLASH_STAR_STAR:
    case DQUOTE:
    case DQUOTESTR:
    case DQUOTESTR_BACKSLASH:
    case DQUOTE2:
    case DQUOTE3:
    case DQUOTE3_BACKSLASH:
    case DQUOTE31:
    case DQUOTE32:
    case ANNOTATION2:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
