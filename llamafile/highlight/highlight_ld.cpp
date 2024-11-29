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
    DQUOTE,
    DQUOTE_BACKSLASH,
    SLASH,
    SLASH_SLASH,
    SLASH_STAR,
    SLASH_STAR_STAR,
};

HighlightLd::HighlightLd()
{
}

HighlightLd::~HighlightLd()
{
}

void
HighlightLd::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;
        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c) || c == '_' || c == '.' || c == ':') {
                t_ = WORD;
                word_ += c;
            } else if (c == '/') {
                t_ = SLASH;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '#' && is_bol_) {
                is_cpp_ = true;
                *r += HI_BUILTIN;
                *r += c;
            } else if (c == '\n') {
                *r += c;
                if (is_cpp_) {
                    *r += HI_RESET;
                    is_cpp_ = false;
                }
            } else {
                *r += c;
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_' || c == '.' || c == '/' ||
                c == ':') {
                word_ += c;
            } else {
                if (is_cpp_) {
                    if (is_keyword_cpp(word_.data(), word_.size())) {
                        *r += HI_BUILTIN;
                        *r += word_;
                        *r += HI_RESET;
                    } else if (is_keyword_c_constant(word_.data(),
                                                     word_.size())) {
                        *r += HI_CONSTANT;
                        *r += word_;
                        *r += HI_RESET;
                    } else {
                        *r += word_;
                    }
                } else {
                    if (is_keyword_ld(word_.data(), word_.size())) {
                        *r += HI_KEYWORD;
                        *r += word_;
                        *r += HI_RESET;
                    } else if (is_keyword_ld_builtin(word_.data(),
                                                     word_.size())) {
                        *r += HI_BUILTIN;
                        *r += word_;
                        *r += HI_RESET;
                    } else if (is_keyword_ld_warning(word_.data(),
                                                     word_.size())) {
                        *r += HI_WARNING;
                        *r += word_;
                        *r += HI_RESET;
                    } else {
                        *r += word_;
                    }
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
            } else if (c == 'D') {
                // for /DISCARD/ warning keyword
                word_ += "/D";
                t_ = WORD;
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
                is_cpp_ = false;
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
HighlightLd::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_cpp_) {
            if (is_keyword_cpp(word_.data(), word_.size())) {
                *r += HI_BUILTIN;
                *r += word_;
                *r += HI_RESET;
            } else if (is_keyword_c_constant(word_.data(), word_.size())) {
                *r += HI_CONSTANT;
                *r += word_;
                *r += HI_RESET;
            } else {
                *r += word_;
                *r += HI_RESET;
            }
        } else {
            if (is_keyword_ld(word_.data(), word_.size())) {
                *r += HI_KEYWORD;
                *r += word_;
                *r += HI_RESET;
            } else if (is_keyword_ld_builtin(word_.data(), word_.size())) {
                *r += HI_BUILTIN;
                *r += word_;
                *r += HI_RESET;
            } else if (is_keyword_ld_warning(word_.data(), word_.size())) {
                *r += HI_WARNING;
                *r += word_;
                *r += HI_RESET;
            } else {
                *r += word_;
            }
        }
        word_.clear();
        break;
    case SLASH:
        *r += '/';
        if (is_cpp_)
            *r += HI_RESET;
        break;
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case SLASH_SLASH:
    case SLASH_STAR:
    case SLASH_STAR_STAR:
        *r += HI_RESET;
        break;
    default:
        if (is_cpp_)
            *r += HI_RESET;
        break;
    }
    is_cpp_ = false;
    is_bol_ = true;
    t_ = NORMAL;
}
