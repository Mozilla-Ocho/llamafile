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
    QUOTE,
    QUOTE_BACKSLASH,
    DQUOTE,
    DQUOTE_BACKSLASH,
    HYPHEN,
    HYPHEN_GT,
    LT,
    LT_LT,
    COLON,
};

HighlightR::HighlightR()
{
}

HighlightR::~HighlightR()
{
}

void
HighlightR::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;
        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c)) {
                t_ = WORD;
                word_ += c;
            } else if (c == '#') {
                *r += HI_COMMENT;
                *r += '#';
                t_ = COMMENT;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '-') {
                t_ = HYPHEN;
            } else if (c == '<') {
                t_ = LT;
            } else if (c == ':') {
                t_ = COLON;
            } else if (c == '$' || c == '@') {
                *r += HI_OPERATOR;
                *r += c;
                *r += HI_RESET;
            } else {
                *r += c;
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_' || c == '.') {
                word_ += c;
            } else {
                if (is_keyword_r(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_r_builtin(word_.data(), word_.size())) {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_r_constant(word_.data(), word_.size())) {
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

        case COLON:
            if (c == ':') {
                *r += HI_OPERATOR;
                *r += "::";
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += ':';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT:
            if (c == '<') {
                t_ = LT_LT;
            } else if (c == '-') {
                *r += HI_OPERATOR;
                *r += "<-";
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += '<';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case HYPHEN:
            if (c == '>') {
                t_ = HYPHEN_GT;
            } else {
                *r += '-';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT:
            if (c == '-') {
                *r += HI_OPERATOR;
                *r += "<<-";
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += "<<";
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case HYPHEN_GT:
            if (c == '>') {
                *r += HI_OPERATOR;
                *r += "->>";
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += HI_OPERATOR;
                *r += "->";
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
HighlightR::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_r(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_r_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_r_constant(word_.data(), word_.size())) {
            *r += HI_CONSTANT;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case HYPHEN:
        *r += '-';
        break;
    case HYPHEN_GT:
        *r += "->";
        break;
    case LT:
        *r += '<';
        break;
    case LT_LT:
        *r += "<<";
        break;
    case COLON:
        *r += ':';
        break;
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case COMMENT:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
