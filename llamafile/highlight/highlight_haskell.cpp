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
    TICK,
    TICK_BACKSLASH,
    CURL,
    CURL_HYPHEN,
    CURL_HYPHEN_HYPHEN,
    CURL_HYPHEN_CURL,
    HYPHEN,
    HYPHEN_HYPHEN,
    HYPHEN_LT,
    EQUAL,
    COLON,
    LT,
};

HighlightHaskell::HighlightHaskell()
{
}

HighlightHaskell::~HighlightHaskell()
{
}

void
HighlightHaskell::feed(std::string* r, std::string_view input)
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
            } else if (c == '-') {
                t_ = HYPHEN;
            } else if (c == '{') {
                t_ = CURL;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '`') {
                t_ = TICK;
                *r += HI_OPERATOR;
                *r += c;
            } else if (c == '!' || //
                       c == '#' || //
                       c == '$' || //
                       c == '*' || //
                       c == ',' || //
                       c == '>' || //
                       c == '?' || //
                       c == '@' || //
                       c == '|' || //
                       c == '~') {
                *r += HI_OPERATOR;
                *r += c;
                *r += HI_RESET;
            } else if (c == '=') {
                t_ = EQUAL;
            } else if (c == ':') {
                t_ = COLON;
            } else if (c == '<') {
                t_ = LT;
            } else {
                *r += c;
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                symbol_ += c;
            } else {
                if (is_keyword_haskell(symbol_.data(), symbol_.size())) {
                    *r += HI_KEYWORD;
                    *r += symbol_;
                    *r += HI_RESET;
                } else {
                    *r += symbol_;
                }
                symbol_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT:
            if (c == '-') {
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

        case EQUAL:
            if (c == '>') {
                *r += HI_OPERATOR;
                *r += "=>";
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += HI_OPERATOR;
                *r += '=';
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case HYPHEN:
            if (c == '-') {
                *r += HI_COMMENT;
                *r += "--";
                t_ = HYPHEN_HYPHEN;
            } else if (c == '<') {
                t_ = HYPHEN_LT;
            } else if (c == '>') {
                *r += HI_OPERATOR;
                *r += "->";
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += '-';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case HYPHEN_LT:
            if (c == '<') {
                *r += HI_OPERATOR;
                *r += "-<<";
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += HI_OPERATOR;
                *r += "-<";
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case HYPHEN_HYPHEN:
            *r += c;
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case CURL:
            if (c == '-') {
                *r += HI_COMMENT;
                *r += "{-";
                t_ = CURL_HYPHEN;
                level_ = 1;
            } else {
                *r += '{';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case CURL_HYPHEN:
            *r += c;
            if (c == '-') {
                t_ = CURL_HYPHEN_HYPHEN;
            } else if (c == '{') {
                t_ = CURL_HYPHEN_CURL;
            }
            break;

        case CURL_HYPHEN_CURL:
            *r += c;
            if (c == '-') {
                t_ = CURL_HYPHEN;
                ++level_;
            } else if (c != '{') {
                t_ = CURL_HYPHEN;
            }
            break;

        case CURL_HYPHEN_HYPHEN:
            *r += c;
            if (c == '}') {
                if (!--level_) {
                    *r += HI_RESET;
                    t_ = NORMAL;
                }
            } else if (c != '-') {
                t_ = CURL_HYPHEN;
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

        case TICK:
            *r += c;
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = TICK_BACKSLASH;
            }
            break;

        case TICK_BACKSLASH:
            *r += c;
            t_ = TICK;
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightHaskell::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_haskell(symbol_.data(), symbol_.size())) {
            *r += HI_KEYWORD;
            *r += symbol_;
            *r += HI_RESET;
        } else {
            *r += symbol_;
        }
        symbol_.clear();
        break;
    case CURL:
        *r += '{';
        break;
    case HYPHEN:
        *r += '-';
        break;
    case EQUAL:
        *r += '=';
        break;
    case COLON:
        *r += ':';
        break;
    case LT:
        *r += '<';
        break;
    case HYPHEN_LT:
        *r += HI_OPERATOR;
        *r += "-<";
        *r += HI_RESET;
        break;
    case TICK:
    case TICK_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case HYPHEN_HYPHEN:
    case CURL_HYPHEN:
    case CURL_HYPHEN_HYPHEN:
    case CURL_HYPHEN_CURL:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
