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
    LPAREN,
    COMMENT,
    COMMENT_STAR,
    COMMENT_LPAREN,
    LCURLY,
    RAWSTR,
    RAWSTR_PIPE,
};

HighlightOcaml::HighlightOcaml()
{
}

HighlightOcaml::~HighlightOcaml()
{
}

void
HighlightOcaml::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;

        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c) || c == '_' || c == '~') {
                t_ = WORD;
                goto Word;
            } else if (c == '(') {
                t_ = LPAREN;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '{') {
                t_ = LCURLY;
            } else {
                *r += c;
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_' || c == '\'' ||
                c == '~') {
                word_ += c;
            } else {
                if (is_keyword_ocaml(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_ocaml_builtin(word_.data(),
                                                    word_.size())) {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_ocaml_constant(word_.data(),
                                                     word_.size())) {
                    *r += HI_CONSTANT;
                    *r += word_;
                    *r += HI_RESET;
                } else if (word_.size() > 1 && word_[0] == '~') {
                    *r += HI_PROPERTY;
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

        case LPAREN:
            if (c == '*') {
                *r += HI_COMMENT;
                *r += "(*";
                t_ = COMMENT;
                nest_ = 1;
            } else {
                *r += '(';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COMMENT:
            *r += c;
            if (c == '*') {
                t_ = COMMENT_STAR;
            } else if (c == '(') {
                t_ = COMMENT_LPAREN;
            }
            break;

        case COMMENT_STAR:
            *r += c;
            if (c == ')') {
                if (!--nest_) {
                    *r += HI_RESET;
                    t_ = NORMAL;
                }
            } else if (c == '(') {
                t_ = COMMENT_LPAREN;
            } else if (c != '*') {
                t_ = COMMENT;
            }
            break;

        case COMMENT_LPAREN:
            *r += c;
            if (c == '*') {
                ++nest_;
                t_ = COMMENT;
            } else if (c != '(') {
                t_ = COMMENT;
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

        case LCURLY:
            if (c == '|') {
                *r += HI_STRING;
                *r += '{';
                *r += word_;
                *r += '|';
                t_ = RAWSTR;
            } else if (isalpha(c) || c == '_') {
                word_ += c;
            } else {
                *r += '{';
                if (word_.empty()) {
                    t_ = NORMAL;
                    goto Normal;
                } else {
                    t_ = WORD;
                    goto Word;
                }
            }
            break;

        case RAWSTR:
            *r += c;
            if (c == '|') {
                t_ = RAWSTR_PIPE;
            }
            break;

        case RAWSTR_PIPE:
            *r += c;
            if (c == '}' && word2_ == word_) {
                *r += HI_RESET;
                word2_.clear();
                word_.clear();
                t_ = NORMAL;
            } else if (c == '|') {
                word2_.clear();
            } else if (isalpha(c) || c == '_') {
                word2_ += c;
            } else {
                word2_.clear();
                t_ = RAWSTR;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightOcaml::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_ocaml(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_ocaml_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_ocaml_constant(word_.data(), word_.size())) {
            *r += HI_CONSTANT;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case LPAREN:
        *r += '(';
        break;
    case LCURLY:
        *r += '{';
        *r += word_;
        word_.clear();
        break;
    case RAWSTR:
    case RAWSTR_PIPE:
        word_.clear();
        word2_.clear();
        *r += HI_RESET;
        break;
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case COMMENT:
    case COMMENT_STAR:
    case COMMENT_LPAREN:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
