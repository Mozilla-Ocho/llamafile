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
    SLASH,
    SLASH_SLASH,
    SLASH_STAR,
    SLASH_SLASH_BACKSLASH,
    SLASH_STAR_STAR,
    R,
    R_DQUOTE,
    RAW,
    QUESTION,
    TRIGRAPH,
    CPP_LT,
    BACKSLASH,
    UNIVERSAL,
};

HighlightC::HighlightC(is_keyword_f* is_keyword, //
                       is_keyword_f* is_type, //
                       is_keyword_f* is_builtin, //
                       is_keyword_f* is_constant)
  : is_keyword_(is_keyword)
  , is_type_(is_type)
  , is_builtin_(is_builtin)
  , is_constant_(is_constant)
{
}

HighlightC::~HighlightC()
{
}

void
HighlightC::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;
        if (c == '\r')
            continue;
        switch (t_) {

        Normal:
        case NORMAL:
            if (c == 'R') {
                t_ = R;
            } else if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                word_ += c;
            } else if (c == '#' && is_bol_ && !is_cpp_ && !is_define_) {
                is_cpp_ = true;
                *r += HI_BUILTIN;
                *r += '#';
            } else if (c == '<' && is_include_) {
                *r += HI_STRING;
                *r += '<';
                t_ = CPP_LT;
            } else if (c == '\\') {
                t_ = BACKSLASH;
            } else if (c == '/') {
                t_ = SLASH;
            } else if (c == '?') {
                t_ = QUESTION;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += '\'';
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += '"';
            } else if (c == '\n') {
                *r += '\n';
                if (is_cpp_) {
                    *r += HI_RESET;
                    is_include_ = false;
                    is_cpp_ = false;
                }
                is_define_ = false;
            } else {
                *r += c;
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_' || c == '$') {
                word_ += c;
            } else {
                if (is_cpp_) {
                    if (is_keyword_cpp(word_.data(), word_.size())) {
                        *r += HI_BUILTIN;
                        *r += word_;
                        *r += HI_RESET;
                        if (word_ == "include" || word_ == "include_next")
                            is_include_ = true;
                        if (word_ == "define") {
                            is_cpp_ = false;
                            is_define_ = true;
                        }
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
                    if (is_keyword_(word_.data(), word_.size())) {
                        *r += HI_KEYWORD;
                        *r += word_;
                        *r += HI_RESET;
                        if (is_keyword_c_pod(word_.data(), word_.size()))
                            is_pod_ = true;
                    } else if (is_pod_ ||
                               (is_type_ &&
                                is_type_(word_.data(), word_.size()))) {
                        *r += HI_TYPE;
                        *r += word_;
                        *r += HI_RESET;
                        is_pod_ = false;
                    } else if (is_builtin_ &&
                               is_builtin_(word_.data(), word_.size())) {
                        *r += HI_BUILTIN;
                        *r += word_;
                        *r += HI_RESET;
                        is_pod_ = false;
                    } else if (is_constant_ &&
                               is_constant_(word_.data(), word_.size())) {
                        *r += HI_CONSTANT;
                        *r += word_;
                        *r += HI_RESET;
                        is_pod_ = false;
                    } else {
                        *r += word_;
                        is_pod_ = false;
                    }
                }
                word_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case BACKSLASH:
            if (c == 'u') {
                *r += HI_ESCAPE;
                *r += "\\u";
                t_ = UNIVERSAL;
                i_ = 4;
            } else if (c == 'U') {
                *r += HI_ESCAPE;
                *r += "\\U";
                t_ = UNIVERSAL;
                i_ = 8;
            } else {
                *r += '\\';
                *r += c;
                t_ = NORMAL;
            }
            break;

        case UNIVERSAL:
            if (isascii(c) && isxdigit(c)) {
                *r += c;
                if (!--i_) {
                    *r += HI_RESET;
                    t_ = NORMAL;
                }
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case QUESTION:
            if (c == '?') {
                t_ = TRIGRAPH;
            } else {
                *r += '?';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case TRIGRAPH:
            if (c == '=' || // '#'
                c == '(' || // '['
                c == '/' || // '\\'
                c == ')' || // ']'
                c == '\'' || // '^'
                c == '<' || // '{'
                c == '!' || // '|'
                c == '>' || // '}'
                c == '-') { // '~'
                *r += HI_ESCAPE;
                *r += "??";
                *r += c;
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += "??";
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
            } else if (c == '\\') {
                t_ = SLASH_SLASH_BACKSLASH;
            }
            break;

        case SLASH_SLASH_BACKSLASH:
            *r += c;
            t_ = SLASH_SLASH;
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

        case CPP_LT:
            *r += c;
            if (c == '>') {
                *r += HI_RESET;
                t_ = NORMAL;
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

        case R:
            if (c == '"') {
                t_ = R_DQUOTE;
                *r += 'R';
                *r += HI_STRING;
                *r += '"';
                heredoc_ = ")";
            } else {
                word_ += 'R';
                t_ = WORD;
                goto Word;
            }
            break;

        case R_DQUOTE:
            *r += c;
            if (c == '(') {
                t_ = RAW;
                i_ = 0;
                heredoc_ += '"';
            } else {
                heredoc_ += c;
            }
            break;

        case RAW:
            *r += c;
            if (heredoc_[i_] == c) {
                if (++i_ == heredoc_.size()) {
                    t_ = NORMAL;
                    *r += HI_RESET;
                }
            } else {
                i_ = 0;
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
HighlightC::flush(std::string* r)
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
            if (is_keyword_(word_.data(), word_.size())) {
                *r += HI_KEYWORD;
                *r += word_;
                *r += HI_RESET;
            } else if (is_pod_ ||
                       (is_type_ && is_type_(word_.data(), word_.size()))) {
                *r += HI_TYPE;
                *r += word_;
                *r += HI_RESET;
            } else if (is_builtin_ && is_builtin_(word_.data(), word_.size())) {
                *r += HI_BUILTIN;
                *r += word_;
                *r += HI_RESET;
            } else if (is_constant_ &&
                       is_constant_(word_.data(), word_.size())) {
                *r += HI_CONSTANT;
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
    case QUESTION:
        *r += '?';
        if (is_cpp_)
            *r += HI_RESET;
        break;
    case TRIGRAPH:
        *r += "??";
        if (is_cpp_)
            *r += HI_RESET;
        break;
    case R:
        *r += 'R';
        if (is_cpp_)
            *r += HI_RESET;
        break;
    case BACKSLASH:
        *r += '\\';
        if (is_cpp_)
            *r += HI_RESET;
        break;
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case SLASH_SLASH:
    case SLASH_SLASH_BACKSLASH:
    case SLASH_STAR:
    case SLASH_STAR_STAR:
    case R_DQUOTE:
    case RAW:
        *r += HI_RESET;
        break;
    default:
        if (is_cpp_)
            *r += HI_RESET;
        break;
    }
    is_include_ = false;
    is_define_ = false;
    is_cpp_ = false;
    is_pod_ = false;
    is_bol_ = true;
    t_ = NORMAL;
}
