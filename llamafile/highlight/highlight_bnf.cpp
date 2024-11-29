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

enum
{
    NORMAL,
    COMMENT,
    DQUOTE,
    DQUOTE_BACKSLASH,
    ESCAPE,
    ESCAPE_HEX,
    ESCAPE_HEX1,
    OPERATOR,
};

static bool
is_operator(const std::string& op)
{
    return op == "::=" || //
           op == "|" || //
           op == "?" || //
           op == "*" || //
           op == "+";
}

static bool
is_operator_char(int c)
{
    switch (c) {
    case '!':
    case '$':
    case '%':
    case '&':
    case '\'':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '/':
    case ':':
    case '=':
    case '?':
    case '@':
    case '^':
    case '_':
    case '`':
    case '|':
    case '~':
        return true;
    default:
        return false;
    }
}

HighlightBnf::HighlightBnf()
{
}

HighlightBnf::~HighlightBnf()
{
}

void
HighlightBnf::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;

        switch (t_) {

        Normal:
        case NORMAL:
            if (c == '#' || c == ';') {
                t_ = COMMENT;
                *r += HI_COMMENT;
                *r += c;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += '"';
            } else if (c == '\\') {
                t_ = ESCAPE;
                *r += HI_ESCAPE;
                *r += '\\';
            } else if (is_operator_char(c)) {
                operator_ += c;
                t_ = OPERATOR;
            } else {
                *r += c;
            }
            break;

        case OPERATOR:
            if (is_operator_char(c)) {
                operator_ += c;
            } else {
                if (is_operator(operator_)) {
                    *r += HI_OPERATOR;
                    *r += operator_;
                    *r += HI_RESET;
                } else {
                    *r += operator_;
                }
                operator_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case ESCAPE:
            *r += c;
            if (c == 'x') {
                t_ = ESCAPE_HEX;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case ESCAPE_HEX:
            if (isxdigit(c)) {
                *r += c;
                t_ = ESCAPE_HEX1;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case ESCAPE_HEX1:
            if (isxdigit(c)) {
                *r += c;
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += HI_RESET;
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
    }
}

void
HighlightBnf::flush(std::string* r)
{
    switch (t_) {
    case OPERATOR:
        if (is_operator(operator_)) {
            *r += HI_OPERATOR;
            *r += operator_;
            *r += HI_RESET;
        } else {
            *r += operator_;
        }
        operator_.clear();
        break;
    case DQUOTE:
    case COMMENT:
    case DQUOTE_BACKSLASH:
    case ESCAPE:
    case ESCAPE_HEX:
    case ESCAPE_HEX1:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
