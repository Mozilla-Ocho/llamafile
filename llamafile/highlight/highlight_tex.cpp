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
#include "util.h"
#include <cosmo.h>
#include <ctype.h>

enum
{
    NORMAL,
    BACKSLASH,
    COMMAND,
    COMMENT,
    DOLLAR,
    MATH,
    MATH_BACKSLASH,
    BACKTICK,
    STRING,
    STRING_QUOTE,
};

HighlightTex::HighlightTex()
{
}

HighlightTex::~HighlightTex()
{
}

void
HighlightTex::feed(std::string* r, std::string_view input)
{
    for (size_t i = 0; i < input.size(); ++i) {
        wchar_t c;
        int b = input[i] & 255;
        if (!u_) {
            if (b < 0300) {
                c = b;
            } else {
                c_ = ThomPikeByte(b);
                u_ = ThomPikeLen(b) - 1;
                continue;
            }
        } else if (ThomPikeCont(b)) {
            c = c_ = ThomPikeMerge(c_, b);
            if (--u_)
                continue;
        } else {
            u_ = 0;
            c = b;
        }
        if (c == 0xFEFF)
            continue; // utf-8 bom
        switch (t_) {

        Normal:
        case NORMAL:
            if (c == '\\') {
                t_ = BACKSLASH;
            } else if (c == '$') {
                t_ = DOLLAR;
            } else if (c == '`') {
                t_ = BACKTICK;
            } else if (c == '%') {
                t_ = COMMENT;
                *r += HI_COMMENT;
                *r += '%';
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case BACKSLASH:
            if (c == '\\') {
                *r += HI_WARNING;
                *r += "\\\\";
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (isspace(c)) {
                *r += '\\';
                t_ = NORMAL;
                goto Normal;
            } else if (isalpha(c) || c == '@') {
                *r += HI_KEYWORD;
                *r += '\\';
                lf::append_wchar(r, c);
                t_ = COMMAND;
            } else {
                *r += HI_ESCAPE;
                *r += '\\';
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case COMMAND:
            if (isalpha(c) || c == '@') {
                lf::append_wchar(r, c);
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case DOLLAR:
            if (c == '$') {
                *r += "$$";
                t_ = NORMAL;
            } else if (c == '\\') {
                *r += HI_MATH;
                *r += "$\\";
                t_ = MATH_BACKSLASH;
            } else {
                *r += HI_MATH;
                *r += "$";
                lf::append_wchar(r, c);
                t_ = MATH;
            }
            break;

        case MATH:
            if (c == '$') {
                *r += "$";
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                *r += '\\';
                t_ = MATH_BACKSLASH;
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case MATH_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = MATH;
            break;

        case COMMENT:
            lf::append_wchar(r, c);
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case BACKTICK:
            if (c == '`') {
                *r += HI_STRING;
                *r += "``";
                t_ = STRING;
            } else {
                *r += '`';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case STRING:
            lf::append_wchar(r, c);
            if (c == '\'')
                t_ = STRING_QUOTE;
            break;

        case STRING_QUOTE:
            lf::append_wchar(r, c);
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                t_ = STRING;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightTex::flush(std::string* r)
{
    switch (t_) {
    case BACKTICK:
        *r += '`';
        break;
    case DOLLAR:
        *r += '$';
        break;
    case BACKSLASH:
        *r += '\\';
        break;
    case COMMAND:
    case COMMENT:
    case MATH:
    case MATH_BACKSLASH:
    case STRING:
    case STRING_QUOTE:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
}
