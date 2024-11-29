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
    WORD,
    WORD_SPACE,
    DOLLAR,
    VAR,
    COMMENT,
    DQUOTE,
    DQUOTE_BACKSLASH,
    DQUOTE_DOLLAR,
    DQUOTE_VAR,
};

HighlightCmake::HighlightCmake()
{
}

HighlightCmake::~HighlightCmake()
{
}

void
HighlightCmake::feed(std::string* r, std::string_view input)
{
    int c;
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
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                lf::append_wchar(&word_, c);
            } else if (c == '#') {
                t_ = COMMENT;
                *r += HI_COMMENT;
                *r += '#';
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += '"';
            } else if (c == '$') {
                t_ = DOLLAR;
            } else if (c == '\\') {
                t_ = BACKSLASH;
                *r += HI_ESCAPE;
                *r += '\\';
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case BACKSLASH:
            lf::append_wchar(r, c);
            *r += HI_RESET;
            t_ = NORMAL;
            break;

        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                lf::append_wchar(&word_, c);
            } else {
                spaces_ = 0;
                t_ = WORD_SPACE;
                goto WordSpace;
            }
            break;

        WordSpace:
        case WORD_SPACE:
            if (c == ' ') {
                ++spaces_;
            } else if (c == '(') {
                if (is_keyword_cmake(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                }
                word_.clear();
                for (int i = 0; i < spaces_; ++i)
                    *r += ' ';
                t_ = NORMAL;
                goto Normal;
            } else {
                *r += word_;
                word_.clear();
                for (int i = 0; i < spaces_; ++i)
                    *r += ' ';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COMMENT:
            lf::append_wchar(r, c);
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case DOLLAR:
            if (c == '{') {
                t_ = VAR;
            } else if (c == '$') {
                *r += '$';
            } else {
                *r += '$';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case VAR:
            if (isalnum(c) || c == '_') {
                lf::append_wchar(&word_, c);
            } else if (c == '}') {
                *r += "${";
                *r += HI_VAR;
                *r += word_;
                *r += HI_RESET;
                *r += '}';
                word_.clear();
                t_ = NORMAL;
            } else {
                *r += "${";
                *r += word_;
                word_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        Dquote:
        case DQUOTE:
            if (c == '"') {
                *r += '"';
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                *r += '\\';
                t_ = DQUOTE_BACKSLASH;
            } else if (c == '$') {
                t_ = DQUOTE_DOLLAR;
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case DQUOTE_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = DQUOTE;
            break;

        case DQUOTE_DOLLAR:
            if (c == '{') {
                t_ = DQUOTE_VAR;
            } else if (c == '$') {
                *r += '$';
            } else {
                *r += '$';
                t_ = DQUOTE;
                goto Dquote;
            }
            break;

        case DQUOTE_VAR:
            if (isalnum(c) || c == '_') {
                lf::append_wchar(&word_, c);
            } else if (c == '}') {
                *r += "${";
                *r += HI_VAR;
                *r += word_;
                *r += HI_RESET;
                *r += HI_STRING;
                *r += '}';
                word_.clear();
                t_ = DQUOTE;
            } else {
                *r += "${";
                *r += word_;
                word_.clear();
                t_ = DQUOTE;
                goto Dquote;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightCmake::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        *r += word_;
        word_.clear();
        break;
    case WORD_SPACE:
        *r += word_;
        word_.clear();
        for (int i = 0; i < spaces_; ++i)
            *r += ' ';
        break;
    case DOLLAR:
        *r += '$';
        break;
    case DQUOTE_DOLLAR:
        *r += '$';
        *r += HI_RESET;
        break;
    case DQUOTE_VAR:
        *r += "${";
        *r += word_;
        *r += HI_RESET;
        word_.clear();
        break;
    case VAR:
        *r += "${";
        *r += word_;
        word_.clear();
        break;
    case DQUOTE:
    case COMMENT:
    case BACKSLASH:
    case DQUOTE_BACKSLASH:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
}
