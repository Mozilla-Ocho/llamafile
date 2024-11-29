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

// The main challenge with C# is that C# v11 introduced a triple quote
// multi-line string literal syntax that's sort of like python and lua
//
//     Console.WriteLine("");
//     Console.WriteLine("\"");
//     Console.WriteLine("""""");
//     Console.WriteLine("""""");
//     Console.WriteLine(""" yo "" hi """);
//     Console.WriteLine("""" yo """ hi """");
//     Console.WriteLine(""""First
//                       """100 Prime"""
//                       Numbers:
//                       """");
//
// As we can see above, you can use four, five, or more dquotes so you
// can embed triple quoted strings inside.

enum
{
    NORMAL,
    WORD,
    QUOTE,
    QUOTE_BACKSLASH,
    SLASH,
    SLASH_SLASH,
    SLASH_STAR,
    SLASH_STAR_STAR,
    DQUOTE,
    STR,
    STR_BACKSLASH,
    DQUOTE_DQUOTE,
    DQUOTE_DQUOTE_DQUOTE,
    TRIPS,
    TRIPS_DQUOTE,
};

HighlightCsharp::HighlightCsharp()
{
}

HighlightCsharp::~HighlightCsharp()
{
}

void
HighlightCsharp::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;
        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c) || c == '_' || c == '#') {
                t_ = WORD;
                word_ += c;
            } else if (c == '/') {
                t_ = SLASH;
            } else if (c == '\'') {
                t_ = QUOTE;
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

        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_' || c == '$' || c == '#') {
                word_ += c;
            } else {
                if (is_keyword_csharp(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_csharp_constant(word_.data(),
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

        case DQUOTE:
            *r += c;
            if (c == '"') {
                t_ = DQUOTE_DQUOTE;
            } else if (c == '\\') {
                t_ = STR_BACKSLASH;
            } else {
                t_ = STR;
            }
            break;

        case STR:
            *r += c;
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = STR_BACKSLASH;
            }
            break;

        case STR_BACKSLASH:
            *r += c;
            t_ = STR;
            break;

        case DQUOTE_DQUOTE:
            if (c == '"') {
                *r += c;
                t_ = DQUOTE_DQUOTE_DQUOTE;
                trips1_ = 3;
                trips2_ = 0;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case DQUOTE_DQUOTE_DQUOTE:
            if (c == '"') {
                *r += c;
                ++trips1_;
                if (++trips2_ == 3) {
                    *r += HI_RESET;
                    t_ = NORMAL;
                }
                break;
            } else {
                trips2_ = 0;
                t_ = TRIPS;
            }
            // fallthrough

        case TRIPS:
            *r += c;
            if (c == '"') {
                t_ = TRIPS_DQUOTE;
                trips2_ = 1;
            }
            break;

        case TRIPS_DQUOTE:
            *r += c;
            if (c == '"') {
                if (++trips2_ == trips1_) {
                    *r += HI_RESET;
                    t_ = NORMAL;
                }
            } else {
                t_ = TRIPS;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightCsharp::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_csharp(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_csharp_constant(word_.data(), word_.size())) {
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
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case STR:
    case STR_BACKSLASH:
    case DQUOTE_DQUOTE:
    case DQUOTE_DQUOTE_DQUOTE:
    case TRIPS:
    case TRIPS_DQUOTE:
    case SLASH_SLASH:
    case SLASH_STAR:
    case SLASH_STAR_STAR:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
