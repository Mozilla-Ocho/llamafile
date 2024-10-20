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

enum {
    NORMAL,
    SELECTOR,
    PROPERTY,
    VALUE,
    QUOTE,
    QUOTE_BACKSLASH,
    DQUOTE,
    DQUOTE_BACKSLASH,
    SLASH,
    SLASH_STAR,
    SLASH_STAR_STAR,
};

HighlightCss::HighlightCss() {
}

HighlightCss::~HighlightCss() {
}

void HighlightCss::feed(std::string *r, std::string_view input) {
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;

    TryAgain:
        switch (t_ & 255) {

        case NORMAL:
            *r += HI_SELECTOR;
            t_ = SELECTOR;
            // fallthrough

        case SELECTOR:
            if (c == '{') {
                t_ = PROPERTY;
                *r += HI_RESET;
                *r += c;
                *r += HI_PROPERTY;
            } else if (c == ',') {
                *r += HI_RESET;
                *r += c;
                *r += HI_SELECTOR;
            } else if (c == '/') {
                t_ = SELECTOR << 8 | SLASH;
            } else if (c == '\'') {
                t_ = SELECTOR << 8 | QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = SELECTOR << 8 | DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else {
                *r += c;
            }
            break;

        case PROPERTY:
            if (c == '/') {
                t_ = PROPERTY << 8 | SLASH;
            } else if (c == '\'') {
                t_ = PROPERTY << 8 | QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = PROPERTY << 8 | DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == ':') {
                t_ = VALUE;
                *r += HI_RESET;
                *r += c;
            } else if (c == '}') {
                t_ = SELECTOR;
                *r += HI_RESET;
                *r += c;
                *r += HI_SELECTOR;
            } else {
                *r += c;
            }
            break;

        case VALUE:
            if (c == '/') {
                t_ = VALUE << 8 | SLASH;
            } else if (c == '\'') {
                t_ = VALUE << 8 | QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = VALUE << 8 | DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == ';') {
                t_ = PROPERTY;
                *r += c;
                *r += HI_PROPERTY;
            } else if (c == '}') {
                t_ = SELECTOR;
                *r += c;
                *r += HI_SELECTOR;
            } else {
                *r += c;
            }
            break;

        case SLASH:
            if (c == '*') {
                *r += HI_COMMENT;
                *r += "/*";
                t_ &= -256;
                t_ |= SLASH_STAR;
            } else {
                *r += '/';
                t_ >>= 8;
                goto TryAgain;
            }
            break;

        case SLASH_STAR:
            *r += c;
            if (c == '*') {
                t_ &= -256;
                t_ |= SLASH_STAR_STAR;
            }
            break;

        case SLASH_STAR_STAR:
            *r += c;
            if (c == '/') {
                *r += HI_RESET;
                goto Pop;
            } else if (c != '*') {
                t_ &= -256;
                t_ |= SLASH_STAR;
            }
            break;

        case QUOTE:
            *r += c;
            if (c == '\'') {
                *r += HI_RESET;
                goto Pop;
            } else if (c == '\\') {
                t_ &= -256;
                t_ |= QUOTE_BACKSLASH;
            }
            break;

        case QUOTE_BACKSLASH:
            *r += c;
            t_ &= -256;
            t_ |= QUOTE;
            break;

        case DQUOTE:
            *r += c;
            if (c == '"') {
                *r += HI_RESET;
                goto Pop;
            } else if (c == '\\') {
                t_ &= -256;
                t_ |= DQUOTE_BACKSLASH;
            }
            break;

        case DQUOTE_BACKSLASH:
            *r += c;
            t_ &= -256;
            t_ |= DQUOTE;
            break;

        Pop:
            t_ >>= 8;
            if (t_ == SELECTOR)
                *r += HI_SELECTOR;
            if (t_ == PROPERTY)
                *r += HI_PROPERTY;
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void HighlightCss::flush(std::string *r) {
    switch (t_ & 255) {
    case SLASH:
        *r += '/';
        break;
    case SELECTOR:
    case PROPERTY:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    *r += HI_RESET;
    t_ = 0;
}
