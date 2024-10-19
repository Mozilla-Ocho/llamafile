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

#define NORMAL 0
#define TICK 1
#define TICK_TICK 2
#define LANG 3
#define CODE 4
#define CODE_TICK 5
#define CODE_TICK_TICK 6
#define STAR 7
#define STRONG 8
#define STRONG_STAR 9
#define BACKSLASH 10
#define INCODE 11

HighlightMarkdown::HighlightMarkdown() {
}

HighlightMarkdown::~HighlightMarkdown() {
}

void HighlightMarkdown::feed(std::string *r, std::string_view input) {
    char c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i];
        switch (t_) {

        Normal:
        case NORMAL:
            if (c == '`') {
                t_ = TICK;
            } else if (c == '*') {
                t_ = STAR;
            } else if (c == '\\') {
                t_ = BACKSLASH;
                *r += c;
            } else {
                *r += c;
            }
            break;

        case BACKSLASH:
            *r += c;
            t_ = NORMAL;
            break;

        case STAR:
            if (c == '*') {
                t_ = STRONG;
                *r += HI_BOLD;
            } else {
                t_ = NORMAL;
            }
            *r += '*';
            *r += c;
            break;

        case STRONG:
            *r += c;
            if (c == '*')
                t_ = STRONG_STAR;
            break;

        case STRONG_STAR:
            *r += c;
            if (c == '*') {
                t_ = NORMAL;
                *r += HI_RESET;
            } else {
                t_ = STRONG;
            }
            break;

        case TICK:
            if (c == '`') {
                t_ = TICK_TICK;
                *r += '`';
                *r += c;
            } else {
                *r += HI_INCODE;
                *r += '`';
                *r += c;
                t_ = INCODE;
            }
            break;

        case INCODE:
            *r += c;
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case TICK_TICK:
            if (c == '`') {
                t_ = LANG;
                *r += c;
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LANG:
            if (!isascii(c) || !isspace(c)) {
                *r += c;
                lang_ += tolower(c);
            } else {
                highlighter_ = Highlight::create(lang_);
                lang_.clear();
                t_ = CODE;
                goto Code;
            }
            break;

        Code:
        case CODE:
            if (c == '`') {
                t_ = CODE_TICK;
            } else {
                char cs[2] = {c};
                highlighter_->feed(r, cs);
            }
            break;

        case CODE_TICK:
            if (c == '`') {
                t_ = CODE_TICK_TICK;
            } else {
                char cs[3] = {'`', c};
                highlighter_->feed(r, cs);
                t_ = CODE;
            }
            break;

        case CODE_TICK_TICK:
            if (c == '`') {
                t_ = NORMAL;
                highlighter_->flush(r);
                delete highlighter_;
                highlighter_ = nullptr;
                *r += "```";
            } else {
                char cs[4] = {'`', '`', c};
                highlighter_->feed(r, cs);
                t_ = CODE;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void HighlightMarkdown::flush(std::string *r) {
    switch (t_) {
    case LANG:
        lang_.clear();
        break;
    case STAR:
        *r += '*';
        break;
    case TICK:
        *r += '`';
        break;
    case INCODE:
    case STRONG:
    case STRONG_STAR:
        *r += HI_RESET;
        break;
    case CODE:
        highlighter_->flush(r);
        delete highlighter_;
        highlighter_ = nullptr;
        break;
    case CODE_TICK:
        highlighter_->flush(r);
        delete highlighter_;
        highlighter_ = nullptr;
        *r += '`';
        break;
    case CODE_TICK_TICK:
        highlighter_->flush(r);
        delete highlighter_;
        highlighter_ = nullptr;
        *r += "``";
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
