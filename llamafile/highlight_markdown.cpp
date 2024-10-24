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

enum {
    NORMAL,
    TICK,
    TICK_TICK,
    LANG,
    CODE,
    CODE_TICK,
    CODE_TICK_TICK,
    STAR,
    STRONG,
    STRONG_BACKSLASH,
    STRONG_STAR,
    BACKSLASH,
    INCODE,
    INCODE2,
    INCODE2_TICK,
    EMPHASIS,
    EMPHASIS_BACKSLASH,
};

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
                break;
            } else if (c == '*') {
                t_ = STAR;
                break;
            } else if (c == '\\') {
                // handle \*\*not bold\*\* etc.
                t_ = BACKSLASH;
                *r += c;
            } else {
                *r += c;
            }
            if (c == '\n') {
                bol_ = true;
                tail_ = false;
            } else {
                tail_ = true;
                if (!isblank(c))
                    bol_ = false;
            }
            break;

        case BACKSLASH:
            *r += c;
            t_ = NORMAL;
            break;

        case STAR:
            if (c == '*') {
                // handle **strong** text
                t_ = STRONG;
                *r += HI_BOLD;
                *r += '*';
                *r += c;
            } else if (bol_ && isblank(c)) {
                *r += '*';
                *r += c;
                t_ = NORMAL;
            } else {
                // handle *emphasized* text
                // inverted because \e[3m has a poorly supported western bias
                *r += '*';
                *r += HI_ITALIC;
                *r += c;
                t_ = EMPHASIS;
                if (c == '\\')
                    t_ = EMPHASIS_BACKSLASH;
            }
            break;

        case EMPHASIS:
            // this is for *emphasized* text
            if (c == '*') {
                t_ = NORMAL;
                *r += HI_RESET;
                *r += c;
            } else if (c == '\\') {
                t_ = EMPHASIS_BACKSLASH;
                *r += c;
            } else {
                *r += c;
            }
            break;

        case EMPHASIS_BACKSLASH:
            // so we can say *unbroken \* italic* and have it work
            *r += c;
            t_ = EMPHASIS;
            break;

        case STRONG:
            *r += c;
            if (c == '*') {
                t_ = STRONG_STAR;
            } else if (c == '\\') {
                t_ = STRONG_BACKSLASH;
            }
            break;

        case STRONG_BACKSLASH:
            // so we can say **unbroken \*\* bold** and have it work
            *r += c;
            t_ = STRONG;
            break;

        case STRONG_STAR:
            *r += c;
            if (c == '*' || // handle **bold** ending
                (c == '\n' && !tail_)) { // handle *** line break
                t_ = NORMAL;
                *r += HI_RESET;
            } else if (c == '\\') {
                t_ = STRONG_BACKSLASH;
            } else {
                t_ = STRONG;
            }
            break;

        case TICK:
            if (c == '`') {
                t_ = TICK_TICK;
            } else {
                *r += HI_INCODE;
                *r += '`';
                *r += c;
                t_ = INCODE;
            }
            break;

        case INCODE:
            // this is for `inline code` like that
            // no backslash escapes are supported here
            *r += c;
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case INCODE2:
            // this is for ``inline ` code`` like that
            // it lets you put backtick inside the code
            *r += c;
            if (c == '`') {
                t_ = INCODE2_TICK;
            }
            break;

        case INCODE2_TICK:
            *r += c;
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                t_ = INCODE2;
            }
            break;

        case TICK_TICK:
            if (c == '`') {
                t_ = LANG;
                *r += "```";
            } else {
                *r += HI_INCODE;
                *r += "``";
                *r += c;
                t_ = INCODE2;
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
    case TICK_TICK:
        *r += "``";
        break;
    case INCODE:
    case INCODE2:
    case INCODE2_TICK:
    case STRONG:
    case STRONG_BACKSLASH:
    case STRONG_STAR:
    case EMPHASIS:
    case EMPHASIS_BACKSLASH:
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
