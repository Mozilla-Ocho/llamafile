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
#include "string.h"
#include <cosmo.h>

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
                *r += '\\';
            } else {
                append_wchar(r, c);
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
            append_wchar(r, c);
            t_ = NORMAL;
            break;

        case STAR:
            if (c == '*') {
                // handle **strong** text
                t_ = STRONG;
                *r += HI_BOLD;
                *r += "**";
            } else if (bol_ && isblank(c)) {
                *r += '*';
                append_wchar(r, c);
                t_ = NORMAL;
            } else {
                // handle *emphasized* text
                // inverted because \e[3m has a poorly supported western bias
                *r += '*';
                *r += HI_ITALIC;
                append_wchar(r, c);
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
                *r += '*';
            } else if (c == '\\') {
                t_ = EMPHASIS_BACKSLASH;
                *r += '\\';
            } else {
                append_wchar(r, c);
            }
            break;

        case EMPHASIS_BACKSLASH:
            // so we can say *unbroken \* italic* and have it work
            append_wchar(r, c);
            t_ = EMPHASIS;
            break;

        case STRONG:
            append_wchar(r, c);
            if (c == '*') {
                t_ = STRONG_STAR;
            } else if (c == '\\') {
                t_ = STRONG_BACKSLASH;
            }
            break;

        case STRONG_BACKSLASH:
            // so we can say **unbroken \*\* bold** and have it work
            append_wchar(r, c);
            t_ = STRONG;
            break;

        case STRONG_STAR:
            append_wchar(r, c);
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
                append_wchar(r, c);
                t_ = INCODE;
            }
            break;

        case INCODE:
            // this is for `inline code` like that
            // no backslash escapes are supported here
            append_wchar(r, c);
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case INCODE2:
            // this is for ``inline ` code`` like that
            // it lets you put backtick inside the code
            append_wchar(r, c);
            if (c == '`') {
                t_ = INCODE2_TICK;
            }
            break;

        case INCODE2_TICK:
            append_wchar(r, c);
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
                append_wchar(r, c);
                t_ = INCODE2;
            }
            break;

        case LANG:
            if (!isascii(c) || !isspace(c)) {
                append_wchar(r, c);
                lang_ += tolower(c);
            } else {
                if (!(highlighter_ = Highlight::create(lang_)))
                    highlighter_ = new HighlightTxt;
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
                char cs[8];
                uint64_t w = tpenc(c);
                WRITE64LE(cs, w);
                highlighter_->feed(r, cs);
            }
            break;

        case CODE_TICK:
            if (c == '`') {
                t_ = CODE_TICK_TICK;
            } else {
                char cs[9];
                uint64_t w = tpenc(c);
                cs[0] = '`';
                WRITE64LE(cs + 1, w);
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
                char cs[10];
                uint64_t w = tpenc(c);
                cs[0] = '`';
                cs[1] = '`';
                WRITE64LE(cs + 2, w);
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
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
    bol_ = true;
    tail_ = false;
}
