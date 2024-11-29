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

// the `less` command was intentionally designed to prevent ansi sgr
// codes from bleeding across lines. for example, saying:
//
//     printf '\e[1m line 1 \n line 2 \e[0m \n'
//
// both lines will appear bold in the terminal. however with:
//
//     printf '\e[1m line 1 \n line 2 \e[0m \n' | less
//
// only the first line will appear bold. this is because less has an
// implicit \e[0m at the end of each line. this is bad, because our
// highlighting code assumes that, when emitting things like comments on
// multiple lines, we only need to set the color at the beginning.
//
// this class may be used to compose a Highlighter object. it'll relay
// the emitted bytes through, taking special care to restore the ansi
// sgr state at the beginning of each line. thus less will now work.
//
// only the subset of ansi escape codes that we use are supported.

enum
{
    NORMAL,
    ESC,
    CSI,
};

ColorBleeder::ColorBleeder(Highlight* h) : h_(h)
{
}

ColorBleeder::~ColorBleeder()
{
}

void
ColorBleeder::restore(std::string* r)
{
    bool got_some = false;
    if (!intensity_ && !inverted_ && !foreground_ && !background_)
        return;
    *r += "\033[";
    if (intensity_) {
        *r += std::to_string(intensity_);
        got_some = true;
    }
    if (inverted_) {
        if (got_some)
            *r += ';';
        *r += '7';
        got_some = true;
    }
    if (foreground_) {
        if (got_some)
            *r += ';';
        *r += std::to_string(foreground_);
        got_some = true;
    }
    if (background_) {
        if (got_some)
            *r += ';';
        *r += std::to_string(background_);
    }
    *r += 'm';
}

void
ColorBleeder::relay(std::string* r, const std::string& s)
{
    for (char c : s) {
        *r += c;
        switch (t_) {

        Normal:
        case NORMAL:
            switch (c) {
            case 033:
                t_ = ESC;
                break;
            case '\n':
                restore(r);
                break;
            }
            break;

        case ESC:
            switch (c) {
            case '[':
                t_ = CSI;
                n_ = 0;
                x_ = 0;
                break;
            default:
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case CSI:
            if (isdigit(c)) {
                x_ *= 10;
                x_ += c - '0';
            } else if (c == ';') {
                if (n_ < sizeof(sgr_codes_)) {
                    sgr_codes_[n_++] = x_;
                    x_ = 0;
                }
            } else if (c == 'm') {
                bool vt100dirty = false;
                if (n_ < sizeof(sgr_codes_)) {
                    sgr_codes_[n_++] = x_;
                    x_ = 0;
                }
                for (int i = 0; i < n_; ++i) {
                    int g = sgr_codes_[i];
                    if (g == 0) {
                        inverted_ = 0;
                        intensity_ = 0;
                        foreground_ = 0;
                        background_ = 0;
                    } else if (g == 1 || g == 2) {
                        intensity_ = g;
                    } else if (g == 22) {
                        intensity_ = 0;
                        vt100dirty = true;
                    } else if (g == 7) {
                        inverted_ = 1;
                    } else if (g == 27) {
                        inverted_ = 0;
                        vt100dirty = true;
                    } else if ((30 <= g && g <= 37) || //
                               (90 <= g && g <= 97)) {
                        foreground_ = g;
                    } else if (g == 39) {
                        foreground_ = 0;
                        vt100dirty = true;
                    } else if ((40 <= g && g <= 47) || //
                               (100 <= g && g <= 107)) {
                        background_ = g;
                    } else if (g == 49) {
                        background_ = 0;
                        vt100dirty = true;
                    }
                }
                if (vt100dirty) {
                    *r += HI_RESET;
                    restore(r);
                }
                t_ = NORMAL;
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
ColorBleeder::feed(std::string* r, std::string_view input)
{
    std::string s;
    h_->feed(&s, input);
    relay(r, s);
}

void
ColorBleeder::flush(std::string* r)
{
    std::string s;
    h_->flush(&s);
    relay(r, s);
}
