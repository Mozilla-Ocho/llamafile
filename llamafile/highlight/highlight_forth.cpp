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
    SYNTAX,
};

HighlightForth::HighlightForth()
{
}

HighlightForth::~HighlightForth()
{
}

void
HighlightForth::feed(std::string* r, std::string_view input)
{
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;
        switch (t_) {

        case NORMAL:
            if (!isspace(c)) {
                word_ += c;
            } else if (!word_.empty()) {
                if (is_label_) {
                    *r += word_;
                    *r += HI_RESET;
                    is_label_ = false;
                } else if (word_ == "\\") { // line comment
                    *r += HI_COMMENT;
                    *r += word_;
                    t_ = SYNTAX;
                    closer_ = '\n';
                } else if (word_ == "(" || // inline comment, e.g. ( arg1
                                           // arg2 -- res1 )
                           word_ == ".(") { // printed comment, e.g. .(
                                            // compiling... )
                    *r += HI_COMMENT;
                    *r += word_;
                    t_ = SYNTAX;
                    closer_ = ')';
                } else if (word_ == ".\"" || // string
                           word_ == "s\"" || // stack string
                           word_ == "S\"" || // stack string
                           word_ == "c\"" || // counted string
                           word_ == "C\"") { // counted string
                    *r += HI_STRING;
                    *r += word_;
                    t_ = SYNTAX;
                    closer_ = '"';
                } else if (is_keyword_forth_def(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_DEF;
                    is_label_ = true;
                } else if (is_keyword_forth(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else {
                    *r += word_;
                }
                word_.clear();
                *r += c;
                break;
            } else {
                *r += c;
            }
            break;

        case SYNTAX:
            *r += c;
            if (c == closer_) {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightForth::flush(std::string* r)
{
    switch (t_) {
    case NORMAL:
        if (is_label_) {
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_forth(word_.data(), word_.size()) ||
                   is_keyword_forth_def(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case SYNTAX:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    is_label_ = false;
    t_ = NORMAL;
}
