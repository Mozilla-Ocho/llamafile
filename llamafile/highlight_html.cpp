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
#define TAG 1
#define TAG2 2
#define TAG_EXCLAIM 3
#define TAG_EXCLAIM_HYPHEN 4
#define KEY 5
#define VAL 6
#define QUOTE 7
#define DQUOTE 8
#define COMMENT 9
#define COMMENT_HYPHEN 10
#define COMMENT_HYPHEN_HYPHEN 11
#define RELAY 12

HighlightHtml::HighlightHtml() {
}

HighlightHtml::~HighlightHtml() {
}

void HighlightHtml::feed(std::string *r, std::string_view input) {
    char c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i];
        switch (t_) {

        case NORMAL:
            if (c == '<') {
                t_ = TAG;
                name_.clear();
            } else {
                *r += c;
            }
            break;

        case TAG:
            if (c == '!') {
                t_ = TAG_EXCLAIM;
            } else if (c == '>' || isspace(c)) {
                *r += '<';
                *r += c;
                t_ = NORMAL;
            } else {
                *r += '<';
                *r += HI_TAG;
                *r += c;
                name_ += tolower(c);
                t_ = TAG2;
            }
            break;

        case TAG2:
            if (c == '>') {
                *r += HI_RESET;
                *r += c;
                goto OnTag;
            } else if (isspace(c)) {
                *r += c;
                t_ = KEY;
                *r += HI_ATTRIB;
            } else {
                *r += c;
                name_ += tolower(c);
            }
            break;

        case TAG_EXCLAIM:
            if (c == '-') {
                t_ = TAG_EXCLAIM_HYPHEN;
            } else {
                *r += "<!";
                *r += c;
                t_ = NORMAL;
            }
            break;

        case TAG_EXCLAIM_HYPHEN:
            if (c == '-') {
                *r += HI_COMMENT;
                *r += "<!--";
                t_ = COMMENT;
            } else {
                *r += "<!-";
                *r += c;
                t_ = NORMAL;
            }
            break;

        case COMMENT:
            *r += c;
            if (c == '-')
                t_ = COMMENT_HYPHEN;
            break;

        case COMMENT_HYPHEN:
            *r += c;
            if (c == '-') {
                t_ = COMMENT_HYPHEN_HYPHEN;
            } else {
                t_ = COMMENT;
            }
            break;

        case COMMENT_HYPHEN_HYPHEN:
            *r += c;
            if (c == '>') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                t_ = COMMENT;
            }
            break;

        case KEY:
            if (c == '=') {
                *r += HI_RESET;
                *r += c;
                t_ = VAL;
            } else if (c == '>') {
                *r += HI_RESET;
                *r += c;
                goto OnTag;
            } else {
                *r += c;
            }
            break;

        case VAL:
            if (isspace(c)) {
                *r += c;
                t_ = KEY;
                *r += HI_ATTRIB;
            } else if (c == '\'') {
                *r += HI_STRING;
                *r += c;
                t_ = QUOTE;
            } else if (c == '"') {
                *r += HI_STRING;
                *r += c;
                t_ = DQUOTE;
            } else if (c == '>') {
                *r += c;
                goto OnTag;
            } else {
                *r += c;
            }
            break;

        case QUOTE:
            *r += c;
            if (c == '\'') {
                *r += HI_RESET;
                t_ = VAL;
            }
            break;

        case DQUOTE:
            *r += c;
            if (c == '"') {
                *r += HI_RESET;
                t_ = VAL;
            }
            break;

        OnTag:
            t_ = NORMAL;
            if (name_ == "script") {
                pending_.clear();
                closer_ = "</script>";
                highlighter_ = Highlight::create("js");
                t_ = RELAY;
                i_ = 0;
            } else if (name_ == "style") {
                pending_.clear();
                closer_ = "</style>";
                highlighter_ = Highlight::create("css");
                t_ = RELAY;
                i_ = 0;
            }
            break;

        case RELAY:
            if (closer_[i_] == tolower(c)) {
                pending_ += c;
                if (++i_ == closer_.size()) {
                    highlighter_->flush(r);
                    delete highlighter_;
                    highlighter_ = nullptr;
                    if (closer_ == "</style>")
                        pending_ = "</" HI_TAG "style" HI_RESET ">";
                    else if (closer_ == "</script>")
                        pending_ = "</" HI_TAG "script" HI_RESET ">";
                    *r += pending_;
                    t_ = NORMAL;
                    i_ = 0;
                }
            } else {
                pending_ += c;
                highlighter_->feed(r, pending_);
                pending_.clear();
                i_ = 0;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void HighlightHtml::flush(std::string *r) {
    switch (t_) {
    case TAG:
        *r += '<';
        break;
    case TAG_EXCLAIM:
        *r += "<!";
        break;
    case TAG_EXCLAIM_HYPHEN:
        *r += "<!-";
        break;
    case COMMENT_HYPHEN_HYPHEN:
    case COMMENT_HYPHEN:
    case COMMENT:
    case DQUOTE:
    case QUOTE:
    case TAG2:
    case KEY:
        *r += HI_RESET;
        break;
    case RELAY:
        highlighter_->feed(r, pending_);
        highlighter_->flush(r);
        delete highlighter_;
        highlighter_ = nullptr;
        break;
    default:
        break;
    }
    pending_.clear();
    closer_.clear();
    name_.clear();
    t_ = NORMAL;
}
