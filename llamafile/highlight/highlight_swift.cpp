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

// apple swift
//
// - string interpolation
//   "hi \(2 * (2 + 2)) there"
//
// - hash strings
//   #"hi \#(2 * (2 + 2)) there"#
//   ##"hi \##(2 * (2 + 2)) there"##
//
// - hash multiline strings
//   #"""hi \#(2 * (2 + 2)) there"""#
//   ##"""hi \##(2 * (2 + 2)) there"""##
//

enum
{
    NORMAL,
    WORD,
    SLASH,
    SLASH_SLASH,
    SLASH_STAR,
    SLASH_SLASH_BACKSLASH,
    SLASH_STAR_STAR,
    HASH,
    DQUOTE,
    DQUOTESTR,
    DQUOTESTR_BACKSLASH,
    DQUOTESTR_END,
    DQUOTE2,
    DQUOTE3,
    DQUOTE3_BACKSLASH,
    DQUOTE31,
    DQUOTE32,
    DQUOTE3_END,
    REGEX,
    REGEX_END,
    REGEX_BACKSLASH,
};

enum
{
    EXPECT_VALUE,
    EXPECT_OPERATOR,
};

HighlightSwift::HighlightSwift()
{
}

HighlightSwift::~HighlightSwift()
{
}

void
HighlightSwift::feed(std::string* r, std::string_view input)
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
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                lf::append_wchar(&word_, c);
            } else if (c == '/') {
                t_ = SLASH;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += '"';
                hash1_ = 0;
                expect_ = EXPECT_OPERATOR;
            } else if (c == '#') {
                t_ = HASH;
                hash1_ = 1;
                expect_ = EXPECT_OPERATOR;
            } else if (c == '(' && nesti_ && nesti_ < sizeof(nest_)) {
                *r += '(';
                nest_[nesti_] = NORMAL;
                hash_[nesti_] = 0;
                nesti_++;
                expect_ = EXPECT_VALUE;
            } else if (c == ')' && nesti_) {
                expect_ = EXPECT_OPERATOR;
                --nesti_;
                t_ = nest_[nesti_];
                hash1_ = hash_[nesti_];
                if (t_ != NORMAL)
                    *r += HI_STRING;
                *r += ')';
            } else if (c == ')' || c == ']' || isdigit(c) || c == '.') {
                expect_ = EXPECT_OPERATOR;
                lf::append_wchar(r, c);
            } else if (ispunct(c) || c == '\n') {
                expect_ = EXPECT_VALUE;
                lf::append_wchar(r, c);
            } else {
                lf::append_wchar(r, c);
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                lf::append_wchar(&word_, c);
            } else {
                if (is_keyword_swift(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_swift_type(word_.data(), word_.size())) {
                    *r += HI_TYPE;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_swift_builtin(word_.data(),
                                                    word_.size())) {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_swift_constant(word_.data(),
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
            } else if (expect_ == EXPECT_VALUE) {
                *r += HI_STRING;
                *r += '/';
                hash1_ = 0;
                t_ = REGEX;
                goto Regex;
            } else {
                *r += '/';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case SLASH_SLASH:
            lf::append_wchar(r, c);
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = SLASH_SLASH_BACKSLASH;
            }
            break;

        case SLASH_SLASH_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = SLASH_SLASH;
            break;

        case SLASH_STAR:
            lf::append_wchar(r, c);
            if (c == '*')
                t_ = SLASH_STAR_STAR;
            break;

        case SLASH_STAR_STAR:
            lf::append_wchar(r, c);
            if (c == '/') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c != '*') {
                t_ = SLASH_STAR;
            }
            break;

        case HASH:
            if (c == '#') {
                ++hash1_;
            } else if (c == '"') {
                *r += HI_STRING;
                for (int i = 0; i < hash1_; ++i)
                    *r += '#';
                *r += '"';
                t_ = DQUOTE;
            } else if (c == '/') {
                *r += HI_STRING;
                for (int i = 0; i < hash1_; ++i)
                    *r += '#';
                *r += '/';
                t_ = REGEX;
            } else {
                for (int i = 0; i < hash1_; ++i)
                    *r += '#';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        Dquote:
        case DQUOTE:
            lf::append_wchar(r, c);
            if (c == '"') {
                t_ = DQUOTE2;
                hash2_ = 0;
            } else if (c == '\\') {
                t_ = DQUOTESTR_BACKSLASH;
                hash2_ = 0;
            } else {
                t_ = DQUOTESTR;
            }
            break;

        Dquotestr:
        case DQUOTESTR:
            lf::append_wchar(r, c);
            if (c == '"') {
                t_ = DQUOTESTR_END;
                hash2_ = 0;
            } else if (c == '\\') {
                t_ = DQUOTESTR_BACKSLASH;
                hash2_ = 0;
            }
            break;

        case DQUOTESTR_END:
            if (hash2_ == hash1_) {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            } else if (c == '#') {
                *r += '#';
                ++hash2_;
            } else {
                t_ = DQUOTESTR;
                goto Dquotestr;
            }
            break;

        case DQUOTESTR_BACKSLASH:
            if (c == '#' && hash2_ < hash1_) {
                *r += '#';
                ++hash2_;
            } else if (c == '(' && hash2_ == hash1_ && nesti_ < sizeof(nest_)) {
                *r += '(';
                *r += HI_RESET;
                nest_[nesti_] = DQUOTESTR;
                hash_[nesti_] = hash1_;
                ++nesti_;
                t_ = NORMAL;
            } else {
                t_ = DQUOTESTR;
                goto Dquotestr;
            }
            break;

        case DQUOTE2:
            if (c == '"') {
                *r += '"';
                t_ = DQUOTE3;
            } else if (c == '#' && hash2_ < hash1_) {
                *r += '#';
                ++hash2_;
            } else if (hash2_ == hash1_) {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            } else {
                t_ = DQUOTESTR;
                goto Dquotestr;
            }
            break;

        Dquote3:
        case DQUOTE3:
            lf::append_wchar(r, c);
            if (c == '"') {
                t_ = DQUOTE31;
            } else if (c == '\\') {
                t_ = DQUOTE3_BACKSLASH;
                hash2_ = 0;
            }
            break;

        case DQUOTE31:
            if (c == '"') {
                *r += '"';
                t_ = DQUOTE32;
            } else {
                t_ = DQUOTE3;
                goto Dquote3;
            }
            break;

        case DQUOTE32:
            if (c == '"') {
                *r += '"';
                t_ = DQUOTESTR_END;
                hash2_ = 0;
            } else {
                t_ = DQUOTE3;
                goto Dquote3;
            }
            break;

        case DQUOTE3_BACKSLASH:
            if (c == '#' && hash2_ < hash1_) {
                *r += '#';
                ++hash2_;
            } else if (c == '(' && hash2_ == hash1_ && nesti_ < sizeof(nest_)) {
                *r += '(';
                *r += HI_RESET;
                nest_[nesti_] = DQUOTE3;
                hash_[nesti_] = hash1_;
                ++nesti_;
                t_ = NORMAL;
            } else {
                t_ = DQUOTE3;
                goto Dquote3;
            }
            break;

        case DQUOTE3_END:
            if (hash2_ == hash1_) {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            } else if (c == '#') {
                *r += '#';
                ++hash2_;
            } else {
                t_ = DQUOTE3;
                goto Dquote3;
            }
            break;

        Regex:
        case REGEX:
            lf::append_wchar(r, c);
            if (c == '/') {
                t_ = REGEX_END;
                hash2_ = 0;
            } else if (c == '\\') {
                t_ = REGEX_BACKSLASH;
                hash2_ = 0;
            }
            break;

        case REGEX_END:
            if (hash2_ == hash1_) {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            } else if (c == '#') {
                *r += '#';
                ++hash2_;
            } else {
                t_ = REGEX;
                goto Regex;
            }
            break;

        case REGEX_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = REGEX;
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightSwift::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_swift(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_swift_type(word_.data(), word_.size())) {
            *r += HI_TYPE;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_swift_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_swift_constant(word_.data(), word_.size())) {
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
    case HASH:
        for (int i = 0; i < hash1_; ++i)
            *r += '#';
        break;
    case SLASH_SLASH:
    case SLASH_SLASH_BACKSLASH:
    case SLASH_STAR:
    case SLASH_STAR_STAR:
    case DQUOTE:
    case DQUOTESTR:
    case DQUOTESTR_BACKSLASH:
    case DQUOTESTR_END:
    case DQUOTE2:
    case DQUOTE3:
    case DQUOTE3_BACKSLASH:
    case DQUOTE31:
    case DQUOTE32:
    case DQUOTE3_END:
    case REGEX:
    case REGEX_END:
    case REGEX_BACKSLASH:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
    expect_ = 0;
    nesti_ = 0;
}
