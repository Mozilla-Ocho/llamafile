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

// syntax highlighting for assembly code
//
// this syntax highlighter aims to support a blended dialect of
//
// - at&t style assembly (e.g. gnu as)
// - intel style assembly (e.g. nasm)
// - arm style assembly
// - c preprocessor
// - m4
//
// doing that requires special care

enum
{
    NORMAL,
    WORD,
    COMMENT,
    BACKSLASH,
    SLASH0,
    SLASH,
    REG0,
    REG,
    SLASH_SLASH,
    SLASH_STAR,
    SLASH_STAR_STAR,
    QUOTE,
    QUOTE_BACKSLASH,
    QUOTE_FINISH,
    DQUOTE,
    DQUOTE_BACKSLASH,
    DOLLAR,
    IMMEDIATE,
    IMMEDIATE_QUOTE,
    IMMEDIATE_QUOTE_BACKSLASH,
    IMMEDIATE_QUOTE_FINISH,
    HASH,
};

static bool
is_immediate(int c)
{
    switch (c) {
    case '&':
    case '*':
    case '+':
    case '-':
    case '/':
    case '0' ... '9':
    case '<':
    case '>':
    case 'A' ... 'Z':
    case '^':
    case '_':
    case 'a' ... 'z':
    case '|':
        return true;
    default:
        return false;
    }
}

HighlightAsm::HighlightAsm()
{
}

HighlightAsm::~HighlightAsm()
{
}

void
HighlightAsm::feed(std::string* r, std::string_view input)
{
    for (size_t i = 0; i < input.size(); ++i) {
        wchar_t c;
        int b = input[i] & 255;
        last_ = c_;
        if (!u_) {
            if (b < 0300) {
                c_ = c = b;
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
            c_ = c = b;
        }
        if (c == 0xFEFF)
            continue; // utf-8 bom
        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || //
                isalnum(c) || //
                c == '-' || //
                c == '.' || //
                c == '@' || //
                c == '_' || //
                (c == '#' && col_ == 0)) {
                t_ = WORD;
                lf::append_wchar(&word_, c);
                break;
            } else if (c == '#' && col_ && isspace(last_)) {
                t_ = HASH;
            } else if ((c == ';' || c == '!') && (!col_ || isspace(last_))) {
                t_ = COMMENT;
                *r += HI_COMMENT;
                lf::append_wchar(r, c);
            } else if (c == '/' && col_ == 0) {
                // bell system five allowed single slash comments
                // anywhere on the line, but we limit that a bit.
                t_ = SLASH0;
                *r += HI_COMMENT;
                *r += '/';
            } else if (c == '/') {
                t_ = SLASH;
                is_first_thing_on_line_ = false;
            } else if (c == '$') {
                *r += '$';
                t_ = DOLLAR;
                is_first_thing_on_line_ = false;
            } else if (c == '%') {
                t_ = REG0;
                is_first_thing_on_line_ = false;
            } else if (c == '\\') {
                t_ = BACKSLASH;
                *r += HI_ESCAPE;
                *r += '\\';
                is_first_thing_on_line_ = false;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else {
                if (c == '\n')
                    is_preprocessor_ = false;
                if (!isspace(c))
                    is_first_thing_on_line_ = false;
                if (c == ':')
                    is_first_thing_on_line_ = true;
                lf::append_wchar(r, c);
            }
            break;

        case DOLLAR:
            if (is_immediate(c) || c == '\'') {
                *r += HI_IMMEDIATE;
                t_ = IMMEDIATE;
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            // fallthrough

        Immediate:
        case IMMEDIATE:
            if (is_immediate(c)) {
                lf::append_wchar(r, c);
            } else if (c == '\'') {
                lf::append_wchar(r, c);
                t_ = IMMEDIATE_QUOTE;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case IMMEDIATE_QUOTE:
            if (c == '\\') {
                lf::append_wchar(r, c);
                t_ = IMMEDIATE_QUOTE_BACKSLASH;
            } else if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            } else {
                lf::append_wchar(r, c);
                t_ = IMMEDIATE_QUOTE_FINISH;
            }
            break;

        case IMMEDIATE_QUOTE_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = IMMEDIATE_QUOTE_FINISH;
            break;

        case IMMEDIATE_QUOTE_FINISH:
            if (c == '\'') {
                lf::append_wchar(r, c);
                t_ = IMMEDIATE;
            } else {
                // yes '" means '"' in bell system five
                t_ = IMMEDIATE;
                goto Immediate;
            }
            break;

        case BACKSLASH:
            lf::append_wchar(r, c);
            *r += HI_RESET;
            t_ = NORMAL;
            break;

        case HASH:
            if (isspace(c)) {
                *r += HI_COMMENT;
                *r += '#';
                t_ = COMMENT;
                goto Comment;
            } else {
                word_ += '#';
                t_ = WORD;
            }
            // fallthrough

        case WORD:
            if (!isascii(c) || isalnum(c) || c == '$' || c == '_' || c == '-' ||
                c == '.') {
                lf::append_wchar(&word_, c);
            } else {
                if (is_first_thing_on_line_) {
                    if (word_.size() > 1 && word_[0] == '#' &&
                        is_keyword_c_builtin(word_.data(), word_.size())) {
                        *r += HI_BUILTIN;
                        *r += word_;
                        *r += HI_RESET;
                        is_first_thing_on_line_ = false;
                        is_preprocessor_ = true;
                    } else if (c == ':') {
                        *r += HI_LABEL;
                        *r += word_;
                        *r += HI_RESET;
                    } else if (word_ == "C" || word_ == "dnl" ||
                               word_ == "m4_dnl") {
                        *r += HI_COMMENT;
                        *r += word_;
                        word_.clear();
                        t_ = COMMENT;
                        goto Comment;
                    } else {
                        *r += HI_KEYWORD;
                        *r += word_;
                        *r += HI_RESET;
                        if (!is_keyword_asm_prefix(word_.data(), word_.size()))
                            is_first_thing_on_line_ = false;
                    }
                } else if (is_preprocessor_ &&
                           is_keyword_c_builtin(word_.data(), word_.size())) {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_asm_qualifier(word_.data(),
                                                    word_.size())) {
                    *r += HI_QUALIFIER;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_c_constant(word_.data(), word_.size())) {
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

        case REG0:
            if (isalpha(c) || c == '(' || c == ')') {
                t_ = REG;
                *r += HI_REGISTER;
                *r += '%';
            } else {
                *r += '%';
                t_ = NORMAL;
                goto Normal;
            }
            // fallthrough

        case REG:
            if (isalnum(c)) {
                lf::append_wchar(r, c);
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case QUOTE:
            *r += c;
            if (c == '\'' || c == '\n') {
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
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = DQUOTE_BACKSLASH;
            }
            break;

        case DQUOTE_BACKSLASH:
            *r += c;
            t_ = DQUOTE;
            break;

        case SLASH0:
            if (c == '*') {
                *r += '*';
                t_ = SLASH_STAR;
            } else {
                t_ = COMMENT;
                goto Comment;
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

        Comment:
        case COMMENT:
        case SLASH_SLASH:
            lf::append_wchar(r, c);
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        default:
            __builtin_unreachable();
        }

        if (c != '\n') {
            col_ += 1;
        } else {
            col_ = 0;
            is_first_thing_on_line_ = true;
        }
    }
}

void
HighlightAsm::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        *r += word_;
        word_.clear();
        break;
    case HASH:
        *r += '#';
        break;
    case REG0:
        *r += '%';
        break;
    case SLASH:
        *r += '/';
        break;
    case REG:
    case SLASH0:
    case COMMENT:
    case BACKSLASH:
    case SLASH_SLASH:
    case SLASH_STAR:
    case SLASH_STAR_STAR:
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case IMMEDIATE:
    case IMMEDIATE_QUOTE:
    case IMMEDIATE_QUOTE_BACKSLASH:
    case IMMEDIATE_QUOTE_FINISH:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
}
