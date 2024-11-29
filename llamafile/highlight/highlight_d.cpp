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

// D probably has the best lexical syntax documentation of any language.
// https://dlang.org/spec/lex.html

enum
{
    NORMAL,
    WORD,
    QUOTE,
    QUOTE_BACKSLASH,
    DQUOTE,
    DQUOTE_BACKSLASH,
    SLASH,
    SLASH_SLASH,
    SLASH_STAR,
    SLASH_STAR_STAR,
    SLASH_PLUS,
    SLASH_PLUS_PLUS,
    SLASH_PLUS_SLASH,
    BACKTICK,
    R,
    R_DQUOTE,
    Q,
    Q_DQUOTE,
    Q_DQUOTE_STRING,
    Q_DQUOTE_STRING_END,
    Q_DQUOTE_IDENT,
    Q_DQUOTE_HEREDOC,
    Q_DQUOTE_HEREDOC_BOL,
    Q_DQUOTE_HEREDOC_END,
    X,
    X_DQUOTE,
};

static bool
is_line_terminator(wchar_t c)
{
    switch (c) {
    case '\r':
    case '\n':
    case 0x2028: // LINE SEPARATOR
    case 0x2029: // PARAGRAPH SEPARATOR
        return true;
    default:
        return false;
    }
}

static int
mirror(int c)
{
    switch (c) {
    case '(':
        return ')';
    case '{':
        return '}';
    case '[':
        return ']';
    case '<':
        return '>';
    default:
        return c;
    }
}

HighlightD::HighlightD()
{
}

HighlightD::~HighlightD()
{
}

void
HighlightD::feed(std::string* r, std::string_view input)
{
    int c;
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
        if (c == '\r')
            continue;
        if (c == 0xFEFF)
            continue; // utf-8 bom
        switch (t_) {

        Normal:
        case NORMAL:
            if (c == 'r') {
                t_ = R;
            } else if (c == 'q') {
                t_ = Q;
            } else if (c == 'x') {
                t_ = X;
            } else if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                lf::append_wchar(&word_, c);
            } else if (c == '`') {
                t_ = BACKTICK;
                *r += HI_STRING;
                *r += '`';
            } else if (c == '/') {
                t_ = SLASH;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += '\'';
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += '"';
            } else {
                lf::append_wchar(r, c);
            }
            break;

        Word:
        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                lf::append_wchar(&word_, c);
            } else {
                if (is_keyword_d(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_d_constant(word_.data(), word_.size())) {
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
            } else if (c == '+') {
                *r += HI_COMMENT;
                *r += "/+";
                t_ = SLASH_PLUS;
                depth_ = 1;
            } else {
                *r += '/';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case SLASH_SLASH:
            lf::append_wchar(r, c);
            if (is_line_terminator(c)) {
                *r += HI_RESET;
                t_ = NORMAL;
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

        case SLASH_PLUS:
            lf::append_wchar(r, c);
            if (c == '+') {
                t_ = SLASH_PLUS_PLUS;
            } else if (c == '/') {
                t_ = SLASH_PLUS_SLASH;
            }
            break;

        case SLASH_PLUS_PLUS:
            lf::append_wchar(r, c);
            if (c == '/') {
                if (!--depth_) {
                    *r += HI_RESET;
                    t_ = NORMAL;
                } else {
                    t_ = SLASH_PLUS;
                }
            } else if (c != '+') {
                t_ = SLASH_PLUS;
            }
            break;

        case SLASH_PLUS_SLASH:
            lf::append_wchar(r, c);
            if (c == '+') {
                ++depth_;
                t_ = SLASH_PLUS;
            } else if (c != '/') {
                t_ = SLASH_PLUS;
            }
            break;

        case QUOTE:
            lf::append_wchar(r, c);
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = QUOTE_BACKSLASH;
            }
            break;

        case QUOTE_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = QUOTE;
            break;

        case DQUOTE:
            lf::append_wchar(r, c);
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = DQUOTE_BACKSLASH;
            }
            break;

        case DQUOTE_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = DQUOTE;
            break;

        case R:
            if (c == '"') {
                t_ = R_DQUOTE;
                *r += 'r';
                *r += HI_STRING;
                *r += '"';
            } else {
                word_ += 'r';
                t_ = WORD;
                goto Word;
            }
            break;

        case R_DQUOTE:
            lf::append_wchar(r, c);
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case BACKTICK:
            lf::append_wchar(r, c);
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case Q:
            if (c == '"') {
                t_ = Q_DQUOTE;
                *r += 'q';
                *r += HI_STRING;
                *r += '"';
            } else {
                word_ += 'q';
                t_ = WORD;
                goto Word;
            }
            break;

        case Q_DQUOTE:
            lf::append_wchar(r, c);
            if (!isascii(c) || isalpha(c) || c == '_') {
                heredoc_.clear();
                lf::append_wchar(&heredoc_, c);
                t_ = Q_DQUOTE_IDENT;
            } else {
                opener_ = c;
                closer_ = mirror(c);
                depth_ = 1;
                t_ = Q_DQUOTE_STRING;
            }
            break;

        QDquoteString:
        case Q_DQUOTE_STRING:
            if (c == closer_) {
                if (closer_ != opener_) {
                    if (depth_) {
                        --depth_;
                    } else {
                        *r += HI_RESET;
                        *r += HI_WARNING;
                    }
                    if (!depth_) {
                        t_ = Q_DQUOTE_STRING_END;
                    }
                } else {
                    t_ = Q_DQUOTE_STRING_END;
                }
            } else if (c == opener_ && closer_ != opener_) {
                ++depth_;
            }
            lf::append_wchar(r, c);
            break;

        case Q_DQUOTE_STRING_END:
            if (c == '"') {
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += HI_RESET;
                *r += HI_WARNING;
                t_ = Q_DQUOTE_STRING;
                goto QDquoteString;
            }
            break;

        case Q_DQUOTE_IDENT:
            if (is_line_terminator(c)) {
                t_ = Q_DQUOTE_HEREDOC_BOL;
                heredoc2_.clear();
            } else if (!isascii(c) || isalpha(c) || c == '_') {
                lf::append_wchar(&heredoc_, c);
            } else {
                *r += HI_RESET;
                *r += HI_WARNING;
                t_ = Q_DQUOTE_HEREDOC_BOL;
                heredoc2_.clear();
            }
            lf::append_wchar(r, c);
            break;

        QDquoteHeredoc:
        case Q_DQUOTE_HEREDOC:
            lf::append_wchar(r, c);
            if (is_line_terminator(c)) {
                t_ = Q_DQUOTE_HEREDOC_BOL;
                heredoc2_.clear();
            }
            break;

        case Q_DQUOTE_HEREDOC_BOL:
            lf::append_wchar(r, c);
            if (is_line_terminator(c)) {
                t_ = Q_DQUOTE_HEREDOC_BOL;
                heredoc2_.clear();
            } else {
                lf::append_wchar(&heredoc2_, c);
                if (heredoc_.starts_with(heredoc2_)) {
                    if (heredoc_ == heredoc2_) {
                        t_ = Q_DQUOTE_HEREDOC_END;
                    }
                } else {
                    t_ = Q_DQUOTE_HEREDOC;
                }
            }
            break;

        case Q_DQUOTE_HEREDOC_END:
            if (c == '"') {
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += HI_RESET;
                *r += HI_WARNING;
                t_ = Q_DQUOTE_HEREDOC;
                goto QDquoteHeredoc;
            }
            break;

        case X:
            if (c == '"') {
                *r += 'x';
                *r += HI_STRING;
                *r += '"';
                t_ = X_DQUOTE;
            } else {
                word_ += 'x';
                t_ = WORD;
                goto Word;
            }
            break;

        case X_DQUOTE:
            if (is_line_terminator(c) || isspace(c) || isxdigit(c)) {
                lf::append_wchar(r, c);
            } else if (c == '"') {
                *r += '"';
                *r += HI_RESET;
                t_ = NORMAL;
            } else {
                *r += HI_RESET;
                *r += HI_WARNING;
                lf::append_wchar(r, c);
                *r += HI_RESET;
                *r += HI_STRING;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightD::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_d(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_d_constant(word_.data(), word_.size())) {
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
    case R:
        *r += 'r';
        break;
    case Q:
        *r += 'q';
        break;
    case X:
        *r += 'x';
        break;
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case SLASH_SLASH:
    case SLASH_STAR:
    case SLASH_STAR_STAR:
    case SLASH_PLUS:
    case SLASH_PLUS_PLUS:
    case SLASH_PLUS_SLASH:
    case BACKTICK:
    case R_DQUOTE:
    case Q_DQUOTE:
    case Q_DQUOTE_STRING:
    case Q_DQUOTE_STRING_END:
    case Q_DQUOTE_IDENT:
    case Q_DQUOTE_HEREDOC:
    case Q_DQUOTE_HEREDOC_BOL:
    case Q_DQUOTE_HEREDOC_END:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
}
