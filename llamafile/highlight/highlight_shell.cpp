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

enum
{
    NORMAL,
    WORD,
    QUOTE,
    DQUOTE,
    DQUOTE_VAR,
    DQUOTE_VAR2,
    DQUOTE_CURL,
    DQUOTE_CURL_BACKSLASH,
    DQUOTE_BACKSLASH,
    TICK,
    TICK_BACKSLASH,
    VAR,
    VAR2,
    CURL,
    CURL_BACKSLASH,
    COMMENT,
    LT,
    LT_LT,
    LT_LT_NAME,
    LT_LT_QNAME,
    HEREDOC_BOL,
    HEREDOC,
    HEREDOC_VAR,
    HEREDOC_VAR2,
    HEREDOC_CURL,
    HEREDOC_CURL_BACKSLASH,
    BACKSLASH,
};

HighlightShell::HighlightShell()
{
}

HighlightShell::~HighlightShell()
{
}

void
HighlightShell::feed(std::string* r, std::string_view input)
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
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += '\'';
            } else if (c == '\\') {
                t_ = BACKSLASH;
                *r += HI_ESCAPE;
                *r += '\\';
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += '"';
            } else if (c == '`') {
                t_ = TICK;
                *r += HI_STRING;
                *r += '`';
            } else if (c == '$') {
                t_ = VAR;
                *r += '$';
            } else if (c == '<') {
                t_ = LT;
                *r += '<';
            } else if (c == '#' && (!last_ || isspace(last_))) {
                *r += HI_COMMENT;
                *r += '#';
                t_ = COMMENT;
            } else if (c == '\n') {
                *r += '\n';
                if (pending_heredoc_) {
                    *r += HI_STRING;
                    pending_heredoc_ = false;
                    t_ = HEREDOC_BOL;
                    i_ = 0;
                }
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case BACKSLASH:
            lf::append_wchar(r, c);
            *r += HI_RESET;
            t_ = NORMAL;
            break;

        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                lf::append_wchar(&word_, c);
            } else {
                if (is_keyword_shell(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_shell_builtin(word_.data(),
                                                    word_.size())) {
                    *r += HI_BUILTIN;
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

        case VAR:
            if (c == '!' || //
                c == '#' || //
                c == '$' || //
                c == '*' || //
                c == '-' || //
                c == '?' || //
                c == '@' || //
                c == '\\' || //
                c == '^') {
                *r += HI_VAR;
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
                break;
            } else if (c == '{') {
                *r += '{';
                *r += HI_VAR;
                t_ = CURL;
                curl_ = 1;
                break;
            } else {
                *r += HI_VAR;
                t_ = VAR2;
            }
            // fallthrough

        case VAR2:
            if (!isascii(c) || isalnum(c) || c == '_') {
                lf::append_wchar(r, c);
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case CURL:
            if (c == '\\') {
                t_ = CURL_BACKSLASH;
                *r += HI_RESET;
                *r += HI_ESCAPE;
                *r += '\\';
            } else if (c == '{') {
                *r += HI_RESET;
                *r += '{';
                *r += HI_VAR;
                ++curl_;
            } else if (c == '}') {
                *r += HI_RESET;
                *r += '}';
                if (!--curl_) {
                    t_ = NORMAL;
                }
            } else if (ispunct(c)) {
                *r += HI_RESET;
                lf::append_wchar(r, c);
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case CURL_BACKSLASH:
            lf::append_wchar(r, c);
            *r += HI_RESET;
            t_ = CURL;
            break;

        case COMMENT:
            lf::append_wchar(r, c);
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case QUOTE:
            lf::append_wchar(r, c);
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        Dquote:
        case DQUOTE:
            if (c == '"') {
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                lf::append_wchar(r, c);
                t_ = DQUOTE_BACKSLASH;
            } else if (c == '$') {
                t_ = DQUOTE_VAR;
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case DQUOTE_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = DQUOTE;
            break;

        case DQUOTE_VAR:
            if (c == '!' || //
                c == '#' || //
                c == '$' || //
                c == '*' || //
                c == '-' || //
                c == '?' || //
                c == '@' || //
                c == '\\' || //
                c == '^') {
                *r += HI_BOLD;
                *r += '$';
                lf::append_wchar(r, c);
                *r += HI_UNBOLD;
                t_ = DQUOTE;
                break;
            } else if (c == '{') {
                *r += HI_BOLD;
                *r += "${";
                t_ = DQUOTE_CURL;
                curl_ = 1;
                break;
            } else if (c == '(') {
                *r += '$';
                t_ = DQUOTE_VAR2;
            } else {
                *r += HI_BOLD;
                *r += '$';
                t_ = DQUOTE_VAR2;
            }
            // fallthrough

        case DQUOTE_VAR2:
            if (!isascii(c) || isalnum(c) || c == '_') {
                lf::append_wchar(r, c);
            } else {
                *r += HI_UNBOLD;
                t_ = DQUOTE;
                goto Dquote;
            }
            break;

        case DQUOTE_CURL:
            if (c == '\\') {
                t_ = DQUOTE_CURL_BACKSLASH;
                *r += '\\';
            } else if (c == '{') {
                *r += '{';
                ++curl_;
            } else if (c == '}') {
                *r += '}';
                if (!--curl_) {
                    *r += HI_UNBOLD;
                    t_ = DQUOTE;
                }
            } else if (ispunct(c)) {
                lf::append_wchar(r, c);
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case DQUOTE_CURL_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = DQUOTE_CURL;
            break;

        case TICK:
            lf::append_wchar(r, c);
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = TICK_BACKSLASH;
            }
            break;

        case TICK_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = TICK;
            break;

        case LT:
            if (c == '<') {
                lf::append_wchar(r, c);
                t_ = LT_LT;
                heredoc_.clear();
                pending_heredoc_ = false;
                indented_heredoc_ = false;
                no_interpolation_ = false;
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT:
            if (c == '-') {
                indented_heredoc_ = true;
                lf::append_wchar(r, c);
            } else if (c == '\\') {
                lf::append_wchar(r, c);
            } else if (c == '\'') {
                t_ = LT_LT_QNAME;
                *r += HI_STRING;
                lf::append_wchar(r, c);
                no_interpolation_ = true;
            } else if (isalpha(c) || c == '_') {
                t_ = LT_LT_NAME;
                lf::append_wchar(&heredoc_, c);
                lf::append_wchar(r, c);
            } else if (isascii(c) && isblank(c)) {
                *r += c;
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT_NAME:
            if (isalnum(c) || c == '_') {
                t_ = LT_LT_NAME;
                lf::append_wchar(&heredoc_, c);
                lf::append_wchar(r, c);
            } else if (c == '\n') {
                lf::append_wchar(r, c);
                *r += HI_STRING;
                t_ = HEREDOC_BOL;
            } else {
                pending_heredoc_ = true;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT_QNAME:
            lf::append_wchar(r, c);
            if (c == '\'') {
                *r += HI_RESET;
                t_ = HEREDOC_BOL;
                pending_heredoc_ = true;
                t_ = NORMAL;
            } else {
                lf::append_wchar(&heredoc_, c);
            }
            break;

        case HEREDOC_BOL:
            lf::append_wchar(r, c);
            if (c == '\n') {
                if (i_ == heredoc_.size()) {
                    t_ = NORMAL;
                    *r += HI_RESET;
                }
                i_ = 0;
            } else if (c == '\t' && indented_heredoc_) {
                // do nothing
            } else if (i_ < heredoc_.size() && (heredoc_[i_] & 255) == c) {
                i_++;
            } else {
                t_ = HEREDOC;
                i_ = 0;
            }
            break;

        Heredoc:
        case HEREDOC:
            if (c == '\n') {
                *r += '\n';
                t_ = HEREDOC_BOL;
            } else if (c == '$' && !no_interpolation_) {
                t_ = HEREDOC_VAR;
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case HEREDOC_VAR:
            if (c == '!' || //
                c == '#' || //
                c == '$' || //
                c == '*' || //
                c == '-' || //
                c == '?' || //
                c == '@' || //
                c == '\\' || //
                c == '^') {
                *r += HI_BOLD;
                *r += '$';
                lf::append_wchar(r, c);
                *r += HI_UNBOLD;
                t_ = HEREDOC;
                break;
            } else if (c == '{') {
                *r += HI_BOLD;
                *r += "${";
                t_ = HEREDOC_CURL;
                curl_ = 1;
                break;
            } else if (c == '(') {
                *r += '$';
                t_ = HEREDOC_VAR2;
            } else {
                *r += HI_BOLD;
                *r += '$';
                t_ = HEREDOC_VAR2;
            }
            // fallthrough

        case HEREDOC_VAR2:
            if (!isascii(c) || isalnum(c) || c == '_') {
                lf::append_wchar(r, c);
            } else {
                *r += HI_UNBOLD;
                t_ = HEREDOC;
                goto Heredoc;
            }
            break;

        case HEREDOC_CURL:
            if (c == '\\') {
                t_ = HEREDOC_CURL_BACKSLASH;
                *r += '\\';
            } else if (c == '{') {
                *r += '{';
                ++curl_;
            } else if (c == '}') {
                *r += '}';
                if (!--curl_) {
                    *r += HI_UNBOLD;
                    t_ = HEREDOC;
                }
            } else if (ispunct(c)) {
                lf::append_wchar(r, c);
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case HEREDOC_CURL_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = HEREDOC_CURL;
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightShell::flush(std::string* r)
{
    switch (t_) {
    case WORD:
        if (is_keyword_shell(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_shell_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case DQUOTE_VAR:
        *r += '$';
        *r += HI_RESET;
        break;
    case HEREDOC_VAR:
        *r += '$';
        *r += HI_RESET;
        break;
    case VAR2:
    case CURL:
    case CURL_BACKSLASH:
    case TICK:
    case TICK_BACKSLASH:
    case QUOTE:
    case DQUOTE:
    case DQUOTE_VAR2:
    case DQUOTE_CURL:
    case DQUOTE_CURL_BACKSLASH:
    case DQUOTE_BACKSLASH:
    case COMMENT:
    case HEREDOC_BOL:
    case HEREDOC:
    case HEREDOC_VAR2:
    case HEREDOC_CURL:
    case HEREDOC_CURL_BACKSLASH:
    case LT_LT_QNAME:
    case BACKSLASH:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
    last_ = 0;
}
