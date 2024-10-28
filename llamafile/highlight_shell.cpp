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
#include <ctype.h>

enum {
    NORMAL,
    WORD,
    QUOTE,
    DQUOTE,
    DQUOTE_BACKSLASH,
    TICK,
    TICK_BACKSLASH,
    VAR,
    VAR2,
    COMMENT,
    LT,
    LT_LT,
    LT_LT_NAME,
    LT_LT_QNAME,
    HEREDOC_BOL,
    HEREDOC,
    BACKSLASH,
};

HighlightShell::HighlightShell() {
}

HighlightShell::~HighlightShell() {
}

void HighlightShell::feed(std::string *r, std::string_view input) {
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
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                append_wchar(&word_, c);
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
            } else if (c == '#') {
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
                append_wchar(r, c);
            }
            break;

        case BACKSLASH:
            append_wchar(r, c);
            *r += HI_RESET;
            t_ = NORMAL;
            break;

        case WORD:
            if (!isascii(c) || isalnum(c) || c == '_') {
                append_wchar(&word_, c);
            } else {
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
                append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
                break;
            } else if (c == '{') {
                append_wchar(r, c);
                *r += HI_VAR;
                t_ = VAR2;
                break;
            } else {
                *r += HI_VAR;
                t_ = VAR2;
            }
            // fallthrough

        case VAR2:
            if (!isascii(c) || isalnum(c) || c == '_') {
                append_wchar(r, c);
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COMMENT:
            append_wchar(r, c);
            if (c == '\n') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case QUOTE:
            append_wchar(r, c);
            if (c == '\'') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case DQUOTE:
            append_wchar(r, c);
            if (c == '"') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = DQUOTE_BACKSLASH;
            }
            break;

        case DQUOTE_BACKSLASH:
            append_wchar(r, c);
            t_ = DQUOTE;
            break;

        case TICK:
            append_wchar(r, c);
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = TICK_BACKSLASH;
            }
            break;

        case TICK_BACKSLASH:
            append_wchar(r, c);
            t_ = TICK;
            break;

        case LT:
            if (c == '<') {
                append_wchar(r, c);
                t_ = LT_LT;
                heredoc_.clear();
                pending_heredoc_ = false;
                indented_heredoc_ = false;
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT:
            if (c == '-') {
                indented_heredoc_ = true;
                append_wchar(r, c);
            } else if (c == '\\') {
                append_wchar(r, c);
            } else if (c == '\'') {
                t_ = LT_LT_QNAME;
                *r += HI_STRING;
                append_wchar(r, c);
            } else if (isalpha(c) || c == '_') {
                t_ = LT_LT_NAME;
                append_wchar(&heredoc_, c);
                append_wchar(r, c);
            } else if (!isblank(c)) {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT_NAME:
            if (isalnum(c) || c == '_') {
                t_ = LT_LT_NAME;
                append_wchar(&heredoc_, c);
                append_wchar(r, c);
            } else if (c == '\n') {
                append_wchar(r, c);
                *r += HI_STRING;
                t_ = HEREDOC_BOL;
            } else {
                pending_heredoc_ = true;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT_QNAME:
            append_wchar(r, c);
            if (c == '\'') {
                *r += HI_RESET;
                t_ = HEREDOC_BOL;
                pending_heredoc_ = true;
                t_ = NORMAL;
            } else {
                append_wchar(&heredoc_, c);
            }
            break;

        case HEREDOC_BOL:
            append_wchar(r, c);
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

        case HEREDOC:
            append_wchar(r, c);
            if (c == '\n')
                t_ = HEREDOC_BOL;
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void HighlightShell::flush(std::string *r) {
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
    case VAR2:
    case TICK:
    case TICK_BACKSLASH:
    case QUOTE:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case COMMENT:
    case HEREDOC_BOL:
    case HEREDOC:
    case LT_LT_QNAME:
    case BACKSLASH:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
