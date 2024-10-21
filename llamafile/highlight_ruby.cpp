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

// ruby lexical syntax is bananas and isn't even formally documented
//
// known issues
//
// 1. we can't tell this backtick apart from command substitution
//
//     def `(command)
//       return "just testing a backquote override"
//     end
//
// 2. we don't know <<arg isn't a heredoc. emacs doesn't get this right
//    either, but somehow vim does.
//
//     when /\.*\.h/
//       options[:includes] <<arg; true
//     when /--(\w+)=\"?(.*)\"?/
//       options[$1.to_sym] = $2; true
//

enum {
    NORMAL,
    WORD,
    EQUAL,
    EQUAL_WORD,
    QUOTE,
    QUOTE_BACKSLASH,
    DQUOTE,
    DQUOTE_BACKSLASH,
    TICK,
    TICK_BACKSLASH,
    COMMENT,
    LT,
    LT_LT,
    LT_LT_NAME,
    LT_LT_QNAME,
    HEREDOC_BOL,
    HEREDOC,
    COLON,
    COLON_WORD,
    AT,
    AT_WORD,
    DOLLAR,
    DOLLAR_WORD,
    PERCENT,
    PERCENT2,
    PERCENT_STRING,
    MULTICOM,
    MULTICOM_BOL,
    REGEX,
    REGEX_BACKSLASH,
};

static int mirror(int c) {
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

static bool isident(int c) {
    return !isascii(c) || //
           isalnum(c) || //
           c == '!' || //
           c == '$' || //
           c == '%' || //
           c == '&' || //
           c == '-' || //
           c == '/' || //
           c == '=' || //
           c == '?' || //
           c == '@' || //
           c == '^' || //
           c == '_';
}

HighlightRuby::HighlightRuby() {
}

HighlightRuby::~HighlightRuby() {
}

void HighlightRuby::feed(std::string *r, std::string_view input) {
    int c;
    for (size_t i = 0; i < input.size(); ++i) {
        c = input[i] & 255;

        if (!isblank(c) && c != '/' && c != '<')
            last_ = c;

        switch (t_) {

        Normal:
        case NORMAL:
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = WORD;
                word_ += c;
            } else if (c == ':') {
                t_ = COLON;
            } else if (c == '@') {
                t_ = AT;
            } else if (c == '=') {
                t_ = EQUAL;
            } else if (c == '$') {
                t_ = DOLLAR;
            } else if (c == '%') {
                t_ = PERCENT;
                q_ = 0;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                *r += c;
            } else if (c == '`') {
                t_ = TICK;
                *r += HI_STRING;
                *r += c;
            } else if (c == '#') {
                *r += HI_COMMENT;
                *r += c;
                t_ = COMMENT;
            } else if (c == '<' && !isalnum(last_)) {
                *r += c;
                t_ = LT;
            } else if (c == '/' && !isalnum(last_)) {
                t_ = REGEX;
                *r += HI_STRING;
                *r += c;
            } else if (c == '\n') {
                *r += c;
                if (pending_heredoc_) {
                    *r += HI_STRING;
                    pending_heredoc_ = false;
                    t_ = HEREDOC_BOL;
                    i_ = 0;
                }
            } else {
                *r += c;
            }
            break;

        case EQUAL:
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = EQUAL_WORD;
                word_ += c;
            } else {
                *r += '=';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case EQUAL_WORD:
            if (isident(c)) {
                word_ += c;
                break;
            } else if (word_ == "begin") {
                *r += HI_COMMENT;
                *r += "=begin";
                *r += c;
                if (c == '\n') {
                    t_ = MULTICOM_BOL;
                    i_ = 0;
                } else {
                    t_ = MULTICOM;
                }
                word_.clear();
                break;
            } else {
                *r += '=';
                t_ = WORD;
            }
            // fallthrough

        case WORD:
            if (isident(c)) {
                word_ += c;
            } else {
                if (is_keyword_ruby(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                    last_ = 0;
                } else if (is_keyword_ruby_builtin(word_.data(), word_.size())) {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                } else if (is_keyword_ruby_constant(word_.data(), word_.size())) {
                    *r += HI_CONSTANT;
                    *r += word_;
                    *r += HI_RESET;
                } else if (!word_.empty() && isupper(word_[0])) {
                    *r += HI_CLASS;
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

        case REGEX:
            *r += c;
            if (c == '/') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = REGEX_BACKSLASH;
            }
            break;

        case REGEX_BACKSLASH:
            *r += c;
            t_ = REGEX;
            break;

        case MULTICOM:
            *r += c;
            if (c == '\n') {
                t_ = MULTICOM_BOL;
                i_ = 0;
            }
            break;

        case MULTICOM_BOL:
            *r += c;
            if (c == "=end"[i_]) {
                if (++i_ == 4) {
                    t_ = NORMAL;
                    *r += HI_RESET;
                }
            } else {
                t_ = MULTICOM;
            }
            break;

        case COLON:
            if (isident(c)) {
                *r += HI_LISPKW;
                *r += ':';
                *r += c;
                t_ = COLON_WORD;
            } else {
                *r += ':';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COLON_WORD:
            if (isident(c)) {
                *r += c;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case AT:
            if (isident(c)) {
                *r += HI_VAR;
                *r += '@';
                *r += c;
                t_ = AT_WORD;
            } else {
                *r += '@';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case AT_WORD:
            if (isident(c)) {
                *r += c;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case PERCENT:
            if (c == 'q' || c == 'Q') {
                q_ = c;
                t_ = PERCENT2;
            } else if (ispunct(c)) {
                level_ = 1;
                opener_ = c;
                closer_ = mirror(c);
                *r += HI_STRING;
                *r += '%';
                *r += c;
                t_ = PERCENT_STRING;
            } else {
                *r += '%';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case PERCENT2:
            if (ispunct(c)) {
                level_ = 1;
                opener_ = c;
                closer_ = mirror(c);
                *r += HI_STRING;
                *r += '%';
                *r += q_;
                *r += c;
                t_ = PERCENT_STRING;
            } else {
                *r += '%';
                *r += q_;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case PERCENT_STRING:
            *r += c;
            if (c == opener_ && opener_ != closer_) {
                ++level_;
            } else if (c == closer_) {
                if (!--level_) {
                    *r += HI_RESET;
                    t_ = NORMAL;
                }
            }
            break;

        case DOLLAR:
            if (isdigit(c) || //
                c == '!' || //
                c == '"' || //
                c == '#' || //
                c == '$' || //
                c == '&' || //
                c == '-' || //
                c == '/' || //
                c == '<' || //
                c == '=' || //
                c == '>' || //
                c == '@' || //
                c == '\'' || //
                c == '\\' || //
                c == '^' || //
                c == '_' || //
                c == '`') {
                *r += HI_VAR;
                *r += '$';
                *r += c;
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (isalpha(c)) {
                *r += HI_VAR;
                *r += '$';
                *r += c;
                t_ = DOLLAR_WORD;
            } else {
                *r += '$';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case DOLLAR_WORD:
            if (isident(c)) {
                *r += c;
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COMMENT:
            if (c == '\n') {
                *r += HI_RESET;
                *r += c;
                t_ = NORMAL;
            } else {
                *r += c;
            }
            break;

        case QUOTE:
            *r += c;
            if (c == '\'') {
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

        case TICK:
            *r += c;
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '\\') {
                t_ = TICK_BACKSLASH;
            }
            break;

        case TICK_BACKSLASH:
            *r += c;
            t_ = TICK;
            break;

        case LT:
            if (c == '<') {
                *r += c;
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
                *r += c;
            } else if (c == '\'' || c == '`' || c == '"') {
                closer_ = c;
                t_ = LT_LT_QNAME;
                *r += HI_STRING;
                *r += c;
            } else if (isalpha(c) || c == '_') {
                t_ = LT_LT_NAME;
                heredoc_ += c;
                *r += c;
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT_NAME:
            if (isalnum(c) || c == '_') {
                t_ = LT_LT_NAME;
                heredoc_ += c;
                *r += c;
            } else if (c == '\n') {
                *r += c;
                *r += HI_STRING;
                t_ = HEREDOC_BOL;
            } else {
                pending_heredoc_ = true;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT_QNAME:
            *r += c;
            if (c == closer_) {
                *r += HI_RESET;
                t_ = HEREDOC_BOL;
                pending_heredoc_ = true;
                t_ = NORMAL;
            } else {
                heredoc_ += c;
            }
            break;

        case HEREDOC_BOL:
            *r += c;
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
            *r += c;
            if (c == '\n')
                t_ = HEREDOC_BOL;
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void HighlightRuby::flush(std::string *r) {
    switch (t_) {
    case EQUAL_WORD:
        *r += '=';
        // fallthrough
    case WORD:
        if (is_keyword_ruby(word_.data(), word_.size())) {
            *r += HI_KEYWORD;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_ruby_builtin(word_.data(), word_.size())) {
            *r += HI_BUILTIN;
            *r += word_;
            *r += HI_RESET;
        } else if (is_keyword_ruby_constant(word_.data(), word_.size())) {
            *r += HI_CONSTANT;
            *r += word_;
            *r += HI_RESET;
        } else if (!word_.empty() && isupper(word_[0])) {
            *r += HI_CLASS;
            *r += word_;
            *r += HI_RESET;
        } else {
            *r += word_;
        }
        word_.clear();
        break;
    case AT:
        *r += '@';
        break;
    case EQUAL:
        *r += '=';
        break;
    case COLON:
        *r += ':';
        break;
    case DOLLAR:
        *r += '$';
        break;
    case PERCENT:
        *r += '%';
        break;
    case PERCENT2:
        *r += '%';
        *r += q_;
        break;
    case REGEX:
    case REGEX_BACKSLASH:
    case PERCENT_STRING:
    case AT_WORD:
    case DOLLAR_WORD:
    case TICK:
    case TICK_BACKSLASH:
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case COMMENT:
    case HEREDOC_BOL:
    case HEREDOC:
    case LT_LT_QNAME:
    case COLON_WORD:
    case MULTICOM:
    case MULTICOM_BOL:
        *r += HI_RESET;
        break;
    default:
        break;
    }
    t_ = NORMAL;
}
