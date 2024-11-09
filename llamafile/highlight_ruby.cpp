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
    DQUOTE_HASH,
    DQUOTE_HASH_DOLLAR,
    DQUOTE_HASH_DOLLAR_WORD,
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
    REGEX_HASH,
    REGEX_HASH_DOLLAR,
    REGEX_HASH_DOLLAR_WORD,
    REGEX_BACKSLASH,
    QUESTION,
    QUESTION_BACKSLASH,
};

enum {
    EXPECT_VALUE,
    EXPECT_OPERATOR,
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

static bool ispunct_overridable(int c) {
    switch (c) {
    case '%':
    case '&':
    case '*':
    case '+':
    case '-':
    case '/':
    case '<':
    case '>':
    case '^':
    case '_':
    case '`':
    case '|':
    case '~':
        return true;
    default:
        return false;
    }
}

static bool is_dollar_one(int c) {
    switch (c) {
    case '!':
    case '"':
    case '#':
    case '$':
    case '&':
    case '-':
    case '/':
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case '<':
    case '=':
    case '>':
    case '@':
    case '\'':
    case '\\':
    case '^':
    case '_':
    case '`':
        return true;
    default:
        return false;
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
            if (!isascii(c) || isalpha(c) || c == '_' ||
                (is_definition_ && ispunct_overridable(c))) {
                t_ = WORD;
                lf::append_wchar(&word_, c);
                is_definition_ = false;
            } else if (c == ':') {
                t_ = COLON;
                expect_ = EXPECT_OPERATOR;
                is_definition_ = false;
            } else if (c == '@') {
                t_ = AT;
                is_definition_ = false;
            } else if (c == '=') {
                t_ = EQUAL;
                expect_ = EXPECT_VALUE;
                is_definition_ = false;
            } else if (c == '?' && expect_ == EXPECT_VALUE) {
                t_ = QUESTION;
                is_definition_ = false;
            } else if (c == '$') {
                t_ = DOLLAR;
                expect_ = EXPECT_OPERATOR;
                is_definition_ = false;
            } else if (c == '%' && expect_ == EXPECT_VALUE) {
                t_ = PERCENT;
                q_ = 0;
                expect_ = EXPECT_OPERATOR;
            } else if (c == '\'') {
                t_ = QUOTE;
                *r += HI_STRING;
                lf::append_wchar(r, c);
                expect_ = EXPECT_OPERATOR;
                is_definition_ = false;
            } else if (c == '"') {
                t_ = DQUOTE;
                *r += HI_STRING;
                lf::append_wchar(r, c);
                expect_ = EXPECT_OPERATOR;
                is_definition_ = false;
            } else if (c == '`') {
                t_ = TICK;
                *r += HI_STRING;
                lf::append_wchar(r, c);
                expect_ = EXPECT_OPERATOR;
            } else if (c == '#') {
                *r += HI_COMMENT;
                lf::append_wchar(r, c);
                t_ = COMMENT;
            } else if (c == '<' && expect_ == EXPECT_VALUE) {
                lf::append_wchar(r, c);
                t_ = LT;
            } else if (c == '/' && expect_ == EXPECT_VALUE) {
                t_ = REGEX;
                *r += HI_STRING;
                lf::append_wchar(r, c);
            } else if (c == '{' && nesti_ && nesti_ < sizeof(nest_)) {
                expect_ = EXPECT_VALUE;
                *r += '{';
                nest_[nesti_++] = NORMAL;
                is_definition_ = false;
            } else if (c == '}' && nesti_) {
                if ((t_ = nest_[--nesti_]) != NORMAL)
                    *r += HI_STRING;
                *r += '}';
                expect_ = EXPECT_OPERATOR;
                is_definition_ = false;
            } else if (c == '\n') {
                expect_ = EXPECT_VALUE;
                lf::append_wchar(r, c);
                if (pending_heredoc_) {
                    *r += HI_STRING;
                    pending_heredoc_ = false;
                    t_ = HEREDOC_BOL;
                    i_ = 0;
                }
            } else if (c == ']') {
                expect_ = EXPECT_OPERATOR;
                *r += ']';
                is_definition_ = false;
            } else if (ispunct(c)) {
                expect_ = EXPECT_VALUE;
                lf::append_wchar(r, c);
                is_definition_ = false;
            } else if (isdigit(c) || c == '.') {
                expect_ = EXPECT_OPERATOR;
                lf::append_wchar(r, c);
                is_definition_ = false;
            } else if (isspace(c)) {
                lf::append_wchar(r, c);
            } else {
                lf::append_wchar(r, c);
                is_definition_ = false;
            }
            break;

        case EQUAL:
            if (!isascii(c) || isalpha(c) || c == '_') {
                t_ = EQUAL_WORD;
                lf::append_wchar(&word_, c);
            } else {
                *r += '=';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case EQUAL_WORD:
            if (isident(c)) {
                lf::append_wchar(&word_, c);
                break;
            } else if (word_ == "begin") {
                *r += HI_COMMENT;
                *r += "=begin";
                lf::append_wchar(r, c);
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
                lf::append_wchar(&word_, c);
            } else {
                if (is_keyword_ruby(word_.data(), word_.size())) {
                    *r += HI_KEYWORD;
                    *r += word_;
                    *r += HI_RESET;
                    expect_ = EXPECT_VALUE;
                    if (word_ == "def") {
                        is_definition_ = true;
                    }
                } else if (is_keyword_ruby_builtin(word_.data(), word_.size())) {
                    *r += HI_BUILTIN;
                    *r += word_;
                    *r += HI_RESET;
                    expect_ = EXPECT_VALUE;
                } else if (is_keyword_ruby_constant(word_.data(), word_.size())) {
                    *r += HI_CONSTANT;
                    *r += word_;
                    *r += HI_RESET;
                    expect_ = EXPECT_OPERATOR;
                } else if (!word_.empty() && isupper(word_[0])) {
                    *r += HI_CLASS;
                    *r += word_;
                    *r += HI_RESET;
                    expect_ = EXPECT_OPERATOR;
                } else {
                    *r += word_;
                    expect_ = EXPECT_OPERATOR;
                }
                word_.clear();
                t_ = NORMAL;
                goto Normal;
            }
            break;

        Regex:
        case REGEX:
            if (c == '/') {
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '#') {
                t_ = REGEX_HASH;
            } else if (c == '\\') {
                lf::append_wchar(r, c);
                t_ = REGEX_BACKSLASH;
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case REGEX_HASH:
            if (c == '{' && nesti_ < sizeof(nest_)) {
                *r += HI_BOLD;
                *r += '#';
                *r += HI_UNBOLD;
                *r += HI_STRING;
                *r += '{';
                *r += HI_RESET;
                expect_ = EXPECT_VALUE;
                nest_[nesti_++] = REGEX;
                t_ = NORMAL;
            } else if (c == '$') {
                t_ = REGEX_HASH_DOLLAR;
            } else {
                *r += '#';
                t_ = REGEX;
                goto Regex;
            }
            break;

        case REGEX_HASH_DOLLAR:
            if (is_dollar_one(c)) {
                *r += '#';
                *r += HI_BOLD;
                *r += '$';
                lf::append_wchar(r, c);
                *r += HI_UNBOLD;
                t_ = REGEX;
            } else if (isalpha(c)) {
                *r += '#';
                *r += HI_BOLD;
                *r += '$';
                lf::append_wchar(r, c);
                t_ = REGEX_HASH_DOLLAR_WORD;
            } else {
                *r += '#';
                *r += '$';
                t_ = REGEX;
                goto Regex;
            }
            break;

        case REGEX_HASH_DOLLAR_WORD:
            if (isident(c)) {
                lf::append_wchar(r, c);
            } else {
                *r += HI_UNBOLD;
                t_ = REGEX;
                goto Regex;
            }
            break;

        case REGEX_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = REGEX;
            break;

        case MULTICOM:
            lf::append_wchar(r, c);
            if (c == '\n') {
                t_ = MULTICOM_BOL;
                i_ = 0;
            }
            break;

        case MULTICOM_BOL:
            lf::append_wchar(r, c);
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
                lf::append_wchar(r, c);
                t_ = COLON_WORD;
            } else {
                *r += ':';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case COLON_WORD:
            if (isident(c)) {
                lf::append_wchar(r, c);
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
                lf::append_wchar(r, c);
                t_ = AT_WORD;
            } else {
                *r += '@';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case AT_WORD:
            if (isident(c)) {
                lf::append_wchar(r, c);
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
                lf::append_wchar(r, c);
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
                lf::append_wchar(r, c);
                t_ = PERCENT_STRING;
            } else {
                *r += '%';
                *r += q_;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case PERCENT_STRING:
            lf::append_wchar(r, c);
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
            if (is_dollar_one(c)) {
                *r += HI_VAR;
                *r += '$';
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (isalpha(c)) {
                *r += HI_VAR;
                *r += '$';
                lf::append_wchar(r, c);
                t_ = DOLLAR_WORD;
            } else {
                *r += '$';
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case DOLLAR_WORD:
            if (isident(c)) {
                lf::append_wchar(r, c);
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
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
            } else if (c == '\\') {
                t_ = QUOTE_BACKSLASH;
            }
            break;

        case QUOTE_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = QUOTE;
            break;

        Dquote:
        case DQUOTE:
            if (c == '"') {
                *r += '"';
                *r += HI_RESET;
                t_ = NORMAL;
            } else if (c == '#') {
                t_ = DQUOTE_HASH;
            } else if (c == '\\') {
                t_ = DQUOTE_BACKSLASH;
                *r += '\\';
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case DQUOTE_HASH:
            if (c == '{' && nesti_ < sizeof(nest_)) {
                *r += HI_BOLD;
                *r += '#';
                *r += HI_UNBOLD;
                *r += HI_STRING;
                *r += '{';
                *r += HI_RESET;
                expect_ = EXPECT_VALUE;
                nest_[nesti_++] = DQUOTE;
                t_ = NORMAL;
            } else if (c == '$') {
                t_ = DQUOTE_HASH_DOLLAR;
            } else {
                *r += '#';
                t_ = DQUOTE;
                goto Dquote;
            }
            break;

        case DQUOTE_BACKSLASH:
            lf::append_wchar(r, c);
            t_ = DQUOTE;
            break;

        case DQUOTE_HASH_DOLLAR:
            if (is_dollar_one(c)) {
                *r += '#';
                *r += HI_BOLD;
                *r += '$';
                lf::append_wchar(r, c);
                *r += HI_UNBOLD;
                t_ = DQUOTE;
            } else if (isalpha(c)) {
                *r += '#';
                *r += HI_BOLD;
                *r += '$';
                lf::append_wchar(r, c);
                t_ = DQUOTE_HASH_DOLLAR_WORD;
            } else {
                *r += '#';
                *r += '$';
                t_ = DQUOTE;
                goto Dquote;
            }
            break;

        case DQUOTE_HASH_DOLLAR_WORD:
            if (isident(c)) {
                lf::append_wchar(r, c);
            } else {
                *r += HI_UNBOLD;
                t_ = DQUOTE;
                goto Dquote;
            }
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
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT:
            if (c == '-') {
                indented_heredoc_ = true;
                lf::append_wchar(r, c);
            } else if (c == '\'' || c == '`' || c == '"') {
                closer_ = c;
                t_ = LT_LT_QNAME;
                *r += HI_STRING;
                lf::append_wchar(r, c);
            } else if (isalpha(c) || c == '_') {
                t_ = LT_LT_NAME;
                heredoc_ += c;
                lf::append_wchar(r, c);
            } else {
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case LT_LT_NAME:
            if (isalnum(c) || c == '_') {
                t_ = LT_LT_NAME;
                heredoc_ += c;
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

        case HEREDOC:
            lf::append_wchar(r, c);
            if (c == '\n')
                t_ = HEREDOC_BOL;
            break;

        case QUESTION:
            if (c == '\\') {
                t_ = QUESTION_BACKSLASH;
            } else if (isspace(c)) {
                *r += '?';
                t_ = NORMAL;
                goto Normal;
            } else {
                *r += HI_ESCAPE;
                *r += '?';
                lf::append_wchar(r, c);
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case QUESTION_BACKSLASH:
            *r += HI_ESCAPE;
            *r += "?\\";
            lf::append_wchar(r, c);
            *r += HI_RESET;
            t_ = NORMAL;
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
    case QUESTION:
        *r += '?';
        break;
    case QUESTION_BACKSLASH:
        *r += "?\\";
        break;
    case DQUOTE_HASH:
        *r += '#';
        *r += HI_RESET;
        break;
    case DQUOTE_HASH_DOLLAR:
        *r += "#$";
        *r += HI_RESET;
        break;
    case REGEX_HASH:
        *r += '#';
        *r += HI_RESET;
        break;
    case REGEX_HASH_DOLLAR:
        *r += "#$";
        *r += HI_RESET;
        break;
    case REGEX:
    case REGEX_BACKSLASH:
    case REGEX_HASH_DOLLAR_WORD:
    case PERCENT_STRING:
    case AT_WORD:
    case DOLLAR_WORD:
    case TICK:
    case TICK_BACKSLASH:
    case QUOTE:
    case QUOTE_BACKSLASH:
    case DQUOTE:
    case DQUOTE_BACKSLASH:
    case DQUOTE_HASH_DOLLAR_WORD:
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
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
    is_definition_ = 0;
    expect_ = EXPECT_VALUE;
    nesti_ = 0;
}
