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
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define LENGTH 10
#define ITERATIONS 200000
#define CHARSET " raQq123{}[]!@#$%^*().\"'`\\/\n-_=&;:<>,"

const char* const kLanguages[] = {
    "ada", //
    "asm", //
    "basic", //
    "bnf", //
    "c", //
    "c#", //
    "c++", //
    "cmake", //
    "cobol", //
    "css", //
    "d", //
    "forth", //
    "fortran", //
    "go", //
    "haskell", //
    "html", //
    "java", //
    "javascript", //
    "json", //
    "julia", //
    "kotlin", //
    "ld", //
    "lisp", //
    "lua", //
    "m4", //
    "make", //
    "markdown", //
    "matlab", //
    "ocaml", //
    "pascal", //
    "perl", //
    "php!", //
    "python", //
    "r", //
    "ruby", //
    "rust", //
    "scala", //
    "shell", //
    "sql", //
    "swift", //
    "tcl", //
    "tex", //
    "txt", //
    "typescript", //
    "zig", //
};

std::string
generate_random_string(int n)
{
    std::string s;
    s.reserve(n);
    for (int i = 0; i < n; ++i)
        s += CHARSET[rand() % (sizeof(CHARSET) - 1)];
    return s;
}

std::string
remove_ansi_sgr_codes(const std::string& input)
{
    std::string result;
    result.reserve(input.length());
    bool in_escape_sequence = false;
    for (char c : input) {
        if (c == '\033') {
            in_escape_sequence = true;
        } else if (in_escape_sequence) {
            if (c == 'm')
                in_escape_sequence = false;
        } else {
            result += c;
        }
    }
    return result;
}

bool
is_color_reset(const std::string& input)
{
    int t = 0;
    int number = 0;
    bool has_color = false;
    for (char c : input) {
        switch (t) {
        case 0:
            if (c == 033)
                t = 1;
            break;
        case 1:
            if (c == '[') {
                t = 2;
                number = 0;
            } else {
                fprintf(stderr, "unexpected ansi escape structure\n");
                exit(1);
            }
            break;
        case 2:
            if (isdigit(c)) {
                number *= 10;
                number += c - '0';
            } else if (c == 'm') {
                has_color = !!number;
                t = 0;
            } else if (c == ';') {
                has_color = !!number;
                number = 0;
            } else {
                fprintf(stderr, "unexpected ansi escape structure\n");
                exit(1);
            }
            break;
        default:
            __builtin_unreachable();
        }
    }
    return !has_color;
}

int
main(int argc, char* argv[])
{
    for (int l = 0; l < sizeof(kLanguages) / sizeof(*kLanguages); ++l) {
        Highlight* h = Highlight::create(kLanguages[l]);
        if (!h) {
            fprintf(stderr, "Highlight::create(%`'s) failed\n", kLanguages[l]);
            exit(1);
        }
        for (int i = 0; i < ITERATIONS; ++i) {
            std::string sauce = generate_random_string(LENGTH);
            std::string colorized;
            h->feed(&colorized, sauce);
            h->flush(&colorized);
            if (!is_color_reset(colorized)) {
                fprintf(stderr,
                        "%s highlight failed to reset color: %`'s -> %`'s\n",
                        kLanguages[l],
                        sauce.c_str(),
                        colorized.c_str());
                exit(1);
            }
            std::string plain = remove_ansi_sgr_codes(colorized);
            if (sauce != plain) {
                fprintf(stderr,
                        "%s highlight failed to preserve code: %`'s -> %`'s -> "
                        "%`'s\n",
                        kLanguages[l],
                        sauce.c_str(),
                        colorized.c_str(),
                        plain.c_str());
                exit(1);
            }
        }
        delete h;
    }
}
