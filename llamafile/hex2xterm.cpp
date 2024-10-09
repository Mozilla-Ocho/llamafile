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

#include "xterm.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "%s: missing operand\n", argv[0]);
        exit(1);
    }

    // process RRGGBB hex arguments
    for (int i = 1; i < argc; ++i) {
        int rgb = strtol(argv[i], 0, 16);
        int xterm = rgb2xterm256(rgb);
        printf("\n");
        printf("xterm code %d\n", xterm);
        printf("html5 code #%06x\n", rgb);
        printf("foreground \033[38;5;%dm\\033[38;5;%dm\033[0m\n", xterm, xterm);
        printf("background \033[48;5;%dm\\033[48;5;%dm\033[0m\n", xterm, xterm);
    }
}
