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
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <unistd.h>

// syntax highlighter demo

int main(int argc, char *argv[]) {

    // process flags
    int opt;
    int infd = 0;
    int outfd = 1;
    const char *lang = nullptr;
    const char *inpath = nullptr;
    while ((opt = getopt(argc, argv, "hl:o:")) != -1) {
        switch (opt) {
        case 'h':
            printf("usage: %s [-l LANG] [-o OUTFILE] [INFILE]\n", argv[0]);
            exit(0);
        case 'l':
            lang = optarg;
            break;
        case 'o':
            if ((outfd = creat(optarg, 0644)) == -1) {
                perror(optarg);
                exit(1);
            }
            break;
        default:
            exit(1);
        }
    }
    if (optind < argc) {
        inpath = argv[optind];
        if ((infd = open(inpath, O_RDONLY)) == -1) {
            perror(inpath);
            exit(1);
        }
    }

    // create syntax highlighter
    Highlight *h;
    const char *ext;
    if (lang) {
        h = Highlight::create(lang);
    } else if (inpath && (ext = strrchr(inpath, '.'))) {
        h = Highlight::create(ext + 1);
    } else {
        h = Highlight::create("markdown");
    }

    // process input
    std::string res;
    ColorBleeder H(h);
    for (;;) {

        // read input chunk
        char buf[256];
        ssize_t rc = read(infd, buf, sizeof(buf));
        if (rc == -1) {
            perror("read");
            exit(1);
        }
        size_t got = rc;
        if (!got)
            break;

        // highlight chunk
        res.clear();
        H.feed(&res, std::string_view(buf, got));

        // write highlighted output chunk
        write(outfd, res.data(), res.size());
    }

    // flush highlighter
    res.clear();
    H.flush(&res);
    write(outfd, res.data(), res.size());
}
