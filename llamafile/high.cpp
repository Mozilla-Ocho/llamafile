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

#include "llamafile/highlight/highlight.h"
#include "string.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <unistd.h>

// syntax highlighter demo

static void highlight(int infd, int outfd, const char *lang, const char *inpath) {
    // create syntax highlighter
    Highlight *h;
    const char *ext;
    if (lang) {
        if (!(h = Highlight::create(lang))) {
            fprintf(stderr, "%s: language not supported\n", lang);
            exit(1);
        }
    } else if (inpath) {
        if (!(h = Highlight::create(lf::tolower(lf::extname(inpath)))))
            h = new HighlightMarkdown;
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

int main(int argc, char *argv[]) {

    // process flags
    int opt;
    int outfd = 1;
    const char *lang = nullptr;
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

    // process files
    if (optind == argc) {
        highlight(0, outfd, lang, 0);
    } else {
        for (int i = optind; i < argc; ++i) {
            int infd;
            const char *inpath = argv[i];
            if ((infd = open(inpath, O_RDONLY)) == -1) {
                perror(inpath);
                exit(1);
            }
            highlight(infd, outfd, lang, inpath);
        }
    }
}
