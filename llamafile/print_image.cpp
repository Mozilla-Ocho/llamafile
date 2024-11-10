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

#include <string>
#include <termios.h>
#include <unistd.h>

#include "llamafile/macros.h"
#include "stb/stb_image.h"
#include "stb/stb_image_resize.h"

/**
 * Prints image to terminal.
 */
int print_image(int fd, const char *path, int max_width) {

    // load image
    int width, height, channels;
    unsigned char *img = stbi_load(path, &width, &height, &channels, 3);
    if (!img)
        return -1;

    // get terminal info
    bool use_rgb = is_rgb_terminal();
    struct winsize ws = {24, 80};
    tcgetwinsize(fd, &ws);

    // calculate new dimensions preserving aspect ratio
    int xn = MIN(max_width, ws.ws_col);
    int yn = (height * xn) / width; // *2 because we use half blocks
    yn = (yn + 1) & -2; // round up to even number

    // resize image
    unsigned char *resized = new unsigned char[xn * yn * 3];
    stbir_resize_uint8(img, width, height, 0, resized, xn, yn, 0, 3);
    stbi_image_free(img);

    // convert image to string using half blocks
    std::string s;
    for (int y = 0; y < yn; y += 2) {
        for (int x = 0; x < xn; ++x) {
            int upr = resized[((y + 0) * xn + x) * 3 + 0];
            int upg = resized[((y + 0) * xn + x) * 3 + 1];
            int upb = resized[((y + 0) * xn + x) * 3 + 2];
            int lor = resized[((y + 1) * xn + x) * 3 + 0];
            int log = resized[((y + 1) * xn + x) * 3 + 1];
            int lob = resized[((y + 1) * xn + x) * 3 + 2];
            char buf[48];
            if (use_rgb) {
                s.append(buf, snprintf(buf, sizeof(buf), "\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm▀",
                                       upr, upg, upb, lor, log, lob));
            } else {
                int upx = rgb2xterm256((upr << 16) | (upg << 8) | upb);
                int lox = rgb2xterm256((lor << 16) | (log << 8) | lob);
                s.append(buf, snprintf(buf, sizeof(buf), "\033[38;5;%dm\033[48;5;%dm▀", upx, lox));
            }
        }
        s += "\033[0m\n";
    }

    // write image to terminal
    int rc = write(fd, s.data(), s.size());
    delete[] resized;
    return rc;
}
