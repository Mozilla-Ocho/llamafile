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

#include <cosmo.h>
#include <stdlib.h>
#include <string.h>

/**
 * Detect if terminal supports 24-bit color.
 */
bool is_rgb_terminal(void) {

    if (IsWindows())
        return true;

    const char *colorterm = getenv("COLORTERM");
    if (colorterm && (!strcasecmp(colorterm, "truecolor") || //
                      !strcasecmp(colorterm, "24bit"))) {
        return true;
    }

    const char *term = getenv("TERM");
    if (term && (strstr(term, "24bit") || //
                 strstr(term, "truecolor") || //
                 strstr(term, "iterm") || //
                 strstr(term, "kitty") || //
                 strstr(term, "wezterm"))) {
        return true;
    }

    return false;
}
