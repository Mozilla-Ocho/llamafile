// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2023 Mozilla Foundation
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

#include "llamafile.h"
#include "version.h"
#include <stdio.h>
#include <stdlib.h>

static const char *llamafile_get_home_dir(void) {
    const char *homedir;
    if (!(homedir = getenv("HOME")) || !*homedir)
        homedir = ".";
    return homedir;
}

/**
 * Returns path of directory for app-specific files.
 */
void llamafile_get_app_dir(char *path, size_t size) {
    snprintf(path, size, "%s/.llamafile/v/%d.%d.%d/", llamafile_get_home_dir(), LLAMAFILE_MAJOR,
             LLAMAFILE_MINOR, LLAMAFILE_PATCH);
}
