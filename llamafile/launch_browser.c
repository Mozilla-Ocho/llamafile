// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set net ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
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
#include <cosmo.h>
#include <spawn.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>

static void report_failure(const char *url,
                           const char *cmd,
                           const char *reason) {
    tinyprint(2, "failed to open ", url, " in a browser tab using ", cmd,
              ": ", reason, "\n", NULL);
}

/**
 * Opens browser tab on host system.
 */
bool llamafile_launch_browser(const char *url) {

    // determine which command opens browser tab
    const char *cmd;
    if (IsWindows()) {
        cmd = "/c/windows/explorer.exe";
    } else if (IsXnu()) {
        cmd = "open";
    } else {
        cmd = "xdg-open";
    }

    // spawn process
    // set process group so ctrl-c won't kill browser
    int pid, err;
    posix_spawnattr_t sa;
    char *args[] = {(char *)cmd, (char *)url, NULL};
    posix_spawnattr_init(&sa);
    posix_spawnattr_setflags(&sa, POSIX_SPAWN_SETPGROUP);
    err = posix_spawnp(&pid, cmd, NULL, &sa, args, environ);
    posix_spawnattr_destroy(&sa);
    if (err) {
        report_failure(url, cmd, strerror(err));
        return false;
    }

    // wait for tab to finish opening
    // the browser will still be running after this completes
    int ws;
    while (waitpid(pid, &ws, 0) == -1) {
        if (errno != EINTR) {
            report_failure(url, cmd, strerror(errno));
            return false;
        }
    }
    if (ws) {
        report_failure(url, cmd, "process exited with non-zero status");
        return false;
    }

    // report success
    return true;
}
