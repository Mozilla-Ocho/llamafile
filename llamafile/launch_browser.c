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
#include "llamafile/log.h"
#include <cosmo.h>
#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static volatile bool g_timed_out;

static void finish(void) {
    if (!IsWindows())
        _exit(0);
}

static void handle_timeout(int sig) {
    g_timed_out = true;
}

static void report_failure(const char *url, const char *cmd, const char *reason) {
    tinylog("failed to open ", url, " in a browser tab using ", cmd, ": ", reason, "\n", NULL);
}

/**
 * Opens browser tab on host system.
 */
void llamafile_launch_browser(const char *url) {

    // perform this task from a subprocess so it doesn't block server
    tinylog("opening browser tab... (pass --nobrowser to disable)\n", NULL);
    if (!IsWindows()) {
        switch (fork()) {
        case 0:
            break;
        default:
            return;
        case -1:
            perror("fork");
            return;
        }
    }

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
    char *args[] = {(char *)cmd, (char *)url, 0};
    posix_spawnattr_init(&sa);
    posix_spawnattr_setflags(&sa, POSIX_SPAWN_SETPGROUP);
    err = posix_spawnp(&pid, cmd, 0, &sa, args, environ);
    posix_spawnattr_destroy(&sa);
    if (err) {
        report_failure(url, cmd, strerror(err));
        return finish();
    }

    // kill command if it takes more than three seconds
    // we need it because xdg-open acts weird on headless systems
    struct sigaction hand;
    hand.sa_flags = 0;
    sigemptyset(&hand.sa_mask);
    hand.sa_handler = handle_timeout;
    sigaction(SIGALRM, &hand, 0);
    alarm(3);

    // wait for tab to return finish opening
    // the browser will still be running after this completes
    int ws;
    while (waitpid(pid, &ws, 0) == -1) {
        if (errno != EINTR) {
            report_failure(url, cmd, strerror(errno));
            kill(pid, SIGKILL);
            return finish();
        }
        if (g_timed_out) {
            report_failure(url, cmd, "process timed out");
            kill(pid, SIGKILL);
            return finish();
        }
    }
    if (ws)
        report_failure(url, cmd, "process exited with non-zero status");

    // we're done
    return finish();
}
