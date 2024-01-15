/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2022 Justine Alexandra Roberts Tunney                              │
│                                                                              │
│ Permission to use, copy, modify, and/or distribute this software for         │
│ any purpose with or without fee is hereby granted, provided that the         │
│ above copyright notice and this permission notice appear in all copies.      │
│                                                                              │
│ THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL                │
│ WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED                │
│ WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE             │
│ AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL         │
│ DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR        │
│ PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER               │
│ TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR             │
│ PERFORMANCE OF THIS SOFTWARE.                                                │
╚─────────────────────────────────────────────────────────────────────────────*/
#include "llamafile.h"
#include <stdbool.h>
#include <string.h>
#include "libc/assert.h"
//#include "libc/calls/calls.h"
#include "libc/errno.h"
//#include "libc/mem/gc.h"
//#include "libc/mem/mem.h"
#include "libc/runtime/runtime.h"
//#include "libc/str/str.h"
//#include "libc/sysv/consts/o.h"
#include "libc/x/x.h"

__static_yoink("zipos");

static struct ZipArgs {
  bool initialized;
  bool loaded;
  int oldargc;
  char *data;
  char **args;
  char **oldargv;
} g_zipargs;

// remap \n → newline
static char *Decode(char *arg) {
  int i, j;
  for (i = j = 0; arg[i]; ++i) {
    if (arg[i] == '\\' && arg[i + 1] == 'n') {
      arg[j++] = '\n';
      ++i;
    } else {
      arg[j++] = arg[i];
    }
  }
  arg[j] = 0;
  return arg;
}

static void AddZipArg(int *argc, char ***argv, char *arg) {
  *argv = xrealloc(*argv, (++(*argc) + 1) * sizeof(*(*argv)));
  (*argv)[*argc - 1] = arg;
  (*argv)[*argc - 0] = 0;
}

void FreeZipArgs(void) {
  if (g_zipargs.loaded) {
    free(g_zipargs.data);
    free(g_zipargs.args);
    __argc = g_zipargs.oldargc;
    __argv = g_zipargs.oldargv;
    g_zipargs.loaded = false;
  }
}

int LoadZipArgsImpl(int *argc, char ***argv, char *data) {
  int i, n;
  bool founddots;
  char *arg, **args, *state, *start;
  assert(!g_zipargs.loaded);
  if (chomp(data)) {
    n = 0;
    args = 0;
    start = data;
    founddots = false;
    AddZipArg(&n, &args, (*argv)[0]);
    while ((arg = strtok_r(start, "\r\n", &state))) {
      if (!strcmp(arg, "...") && !state) {
        founddots = true;
        for (i = 1; i < *argc; ++i) {
          AddZipArg(&n, &args, (*argv)[i]);
        }
      } else {
        AddZipArg(&n, &args, Decode(arg));
      }
      start = 0;
    }

    // If `...` is not found in .args... then assume there is one at the end
    if (!founddots)
    {
        founddots = true;
        for (i = 1; i < *argc; ++i) {
          AddZipArg(&n, &args, (*argv)[i]);
        }
    }

    if (founddots || *argc <= 1) {
      if (!g_zipargs.initialized) {
        atexit(FreeZipArgs);
        g_zipargs.oldargc = __argc;
        g_zipargs.oldargv = __argv;
        g_zipargs.initialized = true;
      }
      g_zipargs.loaded = true;
      g_zipargs.data = data;
      g_zipargs.args = args;
      *argc = n;
      *argv = args;
      __argc = n;
      __argv = args;
    } else {
      free(data);
      free(args);
    }
  }
  return 0;
}

/**
 * Replaces argument list with `/zip/.args` contents if it exists.
 *
 * Your `.args` file should have one argument per line.
 *
 * If the special argument `...` is *not* encountered, then it would be assumed
 * that the developer intent is for whatever CLI args were specified by the user
 * to be appended to the end
 *
 * If the special argument `...` *is* encountered, then it'll be
 * replaced with whatever CLI args were specified by the user.
 *
 * @return 0 on success, or -1 if not found w/o errno clobber
 */
int llamafile_LoadZipArgs(int *argc, char ***argv) {
  int e;
  char *p;
  e = errno;
  if ((p = xslurp("/zip/.args", 0))) {
    return LoadZipArgsImpl(argc, argv, p);
  } else {
    errno = e;
    return -1;
  }
}
