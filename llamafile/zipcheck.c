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

#include "zip.h"
#include <assert.h>
#include <cosmo.h>
#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/uio.h>
#include <third_party/zlib/zlib.h>
#include <time.h>

#define USAGE \
    " [FLAGS] FILE...\n\
\n\
DESCRIPTION\n\
\n\
  Tool for checking alignment of files in PKZIP archive.\n\
\n\
FLAGS\n\
\n\
  -h        help\n\
\n"

#define Min(a, b) ((a) < (b) ? (a) : (b))

static const char *prog;

static wontreturn void Die(const char *thing, const char *reason) {
    tinyprint(2, thing, ": ", reason, "\n", NULL);
    exit(1);
}

static wontreturn void DieSys(const char *thing) {
    perror(thing);
    exit(1);
}

static wontreturn void DieOom(void) {
    Die("apelink", "out of memory");
}

static void *Malloc(size_t n) {
    void *p;
    if (!(p = malloc(n)))
        DieOom();
    return p;
}

static char *StrDup(const char *s) {
    char *p;
    if (!(p = strdup(s)))
        DieOom();
    return p;
}

static void *Realloc(void *p, size_t n) {
    if (!(p = realloc(p, n)))
        DieOom();
    return p;
}

static wontreturn void PrintUsage(int fd, int rc) {
    tinyprint(fd, "SYNOPSIS\n\n  ", prog, USAGE, NULL);
    exit(rc);
}

static void CheckZip(const char *zpath) {

    // open zip archive
    int zfd;
    ssize_t zsize;
    if ((zfd = open(zpath, O_RDONLY)) == -1)
        DieSys(zpath);
    if ((zsize = lseek(zfd, 0, SEEK_END)) == -1)
        DieSys(zpath);

    // read last 64kb of file
    int amt;
    off_t off;
    if (zsize <= 65536) {
        off = 0;
        amt = zsize;
    } else {
        off = zsize - 65536;
        amt = zsize - off;
    }
    static char last64[65536];
    if (pread(zfd, last64, amt, off) != amt)
        DieSys(zpath);

    // search backwards for the
    // end-of-central-directory record the eocd
    // (cdir) says where the central directory
    // (cfile) array is located we consistency
    // check some legacy fields, to be extra sure
    // that it is eocd
    unsigned cnt = 0;
    for (int i = amt - Min(kZipCdirHdrMinSize, kZipCdir64LocatorSize); i >= 0; --i) {
        uint32_t magic = ZIP_READ32(last64 + i);
        if (magic == kZipCdir64LocatorMagic && i + kZipCdir64LocatorSize <= amt &&
            pread(zfd, last64, kZipCdir64HdrMinSize, ZIP_LOCATE64_OFFSET(last64 + i)) ==
                (long)kZipCdir64HdrMinSize &&
            ZIP_READ32(last64) == kZipCdir64HdrMagic &&
            ZIP_CDIR64_RECORDS(last64) == ZIP_CDIR64_RECORDSONDISK(last64) &&
            ZIP_CDIR64_RECORDS(last64) && ZIP_CDIR64_SIZE(last64) <= INT_MAX) {
            cnt = ZIP_CDIR64_RECORDS(last64);
            off = ZIP_CDIR64_OFFSET(last64);
            amt = ZIP_CDIR64_SIZE(last64);
            break;
        }
        if (magic == kZipCdirHdrMagic && i + kZipCdirHdrMinSize <= amt &&
            ZIP_CDIR_RECORDS(last64 + i) == ZIP_CDIR_RECORDSONDISK(last64 + i) &&
            ZIP_CDIR_RECORDS(last64 + i) && ZIP_CDIR_SIZE(last64 + i) <= INT_MAX &&
            ZIP_CDIR_OFFSET(last64 + i) != 0xffffffffu) {
            cnt = ZIP_CDIR_RECORDS(last64 + i);
            off = ZIP_CDIR_OFFSET(last64 + i);
            amt = ZIP_CDIR_SIZE(last64 + i);
            break;
        }
    }
    if (!cnt)
        amt = 0;

    // read central directory
    uint8_t *cdir = Malloc(amt);
    if (pread(zfd, cdir, amt, off) != amt)
        DieSys(zpath);
    if (ZIP_READ32(cdir) != kZipCfileHdrMagic)
        Die(zpath, "unable to locate central directory");
    unsigned entry_index, entry_offset;
    for (entry_index = entry_offset = 0;
         entry_index < cnt && entry_offset + kZipCfileHdrMinSize <= amt &&
         entry_offset + ZIP_CFILE_HDRSIZE(cdir + entry_offset) <= amt;
         ++entry_index, entry_offset += ZIP_CFILE_HDRSIZE(cdir + entry_offset)) {
        if (ZIP_CFILE_MAGIC(cdir + entry_offset) != kZipCfileHdrMagic)
            Die(zpath, "corrupted zip central "
                       "directory entry magic");

        // ignore entries with no content (e.g.
        // dirs)
        if (!get_zip_cfile_compressed_size(cdir + entry_offset))
            continue;

        // read local file header
        off_t off = get_zip_cfile_offset(cdir + entry_offset);
        uint8_t lfile[kZipLfileHdrMinSize];
        if (pread(zfd, lfile, kZipLfileHdrMinSize, off) != kZipLfileHdrMinSize)
            Die(zpath, "failed to pread lfile");
        if (ZIP_LFILE_MAGIC(lfile) != kZipLfileHdrMagic)
            Die(zpath, "corrupted zip local file magic");

        // get offset of local file content
        off += ZIP_LFILE_HDRSIZE(lfile);

        // get alignment of local file
        long align = 1ull << _bsfl(off);
        printf("%.*s has alignment of %ld\n", (int)ZIP_CFILE_NAMESIZE(cdir + entry_offset),
               ZIP_CFILE_NAME(cdir + entry_offset), align);
    }

    // close input
    if (close(zfd))
        DieSys(zpath);
}

int main(int argc, char *argv[]) {

    // get name of program
    prog = argv[0];
    if (!prog)
        prog = "zipcheck";

    // parse flags
    int opt;
    while ((opt = getopt(argc, argv, "h")) != -1) {
        switch (opt) {
        case 'h':
            PrintUsage(1, 0);
        default:
            PrintUsage(2, 1);
        }
    }
    if (optind == argc)
        Die(prog, "missing input");

    for (int i = optind; i < argc; ++i)
        CheckZip(argv[i]);
}
