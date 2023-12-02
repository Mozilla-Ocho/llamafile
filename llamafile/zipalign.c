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

#include <time.h>
#include <fcntl.h>
#include <cosmo.h>
#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <libgen.h>
#include <getopt.h>
#include <stdbool.h>
#include <sys/uio.h>
#include <third_party/zlib/zlib.h>
#include "zip.h"

#define USAGE \
  " ZIP FILE...\n\
\n\
DESCRIPTION\n\
\n\
  Adds aligned uncompressed files to PKZIP archive\n\
\n\
  This tool is designed to concatenate gigabytes of LLM weights to an\n\
  executable. This command goes 10x faster than `zip -j0`. Unlike zip\n\
  you are not required to use the .com file extension for it to work.\n\
  But most importantly, this tool has a flag that lets you insert zip\n\
  files that are aligned on a specific boundary. The result is things\n\
  like GPUs that have specific memory alignment requirements will now\n\
  be able to perform math directly on the zip file's mmap()'d weights\n\
\n\
FLAGS\n\
\n\
  -h        help\n\
  -N        nondeterministic mode\n\
  -a INT    alignment (default 65536)\n\
  -j        strip directory components\n\
  -0        store uncompressed (currently default)\n\
\n"

#define Min(a, b) ((a) < (b) ? (a) : (b))
#define DOS_DATE(YEAR, MONTH_IDX1, DAY_IDX1) \
  (((YEAR)-1980) << 9 | (MONTH_IDX1) << 5 | (DAY_IDX1))
#define DOS_TIME(HOUR, MINUTE, SECOND) \
  ((HOUR) << 11 | (MINUTE) << 5 | (SECOND) >> 1)

static const char *prog;
static int FLAG_junk;
static int FLAG_alignment = 65536;
static bool FLAG_nondeterministic;

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
    if (!(p = malloc(n))) DieOom();
    return p;
}

static char *StrDup(const char *s) {
    char *p;
    if (!(p = strdup(s))) DieOom();
    return p;
}

static void *Realloc(void *p, size_t n) {
    if (!(p = realloc(p, n))) DieOom();
    return p;
}

static wontreturn void PrintUsage(int fd, int rc) {
    tinyprint(fd, "SYNOPSIS\n\n  ", prog, USAGE, NULL);
    exit(rc);
}

static void GetDosLocalTime(int64_t utcunixts,
                            uint16_t *out_time,
                            uint16_t *out_date) {
    struct tm tm;
    gmtime_r(&utcunixts, &tm);
    *out_time = DOS_TIME(tm.tm_hour, tm.tm_min, tm.tm_sec);
    *out_date = DOS_DATE(tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday + 1);
}

int main(int argc, char *argv[]) {

    // get name of program
    prog = argv[0];
    if (!prog) prog = "zipalign";

    // parse flags
    int opt;
    while ((opt = getopt(argc, argv, "hj0Na:")) != -1) {
        switch (opt) {
            case '0':
                break;
            case 'j':
                FLAG_junk = true;
                break;
            case 'N':
                FLAG_nondeterministic = true;
                break;
            case 'a':
                FLAG_alignment = atoi(optarg);
                if (FLAG_alignment < 1) {
                    Die(prog, "FLAG_alignment must be at least 1");
                }
                if (FLAG_alignment & (FLAG_alignment - 1)) {
                    Die(prog, "FLAG_alignment must be two power");
                }
                break;
            case 'h':
                PrintUsage(1, 0);
            default:
                PrintUsage(2, 1);
        }
    }
    if (optind == argc) {
        Die(prog, "missing output argument");
    }

    // open output file
    int zfd;
    ssize_t zsize;
    const char *zpath = argv[optind++];
    if ((zfd = open(zpath, O_CREAT | O_RDWR, 0644)) == -1) {
        DieSys(zpath);
    }
    if ((zsize = lseek(zfd, 0, SEEK_END)) == -1) {
        DieSys(zpath);
    }

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
    if (pread(zfd, last64, amt, off) != amt) {
        DieSys(zpath);
    }

    // search backwards for the end-of-central-directory record
    // the eocd (cdir) says where the central directory (cfile) array is located
    // we consistency check some legacy fields, to be extra sure that it is eocd
    unsigned cnt = 0;
    for (int i = amt - Min(kZipCdirHdrMinSize, kZipCdir64LocatorSize); i >= 0; --i) {
        uint32_t magic = ZIP_READ32(last64 + i);
        if (magic == kZipCdir64LocatorMagic && i + kZipCdir64LocatorSize <= amt &&
            pread(zfd, last64, kZipCdir64HdrMinSize,
                  ZIP_LOCATE64_OFFSET(last64 + i)) == (long)kZipCdir64HdrMinSize &&
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
    if (!cnt) {
        amt = 0;
    }

    // read central directory
    uint8_t *cdir = Malloc(amt);
    if (pread(zfd, cdir, amt, off) != amt) {
        DieSys(zpath);
    }

    // get time
    struct timespec now;
    uint16_t mtime, mdate;
    if (FLAG_nondeterministic) {
        now = timespec_real();
    } else {
        now = timespec_fromseconds(1700000000);
    }
    GetDosLocalTime(now.tv_sec, &mtime, &mdate);

    // add inputs
    for (int i = optind; i < argc; ++i) {

        // open input file
        int fd;
        ssize_t size;
        const char *path = argv[i];
        if ((fd = open(path, O_RDONLY)) == -1) {
            DieSys(path);
        }
        if ((size = lseek(fd, 0, SEEK_END)) == -1) {
            DieSys(path);
        }

        // construct zip entry name
        char *name = StrDup(path);
        if (FLAG_junk) {
            name = basename(name);
        } else {
            while (*name == '/') {
                ++name;
            }
        }

        // determine size and alignment of local file header
        size_t namlen = strlen(name);
        size_t extlen = (2 + 2 + 8 + 8);
        size_t hdrlen = kZipLfileHdrMinSize + namlen + extlen;
        while ((zsize + hdrlen) & (FLAG_alignment - 1)) ++zsize;

        // copy file
        ssize_t rc;
        uint32_t crc = 0;
        _Alignas(4096) static uint8_t iobuf[2097152];
        for (off_t i = 0; i < size; i += rc) {
            if ((rc = pread(fd, iobuf, Min(size, sizeof(iobuf)), i)) <= 0) {
                DieSys(path);
            }
            crc = crc32(crc, iobuf, rc);
            if (pwrite(zfd, iobuf, rc, zsize + hdrlen + i) != rc) {
                DieSys(zpath);
            }
        }

        // write local file header
        uint8_t *lochdr = Malloc(hdrlen);
        uint8_t *p = lochdr;

        p = ZIP_WRITE32(p, kZipLfileHdrMagic);
        p = ZIP_WRITE16(p, kZipEra2001);
        p = ZIP_WRITE16(p, kZipGflagUtf8);
        p = ZIP_WRITE16(p, kZipCompressionNone);
        p = ZIP_WRITE16(p, mtime);
        p = ZIP_WRITE16(p, mdate);
        p = ZIP_WRITE32(p, crc);
        p = ZIP_WRITE32(p, 0xffffffffu);  // compressed size
        p = ZIP_WRITE32(p, 0xffffffffu);  // uncompressed size
        p = ZIP_WRITE16(p, namlen);
        p = ZIP_WRITE16(p, extlen);
        p = mempcpy(p, name, namlen);

        p = ZIP_WRITE16(p, kZipExtraZip64);
        p = ZIP_WRITE16(p, 8 + 8);
        p = ZIP_WRITE64(p, size);  // uncompressed size
        p = ZIP_WRITE64(p, size);  // compressed size

        unassert(p == lochdr + hdrlen);
        if (pwrite(zfd, lochdr, hdrlen, zsize) != hdrlen) {
            DieSys(zpath);
        }
        free(lochdr);

        // create central directory entry
        extlen = (2 + 2 + 8 + 8 + 8);
        hdrlen = kZipCfileHdrMinSize + namlen + extlen;
        cdir = Realloc(cdir, amt + hdrlen);
        uint8_t *cdirhdr = cdir + amt;
        amt += hdrlen;
        p = cdirhdr;

        p = ZIP_WRITE32(p, kZipCfileHdrMagic);
        p = ZIP_WRITE16(p, kZipOsUnix << 8 | kZipEra2001);  // version made by
        p = ZIP_WRITE16(p, kZipEra2001);  // version needed to extract
        p = ZIP_WRITE16(p, kZipGflagUtf8);
        p = ZIP_WRITE16(p, kZipCompressionNone);
        p = ZIP_WRITE16(p, mtime);
        p = ZIP_WRITE16(p, mdate);
        p = ZIP_WRITE32(p, crc);
        p = ZIP_WRITE32(p, 0xffffffffu);  // compressed size
        p = ZIP_WRITE32(p, 0xffffffffu);  // uncompressed size
        p = ZIP_WRITE16(p, namlen);
        p = ZIP_WRITE16(p, extlen);
        p = ZIP_WRITE16(p, 0);  // comment length
        p = ZIP_WRITE16(p, 0);  // disk number start
        p = ZIP_WRITE16(p, kZipIattrBinary);
        p = ZIP_WRITE32(p, 0100644u << 16);  // external file attributes
        p = ZIP_WRITE32(p, 0xffffffffu); // lfile offset
        p = mempcpy(p, name, namlen);

        p = ZIP_WRITE16(p, kZipExtraZip64);
        p = ZIP_WRITE16(p, 8 + 8 + 8);
        p = ZIP_WRITE64(p, size);   // uncompressed size
        p = ZIP_WRITE64(p, size);   // compressed size
        p = ZIP_WRITE64(p, zsize);  // lfile offset
        unassert(p == cdirhdr + hdrlen);

        // finish up
        ++cnt;
        zsize += hdrlen + size;
        if (close(fd)) {
            DieSys(path);
        }
    }

    // write out central directory
    if (pwrite(zfd, cdir, amt, zsize) != amt) {
        DieSys(zpath);
    }
    free(cdir);

    // write out end of central directory
    uint8_t eocd[kZipCdirHdrMinSize + kZipCdir64HdrMinSize + kZipCdir64LocatorSize];
    uint8_t *p = eocd;
    p = ZIP_WRITE32(p, kZipCdir64HdrMagic);
    p = ZIP_WRITE64(p, kZipCdir64HdrMinSize - 12);  // size of eocd64
    p = ZIP_WRITE16(p, kZipOsUnix << 8 | kZipEra2001);  // version made by
    p = ZIP_WRITE16(p, kZipEra2001);  // version needed to extract
    p = ZIP_WRITE32(p, 0);  // number of this disk
    p = ZIP_WRITE32(p, 0);  // number of disk with start of central directory
    p = ZIP_WRITE64(p, cnt);  // number of records on disk
    p = ZIP_WRITE64(p, cnt);  // number of records
    p = ZIP_WRITE64(p, amt);  // size of central directory
    p = ZIP_WRITE64(p, zsize);  // offset of start of central directory
    p = ZIP_WRITE32(p, kZipCdir64LocatorMagic);
    p = ZIP_WRITE32(p, 0);  // number of disk with eocd64
    p = ZIP_WRITE64(p, zsize + amt);  // offset of eocd64
    p = ZIP_WRITE32(p, 1);  // total number of disks
    p = ZIP_WRITE32(p, kZipCdirHdrMagic);
    p = ZIP_WRITE16(p, 0);  // number of this disk
    p = ZIP_WRITE16(p, 0);  // number of disks
    p = ZIP_WRITE16(p, cnt);  // number of records on disk
    p = ZIP_WRITE16(p, cnt);  // number of records
    p = ZIP_WRITE32(p, amt);  // size of central directory
    p = ZIP_WRITE32(p, 0xffffffffu);  // offset of central directory
    p = ZIP_WRITE16(p, 0);  // comment length
    unassert(p == eocd + sizeof(eocd));
    if (pwrite(zfd, eocd, sizeof(eocd), zsize + amt) != sizeof(eocd)) {
        DieSys(zpath);
    }

    // close output
    if (close(zfd)) {
        DieSys(zpath);
    }
}
