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
#include "zip.h"

#include <assert.h>
#include <cosmo.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <third_party/getopt/getopt.internal.h>
#include <third_party/zlib/zlib.h>
#include <time.h>

#define TINYMALLOC_MAX_BYTES (64 * 1024 * 1024)
#include <libc/mem/tinymalloc.inc>

#define CHUNK 2097152

#define Min(a, b) ((a) < (b) ? (a) : (b))
#define DOS_DATE(YEAR, MONTH_IDX1, DAY_IDX1) (((YEAR) - 1980) << 9 | (MONTH_IDX1) << 5 | (DAY_IDX1))
#define DOS_TIME(HOUR, MINUTE, SECOND) ((HOUR) << 11 | (MINUTE) << 5 | (SECOND) >> 1)

static const char *prog;
static int flag_junk;
static int flag_level;
static int flag_verbose;
static int flag_alignment = 65536;
static bool flag_nondeterministic;

static wontreturn void Die(const char *thing, const char *reason) {
    tinyprint(2, thing, ": fatal error: ", reason, "\n", NULL);
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

static void GetDosLocalTime(int64_t utcunixts, uint16_t *out_time, uint16_t *out_date) {
    struct tm tm;
    localtime_r(&utcunixts, &tm);
    *out_time = DOS_TIME(tm.tm_hour, tm.tm_min, tm.tm_sec);
    *out_date = DOS_DATE(tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
}

static int NormalizeMode(int mode) {
    int res = mode & S_IFMT;
    if (mode & 0111)
        res |= 0111;
    return res | 0644;
}

int main(int argc, char *argv[]) {

    if (llamafile_has(argv, "-h") || llamafile_has(argv, "-help") ||
        llamafile_has(argv, "--help")) {
        llamafile_help("/zip/llamafile/zipalign.1.asc");
        __builtin_unreachable();
    }

    // get name of program
    prog = argv[0];
    if (!prog)
        prog = "zipalign";

    // parse flags
    int opt;
    while ((opt = getopt(argc, argv, "0123456789vjNa:")) != -1) {
        switch (opt) {
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            flag_level = opt - '0';
            break;
        case 'v':
            ++flag_verbose;
            break;
        case 'j':
            flag_junk = true;
            break;
        case 'N':
            flag_nondeterministic = true;
            break;
        case 'a':
            flag_alignment = atoi(optarg);
            if (flag_alignment < 1)
                Die(prog, "flag_alignment must be at least 1");
            if (flag_alignment & (flag_alignment - 1))
                Die(prog, "flag_alignment must be two power");
            break;
        default:
            return 1;
        }
    }
    if (optind == argc)
        Die(prog, "missing output argument");

    // use idle scheduling priority
    verynice();

    // open output file
    int zfd;
    ssize_t zsize;
    const char *zpath = argv[optind++];
    if ((zfd = open(zpath, O_CREAT | O_RDWR, 0644)) == -1)
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

    // search backwards for the end-of-central-directory record
    // the eocd (cdir) says where the central directory (cfile) array is located
    // we consistency check some legacy fields, to be extra sure that it is eocd
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
    size_t cdirsize = amt;
    uint8_t *cdir = Malloc(cdirsize);
    if (pread(zfd, cdir, cdirsize, off) != cdirsize)
        DieSys(zpath);

    // create array of zip entry names
    char **names = Malloc(sizeof(char *) * argc);
    for (int i = optind; i < argc; ++i) {
        names[i] = StrDup(argv[i]);
        if (flag_junk)
            names[i] = basename(names[i]);
        else
            while (*names[i] == '/')
                ++names[i];
    }

    // verify there's no duplicate zip asset names
    for (int i = optind; i < argc; ++i)
        for (int j = i + 1; j < argc; ++j)
            if (!strcmp(names[i], names[j]))
                Die(names[i], "zip asset name specified multiple times");

    // delete central directory entries about to be replaced
    int new_count = 0;
    off_t new_index = 0;
    unsigned entry_index, entry_offset;
    for (entry_index = entry_offset = 0;
         entry_index < cnt && entry_offset + kZipCfileHdrMinSize <= cdirsize &&
         entry_offset + ZIP_CFILE_HDRSIZE(cdir + entry_offset) <= cdirsize;
         ++entry_index, entry_offset += ZIP_CFILE_HDRSIZE(cdir + entry_offset)) {
        if (ZIP_CFILE_MAGIC(cdir + entry_offset) != kZipCfileHdrMagic)
            Die(zpath, "corrupted zip central directory entry magic");

        // check if entry name matches any of the new names
        bool found = false;
        for (int i = optind; i < argc; ++i)
            if (ZIP_CFILE_NAMESIZE(cdir + entry_offset) == strlen(names[i]) &&
                !memcmp(ZIP_CFILE_NAME(cdir + entry_offset), names[i],
                        ZIP_CFILE_NAMESIZE(cdir + entry_offset))) {
                found = true;
                break;
            }

        // copy back central directory entry
        if (!found) {
            memmove(cdir + new_index, cdir + entry_offset, ZIP_CFILE_HDRSIZE(cdir + entry_offset));
            new_index += ZIP_CFILE_HDRSIZE(cdir + new_index);
            ++new_count;
        }
    }
    cdirsize = new_index;
    cnt = new_count;

    // add inputs
    for (int i = optind; i < argc; ++i) {

        // open input file
        int fd;
        const char *path = argv[i];
        if ((fd = open(path, O_RDONLY)) == -1)
            DieSys(path);
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

        // get information about file
        uint64_t size;
        struct stat st;
        if (fstat(fd, &st) == -1)
            DieSys(path);
        size = st.st_size;
        if (!S_ISREG(st.st_mode))
            Die(path, "not a regular file");

        // get time
        int64_t ts;
        uint16_t mtime, mdate;
        if (flag_nondeterministic)
            ts = st.st_mtime;
        else
            ts = 1700000000;
        GetDosLocalTime(ts, &mtime, &mdate);

        // determine size and alignment of local file header
        char *name = names[i];
        size_t namlen = strlen(name);
        size_t extlen = (2 + 2 + 8 + 8);
        size_t hdrlen = kZipLfileHdrMinSize + namlen + extlen;
        while ((zsize + hdrlen) & (flag_alignment - 1))
            ++zsize;

        // initialize zlib in raw deflate mode
        z_stream zs;
        int compression;
        if (!flag_level) {
            compression = kZipCompressionNone;
        } else {
            compression = kZipCompressionDeflate;
            zs.zalloc = 0;
            zs.zfree = 0;
            zs.opaque = 0;
            switch (deflateInit2(&zs, flag_level, Z_DEFLATED, -MAX_WBITS, DEF_MEM_LEVEL,
                                 Z_DEFAULT_STRATEGY)) {
            case Z_OK:
                break;
            case Z_MEM_ERROR:
                DieOom();
            default:
                npassert(!"deflateInit2() called with invalid parameters");
            }
        }

        // copy file
        ssize_t rc;
        uint32_t crc = 0;
        uint64_t compsize = 0;
        _Alignas(4096) static uint8_t iobuf[CHUNK];
        _Alignas(4096) static uint8_t cdbuf[CHUNK];
        for (off_t i = 0; i < size; i += rc) {
            // read chunk
            if ((rc = pread(fd, iobuf, Min(size, CHUNK), i)) <= 0)
                DieSys(path);
            posix_fadvise(fd, i, Min(size, CHUNK), POSIX_FADV_DONTNEED);
            crc = crc32(crc, iobuf, rc);
            if (!flag_level) {
                // write uncompressed chunk to output
                if (pwrite(zfd, iobuf, rc, zsize + hdrlen + compsize) != rc)
                    DieSys(zpath);
                compsize += rc;
            } else {
                // compress chunk and write to output
                zs.avail_in = rc;
                zs.next_in = iobuf;
                do {
                    zs.next_out = cdbuf;
                    zs.avail_out = CHUNK;
                    int boop;
                    switch ((boop = deflate(&zs, rc != CHUNK ? Z_FINISH : Z_FULL_FLUSH))) {
                    case Z_MEM_ERROR:
                        DieOom();
                    case Z_STREAM_ERROR:
                        npassert(!"deflate() stream error");
                    default:
                        break;
                    }
                    ssize_t have = CHUNK - zs.avail_out;
                    if (pwrite(zfd, cdbuf, have, zsize + hdrlen + compsize) != have)
                        DieSys(zpath);
                    compsize += have;
                } while (!zs.avail_out);
            }
        }
        if (flag_level)
            npassert(deflateEnd(&zs) == Z_OK);

        // write local file header
        uint8_t *lochdr = Malloc(hdrlen);
        uint8_t *p = lochdr;

        p = ZIP_WRITE32(p, kZipLfileHdrMagic);
        p = ZIP_WRITE16(p, kZipEra2001);
        p = ZIP_WRITE16(p, kZipGflagUtf8);
        p = ZIP_WRITE16(p, compression);
        p = ZIP_WRITE16(p, mtime);
        p = ZIP_WRITE16(p, mdate);
        p = ZIP_WRITE32(p, crc);
        p = ZIP_WRITE32(p, 0xffffffffu); // compressed size
        p = ZIP_WRITE32(p, 0xffffffffu); // uncompressed size
        p = ZIP_WRITE16(p, namlen);
        p = ZIP_WRITE16(p, extlen);
        p = mempcpy(p, name, namlen);

        p = ZIP_WRITE16(p, kZipExtraZip64);
        p = ZIP_WRITE16(p, 8 + 8);
        p = ZIP_WRITE64(p, size); // uncompressed size
        p = ZIP_WRITE64(p, compsize); // compressed size

        npassert(p == lochdr + hdrlen);
        if (pwrite(zfd, lochdr, hdrlen, zsize) != hdrlen)
            DieSys(zpath);
        free(lochdr);

        // create central directory entry
        extlen = (2 + 2 + 8 + 8 + 8);
        hdrlen = kZipCfileHdrMinSize + namlen + extlen;
        cdir = Realloc(cdir, cdirsize + hdrlen);
        uint8_t *cdirhdr = cdir + cdirsize;
        cdirsize += hdrlen;
        p = cdirhdr;

        p = ZIP_WRITE32(p, kZipCfileHdrMagic);
        p = ZIP_WRITE16(p, kZipOsUnix << 8 | kZipEra2001); // version made by
        p = ZIP_WRITE16(p, kZipEra2001); // version needed to extract
        p = ZIP_WRITE16(p, kZipGflagUtf8);
        p = ZIP_WRITE16(p, compression);
        p = ZIP_WRITE16(p, mtime);
        p = ZIP_WRITE16(p, mdate);
        p = ZIP_WRITE32(p, crc);
        p = ZIP_WRITE32(p, 0xffffffffu); // compressed size
        p = ZIP_WRITE32(p, 0xffffffffu); // uncompressed size
        p = ZIP_WRITE16(p, namlen);
        p = ZIP_WRITE16(p, extlen);
        p = ZIP_WRITE16(p, 0); // comment length
        p = ZIP_WRITE16(p, 0); // disk number start
        p = ZIP_WRITE16(p, kZipIattrBinary);
        p = ZIP_WRITE32(p, NormalizeMode(st.st_mode) << 16); // external file attributes
        p = ZIP_WRITE32(p, 0xffffffffu); // lfile offset
        p = mempcpy(p, name, namlen);

        p = ZIP_WRITE16(p, kZipExtraZip64);
        p = ZIP_WRITE16(p, 8 + 8 + 8);
        p = ZIP_WRITE64(p, size); // uncompressed size
        p = ZIP_WRITE64(p, compsize); // compressed size
        p = ZIP_WRITE64(p, zsize); // lfile offset
        npassert(p == cdirhdr + hdrlen);

        // finish up
        ++cnt;
        zsize += hdrlen + compsize;
        if (close(fd))
            DieSys(path);

        // log asset creation
        if (flag_verbose)
            tinyprint(2, path, " -> ", name, "\n", NULL);
    }

    // write out central directory
    if (pwrite(zfd, cdir, cdirsize, zsize) != cdirsize)
        DieSys(zpath);
    free(cdir);

    // write out end of central directory
    uint8_t eocd[kZipCdirHdrMinSize + kZipCdir64HdrMinSize + kZipCdir64LocatorSize];
    uint8_t *p = eocd;
    p = ZIP_WRITE32(p, kZipCdir64HdrMagic);
    p = ZIP_WRITE64(p, kZipCdir64HdrMinSize - 12); // size of eocd64
    p = ZIP_WRITE16(p, kZipOsUnix << 8 | kZipEra2001); // version made by
    p = ZIP_WRITE16(p, kZipEra2001); // version needed to extract
    p = ZIP_WRITE32(p, 0); // number of this disk
    p = ZIP_WRITE32(p, 0); // number of disk with start of central directory
    p = ZIP_WRITE64(p, cnt); // number of records on disk
    p = ZIP_WRITE64(p, cnt); // number of records
    p = ZIP_WRITE64(p, cdirsize); // size of central directory
    p = ZIP_WRITE64(p, zsize); // offset of start of central directory
    p = ZIP_WRITE32(p, kZipCdir64LocatorMagic);
    p = ZIP_WRITE32(p, 0); // number of disk with eocd64
    p = ZIP_WRITE64(p, zsize + cdirsize); // offset of eocd64
    p = ZIP_WRITE32(p, 1); // total number of disks
    p = ZIP_WRITE32(p, kZipCdirHdrMagic);
    p = ZIP_WRITE16(p, 0); // number of this disk
    p = ZIP_WRITE16(p, 0); // number of disks
    p = ZIP_WRITE16(p, cnt); // number of records on disk
    p = ZIP_WRITE16(p, cnt); // number of records
    p = ZIP_WRITE32(p, cdirsize); // size of central directory
    p = ZIP_WRITE32(p, 0xffffffffu); // offset of central directory
    p = ZIP_WRITE16(p, 0); // comment length
    npassert(p == eocd + sizeof(eocd));
    if (pwrite(zfd, eocd, sizeof(eocd), zsize + cdirsize) != sizeof(eocd))
        DieSys(zpath);

    // close output
    if (close(zfd))
        DieSys(zpath);
}
