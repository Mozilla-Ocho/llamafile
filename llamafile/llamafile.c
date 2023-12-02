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
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <unistd.h>
#include <sys/mman.h>
#include "zip.h"

#define Min(a, b) ((a) < (b) ? (a) : (b))

asm(".ident\t\"\\n\\n\
llamafile (Apache 2.0)\\n\
Copyright 2023 Mozilla Foundation\\n\
\\n\
Licensed under the Apache License, Version 2.0 (the \\\"License\\\");\\n\
you may not use this file except in compliance with the License.\\n\
You may obtain a copy of the License at\\n\
\\n\
    http://www.apache.org/licenses/LICENSE-2.0\\n\
\\n\
Unless required by applicable law or agreed to in writing, software\\n\
distributed under the License is distributed on an \\\"AS IS\\\" BASIS,\\n\
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\\n\
See the License for the specific language governing permissions and\\n\
limitations under the License.\"");

struct llamafile {
    FILE *fp;
    size_t size;
    char *content;
    size_t position;
    void *mapping;
    size_t mapsize;
};

struct llamafile *llamafile_open(const char *fname, const char *mode) {
    int fd = -1;
    uint8_t *bufdata = NULL;
    size_t cdirsize = 0;
    uint8_t *cdirdata = NULL;
    struct llamafile *file = NULL;

    if (!(file = malloc(sizeof(struct llamafile)))) {
        goto Failure;
    }

    // open from the filesystem if it exists
    if ((file->fp = fopen(fname, mode))) {
        llamafile_seek(file, 0, SEEK_END);
        file->size = llamafile_tell(file);
        llamafile_seek(file, 0, SEEK_SET);
        return file;
    }
    if (errno != ENOENT) {
        goto Failure;
    }

    // try opening from this executable's zip store
    const char *prog = GetProgramExecutableName();
    if ((fd = open(prog, O_RDONLY | O_CLOEXEC)) == -1) {
        errno = ENOENT;
        goto Failure;
    }
    ssize_t rc;
    if ((rc = lseek(fd, 0, SEEK_END)) == -1) {
        fprintf(stderr, "warning: failed to seek executable: %s: %s\n", prog, strerror(errno));
        goto Failure;
    }
    file->size = rc;

    // read the last 64kb of file
    // the zip file format magic can be anywhere in there
    int amt;
    uint64_t off;
    if (file->size <= 65536) {
        off = 0;
        amt = file->size;
    } else {
        off = file->size - 65536;
        amt = file->size - off;
    }
    if (!(bufdata = malloc(65536))) {
        goto Failure;
    }
    if (pread(fd, bufdata, amt, off) != amt) {
        fprintf(stderr, "warning: failed to pread end of file: %s: %s\n", prog, strerror(errno));
        goto Failure;
    }

    // search backwards for the end-of-central-directory record
    // the eocd (cdir) says where the central directory (cfile) array is located
    // we consistency check some legacy fields, to be extra sure that it is eocd
    unsigned cnt = 0;
    for (int i = amt - Min(kZipCdirHdrMinSize, kZipCdir64LocatorSize); i >= 0; --i) {
        uint32_t magic = ZIP_READ32(bufdata + i);
        if (magic == kZipCdir64LocatorMagic && i + kZipCdir64LocatorSize <= amt &&
            pread(fd, bufdata, kZipCdir64HdrMinSize,
                  ZIP_LOCATE64_OFFSET(bufdata + i)) == (long)kZipCdir64HdrMinSize &&
            ZIP_READ32(bufdata) == kZipCdir64HdrMagic &&
            ZIP_CDIR64_RECORDS(bufdata) == ZIP_CDIR64_RECORDSONDISK(bufdata) &&
            ZIP_CDIR64_RECORDS(bufdata) && ZIP_CDIR64_SIZE(bufdata) <= INT_MAX) {
            cnt = ZIP_CDIR64_RECORDS(bufdata);
            off = ZIP_CDIR64_OFFSET(bufdata);
            amt = ZIP_CDIR64_SIZE(bufdata);
            break;
        }
        if (magic == kZipCdirHdrMagic && i + kZipCdirHdrMinSize <= amt &&
            ZIP_CDIR_RECORDS(bufdata + i) == ZIP_CDIR_RECORDSONDISK(bufdata + i) &&
            ZIP_CDIR_RECORDS(bufdata + i) && ZIP_CDIR_SIZE(bufdata + i) <= INT_MAX &&
            ZIP_CDIR_OFFSET(bufdata + i) != 0xffffffffu) {
            cnt = ZIP_CDIR_RECORDS(bufdata + i);
            off = ZIP_CDIR_OFFSET(bufdata + i);
            amt = ZIP_CDIR_SIZE(bufdata + i);
            break;
        }
    }
    if (cnt <= 0) {
        // this executable isn't a zip file
        goto Failure;
    }

    // read the central directory
    cdirsize = amt;
    if (!(cdirdata = malloc(cdirsize))) {
        goto Failure;
    }
    if (pread(fd, cdirdata, cdirsize, off) != (long)cdirsize) {
        fprintf(stderr, "warning: failed to pread zip cdir: %s: %s\n", prog, strerror(errno));
        goto Failure;
    }
    if (ZIP_READ32(cdirdata) != kZipCfileHdrMagic) {
        fprintf(stderr, "warning: failed to locate zip central directory: %s\n", prog);
        goto Failure;
    }

    // look for filename in the directory
    bool found = false;
    int fname_len = strlen(fname);
    unsigned entry_index, entry_offset;
    for (entry_index = entry_offset = 0;
         entry_index < cnt && entry_offset + kZipCfileHdrMinSize <= cdirsize &&
                       entry_offset + ZIP_CFILE_HDRSIZE(cdirdata + entry_offset) <= cdirsize;
         ++entry_index, entry_offset += ZIP_CFILE_HDRSIZE(cdirdata + entry_offset)) {
        if (ZIP_CFILE_MAGIC(cdirdata + entry_offset) != kZipCfileHdrMagic) {
            fprintf(stderr, "error: corrupted zip central directory entry magic\n");
            goto Failure;
        }
        if (fname_len == ZIP_CFILE_NAMESIZE(cdirdata + entry_offset) &&
            !memcmp(fname, ZIP_CFILE_NAME(cdirdata + entry_offset), fname_len)) {
            off = get_zip_cfile_offset(cdirdata + entry_offset);
            file->size = get_zip_cfile_compressed_size(cdirdata + entry_offset);
            found = true;
            break;
        }
    }
    if (!found) {
        goto Failure;
    }
    if (ZIP_CFILE_COMPRESSIONMETHOD(cdirdata + entry_offset) != kZipCompressionNone) {
        fprintf(stderr, "error: weights stored in the zip executable can't be stored using compression (try zip -0 flag)\n");
        goto Failure;
    }

    // read the zip local file header
    // this is needed to determine offset of file content
    uint8_t lfile[kZipLfileHdrMinSize];
    if (pread(fd, lfile, kZipLfileHdrMinSize, off) != kZipLfileHdrMinSize) {
        fprintf(stderr, "error: failed to pread lfile: %s\n", prog);
        goto Failure;
    }
    if (ZIP_LFILE_MAGIC(lfile) != kZipLfileHdrMagic) {
        fprintf(stderr, "error: corrupted zip local file magic\n");
        goto Failure;
    }
    off += ZIP_LFILE_HDRSIZE(lfile);

    // map the file into memory
    long pagesz = sysconf(_SC_PAGESIZE);
    off_t mapoff = off & -pagesz;
    long skew = off - mapoff;
    file->mapsize = skew + file->size;
    file->mapping = mmap(0, file->mapsize, PROT_READ, MAP_SHARED, fd, mapoff);
    if (file->mapping == MAP_FAILED) {
        fprintf(stderr, "warning: failed to map %s from %s: %s\n",
                fname, prog, strerror(errno));
        goto Failure;
    }

    // setup our synthetic file
    file->position = 0;
    file->content = (char *)file->mapping + skew;

    // return object
    free(cdirdata);
    free(bufdata);
    close(fd);
    return file;

Failure:
    if (fd != -1) {
        close(fd);
    }
    free(cdirdata);
    free(bufdata);
    free(file);
    return 0;
}

FILE *llamafile_fp(struct llamafile *file) {
    return file->fp;
}

size_t llamafile_size(struct llamafile *file) {
    return file->size;
}

void *llamafile_content(struct llamafile *file) {
    return file->content;
}

size_t llamafile_tell(struct llamafile *file) {
    if (!file->fp) {
        return file->position;
    }
    long ret = ftell(file->fp);
    unassert(ret != -1);  // shouldn't fail because we seeked earlier
    return (size_t) ret;
}

void llamafile_seek(struct llamafile *file, size_t offset, int whence) {
    if (!file->fp) {
        switch (whence) {
            case SEEK_SET:
                file->position = offset;
                break;
            case SEEK_CUR:
                file->position += offset;
                break;
            case SEEK_END:
                file->position = file->size + offset;
                break;
        }
        return;
    }
    unassert(!fseek(file->fp, (long) offset, whence));
}

long llamafile_read(struct llamafile *file, void *ptr, size_t len) {
    if (len == 0) {
        return 0;
    }
    if (!file->fp) {
        if (file->position > file->size) {
            return 0;
        }
        size_t remain = file->size - file->position;
        size_t amt = Min(len, remain);
        memcpy(ptr, file->content + file->position, amt);
        file->position += amt;
        return amt;
    }
    errno = 0;
    size_t ret = fread(ptr, len, 1, file->fp);
    if (ferror(file->fp)) {
        return -1;
    }
    if (ret != 1) {
        return 0;
    }
    return len;
}

long llamafile_write(struct llamafile *file, const void *ptr, size_t len) {
    if (len == 0) {
        return 0;
    }
    if (!file->fp) {
        errno = EROFS;
        return -1;
    }
    errno = 0;
    size_t ret = fwrite(ptr, len, 1, file->fp);
    if (ferror(file->fp)) {
        return -1;
    }
    if (ret != 1) {
        return 0;
    }
    return len;
}

void llamafile_close(struct llamafile *file) {
    if (file->fp) {
        fclose(file->fp);
    }
    if (file->mapping && file->mapping != MAP_FAILED) {
        // TODO(jart): reference count this mapping w/ llama_mmap
        // munmap(file->mapping, file->mapsize);
    }
}
