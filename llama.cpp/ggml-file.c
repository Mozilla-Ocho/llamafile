// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set net ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
#define _COSMO_SOURCE
#include <cosmo.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <unistd.h>
#include <sys/mman.h>
#include "ggml.h"
#include "zip.h"

#define Min(a, b) ((a) < (b) ? (a) : (b))

struct ggml_file {
    FILE * fp;
    size_t size;
    char * content;
    size_t position;
    void * mapping;
    size_t mapsize;
};

static uint64_t GetZipCfileOffset(const uint8_t *z) {
    uint64_t x;
    const uint8_t *p, *pe;
    if ((x = ZIP_CFILE_OFFSET(z)) == 0xFFFFFFFF) {
        for (p = ZIP_CFILE_EXTRA(z), pe = p + ZIP_CFILE_EXTRASIZE(z); p < pe;
             p += ZIP_EXTRA_SIZE(p)) {
            if (ZIP_EXTRA_HEADERID(p) == kZipExtraZip64 &&
                16 + 8 <= ZIP_EXTRA_CONTENTSIZE(p)) {
                return ZIP_READ64(ZIP_EXTRA_CONTENT(p) + 16);
            }
        }
    }
    return x;
}

static uint64_t GetZipCfileCompressedSize(const uint8_t *z) {
    uint64_t x;
    const uint8_t *p, *pe;
    if ((x = ZIP_CFILE_COMPRESSEDSIZE(z)) == 0xFFFFFFFF) {
        for (p = ZIP_CFILE_EXTRA(z), pe = p + ZIP_CFILE_EXTRASIZE(z); p < pe;
             p += ZIP_EXTRA_SIZE(p)) {
            if (ZIP_EXTRA_HEADERID(p) == kZipExtraZip64 &&
                8 + 8 <= ZIP_EXTRA_CONTENTSIZE(p)) {
                return ZIP_READ64(ZIP_EXTRA_CONTENT(p) + 8);
            }
        }
    }
    return x;
}

struct ggml_file *ggml_file_open(const char * fname, const char * mode) {
    int fd = -1;
    uint8_t * bufdata = NULL;
    size_t cdirsize = 0;
    uint8_t * cdirdata = NULL;
    struct ggml_file * file = NULL;

    if (!(file = malloc(sizeof(struct ggml_file)))) {
        goto Failure;
    }

    // open from the filesystem if it exists
    if ((file->fp = fopen(fname, mode))) {
        ggml_file_seek(file, 0, SEEK_END);
        file->size = ggml_file_tell(file);
        ggml_file_seek(file, 0, SEEK_SET);
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
    off_t off;
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
    for (int i = amt - Min(kZipCdirHdrMinSize, kZipCdir64LocatorSize); i; --i) {
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
            off = GetZipCfileOffset(cdirdata + entry_offset);
            file->size = GetZipCfileCompressedSize(cdirdata + entry_offset);
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
    if (ggml_is_numa() && posix_madvise(file->mapping, file->mapsize, POSIX_MADV_RANDOM)) {
        fprintf(stderr, "warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                strerror(errno));
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

FILE * ggml_file_fp(struct ggml_file * file) {
    return file->fp;
}

size_t ggml_file_size(struct ggml_file * file) {
    return file->size;
}

void * ggml_file_content(struct ggml_file * file) {
    return file->content;
}

size_t ggml_file_tell(struct ggml_file * file) {
    if (!file->fp) {
        return file->position;
    }
#ifdef _WIN32
    __int64 ret = _ftelli64(file->fp);
#else
    long ret = ftell(file->fp);
#endif
    GGML_ASSERT(ret != -1); // this really shouldn't fail
    return (size_t) ret;
}

void ggml_file_seek(struct ggml_file * file, size_t offset, int whence) {
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
#ifdef _WIN32
    int ret = _fseeki64(file->fp, (__int64) offset, whence);
#else
    int ret = fseek(file->fp, (long) offset, whence);
#endif
    GGML_ASSERT(ret == 0); // same
}

long ggml_file_read(struct ggml_file * file, void * ptr, size_t len) {
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

long ggml_file_write(struct ggml_file * file, const void * ptr, size_t len) {
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

void ggml_file_close(struct ggml_file * file) {
    if (file->fp) {
        fclose(file->fp);
    }
    if (file->mapping && file->mapping != MAP_FAILED) {
        // TODO(jart): reference count this mapping w/ llama_mmap
        // munmap(file->mapping, file->mapsize);
    }
}
