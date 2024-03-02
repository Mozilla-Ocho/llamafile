#ifndef COSMO_ZIP_
#define COSMO_ZIP_

/**
 * @fileoverview PKZIP Data Structures.
 */

#ifdef __COSMOPOLITAN__
#define ZM_(x) ~__veil("r", ~x) /* prevent magic from appearing in binary */
#else
#define ZM_(x) x
#endif

#ifdef TINY
#define _ZE(x) -1
#else
#define _ZE(x) x
#endif

#define kZipOk 0
#define kZipErrorEocdNotFound _ZE(-1)
#define kZipErrorEocdOffsetOverflow _ZE(-2)
#define kZipErrorEocdMagicNotFound _ZE(-3)
#define kZipErrorEocdSizeOverflow _ZE(-4)
#define kZipErrorEocdDiskMismatch _ZE(-5)
#define kZipErrorEocdOffsetSizeOverflow _ZE(-6)
#define kZipErrorEocdRecordsMismatch _ZE(-7)
#define kZipErrorEocdRecordsOverflow _ZE(-8)
#define kZipErrorCdirOffsetPastEocd _ZE(-9)
#define kZipErrorEocdLocatorMagic _ZE(-10)
#define kZipErrorEocdLocatorOffset _ZE(-11)
#define kZipErrorRaceCondition _ZE(-12)
#define kZipErrorMapFailed _ZE(-13)
#define kZipErrorOpenFailed _ZE(-14)

#define kZipCosmopolitanVersion kZipEra2001

#define kZipOsDos 0
#define kZipOsAmiga 1
#define kZipOsOpenvms 2
#define kZipOsUnix 3
#define kZipOsVmcms 4
#define kZipOsAtarist 5
#define kZipOsOs2hpfs 6
#define kZipOsMacintosh 7
#define kZipOsZsystem 8
#define kZipOsCpm 9
#define kZipOsWindowsntfs 10
#define kZipOsMvsos390zos 11
#define kZipOsVse 12
#define kZipOsAcornrisc 13
#define kZipOsVfat 14
#define kZipOsAltmvs 15
#define kZipOsBeos 16
#define kZipOsTandem 17
#define kZipOsOs400 18
#define kZipOsOsxdarwin 19

#define kZipEra1989 10 /* PKZIP 1.0 */
#define kZipEra1993 20 /* PKZIP 2.0: deflate/subdir/etc. support */
#define kZipEra2001 45 /* PKZIP 4.5: kZipExtraZip64 support */

#define kZipIattrBinary 0 /* first bit not set */
#define kZipIattrText 1 /* first bit set */

#define kZipCompressionNone 0
#define kZipCompressionDeflate 8

#define kZipCdirHdrMagic ZM_(0x06054b50) /* PK♣♠ "PK\5\6" */
#define kZipCdirHdrMagicTodo ZM_(0x19184b50) /* PK♣♠ "PK\30\31" */
#define kZipCdirHdrMinSize 22
#define kZipCdirHdrLinkableSize 294
#define kZipCdirDiskOffset 4
#define kZipCdirStartingDiskOffset 6
#define kZipCdirRecordsOnDiskOffset 8
#define kZipCdirRecordsOffset 10
#define kZipCdirSizeOffset 12
#define kZipCdirOffsetOffset 16
#define kZipCdirCommentSizeOffset 20

#define kZipCdir64HdrMagic ZM_(0x06064b50) /* PK♠♠ "PK\6\6" */
#define kZipCdir64HdrMinSize 56
#define kZipCdir64LocatorMagic ZM_(0x07064b50) /* PK♠• "PK\6\7" */
#define kZipCdir64LocatorSize 20

#define kZipCfileHdrMagic ZM_(0x02014b50) /* PK☺☻ "PK\1\2" */
#define kZipCfileHdrMinSize 46
#define kZipCfileOffsetGeneralflag 8
#define kZipCfileOffsetCompressionmethod 10
#define kZipCfileOffsetLastmodifiedtime 12
#define kZipCfileOffsetLastmodifieddate 14
#define kZipCfileOffsetCrc32 16
#define kZipCfileOffsetCompressedsize 20
#define kZipCfileOffsetUncompressedsize 24
#define kZipCfileOffsetNamesize 28
#define kZipCfileOffsetExternalattributes 38
#define kZipCfileOffsetOffset 42

#define kZipLfileHdrMagic ZM_(0x04034b50) /* PK♥♦ "PK\3\4" */
#define kZipLfileHdrMinSize 30
#define kZipLfileOffsetVersionNeeded 4
#define kZipLfileOffsetGeneralflag 6
#define kZipLfileOffsetCompressionmethod 8
#define kZipLfileOffsetLastmodifiedtime 10
#define kZipLfileOffsetLastmodifieddate 12
#define kZipLfileOffsetCrc32 14
#define kZipLfileOffsetNamesize 26
#define kZipLfileOffsetCompressedsize 18
#define kZipLfileOffsetUncompressedsize 22

#define kZipGflagUtf8 0x800

#define kZipExtraHdrSize 4
#define kZipExtraZip64 0x0001
#define kZipExtraNtfs 0x000a
#define kZipExtraUnix 0x000d
#define kZipExtraExtendedTimestamp 0x5455
#define kZipExtraInfoZipNewUnixExtra 0x7875

#define kZipCfileMagic "PK\001\002"

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define ZIP_SWAP16_(P) __builtin_bswap16(P)
#define ZIP_SWAP32_(P) __builtin_bswap32(P)
#define ZIP_SWAP64_(P) __builtin_bswap64(P)
#else
#define ZIP_SWAP16_(P) (P)
#define ZIP_SWAP32_(P) (P)
#define ZIP_SWAP64_(P) (P)
#endif

#define ZIP_READ16(P) \
    __extension__({ \
        uint16_t __x; \
        __builtin_memcpy(&__x, P, 16 / 8); \
        ZIP_SWAP16_(__x); \
    })
#define ZIP_READ32(P) \
    __extension__({ \
        uint32_t __x; \
        __builtin_memcpy(&__x, P, 32 / 8); \
        ZIP_SWAP32_(__x); \
    })
#define ZIP_READ64(P) \
    __extension__({ \
        uint64_t __x; \
        __builtin_memcpy(&__x, P, 64 / 8); \
        ZIP_SWAP64_(__x); \
    })

#define ZIP_WRITE16(P, X) \
    __extension__({ \
        __typeof__(&(P)[0]) __p = (P); \
        uint16_t __x = ZIP_SWAP16_(X); \
        __builtin_memcpy(__p, &__x, 16 / 8); \
        __p + 16 / 8; \
    })
#define ZIP_WRITE32(P, X) \
    __extension__({ \
        __typeof__(&(P)[0]) __p = (P); \
        uint32_t __x = ZIP_SWAP32_(X); \
        __builtin_memcpy(__p, &__x, 32 / 8); \
        __p + 32 / 8; \
    })
#define ZIP_WRITE64(P, X) \
    __extension__({ \
        __typeof__(&(P)[0]) __p = (P); \
        uint64_t __x = ZIP_SWAP64_(X); \
        __builtin_memcpy(__p, &__x, 64 / 8); \
        __p + 64 / 8; \
    })

/* end of central directory record */
#define ZIP_CDIR_MAGIC(P) ZIP_READ32(P)
#define ZIP_CDIR_DISK(P) ZIP_READ16((P) + kZipCdirDiskOffset)
#define ZIP_CDIR_STARTINGDISK(P) ZIP_READ16((P) + kZipCdirStartingDiskOffset)
#define ZIP_CDIR_RECORDSONDISK(P) ZIP_READ16((P) + kZipCdirRecordsOnDiskOffset)
#define ZIP_CDIR_RECORDS(P) ZIP_READ16((P) + kZipCdirRecordsOffset)
#define ZIP_CDIR_SIZE(P) ZIP_READ32((P) + kZipCdirSizeOffset)
#define ZIP_CDIR_OFFSET(P) ZIP_READ32((P) + kZipCdirOffsetOffset)
#define ZIP_CDIR_COMMENTSIZE(P) ZIP_READ16((P) + kZipCdirCommentSizeOffset)
#define ZIP_CDIR_COMMENT(P) ((P) + 22) /* recommend stopping at nul */
#define ZIP_CDIR_HDRSIZE(P) (ZIP_CDIR_COMMENTSIZE(P) + kZipCdirHdrMinSize)

/* zip64 end of central directory record */
#define ZIP_CDIR64_MAGIC(P) ZIP_READ32(P)
#define ZIP_CDIR64_HDRSIZE(P) (ZIP_READ64((P) + 4) + 12)
#define ZIP_CDIR64_VERSIONMADE(P) ZIP_READ16((P) + 12)
#define ZIP_CDIR64_VERSIONNEED(P) ZIP_READ16((P) + 14)
#define ZIP_CDIR64_DISK(P) ZIP_READ32((P) + 16)
#define ZIP_CDIR64_STARTINGDISK(P) ZIP_READ32((P) + 20)
#define ZIP_CDIR64_RECORDSONDISK(P) ZIP_READ64((P) + 24)
#define ZIP_CDIR64_RECORDS(P) ZIP_READ64((P) + 32)
#define ZIP_CDIR64_SIZE(P) ZIP_READ64((P) + 40)
#define ZIP_CDIR64_OFFSET(P) ZIP_READ64((P) + 48)
#define ZIP_CDIR64_COMMENTSIZE(P) (ZIP_CDIR64_HDRSIZE(P) >= 56 ? ZIP_CDIR64_HDRSIZE(P) - 56 : 0)
#define ZIP_CDIR64_COMMENT(P) ((P) + 56) /* recommend stopping at nul */
#define ZIP_LOCATE64_MAGIC(P) ZIP_READ32(P)
#define ZIP_LOCATE64_STARTINGDISK(P) ZIP_READ32((P) + 4)
#define ZIP_LOCATE64_OFFSET(P) ZIP_READ64((P) + 8)
#define ZIP_LOCATE64_TOTALDISKS(P) ZIP_READ32((P) + 12)

/* central directory file header */
#define ZIP_CFILE_MAGIC(P) ZIP_READ32(P)
#define ZIP_CFILE_VERSIONMADE(P) (255 & (P)[4])
#define ZIP_CFILE_FILEATTRCOMPAT(P) (255 & (P)[5])
#define ZIP_CFILE_VERSIONNEED(P) (255 & (P)[6])
#define ZIP_CFILE_OSNEED(P) (255 & (P)[7])
#define ZIP_CFILE_GENERALFLAG(P) ZIP_READ16((P) + kZipCfileOffsetGeneralflag)
#define ZIP_CFILE_COMPRESSIONMETHOD(P) ZIP_READ16((P) + kZipCfileOffsetCompressionmethod)
#define ZIP_CFILE_LASTMODIFIEDTIME(P) \
    ZIP_READ16((P) + kZipCfileOffsetLastmodifiedtime) /* @see DOS_TIME() */
#define ZIP_CFILE_LASTMODIFIEDDATE(P) \
    ZIP_READ16((P) + kZipCfileOffsetLastmodifieddate) /* @see DOS_DATE() */
#define ZIP_CFILE_CRC32(P) ZIP_READ32((P) + kZipCfileOffsetCrc32)
#define ZIP_CFILE_COMPRESSEDSIZE(P) ZIP_READ32(P + kZipCfileOffsetCompressedsize)
#define ZIP_CFILE_UNCOMPRESSEDSIZE(P) ZIP_READ32((P) + kZipCfileOffsetUncompressedsize)
#define ZIP_CFILE_NAMESIZE(P) ZIP_READ16((P) + kZipCfileOffsetNamesize)
#define ZIP_CFILE_EXTRASIZE(P) ZIP_READ16((P) + 30)
#define ZIP_CFILE_COMMENTSIZE(P) ZIP_READ16((P) + 32)
#define ZIP_CFILE_DISK(P) ZIP_READ16((P) + 34)
#define ZIP_CFILE_INTERNALATTRIBUTES(P) ZIP_READ16((P) + 36)
#define ZIP_CFILE_EXTERNALATTRIBUTES(P) ZIP_READ32((P) + kZipCfileOffsetExternalattributes)
#define ZIP_CFILE_OFFSET(P) ZIP_READ32((P) + kZipCfileOffsetOffset)
#define ZIP_CFILE_NAME(P) ((const char *)((P) + 46)) /* not nul-terminated */
#define ZIP_CFILE_EXTRA(P) ((P) + 46 + ZIP_CFILE_NAMESIZE(P))
#define ZIP_CFILE_COMMENT(P) \
    ((const char *)((P) + 46 + ZIP_CFILE_NAMESIZE(P) + \
                    ZIP_CFILE_EXTRASIZE(P))) /* recommend stopping at nul */
#define ZIP_CFILE_HDRSIZE(P) \
    (ZIP_CFILE_NAMESIZE(P) + ZIP_CFILE_EXTRASIZE(P) + ZIP_CFILE_COMMENTSIZE(P) + \
     kZipCfileHdrMinSize)

/* local file header */
#define ZIP_LFILE_MAGIC(P) ZIP_READ32(P)
#define ZIP_LFILE_VERSIONNEED(P) (255 & (P)[4])
#define ZIP_LFILE_OSNEED(P) (255 & (P)[5])
#define ZIP_LFILE_GENERALFLAG(P) ZIP_READ16((P) + kZipLfileOffsetGeneralflag)
#define ZIP_LFILE_COMPRESSIONMETHOD(P) ZIP_READ16((P) + kZipLfileOffsetCompressionmethod)
#define ZIP_LFILE_LASTMODIFIEDTIME(P) \
    ZIP_READ16((P) + kZipLfileOffsetLastmodifiedtime) /* @see DOS_TIME() */
#define ZIP_LFILE_LASTMODIFIEDDATE(P) \
    ZIP_READ16((P) + kZipLfileOffsetLastmodifieddate) /* @see DOS_DATE() */
#define ZIP_LFILE_CRC32(P) ZIP_READ32((P) + kZipLfileOffsetCrc32)
#define ZIP_LFILE_COMPRESSEDSIZE(P) ZIP_READ32((P) + kZipLfileOffsetCompressedsize)
#define ZIP_LFILE_UNCOMPRESSEDSIZE(P) ZIP_READ32((P) + kZipLfileOffsetUncompressedsize)
#define ZIP_LFILE_NAMESIZE(P) ZIP_READ16((P) + kZipLfileOffsetNamesize)
#define ZIP_LFILE_EXTRASIZE(P) ZIP_READ16((P) + 28)
#define ZIP_LFILE_NAME(P) ((const char *)((P) + 30))
#define ZIP_LFILE_EXTRA(P) ((P) + 30 + ZIP_LFILE_NAMESIZE(P))
#define ZIP_LFILE_HDRSIZE(P) (ZIP_LFILE_NAMESIZE(P) + ZIP_LFILE_EXTRASIZE(P) + kZipLfileHdrMinSize)
#define ZIP_LFILE_CONTENT(P) ((P) + ZIP_LFILE_HDRSIZE(P))
#define ZIP_LFILE_SIZE(P) (ZIP_LFILE_HDRSIZE(P) + ZIP_LFILE_COMPRESSEDSIZE(P))

#define ZIP_EXTRA_HEADERID(P) ZIP_READ16(P)
#define ZIP_EXTRA_CONTENTSIZE(P) ZIP_READ16((P) + 2)
#define ZIP_EXTRA_CONTENT(P) ((P) + 4)
#define ZIP_EXTRA_SIZE(P) (ZIP_EXTRA_CONTENTSIZE(P) + kZipExtraHdrSize)

int64_t get_zip_cfile_offset(const uint8_t *);
int64_t get_zip_cfile_compressed_size(const uint8_t *);

#endif /* COSMO_ZIP_ */
