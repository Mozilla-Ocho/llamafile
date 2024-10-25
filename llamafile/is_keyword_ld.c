/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_ld.c llamafile/is_keyword_ld.gperf  */
/* Computed positions: -k'1,3,$' */

#if !((' ' == 32) && ('!' == 33) && ('"' == 34) && ('#' == 35) \
      && ('%' == 37) && ('&' == 38) && ('\'' == 39) && ('(' == 40) \
      && (')' == 41) && ('*' == 42) && ('+' == 43) && (',' == 44) \
      && ('-' == 45) && ('.' == 46) && ('/' == 47) && ('0' == 48) \
      && ('1' == 49) && ('2' == 50) && ('3' == 51) && ('4' == 52) \
      && ('5' == 53) && ('6' == 54) && ('7' == 55) && ('8' == 56) \
      && ('9' == 57) && (':' == 58) && (';' == 59) && ('<' == 60) \
      && ('=' == 61) && ('>' == 62) && ('?' == 63) && ('A' == 65) \
      && ('B' == 66) && ('C' == 67) && ('D' == 68) && ('E' == 69) \
      && ('F' == 70) && ('G' == 71) && ('H' == 72) && ('I' == 73) \
      && ('J' == 74) && ('K' == 75) && ('L' == 76) && ('M' == 77) \
      && ('N' == 78) && ('O' == 79) && ('P' == 80) && ('Q' == 81) \
      && ('R' == 82) && ('S' == 83) && ('T' == 84) && ('U' == 85) \
      && ('V' == 86) && ('W' == 87) && ('X' == 88) && ('Y' == 89) \
      && ('Z' == 90) && ('[' == 91) && ('\\' == 92) && (']' == 93) \
      && ('^' == 94) && ('_' == 95) && ('a' == 97) && ('b' == 98) \
      && ('c' == 99) && ('d' == 100) && ('e' == 101) && ('f' == 102) \
      && ('g' == 103) && ('h' == 104) && ('i' == 105) && ('j' == 106) \
      && ('k' == 107) && ('l' == 108) && ('m' == 109) && ('n' == 110) \
      && ('o' == 111) && ('p' == 112) && ('q' == 113) && ('r' == 114) \
      && ('s' == 115) && ('t' == 116) && ('u' == 117) && ('v' == 118) \
      && ('w' == 119) && ('x' == 120) && ('y' == 121) && ('z' == 122) \
      && ('{' == 123) && ('|' == 124) && ('}' == 125) && ('~' == 126))
/* The character set is not based on ISO-646.  */
#error "gperf generated tables don't work with this execution character set. Please report a bug to <bug-gperf@gnu.org>."
#endif

#line 1 "llamafile/is_keyword_ld.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 69
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 25
#define MIN_HASH_VALUE 6
#define MAX_HASH_VALUE 142
/* maximum key range = 137, duplicates = 0 */

#ifdef __GNUC__
__inline
#else
#ifdef __cplusplus
inline
#endif
#endif
static unsigned int
hash (register const char *str, register size_t len)
{
  static const unsigned char asso_values[] =
    {
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143,   5,   0,  45,  25,  35,
       40,  65,   0,  55, 143,  60,  15,  25,  10,  10,
        0,  20,   0,   0,  10,  10,  65,   0, 143,  55,
      143, 143, 143, 143, 143,   0, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
      143, 143, 143, 143, 143, 143
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_ld (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str6[sizeof("PT_TLS")];
      char stringpool_str7[sizeof("PT_PHDR")];
      char stringpool_str8[sizeof("PT_SHLIB")];
      char stringpool_str9[sizeof("PT_INTERP")];
      char stringpool_str12[sizeof("STARTUP")];
      char stringpool_str14[sizeof("SORT")];
      char stringpool_str15[sizeof("SEARCH_DIR")];
      char stringpool_str17[sizeof("AT")];
      char stringpool_str18[sizeof("SUBALIGN")];
      char stringpool_str20[sizeof("AFTER")];
      char stringpool_str21[sizeof("ASSERT")];
      char stringpool_str22[sizeof("PT_NULL")];
      char stringpool_str25[sizeof("SHORT")];
      char stringpool_str26[sizeof("TARGET")];
      char stringpool_str27[sizeof("SORT_BY_ALIGNMENT")];
      char stringpool_str29[sizeof("PHDR")];
      char stringpool_str30[sizeof("PHDRS")];
      char stringpool_str31[sizeof("OUTPUT_ARCH")];
      char stringpool_str32[sizeof("PT_LOAD")];
      char stringpool_str34[sizeof("PROVIDE_HIDDEN")];
      char stringpool_str35[sizeof("ONLY_IF_RW")];
      char stringpool_str36[sizeof("OUTPUT")];
      char stringpool_str37[sizeof("__DTOR_END__")];
      char stringpool_str38[sizeof("__DTOR_LIST__")];
      char stringpool_str39[sizeof("AS_NEEDED")];
      char stringpool_str40[sizeof("SQUAD")];
      char stringpool_str41[sizeof("HIDDEN")];
      char stringpool_str42[sizeof("PT_NOTE")];
      char stringpool_str43[sizeof("OUTPUT_FORMAT")];
      char stringpool_str44[sizeof("SORT_NONE")];
      char stringpool_str45[sizeof("ONLY_IF_RO")];
      char stringpool_str47[sizeof("SORT_BY_NAME")];
      char stringpool_str49[sizeof("BYTE")];
      char stringpool_str50[sizeof("FLAGS")];
      char stringpool_str52[sizeof("PROVIDE")];
      char stringpool_str53[sizeof("SECTIONS")];
      char stringpool_str54[sizeof("QUAD")];
      char stringpool_str55[sizeof("PT_DYNAMIC")];
      char stringpool_str56[sizeof("NOLOAD")];
      char stringpool_str57[sizeof("__CTOR_END__")];
      char stringpool_str58[sizeof("__CTOR_LIST__")];
      char stringpool_str60[sizeof("LD_FEATURE")];
      char stringpool_str61[sizeof("EXTERN")];
      char stringpool_str62[sizeof("FILEHDR")];
      char stringpool_str66[sizeof("NOCROSSREFS")];
      char stringpool_str67[sizeof("CONSTRUCTORS")];
      char stringpool_str70[sizeof("INPUT")];
      char stringpool_str71[sizeof("INSERT")];
      char stringpool_str72[sizeof("PT_GNU_STACK")];
      char stringpool_str73[sizeof("FORCE_COMMON_ALLOCATION")];
      char stringpool_str74[sizeof("FILL")];
      char stringpool_str75[sizeof("DSECT")];
      char stringpool_str76[sizeof("SORT_BY_INIT_PRIORITY")];
      char stringpool_str77[sizeof("REGION_ALIAS")];
      char stringpool_str79[sizeof("NOCROSSREFS_TO")];
      char stringpool_str80[sizeof("GROUP")];
      char stringpool_str81[sizeof("BEFORE")];
      char stringpool_str82[sizeof("VERSION")];
      char stringpool_str86[sizeof("COMMON")];
      char stringpool_str90[sizeof("INHIBIT_COMMON_ALLOCATION")];
      char stringpool_str94[sizeof("LONG")];
      char stringpool_str99[sizeof("KEEP")];
      char stringpool_str101[sizeof("CREATE_OBJECT_SYMBOLS")];
      char stringpool_str104[sizeof("COPY")];
      char stringpool_str105[sizeof("ENTRY")];
      char stringpool_str107[sizeof("OVERLAY")];
      char stringpool_str109[sizeof("INFO")];
      char stringpool_str111[sizeof("MEMORY")];
      char stringpool_str142[sizeof("INCLUDE")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "PT_TLS",
      "PT_PHDR",
      "PT_SHLIB",
      "PT_INTERP",
      "STARTUP",
      "SORT",
      "SEARCH_DIR",
      "AT",
      "SUBALIGN",
      "AFTER",
      "ASSERT",
      "PT_NULL",
      "SHORT",
      "TARGET",
      "SORT_BY_ALIGNMENT",
      "PHDR",
      "PHDRS",
      "OUTPUT_ARCH",
      "PT_LOAD",
      "PROVIDE_HIDDEN",
      "ONLY_IF_RW",
      "OUTPUT",
      "__DTOR_END__",
      "__DTOR_LIST__",
      "AS_NEEDED",
      "SQUAD",
      "HIDDEN",
      "PT_NOTE",
      "OUTPUT_FORMAT",
      "SORT_NONE",
      "ONLY_IF_RO",
      "SORT_BY_NAME",
      "BYTE",
      "FLAGS",
      "PROVIDE",
      "SECTIONS",
      "QUAD",
      "PT_DYNAMIC",
      "NOLOAD",
      "__CTOR_END__",
      "__CTOR_LIST__",
      "LD_FEATURE",
      "EXTERN",
      "FILEHDR",
      "NOCROSSREFS",
      "CONSTRUCTORS",
      "INPUT",
      "INSERT",
      "PT_GNU_STACK",
      "FORCE_COMMON_ALLOCATION",
      "FILL",
      "DSECT",
      "SORT_BY_INIT_PRIORITY",
      "REGION_ALIAS",
      "NOCROSSREFS_TO",
      "GROUP",
      "BEFORE",
      "VERSION",
      "COMMON",
      "INHIBIT_COMMON_ALLOCATION",
      "LONG",
      "KEEP",
      "CREATE_OBJECT_SYMBOLS",
      "COPY",
      "ENTRY",
      "OVERLAY",
      "INFO",
      "MEMORY",
      "INCLUDE"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142
    };

  if (len <= MAX_WORD_LENGTH && len >= MIN_WORD_LENGTH)
    {
      register unsigned int key = hash (str, len);

      if (key <= MAX_HASH_VALUE)
        {
          register int o = wordlist[key];
          if (o >= 0)
            {
              register const char *s = o + stringpool;

              if (*str == *s && !strncmp (str + 1, s + 1, len - 1) && s[len] == '\0')
                return s;
            }
        }
    }
  return 0;
}
