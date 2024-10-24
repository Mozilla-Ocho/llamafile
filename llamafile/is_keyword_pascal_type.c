/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_pascal_type.c llamafile/is_keyword_pascal_type.gperf  */
/* Computed positions: -k'1-2,4-5' */

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

#line 1 "llamafile/is_keyword_pascal_type.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 86
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 12
#define MIN_HASH_VALUE 9
#define MAX_HASH_VALUE 178
/* maximum key range = 170, duplicates = 0 */

#ifndef GPERF_DOWNCASE
#define GPERF_DOWNCASE 1
static unsigned char gperf_downcase[256] =
  {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
     30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
     45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
     60,  61,  62,  63,  64,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    122,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
    195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
    255
  };
#endif

#ifndef GPERF_CASE_STRNCMP
#define GPERF_CASE_STRNCMP 1
static int
gperf_case_strncmp (register const char *s1, register const char *s2, register size_t n)
{
  for (; n > 0;)
    {
      unsigned char c1 = gperf_downcase[(unsigned char)*s1++];
      unsigned char c2 = gperf_downcase[(unsigned char)*s2++];
      if (c1 != 0 && c1 == c2)
        {
          n--;
          continue;
        }
      return (int)c1 - (int)c2;
    }
  return 0;
}
#endif

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
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179,  45,
       25,   5,  15, 179,   0, 179,   0, 179, 179, 179,
      179, 179, 179, 179, 179,  45,  35,  50,  20,   5,
       60,  80,  40,  40,   0,   5,   0,  60,   0,  30,
        0, 179,   0,  10,   5,  45,   5,   0,   5,  30,
      179, 179, 179, 179, 179, 179, 179,  45,  35,  50,
       20,   5,  60,  80,  40,  40,   0,   5,   0,  60,
        0,  30,   0, 179,   0,  10,   5,  45,   5,   0,
        5,  30, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179, 179, 179, 179, 179,
      179, 179, 179, 179, 179, 179
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
      case 2:
        hval += asso_values[(unsigned char)str[1]];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

const char *
is_keyword_pascal_type (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str9[sizeof("real")];
      char stringpool_str18[sizeof("set")];
      char stringpool_str19[sizeof("text")];
      char stringpool_str23[sizeof("extended")];
      char stringpool_str24[sizeof("pextended")];
      char stringpool_str25[sizeof("pword")];
      char stringpool_str26[sizeof("real48")];
      char stringpool_str29[sizeof("pwordbool")];
      char stringpool_str30[sizeof("pwordarray")];
      char stringpool_str34[sizeof("pwidechar")];
      char stringpool_str35[sizeof("pdate")];
      char stringpool_str36[sizeof("pwidestring")];
      char stringpool_str39[sizeof("pdatetime")];
      char stringpool_str40[sizeof("tdate")];
      char stringpool_str41[sizeof("record")];
      char stringpool_str42[sizeof("pointer")];
      char stringpool_str44[sizeof("tdatetime")];
      char stringpool_str45[sizeof("dword")];
      char stringpool_str47[sizeof("tobject")];
      char stringpool_str48[sizeof("ppointer")];
      char stringpool_str49[sizeof("pshortint")];
      char stringpool_str50[sizeof("pbyte")];
      char stringpool_str51[sizeof("pint64")];
      char stringpool_str52[sizeof("pshortstring")];
      char stringpool_str53[sizeof("pvariant")];
      char stringpool_str54[sizeof("word")];
      char stringpool_str55[sizeof("pbytearray")];
      char stringpool_str56[sizeof("pdword")];
      char stringpool_str57[sizeof("pstring")];
      char stringpool_str58[sizeof("pinteger")];
      char stringpool_str59[sizeof("pcurrency")];
      char stringpool_str60[sizeof("int64")];
      char stringpool_str61[sizeof("string")];
      char stringpool_str63[sizeof("shortint")];
      char stringpool_str64[sizeof("psmallint")];
      char stringpool_str65[sizeof("widestring")];
      char stringpool_str66[sizeof("shortstring")];
      char stringpool_str67[sizeof("phandle")];
      char stringpool_str69[sizeof("bool")];
      char stringpool_str70[sizeof("pbool")];
      char stringpool_str72[sizeof("thandle")];
      char stringpool_str73[sizeof("pboolean")];
      char stringpool_str74[sizeof("byte")];
      char stringpool_str75[sizeof("int32")];
      char stringpool_str77[sizeof("boolean")];
      char stringpool_str78[sizeof("smallint")];
      char stringpool_str79[sizeof("pcardinal")];
      char stringpool_str80[sizeof("ttime")];
      char stringpool_str83[sizeof("textfile")];
      char stringpool_str84[sizeof("comp")];
      char stringpool_str88[sizeof("plongint")];
      char stringpool_str89[sizeof("plongword")];
      char stringpool_str90[sizeof("int16")];
      char stringpool_str91[sizeof("double")];
      char stringpool_str93[sizeof("wordbool")];
      char stringpool_str94[sizeof("char")];
      char stringpool_str95[sizeof("uint8")];
      char stringpool_str96[sizeof("uint64")];
      char stringpool_str97[sizeof("psingle")];
      char stringpool_str98[sizeof("iunknown")];
      char stringpool_str99[sizeof("nativeint")];
      char stringpool_str100[sizeof("pchar")];
      char stringpool_str101[sizeof("uint32")];
      char stringpool_str103[sizeof("widechar")];
      char stringpool_str104[sizeof("pansichar")];
      char stringpool_str105[sizeof("ansistring")];
      char stringpool_str106[sizeof("pansistring")];
      char stringpool_str107[sizeof("pdouble")];
      char stringpool_str108[sizeof("currency")];
      char stringpool_str109[sizeof("file")];
      char stringpool_str113[sizeof("bytebool")];
      char stringpool_str115[sizeof("pcomp")];
      char stringpool_str116[sizeof("tclass")];
      char stringpool_str118[sizeof("longword")];
      char stringpool_str125[sizeof("array")];
      char stringpool_str132[sizeof("integer")];
      char stringpool_str133[sizeof("fixedint")];
      char stringpool_str136[sizeof("single")];
      char stringpool_str141[sizeof("uint16")];
      char stringpool_str142[sizeof("variant")];
      char stringpool_str143[sizeof("ansichar")];
      char stringpool_str153[sizeof("longbool")];
      char stringpool_str157[sizeof("longint")];
      char stringpool_str163[sizeof("cardinal")];
      char stringpool_str168[sizeof("ucs4char")];
      char stringpool_str178[sizeof("ucs2char")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "real",
      "set",
      "text",
      "extended",
      "pextended",
      "pword",
      "real48",
      "pwordbool",
      "pwordarray",
      "pwidechar",
      "pdate",
      "pwidestring",
      "pdatetime",
      "tdate",
      "record",
      "pointer",
      "tdatetime",
      "dword",
      "tobject",
      "ppointer",
      "pshortint",
      "pbyte",
      "pint64",
      "pshortstring",
      "pvariant",
      "word",
      "pbytearray",
      "pdword",
      "pstring",
      "pinteger",
      "pcurrency",
      "int64",
      "string",
      "shortint",
      "psmallint",
      "widestring",
      "shortstring",
      "phandle",
      "bool",
      "pbool",
      "thandle",
      "pboolean",
      "byte",
      "int32",
      "boolean",
      "smallint",
      "pcardinal",
      "ttime",
      "textfile",
      "comp",
      "plongint",
      "plongword",
      "int16",
      "double",
      "wordbool",
      "char",
      "uint8",
      "uint64",
      "psingle",
      "iunknown",
      "nativeint",
      "pchar",
      "uint32",
      "widechar",
      "pansichar",
      "ansistring",
      "pansistring",
      "pdouble",
      "currency",
      "file",
      "bytebool",
      "pcomp",
      "tclass",
      "longword",
      "array",
      "integer",
      "fixedint",
      "single",
      "uint16",
      "variant",
      "ansichar",
      "longbool",
      "longint",
      "cardinal",
      "ucs4char",
      "ucs2char"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str141,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str157,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178
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

              if ((((unsigned char)*str ^ (unsigned char)*s) & ~32) == 0 && !gperf_case_strncmp (str, s, len) && s[len] == '\0')
                return s;
            }
        }
    }
  return 0;
}
