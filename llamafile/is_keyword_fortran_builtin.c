/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_fortran_builtin.c llamafile/is_keyword_fortran_builtin.gperf  */
/* Computed positions: -k'1-2,4,$' */

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

#line 1 "llamafile/is_keyword_fortran_builtin.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 85
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 6
#define MIN_HASH_VALUE 4
#define MAX_HASH_VALUE 309
/* maximum key range = 306, duplicates = 0 */

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
  static const unsigned short asso_values[] =
    {
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310,   0,  45,
       20, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310,   5,  45,  70,   0,  25,
      105,   0,  95,  20,  30, 310,  25,  55,  10, 115,
       45,   5,  17,   0,   0, 120, 310, 310,  30,  20,
      310, 310, 310, 310, 310, 310, 310,   5,  45,  70,
        0,  25, 105,   0,  95,  20,  30, 310,  25,  55,
       10, 115,  45,   5,  17,   0,   0, 120, 310, 310,
       30,  20, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310, 310, 310, 310,
      310, 310, 310, 310, 310, 310, 310
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
      case 2:
        hval += asso_values[(unsigned char)str[1]+1];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_fortran_builtin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str4[sizeof("dcos")];
      char stringpool_str9[sizeof("acos")];
      char stringpool_str14[sizeof("dmod")];
      char stringpool_str15[sizeof("dsign")];
      char stringpool_str19[sizeof("amod")];
      char stringpool_str21[sizeof("sqrt")];
      char stringpool_str22[sizeof("dsqrt")];
      char stringpool_str24[sizeof("dsin")];
      char stringpool_str29[sizeof("asin")];
      char stringpool_str30[sizeof("amin0")];
      char stringpool_str34[sizeof("dint")];
      char stringpool_str35[sizeof("isign")];
      char stringpool_str39[sizeof("aint")];
      char stringpool_str43[sizeof("sin")];
      char stringpool_str44[sizeof("nint")];
      char stringpool_str45[sizeof("aimag")];
      char stringpool_str47[sizeof("ichar")];
      char stringpool_str49[sizeof("dabs")];
      char stringpool_str50[sizeof("amax0")];
      char stringpool_str54[sizeof("sign")];
      char stringpool_str58[sizeof("tan")];
      char stringpool_str59[sizeof("dlog")];
      char stringpool_str60[sizeof("idint")];
      char stringpool_str61[sizeof("dlog10")];
      char stringpool_str64[sizeof("alog")];
      char stringpool_str65[sizeof("datan")];
      char stringpool_str66[sizeof("alog10")];
      char stringpool_str69[sizeof("iabs")];
      char stringpool_str70[sizeof("dmin1")];
      char stringpool_str71[sizeof("idnint")];
      char stringpool_str73[sizeof("log")];
      char stringpool_str74[sizeof("ccos")];
      char stringpool_str75[sizeof("amin1")];
      char stringpool_str76[sizeof("datan2")];
      char stringpool_str78[sizeof("abs")];
      char stringpool_str80[sizeof("dasin")];
      char stringpool_str83[sizeof("llt")];
      char stringpool_str84[sizeof("ifix")];
      char stringpool_str88[sizeof("dim")];
      char stringpool_str89[sizeof("min0")];
      char stringpool_str90[sizeof("dmax1")];
      char stringpool_str92[sizeof("csqrt")];
      char stringpool_str93[sizeof("exp")];
      char stringpool_str94[sizeof("csin")];
      char stringpool_str95[sizeof("amax1")];
      char stringpool_str98[sizeof("min")];
      char stringpool_str100[sizeof("dcosh")];
      char stringpool_str103[sizeof("mod")];
      char stringpool_str104[sizeof("max0")];
      char stringpool_str108[sizeof("lle")];
      char stringpool_str110[sizeof("dsinh")];
      char stringpool_str118[sizeof("cos")];
      char stringpool_str119[sizeof("cabs")];
      char stringpool_str120[sizeof("log10")];
      char stringpool_str123[sizeof("lgt")];
      char stringpool_str124[sizeof("dble")];
      char stringpool_str125[sizeof("dprod")];
      char stringpool_str128[sizeof("char")];
      char stringpool_str129[sizeof("clog")];
      char stringpool_str130[sizeof("dnint")];
      char stringpool_str133[sizeof("max")];
      char stringpool_str135[sizeof("anint")];
      char stringpool_str138[sizeof("int")];
      char stringpool_str139[sizeof("ddim")];
      char stringpool_str140[sizeof("cmplx")];
      char stringpool_str143[sizeof("len")];
      char stringpool_str144[sizeof("dtan")];
      char stringpool_str148[sizeof("lge")];
      char stringpool_str149[sizeof("atan")];
      char stringpool_str150[sizeof("conjg")];
      char stringpool_str159[sizeof("idim")];
      char stringpool_str160[sizeof("atan2")];
      char stringpool_str165[sizeof("dacos")];
      char stringpool_str169[sizeof("sngl")];
      char stringpool_str170[sizeof("float")];
      char stringpool_str176[sizeof("real")];
      char stringpool_str179[sizeof("min1")];
      char stringpool_str194[sizeof("max1")];
      char stringpool_str195[sizeof("index")];
      char stringpool_str199[sizeof("dexp")];
      char stringpool_str224[sizeof("sinh")];
      char stringpool_str230[sizeof("dtanh")];
      char stringpool_str239[sizeof("tanh")];
      char stringpool_str269[sizeof("cexp")];
      char stringpool_str309[sizeof("cosh")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "dcos",
      "acos",
      "dmod",
      "dsign",
      "amod",
      "sqrt",
      "dsqrt",
      "dsin",
      "asin",
      "amin0",
      "dint",
      "isign",
      "aint",
      "sin",
      "nint",
      "aimag",
      "ichar",
      "dabs",
      "amax0",
      "sign",
      "tan",
      "dlog",
      "idint",
      "dlog10",
      "alog",
      "datan",
      "alog10",
      "iabs",
      "dmin1",
      "idnint",
      "log",
      "ccos",
      "amin1",
      "datan2",
      "abs",
      "dasin",
      "llt",
      "ifix",
      "dim",
      "min0",
      "dmax1",
      "csqrt",
      "exp",
      "csin",
      "amax1",
      "min",
      "dcosh",
      "mod",
      "max0",
      "lle",
      "dsinh",
      "cos",
      "cabs",
      "log10",
      "lgt",
      "dble",
      "dprod",
      "char",
      "clog",
      "dnint",
      "max",
      "anint",
      "int",
      "ddim",
      "cmplx",
      "len",
      "dtan",
      "lge",
      "atan",
      "conjg",
      "idim",
      "atan2",
      "dacos",
      "sngl",
      "float",
      "real",
      "min1",
      "max1",
      "index",
      "dexp",
      "sinh",
      "dtanh",
      "tanh",
      "cexp",
      "cosh"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str194,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str195,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str199,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str224,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str230,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str239,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str269,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str309
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
