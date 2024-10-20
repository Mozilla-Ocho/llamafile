/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_lua_builtin.gperf  */
/* Computed positions: -k'1,4-5' */

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

#line 1 "llamafile/is_keyword_lua_builtin.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 59
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 14
#define MIN_HASH_VALUE 2
#define MAX_HASH_VALUE 112
/* maximum key range = 111, duplicates = 0 */

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
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113,  10, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113,   5, 113,  15,   1,  30,
        5,  15,   5,  35,  55,   5, 113,   0,  20,  45,
        0,   0,  25,   5,  25,  40,   0,  20,  60,  50,
        1, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113
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
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

const char *
is_keyword_lua_builtin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str2[sizeof("os")];
      char stringpool_str4[sizeof("next")];
      char stringpool_str7[sizeof("io")];
      char stringpool_str9[sizeof("__lt")];
      char stringpool_str11[sizeof("__bnot")];
      char stringpool_str12[sizeof("__bxor")];
      char stringpool_str13[sizeof("__concat")];
      char stringpool_str14[sizeof("__eq")];
      char stringpool_str15[sizeof("__mod")];
      char stringpool_str16[sizeof("__mode")];
      char stringpool_str17[sizeof("__index")];
      char stringpool_str19[sizeof("type")];
      char stringpool_str20[sizeof("__add")];
      char stringpool_str21[sizeof("__idiv")];
      char stringpool_str24[sizeof("__le")];
      char stringpool_str25[sizeof("__len")];
      char stringpool_str26[sizeof("__band")];
      char stringpool_str29[sizeof("load")];
      char stringpool_str30[sizeof("print")];
      char stringpool_str31[sizeof("__sub")];
      char stringpool_str32[sizeof("__close")];
      char stringpool_str33[sizeof("tostring")];
      char stringpool_str34[sizeof("utf8")];
      char stringpool_str35[sizeof("__bor")];
      char stringpool_str36[sizeof("dofile")];
      char stringpool_str38[sizeof("loadfile")];
      char stringpool_str39[sizeof("__gc")];
      char stringpool_str40[sizeof("table")];
      char stringpool_str41[sizeof("ipairs")];
      char stringpool_str42[sizeof("xpcall")];
      char stringpool_str45[sizeof("error")];
      char stringpool_str46[sizeof("__call")];
      char stringpool_str47[sizeof("package")];
      char stringpool_str50[sizeof("__mul")];
      char stringpool_str51[sizeof("string")];
      char stringpool_str53[sizeof("rawequal")];
      char stringpool_str54[sizeof("warn")];
      char stringpool_str55[sizeof("__unm")];
      char stringpool_str57[sizeof("require")];
      char stringpool_str59[sizeof("coroutine")];
      char stringpool_str60[sizeof("__pow")];
      char stringpool_str61[sizeof("assert")];
      char stringpool_str65[sizeof("debug")];
      char stringpool_str66[sizeof("rawlen")];
      char stringpool_str70[sizeof("pcall")];
      char stringpool_str71[sizeof("__name")];
      char stringpool_str73[sizeof("tonumber")];
      char stringpool_str75[sizeof("__div")];
      char stringpool_str79[sizeof("collectgarbage")];
      char stringpool_str80[sizeof("__newindex")];
      char stringpool_str81[sizeof("rawget")];
      char stringpool_str85[sizeof("__shl")];
      char stringpool_str86[sizeof("rawset")];
      char stringpool_str90[sizeof("__shr")];
      char stringpool_str91[sizeof("select")];
      char stringpool_str95[sizeof("pairs")];
      char stringpool_str104[sizeof("math")];
      char stringpool_str107[sizeof("getmetatable")];
      char stringpool_str112[sizeof("setmetatable")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "os",
      "next",
      "io",
      "__lt",
      "__bnot",
      "__bxor",
      "__concat",
      "__eq",
      "__mod",
      "__mode",
      "__index",
      "type",
      "__add",
      "__idiv",
      "__le",
      "__len",
      "__band",
      "load",
      "print",
      "__sub",
      "__close",
      "tostring",
      "utf8",
      "__bor",
      "dofile",
      "loadfile",
      "__gc",
      "table",
      "ipairs",
      "xpcall",
      "error",
      "__call",
      "package",
      "__mul",
      "string",
      "rawequal",
      "warn",
      "__unm",
      "require",
      "coroutine",
      "__pow",
      "assert",
      "debug",
      "rawlen",
      "pcall",
      "__name",
      "tonumber",
      "__div",
      "collectgarbage",
      "__newindex",
      "rawget",
      "__shl",
      "rawset",
      "__shr",
      "select",
      "pairs",
      "math",
      "getmetatable",
      "setmetatable"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112
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
