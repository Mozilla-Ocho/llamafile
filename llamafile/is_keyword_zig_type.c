/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_zig_type.c llamafile/is_keyword_zig_type.gperf  */
/* Computed positions: -k'1,$' */

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

#line 1 "llamafile/is_keyword_zig_type.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 52
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 14
#define MIN_HASH_VALUE 4
#define MAX_HASH_VALUE 88
/* maximum key range = 85, duplicates = 0 */

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
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 50, 89,
      45, 37, 35, 27, 25,  2, 15, 10, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 15, 30,  0,
      30, 45, 35, 55, 89,  5, 89, 89, 30, 89,
      35, 89, 89, 89, 35, 89,  0,  0, 25, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
      89, 89, 89, 89, 89, 89
    };
  return len + asso_values[(unsigned char)str[len - 1]] + asso_values[(unsigned char)str[0]];
}

const char *
is_keyword_zig_type (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str4[sizeof("u7")];
      char stringpool_str5[sizeof("c_int")];
      char stringpool_str6[sizeof("c_uint")];
      char stringpool_str7[sizeof("c_short")];
      char stringpool_str8[sizeof("c_ushort")];
      char stringpool_str9[sizeof("i7")];
      char stringpool_str12[sizeof("comptime_int")];
      char stringpool_str13[sizeof("u29")];
      char stringpool_str14[sizeof("comptime_float")];
      char stringpool_str17[sizeof("u8")];
      char stringpool_str18[sizeof("i29")];
      char stringpool_str19[sizeof("u128")];
      char stringpool_str22[sizeof("i8")];
      char stringpool_str24[sizeof("i128")];
      char stringpool_str27[sizeof("u6")];
      char stringpool_str28[sizeof("u16")];
      char stringpool_str29[sizeof("u5")];
      char stringpool_str32[sizeof("i6")];
      char stringpool_str33[sizeof("i16")];
      char stringpool_str34[sizeof("i5")];
      char stringpool_str37[sizeof("u4")];
      char stringpool_str38[sizeof("u64")];
      char stringpool_str39[sizeof("u3")];
      char stringpool_str41[sizeof("c_char")];
      char stringpool_str42[sizeof("i4")];
      char stringpool_str43[sizeof("i64")];
      char stringpool_str44[sizeof("i3")];
      char stringpool_str47[sizeof("u2")];
      char stringpool_str48[sizeof("u32")];
      char stringpool_str49[sizeof("type")];
      char stringpool_str50[sizeof("usize")];
      char stringpool_str52[sizeof("i2")];
      char stringpool_str53[sizeof("i32")];
      char stringpool_str54[sizeof("f128")];
      char stringpool_str55[sizeof("isize")];
      char stringpool_str57[sizeof("c_longdouble")];
      char stringpool_str58[sizeof("anyerror")];
      char stringpool_str59[sizeof("void")];
      char stringpool_str61[sizeof("c_long")];
      char stringpool_str62[sizeof("c_ulong")];
      char stringpool_str63[sizeof("f16")];
      char stringpool_str64[sizeof("bool")];
      char stringpool_str65[sizeof("c_longlong")];
      char stringpool_str66[sizeof("c_ulonglong")];
      char stringpool_str67[sizeof("anytype")];
      char stringpool_str68[sizeof("anyframe")];
      char stringpool_str69[sizeof("anyopaque")];
      char stringpool_str73[sizeof("f64")];
      char stringpool_str78[sizeof("noreturn")];
      char stringpool_str83[sizeof("f32")];
      char stringpool_str85[sizeof("error")];
      char stringpool_str88[sizeof("f80")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "u7",
      "c_int",
      "c_uint",
      "c_short",
      "c_ushort",
      "i7",
      "comptime_int",
      "u29",
      "comptime_float",
      "u8",
      "i29",
      "u128",
      "i8",
      "i128",
      "u6",
      "u16",
      "u5",
      "i6",
      "i16",
      "i5",
      "u4",
      "u64",
      "u3",
      "c_char",
      "i4",
      "i64",
      "i3",
      "u2",
      "u32",
      "type",
      "usize",
      "i2",
      "i32",
      "f128",
      "isize",
      "c_longdouble",
      "anyerror",
      "void",
      "c_long",
      "c_ulong",
      "f16",
      "bool",
      "c_longlong",
      "c_ulonglong",
      "anytype",
      "anyframe",
      "anyopaque",
      "f64",
      "noreturn",
      "f32",
      "error",
      "f80"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88
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
