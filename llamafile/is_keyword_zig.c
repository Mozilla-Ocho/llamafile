/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_zig.gperf  */
/* Computed positions: -k'2-3' */

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

#line 1 "llamafile/is_keyword_zig.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 49
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 14
#define MIN_HASH_VALUE 2
#define MAX_HASH_VALUE 81
/* maximum key range = 80, duplicates = 0 */

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
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 40, 60, 20,
      25, 30, 10, 82, 50, 15, 82, 82, 10, 40,
       5,  0, 35, 82,  0, 10,  5,  0, 82, 30,
      15, 25, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
      82, 82, 82, 82, 82, 82
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
        hval += asso_values[(unsigned char)str[1]];
        break;
    }
  return hval;
}

const char *
is_keyword_zig (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str2[sizeof("or")];
      char stringpool_str3[sizeof("for")];
      char stringpool_str5[sizeof("error")];
      char stringpool_str7[sizeof("fn")];
      char stringpool_str8[sizeof("errdefer")];
      char stringpool_str9[sizeof("enum")];
      char stringpool_str10[sizeof("const")];
      char stringpool_str11[sizeof("struct")];
      char stringpool_str12[sizeof("if")];
      char stringpool_str13[sizeof("continue")];
      char stringpool_str16[sizeof("unreachable")];
      char stringpool_str17[sizeof("suspend")];
      char stringpool_str18[sizeof("volatile")];
      char stringpool_str19[sizeof("nosuspend")];
      char stringpool_str21[sizeof("inline")];
      char stringpool_str23[sizeof("noinline")];
      char stringpool_str24[sizeof("else")];
      char stringpool_str25[sizeof("union")];
      char stringpool_str26[sizeof("extern")];
      char stringpool_str28[sizeof("try")];
      char stringpool_str29[sizeof("allowzero")];
      char stringpool_str30[sizeof("align")];
      char stringpool_str31[sizeof("linksection")];
      char stringpool_str33[sizeof("and")];
      char stringpool_str35[sizeof("break")];
      char stringpool_str36[sizeof("orelse")];
      char stringpool_str37[sizeof("anytype")];
      char stringpool_str38[sizeof("anyframe")];
      char stringpool_str39[sizeof("usingnamespace")];
      char stringpool_str40[sizeof("async")];
      char stringpool_str41[sizeof("return")];
      char stringpool_str43[sizeof("var")];
      char stringpool_str44[sizeof("test")];
      char stringpool_str45[sizeof("defer")];
      char stringpool_str46[sizeof("resume")];
      char stringpool_str47[sizeof("noalias")];
      char stringpool_str48[sizeof("comptime")];
      char stringpool_str50[sizeof("catch")];
      char stringpool_str51[sizeof("switch")];
      char stringpool_str53[sizeof("asm")];
      char stringpool_str56[sizeof("export")];
      char stringpool_str58[sizeof("callconv")];
      char stringpool_str59[sizeof("addrspace")];
      char stringpool_str61[sizeof("threadlocal")];
      char stringpool_str63[sizeof("pub")];
      char stringpool_str66[sizeof("packed")];
      char stringpool_str70[sizeof("while")];
      char stringpool_str75[sizeof("await")];
      char stringpool_str81[sizeof("opaque")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "or",
      "for",
      "error",
      "fn",
      "errdefer",
      "enum",
      "const",
      "struct",
      "if",
      "continue",
      "unreachable",
      "suspend",
      "volatile",
      "nosuspend",
      "inline",
      "noinline",
      "else",
      "union",
      "extern",
      "try",
      "allowzero",
      "align",
      "linksection",
      "and",
      "break",
      "orelse",
      "anytype",
      "anyframe",
      "usingnamespace",
      "async",
      "return",
      "var",
      "test",
      "defer",
      "resume",
      "noalias",
      "comptime",
      "catch",
      "switch",
      "asm",
      "export",
      "callconv",
      "addrspace",
      "threadlocal",
      "pub",
      "packed",
      "while",
      "await",
      "opaque"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81
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
