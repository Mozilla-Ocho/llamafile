/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_rust.c llamafile/is_keyword_rust.gperf  */
/* Computed positions: -k'1-3' */

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

#line 1 "llamafile/is_keyword_rust.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 52
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 8
#define MIN_HASH_VALUE 2
#define MAX_HASH_VALUE 85
/* maximum key range = 84, duplicates = 0 */

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
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 10, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86,  0, 15, 40,
      40, 20, 10, 15, 10, 15, 15, 86, 25,  0,
       5,  5, 10, 35,  0,  0,  0,  0, 10, 55,
       5, 20,  5, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
      86, 86, 86, 86, 86, 86, 86
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[2]+1];
      /*FALLTHROUGH*/
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
is_keyword_rust (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str2[sizeof("as")];
      char stringpool_str3[sizeof("mut")];
      char stringpool_str5[sizeof("match")];
      char stringpool_str6[sizeof("struct")];
      char stringpool_str8[sizeof("try")];
      char stringpool_str10[sizeof("async")];
      char stringpool_str11[sizeof("unsafe")];
      char stringpool_str12[sizeof("unsized")];
      char stringpool_str13[sizeof("use")];
      char stringpool_str14[sizeof("true")];
      char stringpool_str15[sizeof("false")];
      char stringpool_str17[sizeof("fn")];
      char stringpool_str18[sizeof("for")];
      char stringpool_str20[sizeof("trait")];
      char stringpool_str21[sizeof("static")];
      char stringpool_str22[sizeof("in")];
      char stringpool_str23[sizeof("abstract")];
      char stringpool_str24[sizeof("self")];
      char stringpool_str25[sizeof("union")];
      char stringpool_str26[sizeof("return")];
      char stringpool_str27[sizeof("if")];
      char stringpool_str28[sizeof("mod")];
      char stringpool_str29[sizeof("priv")];
      char stringpool_str30[sizeof("break")];
      char stringpool_str31[sizeof("extern")];
      char stringpool_str32[sizeof("virtual")];
      char stringpool_str33[sizeof("override")];
      char stringpool_str34[sizeof("Self")];
      char stringpool_str35[sizeof("final")];
      char stringpool_str38[sizeof("ref")];
      char stringpool_str39[sizeof("enum")];
      char stringpool_str40[sizeof("super")];
      char stringpool_str43[sizeof("box")];
      char stringpool_str44[sizeof("loop")];
      char stringpool_str45[sizeof("macro")];
      char stringpool_str47[sizeof("do")];
      char stringpool_str48[sizeof("let")];
      char stringpool_str49[sizeof("else")];
      char stringpool_str50[sizeof("yield")];
      char stringpool_str53[sizeof("pub")];
      char stringpool_str54[sizeof("impl")];
      char stringpool_str55[sizeof("const")];
      char stringpool_str58[sizeof("continue")];
      char stringpool_str59[sizeof("type")];
      char stringpool_str60[sizeof("crate")];
      char stringpool_str61[sizeof("typeof")];
      char stringpool_str64[sizeof("move")];
      char stringpool_str68[sizeof("dyn")];
      char stringpool_str75[sizeof("await")];
      char stringpool_str80[sizeof("where")];
      char stringpool_str81[sizeof("become")];
      char stringpool_str85[sizeof("while")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "as",
      "mut",
      "match",
      "struct",
      "try",
      "async",
      "unsafe",
      "unsized",
      "use",
      "true",
      "false",
      "fn",
      "for",
      "trait",
      "static",
      "in",
      "abstract",
      "self",
      "union",
      "return",
      "if",
      "mod",
      "priv",
      "break",
      "extern",
      "virtual",
      "override",
      "Self",
      "final",
      "ref",
      "enum",
      "super",
      "box",
      "loop",
      "macro",
      "do",
      "let",
      "else",
      "yield",
      "pub",
      "impl",
      "const",
      "continue",
      "type",
      "crate",
      "typeof",
      "move",
      "dyn",
      "await",
      "where",
      "become",
      "while"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85
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
