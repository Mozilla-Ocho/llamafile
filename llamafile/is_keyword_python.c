/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_python.c llamafile/is_keyword_python.gperf  */
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

#line 1 "llamafile/is_keyword_python.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 35
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 8
#define MIN_HASH_VALUE 2
#define MAX_HASH_VALUE 53
/* maximum key range = 52, duplicates = 0 */

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
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      15, 54, 54, 54, 54, 54, 54, 54, 25, 54,
      54, 54, 54, 54, 20, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 10,  0, 25,
       0,  5,  0,  0,  5,  0, 54,  0, 30,  0,
      15,  5,  0, 54, 20, 20,  0, 54, 54, 10,
      54,  5, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54
    };
  return len + asso_values[(unsigned char)str[len - 1]] + asso_values[(unsigned char)str[0]];
}

const char *
is_keyword_python (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str2[sizeof("if")];
      char stringpool_str3[sizeof("def")];
      char stringpool_str4[sizeof("from")];
      char stringpool_str5[sizeof("break")];
      char stringpool_str6[sizeof("import")];
      char stringpool_str8[sizeof("try")];
      char stringpool_str9[sizeof("elif")];
      char stringpool_str10[sizeof("yield")];
      char stringpool_str11[sizeof("except")];
      char stringpool_str12[sizeof("finally")];
      char stringpool_str13[sizeof("and")];
      char stringpool_str14[sizeof("else")];
      char stringpool_str15[sizeof("await")];
      char stringpool_str16[sizeof("assert")];
      char stringpool_str17[sizeof("in")];
      char stringpool_str18[sizeof("not")];
      char stringpool_str19[sizeof("with")];
      char stringpool_str20[sizeof("while")];
      char stringpool_str22[sizeof("is")];
      char stringpool_str23[sizeof("for")];
      char stringpool_str24[sizeof("pass")];
      char stringpool_str25[sizeof("False")];
      char stringpool_str27[sizeof("or")];
      char stringpool_str29[sizeof("True")];
      char stringpool_str30[sizeof("raise")];
      char stringpool_str32[sizeof("as")];
      char stringpool_str33[sizeof("del")];
      char stringpool_str34[sizeof("None")];
      char stringpool_str36[sizeof("global")];
      char stringpool_str38[sizeof("continue")];
      char stringpool_str40[sizeof("async")];
      char stringpool_str41[sizeof("return")];
      char stringpool_str46[sizeof("lambda")];
      char stringpool_str50[sizeof("class")];
      char stringpool_str53[sizeof("nonlocal")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "if",
      "def",
      "from",
      "break",
      "import",
      "try",
      "elif",
      "yield",
      "except",
      "finally",
      "and",
      "else",
      "await",
      "assert",
      "in",
      "not",
      "with",
      "while",
      "is",
      "for",
      "pass",
      "False",
      "or",
      "True",
      "raise",
      "as",
      "del",
      "None",
      "global",
      "continue",
      "async",
      "return",
      "lambda",
      "class",
      "nonlocal"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53
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
