/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_ruby.c llamafile/is_keyword_ruby.gperf  */
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

#line 1 "llamafile/is_keyword_ruby.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 36
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 8
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 60
/* maximum key range = 58, duplicates = 0 */

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
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 15, 61, 61, 61, 30,  0, 61,
       0, 61, 61, 61, 61, 61, 61, 61, 25, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 30, 30,
       0,  5,  5, 15, 61, 61,  5, 20, 25, 61,
      15,  0, 10, 61,  5,  0, 15, 15,  0, 61,
      10, 40, 35, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
      61, 61, 61, 61, 61, 61, 61
    };
  return len + asso_values[(unsigned char)str[len - 1]] + asso_values[(unsigned char)str[0]+1];
}

const char *
is_keyword_ruby (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("END")];
      char stringpool_str4[sizeof("redo")];
      char stringpool_str5[sizeof("class")];
      char stringpool_str6[sizeof("unless")];
      char stringpool_str7[sizeof("do")];
      char stringpool_str8[sizeof("end")];
      char stringpool_str9[sizeof("case")];
      char stringpool_str10[sizeof("undef")];
      char stringpool_str11[sizeof("rescue")];
      char stringpool_str12[sizeof("if")];
      char stringpool_str13[sizeof("def")];
      char stringpool_str14[sizeof("else")];
      char stringpool_str15[sizeof("elsif")];
      char stringpool_str16[sizeof("ensure")];
      char stringpool_str17[sizeof("or")];
      char stringpool_str18[sizeof("not")];
      char stringpool_str19[sizeof("next")];
      char stringpool_str20[sizeof("while")];
      char stringpool_str21[sizeof("return")];
      char stringpool_str22[sizeof("in")];
      char stringpool_str23[sizeof("for")];
      char stringpool_str24[sizeof("self")];
      char stringpool_str25[sizeof("super")];
      char stringpool_str26[sizeof("module")];
      char stringpool_str28[sizeof("defined?")];
      char stringpool_str29[sizeof("when")];
      char stringpool_str30[sizeof("until")];
      char stringpool_str33[sizeof("and")];
      char stringpool_str34[sizeof("then")];
      char stringpool_str35[sizeof("alias")];
      char stringpool_str40[sizeof("yield")];
      char stringpool_str44[sizeof("fail")];
      char stringpool_str45[sizeof("retry")];
      char stringpool_str50[sizeof("begin")];
      char stringpool_str55[sizeof("break")];
      char stringpool_str60[sizeof("BEGIN")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "END",
      "redo",
      "class",
      "unless",
      "do",
      "end",
      "case",
      "undef",
      "rescue",
      "if",
      "def",
      "else",
      "elsif",
      "ensure",
      "or",
      "not",
      "next",
      "while",
      "return",
      "in",
      "for",
      "self",
      "super",
      "module",
      "defined?",
      "when",
      "until",
      "and",
      "then",
      "alias",
      "yield",
      "fail",
      "retry",
      "begin",
      "break",
      "BEGIN"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60
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
