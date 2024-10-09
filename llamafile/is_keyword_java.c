/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf is_keyword_java.gperf  */
/* Computed positions: -k'1,3' */

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

#line 1 "is_keyword_java.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 50
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 12
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 80
/* maximum key range = 78, duplicates = 0 */

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
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81,  5,  5, 15,
      25, 25, 35, 15, 81, 15,  0, 81,  0,  5,
       0, 10,  0, 55,  5,  0, 30, 45,  0,  5,
       0, 81, 15, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
      81, 81, 81, 81, 81, 81, 81
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[2]+1];
      /*FALLTHROUGH*/
      case 2:
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

const char *
is_keyword_java (register const char *str, register size_t len)
{
  static const char * const wordlist[] =
    {
      "", "", "",
      "new",
      "void",
      "short",
      "switch",
      "private",
      "strictfp",
      "protected",
      "while",
      "static",
      "boolean",
      "volatile",
      "long",
      "", "",
      "if",
      "", "", "",
      "public",
      "synchronized",
      "",
      "char",
      "class",
      "",
      "do",
      "",
      "enum",
      "const",
      "double",
      "package",
      "continue",
      "this",
      "throw",
      "throws",
      "",
      "for",
      "",
      "float",
      "assert",
      "",
      "abstract",
      "transient",
      "break",
      "",
      "default",
      "try",
      "case",
      "final",
      "native",
      "finally",
      "",
      "byte",
      "instanceof",
      "return",
      "", "",
      "else",
      "super",
      "", "",
      "int",
      "goto",
      "catch",
      "", "", "",
      "interface",
      "", "", "", "", "", "",
      "import",
      "extends",
      "", "",
      "implements"
    };

  if (len <= MAX_WORD_LENGTH && len >= MIN_WORD_LENGTH)
    {
      register unsigned int key = hash (str, len);

      if (key <= MAX_HASH_VALUE)
        {
          register const char *s = wordlist[key];

          if (*str == *s && !strncmp (str + 1, s + 1, len - 1) && s[len] == '\0')
            return s;
        }
    }
  return 0;
}
