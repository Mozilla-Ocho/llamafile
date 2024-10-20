/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_csharp.gperf  */
/* Computed positions: -k'1-4' */

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

#line 1 "llamafile/is_keyword_csharp.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 121
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 10
#define MIN_HASH_VALUE 8
#define MAX_HASH_VALUE 235
/* maximum key range = 228, duplicates = 0 */

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
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236,  15,  70,  40,
       35,   0,  75, 114,  80,  10,  10,  15,  20,  17,
        0,  25,  95,  55,  55,   5,   0,  30,  55,  63,
       95,  80,   0, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
      236, 236, 236, 236, 236, 236
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
        hval += asso_values[(unsigned char)str[2]];
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
is_keyword_csharp (register const char *str, register size_t len)
{
  static const char * const wordlist[] =
    {
      "", "", "", "", "", "", "", "",
      "set",
      "", "", "",
      "in",
      "int",
      "nint",
      "", "",
      "is",
      "internal",
      "interface",
      "",
      "sizeof",
      "as",
      "let",
      "init",
      "",
      "static",
      "on",
      "not",
      "else",
      "",
      "select",
      "notnull",
      "", "", "", "", "",
      "nameof",
      "into",
      "",
      "namespace",
      "", "",
      "uint",
      "nuint",
      "sealed",
      "", "",
      "join",
      "using",
      "enum",
      "",
      "and",
      "managed",
      "",
      "unsafe",
      "",
      "out",
      "",
      "event",
      "",
      "do",
      "delegate",
      "case",
      "alias",
      "new",
      "", "",
      "ascending",
      "stackalloc",
      "unmanaged",
      "",
      "continue",
      "null",
      "const",
      "string",
      "", "", "",
      "ulong",
      "",
      "or",
      "",
      "switch",
      "class",
      "allows",
      "if",
      "add",
      "true",
      "descending",
      "return",
      "decimal",
      "",
      "base",
      "",
      "struct",
      "",
      "abstract",
      "this",
      "catch",
      "extern",
      "",
      "remove",
      "lock",
      "async",
      "equals",
      "finally",
      "await",
      "file",
      "",
      "object",
      "",
      "readonly",
      "",
      "yield",
      "",
      "get",
      "", "",
      "false",
      "",
      "orderby",
      "volatile",
      "",
      "value",
      "record",
      "virtual",
      "var",
      "void",
      "", "",
      "default",
      "ref",
      "", "", "",
      "dynamic",
      "try",
      "",
      "float",
      "", "",
      "override",
      "bool",
      "break",
      "ushort",
      "when",
      "required",
      "",
      "implicit",
      "",
      "by",
      "",
      "byte",
      "", "",
      "with",
      "for",
      "unchecked",
      "sbyte",
      "",
      "foreach",
      "long",
      "",
      "throw",
      "double",
      "checked",
      "goto",
      "",
      "short",
      "scoped",
      "partial",
      "", "", "",
      "from",
      "",
      "while",
      "", "",
      "typeof",
      "",
      "operator",
      "protected",
      "fixed",
      "params",
      "", "", "", "", "", "",
      "args",
      "char",
      "", "", "", "", "", "", "", "",
      "where",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "",
      "explicit",
      "", "",
      "public",
      "private",
      "", "", "", "", "", "",
      "group",
      "", "", "", "", "",
      "global"
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
