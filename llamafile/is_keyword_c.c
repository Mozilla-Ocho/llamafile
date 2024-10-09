/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_c.gperf  */
/* Computed positions: -k'2-3,$' */

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

#line 1 "llamafile/is_keyword_c.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 104
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 19
#define MIN_HASH_VALUE 10
#define MAX_HASH_VALUE 191
/* maximum key range = 182, duplicates = 0 */

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
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192,   0, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
       30, 192,  10, 192, 192, 192,   0, 192, 192, 192,
      192, 192, 192, 192, 192,  70,  35,  70,   0, 192,
        0,  60, 192,   0, 192, 192, 192, 192,   0, 192,
        5, 192, 192,   5,  70, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192,   5, 192,  60,  65,  70,
       75,   0,  15,  85,   0,  25, 192,  15,   5,  55,
       10,  60,  40, 192,  40,  35,  20,  40,   0,  10,
       10,  35,   5, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
      192, 192, 192, 192, 192, 192
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
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_c (register const char *str, register size_t len)
{
  static const char * const wordlist[] =
    {
      "", "", "", "", "", "", "", "", "",
      "",
      "#else",
      "_Decimal128",
      "", "", "",
      "__volatile",
      "", "",
      "__volatile__ ",
      "",
      "_Decimal64",
      "inline",
      "__FUNCTION__",
      "__extension__",
      "__label__",
      "#elif",
      "__null",
      "",
      "#elifdef",
      "#elifndef",
      "while",
      "#endif",
      "if",
      "__func__",
      "__PRETTY_FUNCTION__",
      "#line",
      "return",
      "",
      "__inline",
      "__imag__ ",
      "_Decimal32",
      "switch",
      "default",
      "#include",
      "else",
      "__inline__",
      "extern",
      "",
      "__typeof",
      "",
      "union",
      "sizeof",
      "alignof",
      "int",
      "__real__ ",
      "__signed__",
      "__real",
      "thread_local",
      "#if",
      "_Static_assert",
      "break",
      "#ifdef",
      "__restrict__",
      "restrict",
      "", "", "", "",
      "#include_next",
      "",
      "false",
      "#undef",
      "alignas",
      "volatile",
      "",
      "__restrict",
      "__attribute",
      "__asm__",
      "continue",
      "_Noreturn",
      "",
      "__alignof__",
      "#define",
      "__attribute__",
      "true",
      "short",
      "struct",
      "_BitInt",
      "_Thread_local",
      "__alignof",
      "float",
      "__complex__",
      "nullptr",
      "typeof_unqual",
      "__complex",
      "const",
      "typeof",
      "typedef",
      "_Alignof",
      "case",
      "_Imaginary",
      "",
      "__const",
      "__builtin_offsetof",
      "char",
      "_Bool",
      "double",
      "",
      "__thread",
      "enum",
      "", "", "",
      "static_assert",
      "", "", "", "",
      "_Alignas",
      "constexpr",
      "",
      "__imag",
      "do",
      "__signed",
      "auto",
      "__asm",
      "", "",
      "unsigned",
      "bool",
      "", "", "",
      "register",
      "", "",
      "#embed",
      "",
      "_Generic",
      "", "", "", "",
      "for",
      "goto",
      "", "", "",
      "_Complex",
      "", "", "", "", "", "", "",
      "static",
      "", "",
      "long",
      "", "", "",
      "#warning",
      "void",
      "", "",
      "_Atomic",
      "", "", "",
      "__builtin_va_arg",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "",
      "signed"
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
