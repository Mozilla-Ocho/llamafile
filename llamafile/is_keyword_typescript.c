/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_typescript.gperf  */
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

#line 1 "llamafile/is_keyword_typescript.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 55
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 10
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 99
/* maximum key range = 97, duplicates = 0 */

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
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100, 40,  0, 15,
      25,  5, 10, 10, 50, 15,100,  0,  0, 35,
      25, 50, 45,100, 20,  5,  0,100,  0, 45,
     100, 35,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100,100,100,100,100,
     100,100,100,100,100,100
    };
  return len + asso_values[(unsigned char)str[len - 1]] + asso_values[(unsigned char)str[0]];
}

const char *
is_keyword_typescript (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("let")];
      char stringpool_str5[sizeof("break")];
      char stringpool_str6[sizeof("target")];
      char stringpool_str8[sizeof("set")];
      char stringpool_str9[sizeof("type")];
      char stringpool_str11[sizeof("export")];
      char stringpool_str13[sizeof("get")];
      char stringpool_str14[sizeof("else")];
      char stringpool_str15[sizeof("keyof")];
      char stringpool_str16[sizeof("typeof")];
      char stringpool_str17[sizeof("extends")];
      char stringpool_str19[sizeof("satisfies")];
      char stringpool_str20[sizeof("const")];
      char stringpool_str21[sizeof("import")];
      char stringpool_str22[sizeof("is")];
      char stringpool_str23[sizeof("var")];
      char stringpool_str24[sizeof("case")];
      char stringpool_str25[sizeof("class")];
      char stringpool_str26[sizeof("static")];
      char stringpool_str27[sizeof("if")];
      char stringpool_str28[sizeof("continue")];
      char stringpool_str29[sizeof("interface")];
      char stringpool_str30[sizeof("implements")];
      char stringpool_str32[sizeof("default")];
      char stringpool_str33[sizeof("for")];
      char stringpool_str35[sizeof("instanceof")];
      char stringpool_str36[sizeof("delete")];
      char stringpool_str37[sizeof("declare")];
      char stringpool_str38[sizeof("try")];
      char stringpool_str39[sizeof("namespace")];
      char stringpool_str40[sizeof("infer")];
      char stringpool_str42[sizeof("in")];
      char stringpool_str43[sizeof("function")];
      char stringpool_str44[sizeof("enum")];
      char stringpool_str45[sizeof("await")];
      char stringpool_str47[sizeof("as")];
      char stringpool_str48[sizeof("abstract")];
      char stringpool_str49[sizeof("from")];
      char stringpool_str50[sizeof("throw")];
      char stringpool_str51[sizeof("return")];
      char stringpool_str52[sizeof("finally")];
      char stringpool_str53[sizeof("debugger")];
      char stringpool_str55[sizeof("while")];
      char stringpool_str57[sizeof("private")];
      char stringpool_str60[sizeof("async")];
      char stringpool_str61[sizeof("switch")];
      char stringpool_str62[sizeof("of")];
      char stringpool_str63[sizeof("readonly")];
      char stringpool_str65[sizeof("yield")];
      char stringpool_str66[sizeof("public")];
      char stringpool_str70[sizeof("catch")];
      char stringpool_str73[sizeof("new")];
      char stringpool_str77[sizeof("do")];
      char stringpool_str79[sizeof("protected")];
      char stringpool_str99[sizeof("with")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "let",
      "break",
      "target",
      "set",
      "type",
      "export",
      "get",
      "else",
      "keyof",
      "typeof",
      "extends",
      "satisfies",
      "const",
      "import",
      "is",
      "var",
      "case",
      "class",
      "static",
      "if",
      "continue",
      "interface",
      "implements",
      "default",
      "for",
      "instanceof",
      "delete",
      "declare",
      "try",
      "namespace",
      "infer",
      "in",
      "function",
      "enum",
      "await",
      "as",
      "abstract",
      "from",
      "throw",
      "return",
      "finally",
      "debugger",
      "while",
      "private",
      "async",
      "switch",
      "of",
      "readonly",
      "yield",
      "public",
      "catch",
      "new",
      "do",
      "protected",
      "with"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
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
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99
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
