/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_kotlin.c llamafile/is_keyword_kotlin.gperf  */
/* Computed positions: -k'1,3,$' */

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

#line 1 "llamafile/is_keyword_kotlin.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 72
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 11
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 125
/* maximum key range = 123, duplicates = 0 */

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
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126,  50,  20,   5,
       35,   5,  65,  40,  30,  15,  20,   5,  50,  55,
        5,  35,  20, 126,   0,  25,   0,   0,   0,   5,
        5,  45, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_kotlin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("var")];
      char stringpool_str10[sizeof("throw")];
      char stringpool_str11[sizeof("return")];
      char stringpool_str13[sizeof("receiver")];
      char stringpool_str15[sizeof("const")];
      char stringpool_str17[sizeof("it")];
      char stringpool_str19[sizeof("when")];
      char stringpool_str20[sizeof("where")];
      char stringpool_str21[sizeof("constructor")];
      char stringpool_str22[sizeof("in")];
      char stringpool_str23[sizeof("continue")];
      char stringpool_str25[sizeof("inner")];
      char stringpool_str27[sizeof("tailrec")];
      char stringpool_str28[sizeof("set")];
      char stringpool_str29[sizeof("interface")];
      char stringpool_str30[sizeof("while")];
      char stringpool_str31[sizeof("expect")];
      char stringpool_str33[sizeof("noinline")];
      char stringpool_str34[sizeof("init")];
      char stringpool_str35[sizeof("break")];
      char stringpool_str37[sizeof("package")];
      char stringpool_str38[sizeof("out")];
      char stringpool_str39[sizeof("else")];
      char stringpool_str40[sizeof("catch")];
      char stringpool_str41[sizeof("import")];
      char stringpool_str42[sizeof("is")];
      char stringpool_str43[sizeof("get")];
      char stringpool_str44[sizeof("this")];
      char stringpool_str46[sizeof("vararg")];
      char stringpool_str47[sizeof("private")];
      char stringpool_str48[sizeof("operator")];
      char stringpool_str49[sizeof("open")];
      char stringpool_str50[sizeof("super")];
      char stringpool_str51[sizeof("public")];
      char stringpool_str52[sizeof("dynamic")];
      char stringpool_str53[sizeof("override")];
      char stringpool_str54[sizeof("typealias")];
      char stringpool_str56[sizeof("crossinline")];
      char stringpool_str57[sizeof("reified")];
      char stringpool_str58[sizeof("lateinit")];
      char stringpool_str61[sizeof("object")];
      char stringpool_str63[sizeof("external")];
      char stringpool_str64[sizeof("enum")];
      char stringpool_str67[sizeof("by")];
      char stringpool_str68[sizeof("for")];
      char stringpool_str70[sizeof("annotation")];
      char stringpool_str72[sizeof("do")];
      char stringpool_str73[sizeof("internal")];
      char stringpool_str74[sizeof("companion")];
      char stringpool_str76[sizeof("inline")];
      char stringpool_str77[sizeof("as")];
      char stringpool_str78[sizeof("fun")];
      char stringpool_str80[sizeof("param")];
      char stringpool_str82[sizeof("if")];
      char stringpool_str83[sizeof("abstract")];
      char stringpool_str85[sizeof("class")];
      char stringpool_str88[sizeof("setparam")];
      char stringpool_str89[sizeof("data")];
      char stringpool_str90[sizeof("infix")];
      char stringpool_str91[sizeof("typeof")];
      char stringpool_str92[sizeof("suspend")];
      char stringpool_str93[sizeof("try")];
      char stringpool_str98[sizeof("delegate")];
      char stringpool_str99[sizeof("protected")];
      char stringpool_str103[sizeof("val")];
      char stringpool_str106[sizeof("actual")];
      char stringpool_str108[sizeof("property")];
      char stringpool_str110[sizeof("field")];
      char stringpool_str116[sizeof("sealed")];
      char stringpool_str122[sizeof("finally")];
      char stringpool_str124[sizeof("file")];
      char stringpool_str125[sizeof("final")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "var",
      "throw",
      "return",
      "receiver",
      "const",
      "it",
      "when",
      "where",
      "constructor",
      "in",
      "continue",
      "inner",
      "tailrec",
      "set",
      "interface",
      "while",
      "expect",
      "noinline",
      "init",
      "break",
      "package",
      "out",
      "else",
      "catch",
      "import",
      "is",
      "get",
      "this",
      "vararg",
      "private",
      "operator",
      "open",
      "super",
      "public",
      "dynamic",
      "override",
      "typealias",
      "crossinline",
      "reified",
      "lateinit",
      "object",
      "external",
      "enum",
      "by",
      "for",
      "annotation",
      "do",
      "internal",
      "companion",
      "inline",
      "as",
      "fun",
      "param",
      "if",
      "abstract",
      "class",
      "setparam",
      "data",
      "infix",
      "typeof",
      "suspend",
      "try",
      "delegate",
      "protected",
      "val",
      "actual",
      "property",
      "field",
      "sealed",
      "finally",
      "file",
      "final"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125
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
