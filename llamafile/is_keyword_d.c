/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_d.gperf  */
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

#line 1 "llamafile/is_keyword_d.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 110
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 19
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 193
/* maximum key range = 191, duplicates = 0 */

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
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
        0, 194, 194, 194, 194, 194,  25,  85, 194, 194,
        0, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194,  30, 194,  25, 110,  15,
       35,   5,  20,  35,   5,   0, 194,  30,  60,  75,
       35,   5,   0, 194,  85,   5,   0,  65,  30,  50,
       20,  40,  10, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194, 194, 194, 194, 194,
      194, 194, 194, 194, 194, 194
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
is_keyword_d (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("int")];
      char stringpool_str6[sizeof("import")];
      char stringpool_str7[sizeof("is")];
      char stringpool_str8[sizeof("out")];
      char stringpool_str9[sizeof("this")];
      char stringpool_str10[sizeof("inout")];
      char stringpool_str11[sizeof("export")];
      char stringpool_str12[sizeof("private")];
      char stringpool_str14[sizeof("interface")];
      char stringpool_str15[sizeof("short")];
      char stringpool_str16[sizeof("switch")];
      char stringpool_str17[sizeof("idouble")];
      char stringpool_str19[sizeof("else")];
      char stringpool_str20[sizeof("scope")];
      char stringpool_str22[sizeof("if")];
      char stringpool_str23[sizeof("override")];
      char stringpool_str24[sizeof("cast")];
      char stringpool_str25[sizeof("catch")];
      char stringpool_str26[sizeof("typeof")];
      char stringpool_str27[sizeof("package")];
      char stringpool_str29[sizeof("case")];
      char stringpool_str30[sizeof("float")];
      char stringpool_str32[sizeof("cdouble")];
      char stringpool_str34[sizeof("auto")];
      char stringpool_str35[sizeof("alias")];
      char stringpool_str36[sizeof("assert")];
      char stringpool_str37[sizeof("in")];
      char stringpool_str38[sizeof("abstract")];
      char stringpool_str39[sizeof("invariant")];
      char stringpool_str41[sizeof("typeid")];
      char stringpool_str42[sizeof("do")];
      char stringpool_str43[sizeof("__traits")];
      char stringpool_str44[sizeof("goto")];
      char stringpool_str46[sizeof("extern")];
      char stringpool_str47[sizeof("__parameters")];
      char stringpool_str49[sizeof("protected")];
      char stringpool_str50[sizeof("class")];
      char stringpool_str51[sizeof("static")];
      char stringpool_str54[sizeof("cent")];
      char stringpool_str55[sizeof("const")];
      char stringpool_str56[sizeof("pragma")];
      char stringpool_str59[sizeof("with")];
      char stringpool_str60[sizeof("while")];
      char stringpool_str62[sizeof("default")];
      char stringpool_str63[sizeof("continue")];
      char stringpool_str65[sizeof("align")];
      char stringpool_str66[sizeof("ifloat")];
      char stringpool_str68[sizeof("__FILE__")];
      char stringpool_str69[sizeof("void")];
      char stringpool_str70[sizeof("ireal")];
      char stringpool_str71[sizeof("shared")];
      char stringpool_str72[sizeof("__FUNCTION__")];
      char stringpool_str73[sizeof("unittest")];
      char stringpool_str74[sizeof("true")];
      char stringpool_str75[sizeof("ucent")];
      char stringpool_str76[sizeof("ushort")];
      char stringpool_str78[sizeof("__FILE_FULL_PATH__")];
      char stringpool_str79[sizeof("__PRETTY_FUNCTION__")];
      char stringpool_str80[sizeof("deprecated")];
      char stringpool_str81[sizeof("cfloat")];
      char stringpool_str83[sizeof("try")];
      char stringpool_str85[sizeof("creal")];
      char stringpool_str87[sizeof("synchronized")];
      char stringpool_str88[sizeof("template")];
      char stringpool_str89[sizeof("immutable")];
      char stringpool_str90[sizeof("false")];
      char stringpool_str92[sizeof("nothrow")];
      char stringpool_str93[sizeof("__LINE__")];
      char stringpool_str94[sizeof("pure")];
      char stringpool_str95[sizeof("super")];
      char stringpool_str96[sizeof("struct")];
      char stringpool_str98[sizeof("function")];
      char stringpool_str100[sizeof("macro")];
      char stringpool_str102[sizeof("finally")];
      char stringpool_str104[sizeof("uint")];
      char stringpool_str105[sizeof("union")];
      char stringpool_str106[sizeof("delete")];
      char stringpool_str108[sizeof("delegate")];
      char stringpool_str109[sizeof("__gshared")];
      char stringpool_str110[sizeof("ulong")];
      char stringpool_str111[sizeof("double")];
      char stringpool_str114[sizeof("lazy")];
      char stringpool_str115[sizeof("ubyte")];
      char stringpool_str117[sizeof("foreach")];
      char stringpool_str119[sizeof("byte")];
      char stringpool_str120[sizeof("final")];
      char stringpool_str121[sizeof("module")];
      char stringpool_str125[sizeof("foreach_reverse")];
      char stringpool_str126[sizeof("return")];
      char stringpool_str128[sizeof("ref")];
      char stringpool_str129[sizeof("char")];
      char stringpool_str130[sizeof("dchar")];
      char stringpool_str131[sizeof("public")];
      char stringpool_str134[sizeof("long")];
      char stringpool_str135[sizeof("mixin")];
      char stringpool_str138[sizeof("new")];
      char stringpool_str140[sizeof("throw")];
      char stringpool_str145[sizeof("wchar")];
      char stringpool_str149[sizeof("enum")];
      char stringpool_str150[sizeof("break")];
      char stringpool_str153[sizeof("__vector")];
      char stringpool_str155[sizeof("__MODULE__")];
      char stringpool_str157[sizeof("version")];
      char stringpool_str159[sizeof("null")];
      char stringpool_str174[sizeof("real")];
      char stringpool_str178[sizeof("asm")];
      char stringpool_str179[sizeof("bool")];
      char stringpool_str185[sizeof("debug")];
      char stringpool_str189[sizeof("body")];
      char stringpool_str193[sizeof("for")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "int",
      "import",
      "is",
      "out",
      "this",
      "inout",
      "export",
      "private",
      "interface",
      "short",
      "switch",
      "idouble",
      "else",
      "scope",
      "if",
      "override",
      "cast",
      "catch",
      "typeof",
      "package",
      "case",
      "float",
      "cdouble",
      "auto",
      "alias",
      "assert",
      "in",
      "abstract",
      "invariant",
      "typeid",
      "do",
      "__traits",
      "goto",
      "extern",
      "__parameters",
      "protected",
      "class",
      "static",
      "cent",
      "const",
      "pragma",
      "with",
      "while",
      "default",
      "continue",
      "align",
      "ifloat",
      "__FILE__",
      "void",
      "ireal",
      "shared",
      "__FUNCTION__",
      "unittest",
      "true",
      "ucent",
      "ushort",
      "__FILE_FULL_PATH__",
      "__PRETTY_FUNCTION__",
      "deprecated",
      "cfloat",
      "try",
      "creal",
      "synchronized",
      "template",
      "immutable",
      "false",
      "nothrow",
      "__LINE__",
      "pure",
      "super",
      "struct",
      "function",
      "macro",
      "finally",
      "uint",
      "union",
      "delete",
      "delegate",
      "__gshared",
      "ulong",
      "double",
      "lazy",
      "ubyte",
      "foreach",
      "byte",
      "final",
      "module",
      "foreach_reverse",
      "return",
      "ref",
      "char",
      "dchar",
      "public",
      "long",
      "mixin",
      "new",
      "throw",
      "wchar",
      "enum",
      "break",
      "__vector",
      "__MODULE__",
      "version",
      "null",
      "real",
      "asm",
      "bool",
      "debug",
      "body",
      "for"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str157,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str185,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str189,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193
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
