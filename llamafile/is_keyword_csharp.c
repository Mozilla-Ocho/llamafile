/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_csharp.c llamafile/is_keyword_csharp.gperf  */
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

#define TOTAL_KEYWORDS 118
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 10
#define MIN_HASH_VALUE 6
#define MAX_HASH_VALUE 272
/* maximum key range = 267, duplicates = 0 */

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
  static const unsigned short asso_values[] =
    {
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273,  15,  65,  40,
       35,   0,  40,  54, 105,  10,  10,   5,  20,  80,
        0,  25,  55,  60,  55,   5,   0,  30,  90, 107,
        0,  90,   0, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273
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
  struct stringpool_t
    {
      char stringpool_str6[sizeof("extern")];
      char stringpool_str8[sizeof("set")];
      char stringpool_str12[sizeof("in")];
      char stringpool_str13[sizeof("int")];
      char stringpool_str14[sizeof("nint")];
      char stringpool_str17[sizeof("is")];
      char stringpool_str18[sizeof("internal")];
      char stringpool_str19[sizeof("interface")];
      char stringpool_str21[sizeof("sizeof")];
      char stringpool_str22[sizeof("as")];
      char stringpool_str23[sizeof("let")];
      char stringpool_str24[sizeof("init")];
      char stringpool_str26[sizeof("static")];
      char stringpool_str27[sizeof("on")];
      char stringpool_str28[sizeof("not")];
      char stringpool_str29[sizeof("else")];
      char stringpool_str31[sizeof("select")];
      char stringpool_str32[sizeof("notnull")];
      char stringpool_str39[sizeof("into")];
      char stringpool_str44[sizeof("uint")];
      char stringpool_str45[sizeof("nuint")];
      char stringpool_str46[sizeof("sealed")];
      char stringpool_str49[sizeof("join")];
      char stringpool_str50[sizeof("using")];
      char stringpool_str52[sizeof("if")];
      char stringpool_str53[sizeof("and")];
      char stringpool_str55[sizeof("fixed")];
      char stringpool_str56[sizeof("unsafe")];
      char stringpool_str57[sizeof("get")];
      char stringpool_str58[sizeof("out")];
      char stringpool_str62[sizeof("do")];
      char stringpool_str63[sizeof("delegate")];
      char stringpool_str64[sizeof("case")];
      char stringpool_str65[sizeof("alias")];
      char stringpool_str69[sizeof("ascending")];
      char stringpool_str70[sizeof("stackalloc")];
      char stringpool_str72[sizeof("finally")];
      char stringpool_str73[sizeof("continue")];
      char stringpool_str74[sizeof("file")];
      char stringpool_str75[sizeof("const")];
      char stringpool_str76[sizeof("string")];
      char stringpool_str80[sizeof("ulong")];
      char stringpool_str82[sizeof("or")];
      char stringpool_str83[sizeof("explicit")];
      char stringpool_str85[sizeof("class")];
      char stringpool_str86[sizeof("allows")];
      char stringpool_str88[sizeof("add")];
      char stringpool_str89[sizeof("base")];
      char stringpool_str90[sizeof("descending")];
      char stringpool_str91[sizeof("return")];
      char stringpool_str92[sizeof("decimal")];
      char stringpool_str93[sizeof("abstract")];
      char stringpool_str94[sizeof("lock")];
      char stringpool_str95[sizeof("event")];
      char stringpool_str96[sizeof("struct")];
      char stringpool_str97[sizeof("default")];
      char stringpool_str98[sizeof("ref")];
      char stringpool_str100[sizeof("catch")];
      char stringpool_str101[sizeof("nameof")];
      char stringpool_str103[sizeof("long")];
      char stringpool_str104[sizeof("namespace")];
      char stringpool_str105[sizeof("float")];
      char stringpool_str106[sizeof("object")];
      char stringpool_str108[sizeof("goto")];
      char stringpool_str110[sizeof("new")];
      char stringpool_str111[sizeof("equals")];
      char stringpool_str113[sizeof("readonly")];
      char stringpool_str114[sizeof("enum")];
      char stringpool_str115[sizeof("async")];
      char stringpool_str117[sizeof("managed")];
      char stringpool_str122[sizeof("orderby")];
      char stringpool_str123[sizeof("for")];
      char stringpool_str124[sizeof("this")];
      char stringpool_str125[sizeof("yield")];
      char stringpool_str126[sizeof("record")];
      char stringpool_str127[sizeof("foreach")];
      char stringpool_str128[sizeof("switch")];
      char stringpool_str131[sizeof("scoped")];
      char stringpool_str132[sizeof("partial")];
      char stringpool_str133[sizeof("args")];
      char stringpool_str134[sizeof("unmanaged")];
      char stringpool_str139[sizeof("bool")];
      char stringpool_str140[sizeof("break")];
      char stringpool_str143[sizeof("operator")];
      char stringpool_str144[sizeof("protected")];
      char stringpool_str146[sizeof("params")];
      char stringpool_str147[sizeof("dynamic")];
      char stringpool_str148[sizeof("try")];
      char stringpool_str151[sizeof("typeof")];
      char stringpool_str152[sizeof("await")];
      char stringpool_str153[sizeof("required")];
      char stringpool_str157[sizeof("by")];
      char stringpool_str158[sizeof("volatile")];
      char stringpool_str159[sizeof("byte")];
      char stringpool_str160[sizeof("value")];
      char stringpool_str161[sizeof("double")];
      char stringpool_str162[sizeof("virtual")];
      char stringpool_str163[sizeof("var")];
      char stringpool_str164[sizeof("void")];
      char stringpool_str165[sizeof("sbyte")];
      char stringpool_str166[sizeof("remove")];
      char stringpool_str169[sizeof("group")];
      char stringpool_str170[sizeof("global")];
      char stringpool_str171[sizeof("ushort")];
      char stringpool_str173[sizeof("implicit")];
      char stringpool_str176[sizeof("public")];
      char stringpool_str178[sizeof("override")];
      char stringpool_str184[sizeof("unchecked")];
      char stringpool_str190[sizeof("throw")];
      char stringpool_str192[sizeof("checked")];
      char stringpool_str195[sizeof("short")];
      char stringpool_str204[sizeof("from")];
      char stringpool_str216[sizeof("when")];
      char stringpool_str217[sizeof("private")];
      char stringpool_str219[sizeof("char")];
      char stringpool_str226[sizeof("with")];
      char stringpool_str247[sizeof("while")];
      char stringpool_str272[sizeof("where")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "extern",
      "set",
      "in",
      "int",
      "nint",
      "is",
      "internal",
      "interface",
      "sizeof",
      "as",
      "let",
      "init",
      "static",
      "on",
      "not",
      "else",
      "select",
      "notnull",
      "into",
      "uint",
      "nuint",
      "sealed",
      "join",
      "using",
      "if",
      "and",
      "fixed",
      "unsafe",
      "get",
      "out",
      "do",
      "delegate",
      "case",
      "alias",
      "ascending",
      "stackalloc",
      "finally",
      "continue",
      "file",
      "const",
      "string",
      "ulong",
      "or",
      "explicit",
      "class",
      "allows",
      "add",
      "base",
      "descending",
      "return",
      "decimal",
      "abstract",
      "lock",
      "event",
      "struct",
      "default",
      "ref",
      "catch",
      "nameof",
      "long",
      "namespace",
      "float",
      "object",
      "goto",
      "new",
      "equals",
      "readonly",
      "enum",
      "async",
      "managed",
      "orderby",
      "for",
      "this",
      "yield",
      "record",
      "foreach",
      "switch",
      "scoped",
      "partial",
      "args",
      "unmanaged",
      "bool",
      "break",
      "operator",
      "protected",
      "params",
      "dynamic",
      "try",
      "typeof",
      "await",
      "required",
      "by",
      "volatile",
      "byte",
      "value",
      "double",
      "virtual",
      "var",
      "void",
      "sbyte",
      "remove",
      "group",
      "global",
      "ushort",
      "implicit",
      "public",
      "override",
      "unchecked",
      "throw",
      "checked",
      "short",
      "from",
      "when",
      "private",
      "char",
      "with",
      "while",
      "where"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str157,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str184,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str192,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str195,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str204,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str217,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str226,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str247,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str272
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
