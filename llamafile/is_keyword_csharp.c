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
  struct stringpool_t
    {
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
      char stringpool_str38[sizeof("nameof")];
      char stringpool_str39[sizeof("into")];
      char stringpool_str41[sizeof("namespace")];
      char stringpool_str44[sizeof("uint")];
      char stringpool_str45[sizeof("nuint")];
      char stringpool_str46[sizeof("sealed")];
      char stringpool_str49[sizeof("join")];
      char stringpool_str50[sizeof("using")];
      char stringpool_str51[sizeof("enum")];
      char stringpool_str53[sizeof("and")];
      char stringpool_str54[sizeof("managed")];
      char stringpool_str56[sizeof("unsafe")];
      char stringpool_str58[sizeof("out")];
      char stringpool_str60[sizeof("event")];
      char stringpool_str62[sizeof("do")];
      char stringpool_str63[sizeof("delegate")];
      char stringpool_str64[sizeof("case")];
      char stringpool_str65[sizeof("alias")];
      char stringpool_str66[sizeof("new")];
      char stringpool_str69[sizeof("ascending")];
      char stringpool_str70[sizeof("stackalloc")];
      char stringpool_str71[sizeof("unmanaged")];
      char stringpool_str73[sizeof("continue")];
      char stringpool_str74[sizeof("null")];
      char stringpool_str75[sizeof("const")];
      char stringpool_str76[sizeof("string")];
      char stringpool_str80[sizeof("ulong")];
      char stringpool_str82[sizeof("or")];
      char stringpool_str84[sizeof("switch")];
      char stringpool_str85[sizeof("class")];
      char stringpool_str86[sizeof("allows")];
      char stringpool_str87[sizeof("if")];
      char stringpool_str88[sizeof("add")];
      char stringpool_str89[sizeof("true")];
      char stringpool_str90[sizeof("descending")];
      char stringpool_str91[sizeof("return")];
      char stringpool_str92[sizeof("decimal")];
      char stringpool_str94[sizeof("base")];
      char stringpool_str96[sizeof("struct")];
      char stringpool_str98[sizeof("abstract")];
      char stringpool_str99[sizeof("this")];
      char stringpool_str100[sizeof("catch")];
      char stringpool_str101[sizeof("extern")];
      char stringpool_str103[sizeof("remove")];
      char stringpool_str104[sizeof("lock")];
      char stringpool_str105[sizeof("async")];
      char stringpool_str106[sizeof("equals")];
      char stringpool_str107[sizeof("finally")];
      char stringpool_str108[sizeof("await")];
      char stringpool_str109[sizeof("file")];
      char stringpool_str111[sizeof("object")];
      char stringpool_str113[sizeof("readonly")];
      char stringpool_str115[sizeof("yield")];
      char stringpool_str117[sizeof("get")];
      char stringpool_str120[sizeof("false")];
      char stringpool_str122[sizeof("orderby")];
      char stringpool_str123[sizeof("volatile")];
      char stringpool_str125[sizeof("value")];
      char stringpool_str126[sizeof("record")];
      char stringpool_str127[sizeof("virtual")];
      char stringpool_str128[sizeof("var")];
      char stringpool_str129[sizeof("void")];
      char stringpool_str132[sizeof("default")];
      char stringpool_str133[sizeof("ref")];
      char stringpool_str137[sizeof("dynamic")];
      char stringpool_str138[sizeof("try")];
      char stringpool_str140[sizeof("float")];
      char stringpool_str143[sizeof("override")];
      char stringpool_str144[sizeof("bool")];
      char stringpool_str145[sizeof("break")];
      char stringpool_str146[sizeof("ushort")];
      char stringpool_str147[sizeof("when")];
      char stringpool_str148[sizeof("required")];
      char stringpool_str150[sizeof("implicit")];
      char stringpool_str152[sizeof("by")];
      char stringpool_str154[sizeof("byte")];
      char stringpool_str157[sizeof("with")];
      char stringpool_str158[sizeof("for")];
      char stringpool_str159[sizeof("unchecked")];
      char stringpool_str160[sizeof("sbyte")];
      char stringpool_str162[sizeof("foreach")];
      char stringpool_str163[sizeof("long")];
      char stringpool_str165[sizeof("throw")];
      char stringpool_str166[sizeof("double")];
      char stringpool_str167[sizeof("checked")];
      char stringpool_str168[sizeof("goto")];
      char stringpool_str170[sizeof("short")];
      char stringpool_str171[sizeof("scoped")];
      char stringpool_str172[sizeof("partial")];
      char stringpool_str176[sizeof("from")];
      char stringpool_str178[sizeof("while")];
      char stringpool_str181[sizeof("typeof")];
      char stringpool_str183[sizeof("operator")];
      char stringpool_str184[sizeof("protected")];
      char stringpool_str185[sizeof("fixed")];
      char stringpool_str186[sizeof("params")];
      char stringpool_str193[sizeof("args")];
      char stringpool_str194[sizeof("char")];
      char stringpool_str203[sizeof("where")];
      char stringpool_str218[sizeof("explicit")];
      char stringpool_str221[sizeof("public")];
      char stringpool_str222[sizeof("private")];
      char stringpool_str229[sizeof("group")];
      char stringpool_str235[sizeof("global")];
    };
  static const struct stringpool_t stringpool_contents =
    {
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
      "nameof",
      "into",
      "namespace",
      "uint",
      "nuint",
      "sealed",
      "join",
      "using",
      "enum",
      "and",
      "managed",
      "unsafe",
      "out",
      "event",
      "do",
      "delegate",
      "case",
      "alias",
      "new",
      "ascending",
      "stackalloc",
      "unmanaged",
      "continue",
      "null",
      "const",
      "string",
      "ulong",
      "or",
      "switch",
      "class",
      "allows",
      "if",
      "add",
      "true",
      "descending",
      "return",
      "decimal",
      "base",
      "struct",
      "abstract",
      "this",
      "catch",
      "extern",
      "remove",
      "lock",
      "async",
      "equals",
      "finally",
      "await",
      "file",
      "object",
      "readonly",
      "yield",
      "get",
      "false",
      "orderby",
      "volatile",
      "value",
      "record",
      "virtual",
      "var",
      "void",
      "default",
      "ref",
      "dynamic",
      "try",
      "float",
      "override",
      "bool",
      "break",
      "ushort",
      "when",
      "required",
      "implicit",
      "by",
      "byte",
      "with",
      "for",
      "unchecked",
      "sbyte",
      "foreach",
      "long",
      "throw",
      "double",
      "checked",
      "goto",
      "short",
      "scoped",
      "partial",
      "from",
      "while",
      "typeof",
      "operator",
      "protected",
      "fixed",
      "params",
      "args",
      "char",
      "where",
      "explicit",
      "public",
      "private",
      "group",
      "global"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1,
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
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str157,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str181,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str184,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str185,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str194,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str203,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str221,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str222,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str235
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
