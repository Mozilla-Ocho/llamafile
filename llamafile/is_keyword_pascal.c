/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_pascal.c llamafile/is_keyword_pascal.gperf  */
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

#line 1 "llamafile/is_keyword_pascal.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 89
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 14
#define MIN_HASH_VALUE 5
#define MAX_HASH_VALUE 218
/* maximum key range = 214, duplicates = 0 */

#ifndef GPERF_DOWNCASE
#define GPERF_DOWNCASE 1
static unsigned char gperf_downcase[256] =
  {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
     30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
     45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
     60,  61,  62,  63,  64,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    122,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
    195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
    255
  };
#endif

#ifndef GPERF_CASE_STRNCMP
#define GPERF_CASE_STRNCMP 1
static int
gperf_case_strncmp (register const char *s1, register const char *s2, register size_t n)
{
  for (; n > 0;)
    {
      unsigned char c1 = gperf_downcase[(unsigned char)*s1++];
      unsigned char c2 = gperf_downcase[(unsigned char)*s2++];
      if (c1 != 0 && c1 == c2)
        {
          n--;
          continue;
        }
      return (int)c1 - (int)c2;
    }
  return 0;
}
#endif

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
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219,  10,  15,  30,  45,   0,
       65,  10,  25,   0, 219,   0,  60,  25,  35,  45,
       20, 219,   5,  30,   0,  80,  85,   0,   5,  55,
      219, 219, 219, 219, 219, 219, 219,  10,  15,  30,
       45,   0,  65,  10,  25,   0, 219,   0,  60,  25,
       35,  45,  20, 219,   5,  30,   0,  80,  85,   0,
        5,  55, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219, 219, 219, 219, 219,
      219, 219, 219, 219, 219, 219
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
is_keyword_pascal (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str5[sizeof("while")];
      char stringpool_str9[sizeof("interface")];
      char stringpool_str10[sizeof("raise")];
      char stringpool_str12[sizeof("rewrite")];
      char stringpool_str13[sizeof("get")];
      char stringpool_str16[sizeof("reintroduce")];
      char stringpool_str18[sizeof("xor")];
      char stringpool_str19[sizeof("threadvar")];
      char stringpool_str20[sizeof("break")];
      char stringpool_str23[sizeof("put")];
      char stringpool_str24[sizeof("type")];
      char stringpool_str25[sizeof("input")];
      char stringpool_str26[sizeof("export")];
      char stringpool_str27[sizeof("private")];
      char stringpool_str28[sizeof("register")];
      char stringpool_str29[sizeof("with")];
      char stringpool_str31[sizeof("repeat")];
      char stringpool_str32[sizeof("is")];
      char stringpool_str33[sizeof("set")];
      char stringpool_str34[sizeof("else")];
      char stringpool_str36[sizeof("except")];
      char stringpool_str37[sizeof("in")];
      char stringpool_str38[sizeof("not")];
      char stringpool_str39[sizeof("then")];
      char stringpool_str40[sizeof("reset")];
      char stringpool_str42[sizeof("as")];
      char stringpool_str43[sizeof("shr")];
      char stringpool_str45[sizeof("alias")];
      char stringpool_str47[sizeof("to")];
      char stringpool_str48[sizeof("abstract")];
      char stringpool_str49[sizeof("initialization")];
      char stringpool_str51[sizeof("output")];
      char stringpool_str52[sizeof("or")];
      char stringpool_str53[sizeof("override")];
      char stringpool_str54[sizeof("assembler")];
      char stringpool_str57[sizeof("exports")];
      char stringpool_str58[sizeof("operator")];
      char stringpool_str59[sizeof("goto")];
      char stringpool_str63[sizeof("asm")];
      char stringpool_str64[sizeof("case")];
      char stringpool_str65[sizeof("begin")];
      char stringpool_str66[sizeof("inline")];
      char stringpool_str67[sizeof("if")];
      char stringpool_str68[sizeof("external")];
      char stringpool_str69[sizeof("implementation")];
      char stringpool_str70[sizeof("const")];
      char stringpool_str71[sizeof("public")];
      char stringpool_str73[sizeof("continue")];
      char stringpool_str74[sizeof("procedure")];
      char stringpool_str75[sizeof("class")];
      char stringpool_str77[sizeof("nostackframe")];
      char stringpool_str78[sizeof("for")];
      char stringpool_str79[sizeof("inherited")];
      char stringpool_str81[sizeof("constructor")];
      char stringpool_str82[sizeof("on")];
      char stringpool_str84[sizeof("unit")];
      char stringpool_str86[sizeof("record")];
      char stringpool_str89[sizeof("published")];
      char stringpool_str90[sizeof("destructor")];
      char stringpool_str92[sizeof("do")];
      char stringpool_str93[sizeof("end")];
      char stringpool_str95[sizeof("cdecl")];
      char stringpool_str96[sizeof("downto")];
      char stringpool_str97[sizeof("program")];
      char stringpool_str98[sizeof("var")];
      char stringpool_str101[sizeof("packed")];
      char stringpool_str103[sizeof("and")];
      char stringpool_str104[sizeof("softfloat")];
      char stringpool_str112[sizeof("of")];
      char stringpool_str113[sizeof("try")];
      char stringpool_str114[sizeof("uses")];
      char stringpool_str116[sizeof("pascal")];
      char stringpool_str117[sizeof("dynamic")];
      char stringpool_str118[sizeof("mod")];
      char stringpool_str119[sizeof("protected")];
      char stringpool_str127[sizeof("varargs")];
      char stringpool_str128[sizeof("property")];
      char stringpool_str137[sizeof("library")];
      char stringpool_str140[sizeof("label")];
      char stringpool_str142[sizeof("stdcall")];
      char stringpool_str143[sizeof("function")];
      char stringpool_str145[sizeof("until")];
      char stringpool_str147[sizeof("finalization")];
      char stringpool_str153[sizeof("shl")];
      char stringpool_str157[sizeof("virtual")];
      char stringpool_str158[sizeof("nil")];
      char stringpool_str162[sizeof("finally")];
      char stringpool_str163[sizeof("safecall")];
      char stringpool_str218[sizeof("div")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "while",
      "interface",
      "raise",
      "rewrite",
      "get",
      "reintroduce",
      "xor",
      "threadvar",
      "break",
      "put",
      "type",
      "input",
      "export",
      "private",
      "register",
      "with",
      "repeat",
      "is",
      "set",
      "else",
      "except",
      "in",
      "not",
      "then",
      "reset",
      "as",
      "shr",
      "alias",
      "to",
      "abstract",
      "initialization",
      "output",
      "or",
      "override",
      "assembler",
      "exports",
      "operator",
      "goto",
      "asm",
      "case",
      "begin",
      "inline",
      "if",
      "external",
      "implementation",
      "const",
      "public",
      "continue",
      "procedure",
      "class",
      "nostackframe",
      "for",
      "inherited",
      "constructor",
      "on",
      "unit",
      "record",
      "published",
      "destructor",
      "do",
      "end",
      "cdecl",
      "downto",
      "program",
      "var",
      "packed",
      "and",
      "softfloat",
      "of",
      "try",
      "uses",
      "pascal",
      "dynamic",
      "mod",
      "protected",
      "varargs",
      "property",
      "library",
      "label",
      "stdcall",
      "function",
      "until",
      "finalization",
      "shl",
      "virtual",
      "nil",
      "finally",
      "safecall",
      "div"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str157,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218
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

              if ((((unsigned char)*str ^ (unsigned char)*s) & ~32) == 0 && !gperf_case_strncmp (str, s, len) && s[len] == '\0')
                return s;
            }
        }
    }
  return 0;
}
