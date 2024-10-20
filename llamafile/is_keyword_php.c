/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_php.gperf  */
/* Computed positions: -k'1,3-4' */

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

#line 1 "llamafile/is_keyword_php.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 78
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 51
#define MIN_HASH_VALUE 2
#define MAX_HASH_VALUE 165
/* maximum key range = 164, duplicates = 0 */

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
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166,  30,  20,  20,  10,  10,
       40,   5,   0,   0, 166, 166,  35,  95,  35,  55,
       35,   0,  40,   0,   0,  80,  10,  85,   5, 115,
      166, 166, 166, 166, 166,  30, 166,  30,  20,  20,
       10,  10,  40,   5,   0,   0, 166, 166,  35,  95,
       35,  55,  35,   0,  40,   0,   0,  80,  10,  85,
        5, 115, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166,   5, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166, 166, 166, 166, 166,
        5, 166, 166, 166, 166, 166, 166, 166, 166, 166,
      166, 166, 166, 166, 166, 166
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
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

const char *
is_keyword_php (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str2[sizeof("if")];
      char stringpool_str6[sizeof("switch")];
      char stringpool_str9[sizeof("insteadof")];
      char stringpool_str10[sizeof("instanceof")];
      char stringpool_str12[sizeof("do")];
      char stringpool_str14[sizeof("exit")];
      char stringpool_str15[sizeof("isset")];
      char stringpool_str19[sizeof("interface")];
      char stringpool_str23[sizeof("die")];
      char stringpool_str24[sizeof("else")];
      char stringpool_str25[sizeof("endif")];
      char stringpool_str26[sizeof("elseif")];
      char stringpool_str27[sizeof("extends")];
      char stringpool_str29[sizeof("endswitch")];
      char stringpool_str32[sizeof("as")];
      char stringpool_str34[sizeof("case")];
      char stringpool_str35[sizeof("trait")];
      char stringpool_str36[sizeof("static")];
      char stringpool_str38[sizeof("abstract")];
      char stringpool_str39[sizeof("list")];
      char stringpool_str40[sizeof("enddeclare")];
      char stringpool_str42[sizeof("fn")];
      char stringpool_str43[sizeof("and")];
      char stringpool_str45[sizeof("catch")];
      char stringpool_str47[sizeof("__DIR__")];
      char stringpool_str48[sizeof("xor")];
      char stringpool_str50[sizeof("empty")];
      char stringpool_str52[sizeof("private")];
      char stringpool_str53[sizeof("var")];
      char stringpool_str55[sizeof("class")];
      char stringpool_str57[sizeof("or")];
      char stringpool_str60[sizeof("const")];
      char stringpool_str62[sizeof("include")];
      char stringpool_str63[sizeof("continue")];
      char stringpool_str64[sizeof("goto")];
      char stringpool_str65[sizeof("break")];
      char stringpool_str66[sizeof("endfor")];
      char stringpool_str67[sizeof("include_once")];
      char stringpool_str69[sizeof("echo")];
      char stringpool_str70[sizeof("endforeach")];
      char stringpool_str72[sizeof("declare")];
      char stringpool_str73[sizeof("__LINE__")];
      char stringpool_str75[sizeof("print")];
      char stringpool_str78[sizeof("__FILE__")];
      char stringpool_str79[sizeof("eval")];
      char stringpool_str80[sizeof("implements")];
      char stringpool_str81[sizeof("__TRAIT__\011\011")];
      char stringpool_str83[sizeof("for")];
      char stringpool_str86[sizeof("global")];
      char stringpool_str87[sizeof("default")];
      char stringpool_str88[sizeof("readonly")];
      char stringpool_str91[sizeof("__\360\235\224\245\360\235\224\236\360\235\224\251\360\235\224\261_\360\235\224\240\360\235\224\254\360\235\224\252\360\235\224\255\360\235\224\246\360\235\224\251\360\235\224\242\360\235\224\257")];
      char stringpool_str93[sizeof("use")];
      char stringpool_str94[sizeof("__CLASS__")];
      char stringpool_str95[sizeof("unset")];
      char stringpool_str96[sizeof("public")];
      char stringpool_str97[sizeof("foreach")];
      char stringpool_str98[sizeof("callable")];
      char stringpool_str99[sizeof("protected")];
      char stringpool_str100[sizeof("throw")];
      char stringpool_str103[sizeof("function")];
      char stringpool_str105[sizeof("array")];
      char stringpool_str108[sizeof("__NAMESPACE__")];
      char stringpool_str110[sizeof("final")];
      char stringpool_str112[sizeof("finally")];
      char stringpool_str113[sizeof("endwhile")];
      char stringpool_str115[sizeof("clone")];
      char stringpool_str118[sizeof("try")];
      char stringpool_str120[sizeof("match")];
      char stringpool_str123[sizeof("new")];
      char stringpool_str125[sizeof("while")];
      char stringpool_str126[sizeof("return")];
      char stringpool_str127[sizeof("require")];
      char stringpool_str132[sizeof("require_once")];
      char stringpool_str145[sizeof("__METHOD__")];
      char stringpool_str149[sizeof("namespace")];
      char stringpool_str162[sizeof("__FUNCTION__")];
      char stringpool_str165[sizeof("yield")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "if",
      "switch",
      "insteadof",
      "instanceof",
      "do",
      "exit",
      "isset",
      "interface",
      "die",
      "else",
      "endif",
      "elseif",
      "extends",
      "endswitch",
      "as",
      "case",
      "trait",
      "static",
      "abstract",
      "list",
      "enddeclare",
      "fn",
      "and",
      "catch",
      "__DIR__",
      "xor",
      "empty",
      "private",
      "var",
      "class",
      "or",
      "const",
      "include",
      "continue",
      "goto",
      "break",
      "endfor",
      "include_once",
      "echo",
      "endforeach",
      "declare",
      "__LINE__",
      "print",
      "__FILE__",
      "eval",
      "implements",
      "__TRAIT__\011\011",
      "for",
      "global",
      "default",
      "readonly",
      "__\360\235\224\245\360\235\224\236\360\235\224\251\360\235\224\261_\360\235\224\240\360\235\224\254\360\235\224\252\360\235\224\255\360\235\224\246\360\235\224\251\360\235\224\242\360\235\224\257",
      "use",
      "__CLASS__",
      "unset",
      "public",
      "foreach",
      "callable",
      "protected",
      "throw",
      "function",
      "array",
      "__NAMESPACE__",
      "final",
      "finally",
      "endwhile",
      "clone",
      "try",
      "match",
      "new",
      "while",
      "return",
      "require",
      "require_once",
      "__METHOD__",
      "namespace",
      "__FUNCTION__",
      "yield"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      -1,
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
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165
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
