/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_ada.c llamafile/is_keyword_ada.gperf  */
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

#line 1 "llamafile/is_keyword_ada.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 74
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 12
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 173
/* maximum key range = 171, duplicates = 0 */

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
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174,  10,  85,  20,  15,   0,
       70,  50,  40,  45, 174,  10,  35,  40,   5,  50,
       60,  15,  50,   0,   5,   0,   0,   0,   0,  25,
      174, 174, 174, 174, 174, 174, 174,  10,  85,  20,
       15,   0,  70,  50,  40,  45, 174,  10,  35,  40,
        5,  50,  60,  15,  50,   0,   5,   0,   0,   0,
        0,  25, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174, 174, 174, 174, 174,
      174, 174, 174, 174, 174, 174
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
is_keyword_ada (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("use")];
      char stringpool_str4[sizeof("else")];
      char stringpool_str8[sizeof("new")];
      char stringpool_str9[sizeof("when")];
      char stringpool_str13[sizeof("abs")];
      char stringpool_str14[sizeof("then")];
      char stringpool_str17[sizeof("at")];
      char stringpool_str18[sizeof("not")];
      char stringpool_str19[sizeof("task")];
      char stringpool_str23[sizeof("abstract")];
      char stringpool_str24[sizeof("case")];
      char stringpool_str32[sizeof("synchronized")];
      char stringpool_str33[sizeof("end")];
      char stringpool_str34[sizeof("exception")];
      char stringpool_str35[sizeof("entry")];
      char stringpool_str36[sizeof("access")];
      char stringpool_str38[sizeof("constant")];
      char stringpool_str41[sizeof("accept")];
      char stringpool_str42[sizeof("declare")];
      char stringpool_str43[sizeof("and")];
      char stringpool_str44[sizeof("some")];
      char stringpool_str45[sizeof("until")];
      char stringpool_str46[sizeof("select")];
      char stringpool_str47[sizeof("is")];
      char stringpool_str49[sizeof("with")];
      char stringpool_str50[sizeof("while")];
      char stringpool_str52[sizeof("in")];
      char stringpool_str54[sizeof("exit")];
      char stringpool_str57[sizeof("reverse")];
      char stringpool_str59[sizeof("interface")];
      char stringpool_str60[sizeof("range")];
      char stringpool_str62[sizeof("renames")];
      char stringpool_str63[sizeof("out")];
      char stringpool_str64[sizeof("terminate")];
      char stringpool_str65[sizeof("delta")];
      char stringpool_str66[sizeof("return")];
      char stringpool_str67[sizeof("do")];
      char stringpool_str68[sizeof("separate")];
      char stringpool_str69[sizeof("type")];
      char stringpool_str70[sizeof("abort")];
      char stringpool_str71[sizeof("digits")];
      char stringpool_str72[sizeof("requeue")];
      char stringpool_str73[sizeof("mod")];
      char stringpool_str75[sizeof("elsif")];
      char stringpool_str76[sizeof("tagged")];
      char stringpool_str77[sizeof("aliased")];
      char stringpool_str79[sizeof("null")];
      char stringpool_str80[sizeof("delay")];
      char stringpool_str82[sizeof("generic")];
      char stringpool_str83[sizeof("all")];
      char stringpool_str86[sizeof("pragma")];
      char stringpool_str87[sizeof("package")];
      char stringpool_str88[sizeof("function")];
      char stringpool_str90[sizeof("array")];
      char stringpool_str91[sizeof("record")];
      char stringpool_str92[sizeof("subtype")];
      char stringpool_str96[sizeof("others")];
      char stringpool_str97[sizeof("limited")];
      char stringpool_str100[sizeof("raise")];
      char stringpool_str102[sizeof("or")];
      char stringpool_str103[sizeof("xor")];
      char stringpool_str109[sizeof("goto")];
      char stringpool_str110[sizeof("overriding")];
      char stringpool_str112[sizeof("private")];
      char stringpool_str117[sizeof("if")];
      char stringpool_str119[sizeof("procedure")];
      char stringpool_str122[sizeof("of")];
      char stringpool_str129[sizeof("body")];
      char stringpool_str133[sizeof("rem")];
      char stringpool_str134[sizeof("protected")];
      char stringpool_str145[sizeof("begin")];
      char stringpool_str149[sizeof("loop")];
      char stringpool_str153[sizeof("parallel")];
      char stringpool_str173[sizeof("for")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "use",
      "else",
      "new",
      "when",
      "abs",
      "then",
      "at",
      "not",
      "task",
      "abstract",
      "case",
      "synchronized",
      "end",
      "exception",
      "entry",
      "access",
      "constant",
      "accept",
      "declare",
      "and",
      "some",
      "until",
      "select",
      "is",
      "with",
      "while",
      "in",
      "exit",
      "reverse",
      "interface",
      "range",
      "renames",
      "out",
      "terminate",
      "delta",
      "return",
      "do",
      "separate",
      "type",
      "abort",
      "digits",
      "requeue",
      "mod",
      "elsif",
      "tagged",
      "aliased",
      "null",
      "delay",
      "generic",
      "all",
      "pragma",
      "package",
      "function",
      "array",
      "record",
      "subtype",
      "others",
      "limited",
      "raise",
      "or",
      "xor",
      "goto",
      "overriding",
      "private",
      "if",
      "procedure",
      "of",
      "body",
      "rem",
      "protected",
      "begin",
      "loop",
      "parallel",
      "for"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173
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
