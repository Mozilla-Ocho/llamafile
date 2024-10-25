/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_lisp.c llamafile/is_keyword_lisp.gperf  */
/* Computed positions: -k'1,4,8,$' */

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

#line 1 "llamafile/is_keyword_lisp.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 107
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 28
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 311
/* maximum key range = 309, duplicates = 0 */

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
  static const unsigned short asso_values[] =
    {
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312,  55, 312, 312,  25, 312, 312, 312,  10,
       25, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312,  62,   0,  10,   5,   0,
       75,  60,  70,  20, 312,   0,  50,  40,  65,  10,
       30, 312,  90,   5,   0,  95,  20,   0,  10,   0,
      312, 312, 312, 312, 312, 312, 312,  62,   0,  10,
        5,   0,  75,  60,  70,  20, 312,   0,  50,  40,
       65,  10,  30, 312,  90,   5,   0,  95,  20,   0,
       10,   0, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312, 312, 312, 312, 312,
      312, 312, 312, 312, 312, 312
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[7]];
      /*FALLTHROUGH*/
      case 7:
      case 6:
      case 5:
      case 4:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
      case 2:
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_lisp (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("the")];
      char stringpool_str7[sizeof("tagbody")];
      char stringpool_str8[sizeof("typecase")];
      char stringpool_str10[sizeof("ecase")];
      char stringpool_str12[sizeof("deftype")];
      char stringpool_str13[sizeof("deftheme")];
      char stringpool_str14[sizeof("case")];
      char stringpool_str15[sizeof("block")];
      char stringpool_str17[sizeof("do")];
      char stringpool_str18[sizeof("defsubst")];
      char stringpool_str20[sizeof("ccase")];
      char stringpool_str23[sizeof("defconst")];
      char stringpool_str24[sizeof("cond")];
      char stringpool_str26[sizeof("defconstant")];
      char stringpool_str28[sizeof("destructuring-bind")];
      char stringpool_str29[sizeof("defstruct")];
      char stringpool_str30[sizeof("check-type")];
      char stringpool_str31[sizeof("dolist")];
      char stringpool_str36[sizeof("ert-deftest")];
      char stringpool_str37[sizeof("dotimes")];
      char stringpool_str38[sizeof("define-widget")];
      char stringpool_str39[sizeof("condition-case")];
      char stringpool_str44[sizeof("etypecase")];
      char stringpool_str48[sizeof("ignore-errors")];
      char stringpool_str49[sizeof("define-derived-mode")];
      char stringpool_str53[sizeof("let")];
      char stringpool_str54[sizeof("ctypecase")];
      char stringpool_str55[sizeof("while")];
      char stringpool_str58[sizeof("define-inline")];
      char stringpool_str59[sizeof("define-symbol-macro")];
      char stringpool_str60[sizeof("symbol-macrolet")];
      char stringpool_str61[sizeof("labels")];
      char stringpool_str62[sizeof("declare")];
      char stringpool_str63[sizeof("do*")];
      char stringpool_str64[sizeof("multiple-value-bind")];
      char stringpool_str66[sizeof("define-compiler-macro")];
      char stringpool_str67[sizeof("break")];
      char stringpool_str68[sizeof("assert")];
      char stringpool_str69[sizeof("defmethod")];
      char stringpool_str70[sizeof("multiple-value-prog1")];
      char stringpool_str72[sizeof("go")];
      char stringpool_str73[sizeof("defmacro")];
      char stringpool_str74[sizeof("defcustom")];
      char stringpool_str79[sizeof("flet")];
      char stringpool_str82[sizeof("define-minor-mode")];
      char stringpool_str84[sizeof("with-open-file")];
      char stringpool_str85[sizeof("defalias")];
      char stringpool_str86[sizeof("defadvice")];
      char stringpool_str87[sizeof("defface")];
      char stringpool_str91[sizeof("defvaralias")];
      char stringpool_str92[sizeof("defsetf")];
      char stringpool_str94[sizeof("define-modify-macro")];
      char stringpool_str95[sizeof("with-slots")];
      char stringpool_str97[sizeof("if")];
      char stringpool_str99[sizeof("with-accessors")];
      char stringpool_str100[sizeof("define-advice")];
      char stringpool_str102[sizeof("declaim")];
      char stringpool_str104[sizeof("define-generic-mode")];
      char stringpool_str105[sizeof("prog1")];
      char stringpool_str106[sizeof("unless")];
      char stringpool_str107[sizeof("defpackage")];
      char stringpool_str109[sizeof("define-global-minor-mode")];
      char stringpool_str110[sizeof("define-skeleton")];
      char stringpool_str112[sizeof("handler-case")];
      char stringpool_str113[sizeof("define-globalized-minor-mode")];
      char stringpool_str114[sizeof("loop")];
      char stringpool_str115[sizeof("progv")];
      char stringpool_str116[sizeof("define-condition")];
      char stringpool_str117[sizeof("handler-bind")];
      char stringpool_str118[sizeof("lambda")];
      char stringpool_str119[sizeof("locally")];
      char stringpool_str120[sizeof("prog2")];
      char stringpool_str121[sizeof("cl-defsubst")];
      char stringpool_str122[sizeof("in-package")];
      char stringpool_str124[sizeof("eval-when")];
      char stringpool_str126[sizeof("with-open-stream")];
      char stringpool_str127[sizeof("restart-case")];
      char stringpool_str128[sizeof("proclaim")];
      char stringpool_str129[sizeof("with-simple-restart")];
      char stringpool_str131[sizeof("with-compilation-unit")];
      char stringpool_str132[sizeof("restart-bind")];
      char stringpool_str133[sizeof("defgroup")];
      char stringpool_str134[sizeof("when")];
      char stringpool_str135[sizeof("flet*")];
      char stringpool_str137[sizeof("defvar-local")];
      char stringpool_str138[sizeof("macrolet")];
      char stringpool_str139[sizeof("defn")];
      char stringpool_str140[sizeof("define-setf-expander")];
      char stringpool_str142[sizeof("compiler-let")];
      char stringpool_str150[sizeof("prog*")];
      char stringpool_str151[sizeof("with-output-to-string")];
      char stringpool_str154[sizeof("prog")];
      char stringpool_str155[sizeof("define-method-combination")];
      char stringpool_str159[sizeof("unwind-protect")];
      char stringpool_str160[sizeof("progn")];
      char stringpool_str163[sizeof("with-condition-restarts")];
      char stringpool_str164[sizeof("let*")];
      char stringpool_str165[sizeof("with-standard-io-syntax")];
      char stringpool_str170[sizeof("defun")];
      char stringpool_str175[sizeof("defgeneric")];
      char stringpool_str177[sizeof("defparameter")];
      char stringpool_str182[sizeof("with-input-from-string")];
      char stringpool_str189[sizeof("with-hash-table-iterator")];
      char stringpool_str191[sizeof("with-package-iterator")];
      char stringpool_str204[sizeof("letf")];
      char stringpool_str256[sizeof("return")];
      char stringpool_str311[sizeof("return-from")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "the",
      "tagbody",
      "typecase",
      "ecase",
      "deftype",
      "deftheme",
      "case",
      "block",
      "do",
      "defsubst",
      "ccase",
      "defconst",
      "cond",
      "defconstant",
      "destructuring-bind",
      "defstruct",
      "check-type",
      "dolist",
      "ert-deftest",
      "dotimes",
      "define-widget",
      "condition-case",
      "etypecase",
      "ignore-errors",
      "define-derived-mode",
      "let",
      "ctypecase",
      "while",
      "define-inline",
      "define-symbol-macro",
      "symbol-macrolet",
      "labels",
      "declare",
      "do*",
      "multiple-value-bind",
      "define-compiler-macro",
      "break",
      "assert",
      "defmethod",
      "multiple-value-prog1",
      "go",
      "defmacro",
      "defcustom",
      "flet",
      "define-minor-mode",
      "with-open-file",
      "defalias",
      "defadvice",
      "defface",
      "defvaralias",
      "defsetf",
      "define-modify-macro",
      "with-slots",
      "if",
      "with-accessors",
      "define-advice",
      "declaim",
      "define-generic-mode",
      "prog1",
      "unless",
      "defpackage",
      "define-global-minor-mode",
      "define-skeleton",
      "handler-case",
      "define-globalized-minor-mode",
      "loop",
      "progv",
      "define-condition",
      "handler-bind",
      "lambda",
      "locally",
      "prog2",
      "cl-defsubst",
      "in-package",
      "eval-when",
      "with-open-stream",
      "restart-case",
      "proclaim",
      "with-simple-restart",
      "with-compilation-unit",
      "restart-bind",
      "defgroup",
      "when",
      "flet*",
      "defvar-local",
      "macrolet",
      "defn",
      "define-setf-expander",
      "compiler-let",
      "prog*",
      "with-output-to-string",
      "prog",
      "define-method-combination",
      "unwind-protect",
      "progn",
      "with-condition-restarts",
      "let*",
      "with-standard-io-syntax",
      "defun",
      "defgeneric",
      "defparameter",
      "with-input-from-string",
      "with-hash-table-iterator",
      "with-package-iterator",
      "letf",
      "return",
      "return-from"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str175,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str177,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str182,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str189,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str204,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str256,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str311
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
