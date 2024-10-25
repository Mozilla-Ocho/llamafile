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

#define TOTAL_KEYWORDS 108
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 28
#define MIN_HASH_VALUE 8
#define MAX_HASH_VALUE 349
/* maximum key range = 342, duplicates = 0 */

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
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350,  55, 350, 350,   0, 350, 350, 350,  10,
       15, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 115,   0,  10,   5,   0,
      110,  75,  40,  15, 350,   5, 125,  30,   0,  25,
        0, 350,  60,  20,   5,  30,   5,  15,   0,   0,
      350, 350, 350, 350, 350, 350, 350, 115,   0,  10,
        5,   0, 110,  75,  40,  15, 350,   5, 125,  30,
        0,  25,   0, 350,  60,  20,   5,  30,   5,  15,
        0,   0, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
      350, 350, 350, 350, 350, 350
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
      char stringpool_str8[sizeof("the")];
      char stringpool_str9[sizeof("defn")];
      char stringpool_str12[sizeof("tagbody")];
      char stringpool_str13[sizeof("typecase")];
      char stringpool_str14[sizeof("case")];
      char stringpool_str17[sizeof("deftype")];
      char stringpool_str18[sizeof("deftheme")];
      char stringpool_str19[sizeof("when")];
      char stringpool_str20[sizeof("block")];
      char stringpool_str21[sizeof("ert-deftest")];
      char stringpool_str24[sizeof("cond")];
      char stringpool_str25[sizeof("ecase")];
      char stringpool_str26[sizeof("define")];
      char stringpool_str29[sizeof("etypecase")];
      char stringpool_str30[sizeof("check-type")];
      char stringpool_str31[sizeof("dolist")];
      char stringpool_str32[sizeof("do")];
      char stringpool_str33[sizeof("defconst")];
      char stringpool_str35[sizeof("ccase")];
      char stringpool_str36[sizeof("defconstant")];
      char stringpool_str38[sizeof("destructuring-bind")];
      char stringpool_str39[sizeof("ctypecase")];
      char stringpool_str40[sizeof("defun")];
      char stringpool_str43[sizeof("defsubst")];
      char stringpool_str44[sizeof("define-derived-mode")];
      char stringpool_str46[sizeof("define-condition")];
      char stringpool_str47[sizeof("dotimes")];
      char stringpool_str48[sizeof("define-inline")];
      char stringpool_str49[sizeof("defstruct")];
      char stringpool_str53[sizeof("define-widget")];
      char stringpool_str54[sizeof("condition-case")];
      char stringpool_str55[sizeof("define-skeleton")];
      char stringpool_str56[sizeof("unless")];
      char stringpool_str57[sizeof("handler-case")];
      char stringpool_str59[sizeof("multiple-value-bind")];
      char stringpool_str61[sizeof("cl-defsubst")];
      char stringpool_str62[sizeof("handler-bind")];
      char stringpool_str63[sizeof("do*")];
      char stringpool_str64[sizeof("unwind-protect")];
      char stringpool_str65[sizeof("multiple-value-prog1")];
      char stringpool_str67[sizeof("define-minor-mode")];
      char stringpool_str69[sizeof("with-open-file")];
      char stringpool_str70[sizeof("symbol-macrolet")];
      char stringpool_str73[sizeof("ignore-errors")];
      char stringpool_str74[sizeof("defmethod")];
      char stringpool_str75[sizeof("define-method-combination")];
      char stringpool_str76[sizeof("define-compiler-macro")];
      char stringpool_str77[sizeof("restart-case")];
      char stringpool_str78[sizeof("proclaim")];
      char stringpool_str79[sizeof("defcustom")];
      char stringpool_str80[sizeof("progn")];
      char stringpool_str82[sizeof("restart-bind")];
      char stringpool_str84[sizeof("define-symbol-macro")];
      char stringpool_str85[sizeof("progv")];
      char stringpool_str87[sizeof("compiler-let")];
      char stringpool_str88[sizeof("defgroup")];
      char stringpool_str90[sizeof("prog1")];
      char stringpool_str93[sizeof("defmacro")];
      char stringpool_str94[sizeof("define-modify-macro")];
      char stringpool_str95[sizeof("prog2")];
      char stringpool_str96[sizeof("return")];
      char stringpool_str98[sizeof("with-condition-restarts")];
      char stringpool_str99[sizeof("with-accessors")];
      char stringpool_str101[sizeof("with-open-stream")];
      char stringpool_str102[sizeof("go")];
      char stringpool_str107[sizeof("defparameter")];
      char stringpool_str108[sizeof("macrolet")];
      char stringpool_str109[sizeof("with-simple-restart")];
      char stringpool_str110[sizeof("with-slots")];
      char stringpool_str111[sizeof("with-compilation-unit")];
      char stringpool_str114[sizeof("define-generic-mode")];
      char stringpool_str119[sizeof("define-global-minor-mode")];
      char stringpool_str120[sizeof("define-setf-expander")];
      char stringpool_str122[sizeof("defface")];
      char stringpool_str123[sizeof("define-globalized-minor-mode")];
      char stringpool_str124[sizeof("flet")];
      char stringpool_str125[sizeof("break")];
      char stringpool_str126[sizeof("assert")];
      char stringpool_str127[sizeof("if")];
      char stringpool_str129[sizeof("loop")];
      char stringpool_str130[sizeof("defpackage")];
      char stringpool_str133[sizeof("let")];
      char stringpool_str134[sizeof("eval-when")];
      char stringpool_str135[sizeof("prog*")];
      char stringpool_str137[sizeof("declare")];
      char stringpool_str139[sizeof("defadvice")];
      char stringpool_str140[sizeof("in-package")];
      char stringpool_str142[sizeof("defsetf")];
      char stringpool_str145[sizeof("while")];
      char stringpool_str146[sizeof("with-package-iterator")];
      char stringpool_str148[sizeof("define-advice")];
      char stringpool_str151[sizeof("labels")];
      char stringpool_str152[sizeof("with-input-from-string")];
      char stringpool_str154[sizeof("prog")];
      char stringpool_str156[sizeof("with-output-to-string")];
      char stringpool_str159[sizeof("with-hash-table-iterator")];
      char stringpool_str160[sizeof("defgeneric")];
      char stringpool_str166[sizeof("defvaralias")];
      char stringpool_str167[sizeof("declaim")];
      char stringpool_str168[sizeof("defalias")];
      char stringpool_str175[sizeof("flet*")];
      char stringpool_str193[sizeof("with-standard-io-syntax")];
      char stringpool_str239[sizeof("let*")];
      char stringpool_str241[sizeof("return-from")];
      char stringpool_str246[sizeof("lambda")];
      char stringpool_str247[sizeof("locally")];
      char stringpool_str272[sizeof("defvar-local")];
      char stringpool_str349[sizeof("letf")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "the",
      "defn",
      "tagbody",
      "typecase",
      "case",
      "deftype",
      "deftheme",
      "when",
      "block",
      "ert-deftest",
      "cond",
      "ecase",
      "define",
      "etypecase",
      "check-type",
      "dolist",
      "do",
      "defconst",
      "ccase",
      "defconstant",
      "destructuring-bind",
      "ctypecase",
      "defun",
      "defsubst",
      "define-derived-mode",
      "define-condition",
      "dotimes",
      "define-inline",
      "defstruct",
      "define-widget",
      "condition-case",
      "define-skeleton",
      "unless",
      "handler-case",
      "multiple-value-bind",
      "cl-defsubst",
      "handler-bind",
      "do*",
      "unwind-protect",
      "multiple-value-prog1",
      "define-minor-mode",
      "with-open-file",
      "symbol-macrolet",
      "ignore-errors",
      "defmethod",
      "define-method-combination",
      "define-compiler-macro",
      "restart-case",
      "proclaim",
      "defcustom",
      "progn",
      "restart-bind",
      "define-symbol-macro",
      "progv",
      "compiler-let",
      "defgroup",
      "prog1",
      "defmacro",
      "define-modify-macro",
      "prog2",
      "return",
      "with-condition-restarts",
      "with-accessors",
      "with-open-stream",
      "go",
      "defparameter",
      "macrolet",
      "with-simple-restart",
      "with-slots",
      "with-compilation-unit",
      "define-generic-mode",
      "define-global-minor-mode",
      "define-setf-expander",
      "defface",
      "define-globalized-minor-mode",
      "flet",
      "break",
      "assert",
      "if",
      "loop",
      "defpackage",
      "let",
      "eval-when",
      "prog*",
      "declare",
      "defadvice",
      "in-package",
      "defsetf",
      "while",
      "with-package-iterator",
      "define-advice",
      "labels",
      "with-input-from-string",
      "prog",
      "with-output-to-string",
      "with-hash-table-iterator",
      "defgeneric",
      "defvaralias",
      "declaim",
      "defalias",
      "flet*",
      "with-standard-io-syntax",
      "let*",
      "return-from",
      "lambda",
      "locally",
      "defvar-local",
      "letf"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str175,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str239,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str241,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str246,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str247,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str272,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str349
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
