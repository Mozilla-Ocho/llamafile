/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_fortran.c llamafile/is_keyword_fortran.gperf  */
/* Computed positions: -k'1-3,$' */

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

#line 1 "llamafile/is_keyword_fortran.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 122
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 15
#define MIN_HASH_VALUE 9
#define MAX_HASH_VALUE 272
/* maximum key range = 264, duplicates = 0 */

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
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273, 273,  70, 273, 273, 273,
      273, 273, 273, 273, 273, 273, 273, 273, 273, 273,
      273, 273, 273, 273, 273,   0,  25,  30,  70,   5,
       95,  74,  10,  80, 273,  12,   5,  30,  10,  10,
       50,  95,  25,   0,  20,  90,   0, 110,   5, 122,
      273, 273, 273, 273, 273, 273, 273,   0,  25,  30,
       70,   5,  95,  74,  10,  80, 273,  12,   5,  30,
       10,  10,  50,  95,  25,   0,  20,  90,   0, 110,
        5, 122, 273, 273, 273, 273, 273, 273, 273, 273,
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
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
        hval += asso_values[(unsigned char)str[1]];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_fortran (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str9[sizeof("save")];
      char stringpool_str14[sizeof("associate")];
      char stringpool_str15[sizeof("value")];
      char stringpool_str16[sizeof("assign")];
      char stringpool_str18[sizeof("all")];
      char stringpool_str19[sizeof("else")];
      char stringpool_str23[sizeof("allocate")];
      char stringpool_str24[sizeof("elsewhere")];
      char stringpool_str26[sizeof("allocatable")];
      char stringpool_str28[sizeof("volatile")];
      char stringpool_str29[sizeof("elemental")];
      char stringpool_str36[sizeof("select")];
      char stringpool_str37[sizeof("extends")];
      char stringpool_str39[sizeof("case")];
      char stringpool_str40[sizeof("class")];
      char stringpool_str42[sizeof("to")];
      char stringpool_str43[sizeof("external")];
      char stringpool_str44[sizeof("call")];
      char stringpool_str48[sizeof("non_recursive")];
      char stringpool_str49[sizeof("then")];
      char stringpool_str50[sizeof("non_overridable")];
      char stringpool_str51[sizeof("rank")];
      char stringpool_str53[sizeof("abstract")];
      char stringpool_str54[sizeof("pass")];
      char stringpool_str55[sizeof("close")];
      char stringpool_str56[sizeof("result")];
      char stringpool_str57[sizeof("block")];
      char stringpool_str58[sizeof("contains")];
      char stringpool_str60[sizeof("contiguous")];
      char stringpool_str61[sizeof("lock")];
      char stringpool_str63[sizeof("continue")];
      char stringpool_str66[sizeof("return")];
      char stringpool_str68[sizeof("namelist")];
      char stringpool_str69[sizeof("backspace")];
      char stringpool_str71[sizeof("target")];
      char stringpool_str74[sizeof("recursive")];
      char stringpool_str76[sizeof("nopass")];
      char stringpool_str79[sizeof("open")];
      char stringpool_str80[sizeof("concurrent")];
      char stringpool_str84[sizeof("stop")];
      char stringpool_str85[sizeof("error")];
      char stringpool_str86[sizeof("common")];
      char stringpool_str90[sizeof("deallocate")];
      char stringpool_str92[sizeof("do")];
      char stringpool_str93[sizeof("optional")];
      char stringpool_str94[sizeof("data")];
      char stringpool_str96[sizeof("go")];
      char stringpool_str97[sizeof("endfile")];
      char stringpool_str98[sizeof("operator")];
      char stringpool_str99[sizeof("procedure")];
      char stringpool_str100[sizeof("enddo")];
      char stringpool_str103[sizeof("use")];
      char stringpool_str104[sizeof("read")];
      char stringpool_str109[sizeof("parameter")];
      char stringpool_str111[sizeof("elseif")];
      char stringpool_str113[sizeof("sequence")];
      char stringpool_str114[sizeof("exit")];
      char stringpool_str116[sizeof("images")];
      char stringpool_str118[sizeof("goto")];
      char stringpool_str121[sizeof("module")];
      char stringpool_str122[sizeof("program")];
      char stringpool_str123[sizeof("unlock")];
      char stringpool_str124[sizeof("interface")];
      char stringpool_str126[sizeof("generic")];
      char stringpool_str129[sizeof("submodule")];
      char stringpool_str130[sizeof("subroutine")];
      char stringpool_str131[sizeof("codimension")];
      char stringpool_str132[sizeof("include")];
      char stringpool_str134[sizeof("asynchronous")];
      char stringpool_str135[sizeof("where")];
      char stringpool_str136[sizeof("intent")];
      char stringpool_str139[sizeof("enum")];
      char stringpool_str140[sizeof("enumerator")];
      char stringpool_str141[sizeof("forall")];
      char stringpool_str148[sizeof("critical")];
      char stringpool_str149[sizeof("intrinsic")];
      char stringpool_str150[sizeof("pause")];
      char stringpool_str151[sizeof("only")];
      char stringpool_str152[sizeof("rewrite")];
      char stringpool_str154[sizeof(".le.")];
      char stringpool_str155[sizeof(".and.")];
      char stringpool_str156[sizeof("format")];
      char stringpool_str158[sizeof("end")];
      char stringpool_str159[sizeof(".ne.")];
      char stringpool_str161[sizeof(".neqv.")];
      char stringpool_str162[sizeof("entry")];
      char stringpool_str164[sizeof("protected")];
      char stringpool_str165[sizeof(".not.")];
      char stringpool_str166[sizeof("sync")];
      char stringpool_str167[sizeof("private")];
      char stringpool_str169[sizeof(".lt.")];
      char stringpool_str171[sizeof("impure")];
      char stringpool_str172[sizeof("pointer")];
      char stringpool_str174[sizeof("pure")];
      char stringpool_str179[sizeof(".or.")];
      char stringpool_str180[sizeof("print")];
      char stringpool_str185[sizeof("endif")];
      char stringpool_str186[sizeof("import")];
      char stringpool_str188[sizeof("implicit")];
      char stringpool_str189[sizeof("bind")];
      char stringpool_str191[sizeof(".true.")];
      char stringpool_str192[sizeof("cycle")];
      char stringpool_str193[sizeof("memory")];
      char stringpool_str195[sizeof("final")];
      char stringpool_str197[sizeof("inquire")];
      char stringpool_str199[sizeof("dimension")];
      char stringpool_str201[sizeof("public")];
      char stringpool_str205[sizeof("flush")];
      char stringpool_str206[sizeof("equivalence")];
      char stringpool_str210[sizeof("while")];
      char stringpool_str213[sizeof("function")];
      char stringpool_str214[sizeof("wait")];
      char stringpool_str216[sizeof("rewind")];
      char stringpool_str223[sizeof(".ge.")];
      char stringpool_str225[sizeof("write")];
      char stringpool_str234[sizeof("nullify")];
      char stringpool_str238[sizeof(".gt.")];
      char stringpool_str242[sizeof(".false.")];
      char stringpool_str244[sizeof(".eq.")];
      char stringpool_str245[sizeof(".eqv.")];
      char stringpool_str248[sizeof("deferred")];
      char stringpool_str272[sizeof("if")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "save",
      "associate",
      "value",
      "assign",
      "all",
      "else",
      "allocate",
      "elsewhere",
      "allocatable",
      "volatile",
      "elemental",
      "select",
      "extends",
      "case",
      "class",
      "to",
      "external",
      "call",
      "non_recursive",
      "then",
      "non_overridable",
      "rank",
      "abstract",
      "pass",
      "close",
      "result",
      "block",
      "contains",
      "contiguous",
      "lock",
      "continue",
      "return",
      "namelist",
      "backspace",
      "target",
      "recursive",
      "nopass",
      "open",
      "concurrent",
      "stop",
      "error",
      "common",
      "deallocate",
      "do",
      "optional",
      "data",
      "go",
      "endfile",
      "operator",
      "procedure",
      "enddo",
      "use",
      "read",
      "parameter",
      "elseif",
      "sequence",
      "exit",
      "images",
      "goto",
      "module",
      "program",
      "unlock",
      "interface",
      "generic",
      "submodule",
      "subroutine",
      "codimension",
      "include",
      "asynchronous",
      "where",
      "intent",
      "enum",
      "enumerator",
      "forall",
      "critical",
      "intrinsic",
      "pause",
      "only",
      "rewrite",
      ".le.",
      ".and.",
      "format",
      "end",
      ".ne.",
      ".neqv.",
      "entry",
      "protected",
      ".not.",
      "sync",
      "private",
      ".lt.",
      "impure",
      "pointer",
      "pure",
      ".or.",
      "print",
      "endif",
      "import",
      "implicit",
      "bind",
      ".true.",
      "cycle",
      "memory",
      "final",
      "inquire",
      "dimension",
      "public",
      "flush",
      "equivalence",
      "while",
      "function",
      "wait",
      "rewind",
      ".ge.",
      "write",
      "nullify",
      ".gt.",
      ".false.",
      ".eq.",
      ".eqv.",
      "deferred",
      "if"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str141,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str185,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str189,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str192,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str195,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str197,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str199,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str201,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str205,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str210,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str213,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str223,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str225,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str234,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str238,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str242,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str244,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str245,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str248,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
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

              if ((((unsigned char)*str ^ (unsigned char)*s) & ~32) == 0 && !gperf_case_strncmp (str, s, len) && s[len] == '\0')
                return s;
            }
        }
    }
  return 0;
}
