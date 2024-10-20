/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_fortran.gperf  */
/* Computed positions: -k'2-3,$' */

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

#define TOTAL_KEYWORDS 120
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 15
#define MIN_HASH_VALUE 11
#define MAX_HASH_VALUE 266
/* maximum key range = 256, duplicates = 0 */

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
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267,  80, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267,  50,  75, 110,  67,   0,
       35,  90,  52, 118,  50,  80,  55,  40,  10,   5,
       85,  35,  20,  20,  15,  40,  70,  40,  95, 112,
        0, 267, 267, 267, 267, 267, 267,  50,  75, 110,
       67,   0,  35,  90,  52, 118,  50,  80,  55,  40,
       10,   5,  85,  35,  20,  20,  15,  40,  70,  40,
       95, 112,   0, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267, 267, 267, 267,
      267, 267, 267, 267, 267, 267, 267
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[2]+1];
      /*FALLTHROUGH*/
      case 2:
        hval += asso_values[(unsigned char)str[1]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_fortran (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str11[sizeof("module")];
      char stringpool_str12[sizeof("do")];
      char stringpool_str17[sizeof("endfile")];
      char stringpool_str18[sizeof("continue")];
      char stringpool_str20[sizeof("enddo")];
      char stringpool_str23[sizeof("non_recursive")];
      char stringpool_str25[sizeof("non_overridable")];
      char stringpool_str26[sizeof("codimension")];
      char stringpool_str28[sizeof("sequence")];
      char stringpool_str31[sizeof("common")];
      char stringpool_str35[sizeof("concurrent")];
      char stringpool_str36[sizeof("result")];
      char stringpool_str37[sizeof("inquire")];
      char stringpool_str38[sizeof("contains")];
      char stringpool_str40[sizeof("contiguous")];
      char stringpool_str44[sizeof("associate")];
      char stringpool_str46[sizeof("format")];
      char stringpool_str50[sizeof("endif")];
      char stringpool_str51[sizeof("assign")];
      char stringpool_str52[sizeof("asynchronous")];
      char stringpool_str53[sizeof("volatile")];
      char stringpool_str54[sizeof("goto")];
      char stringpool_str56[sizeof("return")];
      char stringpool_str58[sizeof("use")];
      char stringpool_str59[sizeof("interface")];
      char stringpool_str61[sizeof("select")];
      char stringpool_str63[sizeof("function")];
      char stringpool_str64[sizeof("pure")];
      char stringpool_str65[sizeof("error")];
      char stringpool_str66[sizeof("nopass")];
      char stringpool_str69[sizeof("case")];
      char stringpool_str71[sizeof("intent")];
      char stringpool_str72[sizeof("if")];
      char stringpool_str74[sizeof("else")];
      char stringpool_str75[sizeof("write")];
      char stringpool_str76[sizeof("recursive")];
      char stringpool_str77[sizeof("private")];
      char stringpool_str79[sizeof("elsewhere")];
      char stringpool_str80[sizeof("end")];
      char stringpool_str81[sizeof("impure")];
      char stringpool_str82[sizeof("pointer")];
      char stringpool_str83[sizeof("namelist")];
      char stringpool_str84[sizeof("include")];
      char stringpool_str85[sizeof("deallocate")];
      char stringpool_str86[sizeof("forall")];
      char stringpool_str89[sizeof("pass")];
      char stringpool_str90[sizeof("print")];
      char stringpool_str91[sizeof("target")];
      char stringpool_str92[sizeof("where")];
      char stringpool_str94[sizeof("save")];
      char stringpool_str95[sizeof("value")];
      char stringpool_str96[sizeof("import")];
      char stringpool_str98[sizeof("implicit")];
      char stringpool_str99[sizeof("parameter")];
      char stringpool_str101[sizeof("then")];
      char stringpool_str102[sizeof("rewrite")];
      char stringpool_str103[sizeof("allocate")];
      char stringpool_str104[sizeof(".eq.")];
      char stringpool_str105[sizeof(".eqv.")];
      char stringpool_str106[sizeof("allocatable")];
      char stringpool_str107[sizeof("while")];
      char stringpool_str109[sizeof(".or.")];
      char stringpool_str110[sizeof("enumerator")];
      char stringpool_str111[sizeof("elseif")];
      char stringpool_str113[sizeof("abstract")];
      char stringpool_str114[sizeof("procedure")];
      char stringpool_str116[sizeof("equivalence")];
      char stringpool_str119[sizeof("wait")];
      char stringpool_str121[sizeof(".true.")];
      char stringpool_str122[sizeof("generic")];
      char stringpool_str124[sizeof("enum")];
      char stringpool_str125[sizeof("pause")];
      char stringpool_str126[sizeof("backspace")];
      char stringpool_str128[sizeof("memory")];
      char stringpool_str129[sizeof(".ne.")];
      char stringpool_str131[sizeof(".neqv.")];
      char stringpool_str133[sizeof("critical")];
      char stringpool_str134[sizeof("open")];
      char stringpool_str136[sizeof("unlock")];
      char stringpool_str139[sizeof("rank")];
      char stringpool_str140[sizeof(".and.")];
      char stringpool_str141[sizeof("images")];
      char stringpool_str144[sizeof("data")];
      char stringpool_str145[sizeof("close")];
      char stringpool_str146[sizeof("read")];
      char stringpool_str147[sizeof("dimension")];
      char stringpool_str148[sizeof("operator")];
      char stringpool_str149[sizeof("call")];
      char stringpool_str152[sizeof("program")];
      char stringpool_str153[sizeof("all")];
      char stringpool_str154[sizeof("elemental")];
      char stringpool_str155[sizeof("class")];
      char stringpool_str156[sizeof("lock")];
      char stringpool_str159[sizeof("submodule")];
      char stringpool_str160[sizeof("subroutine")];
      char stringpool_str162[sizeof("extends")];
      char stringpool_str164[sizeof("exit")];
      char stringpool_str165[sizeof("deferred")];
      char stringpool_str166[sizeof("only")];
      char stringpool_str167[sizeof("entry")];
      char stringpool_str168[sizeof("rewind")];
      char stringpool_str169[sizeof("intrinsic")];
      char stringpool_str174[sizeof(".le.")];
      char stringpool_str179[sizeof(".lt.")];
      char stringpool_str180[sizeof(".not.")];
      char stringpool_str181[sizeof("protected")];
      char stringpool_str182[sizeof("flush")];
      char stringpool_str183[sizeof("final")];
      char stringpool_str184[sizeof("cycle")];
      char stringpool_str188[sizeof("optional")];
      char stringpool_str189[sizeof("stop")];
      char stringpool_str194[sizeof("bind")];
      char stringpool_str197[sizeof(".false.")];
      char stringpool_str198[sizeof("external")];
      char stringpool_str199[sizeof("nullify")];
      char stringpool_str209[sizeof(".ge.")];
      char stringpool_str214[sizeof(".gt.")];
      char stringpool_str225[sizeof("block")];
      char stringpool_str231[sizeof("sync")];
      char stringpool_str266[sizeof("public")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "module",
      "do",
      "endfile",
      "continue",
      "enddo",
      "non_recursive",
      "non_overridable",
      "codimension",
      "sequence",
      "common",
      "concurrent",
      "result",
      "inquire",
      "contains",
      "contiguous",
      "associate",
      "format",
      "endif",
      "assign",
      "asynchronous",
      "volatile",
      "goto",
      "return",
      "use",
      "interface",
      "select",
      "function",
      "pure",
      "error",
      "nopass",
      "case",
      "intent",
      "if",
      "else",
      "write",
      "recursive",
      "private",
      "elsewhere",
      "end",
      "impure",
      "pointer",
      "namelist",
      "include",
      "deallocate",
      "forall",
      "pass",
      "print",
      "target",
      "where",
      "save",
      "value",
      "import",
      "implicit",
      "parameter",
      "then",
      "rewrite",
      "allocate",
      ".eq.",
      ".eqv.",
      "allocatable",
      "while",
      ".or.",
      "enumerator",
      "elseif",
      "abstract",
      "procedure",
      "equivalence",
      "wait",
      ".true.",
      "generic",
      "enum",
      "pause",
      "backspace",
      "memory",
      ".ne.",
      ".neqv.",
      "critical",
      "open",
      "unlock",
      "rank",
      ".and.",
      "images",
      "data",
      "close",
      "read",
      "dimension",
      "operator",
      "call",
      "program",
      "all",
      "elemental",
      "class",
      "lock",
      "submodule",
      "subroutine",
      "extends",
      "exit",
      "deferred",
      "only",
      "entry",
      "rewind",
      "intrinsic",
      ".le.",
      ".lt.",
      ".not.",
      "protected",
      "flush",
      "final",
      "cycle",
      "optional",
      "stop",
      "bind",
      ".false.",
      "external",
      "nullify",
      ".ge.",
      ".gt.",
      "block",
      "sync",
      "public"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str141,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str181,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str182,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str184,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str189,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str194,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str197,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str198,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str199,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str209,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str225,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str231,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str266
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
