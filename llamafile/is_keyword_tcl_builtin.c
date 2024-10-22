/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_tcl_builtin.c llamafile/is_keyword_tcl_builtin.gperf  */
/* Computed positions: -k'1-2,4' */

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

#line 1 "llamafile/is_keyword_tcl_builtin.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 64
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 10
#define MIN_HASH_VALUE 5
#define MAX_HASH_VALUE 112
/* maximum key range = 108, duplicates = 0 */

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
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113,  30,  10,   5,
       30,   5,  55,  10,   0,  10,   5,  40,   0,   0,
        0,  20,  35, 113,   0,   0,  75,  70,   5,   0,
       55, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
      113, 113, 113, 113, 113, 113
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
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
is_keyword_tcl_builtin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str5[sizeof("lsort")];
      char stringpool_str6[sizeof("lrange")];
      char stringpool_str7[sizeof("llength")];
      char stringpool_str8[sizeof("set")];
      char stringpool_str9[sizeof("scan")];
      char stringpool_str10[sizeof("close")];
      char stringpool_str11[sizeof("regsub")];
      char stringpool_str14[sizeof("incr")];
      char stringpool_str15[sizeof("clock")];
      char stringpool_str16[sizeof("regexp")];
      char stringpool_str17[sizeof("linsert")];
      char stringpool_str19[sizeof("gets")];
      char stringpool_str20[sizeof("vwait")];
      char stringpool_str21[sizeof("interp")];
      char stringpool_str23[sizeof("registry")];
      char stringpool_str24[sizeof("glob")];
      char stringpool_str26[sizeof("source")];
      char stringpool_str27[sizeof("bgerror")];
      char stringpool_str28[sizeof("eof")];
      char stringpool_str29[sizeof("join")];
      char stringpool_str32[sizeof("console")];
      char stringpool_str33[sizeof("encoding")];
      char stringpool_str34[sizeof("info")];
      char stringpool_str36[sizeof("concat")];
      char stringpool_str37[sizeof("cd")];
      char stringpool_str38[sizeof("pwd")];
      char stringpool_str39[sizeof("read")];
      char stringpool_str41[sizeof("rename")];
      char stringpool_str43[sizeof("lreplace")];
      char stringpool_str44[sizeof("namespace")];
      char stringpool_str45[sizeof("catch")];
      char stringpool_str46[sizeof("lindex")];
      char stringpool_str48[sizeof("pid")];
      char stringpool_str49[sizeof("seek")];
      char stringpool_str50[sizeof("split")];
      char stringpool_str54[sizeof("load")];
      char stringpool_str56[sizeof("binary")];
      char stringpool_str59[sizeof("open")];
      char stringpool_str60[sizeof("flush")];
      char stringpool_str63[sizeof("dde")];
      char stringpool_str64[sizeof("expr")];
      char stringpool_str65[sizeof("array")];
      char stringpool_str66[sizeof("socket")];
      char stringpool_str69[sizeof("exec")];
      char stringpool_str70[sizeof("fconfigure")];
      char stringpool_str72[sizeof("lappend")];
      char stringpool_str74[sizeof("file")];
      char stringpool_str75[sizeof("subst")];
      char stringpool_str76[sizeof("append")];
      char stringpool_str77[sizeof("unknown")];
      char stringpool_str79[sizeof("fileevent")];
      char stringpool_str80[sizeof("unset")];
      char stringpool_str81[sizeof("format")];
      char stringpool_str84[sizeof("tell")];
      char stringpool_str85[sizeof("trace")];
      char stringpool_str89[sizeof("list")];
      char stringpool_str91[sizeof("string")];
      char stringpool_str92[sizeof("history")];
      char stringpool_str93[sizeof("fblocked")];
      char stringpool_str94[sizeof("time")];
      char stringpool_str95[sizeof("after")];
      char stringpool_str100[sizeof("fcopy")];
      char stringpool_str109[sizeof("puts")];
      char stringpool_str112[sizeof("package")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "lsort",
      "lrange",
      "llength",
      "set",
      "scan",
      "close",
      "regsub",
      "incr",
      "clock",
      "regexp",
      "linsert",
      "gets",
      "vwait",
      "interp",
      "registry",
      "glob",
      "source",
      "bgerror",
      "eof",
      "join",
      "console",
      "encoding",
      "info",
      "concat",
      "cd",
      "pwd",
      "read",
      "rename",
      "lreplace",
      "namespace",
      "catch",
      "lindex",
      "pid",
      "seek",
      "split",
      "load",
      "binary",
      "open",
      "flush",
      "dde",
      "expr",
      "array",
      "socket",
      "exec",
      "fconfigure",
      "lappend",
      "file",
      "subst",
      "append",
      "unknown",
      "fileevent",
      "unset",
      "format",
      "tell",
      "trace",
      "list",
      "string",
      "history",
      "fblocked",
      "time",
      "after",
      "fcopy",
      "puts",
      "package"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1,
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
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112
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
