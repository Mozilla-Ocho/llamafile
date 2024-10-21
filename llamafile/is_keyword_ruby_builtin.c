/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_ruby_builtin.gperf  */
/* Computed positions: -k'1-2,4,6' */

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

#line 1 "llamafile/is_keyword_ruby_builtin.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 67
#define MIN_WORD_LENGTH 1
#define MAX_WORD_LENGTH 20
#define MIN_HASH_VALUE 4
#define MAX_HASH_VALUE 136
/* maximum key range = 133, duplicates = 0 */

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
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137,  20, 137,   0,  45,   5,
       25,   0,  85,   0,  55,  35, 137,   0,   5,   5,
       10,  40,   5, 137,   0,  20,   0,  10,  40,  65,
       80,  20, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137, 137, 137, 137, 137,
      137, 137, 137, 137, 137, 137
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[5]];
      /*FALLTHROUGH*/
      case 5:
      case 4:
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
is_keyword_ruby_builtin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str4[sizeof("attr")];
      char stringpool_str6[sizeof("p")];
      char stringpool_str9[sizeof("trap")];
      char stringpool_str11[sizeof("attr_reader")];
      char stringpool_str13[sizeof("attr_accessor")];
      char stringpool_str14[sizeof("proc")];
      char stringpool_str15[sizeof("catch")];
      char stringpool_str16[sizeof("caller")];
      char stringpool_str17[sizeof("require")];
      char stringpool_str19[sizeof("protected")];
      char stringpool_str20[sizeof("print")];
      char stringpool_str21[sizeof("callcc")];
      char stringpool_str24[sizeof("putc")];
      char stringpool_str25[sizeof("raise")];
      char stringpool_str26[sizeof("require_relative")];
      char stringpool_str27[sizeof("prepend")];
      char stringpool_str29[sizeof("rand")];
      char stringpool_str30[sizeof("sleep")];
      char stringpool_str31[sizeof("public")];
      char stringpool_str34[sizeof("trace_var")];
      char stringpool_str35[sizeof("srand")];
      char stringpool_str36[sizeof("untrace_var")];
      char stringpool_str37[sizeof("alias_method")];
      char stringpool_str39[sizeof("puts")];
      char stringpool_str40[sizeof("public_constant")];
      char stringpool_str41[sizeof("refine")];
      char stringpool_str42[sizeof("at_exit")];
      char stringpool_str44[sizeof("public_class_method")];
      char stringpool_str45[sizeof("using")];
      char stringpool_str49[sizeof("eval")];
      char stringpool_str50[sizeof("abort")];
      char stringpool_str51[sizeof("system")];
      char stringpool_str52[sizeof("private")];
      char stringpool_str54[sizeof("loop")];
      char stringpool_str55[sizeof("__callee__")];
      char stringpool_str56[sizeof("lambda")];
      char stringpool_str57[sizeof("syscall")];
      char stringpool_str59[sizeof("open")];
      char stringpool_str61[sizeof("private_constant")];
      char stringpool_str65[sizeof("private_class_method")];
      char stringpool_str67[sizeof("sprintf")];
      char stringpool_str68[sizeof("readline")];
      char stringpool_str69[sizeof("readlines")];
      char stringpool_str70[sizeof("module_function")];
      char stringpool_str71[sizeof("global_variables")];
      char stringpool_str73[sizeof("define_method")];
      char stringpool_str74[sizeof("load")];
      char stringpool_str76[sizeof("attr_writer")];
      char stringpool_str79[sizeof("warn")];
      char stringpool_str80[sizeof("local_variables")];
      char stringpool_str82[sizeof("include")];
      char stringpool_str84[sizeof("exit")];
      char stringpool_str85[sizeof("exit!")];
      char stringpool_str87[sizeof("block_given?")];
      char stringpool_str89[sizeof("exec")];
      char stringpool_str94[sizeof("fail")];
      char stringpool_str95[sizeof("spawn")];
      char stringpool_str98[sizeof("autoload")];
      char stringpool_str99[sizeof("autoload?")];
      char stringpool_str100[sizeof("throw")];
      char stringpool_str102[sizeof("__dir__")];
      char stringpool_str105[sizeof("__method__")];
      char stringpool_str106[sizeof("printf")];
      char stringpool_str111[sizeof("extend")];
      char stringpool_str122[sizeof("binding")];
      char stringpool_str129[sizeof("fork")];
      char stringpool_str136[sizeof("format")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "attr",
      "p",
      "trap",
      "attr_reader",
      "attr_accessor",
      "proc",
      "catch",
      "caller",
      "require",
      "protected",
      "print",
      "callcc",
      "putc",
      "raise",
      "require_relative",
      "prepend",
      "rand",
      "sleep",
      "public",
      "trace_var",
      "srand",
      "untrace_var",
      "alias_method",
      "puts",
      "public_constant",
      "refine",
      "at_exit",
      "public_class_method",
      "using",
      "eval",
      "abort",
      "system",
      "private",
      "loop",
      "__callee__",
      "lambda",
      "syscall",
      "open",
      "private_constant",
      "private_class_method",
      "sprintf",
      "readline",
      "readlines",
      "module_function",
      "global_variables",
      "define_method",
      "load",
      "attr_writer",
      "warn",
      "local_variables",
      "include",
      "exit",
      "exit!",
      "block_given?",
      "exec",
      "fail",
      "spawn",
      "autoload",
      "autoload?",
      "throw",
      "__dir__",
      "__method__",
      "printf",
      "extend",
      "binding",
      "fork",
      "format"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136
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
