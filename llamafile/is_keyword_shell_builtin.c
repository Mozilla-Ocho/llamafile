/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_shell_builtin.gperf  */
/* Computed positions: -k'1-2' */

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

#line 1 "llamafile/is_keyword_shell_builtin.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 40
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 9
#define MIN_HASH_VALUE 4
#define MAX_HASH_VALUE 64
/* maximum key range = 61, duplicates = 0 */

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
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 25, 25,  5,
      45,  5, 15, 10, 10, 20,  1, 40,  5, 30,
      15,  0,  5, 65, 20,  0,  0, 10, 50, 25,
      65, 40, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
      65, 65, 65, 65, 65, 65
    };
  return len + asso_values[(unsigned char)str[1]] + asso_values[(unsigned char)str[0]];
}

const char *
is_keyword_shell_builtin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str4[sizeof("stop")];
      char stringpool_str5[sizeof("jobs")];
      char stringpool_str6[sizeof("source")];
      char stringpool_str8[sizeof("set")];
      char stringpool_str9[sizeof("test")];
      char stringpool_str10[sizeof("local")];
      char stringpool_str11[sizeof("logout")];
      char stringpool_str12[sizeof("command")];
      char stringpool_str13[sizeof("let")];
      char stringpool_str14[sizeof("echo")];
      char stringpool_str15[sizeof("shift")];
      char stringpool_str17[sizeof("suspend")];
      char stringpool_str19[sizeof("help")];
      char stringpool_str20[sizeof("chdir")];
      char stringpool_str21[sizeof("ulimit")];
      char stringpool_str22[sizeof("getopts")];
      char stringpool_str24[sizeof("true")];
      char stringpool_str25[sizeof("times")];
      char stringpool_str26[sizeof("enable")];
      char stringpool_str27[sizeof("fg")];
      char stringpool_str29[sizeof("read")];
      char stringpool_str30[sizeof("unset")];
      char stringpool_str31[sizeof("printf")];
      char stringpool_str32[sizeof("unalias")];
      char stringpool_str34[sizeof("readarray")];
      char stringpool_str35[sizeof("alias")];
      char stringpool_str36[sizeof("caller")];
      char stringpool_str37[sizeof("bg")];
      char stringpool_str39[sizeof("hash")];
      char stringpool_str42[sizeof("builtin")];
      char stringpool_str44[sizeof("type")];
      char stringpool_str45[sizeof("false")];
      char stringpool_str47[sizeof("typeset")];
      char stringpool_str49[sizeof("bind")];
      char stringpool_str52[sizeof("cd")];
      char stringpool_str54[sizeof("wait")];
      char stringpool_str57[sizeof("declare")];
      char stringpool_str59[sizeof("eval")];
      char stringpool_str62[sizeof("mapfile")];
      char stringpool_str64[sizeof("kill")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "stop",
      "jobs",
      "source",
      "set",
      "test",
      "local",
      "logout",
      "command",
      "let",
      "echo",
      "shift",
      "suspend",
      "help",
      "chdir",
      "ulimit",
      "getopts",
      "true",
      "times",
      "enable",
      "fg",
      "read",
      "unset",
      "printf",
      "unalias",
      "readarray",
      "alias",
      "caller",
      "bg",
      "hash",
      "builtin",
      "type",
      "false",
      "typeset",
      "bind",
      "cd",
      "wait",
      "declare",
      "eval",
      "mapfile",
      "kill"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64
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
