/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_m4.c llamafile/is_keyword_m4.gperf  */
/* Computed positions: -k'1,4-7,9' */

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

#line 1 "llamafile/is_keyword_m4.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 166
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 25
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 439
/* maximum key range = 437, duplicates = 0 */

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
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
       30, 440,   5, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440,  40, 145,  70, 120,  40,
       10,   0,  10,  70,  65,   5,  35, 100,  35,   0,
        0,   0,  25, 155,  40,  55,   5,  15, 150,  10,
      145, 130,  10,   0, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440, 440, 440, 440,
      440, 440, 440, 440, 440, 440, 440
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[8]+1];
      /*FALLTHROUGH*/
      case 8:
      case 7:
        hval += asso_values[(unsigned char)str[6]];
      /*FALLTHROUGH*/
      case 6:
        hval += asso_values[(unsigned char)str[5]];
      /*FALLTHROUGH*/
      case 5:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
        hval += asso_values[(unsigned char)str[3]+1];
      /*FALLTHROUGH*/
      case 3:
      case 2:
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

const char *
is_keyword_m4 (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("os2")];
      char stringpool_str4[sizeof("eval")];
      char stringpool_str6[sizeof("m4_len")];
      char stringpool_str9[sizeof("exch")];
      char stringpool_str12[sizeof("m4_line")];
      char stringpool_str14[sizeof("defn")];
      char stringpool_str17[sizeof("m4_defn")];
      char stringpool_str22[sizeof("traceon")];
      char stringpool_str23[sizeof("maketemp")];
      char stringpool_str24[sizeof("file")];
      char stringpool_str27[sizeof("m4_cond")];
      char stringpool_str30[sizeof("ifdef")];
      char stringpool_str31[sizeof("divnum")];
      char stringpool_str33[sizeof("traceoff")];
      char stringpool_str34[sizeof("m4_define")];
      char stringpool_str37[sizeof("include")];
      char stringpool_str38[sizeof("len")];
      char stringpool_str39[sizeof("join")];
      char stringpool_str40[sizeof("m4_define_blind")];
      char stringpool_str41[sizeof("popdef")];
      char stringpool_str42[sizeof("forloop")];
      char stringpool_str44[sizeof("cond")];
      char stringpool_str46[sizeof("dquote")];
      char stringpool_str47[sizeof("mkstemp")];
      char stringpool_str48[sizeof("undefine")];
      char stringpool_str49[sizeof("line")];
      char stringpool_str51[sizeof("define")];
      char stringpool_str53[sizeof("m4_index")];
      char stringpool_str54[sizeof("copy")];
      char stringpool_str57[sizeof("pushdef")];
      char stringpool_str58[sizeof("m4_indir")];
      char stringpool_str60[sizeof("m4_dumpdef")];
      char stringpool_str62[sizeof("esyscmd")];
      char stringpool_str63[sizeof("m4_ifdef")];
      char stringpool_str64[sizeof("incr")];
      char stringpool_str66[sizeof("ifelse")];
      char stringpool_str67[sizeof("example")];
      char stringpool_str68[sizeof("m4_quote")];
      char stringpool_str69[sizeof("decr")];
      char stringpool_str71[sizeof("divert")];
      char stringpool_str73[sizeof("gnu")];
      char stringpool_str74[sizeof("argn")];
      char stringpool_str81[sizeof("syscmd")];
      char stringpool_str82[sizeof("windows")];
      char stringpool_str85[sizeof("indir")];
      char stringpool_str86[sizeof("m4_gnu")];
      char stringpool_str87[sizeof("m4_decr")];
      char stringpool_str90[sizeof("dquote_elt")];
      char stringpool_str91[sizeof("format")];
      char stringpool_str93[sizeof("m4_shift")];
      char stringpool_str97[sizeof("define_blind")];
      char stringpool_str99[sizeof("m4_ifelse")];
      char stringpool_str101[sizeof("__windows__")];
      char stringpool_str108[sizeof("translit")];
      char stringpool_str111[sizeof("substr")];
      char stringpool_str112[sizeof("m4_join")];
      char stringpool_str113[sizeof("m4_curry")];
      char stringpool_str115[sizeof("m4_joinall")];
      char stringpool_str116[sizeof("m4_os2")];
      char stringpool_str117[sizeof("m4_file")];
      char stringpool_str120[sizeof("m4_include")];
      char stringpool_str122[sizeof("m4_incr")];
      char stringpool_str123[sizeof("__line__")];
      char stringpool_str125[sizeof("nargs")];
      char stringpool_str129[sizeof("m4_cleardivert")];
      char stringpool_str130[sizeof("m4_foreach")];
      char stringpool_str131[sizeof("m4_foreachq")];
      char stringpool_str133[sizeof("sinclude")];
      char stringpool_str134[sizeof("m4_format")];
      char stringpool_str135[sizeof("shift")];
      char stringpool_str136[sizeof("m4_translit")];
      char stringpool_str137[sizeof("builtin")];
      char stringpool_str140[sizeof("m4_builtin")];
      char stringpool_str141[sizeof("m4_downcase")];
      char stringpool_str142[sizeof("__gnu__")];
      char stringpool_str144[sizeof("m4_rename")];
      char stringpool_str146[sizeof("m4exit")];
      char stringpool_str147[sizeof("m4_debugmode")];
      char stringpool_str149[sizeof("unix")];
      char stringpool_str152[sizeof("reverse")];
      char stringpool_str156[sizeof("m4wrap")];
      char stringpool_str158[sizeof("__file__")];
      char stringpool_str159[sizeof("changecom")];
      char stringpool_str161[sizeof("m4_errprint")];
      char stringpool_str162[sizeof("__os2__")];
      char stringpool_str164[sizeof("m4_divnum")];
      char stringpool_str165[sizeof("index")];
      char stringpool_str166[sizeof("rename")];
      char stringpool_str167[sizeof("m4_changecom")];
      char stringpool_str168[sizeof("m4_changeword")];
      char stringpool_str169[sizeof("m4_changequote")];
      char stringpool_str170[sizeof("m4_mkstemp")];
      char stringpool_str172[sizeof("m4_copy")];
      char stringpool_str174[sizeof("m4_m4exit")];
      char stringpool_str175[sizeof("quote")];
      char stringpool_str176[sizeof("sysval")];
      char stringpool_str179[sizeof("m4_divert")];
      char stringpool_str180[sizeof("m4_forloop")];
      char stringpool_str182[sizeof("joinall")];
      char stringpool_str183[sizeof("downcase")];
      char stringpool_str185[sizeof("changeword")];
      char stringpool_str186[sizeof("m4_undivert")];
      char stringpool_str188[sizeof("m4_nargs")];
      char stringpool_str189[sizeof("m4_dquote")];
      char stringpool_str191[sizeof("m4_maketemp")];
      char stringpool_str192[sizeof("dumpdef")];
      char stringpool_str193[sizeof("m4_dquote_elt")];
      char stringpool_str195[sizeof("capitalize")];
      char stringpool_str196[sizeof("upcase")];
      char stringpool_str200[sizeof("m4_traceon")];
      char stringpool_str201[sizeof("m4_traceoff")];
      char stringpool_str202[sizeof("foreach")];
      char stringpool_str203[sizeof("foreachq")];
      char stringpool_str206[sizeof("m4_undefine")];
      char stringpool_str208[sizeof("errprint")];
      char stringpool_str211[sizeof("m4_sinclude")];
      char stringpool_str214[sizeof("m4___windows__")];
      char stringpool_str216[sizeof("__program__")];
      char stringpool_str217[sizeof("m4_debugfile")];
      char stringpool_str219[sizeof("m4_m4wrap")];
      char stringpool_str220[sizeof("m4_reverse")];
      char stringpool_str221[sizeof("m4___file__")];
      char stringpool_str226[sizeof("regexp")];
      char stringpool_str228[sizeof("patsubst")];
      char stringpool_str230[sizeof("curry")];
      char stringpool_str235[sizeof("m4_example")];
      char stringpool_str236[sizeof("cleardivert")];
      char stringpool_str237[sizeof("m4_argn")];
      char stringpool_str238[sizeof("__unix__")];
      char stringpool_str239[sizeof("m4_syscmd")];
      char stringpool_str243[sizeof("m4_capitalize")];
      char stringpool_str246[sizeof("m4___line__")];
      char stringpool_str248[sizeof("undivert")];
      char stringpool_str249[sizeof("debugmode")];
      char stringpool_str259[sizeof("m4_substr")];
      char stringpool_str260[sizeof("m4_esyscmd")];
      char stringpool_str264[sizeof("debugfile")];
      char stringpool_str267[sizeof("m4_exch")];
      char stringpool_str269[sizeof("m4_popdef")];
      char stringpool_str271[sizeof("fatal_error")];
      char stringpool_str272[sizeof("m4_eval")];
      char stringpool_str278[sizeof("m4_array")];
      char stringpool_str281[sizeof("m4_stack_foreach")];
      char stringpool_str283[sizeof("stack_foreach")];
      char stringpool_str285[sizeof("m4_stack_foreach_sep")];
      char stringpool_str286[sizeof("m4_stack_foreach_lifo")];
      char stringpool_str287[sizeof("stack_foreach_sep")];
      char stringpool_str288[sizeof("stack_foreach_lifo")];
      char stringpool_str289[sizeof("m4_regexp")];
      char stringpool_str290[sizeof("m4_stack_foreach_sep_lifo")];
      char stringpool_str292[sizeof("stack_foreach_sep_lifo")];
      char stringpool_str301[sizeof("changequote")];
      char stringpool_str304[sizeof("m4_upcase")];
      char stringpool_str307[sizeof("m4_unix")];
      char stringpool_str310[sizeof("m4_pushdef")];
      char stringpool_str315[sizeof("m4_windows")];
      char stringpool_str325[sizeof("array")];
      char stringpool_str329[sizeof("m4___program__")];
      char stringpool_str336[sizeof("m4_patsubst")];
      char stringpool_str341[sizeof("m4___unix__")];
      char stringpool_str349[sizeof("m4_sysval")];
      char stringpool_str374[sizeof("m4_fatal_error")];
      char stringpool_str395[sizeof("m4___os2__")];
      char stringpool_str410[sizeof("m4___gnu__")];
      char stringpool_str427[sizeof("m4_array_set")];
      char stringpool_str439[sizeof("array_set")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "os2",
      "eval",
      "m4_len",
      "exch",
      "m4_line",
      "defn",
      "m4_defn",
      "traceon",
      "maketemp",
      "file",
      "m4_cond",
      "ifdef",
      "divnum",
      "traceoff",
      "m4_define",
      "include",
      "len",
      "join",
      "m4_define_blind",
      "popdef",
      "forloop",
      "cond",
      "dquote",
      "mkstemp",
      "undefine",
      "line",
      "define",
      "m4_index",
      "copy",
      "pushdef",
      "m4_indir",
      "m4_dumpdef",
      "esyscmd",
      "m4_ifdef",
      "incr",
      "ifelse",
      "example",
      "m4_quote",
      "decr",
      "divert",
      "gnu",
      "argn",
      "syscmd",
      "windows",
      "indir",
      "m4_gnu",
      "m4_decr",
      "dquote_elt",
      "format",
      "m4_shift",
      "define_blind",
      "m4_ifelse",
      "__windows__",
      "translit",
      "substr",
      "m4_join",
      "m4_curry",
      "m4_joinall",
      "m4_os2",
      "m4_file",
      "m4_include",
      "m4_incr",
      "__line__",
      "nargs",
      "m4_cleardivert",
      "m4_foreach",
      "m4_foreachq",
      "sinclude",
      "m4_format",
      "shift",
      "m4_translit",
      "builtin",
      "m4_builtin",
      "m4_downcase",
      "__gnu__",
      "m4_rename",
      "m4exit",
      "m4_debugmode",
      "unix",
      "reverse",
      "m4wrap",
      "__file__",
      "changecom",
      "m4_errprint",
      "__os2__",
      "m4_divnum",
      "index",
      "rename",
      "m4_changecom",
      "m4_changeword",
      "m4_changequote",
      "m4_mkstemp",
      "m4_copy",
      "m4_m4exit",
      "quote",
      "sysval",
      "m4_divert",
      "m4_forloop",
      "joinall",
      "downcase",
      "changeword",
      "m4_undivert",
      "m4_nargs",
      "m4_dquote",
      "m4_maketemp",
      "dumpdef",
      "m4_dquote_elt",
      "capitalize",
      "upcase",
      "m4_traceon",
      "m4_traceoff",
      "foreach",
      "foreachq",
      "m4_undefine",
      "errprint",
      "m4_sinclude",
      "m4___windows__",
      "__program__",
      "m4_debugfile",
      "m4_m4wrap",
      "m4_reverse",
      "m4___file__",
      "regexp",
      "patsubst",
      "curry",
      "m4_example",
      "cleardivert",
      "m4_argn",
      "__unix__",
      "m4_syscmd",
      "m4_capitalize",
      "m4___line__",
      "undivert",
      "debugmode",
      "m4_substr",
      "m4_esyscmd",
      "debugfile",
      "m4_exch",
      "m4_popdef",
      "fatal_error",
      "m4_eval",
      "m4_array",
      "m4_stack_foreach",
      "stack_foreach",
      "m4_stack_foreach_sep",
      "m4_stack_foreach_lifo",
      "stack_foreach_sep",
      "stack_foreach_lifo",
      "m4_regexp",
      "m4_stack_foreach_sep_lifo",
      "stack_foreach_sep_lifo",
      "changequote",
      "m4_upcase",
      "m4_unix",
      "m4_pushdef",
      "m4_windows",
      "array",
      "m4___program__",
      "m4_patsubst",
      "m4___unix__",
      "m4_sysval",
      "m4_fatal_error",
      "m4___os2__",
      "m4___gnu__",
      "m4_array_set",
      "array_set"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str141,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      -1, -1, -1,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str175,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str182,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      -1,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str196,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str200,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str201,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str202,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str203,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str208,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str211,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str217,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str220,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str221,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str226,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str228,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str230,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str235,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str236,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str237,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str238,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str239,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str243,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str246,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str248,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str249,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str259,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str260,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str264,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str267,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str269,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str271,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str272,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str278,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str281,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str283,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str285,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str286,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str287,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str288,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str289,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str290,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str292,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str301,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str304,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str307,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str310,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str315,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str325,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str329,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str336,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str341,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str349,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str374,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str395,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str410,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str427,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str439
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
