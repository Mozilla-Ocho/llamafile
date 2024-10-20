/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_c.gperf  */
/* Computed positions: -k'1,4-5,$' */

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

#line 1 "llamafile/is_keyword_c.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 155
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 27
#define MIN_HASH_VALUE 4
#define MAX_HASH_VALUE 297
/* maximum key range = 294, duplicates = 0 */

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
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298,  20, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298,  20, 298,
        5, 298,  60, 298,  80, 298,   0, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298,  15, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298,   0, 298, 110,  95,  35,
       85,   0,  65, 115,  65, 105,  30,   0,  40,  40,
       20,   0, 100,  30,  15,  10,   5,   5,   0,  55,
      175,  75,  15, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298, 298, 298, 298,
      298, 298, 298, 298, 298, 298, 298
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[4]+1];
      /*FALLTHROUGH*/
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
is_keyword_c (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str4[sizeof("else")];
      char stringpool_str10[sizeof("__fentry__")];
      char stringpool_str12[sizeof("__const")];
      char stringpool_str13[sizeof("__func__")];
      char stringpool_str14[sizeof("__funline")];
      char stringpool_str15[sizeof("__return__")];
      char stringpool_str17[sizeof("__restrict__")];
      char stringpool_str18[sizeof("__word__")];
      char stringpool_str20[sizeof("__restrict")];
      char stringpool_str21[sizeof("__attribute")];
      char stringpool_str22[sizeof("__noreturn__")];
      char stringpool_str23[sizeof("__attribute__")];
      char stringpool_str26[sizeof("__strfmon__")];
      char stringpool_str27[sizeof("__strftime__")];
      char stringpool_str30[sizeof("union")];
      char stringpool_str31[sizeof("__complex__")];
      char stringpool_str32[sizeof("_BitInt")];
      char stringpool_str34[sizeof("__gnu_inline__")];
      char stringpool_str36[sizeof("extern")];
      char stringpool_str37[sizeof("__asm__")];
      char stringpool_str39[sizeof("case")];
      char stringpool_str40[sizeof("short")];
      char stringpool_str41[sizeof("__pointer__")];
      char stringpool_str42[sizeof("__noinline__")];
      char stringpool_str43[sizeof("restrict")];
      char stringpool_str47[sizeof("__builtin___")];
      char stringpool_str50[sizeof("__volatile")];
      char stringpool_str51[sizeof("__cmn_err__")];
      char stringpool_str52[sizeof("__volatile__")];
      char stringpool_str55[sizeof("__printf__")];
      char stringpool_str56[sizeof("return")];
      char stringpool_str58[sizeof("__unix__")];
      char stringpool_str60[sizeof("const")];
      char stringpool_str62[sizeof("_Atomic")];
      char stringpool_str63[sizeof("static_assert")];
      char stringpool_str67[sizeof("returnspointerwithnoaliases")];
      char stringpool_str68[sizeof("__inline")];
      char stringpool_str69[sizeof("char")];
      char stringpool_str70[sizeof("__inline__")];
      char stringpool_str73[sizeof("__volatile__ ")];
      char stringpool_str74[sizeof("constexpr")];
      char stringpool_str75[sizeof("__asm")];
      char stringpool_str76[sizeof("_Decimal128")];
      char stringpool_str77[sizeof("typedef")];
      char stringpool_str78[sizeof("continue")];
      char stringpool_str80[sizeof("_Decimal32")];
      char stringpool_str81[sizeof("__alignof__")];
      char stringpool_str83[sizeof("for")];
      char stringpool_str84[sizeof("enum")];
      char stringpool_str85[sizeof("_Bool")];
      char stringpool_str86[sizeof("static")];
      char stringpool_str87[sizeof("do")];
      char stringpool_str88[sizeof("__byte__")];
      char stringpool_str91[sizeof("__null")];
      char stringpool_str93[sizeof("strftimeesque")];
      char stringpool_str96[sizeof("__noclone__")];
      char stringpool_str100[sizeof("wontreturn")];
      char stringpool_str103[sizeof("__real__")];
      char stringpool_str104[sizeof("_Float128")];
      char stringpool_str105[sizeof("__symver__")];
      char stringpool_str107[sizeof("reallocesque")];
      char stringpool_str108[sizeof("_Float32")];
      char stringpool_str109[sizeof("_Noreturn")];
      char stringpool_str110[sizeof("__muarch__")];
      char stringpool_str111[sizeof("struct")];
      char stringpool_str112[sizeof("nullptr")];
      char stringpool_str113[sizeof("int")];
      char stringpool_str114[sizeof("auto")];
      char stringpool_str115[sizeof("__typeof__")];
      char stringpool_str116[sizeof("memcpyesque")];
      char stringpool_str118[sizeof("__builtin_offsetof")];
      char stringpool_str119[sizeof("goto")];
      char stringpool_str122[sizeof("__typeof_unqual__")];
      char stringpool_str123[sizeof("volatile")];
      char stringpool_str124[sizeof("__real__ ")];
      char stringpool_str126[sizeof("strlenesque")];
      char stringpool_str128[sizeof("_Generic")];
      char stringpool_str129[sizeof("returnsaligned")];
      char stringpool_str130[sizeof("__target__")];
      char stringpool_str131[sizeof("__transparent_union__")];
      char stringpool_str132[sizeof("__hardbool__")];
      char stringpool_str133[sizeof("_Thread_local")];
      char stringpool_str134[sizeof("_Static_assert")];
      char stringpool_str135[sizeof("_Decimal64")];
      char stringpool_str136[sizeof("printfesque")];
      char stringpool_str137[sizeof("thatispacked")];
      char stringpool_str138[sizeof("__may_alias__")];
      char stringpool_str139[sizeof("__scanf__")];
      char stringpool_str140[sizeof("__mcarch__")];
      char stringpool_str141[sizeof("__real")];
      char stringpool_str142[sizeof("nosideeffect")];
      char stringpool_str143[sizeof("__imag__")];
      char stringpool_str144[sizeof("__alignof")];
      char stringpool_str146[sizeof("interruptfn")];
      char stringpool_str148[sizeof("register")];
      char stringpool_str149[sizeof("libcesque")];
      char stringpool_str151[sizeof("vallocesque")];
      char stringpool_str152[sizeof("thread_local")];
      char stringpool_str153[sizeof("asm")];
      char stringpool_str154[sizeof("__label__")];
      char stringpool_str155[sizeof("scanfesque")];
      char stringpool_str156[sizeof("__bf16")];
      char stringpool_str158[sizeof("typeof_unqual")];
      char stringpool_str159[sizeof("dontthrow")];
      char stringpool_str160[sizeof("__mcffpu__")];
      char stringpool_str161[sizeof("__mcfarch__")];
      char stringpool_str162[sizeof("__mcfhwdiv__")];
      char stringpool_str163[sizeof("_Float64")];
      char stringpool_str164[sizeof("__imag__ ")];
      char stringpool_str165[sizeof("while")];
      char stringpool_str166[sizeof("__builtin_va_arg")];
      char stringpool_str168[sizeof("__thread")];
      char stringpool_str169[sizeof("__float80")];
      char stringpool_str171[sizeof("switch")];
      char stringpool_str172[sizeof("if")];
      char stringpool_str174[sizeof("void")];
      char stringpool_str176[sizeof("typeof")];
      char stringpool_str177[sizeof("__pie__")];
      char stringpool_str178[sizeof("__typeof")];
      char stringpool_str179[sizeof("bool")];
      char stringpool_str180[sizeof("__signed__")];
      char stringpool_str181[sizeof("sizeof")];
      char stringpool_str183[sizeof("_Float16")];
      char stringpool_str186[sizeof("signed")];
      char stringpool_str187[sizeof("dontcallback")];
      char stringpool_str188[sizeof("_Alignas")];
      char stringpool_str190[sizeof("float")];
      char stringpool_str191[sizeof("mallocesque")];
      char stringpool_str192[sizeof("__vax__")];
      char stringpool_str193[sizeof("__extension__")];
      char stringpool_str195[sizeof("forcealign")];
      char stringpool_str197[sizeof("__pic__")];
      char stringpool_str199[sizeof("pureconst")];
      char stringpool_str200[sizeof("forcealignargpointer")];
      char stringpool_str202[sizeof("nocallersavedregisters")];
      char stringpool_str204[sizeof("__complex")];
      char stringpool_str205[sizeof("__packed__")];
      char stringpool_str206[sizeof("textwindows")];
      char stringpool_str207[sizeof("default")];
      char stringpool_str216[sizeof("inline")];
      char stringpool_str217[sizeof("__msabi")];
      char stringpool_str225[sizeof("privileged")];
      char stringpool_str226[sizeof("double")];
      char stringpool_str232[sizeof("__always_inline__")];
      char stringpool_str242[sizeof("alignas")];
      char stringpool_str243[sizeof("_Alignof")];
      char stringpool_str250[sizeof("break")];
      char stringpool_str253[sizeof("_Complex")];
      char stringpool_str256[sizeof("__imag")];
      char stringpool_str260[sizeof("_Imaginary")];
      char stringpool_str263[sizeof("__signed")];
      char stringpool_str268[sizeof("unsigned")];
      char stringpool_str274[sizeof("long")];
      char stringpool_str283[sizeof("paramsnonnull")];
      char stringpool_str297[sizeof("alignof")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "else",
      "__fentry__",
      "__const",
      "__func__",
      "__funline",
      "__return__",
      "__restrict__",
      "__word__",
      "__restrict",
      "__attribute",
      "__noreturn__",
      "__attribute__",
      "__strfmon__",
      "__strftime__",
      "union",
      "__complex__",
      "_BitInt",
      "__gnu_inline__",
      "extern",
      "__asm__",
      "case",
      "short",
      "__pointer__",
      "__noinline__",
      "restrict",
      "__builtin___",
      "__volatile",
      "__cmn_err__",
      "__volatile__",
      "__printf__",
      "return",
      "__unix__",
      "const",
      "_Atomic",
      "static_assert",
      "returnspointerwithnoaliases",
      "__inline",
      "char",
      "__inline__",
      "__volatile__ ",
      "constexpr",
      "__asm",
      "_Decimal128",
      "typedef",
      "continue",
      "_Decimal32",
      "__alignof__",
      "for",
      "enum",
      "_Bool",
      "static",
      "do",
      "__byte__",
      "__null",
      "strftimeesque",
      "__noclone__",
      "wontreturn",
      "__real__",
      "_Float128",
      "__symver__",
      "reallocesque",
      "_Float32",
      "_Noreturn",
      "__muarch__",
      "struct",
      "nullptr",
      "int",
      "auto",
      "__typeof__",
      "memcpyesque",
      "__builtin_offsetof",
      "goto",
      "__typeof_unqual__",
      "volatile",
      "__real__ ",
      "strlenesque",
      "_Generic",
      "returnsaligned",
      "__target__",
      "__transparent_union__",
      "__hardbool__",
      "_Thread_local",
      "_Static_assert",
      "_Decimal64",
      "printfesque",
      "thatispacked",
      "__may_alias__",
      "__scanf__",
      "__mcarch__",
      "__real",
      "nosideeffect",
      "__imag__",
      "__alignof",
      "interruptfn",
      "register",
      "libcesque",
      "vallocesque",
      "thread_local",
      "asm",
      "__label__",
      "scanfesque",
      "__bf16",
      "typeof_unqual",
      "dontthrow",
      "__mcffpu__",
      "__mcfarch__",
      "__mcfhwdiv__",
      "_Float64",
      "__imag__ ",
      "while",
      "__builtin_va_arg",
      "__thread",
      "__float80",
      "switch",
      "if",
      "void",
      "typeof",
      "__pie__",
      "__typeof",
      "bool",
      "__signed__",
      "sizeof",
      "_Float16",
      "signed",
      "dontcallback",
      "_Alignas",
      "float",
      "mallocesque",
      "__vax__",
      "__extension__",
      "forcealign",
      "__pic__",
      "pureconst",
      "forcealignargpointer",
      "nocallersavedregisters",
      "__complex",
      "__packed__",
      "textwindows",
      "default",
      "inline",
      "__msabi",
      "privileged",
      "double",
      "__always_inline__",
      "alignas",
      "_Alignof",
      "break",
      "_Complex",
      "__imag",
      "_Imaginary",
      "__signed",
      "unsigned",
      "long",
      "paramsnonnull",
      "alignof"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      -1, -1, -1, -1, -1,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str141,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str177,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str181,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str187,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str192,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str195,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str197,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str199,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str200,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str202,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str204,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str205,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str207,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str217,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str225,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str226,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str232,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str242,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str243,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str250,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str253,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str256,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str260,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str263,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str268,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str274,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str283,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str297
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
