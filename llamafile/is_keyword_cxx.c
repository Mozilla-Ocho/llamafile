/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_cxx.gperf  */
/* Computed positions: -k'1,3,5,$' */

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

#line 1 "llamafile/is_keyword_cxx.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 165
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 27
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 450
/* maximum key range = 448, duplicates = 0 */

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
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451,  30, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451,  10,  60,
       50,  30,  35, 451,  25, 451,   0, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451,   0,  10,
       10, 451, 451, 451, 451, 451, 451, 451,   5, 451,
        5, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451,   0, 451,  15, 130,   0,
       20,  10,  80, 145, 140,  45, 451,  30,  70,  35,
        0,  65,  95, 120,   5,  15,   0, 110,  85, 140,
       85,   5,  10, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451, 451, 451, 451, 451,
      451, 451, 451, 451, 451, 451
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
      case 3:
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
is_keyword_cxx (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("not")];
      char stringpool_str5[sizeof("const")];
      char stringpool_str7[sizeof("__const")];
      char stringpool_str9[sizeof("constinit")];
      char stringpool_str10[sizeof("const_cast")];
      char stringpool_str12[sizeof("__constant__")];
      char stringpool_str13[sizeof("try")];
      char stringpool_str14[sizeof("constexpr")];
      char stringpool_str16[sizeof("return")];
      char stringpool_str17[sizeof("concept")];
      char stringpool_str18[sizeof("noexcept")];
      char stringpool_str19[sizeof("co_return")];
      char stringpool_str21[sizeof("extern")];
      char stringpool_str22[sizeof("char8_t")];
      char stringpool_str23[sizeof("__extension__")];
      char stringpool_str24[sizeof("char")];
      char stringpool_str26[sizeof("struct")];
      char stringpool_str27[sizeof("__FUNCTION__")];
      char stringpool_str28[sizeof("__attribute__")];
      char stringpool_str29[sizeof("case")];
      char stringpool_str30[sizeof("__restrict")];
      char stringpool_str32[sizeof("__restrict__")];
      char stringpool_str33[sizeof("__thread")];
      char stringpool_str34[sizeof("__PRETTY_FUNCTION__")];
      char stringpool_str36[sizeof("__attribute")];
      char stringpool_str38[sizeof("decltype")];
      char stringpool_str39[sizeof("else")];
      char stringpool_str40[sizeof("__shared__")];
      char stringpool_str43[sizeof("strftimeesque")];
      char stringpool_str44[sizeof("returnsaligned")];
      char stringpool_str46[sizeof("__complex__")];
      char stringpool_str47[sizeof("nosideeffect")];
      char stringpool_str48[sizeof("int")];
      char stringpool_str50[sizeof("class")];
      char stringpool_str51[sizeof("strlenesque")];
      char stringpool_str52[sizeof("returnspointerwithnoaliases")];
      char stringpool_str53[sizeof("char32_t")];
      char stringpool_str57[sizeof("__asm__")];
      char stringpool_str58[sizeof("and")];
      char stringpool_str59[sizeof("__real__ ")];
      char stringpool_str61[sizeof("interruptfn")];
      char stringpool_str62[sizeof("dontcallback")];
      char stringpool_str63[sizeof("continue")];
      char stringpool_str64[sizeof("this")];
      char stringpool_str66[sizeof("reinterpret_cast")];
      char stringpool_str67[sizeof("dynamic_cast")];
      char stringpool_str69[sizeof("namespace")];
      char stringpool_str71[sizeof("__alignof__")];
      char stringpool_str72[sizeof("or")];
      char stringpool_str73[sizeof("co_yield")];
      char stringpool_str79[sizeof("consteval")];
      char stringpool_str81[sizeof("static")];
      char stringpool_str82[sizeof("alignas")];
      char stringpool_str83[sizeof("char16_t")];
      char stringpool_str84[sizeof("auto")];
      char stringpool_str85[sizeof("short")];
      char stringpool_str86[sizeof("static_cast")];
      char stringpool_str87[sizeof("do")];
      char stringpool_str88[sizeof("asm")];
      char stringpool_str90[sizeof("__asm")];
      char stringpool_str92[sizeof("thatispacked")];
      char stringpool_str93[sizeof("for")];
      char stringpool_str94[sizeof("_Float128")];
      char stringpool_str96[sizeof("__real")];
      char stringpool_str98[sizeof("xor")];
      char stringpool_str99[sizeof("__imag__ ")];
      char stringpool_str100[sizeof("__forceinline__")];
      char stringpool_str102[sizeof("thread_local")];
      char stringpool_str103[sizeof("operator")];
      char stringpool_str105[sizeof("forcealign")];
      char stringpool_str106[sizeof("delete")];
      char stringpool_str107[sizeof("nocallersavedregisters")];
      char stringpool_str108[sizeof("reflexpr")];
      char stringpool_str109[sizeof("pureconst")];
      char stringpool_str112[sizeof("reallocesque")];
      char stringpool_str113[sizeof("typename")];
      char stringpool_str114[sizeof("threadIdx")];
      char stringpool_str115[sizeof("__device__")];
      char stringpool_str116[sizeof("export")];
      char stringpool_str118[sizeof("_Float16")];
      char stringpool_str120[sizeof("forcealignargpointer")];
      char stringpool_str123[sizeof("template")];
      char stringpool_str125[sizeof("__inline__")];
      char stringpool_str127[sizeof("if")];
      char stringpool_str128[sizeof("_Float64")];
      char stringpool_str129[sizeof("__complex")];
      char stringpool_str130[sizeof("scanfesque")];
      char stringpool_str131[sizeof("inline")];
      char stringpool_str133[sizeof("__inline")];
      char stringpool_str136[sizeof("not_eq")];
      char stringpool_str138[sizeof("atomic_commit")];
      char stringpool_str140[sizeof("atomic_noexcept")];
      char stringpool_str143[sizeof("_Float32")];
      char stringpool_str145[sizeof("bitor")];
      char stringpool_str146[sizeof("__null")];
      char stringpool_str147[sizeof("alignof")];
      char stringpool_str148[sizeof("co_await")];
      char stringpool_str149[sizeof("__alignof")];
      char stringpool_str150[sizeof("float")];
      char stringpool_str151[sizeof("friend")];
      char stringpool_str154[sizeof("void")];
      char stringpool_str155[sizeof("wontreturn")];
      char stringpool_str156[sizeof("bitand")];
      char stringpool_str158[sizeof("explicit")];
      char stringpool_str159[sizeof("enum")];
      char stringpool_str160[sizeof("union")];
      char stringpool_str161[sizeof("printfesque")];
      char stringpool_str163[sizeof("__host__")];
      char stringpool_str164[sizeof("__float80")];
      char stringpool_str166[sizeof("typeid")];
      char stringpool_str169[sizeof("dontthrow")];
      char stringpool_str170[sizeof("__signed__")];
      char stringpool_str171[sizeof("and_eq")];
      char stringpool_str172[sizeof("private")];
      char stringpool_str173[sizeof("volatile")];
      char stringpool_str175[sizeof("__volatile")];
      char stringpool_str176[sizeof("sizeof")];
      char stringpool_str177[sizeof("nullptr")];
      char stringpool_str178[sizeof("register")];
      char stringpool_str180[sizeof("compl")];
      char stringpool_str182[sizeof("mutable")];
      char stringpool_str183[sizeof("__typeof")];
      char stringpool_str186[sizeof("memcpyesque")];
      char stringpool_str187[sizeof("synchronized")];
      char stringpool_str188[sizeof("__signed")];
      char stringpool_str191[sizeof("mallocesque")];
      char stringpool_str193[sizeof("requires")];
      char stringpool_str196[sizeof("signed")];
      char stringpool_str197[sizeof("__launch_bounds__")];
      char stringpool_str198[sizeof("__volatile__ ")];
      char stringpool_str199[sizeof("protected")];
      char stringpool_str202[sizeof("typedef")];
      char stringpool_str205[sizeof("break")];
      char stringpool_str206[sizeof("switch")];
      char stringpool_str208[sizeof("atomic_cancel")];
      char stringpool_str209[sizeof("__label__")];
      char stringpool_str210[sizeof("while")];
      char stringpool_str211[sizeof("__imag")];
      char stringpool_str214[sizeof("goto")];
      char stringpool_str215[sizeof("privileged")];
      char stringpool_str216[sizeof("double")];
      char stringpool_str217[sizeof("default")];
      char stringpool_str218[sizeof("paramsnonnull")];
      char stringpool_str219[sizeof("long")];
      char stringpool_str220[sizeof("__global__")];
      char stringpool_str221[sizeof("__bf16")];
      char stringpool_str226[sizeof("xor_eq")];
      char stringpool_str229[sizeof("libcesque")];
      char stringpool_str232[sizeof("gridDim")];
      char stringpool_str241[sizeof("vallocesque")];
      char stringpool_str251[sizeof("textwindows")];
      char stringpool_str268[sizeof("blockDim")];
      char stringpool_str269[sizeof("bool")];
      char stringpool_str273[sizeof("__builtin_offsetof")];
      char stringpool_str276[sizeof("public")];
      char stringpool_str277[sizeof("virtual")];
      char stringpool_str283[sizeof("new")];
      char stringpool_str285[sizeof("catch")];
      char stringpool_str290[sizeof("throw")];
      char stringpool_str292[sizeof("wchar_t")];
      char stringpool_str298[sizeof("unsigned")];
      char stringpool_str310[sizeof("or_eq")];
      char stringpool_str318[sizeof("blockIdx")];
      char stringpool_str336[sizeof("__builtin_va_arg")];
      char stringpool_str450[sizeof("using")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "not",
      "const",
      "__const",
      "constinit",
      "const_cast",
      "__constant__",
      "try",
      "constexpr",
      "return",
      "concept",
      "noexcept",
      "co_return",
      "extern",
      "char8_t",
      "__extension__",
      "char",
      "struct",
      "__FUNCTION__",
      "__attribute__",
      "case",
      "__restrict",
      "__restrict__",
      "__thread",
      "__PRETTY_FUNCTION__",
      "__attribute",
      "decltype",
      "else",
      "__shared__",
      "strftimeesque",
      "returnsaligned",
      "__complex__",
      "nosideeffect",
      "int",
      "class",
      "strlenesque",
      "returnspointerwithnoaliases",
      "char32_t",
      "__asm__",
      "and",
      "__real__ ",
      "interruptfn",
      "dontcallback",
      "continue",
      "this",
      "reinterpret_cast",
      "dynamic_cast",
      "namespace",
      "__alignof__",
      "or",
      "co_yield",
      "consteval",
      "static",
      "alignas",
      "char16_t",
      "auto",
      "short",
      "static_cast",
      "do",
      "asm",
      "__asm",
      "thatispacked",
      "for",
      "_Float128",
      "__real",
      "xor",
      "__imag__ ",
      "__forceinline__",
      "thread_local",
      "operator",
      "forcealign",
      "delete",
      "nocallersavedregisters",
      "reflexpr",
      "pureconst",
      "reallocesque",
      "typename",
      "threadIdx",
      "__device__",
      "export",
      "_Float16",
      "forcealignargpointer",
      "template",
      "__inline__",
      "if",
      "_Float64",
      "__complex",
      "scanfesque",
      "inline",
      "__inline",
      "not_eq",
      "atomic_commit",
      "atomic_noexcept",
      "_Float32",
      "bitor",
      "__null",
      "alignof",
      "co_await",
      "__alignof",
      "float",
      "friend",
      "void",
      "wontreturn",
      "bitand",
      "explicit",
      "enum",
      "union",
      "printfesque",
      "__host__",
      "__float80",
      "typeid",
      "dontthrow",
      "__signed__",
      "and_eq",
      "private",
      "volatile",
      "__volatile",
      "sizeof",
      "nullptr",
      "register",
      "compl",
      "mutable",
      "__typeof",
      "memcpyesque",
      "synchronized",
      "__signed",
      "mallocesque",
      "requires",
      "signed",
      "__launch_bounds__",
      "__volatile__ ",
      "protected",
      "typedef",
      "break",
      "switch",
      "atomic_cancel",
      "__label__",
      "while",
      "__imag",
      "goto",
      "privileged",
      "double",
      "default",
      "paramsnonnull",
      "long",
      "__global__",
      "__bf16",
      "xor_eq",
      "libcesque",
      "gridDim",
      "vallocesque",
      "textwindows",
      "blockDim",
      "bool",
      "__builtin_offsetof",
      "public",
      "virtual",
      "new",
      "catch",
      "throw",
      "wchar_t",
      "unsigned",
      "or_eq",
      "blockIdx",
      "__builtin_va_arg",
      "using"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1,
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
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str175,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str177,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str182,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str187,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str196,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str197,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str198,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str199,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str202,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str205,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str208,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str209,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str210,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str211,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str215,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str217,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str220,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str221,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str226,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str232,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str241,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str251,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str268,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str269,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str273,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str276,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str277,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str283,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str285,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str290,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str292,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str298,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str310,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str318,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str336,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str450
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
