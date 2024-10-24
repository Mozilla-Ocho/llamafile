/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_cxx.c llamafile/is_keyword_cxx.gperf  */
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

#define TOTAL_KEYWORDS 173
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 27
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 508
/* maximum key range = 506, duplicates = 0 */

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
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509,  30, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509,  10,  40,
       40,  35,  20, 509,  10, 509,  15, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509,   0,  15,
        0, 509, 509, 509, 509, 509, 509, 509,   0, 509,
       10, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509,   0, 509,  25, 155,   0,
       80,  10,  35, 110, 145,  30, 509,  50,  60,  90,
       10, 125, 100,  55,   5,  25,   0,  65, 140,  75,
      170,   0,   0, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509, 509, 509, 509, 509,
      509, 509, 509, 509, 509, 509
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
      char stringpool_str3[sizeof("try")];
      char stringpool_str12[sizeof("__FUNCTION__")];
      char stringpool_str13[sizeof("not")];
      char stringpool_str15[sizeof("const")];
      char stringpool_str17[sizeof("__const")];
      char stringpool_str19[sizeof("constinit")];
      char stringpool_str20[sizeof("const_cast")];
      char stringpool_str22[sizeof("__constant__")];
      char stringpool_str23[sizeof("__extension__")];
      char stringpool_str24[sizeof("constexpr")];
      char stringpool_str26[sizeof("return")];
      char stringpool_str27[sizeof("concept")];
      char stringpool_str28[sizeof("noexcept")];
      char stringpool_str29[sizeof("co_return")];
      char stringpool_str31[sizeof("extern")];
      char stringpool_str32[sizeof("returnstwice")];
      char stringpool_str33[sizeof("int")];
      char stringpool_str34[sizeof("char")];
      char stringpool_str36[sizeof("struct")];
      char stringpool_str38[sizeof("__attribute__")];
      char stringpool_str39[sizeof("case")];
      char stringpool_str40[sizeof("__restrict")];
      char stringpool_str42[sizeof("__restrict__")];
      char stringpool_str43[sizeof("autotype")];
      char stringpool_str44[sizeof("__PRETTY_FUNCTION__")];
      char stringpool_str46[sizeof("__attribute")];
      char stringpool_str47[sizeof("char8_t")];
      char stringpool_str48[sizeof("for")];
      char stringpool_str49[sizeof("else")];
      char stringpool_str51[sizeof("reinterpret_cast")];
      char stringpool_str53[sizeof("strftimeesque")];
      char stringpool_str55[sizeof("__forceinline__")];
      char stringpool_str56[sizeof("interruptfn")];
      char stringpool_str58[sizeof("continue")];
      char stringpool_str59[sizeof("this")];
      char stringpool_str60[sizeof("__shared__")];
      char stringpool_str61[sizeof("strlenesque")];
      char stringpool_str62[sizeof("returnspointerwithnoaliases")];
      char stringpool_str63[sizeof("reflexpr")];
      char stringpool_str66[sizeof("__alignof__")];
      char stringpool_str67[sizeof("if")];
      char stringpool_str68[sizeof("char32_t")];
      char stringpool_str69[sizeof("__real__ ")];
      char stringpool_str70[sizeof("forcealign")];
      char stringpool_str71[sizeof("forceinline")];
      char stringpool_str73[sizeof("char16_t")];
      char stringpool_str75[sizeof("forcealignargpointer")];
      char stringpool_str79[sizeof("consteval")];
      char stringpool_str80[sizeof("class")];
      char stringpool_str81[sizeof("not_eq")];
      char stringpool_str83[sizeof("co_await")];
      char stringpool_str84[sizeof("returnsnonnull")];
      char stringpool_str86[sizeof("static")];
      char stringpool_str90[sizeof("__wur")];
      char stringpool_str91[sizeof("static_cast")];
      char stringpool_str93[sizeof("__thread")];
      char stringpool_str94[sizeof("__imag__ ")];
      char stringpool_str96[sizeof("__real")];
      char stringpool_str97[sizeof("alignas")];
      char stringpool_str98[sizeof("decltype")];
      char stringpool_str99[sizeof("__alignof")];
      char stringpool_str100[sizeof("__inline__")];
      char stringpool_str101[sizeof("__complex__")];
      char stringpool_str102[sizeof("thread_local")];
      char stringpool_str103[sizeof("_Float16")];
      char stringpool_str104[sizeof("returnsaligned")];
      char stringpool_str105[sizeof("scanfesque")];
      char stringpool_str107[sizeof("alignof")];
      char stringpool_str108[sizeof("__inline")];
      char stringpool_str109[sizeof("_Float128")];
      char stringpool_str110[sizeof("wontreturn")];
      char stringpool_str112[sizeof("reallocesque")];
      char stringpool_str113[sizeof("_Float64")];
      char stringpool_str114[sizeof("pureconst")];
      char stringpool_str116[sizeof("inline")];
      char stringpool_str117[sizeof("nocallersavedregisters")];
      char stringpool_str118[sizeof("co_yield")];
      char stringpool_str120[sizeof("union")];
      char stringpool_str121[sizeof("export")];
      char stringpool_str122[sizeof("__asm__")];
      char stringpool_str123[sizeof("requires")];
      char stringpool_str127[sizeof("nosideeffect")];
      char stringpool_str128[sizeof("typename")];
      char stringpool_str130[sizeof("while")];
      char stringpool_str132[sizeof("or")];
      char stringpool_str133[sizeof("_Float32")];
      char stringpool_str136[sizeof("__null")];
      char stringpool_str142[sizeof("__launch_bounds__")];
      char stringpool_str143[sizeof("__typeof")];
      char stringpool_str144[sizeof("namespace")];
      char stringpool_str145[sizeof("__signed__")];
      char stringpool_str147[sizeof("thatispacked")];
      char stringpool_str148[sizeof("explicit")];
      char stringpool_str151[sizeof("printfesque")];
      char stringpool_str152[sizeof("dontcallback")];
      char stringpool_str153[sizeof("register")];
      char stringpool_str154[sizeof("auto")];
      char stringpool_str155[sizeof("short")];
      char stringpool_str156[sizeof("delete")];
      char stringpool_str160[sizeof("throw")];
      char stringpool_str161[sizeof("friend")];
      char stringpool_str163[sizeof("new")];
      char stringpool_str164[sizeof("nullterminated")];
      char stringpool_str165[sizeof("float")];
      char stringpool_str168[sizeof("template")];
      char stringpool_str169[sizeof("enum")];
      char stringpool_str170[sizeof("bitor")];
      char stringpool_str171[sizeof("__imag")];
      char stringpool_str172[sizeof("private")];
      char stringpool_str173[sizeof("operator")];
      char stringpool_str174[sizeof("dontthrow")];
      char stringpool_str176[sizeof("and_eq")];
      char stringpool_str178[sizeof("__host__")];
      char stringpool_str179[sizeof("__float80")];
      char stringpool_str180[sizeof("hasatleast")];
      char stringpool_str182[sizeof("nullptr")];
      char stringpool_str183[sizeof("xor")];
      char stringpool_str184[sizeof("long")];
      char stringpool_str187[sizeof("default")];
      char stringpool_str188[sizeof("and")];
      char stringpool_str191[sizeof("sizeof")];
      char stringpool_str192[sizeof("dynamic_cast")];
      char stringpool_str193[sizeof("atomic_commit")];
      char stringpool_str195[sizeof("atomic_noexcept")];
      char stringpool_str206[sizeof("switch")];
      char stringpool_str207[sizeof("do")];
      char stringpool_str208[sizeof("asm")];
      char stringpool_str209[sizeof("threadIdx")];
      char stringpool_str210[sizeof("__asm")];
      char stringpool_str211[sizeof("__bf16")];
      char stringpool_str215[sizeof("compl")];
      char stringpool_str216[sizeof("typeid")];
      char stringpool_str218[sizeof("volatile")];
      char stringpool_str220[sizeof("__volatile")];
      char stringpool_str221[sizeof("double")];
      char stringpool_str222[sizeof("typedef")];
      char stringpool_str223[sizeof("__signed")];
      char stringpool_str224[sizeof("__label__")];
      char stringpool_str230[sizeof("__device__")];
      char stringpool_str231[sizeof("signed")];
      char stringpool_str232[sizeof("wchar_t")];
      char stringpool_str237[sizeof("gridDim")];
      char stringpool_str238[sizeof("__builtin_offsetof")];
      char stringpool_str239[sizeof("goto")];
      char stringpool_str240[sizeof("or_eq")];
      char stringpool_str243[sizeof("__volatile__ ")];
      char stringpool_str244[sizeof("libcesque")];
      char stringpool_str245[sizeof("__global__")];
      char stringpool_str246[sizeof("xor_eq")];
      char stringpool_str250[sizeof("privileged")];
      char stringpool_str251[sizeof("bitand")];
      char stringpool_str253[sizeof("atomic_cancel")];
      char stringpool_str254[sizeof("void")];
      char stringpool_str262[sizeof("mutable")];
      char stringpool_str264[sizeof("relegated")];
      char stringpool_str268[sizeof("paramsnonnull")];
      char stringpool_str269[sizeof("__complex")];
      char stringpool_str270[sizeof("break")];
      char stringpool_str272[sizeof("synchronized")];
      char stringpool_str277[sizeof("virtual")];
      char stringpool_str281[sizeof("textwindows")];
      char stringpool_str288[sizeof("unsigned")];
      char stringpool_str291[sizeof("public")];
      char stringpool_str295[sizeof("catch")];
      char stringpool_str296[sizeof("mallocesque")];
      char stringpool_str301[sizeof("memcpyesque")];
      char stringpool_str311[sizeof("__builtin_va_arg")];
      char stringpool_str320[sizeof("using")];
      char stringpool_str324[sizeof("protected")];
      char stringpool_str344[sizeof("bool")];
      char stringpool_str346[sizeof("vallocesque")];
      char stringpool_str428[sizeof("blockDim")];
      char stringpool_str508[sizeof("blockIdx")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "try",
      "__FUNCTION__",
      "not",
      "const",
      "__const",
      "constinit",
      "const_cast",
      "__constant__",
      "__extension__",
      "constexpr",
      "return",
      "concept",
      "noexcept",
      "co_return",
      "extern",
      "returnstwice",
      "int",
      "char",
      "struct",
      "__attribute__",
      "case",
      "__restrict",
      "__restrict__",
      "autotype",
      "__PRETTY_FUNCTION__",
      "__attribute",
      "char8_t",
      "for",
      "else",
      "reinterpret_cast",
      "strftimeesque",
      "__forceinline__",
      "interruptfn",
      "continue",
      "this",
      "__shared__",
      "strlenesque",
      "returnspointerwithnoaliases",
      "reflexpr",
      "__alignof__",
      "if",
      "char32_t",
      "__real__ ",
      "forcealign",
      "forceinline",
      "char16_t",
      "forcealignargpointer",
      "consteval",
      "class",
      "not_eq",
      "co_await",
      "returnsnonnull",
      "static",
      "__wur",
      "static_cast",
      "__thread",
      "__imag__ ",
      "__real",
      "alignas",
      "decltype",
      "__alignof",
      "__inline__",
      "__complex__",
      "thread_local",
      "_Float16",
      "returnsaligned",
      "scanfesque",
      "alignof",
      "__inline",
      "_Float128",
      "wontreturn",
      "reallocesque",
      "_Float64",
      "pureconst",
      "inline",
      "nocallersavedregisters",
      "co_yield",
      "union",
      "export",
      "__asm__",
      "requires",
      "nosideeffect",
      "typename",
      "while",
      "or",
      "_Float32",
      "__null",
      "__launch_bounds__",
      "__typeof",
      "namespace",
      "__signed__",
      "thatispacked",
      "explicit",
      "printfesque",
      "dontcallback",
      "register",
      "auto",
      "short",
      "delete",
      "throw",
      "friend",
      "new",
      "nullterminated",
      "float",
      "template",
      "enum",
      "bitor",
      "__imag",
      "private",
      "operator",
      "dontthrow",
      "and_eq",
      "__host__",
      "__float80",
      "hasatleast",
      "nullptr",
      "xor",
      "long",
      "default",
      "and",
      "sizeof",
      "dynamic_cast",
      "atomic_commit",
      "atomic_noexcept",
      "switch",
      "do",
      "asm",
      "threadIdx",
      "__asm",
      "__bf16",
      "compl",
      "typeid",
      "volatile",
      "__volatile",
      "double",
      "typedef",
      "__signed",
      "__label__",
      "__device__",
      "signed",
      "wchar_t",
      "gridDim",
      "__builtin_offsetof",
      "goto",
      "or_eq",
      "__volatile__ ",
      "libcesque",
      "__global__",
      "xor_eq",
      "privileged",
      "bitand",
      "atomic_cancel",
      "void",
      "mutable",
      "relegated",
      "paramsnonnull",
      "__complex",
      "break",
      "synchronized",
      "virtual",
      "textwindows",
      "unsigned",
      "public",
      "catch",
      "mallocesque",
      "memcpyesque",
      "__builtin_va_arg",
      "using",
      "protected",
      "bool",
      "vallocesque",
      "blockDim",
      "blockIdx"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
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
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str182,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str184,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str187,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str192,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str195,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str207,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str208,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str209,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str210,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str211,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str215,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str220,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str221,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str222,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str223,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str224,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str230,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str231,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str232,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str237,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str238,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str239,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str240,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str243,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str244,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str245,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str246,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str250,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str251,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str253,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str254,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str262,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str264,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str268,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str269,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str270,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str272,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str277,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str281,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str288,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str291,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str295,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str296,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str301,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str311,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str320,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str324,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str344,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str346,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str428,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str508
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
