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

#define TOTAL_KEYWORDS 172
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 27
#define MIN_HASH_VALUE 8
#define MAX_HASH_VALUE 449
/* maximum key range = 442, duplicates = 0 */

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
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 150, 450, 450,  85, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450,   0,  30,
       70,  25,  45, 450,   5, 450,   5, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450,   0,
        0, 450, 450, 450, 450, 450, 450, 450,   0, 450,
        0, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450,  55, 450,   5, 140,   5,
      135,   0,  15, 100,  40,  15, 450,   5,  70,  30,
        5, 135, 170,  15,  10,  35,   5, 105,  10,  50,
      200,   0,   5, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
      450, 450, 450, 450, 450, 450
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
      char stringpool_str8[sizeof("try")];
      char stringpool_str18[sizeof("not")];
      char stringpool_str22[sizeof("concept")];
      char stringpool_str23[sizeof("noexcept")];
      char stringpool_str24[sizeof("char")];
      char stringpool_str25[sizeof("const")];
      char stringpool_str26[sizeof("extern")];
      char stringpool_str27[sizeof("char8_t")];
      char stringpool_str28[sizeof("int")];
      char stringpool_str29[sizeof("constinit")];
      char stringpool_str30[sizeof("const_cast")];
      char stringpool_str31[sizeof("not_eq")];
      char stringpool_str32[sizeof("if")];
      char stringpool_str33[sizeof("continue")];
      char stringpool_str34[sizeof("constexpr")];
      char stringpool_str36[sizeof("return")];
      char stringpool_str38[sizeof("for")];
      char stringpool_str39[sizeof("else")];
      char stringpool_str40[sizeof("forcealign")];
      char stringpool_str43[sizeof("reflexpr")];
      char stringpool_str44[sizeof("case")];
      char stringpool_str46[sizeof("interruptfn")];
      char stringpool_str47[sizeof("alignof")];
      char stringpool_str48[sizeof("char32_t")];
      char stringpool_str51[sizeof("reinterpret_cast")];
      char stringpool_str53[sizeof("char16_t")];
      char stringpool_str55[sizeof("forcealignargpointer")];
      char stringpool_str56[sizeof("strlenesque")];
      char stringpool_str59[sizeof("this")];
      char stringpool_str61[sizeof("struct")];
      char stringpool_str63[sizeof("strftimeesque")];
      char stringpool_str65[sizeof("scanfesque")];
      char stringpool_str66[sizeof("static")];
      char stringpool_str67[sizeof("alignas")];
      char stringpool_str68[sizeof("asm")];
      char stringpool_str70[sizeof("while")];
      char stringpool_str71[sizeof("static_cast")];
      char stringpool_str73[sizeof("static_assert")];
      char stringpool_str74[sizeof("co_return")];
      char stringpool_str76[sizeof("__attribute")];
      char stringpool_str77[sizeof("__const")];
      char stringpool_str79[sizeof("namespace")];
      char stringpool_str80[sizeof("wontreturn")];
      char stringpool_str83[sizeof("requires")];
      char stringpool_str85[sizeof("class")];
      char stringpool_str87[sizeof("returnspointerwithnoaliases")];
      char stringpool_str90[sizeof("false")];
      char stringpool_str93[sizeof("volatile")];
      char stringpool_str94[sizeof("consteval")];
      char stringpool_str95[sizeof("catch")];
      char stringpool_str96[sizeof("inline")];
      char stringpool_str97[sizeof("reallocesque")];
      char stringpool_str99[sizeof("__alignof")];
      char stringpool_str101[sizeof("switch")];
      char stringpool_str102[sizeof("thread_local")];
      char stringpool_str105[sizeof("#line")];
      char stringpool_str107[sizeof("#define")];
      char stringpool_str108[sizeof("new")];
      char stringpool_str111[sizeof("#undef")];
      char stringpool_str112[sizeof("wchar_t")];
      char stringpool_str113[sizeof("template")];
      char stringpool_str114[sizeof("true")];
      char stringpool_str115[sizeof("__restrict")];
      char stringpool_str118[sizeof("#if")];
      char stringpool_str120[sizeof("throw")];
      char stringpool_str121[sizeof("#ifdef")];
      char stringpool_str122[sizeof("__FUNCTION__")];
      char stringpool_str123[sizeof("co_await")];
      char stringpool_str125[sizeof("__asm")];
      char stringpool_str126[sizeof("#endif")];
      char stringpool_str128[sizeof("__extension__")];
      char stringpool_str129[sizeof("__PRETTY_FUNCTION__")];
      char stringpool_str133[sizeof("__attribute__")];
      char stringpool_str135[sizeof("union")];
      char stringpool_str137[sizeof("nocallersavedregisters")];
      char stringpool_str138[sizeof("__func__")];
      char stringpool_str139[sizeof("enum")];
      char stringpool_str141[sizeof("__alignof__")];
      char stringpool_str143[sizeof("_Float16")];
      char stringpool_str144[sizeof("_Float128")];
      char stringpool_str145[sizeof("__volatile")];
      char stringpool_str146[sizeof("__real")];
      char stringpool_str147[sizeof("or")];
      char stringpool_str148[sizeof("__inline")];
      char stringpool_str149[sizeof("auto")];
      char stringpool_str152[sizeof("__asm__")];
      char stringpool_str153[sizeof("decltype")];
      char stringpool_str155[sizeof("break")];
      char stringpool_str156[sizeof("__complex__")];
      char stringpool_str158[sizeof("operator")];
      char stringpool_str160[sizeof("#else")];
      char stringpool_str161[sizeof("and_eq")];
      char stringpool_str162[sizeof("dontcallback")];
      char stringpool_str163[sizeof("register")];
      char stringpool_str164[sizeof("void")];
      char stringpool_str165[sizeof("float")];
      char stringpool_str167[sizeof("__restrict__")];
      char stringpool_str168[sizeof("#include")];
      char stringpool_str170[sizeof("bitor")];
      char stringpool_str172[sizeof("thatispacked")];
      char stringpool_str173[sizeof("atomic_commit")];
      char stringpool_str174[sizeof("returnsaligned")];
      char stringpool_str175[sizeof("atomic_noexcept")];
      char stringpool_str176[sizeof("friend")];
      char stringpool_str178[sizeof("#include_next")];
      char stringpool_str179[sizeof("long")];
      char stringpool_str180[sizeof("compl")];
      char stringpool_str181[sizeof("__imag")];
      char stringpool_str182[sizeof("mutable")];
      char stringpool_str183[sizeof("_Float64")];
      char stringpool_str185[sizeof("short")];
      char stringpool_str187[sizeof("dynamic_cast")];
      char stringpool_str188[sizeof("typename")];
      char stringpool_str190[sizeof("#elif")];
      char stringpool_str191[sizeof("export")];
      char stringpool_str192[sizeof("nosideeffect")];
      char stringpool_str193[sizeof("#elifdef")];
      char stringpool_str194[sizeof("#elifndef")];
      char stringpool_str196[sizeof("sizeof")];
      char stringpool_str197[sizeof("private")];
      char stringpool_str198[sizeof("explicit")];
      char stringpool_str199[sizeof("pureconst")];
      char stringpool_str201[sizeof("printfesque")];
      char stringpool_str202[sizeof("virtual")];
      char stringpool_str203[sizeof("#warning")];
      char stringpool_str204[sizeof("dontthrow")];
      char stringpool_str205[sizeof("__inline__")];
      char stringpool_str206[sizeof("__null")];
      char stringpool_str208[sizeof("_Float32")];
      char stringpool_str213[sizeof("__thread")];
      char stringpool_str214[sizeof("__float80")];
      char stringpool_str216[sizeof("delete")];
      char stringpool_str218[sizeof("co_yield")];
      char stringpool_str219[sizeof("libcesque")];
      char stringpool_str223[sizeof("xor")];
      char stringpool_str225[sizeof("or_eq")];
      char stringpool_str226[sizeof("vallocesque")];
      char stringpool_str227[sizeof("synchronized")];
      char stringpool_str229[sizeof("__real__ ")];
      char stringpool_str231[sizeof("xor_eq")];
      char stringpool_str234[sizeof("__imag__ ")];
      char stringpool_str236[sizeof("__bf16")];
      char stringpool_str238[sizeof("atomic_cancel")];
      char stringpool_str241[sizeof("memcpyesque")];
      char stringpool_str243[sizeof("__builtin_offsetof")];
      char stringpool_str244[sizeof("goto")];
      char stringpool_str246[sizeof("mallocesque")];
      char stringpool_str253[sizeof("__typeof")];
      char stringpool_str255[sizeof("__signed__")];
      char stringpool_str256[sizeof("#embed")];
      char stringpool_str262[sizeof("nullptr")];
      char stringpool_str267[sizeof("default")];
      char stringpool_str272[sizeof("do")];
      char stringpool_str276[sizeof("signed")];
      char stringpool_str278[sizeof("and")];
      char stringpool_str291[sizeof("bitand")];
      char stringpool_str293[sizeof("paramsnonnull")];
      char stringpool_str298[sizeof("__volatile__ ")];
      char stringpool_str299[sizeof("__complex")];
      char stringpool_str301[sizeof("textwindows")];
      char stringpool_str316[sizeof("double")];
      char stringpool_str325[sizeof("using")];
      char stringpool_str326[sizeof("__builtin_va_arg")];
      char stringpool_str329[sizeof("__label__")];
      char stringpool_str331[sizeof("typeid")];
      char stringpool_str332[sizeof("typedef")];
      char stringpool_str333[sizeof("__signed")];
      char stringpool_str336[sizeof("public")];
      char stringpool_str345[sizeof("privileged")];
      char stringpool_str349[sizeof("bool")];
      char stringpool_str383[sizeof("unsigned")];
      char stringpool_str449[sizeof("protected")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "try",
      "not",
      "concept",
      "noexcept",
      "char",
      "const",
      "extern",
      "char8_t",
      "int",
      "constinit",
      "const_cast",
      "not_eq",
      "if",
      "continue",
      "constexpr",
      "return",
      "for",
      "else",
      "forcealign",
      "reflexpr",
      "case",
      "interruptfn",
      "alignof",
      "char32_t",
      "reinterpret_cast",
      "char16_t",
      "forcealignargpointer",
      "strlenesque",
      "this",
      "struct",
      "strftimeesque",
      "scanfesque",
      "static",
      "alignas",
      "asm",
      "while",
      "static_cast",
      "static_assert",
      "co_return",
      "__attribute",
      "__const",
      "namespace",
      "wontreturn",
      "requires",
      "class",
      "returnspointerwithnoaliases",
      "false",
      "volatile",
      "consteval",
      "catch",
      "inline",
      "reallocesque",
      "__alignof",
      "switch",
      "thread_local",
      "#line",
      "#define",
      "new",
      "#undef",
      "wchar_t",
      "template",
      "true",
      "__restrict",
      "#if",
      "throw",
      "#ifdef",
      "__FUNCTION__",
      "co_await",
      "__asm",
      "#endif",
      "__extension__",
      "__PRETTY_FUNCTION__",
      "__attribute__",
      "union",
      "nocallersavedregisters",
      "__func__",
      "enum",
      "__alignof__",
      "_Float16",
      "_Float128",
      "__volatile",
      "__real",
      "or",
      "__inline",
      "auto",
      "__asm__",
      "decltype",
      "break",
      "__complex__",
      "operator",
      "#else",
      "and_eq",
      "dontcallback",
      "register",
      "void",
      "float",
      "__restrict__",
      "#include",
      "bitor",
      "thatispacked",
      "atomic_commit",
      "returnsaligned",
      "atomic_noexcept",
      "friend",
      "#include_next",
      "long",
      "compl",
      "__imag",
      "mutable",
      "_Float64",
      "short",
      "dynamic_cast",
      "typename",
      "#elif",
      "export",
      "nosideeffect",
      "#elifdef",
      "#elifndef",
      "sizeof",
      "private",
      "explicit",
      "pureconst",
      "printfesque",
      "virtual",
      "#warning",
      "dontthrow",
      "__inline__",
      "__null",
      "_Float32",
      "__thread",
      "__float80",
      "delete",
      "co_yield",
      "libcesque",
      "xor",
      "or_eq",
      "vallocesque",
      "synchronized",
      "__real__ ",
      "xor_eq",
      "__imag__ ",
      "__bf16",
      "atomic_cancel",
      "memcpyesque",
      "__builtin_offsetof",
      "goto",
      "mallocesque",
      "__typeof",
      "__signed__",
      "#embed",
      "nullptr",
      "default",
      "do",
      "signed",
      "and",
      "bitand",
      "paramsnonnull",
      "__volatile__ ",
      "__complex",
      "textwindows",
      "double",
      "using",
      "__builtin_va_arg",
      "__label__",
      "typeid",
      "typedef",
      "__signed",
      "public",
      "privileged",
      "bool",
      "unsigned",
      "protected"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
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
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str141,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str175,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str181,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str182,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str185,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str187,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str192,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str194,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str196,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str197,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str198,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str199,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str201,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str202,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str203,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str204,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str205,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str208,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str213,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str223,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str225,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str226,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str227,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str231,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str234,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str236,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str238,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str241,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str243,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str244,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str246,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str253,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str255,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str256,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str262,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str267,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str272,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str276,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str278,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str291,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str293,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str298,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str299,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str301,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str316,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str325,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str326,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str329,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str331,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str332,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str333,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str336,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str345,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str349,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str383,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str449
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
