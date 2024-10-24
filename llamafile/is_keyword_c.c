/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_c.c llamafile/is_keyword_c.gperf  */
/* Computed positions: -k'1,3-5,7,9' */

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

#define TOTAL_KEYWORDS 245
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 34
#define MIN_HASH_VALUE 16
#define MAX_HASH_VALUE 857
/* maximum key range = 842, duplicates = 0 */

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
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858,   0, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858,   0,  20,
      858,  30, 858, 858,  10, 858,  10, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858,   0, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858,   0, 858,  25,  75,  65,
       75,   0, 336, 225, 110,  20, 858,   0, 170, 155,
        0,   0, 291,  15,  10,  75,   0, 135,  20,  80,
       40, 170,   0, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858, 858, 858, 858, 858,
      858, 858, 858, 858, 858, 858
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[8]];
      /*FALLTHROUGH*/
      case 8:
      case 7:
        hval += asso_values[(unsigned char)str[6]];
      /*FALLTHROUGH*/
      case 6:
      case 5:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

const char *
is_keyword_c (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str16[sizeof("extern")];
      char stringpool_str22[sizeof("if")];
      char stringpool_str23[sizeof("int")];
      char stringpool_str24[sizeof("__no_reorder__")];
      char stringpool_str26[sizeof("__no_instrument_function__")];
      char stringpool_str27[sizeof("_BitInt")];
      char stringpool_str28[sizeof("_Generic")];
      char stringpool_str29[sizeof("auto")];
      char stringpool_str30[sizeof("__return__")];
      char stringpool_str32[sizeof("__noreturn__")];
      char stringpool_str39[sizeof("__error__")];
      char stringpool_str40[sizeof("__retain__")];
      char stringpool_str41[sizeof("__real")];
      char stringpool_str43[sizeof("__real__")];
      char stringpool_str44[sizeof("__real__ ")];
      char stringpool_str45[sizeof("__target__")];
      char stringpool_str46[sizeof("__read_only")];
      char stringpool_str47[sizeof("thread_local")];
      char stringpool_str48[sizeof("__read_only__")];
      char stringpool_str50[sizeof("__target_clones")];
      char stringpool_str52[sizeof("__target_clones__")];
      char stringpool_str54[sizeof("__noipa__")];
      char stringpool_str57[sizeof("__read_write")];
      char stringpool_str58[sizeof("__zero_call_used_regs__")];
      char stringpool_str59[sizeof("__read_write__")];
      char stringpool_str60[sizeof("__no_sanitize__")];
      char stringpool_str65[sizeof("__vex")];
      char stringpool_str67[sizeof("__no_sanitize_thread__")];
      char stringpool_str68[sizeof("__no_sanitize_address__")];
      char stringpool_str70[sizeof("__no_sanitize_undefined__")];
      char stringpool_str72[sizeof("__const")];
      char stringpool_str73[sizeof("__extension__")];
      char stringpool_str74[sizeof("__const__")];
      char stringpool_str75[sizeof("__no_icf__")];
      char stringpool_str76[sizeof("__noclone__")];
      char stringpool_str77[sizeof("do")];
      char stringpool_str79[sizeof("else")];
      char stringpool_str81[sizeof("sizeof")];
      char stringpool_str83[sizeof("__no_stack_limit__")];
      char stringpool_str87[sizeof("__no_stack_protector__")];
      char stringpool_str90[sizeof("short")];
      char stringpool_str91[sizeof("__avx2")];
      char stringpool_str92[sizeof("__vax__")];
      char stringpool_str95[sizeof("__strong__")];
      char stringpool_str97[sizeof("__externally_visible__")];
      char stringpool_str98[sizeof("__word__")];
      char stringpool_str100[sizeof("__vector_size__")];
      char stringpool_str101[sizeof("__nothrow__")];
      char stringpool_str104[sizeof("char")];
      char stringpool_str105[sizeof("break")];
      char stringpool_str107[sizeof("__sentinel__")];
      char stringpool_str110[sizeof("wontreturn")];
      char stringpool_str112[sizeof("__returns_twice__")];
      char stringpool_str113[sizeof("__weak__")];
      char stringpool_str114[sizeof("__returns_nonnull__")];
      char stringpool_str115[sizeof("__no_address_safety_analysis__")];
      char stringpool_str117[sizeof("__hot__")];
      char stringpool_str119[sizeof("void")];
      char stringpool_str122[sizeof("__write_only")];
      char stringpool_str124[sizeof("__write_only__")];
      char stringpool_str126[sizeof("static")];
      char stringpool_str129[sizeof("_Static_assert")];
      char stringpool_str130[sizeof("_Decimal64")];
      char stringpool_str131[sizeof("textwindows")];
      char stringpool_str136[sizeof("__tainted_args__")];
      char stringpool_str137[sizeof("__warn_unused_result__")];
      char stringpool_str141[sizeof("_Decimal128")];
      char stringpool_str144[sizeof("case")];
      char stringpool_str145[sizeof("const")];
      char stringpool_str150[sizeof("_Decimal32")];
      char stringpool_str153[sizeof("__thread")];
      char stringpool_str154[sizeof("_Noreturn")];
      char stringpool_str156[sizeof("__transparent_union__")];
      char stringpool_str160[sizeof("union")];
      char stringpool_str161[sizeof("return")];
      char stringpool_str163[sizeof("__unix__")];
      char stringpool_str166[sizeof("switch")];
      char stringpool_str168[sizeof("restrict")];
      char stringpool_str170[sizeof("__restrict")];
      char stringpool_str171[sizeof("__section__")];
      char stringpool_str172[sizeof("__restrict__")];
      char stringpool_str174[sizeof("dontthrow")];
      char stringpool_str175[sizeof("_Bool")];
      char stringpool_str176[sizeof("interruptfn")];
      char stringpool_str178[sizeof("__interrupt__")];
      char stringpool_str183[sizeof("asm")];
      char stringpool_str186[sizeof("__interrupt_handler__")];
      char stringpool_str191[sizeof("__attribute")];
      char stringpool_str193[sizeof("__attribute__")];
      char stringpool_str198[sizeof("__inline")];
      char stringpool_str199[sizeof("constexpr")];
      char stringpool_str200[sizeof("__inline__")];
      char stringpool_str202[sizeof("__noinline__")];
      char stringpool_str203[sizeof("__leaf__")];
      char stringpool_str206[sizeof("__imag")];
      char stringpool_str208[sizeof("__imag__")];
      char stringpool_str209[sizeof("__imag__ ")];
      char stringpool_str213[sizeof("_Float64")];
      char stringpool_str215[sizeof("__constructor__")];
      char stringpool_str216[sizeof("inline")];
      char stringpool_str218[sizeof("__used__")];
      char stringpool_str223[sizeof("_Float16")];
      char stringpool_str224[sizeof("__no_caller_saved_registers__")];
      char stringpool_str227[sizeof("_Atomic")];
      char stringpool_str228[sizeof("continue")];
      char stringpool_str229[sizeof("goto")];
      char stringpool_str230[sizeof("__wur")];
      char stringpool_str232[sizeof("__hardbool__")];
      char stringpool_str233[sizeof("_Float32")];
      char stringpool_str234[sizeof("_Float128")];
      char stringpool_str238[sizeof("__mode__")];
      char stringpool_str239[sizeof("__destructor__")];
      char stringpool_str240[sizeof("__access__")];
      char stringpool_str241[sizeof("__cmn_err__")];
      char stringpool_str242[sizeof("__builtin___")];
      char stringpool_str243[sizeof("__cold__")];
      char stringpool_str244[sizeof("returnsnonnull")];
      char stringpool_str246[sizeof("__builtin_va_arg")];
      char stringpool_str248[sizeof("__builtin_offsetof")];
      char stringpool_str249[sizeof("bool")];
      char stringpool_str251[sizeof("__strfmon__")];
      char stringpool_str252[sizeof("__strftime__")];
      char stringpool_str253[sizeof("__byte__")];
      char stringpool_str257[sizeof("returnspointerwithnoaliases")];
      char stringpool_str258[sizeof("__simd__")];
      char stringpool_str260[sizeof("__asm")];
      char stringpool_str262[sizeof("__asm__")];
      char stringpool_str275[sizeof("while")];
      char stringpool_str280[sizeof("__unused__")];
      char stringpool_str281[sizeof("strlenesque")];
      char stringpool_str282[sizeof("__msabi")];
      char stringpool_str291[sizeof("struct")];
      char stringpool_str294[sizeof("enum")];
      char stringpool_str295[sizeof("hasatleast")];
      char stringpool_str297[sizeof("typeof")];
      char stringpool_str299[sizeof("__alias__")];
      char stringpool_str300[sizeof("__noplt__")];
      char stringpool_str301[sizeof("__aligned__")];
      char stringpool_str304[sizeof("typeof_unqual")];
      char stringpool_str306[sizeof("signed")];
      char stringpool_str311[sizeof("__null")];
      char stringpool_str315[sizeof("__ms_abi__")];
      char stringpool_str316[sizeof("__nonnull__")];
      char stringpool_str318[sizeof("__pie__")];
      char stringpool_str320[sizeof("__mcarch__")];
      char stringpool_str322[sizeof("returnstwice")];
      char stringpool_str324[sizeof("autotype")];
      char stringpool_str328[sizeof("__signed")];
      char stringpool_str329[sizeof("__no_split_stack__")];
      char stringpool_str330[sizeof("__signed__")];
      char stringpool_str331[sizeof("__printf__")];
      char stringpool_str332[sizeof("__pointer__")];
      char stringpool_str334[sizeof("libcesque")];
      char stringpool_str337[sizeof("printfesque")];
      char stringpool_str338[sizeof("register")];
      char stringpool_str341[sizeof("__auto_type")];
      char stringpool_str348[sizeof("__assume_aligned__")];
      char stringpool_str349[sizeof("for")];
      char stringpool_str352[sizeof("alignas")];
      char stringpool_str356[sizeof("__fentry__")];
      char stringpool_str357[sizeof("nocallersavedregisters")];
      char stringpool_str360[sizeof("__malloc__")];
      char stringpool_str361[sizeof("privileged")];
      char stringpool_str364[sizeof("__copy__")];
      char stringpool_str366[sizeof("float")];
      char stringpool_str370[sizeof("__volatile")];
      char stringpool_str371[sizeof("__warning__")];
      char stringpool_str372[sizeof("__volatile__")];
      char stringpool_str373[sizeof("__volatile__ ")];
      char stringpool_str374[sizeof("__visibility__")];
      char stringpool_str375[sizeof("pureconst")];
      char stringpool_str378[sizeof("_Thread_local")];
      char stringpool_str380[sizeof("__no_profile_instrument_function__")];
      char stringpool_str381[sizeof("__format__")];
      char stringpool_str383[sizeof("__pic__")];
      char stringpool_str384[sizeof("nullterminated")];
      char stringpool_str385[sizeof("__format_arg__")];
      char stringpool_str386[sizeof("vallocesque")];
      char stringpool_str390[sizeof("__muarch__")];
      char stringpool_str391[sizeof("__packed__")];
      char stringpool_str393[sizeof("volatile")];
      char stringpool_str397[sizeof("dontcallback")];
      char stringpool_str398[sizeof("__force_align_arg_pointer__")];
      char stringpool_str399[sizeof("long")];
      char stringpool_str405[sizeof("__deprecated__")];
      char stringpool_str407[sizeof("__sysv_abi__")];
      char stringpool_str408[sizeof("__may_alias__")];
      char stringpool_str410[sizeof("__symver__")];
      char stringpool_str413[sizeof("thatispacked")];
      char stringpool_str414[sizeof("returnsaligned")];
      char stringpool_str423[sizeof("_Alignof")];
      char stringpool_str425[sizeof("_Imaginary")];
      char stringpool_str429[sizeof("__fd_arg")];
      char stringpool_str431[sizeof("__fd_arg__")];
      char stringpool_str437[sizeof("__bf16")];
      char stringpool_str439[sizeof("__complex")];
      char stringpool_str441[sizeof("__complex__")];
      char stringpool_str442[sizeof("forceinline")];
      char stringpool_str444[sizeof("__pure__")];
      char stringpool_str448[sizeof("_Alignas")];
      char stringpool_str449[sizeof("__label__")];
      char stringpool_str450[sizeof("__artificial__")];
      char stringpool_str454[sizeof("_Complex")];
      char stringpool_str458[sizeof("__optimize__")];
      char stringpool_str461[sizeof("double")];
      char stringpool_str462[sizeof("__always_inline__")];
      char stringpool_str463[sizeof("unsigned")];
      char stringpool_str469[sizeof("__typeof")];
      char stringpool_str470[sizeof("__alloc_align__")];
      char stringpool_str471[sizeof("__typeof__")];
      char stringpool_str473[sizeof("__gnu_scanf__")];
      char stringpool_str474[sizeof("__warn_if_not_aligned__")];
      char stringpool_str478[sizeof("__typeof_unqual__")];
      char stringpool_str489[sizeof("relegated")];
      char stringpool_str494[sizeof("paramsnonnull")];
      char stringpool_str499[sizeof("__params_nonnull__")];
      char stringpool_str500[sizeof("__funline")];
      char stringpool_str510[sizeof("__scanf__")];
      char stringpool_str515[sizeof("__float80")];
      char stringpool_str518[sizeof("nosideeffect")];
      char stringpool_str519[sizeof("__alloc_size__")];
      char stringpool_str521[sizeof("mallocesque")];
      char stringpool_str527[sizeof("reallocesque")];
      char stringpool_str529[sizeof("__patchable_function_entry__")];
      char stringpool_str533[sizeof("__seg_gs")];
      char stringpool_str542[sizeof("__flatten__")];
      char stringpool_str560[sizeof("__alignof")];
      char stringpool_str562[sizeof("__alignof__")];
      char stringpool_str564[sizeof("__gnu_inline__")];
      char stringpool_str565[sizeof("__ifunc__")];
      char stringpool_str578[sizeof("default")];
      char stringpool_str589[sizeof("strftimeesque")];
      char stringpool_str613[sizeof("alignof")];
      char stringpool_str644[sizeof("__seg_fs")];
      char stringpool_str648[sizeof("nullptr")];
      char stringpool_str656[sizeof("scanfesque")];
      char stringpool_str668[sizeof("__mcfhwdiv__")];
      char stringpool_str685[sizeof("__gnu_printf__")];
      char stringpool_str687[sizeof("__mcfarch__")];
      char stringpool_str692[sizeof("memcpyesque")];
      char stringpool_str709[sizeof("typedef")];
      char stringpool_str720[sizeof("__gnu_format__")];
      char stringpool_str816[sizeof("forcealign")];
      char stringpool_str826[sizeof("forcealignargpointer")];
      char stringpool_str857[sizeof("__mcffpu__")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "extern",
      "if",
      "int",
      "__no_reorder__",
      "__no_instrument_function__",
      "_BitInt",
      "_Generic",
      "auto",
      "__return__",
      "__noreturn__",
      "__error__",
      "__retain__",
      "__real",
      "__real__",
      "__real__ ",
      "__target__",
      "__read_only",
      "thread_local",
      "__read_only__",
      "__target_clones",
      "__target_clones__",
      "__noipa__",
      "__read_write",
      "__zero_call_used_regs__",
      "__read_write__",
      "__no_sanitize__",
      "__vex",
      "__no_sanitize_thread__",
      "__no_sanitize_address__",
      "__no_sanitize_undefined__",
      "__const",
      "__extension__",
      "__const__",
      "__no_icf__",
      "__noclone__",
      "do",
      "else",
      "sizeof",
      "__no_stack_limit__",
      "__no_stack_protector__",
      "short",
      "__avx2",
      "__vax__",
      "__strong__",
      "__externally_visible__",
      "__word__",
      "__vector_size__",
      "__nothrow__",
      "char",
      "break",
      "__sentinel__",
      "wontreturn",
      "__returns_twice__",
      "__weak__",
      "__returns_nonnull__",
      "__no_address_safety_analysis__",
      "__hot__",
      "void",
      "__write_only",
      "__write_only__",
      "static",
      "_Static_assert",
      "_Decimal64",
      "textwindows",
      "__tainted_args__",
      "__warn_unused_result__",
      "_Decimal128",
      "case",
      "const",
      "_Decimal32",
      "__thread",
      "_Noreturn",
      "__transparent_union__",
      "union",
      "return",
      "__unix__",
      "switch",
      "restrict",
      "__restrict",
      "__section__",
      "__restrict__",
      "dontthrow",
      "_Bool",
      "interruptfn",
      "__interrupt__",
      "asm",
      "__interrupt_handler__",
      "__attribute",
      "__attribute__",
      "__inline",
      "constexpr",
      "__inline__",
      "__noinline__",
      "__leaf__",
      "__imag",
      "__imag__",
      "__imag__ ",
      "_Float64",
      "__constructor__",
      "inline",
      "__used__",
      "_Float16",
      "__no_caller_saved_registers__",
      "_Atomic",
      "continue",
      "goto",
      "__wur",
      "__hardbool__",
      "_Float32",
      "_Float128",
      "__mode__",
      "__destructor__",
      "__access__",
      "__cmn_err__",
      "__builtin___",
      "__cold__",
      "returnsnonnull",
      "__builtin_va_arg",
      "__builtin_offsetof",
      "bool",
      "__strfmon__",
      "__strftime__",
      "__byte__",
      "returnspointerwithnoaliases",
      "__simd__",
      "__asm",
      "__asm__",
      "while",
      "__unused__",
      "strlenesque",
      "__msabi",
      "struct",
      "enum",
      "hasatleast",
      "typeof",
      "__alias__",
      "__noplt__",
      "__aligned__",
      "typeof_unqual",
      "signed",
      "__null",
      "__ms_abi__",
      "__nonnull__",
      "__pie__",
      "__mcarch__",
      "returnstwice",
      "autotype",
      "__signed",
      "__no_split_stack__",
      "__signed__",
      "__printf__",
      "__pointer__",
      "libcesque",
      "printfesque",
      "register",
      "__auto_type",
      "__assume_aligned__",
      "for",
      "alignas",
      "__fentry__",
      "nocallersavedregisters",
      "__malloc__",
      "privileged",
      "__copy__",
      "float",
      "__volatile",
      "__warning__",
      "__volatile__",
      "__volatile__ ",
      "__visibility__",
      "pureconst",
      "_Thread_local",
      "__no_profile_instrument_function__",
      "__format__",
      "__pic__",
      "nullterminated",
      "__format_arg__",
      "vallocesque",
      "__muarch__",
      "__packed__",
      "volatile",
      "dontcallback",
      "__force_align_arg_pointer__",
      "long",
      "__deprecated__",
      "__sysv_abi__",
      "__may_alias__",
      "__symver__",
      "thatispacked",
      "returnsaligned",
      "_Alignof",
      "_Imaginary",
      "__fd_arg",
      "__fd_arg__",
      "__bf16",
      "__complex",
      "__complex__",
      "forceinline",
      "__pure__",
      "_Alignas",
      "__label__",
      "__artificial__",
      "_Complex",
      "__optimize__",
      "double",
      "__always_inline__",
      "unsigned",
      "__typeof",
      "__alloc_align__",
      "__typeof__",
      "__gnu_scanf__",
      "__warn_if_not_aligned__",
      "__typeof_unqual__",
      "relegated",
      "paramsnonnull",
      "__params_nonnull__",
      "__funline",
      "__scanf__",
      "__float80",
      "nosideeffect",
      "__alloc_size__",
      "mallocesque",
      "reallocesque",
      "__patchable_function_entry__",
      "__seg_gs",
      "__flatten__",
      "__alignof",
      "__alignof__",
      "__gnu_inline__",
      "__ifunc__",
      "default",
      "strftimeesque",
      "alignof",
      "__seg_fs",
      "nullptr",
      "scanfesque",
      "__mcfhwdiv__",
      "__gnu_printf__",
      "__mcfarch__",
      "memcpyesque",
      "typedef",
      "__gnu_format__",
      "forcealign",
      "forcealignargpointer",
      "__mcffpu__"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      -1, -1, -1, -1, -1,
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
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str141,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str175,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str198,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str199,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str200,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str202,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str203,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str208,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str209,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str213,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str215,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str223,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str224,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str227,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str228,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str230,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str232,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str233,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str234,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str238,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str239,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str240,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str241,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str242,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str243,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str244,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str246,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str248,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str249,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str251,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str252,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str253,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str257,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str258,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str260,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str262,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str275,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str280,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str281,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str282,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str291,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str294,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str295,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str297,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str299,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str300,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str301,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str304,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str306,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str311,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str315,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str316,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str318,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str320,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str322,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str324,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str328,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str329,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str330,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str331,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str332,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str334,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str337,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str338,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str341,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str348,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str349,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str352,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str356,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str357,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str360,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str361,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str364,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str366,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str370,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str371,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str372,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str373,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str374,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str375,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str378,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str380,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str381,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str383,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str384,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str385,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str386,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str390,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str391,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str393,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str397,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str398,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str399,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str405,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str407,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str408,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str410,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str413,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str414,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str423,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str425,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str429,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str431,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str437,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str439,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str441,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str442,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str444,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str448,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str449,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str450,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str454,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str458,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str461,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str462,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str463,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str469,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str470,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str471,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str473,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str474,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str478,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str489,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str494,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str499,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str500,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str510,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str515,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str518,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str519,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str521,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str527,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str529,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str533,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str542,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str560,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str562,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str564,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str565,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str578,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str589,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str613,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str644,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str648,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str656,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str668,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str685,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str687,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str692,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str709,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str720,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str816,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str826,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str857
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
