/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_c.gperf  */
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

#define TOTAL_KEYWORDS 237
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 34
#define MIN_HASH_VALUE 16
#define MAX_HASH_VALUE 632
/* maximum key range = 617, duplicates = 0 */

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
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633,   0, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633,   0,  20,
      633,  90, 633, 633,  45, 633,   5, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633,   0, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633,   0, 633,  25, 195,  55,
       15,   0, 125, 185,  65,  80, 633,   5, 160, 158,
        0,   0, 219,   0,  10,  65,   0, 115,  45, 145,
      248, 235,   0, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633, 633, 633, 633, 633,
      633, 633, 633, 633, 633, 633
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
      char stringpool_str17[sizeof("do")];
      char stringpool_str24[sizeof("__no_reorder__")];
      char stringpool_str26[sizeof("__no_instrument_function__")];
      char stringpool_str29[sizeof("auto")];
      char stringpool_str30[sizeof("__return__")];
      char stringpool_str32[sizeof("__noreturn__")];
      char stringpool_str39[sizeof("__error__")];
      char stringpool_str41[sizeof("__real")];
      char stringpool_str43[sizeof("__real__")];
      char stringpool_str44[sizeof("__real__ ")];
      char stringpool_str45[sizeof("__target__")];
      char stringpool_str46[sizeof("__read_only")];
      char stringpool_str47[sizeof("thread_local")];
      char stringpool_str48[sizeof("__read_only__")];
      char stringpool_str50[sizeof("__target_clones")];
      char stringpool_str52[sizeof("__target_clones__")];
      char stringpool_str55[sizeof("__no_address_safety_analysis__")];
      char stringpool_str57[sizeof("__read_write")];
      char stringpool_str58[sizeof("__zero_call_used_regs__")];
      char stringpool_str59[sizeof("__read_write__")];
      char stringpool_str62[sizeof("__const")];
      char stringpool_str64[sizeof("__const__")];
      char stringpool_str65[sizeof("__no_icf__")];
      char stringpool_str66[sizeof("__noclone__")];
      char stringpool_str69[sizeof("else")];
      char stringpool_str71[sizeof("sizeof")];
      char stringpool_str72[sizeof("__hot__")];
      char stringpool_str73[sizeof("__no_stack_limit__")];
      char stringpool_str77[sizeof("__no_stack_protector__")];
      char stringpool_str80[sizeof("short")];
      char stringpool_str82[sizeof("if")];
      char stringpool_str83[sizeof("int")];
      char stringpool_str85[sizeof("__strong__")];
      char stringpool_str87[sizeof("_BitInt")];
      char stringpool_str88[sizeof("_Generic")];
      char stringpool_str94[sizeof("char")];
      char stringpool_str100[sizeof("__retain__")];
      char stringpool_str102[sizeof("__returns_twice__")];
      char stringpool_str104[sizeof("__returns_nonnull__")];
      char stringpool_str108[sizeof("__thread")];
      char stringpool_str114[sizeof("__noipa__")];
      char stringpool_str115[sizeof("__vector_size__")];
      char stringpool_str119[sizeof("_Static_assert")];
      char stringpool_str120[sizeof("__no_sanitize__")];
      char stringpool_str124[sizeof("case")];
      char stringpool_str125[sizeof("const")];
      char stringpool_str127[sizeof("__no_sanitize_thread__")];
      char stringpool_str128[sizeof("__no_sanitize_address__")];
      char stringpool_str130[sizeof("__no_sanitize_undefined__")];
      char stringpool_str134[sizeof("_Noreturn")];
      char stringpool_str136[sizeof("__tainted_args__")];
      char stringpool_str138[sizeof("for")];
      char stringpool_str141[sizeof("return")];
      char stringpool_str144[sizeof("void")];
      char stringpool_str145[sizeof("__fentry__")];
      char stringpool_str146[sizeof("__transparent_union__")];
      char stringpool_str148[sizeof("restrict")];
      char stringpool_str150[sizeof("__restrict")];
      char stringpool_str152[sizeof("__restrict__")];
      char stringpool_str155[sizeof("float")];
      char stringpool_str157[sizeof("__sentinel__")];
      char stringpool_str158[sizeof("__fd_arg")];
      char stringpool_str159[sizeof("__destructor__")];
      char stringpool_str160[sizeof("__fd_arg__")];
      char stringpool_str163[sizeof("__word__")];
      char stringpool_str165[sizeof("_Bool")];
      char stringpool_str166[sizeof("__nothrow__")];
      char stringpool_str169[sizeof("__no_profile_instrument_function__")];
      char stringpool_str170[sizeof("__format__")];
      char stringpool_str174[sizeof("__format_arg__")];
      char stringpool_str175[sizeof("wontreturn")];
      char stringpool_str176[sizeof("static")];
      char stringpool_str178[sizeof("__weak__")];
      char stringpool_str179[sizeof("dontthrow")];
      char stringpool_str181[sizeof("__mode__")];
      char stringpool_str185[sizeof("__constructor__")];
      char stringpool_str186[sizeof("asm")];
      char stringpool_str187[sizeof("__force_align_arg_pointer__")];
      char stringpool_str188[sizeof("__used__")];
      char stringpool_str189[sizeof("goto")];
      char stringpool_str191[sizeof("_Decimal128")];
      char stringpool_str193[sizeof("__leaf__")];
      char stringpool_str200[sizeof("union")];
      char stringpool_str202[sizeof("__warn_unused_result__")];
      char stringpool_str203[sizeof("__unix__")];
      char stringpool_str206[sizeof("switch")];
      char stringpool_str210[sizeof("__access__")];
      char stringpool_str211[sizeof("__section__")];
      char stringpool_str213[sizeof("_Float16")];
      char stringpool_str214[sizeof("__no_caller_saved_registers__")];
      char stringpool_str215[sizeof("_Decimal64")];
      char stringpool_str216[sizeof("interruptfn")];
      char stringpool_str218[sizeof("__interrupt__")];
      char stringpool_str219[sizeof("_Float128")];
      char stringpool_str220[sizeof("_Atomic")];
      char stringpool_str223[sizeof("__cold__")];
      char stringpool_str225[sizeof("typeof")];
      char stringpool_str226[sizeof("__interrupt_handler__")];
      char stringpool_str227[sizeof("returnspointerwithnoaliases")];
      char stringpool_str228[sizeof("__noplt__")];
      char stringpool_str229[sizeof("__artificial__")];
      char stringpool_str230[sizeof("break")];
      char stringpool_str231[sizeof("__attribute")];
      char stringpool_str232[sizeof("typeof_unqual")];
      char stringpool_str233[sizeof("__attribute__")];
      char stringpool_str234[sizeof("__cmn_err__")];
      char stringpool_str238[sizeof("_Float64")];
      char stringpool_str240[sizeof("__unused__")];
      char stringpool_str244[sizeof("__strfmon__")];
      char stringpool_str245[sizeof("__strftime__")];
      char stringpool_str246[sizeof("strlenesque")];
      char stringpool_str247[sizeof("__write_only")];
      char stringpool_str248[sizeof("__inline")];
      char stringpool_str249[sizeof("__write_only__")];
      char stringpool_str250[sizeof("__inline__")];
      char stringpool_str251[sizeof("struct")];
      char stringpool_str252[sizeof("__noinline__")];
      char stringpool_str253[sizeof("__asm")];
      char stringpool_str255[sizeof("__asm__")];
      char stringpool_str256[sizeof("signed")];
      char stringpool_str258[sizeof("continue")];
      char stringpool_str260[sizeof("_Decimal32")];
      char stringpool_str263[sizeof("_Thread_local")];
      char stringpool_str269[sizeof("__imag")];
      char stringpool_str271[sizeof("__imag__")];
      char stringpool_str272[sizeof("__imag__ ")];
      char stringpool_str273[sizeof("__deprecated__")];
      char stringpool_str277[sizeof("enum")];
      char stringpool_str279[sizeof("__scanf__")];
      char stringpool_str281[sizeof("__null")];
      char stringpool_str282[sizeof("__copy__")];
      char stringpool_str283[sizeof("_Float32")];
      char stringpool_str286[sizeof("__nonnull__")];
      char stringpool_str287[sizeof("default")];
      char stringpool_str291[sizeof("__aligned__")];
      char stringpool_str293[sizeof("pureconst")];
      char stringpool_str294[sizeof("__float80")];
      char stringpool_str297[sizeof("nosideeffect")];
      char stringpool_str298[sizeof("__vex")];
      char stringpool_str303[sizeof("__mcarch__")];
      char stringpool_str305[sizeof("__externally_visible__")];
      char stringpool_str306[sizeof("__pie__")];
      char stringpool_str307[sizeof("__hardbool__")];
      char stringpool_str309[sizeof("__packed__")];
      char stringpool_str310[sizeof("printfesque")];
      char stringpool_str311[sizeof("__simd__")];
      char stringpool_str317[sizeof("__no_split_stack__")];
      char stringpool_str319[sizeof("__printf__")];
      char stringpool_str320[sizeof("__pointer__")];
      char stringpool_str321[sizeof("__flatten__")];
      char stringpool_str324[sizeof("__avx2")];
      char stringpool_str325[sizeof("__vax__")];
      char stringpool_str326[sizeof("inline")];
      char stringpool_str327[sizeof("nocallersavedregisters")];
      char stringpool_str328[sizeof("__warn_if_not_aligned__")];
      char stringpool_str329[sizeof("__funline")];
      char stringpool_str331[sizeof("__assume_aligned__")];
      char stringpool_str335[sizeof("__msabi")];
      char stringpool_str338[sizeof("__signed")];
      char stringpool_str339[sizeof("__alias__")];
      char stringpool_str340[sizeof("__signed__")];
      char stringpool_str341[sizeof("__extension__")];
      char stringpool_str346[sizeof("__bf16")];
      char stringpool_str348[sizeof("register")];
      char stringpool_str349[sizeof("long")];
      char stringpool_str352[sizeof("__pure__")];
      char stringpool_str353[sizeof("__malloc__")];
      char stringpool_str359[sizeof("bool")];
      char stringpool_str361[sizeof("__pic__")];
      char stringpool_str362[sizeof("alignas")];
      char stringpool_str363[sizeof("__muarch__")];
      char stringpool_str366[sizeof("typedef")];
      char stringpool_str371[sizeof("strftimeesque")];
      char stringpool_str374[sizeof("returnsaligned")];
      char stringpool_str375[sizeof("__volatile")];
      char stringpool_str376[sizeof("vallocesque")];
      char stringpool_str377[sizeof("__volatile__")];
      char stringpool_str378[sizeof("__volatile__ ")];
      char stringpool_str383[sizeof("__seg_fs")];
      char stringpool_str384[sizeof("__ifunc__")];
      char stringpool_str385[sizeof("_Complex")];
      char stringpool_str386[sizeof("__auto_type")];
      char stringpool_str387[sizeof("constexpr")];
      char stringpool_str388[sizeof("_Imaginary")];
      char stringpool_str389[sizeof("__optimize__")];
      char stringpool_str390[sizeof("while")];
      char stringpool_str391[sizeof("thatispacked")];
      char stringpool_str398[sizeof("volatile")];
      char stringpool_str399[sizeof("__alignof")];
      char stringpool_str401[sizeof("__alignof__")];
      char stringpool_str402[sizeof("__builtin___")];
      char stringpool_str403[sizeof("__gnu_scanf__")];
      char stringpool_str404[sizeof("textwindows")];
      char stringpool_str405[sizeof("scanfesque")];
      char stringpool_str406[sizeof("__builtin_va_arg")];
      char stringpool_str408[sizeof("__builtin_offsetof")];
      char stringpool_str419[sizeof("libcesque")];
      char stringpool_str422[sizeof("alignof")];
      char stringpool_str424[sizeof("__mcfarch__")];
      char stringpool_str425[sizeof("paramsnonnull")];
      char stringpool_str428[sizeof("__ms_abi__")];
      char stringpool_str430[sizeof("__params_nonnull__")];
      char stringpool_str433[sizeof("_Alignof")];
      char stringpool_str434[sizeof("privileged")];
      char stringpool_str437[sizeof("dontcallback")];
      char stringpool_str438[sizeof("__byte__")];
      char stringpool_str440[sizeof("__alloc_align__")];
      char stringpool_str443[sizeof("__seg_gs")];
      char stringpool_str449[sizeof("__gnu_format__")];
      char stringpool_str453[sizeof("unsigned")];
      char stringpool_str456[sizeof("__warning__")];
      char stringpool_str458[sizeof("_Alignas")];
      char stringpool_str462[sizeof("__typeof")];
      char stringpool_str464[sizeof("__typeof__")];
      char stringpool_str468[sizeof("__symver__")];
      char stringpool_str471[sizeof("__typeof_unqual__")];
      char stringpool_str479[sizeof("__alloc_size__")];
      char stringpool_str487[sizeof("reallocesque")];
      char stringpool_str489[sizeof("mallocesque")];
      char stringpool_str491[sizeof("double")];
      char stringpool_str532[sizeof("__patchable_function_entry__")];
      char stringpool_str536[sizeof("__may_alias__")];
      char stringpool_str545[sizeof("forcealign")];
      char stringpool_str549[sizeof("__label__")];
      char stringpool_str554[sizeof("__gnu_inline__")];
      char stringpool_str555[sizeof("forcealignargpointer")];
      char stringpool_str556[sizeof("nullptr")];
      char stringpool_str559[sizeof("__visibility__")];
      char stringpool_str567[sizeof("__mcffpu__")];
      char stringpool_str572[sizeof("__sysv_abi__")];
      char stringpool_str575[sizeof("__mcfhwdiv__")];
      char stringpool_str582[sizeof("__always_inline__")];
      char stringpool_str601[sizeof("memcpyesque")];
      char stringpool_str613[sizeof("__gnu_printf__")];
      char stringpool_str630[sizeof("__complex")];
      char stringpool_str632[sizeof("__complex__")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "extern",
      "do",
      "__no_reorder__",
      "__no_instrument_function__",
      "auto",
      "__return__",
      "__noreturn__",
      "__error__",
      "__real",
      "__real__",
      "__real__ ",
      "__target__",
      "__read_only",
      "thread_local",
      "__read_only__",
      "__target_clones",
      "__target_clones__",
      "__no_address_safety_analysis__",
      "__read_write",
      "__zero_call_used_regs__",
      "__read_write__",
      "__const",
      "__const__",
      "__no_icf__",
      "__noclone__",
      "else",
      "sizeof",
      "__hot__",
      "__no_stack_limit__",
      "__no_stack_protector__",
      "short",
      "if",
      "int",
      "__strong__",
      "_BitInt",
      "_Generic",
      "char",
      "__retain__",
      "__returns_twice__",
      "__returns_nonnull__",
      "__thread",
      "__noipa__",
      "__vector_size__",
      "_Static_assert",
      "__no_sanitize__",
      "case",
      "const",
      "__no_sanitize_thread__",
      "__no_sanitize_address__",
      "__no_sanitize_undefined__",
      "_Noreturn",
      "__tainted_args__",
      "for",
      "return",
      "void",
      "__fentry__",
      "__transparent_union__",
      "restrict",
      "__restrict",
      "__restrict__",
      "float",
      "__sentinel__",
      "__fd_arg",
      "__destructor__",
      "__fd_arg__",
      "__word__",
      "_Bool",
      "__nothrow__",
      "__no_profile_instrument_function__",
      "__format__",
      "__format_arg__",
      "wontreturn",
      "static",
      "__weak__",
      "dontthrow",
      "__mode__",
      "__constructor__",
      "asm",
      "__force_align_arg_pointer__",
      "__used__",
      "goto",
      "_Decimal128",
      "__leaf__",
      "union",
      "__warn_unused_result__",
      "__unix__",
      "switch",
      "__access__",
      "__section__",
      "_Float16",
      "__no_caller_saved_registers__",
      "_Decimal64",
      "interruptfn",
      "__interrupt__",
      "_Float128",
      "_Atomic",
      "__cold__",
      "typeof",
      "__interrupt_handler__",
      "returnspointerwithnoaliases",
      "__noplt__",
      "__artificial__",
      "break",
      "__attribute",
      "typeof_unqual",
      "__attribute__",
      "__cmn_err__",
      "_Float64",
      "__unused__",
      "__strfmon__",
      "__strftime__",
      "strlenesque",
      "__write_only",
      "__inline",
      "__write_only__",
      "__inline__",
      "struct",
      "__noinline__",
      "__asm",
      "__asm__",
      "signed",
      "continue",
      "_Decimal32",
      "_Thread_local",
      "__imag",
      "__imag__",
      "__imag__ ",
      "__deprecated__",
      "enum",
      "__scanf__",
      "__null",
      "__copy__",
      "_Float32",
      "__nonnull__",
      "default",
      "__aligned__",
      "pureconst",
      "__float80",
      "nosideeffect",
      "__vex",
      "__mcarch__",
      "__externally_visible__",
      "__pie__",
      "__hardbool__",
      "__packed__",
      "printfesque",
      "__simd__",
      "__no_split_stack__",
      "__printf__",
      "__pointer__",
      "__flatten__",
      "__avx2",
      "__vax__",
      "inline",
      "nocallersavedregisters",
      "__warn_if_not_aligned__",
      "__funline",
      "__assume_aligned__",
      "__msabi",
      "__signed",
      "__alias__",
      "__signed__",
      "__extension__",
      "__bf16",
      "register",
      "long",
      "__pure__",
      "__malloc__",
      "bool",
      "__pic__",
      "alignas",
      "__muarch__",
      "typedef",
      "strftimeesque",
      "returnsaligned",
      "__volatile",
      "vallocesque",
      "__volatile__",
      "__volatile__ ",
      "__seg_fs",
      "__ifunc__",
      "_Complex",
      "__auto_type",
      "constexpr",
      "_Imaginary",
      "__optimize__",
      "while",
      "thatispacked",
      "volatile",
      "__alignof",
      "__alignof__",
      "__builtin___",
      "__gnu_scanf__",
      "textwindows",
      "scanfesque",
      "__builtin_va_arg",
      "__builtin_offsetof",
      "libcesque",
      "alignof",
      "__mcfarch__",
      "paramsnonnull",
      "__ms_abi__",
      "__params_nonnull__",
      "_Alignof",
      "privileged",
      "dontcallback",
      "__byte__",
      "__alloc_align__",
      "__seg_gs",
      "__gnu_format__",
      "unsigned",
      "__warning__",
      "_Alignas",
      "__typeof",
      "__typeof__",
      "__symver__",
      "__typeof_unqual__",
      "__alloc_size__",
      "reallocesque",
      "mallocesque",
      "double",
      "__patchable_function_entry__",
      "__may_alias__",
      "forcealign",
      "__label__",
      "__gnu_inline__",
      "forcealignargpointer",
      "nullptr",
      "__visibility__",
      "__mcffpu__",
      "__sysv_abi__",
      "__mcfhwdiv__",
      "__always_inline__",
      "memcpyesque",
      "__gnu_printf__",
      "__complex",
      "__complex__"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str26,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1,
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
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str141,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str157,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str175,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str181,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str185,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str187,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str189,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str200,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str202,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str203,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str210,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str211,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str213,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str215,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str220,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str223,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str225,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str226,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str227,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str228,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str230,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str231,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str232,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str233,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str234,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str238,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str240,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str244,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str245,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str246,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str247,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str248,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str249,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str250,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str251,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str252,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str253,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str255,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str256,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str258,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str260,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str263,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str269,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str271,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str272,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str273,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str277,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str279,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str281,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str282,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str283,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str286,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str287,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str291,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str293,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str294,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str297,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str298,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str303,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str305,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str306,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str307,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str309,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str310,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str311,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str317,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str319,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str320,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str321,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str324,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str325,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str326,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str327,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str328,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str329,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str331,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str335,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str338,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str339,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str340,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str341,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str346,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str348,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str349,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str352,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str353,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str359,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str361,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str362,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str363,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str366,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str371,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str374,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str375,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str376,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str377,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str378,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str383,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str384,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str385,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str386,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str387,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str388,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str389,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str390,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str391,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str398,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str399,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str401,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str402,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str403,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str404,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str405,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str406,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str408,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str419,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str422,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str424,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str425,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str428,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str430,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str433,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str434,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str437,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str438,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str440,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str443,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str449,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str453,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str456,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str458,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str462,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str464,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str468,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str471,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str479,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str487,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str489,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str491,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str532,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str536,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str545,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str549,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str554,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str555,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str556,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str559,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str567,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str572,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str575,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str582,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str601,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str613,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str630,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str632
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
