/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_swift.c llamafile/is_keyword_swift.gperf  */
/* Computed positions: -k'1-3,$' */

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

#line 1 "llamafile/is_keyword_swift.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 218
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 31
#define MIN_HASH_VALUE 18
#define MAX_HASH_VALUE 637
/* maximum key range = 620, duplicates = 0 */

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
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638,   0,   5,   0,   0, 638,
      638, 638, 638, 638, 638, 638, 638, 638,  30, 638,
       10, 638,   0,   5, 140,   0, 638, 638, 638, 638,
      638, 638, 638, 638, 638,  50, 638,  20, 249,  30,
        0,   5, 190, 174, 130, 125,   0, 115, 100, 250,
       20,  25,  95,   5,  15,  10,   0, 154, 140,  95,
      130,  25,  20, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638, 638, 638, 638, 638,
      638, 638, 638, 638, 638, 638
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
        hval += asso_values[(unsigned char)str[1]];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_swift (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str18[sizeof("set")];
      char stringpool_str28[sizeof("attached")];
      char stringpool_str31[sizeof("struct")];
      char stringpool_str33[sizeof("required")];
      char stringpool_str35[sizeof("derivative")];
      char stringpool_str36[sizeof("retroactive")];
      char stringpool_str38[sizeof("rethrows")];
      char stringpool_str41[sizeof("target")];
      char stringpool_str42[sizeof("as")];
      char stringpool_str43[sizeof("Sendable")];
      char stringpool_str44[sizeof("read")];
      char stringpool_str46[sizeof("return")];
      char stringpool_str47[sizeof("renamed")];
      char stringpool_str49[sizeof("transpose")];
      char stringpool_str50[sizeof("assignment")];
      char stringpool_str51[sizeof("addressWithOwner")];
      char stringpool_str52[sizeof("do")];
      char stringpool_str57[sizeof("addressWithNativeOwner")];
      char stringpool_str59[sizeof("associatedtype")];
      char stringpool_str62[sizeof("noDerivative")];
      char stringpool_str63[sizeof("noescape")];
      char stringpool_str66[sizeof("static")];
      char stringpool_str68[sizeof("try")];
      char stringpool_str69[sizeof("case")];
      char stringpool_str70[sizeof("actor")];
      char stringpool_str71[sizeof("scoped")];
      char stringpool_str72[sizeof("_RefCountedObject")];
      char stringpool_str73[sizeof("Any")];
      char stringpool_str74[sizeof("none")];
      char stringpool_str75[sizeof("_read")];
      char stringpool_str76[sizeof("nonisolated")];
      char stringpool_str77[sizeof("reasync")];
      char stringpool_str78[sizeof("associativity")];
      char stringpool_str79[sizeof("canImport")];
      char stringpool_str82[sizeof("dynamic")];
      char stringpool_str83[sizeof("_BridgeObject")];
      char stringpool_str84[sizeof("_UnknownLayout")];
      char stringpool_str85[sizeof("_semantics")];
      char stringpool_str87[sizeof("consume")];
      char stringpool_str88[sizeof("continue")];
      char stringpool_str90[sizeof("async")];
      char stringpool_str91[sizeof("convenience")];
      char stringpool_str93[sizeof("any")];
      char stringpool_str94[sizeof("_dynamicReplacement")];
      char stringpool_str98[sizeof("accesses")];
      char stringpool_str101[sizeof("_typeEraser")];
      char stringpool_str102[sizeof("noasync")];
      char stringpool_str105[sizeof("convention")];
      char stringpool_str108[sizeof("let")];
      char stringpool_str109[sizeof("_documentation")];
      char stringpool_str110[sizeof("deprecated")];
      char stringpool_str111[sizeof("_const")];
      char stringpool_str112[sizeof("_nonSendable")];
      char stringpool_str113[sizeof("wrt")];
      char stringpool_str118[sizeof("__shared")];
      char stringpool_str119[sizeof("_PackageDescription")];
      char stringpool_str121[sizeof("repeat")];
      char stringpool_str122[sizeof("_NativeClass")];
      char stringpool_str123[sizeof("_NativeRefCountedObject")];
      char stringpool_str124[sizeof("else")];
      char stringpool_str125[sizeof("_compilerInitialized")];
      char stringpool_str126[sizeof("_noMetadata")];
      char stringpool_str129[sizeof("dependsOn")];
      char stringpool_str130[sizeof("_originallyDefinedIn")];
      char stringpool_str131[sizeof("didSet")];
      char stringpool_str132[sizeof("__owned")];
      char stringpool_str135[sizeof("__setter_access")];
      char stringpool_str136[sizeof("deinit")];
      char stringpool_str139[sizeof("typealias")];
      char stringpool_str140[sizeof("await")];
      char stringpool_str142[sizeof("discard")];
      char stringpool_str146[sizeof("distributed")];
      char stringpool_str147[sizeof("is")];
      char stringpool_str148[sizeof("operator")];
      char stringpool_str149[sizeof("open")];
      char stringpool_str153[sizeof("indirect")];
      char stringpool_str154[sizeof("preconcurrency")];
      char stringpool_str155[sizeof("introduced")];
      char stringpool_str157[sizeof("package")];
      char stringpool_str158[sizeof("Protocol")];
      char stringpool_str159[sizeof("then")];
      char stringpool_str160[sizeof("yield")];
      char stringpool_str161[sizeof("throws")];
      char stringpool_str164[sizeof("extension")];
      char stringpool_str165[sizeof("class")];
      char stringpool_str166[sizeof("_Class")];
      char stringpool_str167[sizeof("in")];
      char stringpool_str168[sizeof("isolated")];
      char stringpool_str169[sizeof("lazy")];
      char stringpool_str171[sizeof("_specialize")];
      char stringpool_str172[sizeof("reverse")];
      char stringpool_str173[sizeof("_private")];
      char stringpool_str174[sizeof("_spi_available")];
      char stringpool_str175[sizeof("inout")];
      char stringpool_str178[sizeof("true")];
      char stringpool_str179[sizeof("copy")];
      char stringpool_str180[sizeof("_alignment")];
      char stringpool_str182[sizeof("get")];
      char stringpool_str183[sizeof("override")];
      char stringpool_str184[sizeof("_optimize")];
      char stringpool_str185[sizeof("catch")];
      char stringpool_str186[sizeof("_cdecl")];
      char stringpool_str189[sizeof("each")];
      char stringpool_str190[sizeof("autoclosure")];
      char stringpool_str191[sizeof("_swift_native_objc_runtime_base")];
      char stringpool_str193[sizeof("var")];
      char stringpool_str194[sizeof("available")];
      char stringpool_str195[sizeof("unsafe")];
      char stringpool_str197[sizeof("_expose")];
      char stringpool_str201[sizeof("exclusivity")];
      char stringpool_str202[sizeof("default")];
      char stringpool_str204[sizeof("sourceFile")];
      char stringpool_str205[sizeof("cType")];
      char stringpool_str206[sizeof("unowned")];
      char stringpool_str207[sizeof("unsafeAddress")];
      char stringpool_str208[sizeof("_projectedValueProperty")];
      char stringpool_str210[sizeof("unavailable")];
      char stringpool_str213[sizeof("unchecked")];
      char stringpool_str214[sizeof("unsafeMutableAddress")];
      char stringpool_str215[sizeof("defer")];
      char stringpool_str216[sizeof("sending")];
      char stringpool_str217[sizeof("availability")];
      char stringpool_str219[sizeof("_TrivialAtMost")];
      char stringpool_str223[sizeof("_version")];
      char stringpool_str224[sizeof("_TrivialStride")];
      char stringpool_str225[sizeof("precedencegroup")];
      char stringpool_str227[sizeof("escaping")];
      char stringpool_str228[sizeof("optional")];
      char stringpool_str229[sizeof("safe")];
      char stringpool_str234[sizeof("witness_method")];
      char stringpool_str235[sizeof("swift")];
      char stringpool_str237[sizeof("forward")];
      char stringpool_str238[sizeof("exported")];
      char stringpool_str239[sizeof("weak")];
      char stringpool_str240[sizeof("where")];
      char stringpool_str243[sizeof("protocol")];
      char stringpool_str244[sizeof("spiModule")];
      char stringpool_str245[sizeof("throw")];
      char stringpool_str247[sizeof("private")];
      char stringpool_str248[sizeof("for")];
      char stringpool_str249[sizeof("lowerThan")];
      char stringpool_str250[sizeof("nonmutating")];
      char stringpool_str251[sizeof("prefix")];
      char stringpool_str253[sizeof("internal")];
      char stringpool_str254[sizeof("line")];
      char stringpool_str256[sizeof("inline")];
      char stringpool_str258[sizeof("consuming")];
      char stringpool_str262[sizeof("_underlyingVersion")];
      char stringpool_str263[sizeof("_effects")];
      char stringpool_str264[sizeof("kind")];
      char stringpool_str266[sizeof("linear")];
      char stringpool_str267[sizeof("postfix")];
      char stringpool_str269[sizeof("Type")];
      char stringpool_str273[sizeof("_forward")];
      char stringpool_str274[sizeof("init")];
      char stringpool_str275[sizeof("_unavailableFromAsync")];
      char stringpool_str277[sizeof("message")];
      char stringpool_str279[sizeof("super")];
      char stringpool_str281[sizeof("_local")];
      char stringpool_str283[sizeof("metadata")];
      char stringpool_str284[sizeof("_spi")];
      char stringpool_str286[sizeof("module")];
      char stringpool_str289[sizeof("_consuming")];
      char stringpool_str291[sizeof("initializes")];
      char stringpool_str293[sizeof("obsoleted")];
      char stringpool_str294[sizeof("some")];
      char stringpool_str297[sizeof("_linear")];
      char stringpool_str299[sizeof("left")];
      char stringpool_str304[sizeof("Self")];
      char stringpool_str306[sizeof("modify")];
      char stringpool_str308[sizeof("objc")];
      char stringpool_str309[sizeof("self")];
      char stringpool_str310[sizeof("visibility")];
      char stringpool_str311[sizeof("backDeployed")];
      char stringpool_str313[sizeof("_Trivial")];
      char stringpool_str315[sizeof("__consuming")];
      char stringpool_str319[sizeof("right")];
      char stringpool_str320[sizeof("false")];
      char stringpool_str327[sizeof("willSet")];
      char stringpool_str328[sizeof("compiler")];
      char stringpool_str330[sizeof("macro")];
      char stringpool_str334[sizeof("differentiable")];
      char stringpool_str335[sizeof("_move")];
      char stringpool_str345[sizeof("_objcRuntimeName")];
      char stringpool_str348[sizeof("nil")];
      char stringpool_str353[sizeof("guard")];
      char stringpool_str355[sizeof("_backDeploy")];
      char stringpool_str357[sizeof("_modify")];
      char stringpool_str358[sizeof("spi")];
      char stringpool_str360[sizeof("while")];
      char stringpool_str363[sizeof("_objcImplementation")];
      char stringpool_str366[sizeof("switch")];
      char stringpool_str379[sizeof("_opaqueReturnTypeOf")];
      char stringpool_str389[sizeof("break")];
      char stringpool_str396[sizeof("freestanding")];
      char stringpool_str398[sizeof("func")];
      char stringpool_str407[sizeof("of")];
      char stringpool_str422[sizeof("subscript")];
      char stringpool_str424[sizeof("file")];
      char stringpool_str426[sizeof("_borrow")];
      char stringpool_str431[sizeof("fileprivate")];
      char stringpool_str433[sizeof("enum")];
      char stringpool_str440[sizeof("final")];
      char stringpool_str442[sizeof("mutableAddressWithOwner")];
      char stringpool_str446[sizeof("_implements")];
      char stringpool_str448[sizeof("mutableAddressWithNativeOwner")];
      char stringpool_str451[sizeof("fallthrough")];
      char stringpool_str455[sizeof("before")];
      char stringpool_str459[sizeof("higherThan")];
      char stringpool_str470[sizeof("infix")];
      char stringpool_str472[sizeof("borrowing")];
      char stringpool_str476[sizeof("import")];
      char stringpool_str494[sizeof("block")];
      char stringpool_str507[sizeof("if")];
      char stringpool_str508[sizeof("_borrowing")];
      char stringpool_str534[sizeof("public")];
      char stringpool_str586[sizeof("mutating")];
      char stringpool_str637[sizeof("_mutating")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "set",
      "attached",
      "struct",
      "required",
      "derivative",
      "retroactive",
      "rethrows",
      "target",
      "as",
      "Sendable",
      "read",
      "return",
      "renamed",
      "transpose",
      "assignment",
      "addressWithOwner",
      "do",
      "addressWithNativeOwner",
      "associatedtype",
      "noDerivative",
      "noescape",
      "static",
      "try",
      "case",
      "actor",
      "scoped",
      "_RefCountedObject",
      "Any",
      "none",
      "_read",
      "nonisolated",
      "reasync",
      "associativity",
      "canImport",
      "dynamic",
      "_BridgeObject",
      "_UnknownLayout",
      "_semantics",
      "consume",
      "continue",
      "async",
      "convenience",
      "any",
      "_dynamicReplacement",
      "accesses",
      "_typeEraser",
      "noasync",
      "convention",
      "let",
      "_documentation",
      "deprecated",
      "_const",
      "_nonSendable",
      "wrt",
      "__shared",
      "_PackageDescription",
      "repeat",
      "_NativeClass",
      "_NativeRefCountedObject",
      "else",
      "_compilerInitialized",
      "_noMetadata",
      "dependsOn",
      "_originallyDefinedIn",
      "didSet",
      "__owned",
      "__setter_access",
      "deinit",
      "typealias",
      "await",
      "discard",
      "distributed",
      "is",
      "operator",
      "open",
      "indirect",
      "preconcurrency",
      "introduced",
      "package",
      "Protocol",
      "then",
      "yield",
      "throws",
      "extension",
      "class",
      "_Class",
      "in",
      "isolated",
      "lazy",
      "_specialize",
      "reverse",
      "_private",
      "_spi_available",
      "inout",
      "true",
      "copy",
      "_alignment",
      "get",
      "override",
      "_optimize",
      "catch",
      "_cdecl",
      "each",
      "autoclosure",
      "_swift_native_objc_runtime_base",
      "var",
      "available",
      "unsafe",
      "_expose",
      "exclusivity",
      "default",
      "sourceFile",
      "cType",
      "unowned",
      "unsafeAddress",
      "_projectedValueProperty",
      "unavailable",
      "unchecked",
      "unsafeMutableAddress",
      "defer",
      "sending",
      "availability",
      "_TrivialAtMost",
      "_version",
      "_TrivialStride",
      "precedencegroup",
      "escaping",
      "optional",
      "safe",
      "witness_method",
      "swift",
      "forward",
      "exported",
      "weak",
      "where",
      "protocol",
      "spiModule",
      "throw",
      "private",
      "for",
      "lowerThan",
      "nonmutating",
      "prefix",
      "internal",
      "line",
      "inline",
      "consuming",
      "_underlyingVersion",
      "_effects",
      "kind",
      "linear",
      "postfix",
      "Type",
      "_forward",
      "init",
      "_unavailableFromAsync",
      "message",
      "super",
      "_local",
      "metadata",
      "_spi",
      "module",
      "_consuming",
      "initializes",
      "obsoleted",
      "some",
      "_linear",
      "left",
      "Self",
      "modify",
      "objc",
      "self",
      "visibility",
      "backDeployed",
      "_Trivial",
      "__consuming",
      "right",
      "false",
      "willSet",
      "compiler",
      "macro",
      "differentiable",
      "_move",
      "_objcRuntimeName",
      "nil",
      "guard",
      "_backDeploy",
      "_modify",
      "spi",
      "while",
      "_objcImplementation",
      "switch",
      "_opaqueReturnTypeOf",
      "break",
      "freestanding",
      "func",
      "of",
      "subscript",
      "file",
      "_borrow",
      "fileprivate",
      "enum",
      "final",
      "mutableAddressWithOwner",
      "_implements",
      "mutableAddressWithNativeOwner",
      "fallthrough",
      "before",
      "higherThan",
      "infix",
      "borrowing",
      "import",
      "block",
      "if",
      "_borrowing",
      "public",
      "mutating",
      "_mutating"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str157,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str175,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str182,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str184,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str185,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str189,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str194,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str195,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str197,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str201,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str202,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str204,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str205,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str207,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str208,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str210,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str213,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str215,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str217,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str223,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str224,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str225,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str227,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str228,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str234,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str235,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str237,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str238,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str239,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str240,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str243,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str244,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str245,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str247,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str248,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str249,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str250,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str251,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str253,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str254,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str256,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str258,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str262,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str263,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str264,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str266,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str267,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str269,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str273,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str274,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str275,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str277,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str279,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str281,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str283,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str284,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str286,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str289,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str291,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str293,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str294,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str297,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str299,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str304,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str306,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str308,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str309,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str310,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str311,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str313,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str315,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str319,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str320,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str327,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str328,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str330,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str334,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str335,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str345,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str348,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str353,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str355,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str357,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str358,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str360,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str363,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str366,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str379,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str389,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str396,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str398,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str407,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str422,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str424,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str426,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str431,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str433,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str440,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str442,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str446,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str448,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str451,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str455,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str459,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str470,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str472,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str476,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str494,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str507,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str508,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str534,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str586,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str637
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
