/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_zig_builtin.gperf  */
/* Computed positions: -k'2-4,$' */

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

#line 1 "llamafile/is_keyword_zig_builtin.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 117
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 19
#define MIN_HASH_VALUE 8
#define MAX_HASH_VALUE 269
/* maximum key range = 262, duplicates = 0 */

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
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270,  10, 270,
       25, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270,   5,   0, 270,
       25, 270, 270,   5, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270,  20,   0,  70, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270,  15,  60,  70,
        5,   5,  15,  45, 100,   0, 270,   0,  10,  50,
        0,  15,  90,   0,   5,  10,   0,  65, 112,  20,
       40,  40,  10, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270, 270, 270, 270, 270,
      270, 270, 270, 270, 270, 270
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
        hval += asso_values[(unsigned char)str[1]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_zig_builtin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str8[sizeof("@intCast")];
      char stringpool_str13[sizeof("@intFromFloat")];
      char stringpool_str14[sizeof("@sin")];
      char stringpool_str16[sizeof("@intFromPtr")];
      char stringpool_str18[sizeof("@intFromError")];
      char stringpool_str19[sizeof("@tan")];
      char stringpool_str20[sizeof("@sqrt")];
      char stringpool_str21[sizeof("@inComptime")];
      char stringpool_str22[sizeof("@intFromBool")];
      char stringpool_str25[sizeof("@errorCast")];
      char stringpool_str27[sizeof("@reduce")];
      char stringpool_str28[sizeof("@errorFromInt")];
      char stringpool_str29[sizeof("@setAlignStack")];
      char stringpool_str30[sizeof("@errorName")];
      char stringpool_str31[sizeof("@field")];
      char stringpool_str32[sizeof("@select")];
      char stringpool_str33[sizeof("@setFloatMode")];
      char stringpool_str34[sizeof("@returnAddress")];
      char stringpool_str35[sizeof("@alignCast")];
      char stringpool_str37[sizeof("@errorReturnTrace")];
      char stringpool_str38[sizeof("@as")];
      char stringpool_str39[sizeof("@addrSpaceCast")];
      char stringpool_str40[sizeof("@fieldParentPtr")];
      char stringpool_str42[sizeof("@sizeOf")];
      char stringpool_str45[sizeof("@FieldType")];
      char stringpool_str46[sizeof("@atomicLoad")];
      char stringpool_str47[sizeof("@atomicStore")];
      char stringpool_str48[sizeof("@alignOf")];
      char stringpool_str49[sizeof("@setEvalBranchQuota")];
      char stringpool_str50[sizeof("@floatCast")];
      char stringpool_str51[sizeof("@floor")];
      char stringpool_str52[sizeof("@extern")];
      char stringpool_str53[sizeof("@floatFromInt")];
      char stringpool_str54[sizeof("@min")];
      char stringpool_str56[sizeof("@workItemId")];
      char stringpool_str57[sizeof("@workGroupId")];
      char stringpool_str58[sizeof("@frameAddress")];
      char stringpool_str59[sizeof("@workGroupSize")];
      char stringpool_str60[sizeof("@atomicRmw")];
      char stringpool_str61[sizeof("@addWithOverflow")];
      char stringpool_str62[sizeof("@intFromEnum")];
      char stringpool_str65[sizeof("@wasmMemorySize")];
      char stringpool_str68[sizeof("@bitCast")];
      char stringpool_str69[sizeof("@offsetOf")];
      char stringpool_str72[sizeof("@setRuntimeSafety")];
      char stringpool_str73[sizeof("@tagName")];
      char stringpool_str75[sizeof("@unionInit")];
      char stringpool_str76[sizeof("@bitReverse")];
      char stringpool_str79[sizeof("@mod")];
      char stringpool_str80[sizeof("@wasmMemoryGrow")];
      char stringpool_str81[sizeof("@breakpoint")];
      char stringpool_str82[sizeof("@enumFromInt")];
      char stringpool_str84[sizeof("@truncate")];
      char stringpool_str85[sizeof("@bitSizeOf")];
      char stringpool_str86[sizeof("@log10")];
      char stringpool_str87[sizeof("@bitOffsetOf")];
      char stringpool_str88[sizeof("@cDefine")];
      char stringpool_str89[sizeof("@cInclude")];
      char stringpool_str90[sizeof("@ceil")];
      char stringpool_str91[sizeof("@branchHint")];
      char stringpool_str92[sizeof("@cUndef")];
      char stringpool_str94[sizeof("@ctz")];
      char stringpool_str95[sizeof("@constCast")];
      char stringpool_str96[sizeof("@round")];
      char stringpool_str99[sizeof("@abs")];
      char stringpool_str100[sizeof("@log2")];
      char stringpool_str103[sizeof("@ptrCast")];
      char stringpool_str104[sizeof("@clz")];
      char stringpool_str106[sizeof("@ptrFromInt")];
      char stringpool_str109[sizeof("@cos")];
      char stringpool_str110[sizeof("@call")];
      char stringpool_str112[sizeof("@memset")];
      char stringpool_str114[sizeof("@rem")];
      char stringpool_str115[sizeof("@trap")];
      char stringpool_str116[sizeof("@splat")];
      char stringpool_str119[sizeof("@log")];
      char stringpool_str124[sizeof("@shrExact")];
      char stringpool_str126[sizeof("@divExact")];
      char stringpool_str129[sizeof("@shlExact")];
      char stringpool_str130[sizeof("@embedFile")];
      char stringpool_str131[sizeof("@divFloor")];
      char stringpool_str133[sizeof("@cImport")];
      char stringpool_str135[sizeof("@This")];
      char stringpool_str137[sizeof("@mulAdd")];
      char stringpool_str139[sizeof("@hasField")];
      char stringpool_str142[sizeof("@export")];
      char stringpool_str143[sizeof("@hasDecl")];
      char stringpool_str144[sizeof("@typeName")];
      char stringpool_str146[sizeof("@trunc")];
      char stringpool_str147[sizeof("@import")];
      char stringpool_str149[sizeof("@max")];
      char stringpool_str150[sizeof("@volatileCast")];
      char stringpool_str152[sizeof("@memcpy")];
      char stringpool_str153[sizeof("@compileError")];
      char stringpool_str154[sizeof("@typeInfo")];
      char stringpool_str156[sizeof("@shlWithOverflow")];
      char stringpool_str157[sizeof("@Vector")];
      char stringpool_str159[sizeof("@src")];
      char stringpool_str160[sizeof("@Type")];
      char stringpool_str161[sizeof("@mulWithOverflow")];
      char stringpool_str164[sizeof("@cVaStart")];
      char stringpool_str165[sizeof("@exp2")];
      char stringpool_str167[sizeof("@cVaEnd")];
      char stringpool_str171[sizeof("@subWithOverflow")];
      char stringpool_str172[sizeof("@TypeOf")];
      char stringpool_str181[sizeof("@panic")];
      char stringpool_str188[sizeof("@shuffle")];
      char stringpool_str191[sizeof("@compileLog")];
      char stringpool_str196[sizeof("@divTrunc")];
      char stringpool_str199[sizeof("@byteSwap")];
      char stringpool_str203[sizeof("@cVaCopy")];
      char stringpool_str204[sizeof("@popCount")];
      char stringpool_str207[sizeof("@cVaArg")];
      char stringpool_str209[sizeof("@prefetch")];
      char stringpool_str222[sizeof("@cmpxchgWeak")];
      char stringpool_str229[sizeof("@exp")];
      char stringpool_str269[sizeof("@cmpxchgStrong")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "@intCast",
      "@intFromFloat",
      "@sin",
      "@intFromPtr",
      "@intFromError",
      "@tan",
      "@sqrt",
      "@inComptime",
      "@intFromBool",
      "@errorCast",
      "@reduce",
      "@errorFromInt",
      "@setAlignStack",
      "@errorName",
      "@field",
      "@select",
      "@setFloatMode",
      "@returnAddress",
      "@alignCast",
      "@errorReturnTrace",
      "@as",
      "@addrSpaceCast",
      "@fieldParentPtr",
      "@sizeOf",
      "@FieldType",
      "@atomicLoad",
      "@atomicStore",
      "@alignOf",
      "@setEvalBranchQuota",
      "@floatCast",
      "@floor",
      "@extern",
      "@floatFromInt",
      "@min",
      "@workItemId",
      "@workGroupId",
      "@frameAddress",
      "@workGroupSize",
      "@atomicRmw",
      "@addWithOverflow",
      "@intFromEnum",
      "@wasmMemorySize",
      "@bitCast",
      "@offsetOf",
      "@setRuntimeSafety",
      "@tagName",
      "@unionInit",
      "@bitReverse",
      "@mod",
      "@wasmMemoryGrow",
      "@breakpoint",
      "@enumFromInt",
      "@truncate",
      "@bitSizeOf",
      "@log10",
      "@bitOffsetOf",
      "@cDefine",
      "@cInclude",
      "@ceil",
      "@branchHint",
      "@cUndef",
      "@ctz",
      "@constCast",
      "@round",
      "@abs",
      "@log2",
      "@ptrCast",
      "@clz",
      "@ptrFromInt",
      "@cos",
      "@call",
      "@memset",
      "@rem",
      "@trap",
      "@splat",
      "@log",
      "@shrExact",
      "@divExact",
      "@shlExact",
      "@embedFile",
      "@divFloor",
      "@cImport",
      "@This",
      "@mulAdd",
      "@hasField",
      "@export",
      "@hasDecl",
      "@typeName",
      "@trunc",
      "@import",
      "@max",
      "@volatileCast",
      "@memcpy",
      "@compileError",
      "@typeInfo",
      "@shlWithOverflow",
      "@Vector",
      "@src",
      "@Type",
      "@mulWithOverflow",
      "@cVaStart",
      "@exp2",
      "@cVaEnd",
      "@subWithOverflow",
      "@TypeOf",
      "@panic",
      "@shuffle",
      "@compileLog",
      "@divTrunc",
      "@byteSwap",
      "@cVaCopy",
      "@popCount",
      "@cVaArg",
      "@prefetch",
      "@cmpxchgWeak",
      "@exp",
      "@cmpxchgStrong"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str22,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str31,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str157,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str181,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str196,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str199,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str203,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str204,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str207,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str209,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str222,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str269
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
