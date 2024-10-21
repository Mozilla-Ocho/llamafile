/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_js_builtin.gperf  */
/* Computed positions: -k'1,5-6' */

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

#line 1 "llamafile/is_keyword_js_builtin.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 65
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 22
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 102
/* maximum key range = 100, duplicates = 0 */

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
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103,  25,
       15,  15, 103, 103,  10, 103,   5, 103, 103, 103,
      103, 103, 103, 103, 103,  10,  25,  25,  50,  15,
       25,   0, 103,   0,  70, 103, 103,  60,   5,  10,
        5, 103,  25,   0,  40,  25,  15,  20, 103, 103,
      103, 103, 103, 103, 103, 103, 103,  15, 103,  10,
        0,   5, 103,  30, 103,   0, 103, 103,   0, 103,
        0,   0,   0, 103,   5,   0,  15,  35, 103, 103,
        0,   5, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
      103, 103, 103, 103, 103, 103
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[5]];
      /*FALLTHROUGH*/
      case 5:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
      case 3:
      case 2:
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

const char *
is_keyword_js_builtin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("Set")];
      char stringpool_str4[sizeof("Intl")];
      char stringpool_str6[sizeof("Symbol")];
      char stringpool_str8[sizeof("isFinite")];
      char stringpool_str9[sizeof("eval")];
      char stringpool_str10[sizeof("isNaN")];
      char stringpool_str12[sizeof("Promise")];
      char stringpool_str13[sizeof("parseInt")];
      char stringpool_str14[sizeof("decodeURI")];
      char stringpool_str15[sizeof("Proxy")];
      char stringpool_str16[sizeof("escape")];
      char stringpool_str18[sizeof("InternalError")];
      char stringpool_str19[sizeof("encodeURI")];
      char stringpool_str20[sizeof("Array")];
      char stringpool_str21[sizeof("Number")];
      char stringpool_str22[sizeof("SharedArrayBuffer")];
      char stringpool_str23[sizeof("decodeURIComponent")];
      char stringpool_str24[sizeof("Int8Array")];
      char stringpool_str25[sizeof("Error")];
      char stringpool_str26[sizeof("SyntaxError")];
      char stringpool_str27[sizeof("Atomics")];
      char stringpool_str28[sizeof("encodeURIComponent")];
      char stringpool_str29[sizeof("Generator")];
      char stringpool_str30[sizeof("Int16Array")];
      char stringpool_str31[sizeof("RegExp")];
      char stringpool_str32[sizeof("WeakSet")];
      char stringpool_str33[sizeof("AsyncIterator")];
      char stringpool_str34[sizeof("AsyncGenerator")];
      char stringpool_str35[sizeof("Int32Array")];
      char stringpool_str36[sizeof("String")];
      char stringpool_str37[sizeof("GeneratorFunction")];
      char stringpool_str38[sizeof("Iterator")];
      char stringpool_str39[sizeof("BigUint64Array")];
      char stringpool_str40[sizeof("parseFloat")];
      char stringpool_str41[sizeof("Object")];
      char stringpool_str42[sizeof("AsyncGeneratorFunction")];
      char stringpool_str43[sizeof("URIError")];
      char stringpool_str44[sizeof("EvalError")];
      char stringpool_str45[sizeof("FinalizationRegistry")];
      char stringpool_str46[sizeof("BigInt")];
      char stringpool_str47[sizeof("Reflect")];
      char stringpool_str48[sizeof("Function")];
      char stringpool_str49[sizeof("ReferenceError")];
      char stringpool_str50[sizeof("Uint8Array")];
      char stringpool_str51[sizeof("ArrayBuffer")];
      char stringpool_str52[sizeof("Boolean")];
      char stringpool_str53[sizeof("BigInt64Array")];
      char stringpool_str54[sizeof("Date")];
      char stringpool_str55[sizeof("RangeError")];
      char stringpool_str57[sizeof("WeakRef")];
      char stringpool_str58[sizeof("AsyncFunction")];
      char stringpool_str59[sizeof("AggregateError")];
      char stringpool_str62[sizeof("Float64Array")];
      char stringpool_str63[sizeof("Map")];
      char stringpool_str64[sizeof("Math")];
      char stringpool_str66[sizeof("Uint32Array")];
      char stringpool_str67[sizeof("Float32Array")];
      char stringpool_str68[sizeof("unescape")];
      char stringpool_str69[sizeof("TypeError")];
      char stringpool_str71[sizeof("Uint16Array")];
      char stringpool_str72[sizeof("Uint8ClampedArray")];
      char stringpool_str73[sizeof("DataView")];
      char stringpool_str74[sizeof("JSON")];
      char stringpool_str77[sizeof("Float16Array")];
      char stringpool_str102[sizeof("WeakMap")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "Set",
      "Intl",
      "Symbol",
      "isFinite",
      "eval",
      "isNaN",
      "Promise",
      "parseInt",
      "decodeURI",
      "Proxy",
      "escape",
      "InternalError",
      "encodeURI",
      "Array",
      "Number",
      "SharedArrayBuffer",
      "decodeURIComponent",
      "Int8Array",
      "Error",
      "SyntaxError",
      "Atomics",
      "encodeURIComponent",
      "Generator",
      "Int16Array",
      "RegExp",
      "WeakSet",
      "AsyncIterator",
      "AsyncGenerator",
      "Int32Array",
      "String",
      "GeneratorFunction",
      "Iterator",
      "BigUint64Array",
      "parseFloat",
      "Object",
      "AsyncGeneratorFunction",
      "URIError",
      "EvalError",
      "FinalizationRegistry",
      "BigInt",
      "Reflect",
      "Function",
      "ReferenceError",
      "Uint8Array",
      "ArrayBuffer",
      "Boolean",
      "BigInt64Array",
      "Date",
      "RangeError",
      "WeakRef",
      "AsyncFunction",
      "AggregateError",
      "Float64Array",
      "Map",
      "Math",
      "Uint32Array",
      "Float32Array",
      "unescape",
      "TypeError",
      "Uint16Array",
      "Uint8ClampedArray",
      "DataView",
      "JSON",
      "Float16Array",
      "WeakMap"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1, -1,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102
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
