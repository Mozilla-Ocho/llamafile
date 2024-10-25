/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_basic_constant.c llamafile/is_keyword_basic_constant.gperf  */
/* Computed positions: -k'3-4,8,$' */

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

#line 1 "llamafile/is_keyword_basic_constant.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 105
#define MIN_WORD_LENGTH 4
#define MAX_WORD_LENGTH 21
#define MIN_HASH_VALUE 8
#define MAX_HASH_VALUE 230
/* maximum key range = 223, duplicates = 0 */

#ifndef GPERF_DOWNCASE
#define GPERF_DOWNCASE 1
static unsigned char gperf_downcase[256] =
  {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
     30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
     45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
     60,  61,  62,  63,  64,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    122,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
    195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
    255
  };
#endif

#ifndef GPERF_CASE_STRNCMP
#define GPERF_CASE_STRNCMP 1
static int
gperf_case_strncmp (register const char *s1, register const char *s2, register size_t n)
{
  for (; n > 0;)
    {
      unsigned char c1 = gperf_downcase[(unsigned char)*s1++];
      unsigned char c2 = gperf_downcase[(unsigned char)*s2++];
      if (c1 != 0 && c1 == c2)
        {
          n--;
          continue;
        }
      return (int)c1 - (int)c2;
    }
  return 0;
}
#endif

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
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231,   0,
       25,   5, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231,   5,  30,  90,  75,   0,
       25,  50,  70,   0,   0, 105,  10,   0,  10,   5,
       25,   5,  30,   0,  30,  35,  50,  75, 110,   5,
      231, 231, 231, 231, 231, 231, 231,   5,  30,  90,
       75,   0,  25,  50,  70,   0,   0, 105,  10,   0,
       10,   5,  25,   5,  30,   0,  30,  35,  50,  75,
      110,   5, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
      231, 231, 231, 231, 231, 231
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[7]];
      /*FALLTHROUGH*/
      case 7:
      case 6:
      case 5:
      case 4:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
        hval += asso_values[(unsigned char)str[2]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_basic_constant (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str8[sizeof("vbSingle")];
      char stringpool_str10[sizeof("vbYes")];
      char stringpool_str12[sizeof("vbEmpty")];
      char stringpool_str13[sizeof("vbSystem")];
      char stringpool_str15[sizeof("False")];
      char stringpool_str16[sizeof("vbMinimizedFocus")];
      char stringpool_str17[sizeof("vbYesNo")];
      char stringpool_str18[sizeof("vbMinimizedNoFocus")];
      char stringpool_str19[sizeof("vbSimplifiedChinese")];
      char stringpool_str21[sizeof("vbMaximizedFocus")];
      char stringpool_str23[sizeof("vbMonday")];
      char stringpool_str24[sizeof("vbNo")];
      char stringpool_str25[sizeof("vbLongTime")];
      char stringpool_str28[sizeof("vbSystemModal")];
      char stringpool_str29[sizeof("vbNewLine")];
      char stringpool_str30[sizeof("vbLongDate")];
      char stringpool_str33[sizeof("vbInformation")];
      char stringpool_str35[sizeof("vbSet")];
      char stringpool_str36[sizeof("vbFirstJan1")];
      char stringpool_str37[sizeof("vbFalse")];
      char stringpool_str38[sizeof("vbNormalFocus")];
      char stringpool_str39[sizeof("True")];
      char stringpool_str40[sizeof("vbNormalNoFocus")];
      char stringpool_str41[sizeof("vbByte")];
      char stringpool_str42[sizeof("vbRetry")];
      char stringpool_str43[sizeof("vbNormal")];
      char stringpool_str45[sizeof("vbLet")];
      char stringpool_str46[sizeof("vbUseSystem")];
      char stringpool_str47[sizeof("vbArray")];
      char stringpool_str48[sizeof("vbTextCompare")];
      char stringpool_str49[sizeof("vbInteger")];
      char stringpool_str50[sizeof("vbBinaryCompare")];
      char stringpool_str52[sizeof("vbUserDefinedType")];
      char stringpool_str53[sizeof("vbSunday")];
      char stringpool_str55[sizeof("vbReadOnly")];
      char stringpool_str58[sizeof("vbIgnore")];
      char stringpool_str59[sizeof("vbBoolean")];
      char stringpool_str60[sizeof("vbQuestion")];
      char stringpool_str61[sizeof("vbNull")];
      char stringpool_str63[sizeof("vbVolume")];
      char stringpool_str64[sizeof("vbLf")];
      char stringpool_str65[sizeof("vbFirstFourDays")];
      char stringpool_str66[sizeof("vbTrue")];
      char stringpool_str68[sizeof("vbGeneralDate")];
      char stringpool_str70[sizeof("vbTab")];
      char stringpool_str71[sizeof("vbLong")];
      char stringpool_str72[sizeof("vbAbort")];
      char stringpool_str73[sizeof("vbFriday")];
      char stringpool_str76[sizeof("vbHide")];
      char stringpool_str78[sizeof("vbLinguisticCasing")];
      char stringpool_str81[sizeof("vbWide")];
      char stringpool_str83[sizeof("vbAbortRetryIgnore")];
      char stringpool_str84[sizeof("vbTuesday")];
      char stringpool_str85[sizeof("vbGet")];
      char stringpool_str86[sizeof("vbDate")];
      char stringpool_str88[sizeof("vbDouble")];
      char stringpool_str90[sizeof("vbHiragana")];
      char stringpool_str91[sizeof("vbWednesday")];
      char stringpool_str94[sizeof("vbArchive")];
      char stringpool_str95[sizeof("vbSaturday")];
      char stringpool_str97[sizeof("vbProperCase")];
      char stringpool_str98[sizeof("vbHidden")];
      char stringpool_str99[sizeof("vbDecimal")];
      char stringpool_str101[sizeof("vbDefaultButton1")];
      char stringpool_str102[sizeof("vbUseDefault")];
      char stringpool_str103[sizeof("vbObject")];
      char stringpool_str104[sizeof("vbVariant")];
      char stringpool_str106[sizeof("vbDefaultButton3")];
      char stringpool_str108[sizeof("vbObjectError")];
      char stringpool_str110[sizeof("vbTraditionalChinese")];
      char stringpool_str111[sizeof("vbShortTime")];
      char stringpool_str115[sizeof("vbFormFeed")];
      char stringpool_str116[sizeof("vbLowerCase")];
      char stringpool_str118[sizeof("vbYesNoCancel")];
      char stringpool_str121[sizeof("vbDirectory")];
      char stringpool_str123[sizeof("vbCancel")];
      char stringpool_str126[sizeof("vbDefaultButton2")];
      char stringpool_str128[sizeof("vbOKOnly")];
      char stringpool_str130[sizeof("vbKatakana")];
      char stringpool_str133[sizeof("vbExclamation")];
      char stringpool_str137[sizeof("vbNullString")];
      char stringpool_str138[sizeof("vbString")];
      char stringpool_str143[sizeof("vbRetryCancel")];
      char stringpool_str146[sizeof("vbBack")];
      char stringpool_str147[sizeof("vbMsgBoxHelp")];
      char stringpool_str148[sizeof("vbApplicationModal")];
      char stringpool_str150[sizeof("vbCurrency")];
      char stringpool_str151[sizeof("vbCrLf")];
      char stringpool_str153[sizeof("vbMsgBoxRight")];
      char stringpool_str154[sizeof("vbCr")];
      char stringpool_str155[sizeof("vbNullChar")];
      char stringpool_str156[sizeof("vbShortDate")];
      char stringpool_str157[sizeof("Nothing")];
      char stringpool_str158[sizeof("vbMethod")];
      char stringpool_str160[sizeof("vbUseSystemDayOfWeek")];
      char stringpool_str161[sizeof("vbUpperCase")];
      char stringpool_str170[sizeof("vbFirstFullWeek")];
      char stringpool_str173[sizeof("vbNarrow")];
      char stringpool_str178[sizeof("vbMsgBoxRtlReading")];
      char stringpool_str183[sizeof("vbVerticalTab")];
      char stringpool_str190[sizeof("vbThursday")];
      char stringpool_str206[sizeof("vbMsgBoxSetForeground")];
      char stringpool_str219[sizeof("vbOK")];
      char stringpool_str220[sizeof("vbOKCancel")];
      char stringpool_str230[sizeof("vbCritical")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "vbSingle",
      "vbYes",
      "vbEmpty",
      "vbSystem",
      "False",
      "vbMinimizedFocus",
      "vbYesNo",
      "vbMinimizedNoFocus",
      "vbSimplifiedChinese",
      "vbMaximizedFocus",
      "vbMonday",
      "vbNo",
      "vbLongTime",
      "vbSystemModal",
      "vbNewLine",
      "vbLongDate",
      "vbInformation",
      "vbSet",
      "vbFirstJan1",
      "vbFalse",
      "vbNormalFocus",
      "True",
      "vbNormalNoFocus",
      "vbByte",
      "vbRetry",
      "vbNormal",
      "vbLet",
      "vbUseSystem",
      "vbArray",
      "vbTextCompare",
      "vbInteger",
      "vbBinaryCompare",
      "vbUserDefinedType",
      "vbSunday",
      "vbReadOnly",
      "vbIgnore",
      "vbBoolean",
      "vbQuestion",
      "vbNull",
      "vbVolume",
      "vbLf",
      "vbFirstFourDays",
      "vbTrue",
      "vbGeneralDate",
      "vbTab",
      "vbLong",
      "vbAbort",
      "vbFriday",
      "vbHide",
      "vbLinguisticCasing",
      "vbWide",
      "vbAbortRetryIgnore",
      "vbTuesday",
      "vbGet",
      "vbDate",
      "vbDouble",
      "vbHiragana",
      "vbWednesday",
      "vbArchive",
      "vbSaturday",
      "vbProperCase",
      "vbHidden",
      "vbDecimal",
      "vbDefaultButton1",
      "vbUseDefault",
      "vbObject",
      "vbVariant",
      "vbDefaultButton3",
      "vbObjectError",
      "vbTraditionalChinese",
      "vbShortTime",
      "vbFormFeed",
      "vbLowerCase",
      "vbYesNoCancel",
      "vbDirectory",
      "vbCancel",
      "vbDefaultButton2",
      "vbOKOnly",
      "vbKatakana",
      "vbExclamation",
      "vbNullString",
      "vbString",
      "vbRetryCancel",
      "vbBack",
      "vbMsgBoxHelp",
      "vbApplicationModal",
      "vbCurrency",
      "vbCrLf",
      "vbMsgBoxRight",
      "vbCr",
      "vbNullChar",
      "vbShortDate",
      "Nothing",
      "vbMethod",
      "vbUseSystemDayOfWeek",
      "vbUpperCase",
      "vbFirstFullWeek",
      "vbNarrow",
      "vbMsgBoxRtlReading",
      "vbVerticalTab",
      "vbThursday",
      "vbMsgBoxSetForeground",
      "vbOK",
      "vbOKCancel",
      "vbCritical"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str21,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str157,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str220,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str230
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

              if ((((unsigned char)*str ^ (unsigned char)*s) & ~32) == 0 && !gperf_case_strncmp (str, s, len) && s[len] == '\0')
                return s;
            }
        }
    }
  return 0;
}
