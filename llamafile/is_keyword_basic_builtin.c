/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_basic_builtin.c llamafile/is_keyword_basic_builtin.gperf  */
/* Computed positions: -k'1-2,5,$' */

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

#line 1 "llamafile/is_keyword_basic_builtin.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 113
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 13
#define MIN_HASH_VALUE 10
#define MAX_HASH_VALUE 325
/* maximum key range = 316, duplicates = 0 */

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
  static const unsigned short asso_values[] =
    {
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326,  65,  50,  80,  15,  25,
       10,  32,  95,  50,   0,  70,  90,  22,  25, 100,
       10,  70,   5,  15,   0,  40,  60,  15,  12, 120,
      326, 326, 326, 326, 326, 326, 326,  65,  50,  80,
       15,  25,  10,  32,  95,  50,   0,  70,  90,  22,
       25, 100,  10,  70,   5,  15,   0,  40,  60,  15,
       12, 120, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326, 326, 326, 326,
      326, 326, 326, 326, 326, 326, 326
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
      case 3:
      case 2:
        hval += asso_values[(unsigned char)str[1]+1];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_basic_builtin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str10[sizeof("Right")];
      char stringpool_str15[sizeof("Timer")];
      char stringpool_str20[sizeof("Reset")];
      char stringpool_str23[sizeof("Dir")];
      char stringpool_str24[sizeof("Sqrt")];
      char stringpool_str25[sizeof("Fix")];
      char stringpool_str27[sizeof("FilePut")];
      char stringpool_str28[sizeof("Sqr")];
      char stringpool_str29[sizeof("Time")];
      char stringpool_str30[sizeof("Print")];
      char stringpool_str33[sizeof("FilePutObject")];
      char stringpool_str34[sizeof("Read")];
      char stringpool_str37[sizeof("SetAttr")];
      char stringpool_str38[sizeof("Pmt")];
      char stringpool_str39[sizeof("Join")];
      char stringpool_str40[sizeof("Mid")];
      char stringpool_str43[sizeof("Sin")];
      char stringpool_str44[sizeof("Sign")];
      char stringpool_str48[sizeof("EOF")];
      char stringpool_str49[sizeof("FileGet")];
      char stringpool_str50[sizeof("Round")];
      char stringpool_str53[sizeof("Now")];
      char stringpool_str54[sizeof("GetAttr")];
      char stringpool_str55[sizeof("FileGetObject")];
      char stringpool_str58[sizeof("IsObject")];
      char stringpool_str62[sizeof("FileDateTime")];
      char stringpool_str63[sizeof("Str")];
      char stringpool_str67[sizeof("IsError")];
      char stringpool_str68[sizeof("FreeFile")];
      char stringpool_str71[sizeof("Second")];
      char stringpool_str73[sizeof("DoEvents")];
      char stringpool_str76[sizeof("WeekdayName")];
      char stringpool_str78[sizeof("Tan")];
      char stringpool_str81[sizeof("IsDate")];
      char stringpool_str83[sizeof("DatePart")];
      char stringpool_str84[sizeof("PPmt")];
      char stringpool_str85[sizeof("Write")];
      char stringpool_str87[sizeof("PV")];
      char stringpool_str88[sizeof("FileAttr")];
      char stringpool_str89[sizeof("WriteLine")];
      char stringpool_str90[sizeof("Split")];
      char stringpool_str91[sizeof("Format")];
      char stringpool_str93[sizeof("DDB")];
      char stringpool_str94[sizeof("Date")];
      char stringpool_str98[sizeof("DateDiff")];
      char stringpool_str99[sizeof("Seek")];
      char stringpool_str101[sizeof("GetObject")];
      char stringpool_str103[sizeof("TAB")];
      char stringpool_str104[sizeof("Left")];
      char stringpool_str106[sizeof("IsMissing")];
      char stringpool_str108[sizeof("Cos")];
      char stringpool_str110[sizeof("UCase")];
      char stringpool_str112[sizeof("Replace")];
      char stringpool_str113[sizeof("LOF")];
      char stringpool_str115[sizeof("StrReverse")];
      char stringpool_str118[sizeof("Oct")];
      char stringpool_str120[sizeof("Hex")];
      char stringpool_str123[sizeof("Rnd")];
      char stringpool_str124[sizeof("FileClose")];
      char stringpool_str128[sizeof("Len")];
      char stringpool_str129[sizeof("FileWidth")];
      char stringpool_str132[sizeof("FileLen")];
      char stringpool_str133[sizeof("Atn")];
      char stringpool_str134[sizeof("Atan")];
      char stringpool_str135[sizeof("Log")];
      char stringpool_str138[sizeof("Chr")];
      char stringpool_str139[sizeof("Year")];
      char stringpool_str140[sizeof("MsgBox")];
      char stringpool_str143[sizeof("FileOpen")];
      char stringpool_str144[sizeof("Partition")];
      char stringpool_str148[sizeof("Asc")];
      char stringpool_str149[sizeof("LineInput")];
      char stringpool_str152[sizeof("DateAdd")];
      char stringpool_str153[sizeof("Int")];
      char stringpool_str155[sizeof("Input")];
      char stringpool_str158[sizeof("Exp")];
      char stringpool_str159[sizeof("DateValue")];
      char stringpool_str160[sizeof("LCase")];
      char stringpool_str161[sizeof("MonthName")];
      char stringpool_str162[sizeof("Environ")];
      char stringpool_str163[sizeof("Abs")];
      char stringpool_str164[sizeof("Kill")];
      char stringpool_str165[sizeof("InStr")];
      char stringpool_str166[sizeof("UBound")];
      char stringpool_str167[sizeof("Weekday")];
      char stringpool_str168[sizeof("SPC")];
      char stringpool_str170[sizeof("InputBox")];
      char stringpool_str172[sizeof("StrComp")];
      char stringpool_str174[sizeof("Lock")];
      char stringpool_str176[sizeof("Choose")];
      char stringpool_str177[sizeof("Command")];
      char stringpool_str180[sizeof("DateSerial")];
      char stringpool_str182[sizeof("IsArray")];
      char stringpool_str183[sizeof("Loc")];
      char stringpool_str187[sizeof("IsEmpty")];
      char stringpool_str188[sizeof("Day")];
      char stringpool_str189[sizeof("Randomize")];
      char stringpool_str193[sizeof("InputString")];
      char stringpool_str201[sizeof("CurDir")];
      char stringpool_str203[sizeof("Val")];
      char stringpool_str208[sizeof("Switch")];
      char stringpool_str215[sizeof("CallByName")];
      char stringpool_str216[sizeof("LBound")];
      char stringpool_str218[sizeof("FileCopy")];
      char stringpool_str222[sizeof("StrConv")];
      char stringpool_str223[sizeof("InStrRev")];
      char stringpool_str227[sizeof("Month")];
      char stringpool_str236[sizeof("IsNull")];
      char stringpool_str250[sizeof("Shell")];
      char stringpool_str252[sizeof("QBColor")];
      char stringpool_str262[sizeof("VarType")];
      char stringpool_str296[sizeof("Unlock")];
      char stringpool_str325[sizeof("Array")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "Right",
      "Timer",
      "Reset",
      "Dir",
      "Sqrt",
      "Fix",
      "FilePut",
      "Sqr",
      "Time",
      "Print",
      "FilePutObject",
      "Read",
      "SetAttr",
      "Pmt",
      "Join",
      "Mid",
      "Sin",
      "Sign",
      "EOF",
      "FileGet",
      "Round",
      "Now",
      "GetAttr",
      "FileGetObject",
      "IsObject",
      "FileDateTime",
      "Str",
      "IsError",
      "FreeFile",
      "Second",
      "DoEvents",
      "WeekdayName",
      "Tan",
      "IsDate",
      "DatePart",
      "PPmt",
      "Write",
      "PV",
      "FileAttr",
      "WriteLine",
      "Split",
      "Format",
      "DDB",
      "Date",
      "DateDiff",
      "Seek",
      "GetObject",
      "TAB",
      "Left",
      "IsMissing",
      "Cos",
      "UCase",
      "Replace",
      "LOF",
      "StrReverse",
      "Oct",
      "Hex",
      "Rnd",
      "FileClose",
      "Len",
      "FileWidth",
      "FileLen",
      "Atn",
      "Atan",
      "Log",
      "Chr",
      "Year",
      "MsgBox",
      "FileOpen",
      "Partition",
      "Asc",
      "LineInput",
      "DateAdd",
      "Int",
      "Input",
      "Exp",
      "DateValue",
      "LCase",
      "MonthName",
      "Environ",
      "Abs",
      "Kill",
      "InStr",
      "UBound",
      "Weekday",
      "SPC",
      "InputBox",
      "StrComp",
      "Lock",
      "Choose",
      "Command",
      "DateSerial",
      "IsArray",
      "Loc",
      "IsEmpty",
      "Day",
      "Randomize",
      "InputString",
      "CurDir",
      "Val",
      "Switch",
      "CallByName",
      "LBound",
      "FileCopy",
      "StrConv",
      "InStrRev",
      "Month",
      "IsNull",
      "Shell",
      "QBColor",
      "VarType",
      "Unlock",
      "Array"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str25,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str28,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str33,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str63,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str67,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1,
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
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str90,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str177,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str182,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str187,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str189,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str201,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str203,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str208,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str215,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str222,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str223,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str227,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str236,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str250,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str252,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str262,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str296,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str325
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
