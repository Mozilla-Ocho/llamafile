/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_basic.c llamafile/is_keyword_basic.gperf  */
/* Computed positions: -k'1-3,5' */

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

#line 1 "llamafile/is_keyword_basic.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 145
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 15
#define MIN_HASH_VALUE 2
#define MAX_HASH_VALUE 415
/* maximum key range = 414, duplicates = 0 */

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
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416,  45,  95, 120,  35,   5,
       45,  90,  65,   5,   0,   0, 175,  35,  10,   0,
       70, 416,   0,  25,   5,  50,  45, 210,   5, 125,
        0, 416, 416, 416, 416, 416, 416,  45,  95, 120,
       35,   5,  45,  90,  65,   5,   0,   0, 175,  35,
       10,   0,  70, 416,   0,  25,   5,  50,  45, 210,
        5, 125,   0, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
      416, 416, 416, 416, 416, 416, 416
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[4]+1];
      /*FALLTHROUGH*/
      case 4:
      case 3:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
        hval += asso_values[(unsigned char)str[1]];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval;
}

const char *
is_keyword_basic (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str2[sizeof("Or")];
      char stringpool_str7[sizeof("To")];
      char stringpool_str8[sizeof("Xor")];
      char stringpool_str12[sizeof("On")];
      char stringpool_str14[sizeof("Trim")];
      char stringpool_str16[sizeof("OrElse")];
      char stringpool_str17[sizeof("In")];
      char stringpool_str18[sizeof("Not")];
      char stringpool_str19[sizeof("Exit")];
      char stringpool_str20[sizeof("RTrim")];
      char stringpool_str24[sizeof("Next")];
      char stringpool_str29[sizeof("NotInheritable")];
      char stringpool_str32[sizeof("Is")];
      char stringpool_str34[sizeof("Stop")];
      char stringpool_str35[sizeof("Error")];
      char stringpool_str37[sizeof("Do")];
      char stringpool_str38[sizeof("Set")];
      char stringpool_str39[sizeof("Step")];
      char stringpool_str41[sizeof("Return")];
      char stringpool_str42[sizeof("Me")];
      char stringpool_str43[sizeof("REM")];
      char stringpool_str44[sizeof("MIRR")];
      char stringpool_str46[sizeof("Resume")];
      char stringpool_str47[sizeof("Of")];
      char stringpool_str48[sizeof("For")];
      char stringpool_str52[sizeof("If")];
      char stringpool_str53[sizeof("End")];
      char stringpool_str54[sizeof("Interface")];
      char stringpool_str55[sizeof("ReDim")];
      char stringpool_str56[sizeof("Friend")];
      char stringpool_str58[sizeof("Out")];
      char stringpool_str64[sizeof("Attribute")];
      char stringpool_str69[sizeof("Enum")];
      char stringpool_str72[sizeof("As")];
      char stringpool_str73[sizeof("Mod")];
      char stringpool_str74[sizeof("Structure")];
      char stringpool_str78[sizeof("Dim")];
      char stringpool_str79[sizeof("Then")];
      char stringpool_str80[sizeof("Throw")];
      char stringpool_str81[sizeof("Static")];
      char stringpool_str84[sizeof("Overrides")];
      char stringpool_str85[sizeof("DirectCast")];
      char stringpool_str86[sizeof("Overridable")];
      char stringpool_str89[sizeof("NPer")];
      char stringpool_str93[sizeof("And")];
      char stringpool_str94[sizeof("Overloads")];
      char stringpool_str95[sizeof("IsNot")];
      char stringpool_str99[sizeof("GoTo")];
      char stringpool_str100[sizeof("Erase")];
      char stringpool_str102[sizeof("Finally")];
      char stringpool_str103[sizeof("Get")];
      char stringpool_str104[sizeof("Namespace")];
      char stringpool_str105[sizeof("RaiseEvent")];
      char stringpool_str107[sizeof("GetType")];
      char stringpool_str110[sizeof("Event")];
      char stringpool_str111[sizeof("Module")];
      char stringpool_str113[sizeof("Inherits")];
      char stringpool_str121[sizeof("MustInherit")];
      char stringpool_str122[sizeof("Partial")];
      char stringpool_str123[sizeof("Property")];
      char stringpool_str124[sizeof("Protected")];
      char stringpool_str125[sizeof("GetXMLNamespace")];
      char stringpool_str128[sizeof("ReadOnly")];
      char stringpool_str129[sizeof("Statement")];
      char stringpool_str132[sizeof("AndAlso")];
      char stringpool_str133[sizeof("Try")];
      char stringpool_str134[sizeof("Narrowing")];
      char stringpool_str135[sizeof("ParamArray")];
      char stringpool_str137[sizeof("Default")];
      char stringpool_str138[sizeof("Continue")];
      char stringpool_str139[sizeof("CInt")];
      char stringpool_str142[sizeof("Imports")];
      char stringpool_str145[sizeof("EndIf")];
      char stringpool_str150[sizeof("Using")];
      char stringpool_str151[sizeof("Option")];
      char stringpool_str153[sizeof("Optional")];
      char stringpool_str154[sizeof("CStr")];
      char stringpool_str159[sizeof("CSng")];
      char stringpool_str162[sizeof("Handles")];
      char stringpool_str163[sizeof("Function")];
      char stringpool_str164[sizeof("CDec")];
      char stringpool_str165[sizeof("Implements")];
      char stringpool_str166[sizeof("NameOf")];
      char stringpool_str169[sizeof("AddressOf")];
      char stringpool_str173[sizeof("Sub")];
      char stringpool_str174[sizeof("Each")];
      char stringpool_str177[sizeof("Private")];
      char stringpool_str178[sizeof("Operator")];
      char stringpool_str179[sizeof("Loop")];
      char stringpool_str180[sizeof("Catch")];
      char stringpool_str184[sizeof("Like")];
      char stringpool_str185[sizeof("Const")];
      char stringpool_str186[sizeof("Shared")];
      char stringpool_str188[sizeof("Let")];
      char stringpool_str190[sizeof("Constraint")];
      char stringpool_str192[sizeof("MustOverride")];
      char stringpool_str194[sizeof("Case")];
      char stringpool_str195[sizeof("LTrim")];
      char stringpool_str203[sizeof("SyncLock")];
      char stringpool_str204[sizeof("Type")];
      char stringpool_str209[sizeof("Else")];
      char stringpool_str210[sizeof("MacID")];
      char stringpool_str211[sizeof("ElseIf")];
      char stringpool_str212[sizeof("Shadows")];
      char stringpool_str219[sizeof("CObj")];
      char stringpool_str220[sizeof("AddHandler")];
      char stringpool_str221[sizeof("Public")];
      char stringpool_str224[sizeof("With")];
      char stringpool_str228[sizeof("New")];
      char stringpool_str229[sizeof("Wend")];
      char stringpool_str230[sizeof("CUInt")];
      char stringpool_str232[sizeof("TryCast")];
      char stringpool_str235[sizeof("Alias")];
      char stringpool_str239[sizeof("NotOverridable")];
      char stringpool_str240[sizeof("GoSub")];
      char stringpool_str241[sizeof("CShort")];
      char stringpool_str244[sizeof("MacScript")];
      char stringpool_str246[sizeof("Select")];
      char stringpool_str250[sizeof("CDate")];
      char stringpool_str254[sizeof("CDbl")];
      char stringpool_str255[sizeof("CBool")];
      char stringpool_str258[sizeof("Widening")];
      char stringpool_str262[sizeof("Declare")];
      char stringpool_str263[sizeof("RemoveHandler")];
      char stringpool_str266[sizeof("MyBase")];
      char stringpool_str269[sizeof("WriteOnly")];
      char stringpool_str272[sizeof("CUShort")];
      char stringpool_str275[sizeof("WithEvents")];
      char stringpool_str276[sizeof("TypeOf")];
      char stringpool_str278[sizeof("Lib")];
      char stringpool_str284[sizeof("When")];
      char stringpool_str288[sizeof("Delegate")];
      char stringpool_str296[sizeof("CSByte")];
      char stringpool_str300[sizeof("CType")];
      char stringpool_str305[sizeof("ByVal")];
      char stringpool_str309[sizeof("CLng")];
      char stringpool_str315[sizeof("ByRef")];
      char stringpool_str330[sizeof("While")];
      char stringpool_str335[sizeof("CChar")];
      char stringpool_str344[sizeof("Call")];
      char stringpool_str350[sizeof("Class")];
      char stringpool_str366[sizeof("Global")];
      char stringpool_str382[sizeof("MyClass")];
      char stringpool_str390[sizeof("CByte")];
      char stringpool_str415[sizeof("CULng")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "Or",
      "To",
      "Xor",
      "On",
      "Trim",
      "OrElse",
      "In",
      "Not",
      "Exit",
      "RTrim",
      "Next",
      "NotInheritable",
      "Is",
      "Stop",
      "Error",
      "Do",
      "Set",
      "Step",
      "Return",
      "Me",
      "REM",
      "MIRR",
      "Resume",
      "Of",
      "For",
      "If",
      "End",
      "Interface",
      "ReDim",
      "Friend",
      "Out",
      "Attribute",
      "Enum",
      "As",
      "Mod",
      "Structure",
      "Dim",
      "Then",
      "Throw",
      "Static",
      "Overrides",
      "DirectCast",
      "Overridable",
      "NPer",
      "And",
      "Overloads",
      "IsNot",
      "GoTo",
      "Erase",
      "Finally",
      "Get",
      "Namespace",
      "RaiseEvent",
      "GetType",
      "Event",
      "Module",
      "Inherits",
      "MustInherit",
      "Partial",
      "Property",
      "Protected",
      "GetXMLNamespace",
      "ReadOnly",
      "Statement",
      "AndAlso",
      "Try",
      "Narrowing",
      "ParamArray",
      "Default",
      "Continue",
      "CInt",
      "Imports",
      "EndIf",
      "Using",
      "Option",
      "Optional",
      "CStr",
      "CSng",
      "Handles",
      "Function",
      "CDec",
      "Implements",
      "NameOf",
      "AddressOf",
      "Sub",
      "Each",
      "Private",
      "Operator",
      "Loop",
      "Catch",
      "Like",
      "Const",
      "Shared",
      "Let",
      "Constraint",
      "MustOverride",
      "Case",
      "LTrim",
      "SyncLock",
      "Type",
      "Else",
      "MacID",
      "ElseIf",
      "Shadows",
      "CObj",
      "AddHandler",
      "Public",
      "With",
      "New",
      "Wend",
      "CUInt",
      "TryCast",
      "Alias",
      "NotOverridable",
      "GoSub",
      "CShort",
      "MacScript",
      "Select",
      "CDate",
      "CDbl",
      "CBool",
      "Widening",
      "Declare",
      "RemoveHandler",
      "MyBase",
      "WriteOnly",
      "CUShort",
      "WithEvents",
      "TypeOf",
      "Lib",
      "When",
      "Delegate",
      "CSByte",
      "CType",
      "ByVal",
      "CLng",
      "ByRef",
      "While",
      "CChar",
      "Call",
      "Class",
      "Global",
      "MyClass",
      "CByte",
      "CULng"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str8,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str20,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str24,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str69,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str73,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str74,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str93,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str99,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str105,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str110,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str123,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str177,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str184,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str185,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str192,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str194,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str195,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str203,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str204,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str209,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str210,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str211,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str212,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str220,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str221,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str224,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str228,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str230,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str232,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str235,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str239,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str240,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str241,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str244,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str246,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str250,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str254,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str255,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str258,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str262,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str263,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str266,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str269,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str272,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str275,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str276,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str278,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str284,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str288,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str296,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str300,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str305,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str309,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str315,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str330,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str335,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str344,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str350,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str366,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str382,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str390,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str415
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
