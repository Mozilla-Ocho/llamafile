/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_cobol.gperf  */
/* Computed positions: -k'1-3,6-7,12,$' */

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

#line 1 "llamafile/is_keyword_cobol.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 418
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 19
#define MIN_HASH_VALUE 27
#define MAX_HASH_VALUE 1817
/* maximum key range = 1791, duplicates = 0 */

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
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818,    0, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818,  423, 1818, 1818, 1818,    0,
        30,   15,   35,   10, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818,   50,  455,  105,   15,    5,
       443,  245,  191,  130,    0,    0,   70,  395,    5,  155,
       335,  230,   20,    0,   15,  295,  438,  165,   25,  508,
        70, 1818, 1818, 1818, 1818,    5, 1818,   50,  455,  105,
        15,    5,  443,  245,  191,  130,    0,    0,   70,  395,
         5,  155,  335,  230,   20,    0,   15,  295,  438,  165,
        25,  508,   70, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818, 1818,
      1818, 1818, 1818, 1818, 1818, 1818
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[11]];
      /*FALLTHROUGH*/
      case 11:
      case 10:
      case 9:
      case 8:
      case 7:
        hval += asso_values[(unsigned char)str[6]];
      /*FALLTHROUGH*/
      case 6:
        hval += asso_values[(unsigned char)str[5]];
      /*FALLTHROUGH*/
      case 5:
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
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_cobol (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str27[sizeof("DE")];
      char stringpool_str29[sizeof("SEND")];
      char stringpool_str30[sizeof("EJECT")];
      char stringpool_str32[sizeof("SD")];
      char stringpool_str38[sizeof("SET")];
      char stringpool_str39[sizeof("TEST")];
      char stringpool_str42[sizeof("RENAMES")];
      char stringpool_str43[sizeof("END")];
      char stringpool_str45[sizeof("RESET")];
      char stringpool_str50[sizeof("ENTER")];
      char stringpool_str52[sizeof("RD")];
      char stringpool_str54[sizeof("NEXT")];
      char stringpool_str55[sizeof("RERUN")];
      char stringpool_str56[sizeof("RETURN")];
      char stringpool_str60[sizeof("END-RETURN")];
      char stringpool_str64[sizeof("TEXT")];
      char stringpool_str66[sizeof("ASSIGN")];
      char stringpool_str70[sizeof("ERROR")];
      char stringpool_str71[sizeof("STATUS")];
      char stringpool_str76[sizeof("ENABLE")];
      char stringpool_str77[sizeof("END-ADD")];
      char stringpool_str79[sizeof("LESS")];
      char stringpool_str80[sizeof("AREAS")];
      char stringpool_str81[sizeof("EXTEND")];
      char stringpool_str82[sizeof("AT")];
      char stringpool_str83[sizeof("ARE")];
      char stringpool_str84[sizeof("ELSE")];
      char stringpool_str85[sizeof("START")];
      char stringpool_str86[sizeof("NATIVE")];
      char stringpool_str87[sizeof("ADDRESS")];
      char stringpool_str88[sizeof("AND")];
      char stringpool_str91[sizeof("DESTINATION")];
      char stringpool_str94[sizeof("READ")];
      char stringpool_str95[sizeof("TRACE")];
      char stringpool_str98[sizeof("ADD")];
      char stringpool_str100[sizeof("ZEROS")];
      char stringpool_str101[sizeof("ZEROES")];
      char stringpool_str103[sizeof("END-READ")];
      char stringpool_str104[sizeof("REEL")];
      char stringpool_str106[sizeof("DELETE")];
      char stringpool_str109[sizeof("TERMINATE")];
      char stringpool_str111[sizeof("SELECT")];
      char stringpool_str112[sizeof("RELEASE")];
      char stringpool_str114[sizeof("END-START")];
      char stringpool_str115[sizeof("END-DELETE")];
      char stringpool_str129[sizeof("AREA")];
      char stringpool_str131[sizeof("RELOAD")];
      char stringpool_str132[sizeof("IS")];
      char stringpool_str133[sizeof("SENTENCE")];
      char stringpool_str134[sizeof("DATA")];
      char stringpool_str135[sizeof("SKIP1")];
      char stringpool_str137[sizeof("CD")];
      char stringpool_str139[sizeof("LAST")];
      char stringpool_str142[sizeof("IN")];
      char stringpool_str145[sizeof("STANDARD-1")];
      char stringpool_str147[sizeof("SERVICE")];
      char stringpool_str150[sizeof("SKIP3")];
      char stringpool_str151[sizeof("END-RECEIVE")];
      char stringpool_str152[sizeof("RECORDS")];
      char stringpool_str158[sizeof("STANDARD")];
      char stringpool_str160[sizeof("ALTER")];
      char stringpool_str162[sizeof("ID")];
      char stringpool_str165[sizeof("SKIP2")];
      char stringpool_str166[sizeof("RECORD")];
      char stringpool_str167[sizeof("ON")];
      char stringpool_str170[sizeof("TITLE")];
      char stringpool_str171[sizeof("INSERT")];
      char stringpool_str173[sizeof("TERMINAL")];
      char stringpool_str175[sizeof("STANDARD-2")];
      char stringpool_str178[sizeof("EXTERNAL")];
      char stringpool_str179[sizeof("EXIT")];
      char stringpool_str180[sizeof("INDEX")];
      char stringpool_str181[sizeof("DETAIL")];
      char stringpool_str184[sizeof("REDEFINES")];
      char stringpool_str189[sizeof("END-WRITE")];
      char stringpool_str190[sizeof("KANJI")];
      char stringpool_str192[sizeof("INDEXED")];
      char stringpool_str193[sizeof("NOT")];
      char stringpool_str194[sizeof("SORT")];
      char stringpool_str197[sizeof("OR")];
      char stringpool_str204[sizeof("ALTERNATE")];
      char stringpool_str207[sizeof("DECLARATIVES")];
      char stringpool_str209[sizeof("SIZE")];
      char stringpool_str210[sizeof("LINES")];
      char stringpool_str211[sizeof("END-REWRITE")];
      char stringpool_str214[sizeof("LINE")];
      char stringpool_str215[sizeof("ORDER")];
      char stringpool_str216[sizeof("SORT-RETURN")];
      char stringpool_str220[sizeof("THEN")];
      char stringpool_str221[sizeof("LINAGE")];
      char stringpool_str222[sizeof("REWRITE")];
      char stringpool_str223[sizeof("END-CALL")];
      char stringpool_str226[sizeof("REWIND")];
      char stringpool_str228[sizeof("INDICATE")];
      char stringpool_str230[sizeof("CLASS")];
      char stringpool_str254[sizeof("ZERO")];
      char stringpool_str259[sizeof("DELIMITED")];
      char stringpool_str263[sizeof("ALL")];
      char stringpool_str264[sizeof("DELIMITER")];
      char stringpool_str265[sizeof("THAN")];
      char stringpool_str266[sizeof("ACCESS")];
      char stringpool_str268[sizeof("ESI")];
      char stringpool_str274[sizeof("RECURSIVE")];
      char stringpool_str277[sizeof("INSPECT")];
      char stringpool_str278[sizeof("DECIMAL-POINT")];
      char stringpool_str279[sizeof("ALSO")];
      char stringpool_str281[sizeof("END-SEARCH")];
      char stringpool_str282[sizeof("SECTION")];
      char stringpool_str284[sizeof("CODE")];
      char stringpool_str287[sizeof("DATE-WRITTEN")];
      char stringpool_str292[sizeof("SEGMENT")];
      char stringpool_str294[sizeof("EXCEPTION")];
      char stringpool_str295[sizeof("DESCENDING")];
      char stringpool_str296[sizeof("ACCEPT")];
      char stringpool_str297[sizeof("INSTALLATION")];
      char stringpool_str299[sizeof("CALL")];
      char stringpool_str303[sizeof("CODE-SET")];
      char stringpool_str304[sizeof("CORR")];
      char stringpool_str306[sizeof("CANCEL")];
      char stringpool_str307[sizeof("CONTENT")];
      char stringpool_str308[sizeof("USE")];
      char stringpool_str309[sizeof("INTO")];
      char stringpool_str314[sizeof("JUST")];
      char stringpool_str315[sizeof("END-STRING")];
      char stringpool_str317[sizeof("NO")];
      char stringpool_str322[sizeof("GREATER")];
      char stringpool_str325[sizeof("WRITE")];
      char stringpool_str327[sizeof("TO")];
      char stringpool_str328[sizeof("RUN")];
      char stringpool_str333[sizeof("GENERATE")];
      char stringpool_str334[sizeof("LOCK")];
      char stringpool_str335[sizeof("SEQUENTIAL")];
      char stringpool_str339[sizeof("TRUE")];
      char stringpool_str340[sizeof("CLOSE")];
      char stringpool_str343[sizeof("INITIATE")];
      char stringpool_str344[sizeof("DOWN")];
      char stringpool_str345[sizeof("WORDS")];
      char stringpool_str346[sizeof("COLUMN")];
      char stringpool_str355[sizeof("USAGE")];
      char stringpool_str358[sizeof("SEQUENCE")];
      char stringpool_str359[sizeof("EGCS")];
      char stringpool_str370[sizeof("WHEN")];
      char stringpool_str371[sizeof("OCCURS")];
      char stringpool_str373[sizeof("DATE-COMPILED")];
      char stringpool_str375[sizeof("NULLS")];
      char stringpool_str382[sizeof("REPORTS")];
      char stringpool_str384[sizeof("SIGN")];
      char stringpool_str386[sizeof("OTHER")];
      char stringpool_str390[sizeof("UNTIL")];
      char stringpool_str391[sizeof("SPACES")];
      char stringpool_str395[sizeof("SPACE")];
      char stringpool_str396[sizeof("REPORT")];
      char stringpool_str400[sizeof("INITIALIZE")];
      char stringpool_str404[sizeof("RH")];
      char stringpool_str406[sizeof("AUTHOR")];
      char stringpool_str408[sizeof("CONTAINS")];
      char stringpool_str409[sizeof("TAPE")];
      char stringpool_str415[sizeof("RIGHT")];
      char stringpool_str418[sizeof("SEPARATE")];
      char stringpool_str427[sizeof("REMARKS")];
      char stringpool_str428[sizeof("SEGMENT-LIMIT")];
      char stringpool_str429[sizeof("RETURNING")];
      char stringpool_str430[sizeof("MERGE")];
      char stringpool_str443[sizeof("SEARCH")];
      char stringpool_str444[sizeof("NULL")];
      char stringpool_str449[sizeof("UNIT")];
      char stringpool_str451[sizeof("METHOD")];
      char stringpool_str454[sizeof("SAME")];
      char stringpool_str456[sizeof("END-PERFORM")];
      char stringpool_str462[sizeof("INITIAL")];
      char stringpool_str466[sizeof("SOURCE")];
      char stringpool_str467[sizeof("LINKAGE")];
      char stringpool_str468[sizeof("LENGTH")];
      char stringpool_str469[sizeof("REMAINDER")];
      char stringpool_str473[sizeof("TRAILING")];
      char stringpool_str475[sizeof("FD")];
      char stringpool_str476[sizeof("CHARACTERS")];
      char stringpool_str478[sizeof("SPECIAL-NAMES")];
      char stringpool_str479[sizeof("INHERITS")];
      char stringpool_str480[sizeof("RESERVE")];
      char stringpool_str482[sizeof("REPLACE")];
      char stringpool_str483[sizeof("END-INVOKE")];
      char stringpool_str484[sizeof("RETURN-CODE")];
      char stringpool_str488[sizeof("REFERENCES")];
      char stringpool_str489[sizeof("CH")];
      char stringpool_str490[sizeof("INPUT")];
      char stringpool_str491[sizeof("REVERSED")];
      char stringpool_str492[sizeof("REFERENCE")];
      char stringpool_str495[sizeof("CHARACTER")];
      char stringpool_str498[sizeof("CONTROLS")];
      char stringpool_str501[sizeof("OUTPUT")];
      char stringpool_str504[sizeof("OPEN")];
      char stringpool_str505[sizeof("WITH")];
      char stringpool_str509[sizeof("STOP")];
      char stringpool_str510[sizeof("BASIS")];
      char stringpool_str512[sizeof("ROUNDED")];
      char stringpool_str513[sizeof("EGI")];
      char stringpool_str517[sizeof("LINE-COUNTER")];
      char stringpool_str520[sizeof("LOW-VALUES")];
      char stringpool_str521[sizeof("ENDING")];
      char stringpool_str522[sizeof("SELF")];
      char stringpool_str523[sizeof("TALLYING")];
      char stringpool_str524[sizeof("LOW-VALUE")];
      char stringpool_str525[sizeof("THRU")];
      char stringpool_str529[sizeof("RECORDING")];
      char stringpool_str530[sizeof("TABLE")];
      char stringpool_str531[sizeof("STRING")];
      char stringpool_str532[sizeof("END-UNSTRING")];
      char stringpool_str533[sizeof("AFTER")];
      char stringpool_str535[sizeof("END-EVALUATE")];
      char stringpool_str537[sizeof("LEFT")];
      char stringpool_str538[sizeof("ENTRY")];
      char stringpool_str540[sizeof("QUEUE")];
      char stringpool_str544[sizeof("METACLASS")];
      char stringpool_str545[sizeof("TIMES")];
      char stringpool_str549[sizeof("TIME")];
      char stringpool_str552[sizeof("PROCEED")];
      char stringpool_str554[sizeof("ASCENDING")];
      char stringpool_str555[sizeof("CONVERTING")];
      char stringpool_str557[sizeof("GO")];
      char stringpool_str564[sizeof("VALUES")];
      char stringpool_str567[sizeof("CONTROL")];
      char stringpool_str568[sizeof("VALUE")];
      char stringpool_str571[sizeof("EVALUATE")];
      char stringpool_str573[sizeof("FALSE")];
      char stringpool_str574[sizeof("MODE")];
      char stringpool_str575[sizeof("COUNT")];
      char stringpool_str577[sizeof("MODULES")];
      char stringpool_str578[sizeof("CONTINUE")];
      char stringpool_str579[sizeof("DBCS")];
      char stringpool_str580[sizeof("BLANK")];
      char stringpool_str583[sizeof("PASSWORD")];
      char stringpool_str584[sizeof("SORT-CORE-SIZE")];
      char stringpool_str585[sizeof("RECEIVE")];
      char stringpool_str587[sizeof("SORT-CONTROL")];
      char stringpool_str588[sizeof("READY")];
      char stringpool_str589[sizeof("INVOKE")];
      char stringpool_str590[sizeof("SORT-MERGE")];
      char stringpool_str591[sizeof("END-COMPUTE")];
      char stringpool_str597[sizeof("SORT-MESSAGE")];
      char stringpool_str599[sizeof("DIVIDE")];
      char stringpool_str601[sizeof("LIMITS")];
      char stringpool_str605[sizeof("EQUAL")];
      char stringpool_str607[sizeof("PICTURE")];
      char stringpool_str608[sizeof("END-DIVIDE")];
      char stringpool_str613[sizeof("FIRST")];
      char stringpool_str615[sizeof("LIMIT")];
      char stringpool_str616[sizeof("GLOBAL")];
      char stringpool_str617[sizeof("REMOVAL")];
      char stringpool_str621[sizeof("NATIVE_BINARY")];
      char stringpool_str622[sizeof("SYNC")];
      char stringpool_str627[sizeof("LEADING")];
      char stringpool_str630[sizeof("WORKING-STORAGE")];
      char stringpool_str634[sizeof("ENVIRONMENT")];
      char stringpool_str638[sizeof("OPTIONAL")];
      char stringpool_str639[sizeof("PAGE")];
      char stringpool_str641[sizeof("FOR")];
      char stringpool_str642[sizeof("ORGANIZATION")];
      char stringpool_str643[sizeof("SUPPRESS")];
      char stringpool_str646[sizeof("OBJECT")];
      char stringpool_str648[sizeof("TALLY")];
      char stringpool_str650[sizeof("LABEL")];
      char stringpool_str652[sizeof("FILE")];
      char stringpool_str653[sizeof("FINAL")];
      char stringpool_str654[sizeof("WHEN-COMPILED")];
      char stringpool_str655[sizeof("SUPER")];
      char stringpool_str660[sizeof("PURGE")];
      char stringpool_str661[sizeof("COMP-1")];
      char stringpool_str662[sizeof("MESSAGE")];
      char stringpool_str663[sizeof("EMI")];
      char stringpool_str671[sizeof("COMMON")];
      char stringpool_str672[sizeof("POINTER")];
      char stringpool_str675[sizeof("USING")];
      char stringpool_str676[sizeof("RELATIVE")];
      char stringpool_str678[sizeof("PIC")];
      char stringpool_str681[sizeof("COMP-5")];
      char stringpool_str682[sizeof("LINAGE-COUNTER")];
      char stringpool_str685[sizeof("BLOCK")];
      char stringpool_str686[sizeof("QUOTES")];
      char stringpool_str687[sizeof("COMPUTE")];
      char stringpool_str688[sizeof("UNSTRING")];
      char stringpool_str689[sizeof("FILLER")];
      char stringpool_str690[sizeof("QUOTE")];
      char stringpool_str691[sizeof("COMP-3")];
      char stringpool_str701[sizeof("MORE-LABELS")];
      char stringpool_str703[sizeof("CBL")];
      char stringpool_str704[sizeof("PLUS")];
      char stringpool_str710[sizeof("COMMA")];
      char stringpool_str712[sizeof("DISPLAY-1")];
      char stringpool_str719[sizeof("PH")];
      char stringpool_str721[sizeof("COMP-2")];
      char stringpool_str722[sizeof("OMITTED")];
      char stringpool_str729[sizeof("COLLATING")];
      char stringpool_str730[sizeof("SYNCHRONIZED")];
      char stringpool_str731[sizeof("COMP-4")];
      char stringpool_str740[sizeof("INVALID")];
      char stringpool_str741[sizeof("NUMBER")];
      char stringpool_str748[sizeof("HEADING")];
      char stringpool_str754[sizeof("DEPENDING")];
      char stringpool_str756[sizeof("OVERRIDE")];
      char stringpool_str759[sizeof("REPORTING")];
      char stringpool_str760[sizeof("GROUP")];
      char stringpool_str765[sizeof("PROCESSING")];
      char stringpool_str771[sizeof("SECURITY")];
      char stringpool_str785[sizeof("COMPUTATIONAL-1")];
      char stringpool_str788[sizeof("POSITION")];
      char stringpool_str790[sizeof("COBOL")];
      char stringpool_str794[sizeof("UPON")];
      char stringpool_str795[sizeof("COMPUTATIONAL-5")];
      char stringpool_str800[sizeof("COMPUTATIONAL-3")];
      char stringpool_str801[sizeof("CLASS-ID")];
      char stringpool_str810[sizeof("DUPLICATES")];
      char stringpool_str815[sizeof("COMPUTATIONAL-2")];
      char stringpool_str817[sizeof("END-SUBTRACT")];
      char stringpool_str820[sizeof("COMPUTATIONAL-4")];
      char stringpool_str830[sizeof("PROCEDURES")];
      char stringpool_str833[sizeof("EOP")];
      char stringpool_str834[sizeof("PROCEDURE")];
      char stringpool_str836[sizeof("NEGATIVE")];
      char stringpool_str843[sizeof("TOP")];
      char stringpool_str849[sizeof("REPLACING")];
      char stringpool_str853[sizeof("COMPUTATIONAL")];
      char stringpool_str860[sizeof("THROUGH")];
      char stringpool_str861[sizeof("GOBACK")];
      char stringpool_str866[sizeof("I-O")];
      char stringpool_str867[sizeof("TYPE")];
      char stringpool_str871[sizeof("RANDOM")];
      char stringpool_str872[sizeof("IDENTIFICATION")];
      char stringpool_str873[sizeof("PRINTING")];
      char stringpool_str874[sizeof("SORT-MODE-SIZE")];
      char stringpool_str875[sizeof("DYNAMIC")];
      char stringpool_str877[sizeof("METHOD-ID")];
      char stringpool_str878[sizeof("CORRESPONDING")];
      char stringpool_str879[sizeof("JUSTIFIED\011 ")];
      char stringpool_str881[sizeof("DIVISION")];
      char stringpool_str887[sizeof("SHIFT-IN")];
      char stringpool_str894[sizeof("ALPHANUMERIC-EDITED")];
      char stringpool_str897[sizeof("SORT-FILE-SIZE")];
      char stringpool_str902[sizeof("PADDING")];
      char stringpool_str907[sizeof("END-OF-PAGE")];
      char stringpool_str908[sizeof("RF")];
      char stringpool_str909[sizeof("DEBUG-SUB-1")];
      char stringpool_str917[sizeof("END-IF")];
      char stringpool_str918[sizeof("DEBUG-NAME")];
      char stringpool_str919[sizeof("BEFORE")];
      char stringpool_str923[sizeof("SHIFT-OUT")];
      char stringpool_str924[sizeof("DEBUG-SUB-3")];
      char stringpool_str928[sizeof("SUBTRACT")];
      char stringpool_str938[sizeof("ALPHABET")];
      char stringpool_str939[sizeof("DEBUG-SUB-2")];
      char stringpool_str942[sizeof("PAGE-COUNTER")];
      char stringpool_str949[sizeof("I-O-CONTROL")];
      char stringpool_str961[sizeof("EVERY")];
      char stringpool_str963[sizeof("COMMUNICATION")];
      char stringpool_str967[sizeof("UP")];
      char stringpool_str974[sizeof("NUMERIC-EDITED")];
      char stringpool_str977[sizeof("ALPHANUMERIC")];
      char stringpool_str978[sizeof("CONFIGURATION")];
      char stringpool_str980[sizeof("PROGRAM-ID")];
      char stringpool_str983[sizeof("DEBUG-LINE")];
      char stringpool_str992[sizeof("ADVANCING")];
      char stringpool_str993[sizeof("CF")];
      char stringpool_str994[sizeof("COMP")];
      char stringpool_str996[sizeof("OVERFLOW")];
      char stringpool_str997[sizeof("MOVE")];
      char stringpool_str1010[sizeof("VARYING")];
      char stringpool_str1012[sizeof("PROCEDURE-POINTER")];
      char stringpool_str1016[sizeof("LOCAL-STORAGE")];
      char stringpool_str1017[sizeof("FROM")];
      char stringpool_str1018[sizeof("IF")];
      char stringpool_str1021[sizeof("ALPHABETIC-LOWER")];
      char stringpool_str1022[sizeof("DEBUG-CONTENTS")];
      char stringpool_str1023[sizeof("REPOSITORY")];
      char stringpool_str1024[sizeof("KEY")];
      char stringpool_str1030[sizeof("ALPHABETIC")];
      char stringpool_str1041[sizeof("FUNCTION")];
      char stringpool_str1042[sizeof("NUMERIC")];
      char stringpool_str1043[sizeof("OF")];
      char stringpool_str1046[sizeof("CURRENCY")];
      char stringpool_str1055[sizeof("FILE-CONTROL")];
      char stringpool_str1059[sizeof("CLOCK-UNITS")];
      char stringpool_str1061[sizeof("SUB-QUEUE-1")];
      char stringpool_str1065[sizeof("HIGH-VALUES")];
      char stringpool_str1069[sizeof("HIGH-VALUE")];
      char stringpool_str1071[sizeof("POSITIVE")];
      char stringpool_str1074[sizeof("ANY")];
      char stringpool_str1076[sizeof("SUB-QUEUE-3")];
      char stringpool_str1084[sizeof("DAY")];
      char stringpool_str1088[sizeof("SUM")];
      char stringpool_str1090[sizeof("INPUT-OUTPUT")];
      char stringpool_str1091[sizeof("SUB-QUEUE-2")];
      char stringpool_str1094[sizeof("BEGINNING")];
      char stringpool_str1104[sizeof("DEBUGGING")];
      char stringpool_str1107[sizeof("COPY")];
      char stringpool_str1157[sizeof("COM-REG")];
      char stringpool_str1177[sizeof("PERFORM")];
      char stringpool_str1178[sizeof("MULTIPLE")];
      char stringpool_str1208[sizeof("SOURCE-COMPUTER")];
      char stringpool_str1216[sizeof("SYMBOLIC")];
      char stringpool_str1218[sizeof("DISPLAY")];
      char stringpool_str1223[sizeof("PF")];
      char stringpool_str1233[sizeof("APPLY")];
      char stringpool_str1246[sizeof("ALPHABETIC-UPPER")];
      char stringpool_str1255[sizeof("FOOTING")];
      char stringpool_str1309[sizeof("GIVING")];
      char stringpool_str1357[sizeof("PROGRAM")];
      char stringpool_str1378[sizeof("OBJECT-COMPUTER")];
      char stringpool_str1407[sizeof("PACKED-DECIMAL")];
      char stringpool_str1411[sizeof("WRITE-ONLY")];
      char stringpool_str1418[sizeof("END-MULTIPLY")];
      char stringpool_str1421[sizeof("BOTTOM")];
      char stringpool_str1433[sizeof("DEBUG-ITEM")];
      char stringpool_str1450[sizeof("DAY-OF-WEEK")];
      char stringpool_str1473[sizeof("BY")];
      char stringpool_str1487[sizeof("OFF")];
      char stringpool_str1612[sizeof("BINARY")];
      char stringpool_str1681[sizeof("MULTIPLY")];
      char stringpool_str1817[sizeof("MEMORY")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "DE",
      "SEND",
      "EJECT",
      "SD",
      "SET",
      "TEST",
      "RENAMES",
      "END",
      "RESET",
      "ENTER",
      "RD",
      "NEXT",
      "RERUN",
      "RETURN",
      "END-RETURN",
      "TEXT",
      "ASSIGN",
      "ERROR",
      "STATUS",
      "ENABLE",
      "END-ADD",
      "LESS",
      "AREAS",
      "EXTEND",
      "AT",
      "ARE",
      "ELSE",
      "START",
      "NATIVE",
      "ADDRESS",
      "AND",
      "DESTINATION",
      "READ",
      "TRACE",
      "ADD",
      "ZEROS",
      "ZEROES",
      "END-READ",
      "REEL",
      "DELETE",
      "TERMINATE",
      "SELECT",
      "RELEASE",
      "END-START",
      "END-DELETE",
      "AREA",
      "RELOAD",
      "IS",
      "SENTENCE",
      "DATA",
      "SKIP1",
      "CD",
      "LAST",
      "IN",
      "STANDARD-1",
      "SERVICE",
      "SKIP3",
      "END-RECEIVE",
      "RECORDS",
      "STANDARD",
      "ALTER",
      "ID",
      "SKIP2",
      "RECORD",
      "ON",
      "TITLE",
      "INSERT",
      "TERMINAL",
      "STANDARD-2",
      "EXTERNAL",
      "EXIT",
      "INDEX",
      "DETAIL",
      "REDEFINES",
      "END-WRITE",
      "KANJI",
      "INDEXED",
      "NOT",
      "SORT",
      "OR",
      "ALTERNATE",
      "DECLARATIVES",
      "SIZE",
      "LINES",
      "END-REWRITE",
      "LINE",
      "ORDER",
      "SORT-RETURN",
      "THEN",
      "LINAGE",
      "REWRITE",
      "END-CALL",
      "REWIND",
      "INDICATE",
      "CLASS",
      "ZERO",
      "DELIMITED",
      "ALL",
      "DELIMITER",
      "THAN",
      "ACCESS",
      "ESI",
      "RECURSIVE",
      "INSPECT",
      "DECIMAL-POINT",
      "ALSO",
      "END-SEARCH",
      "SECTION",
      "CODE",
      "DATE-WRITTEN",
      "SEGMENT",
      "EXCEPTION",
      "DESCENDING",
      "ACCEPT",
      "INSTALLATION",
      "CALL",
      "CODE-SET",
      "CORR",
      "CANCEL",
      "CONTENT",
      "USE",
      "INTO",
      "JUST",
      "END-STRING",
      "NO",
      "GREATER",
      "WRITE",
      "TO",
      "RUN",
      "GENERATE",
      "LOCK",
      "SEQUENTIAL",
      "TRUE",
      "CLOSE",
      "INITIATE",
      "DOWN",
      "WORDS",
      "COLUMN",
      "USAGE",
      "SEQUENCE",
      "EGCS",
      "WHEN",
      "OCCURS",
      "DATE-COMPILED",
      "NULLS",
      "REPORTS",
      "SIGN",
      "OTHER",
      "UNTIL",
      "SPACES",
      "SPACE",
      "REPORT",
      "INITIALIZE",
      "RH",
      "AUTHOR",
      "CONTAINS",
      "TAPE",
      "RIGHT",
      "SEPARATE",
      "REMARKS",
      "SEGMENT-LIMIT",
      "RETURNING",
      "MERGE",
      "SEARCH",
      "NULL",
      "UNIT",
      "METHOD",
      "SAME",
      "END-PERFORM",
      "INITIAL",
      "SOURCE",
      "LINKAGE",
      "LENGTH",
      "REMAINDER",
      "TRAILING",
      "FD",
      "CHARACTERS",
      "SPECIAL-NAMES",
      "INHERITS",
      "RESERVE",
      "REPLACE",
      "END-INVOKE",
      "RETURN-CODE",
      "REFERENCES",
      "CH",
      "INPUT",
      "REVERSED",
      "REFERENCE",
      "CHARACTER",
      "CONTROLS",
      "OUTPUT",
      "OPEN",
      "WITH",
      "STOP",
      "BASIS",
      "ROUNDED",
      "EGI",
      "LINE-COUNTER",
      "LOW-VALUES",
      "ENDING",
      "SELF",
      "TALLYING",
      "LOW-VALUE",
      "THRU",
      "RECORDING",
      "TABLE",
      "STRING",
      "END-UNSTRING",
      "AFTER",
      "END-EVALUATE",
      "LEFT",
      "ENTRY",
      "QUEUE",
      "METACLASS",
      "TIMES",
      "TIME",
      "PROCEED",
      "ASCENDING",
      "CONVERTING",
      "GO",
      "VALUES",
      "CONTROL",
      "VALUE",
      "EVALUATE",
      "FALSE",
      "MODE",
      "COUNT",
      "MODULES",
      "CONTINUE",
      "DBCS",
      "BLANK",
      "PASSWORD",
      "SORT-CORE-SIZE",
      "RECEIVE",
      "SORT-CONTROL",
      "READY",
      "INVOKE",
      "SORT-MERGE",
      "END-COMPUTE",
      "SORT-MESSAGE",
      "DIVIDE",
      "LIMITS",
      "EQUAL",
      "PICTURE",
      "END-DIVIDE",
      "FIRST",
      "LIMIT",
      "GLOBAL",
      "REMOVAL",
      "NATIVE_BINARY",
      "SYNC",
      "LEADING",
      "WORKING-STORAGE",
      "ENVIRONMENT",
      "OPTIONAL",
      "PAGE",
      "FOR",
      "ORGANIZATION",
      "SUPPRESS",
      "OBJECT",
      "TALLY",
      "LABEL",
      "FILE",
      "FINAL",
      "WHEN-COMPILED",
      "SUPER",
      "PURGE",
      "COMP-1",
      "MESSAGE",
      "EMI",
      "COMMON",
      "POINTER",
      "USING",
      "RELATIVE",
      "PIC",
      "COMP-5",
      "LINAGE-COUNTER",
      "BLOCK",
      "QUOTES",
      "COMPUTE",
      "UNSTRING",
      "FILLER",
      "QUOTE",
      "COMP-3",
      "MORE-LABELS",
      "CBL",
      "PLUS",
      "COMMA",
      "DISPLAY-1",
      "PH",
      "COMP-2",
      "OMITTED",
      "COLLATING",
      "SYNCHRONIZED",
      "COMP-4",
      "INVALID",
      "NUMBER",
      "HEADING",
      "DEPENDING",
      "OVERRIDE",
      "REPORTING",
      "GROUP",
      "PROCESSING",
      "SECURITY",
      "COMPUTATIONAL-1",
      "POSITION",
      "COBOL",
      "UPON",
      "COMPUTATIONAL-5",
      "COMPUTATIONAL-3",
      "CLASS-ID",
      "DUPLICATES",
      "COMPUTATIONAL-2",
      "END-SUBTRACT",
      "COMPUTATIONAL-4",
      "PROCEDURES",
      "EOP",
      "PROCEDURE",
      "NEGATIVE",
      "TOP",
      "REPLACING",
      "COMPUTATIONAL",
      "THROUGH",
      "GOBACK",
      "I-O",
      "TYPE",
      "RANDOM",
      "IDENTIFICATION",
      "PRINTING",
      "SORT-MODE-SIZE",
      "DYNAMIC",
      "METHOD-ID",
      "CORRESPONDING",
      "JUSTIFIED\011 ",
      "DIVISION",
      "SHIFT-IN",
      "ALPHANUMERIC-EDITED",
      "SORT-FILE-SIZE",
      "PADDING",
      "END-OF-PAGE",
      "RF",
      "DEBUG-SUB-1",
      "END-IF",
      "DEBUG-NAME",
      "BEFORE",
      "SHIFT-OUT",
      "DEBUG-SUB-3",
      "SUBTRACT",
      "ALPHABET",
      "DEBUG-SUB-2",
      "PAGE-COUNTER",
      "I-O-CONTROL",
      "EVERY",
      "COMMUNICATION",
      "UP",
      "NUMERIC-EDITED",
      "ALPHANUMERIC",
      "CONFIGURATION",
      "PROGRAM-ID",
      "DEBUG-LINE",
      "ADVANCING",
      "CF",
      "COMP",
      "OVERFLOW",
      "MOVE",
      "VARYING",
      "PROCEDURE-POINTER",
      "LOCAL-STORAGE",
      "FROM",
      "IF",
      "ALPHABETIC-LOWER",
      "DEBUG-CONTENTS",
      "REPOSITORY",
      "KEY",
      "ALPHABETIC",
      "FUNCTION",
      "NUMERIC",
      "OF",
      "CURRENCY",
      "FILE-CONTROL",
      "CLOCK-UNITS",
      "SUB-QUEUE-1",
      "HIGH-VALUES",
      "HIGH-VALUE",
      "POSITIVE",
      "ANY",
      "SUB-QUEUE-3",
      "DAY",
      "SUM",
      "INPUT-OUTPUT",
      "SUB-QUEUE-2",
      "BEGINNING",
      "DEBUGGING",
      "COPY",
      "COM-REG",
      "PERFORM",
      "MULTIPLE",
      "SOURCE-COMPUTER",
      "SYMBOLIC",
      "DISPLAY",
      "PF",
      "APPLY",
      "ALPHABETIC-UPPER",
      "FOOTING",
      "GIVING",
      "PROGRAM",
      "OBJECT-COMPUTER",
      "PACKED-DECIMAL",
      "WRITE-ONLY",
      "END-MULTIPLY",
      "BOTTOM",
      "DEBUG-ITEM",
      "DAY-OF-WEEK",
      "BY",
      "OFF",
      "BINARY",
      "MULTIPLY",
      "MEMORY"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str29,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str45,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str54,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str55,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str70,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str81,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str83,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str131,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str132,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str134,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str150,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str170,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str175,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str181,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str184,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str189,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str192,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str194,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str197,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str204,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str207,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str209,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str210,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str211,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str215,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str220,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str221,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str222,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str223,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str226,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str228,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str230,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str254,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str259,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str263,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str264,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str265,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str266,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str268,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str274,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str277,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str278,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str279,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str281,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str282,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str284,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str287,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str292,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str294,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str295,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str296,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str297,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str299,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str303,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str304,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str306,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str307,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str308,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str309,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str314,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str315,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str317,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str322,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str325,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str327,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str328,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str333,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str334,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str335,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str339,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str340,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str343,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str344,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str345,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str346,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str355,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str358,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str359,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str370,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str371,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str373,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str375,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str382,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str384,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str386,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str390,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str391,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str395,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str396,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str400,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str404,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str406,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str408,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str409,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str415,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str418,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str427,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str428,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str429,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str430,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str443,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str444,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str449,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str451,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str454,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str456,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str462,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str466,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str467,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str468,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str469,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str473,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str475,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str476,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str478,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str479,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str480,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str482,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str483,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str484,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str488,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str489,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str490,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str491,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str492,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str495,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str498,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str501,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str504,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str505,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str509,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str510,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str512,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str513,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str517,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str520,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str521,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str522,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str523,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str524,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str525,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str529,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str530,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str531,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str532,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str533,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str535,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str537,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str538,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str540,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str544,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str545,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str549,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str552,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str554,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str555,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str557,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str564,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str567,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str568,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str571,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str573,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str574,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str575,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str577,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str578,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str579,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str580,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str583,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str584,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str585,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str587,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str588,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str589,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str590,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str591,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str597,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str599,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str601,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str605,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str607,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str608,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str613,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str615,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str616,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str617,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str621,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str622,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str627,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str630,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str634,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str638,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str639,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str641,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str642,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str643,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str646,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str648,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str650,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str652,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str653,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str654,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str655,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str660,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str661,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str662,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str663,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str671,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str672,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str675,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str676,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str678,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str681,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str682,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str685,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str686,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str687,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str688,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str689,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str690,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str691,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str701,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str703,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str704,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str710,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str712,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str719,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str721,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str722,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str729,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str730,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str731,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str740,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str741,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str748,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str754,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str756,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str759,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str760,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str765,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str771,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str785,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str788,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str790,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str794,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str795,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str800,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str801,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str810,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str815,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str817,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str820,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str830,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str833,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str834,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str836,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str843,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str849,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str853,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str860,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str861,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str866,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str867,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str871,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str872,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str873,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str874,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str875,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str877,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str878,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str879,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str881,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str887,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str894,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str897,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str902,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str907,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str908,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str909,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str917,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str918,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str919,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str923,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str924,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str928,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str938,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str939,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str942,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str949,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str961,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str963,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str967,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str974,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str977,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str978,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str980,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str983,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str992,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str993,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str994,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str996,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str997,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1010,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1012,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1016,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1017,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1018,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1021,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1022,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1023,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1024,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1030,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1041,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1042,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1043,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1046,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1055,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1059,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1061,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1065,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1069,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1071,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1074,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1076,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1084,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1088,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1090,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1091,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1094,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1104,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1107,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1157,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1177,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1178,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1208,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1216,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1218,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1223,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1233,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1246,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1255,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1309,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1357,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1378,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1407,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1411,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1418,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1421,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1433,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1450,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1473,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1487,
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
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1612,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1681,
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
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1817
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
