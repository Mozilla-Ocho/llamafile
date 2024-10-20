/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_sql.gperf  */
/* Computed positions: -k'1-4,6-7,9,12,$' */

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

#line 1 "llamafile/is_keyword_sql.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 927
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 32
#define MIN_HASH_VALUE 17
#define MAX_HASH_VALUE 7452
/* maximum key range = 7436, duplicates = 0 */

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
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453,    0, 7453, 7453,    0,   25,
         0,   20,   10, 7453, 7453, 7453,    5, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453,  175, 1604,  430,   50,    0,
       970,  600, 1909,  255, 1349, 1941,  330,  780,   15,   15,
       890, 1849,   65,   10,    5,   70, 1671, 1693, 1264, 1714,
       235,    5, 7453, 7453, 7453,  450, 7453,  175, 1604,  430,
        50,    0,  970,  600, 1909,  255, 1349, 1941,  330,  780,
        15,   15,  890, 1849,   65,   10,    5,   70, 1671, 1693,
      1264, 1714,  235,    5, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453, 7453,
      7453, 7453, 7453, 7453, 7453, 7453, 7453
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
        hval += asso_values[(unsigned char)str[8]];
      /*FALLTHROUGH*/
      case 8:
      case 7:
        hval += asso_values[(unsigned char)str[6]];
      /*FALLTHROUGH*/
      case 6:
        hval += asso_values[(unsigned char)str[5]];
      /*FALLTHROUGH*/
      case 5:
      case 4:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
        hval += asso_values[(unsigned char)str[1]];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]+1];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_sql (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str17[sizeof("NE")];
      char stringpool_str18[sizeof("SET")];
      char stringpool_str23[sizeof("RET")];
      char stringpool_str27[sizeof("SS")];
      char stringpool_str32[sizeof("DO")];
      char stringpool_str34[sizeof("SETS")];
      char stringpool_str39[sizeof("ZONE")];
      char stringpool_str43[sizeof("NOT")];
      char stringpool_str47[sizeof("NO")];
      char stringpool_str49[sizeof("NONE")];
      char stringpool_str62[sizeof("CT")];
      char stringpool_str72[sizeof("CS")];
      char stringpool_str76[sizeof("MONTHS")];
      char stringpool_str77[sizeof("SESSION")];
      char stringpool_str84[sizeof("MODE")];
      char stringpool_str88[sizeof("COS")];
      char stringpool_str96[sizeof("RESUME")];
      char stringpool_str97[sizeof("RESTORE")];
      char stringpool_str102[sizeof("TO")];
      char stringpool_str106[sizeof("RESULT")];
      char stringpool_str107[sizeof("RESTART")];
      char stringpool_str113[sizeof("RUN")];
      char stringpool_str115[sizeof("MONSESSION")];
      char stringpool_str116[sizeof("STORES")];
      char stringpool_str117[sizeof("CONTENT")];
      char stringpool_str121[sizeof("RETURN")];
      char stringpool_str122[sizeof("ROUTINE")];
      char stringpool_str127[sizeof("RETURNS")];
      char stringpool_str133[sizeof("MOD")];
      char stringpool_str152[sizeof("CD")];
      char stringpool_str155[sizeof("CROSS")];
      char stringpool_str156[sizeof("MODULE")];
      char stringpool_str160[sizeof("COUNT")];
      char stringpool_str161[sizeof("SOURCE")];
      char stringpool_str165[sizeof("SETRESRATE")];
      char stringpool_str178[sizeof("CONTINUE")];
      char stringpool_str184[sizeof("DATE")];
      char stringpool_str195[sizeof("STATE")];
      char stringpool_str196[sizeof("STORED")];
      char stringpool_str206[sizeof("RENAME")];
      char stringpool_str209[sizeof("TRUE")];
      char stringpool_str211[sizeof("MONRESOURCE")];
      char stringpool_str217[sizeof("SETUSER")];
      char stringpool_str226[sizeof("SETSESSRATE")];
      char stringpool_str229[sizeof("STRUCTURE")];
      char stringpool_str239[sizeof("CASE")];
      char stringpool_str249[sizeof("CAST")];
      char stringpool_str250[sizeof("READS")];
      char stringpool_str252[sizeof("QUERYNO")];
      char stringpool_str260[sizeof("START")];
      char stringpool_str264[sizeof("CORR")];
      char stringpool_str267[sizeof("SESSION_USER")];
      char stringpool_str271[sizeof("CASE_N")];
      char stringpool_str278[sizeof("TAN")];
      char stringpool_str281[sizeof("DSSIZE")];
      char stringpool_str282[sizeof("CURRENT")];
      char stringpool_str287[sizeof("CURRENT_TIME")];
      char stringpool_str289[sizeof("READ")];
      char stringpool_str291[sizeof("CURRENT_TIMEZONE")];
      char stringpool_str293[sizeof("SIN")];
      char stringpool_str294[sizeof("HOST")];
      char stringpool_str296[sizeof("CREATE")];
      char stringpool_str299[sizeof("SENSITIVE")];
      char stringpool_str303[sizeof("MIN")];
      char stringpool_str306[sizeof("CONSTRUCTOR")];
      char stringpool_str320[sizeof("TREAT")];
      char stringpool_str322[sizeof("CURRENT_TRANSFORM_GROUP_FOR_TYPE")];
      char stringpool_str331[sizeof("CURSOR")];
      char stringpool_str332[sizeof("CURRENT_DATE")];
      char stringpool_str336[sizeof("DELETE")];
      char stringpool_str344[sizeof("TRAN")];
      char stringpool_str347[sizeof("CURRENT_ROLE")];
      char stringpool_str351[sizeof("SELECT")];
      char stringpool_str357[sizeof("RELEASE")];
      char stringpool_str359[sizeof("ROLE")];
      char stringpool_str360[sizeof("CONSTRAINT")];
      char stringpool_str361[sizeof("MINUTE")];
      char stringpool_str366[sizeof("CONSTRAINTS")];
      char stringpool_str370[sizeof("MINUS")];
      char stringpool_str373[sizeof("CONTAINS")];
      char stringpool_str382[sizeof("MINUTES")];
      char stringpool_str408[sizeof("TRUNCATE")];
      char stringpool_str410[sizeof("CLOSE")];
      char stringpool_str414[sizeof("RULE")];
      char stringpool_str415[sizeof("CLONE")];
      char stringpool_str420[sizeof("HOURS")];
      char stringpool_str429[sizeof("CONDITION")];
      char stringpool_str442[sizeof("BT")];
      char stringpool_str443[sizeof("MULTISET")];
      char stringpool_str452[sizeof("MONITOR")];
      char stringpool_str469[sizeof("CURRENT_SCHEMA")];
      char stringpool_str474[sizeof("HOUR")];
      char stringpool_str482[sizeof("CURRENT_USER")];
      char stringpool_str490[sizeof("YEARS")];
      char stringpool_str491[sizeof("HOUR_SECOND")];
      char stringpool_str492[sizeof("SECTION")];
      char stringpool_str498[sizeof("STANDARD")];
      char stringpool_str499[sizeof("SIZE")];
      char stringpool_str501[sizeof("COLUMN")];
      char stringpool_str513[sizeof("BUT")];
      char stringpool_str526[sizeof("CONCAT")];
      char stringpool_str527[sizeof("SECONDS")];
      char stringpool_str532[sizeof("RADIANS")];
      char stringpool_str534[sizeof("DATA")];
      char stringpool_str538[sizeof("RESOURCE")];
      char stringpool_str542[sizeof("CONNECT")];
      char stringpool_str543[sizeof("DOCUMENT")];
      char stringpool_str544[sizeof("YEAR")];
      char stringpool_str547[sizeof("NOAUDIT")];
      char stringpool_str548[sizeof("DATABASE")];
      char stringpool_str556[sizeof("SECOND")];
      char stringpool_str569[sizeof("DATABASES")];
      char stringpool_str570[sizeof("CONNECTION")];
      char stringpool_str580[sizeof("CLASS")];
      char stringpool_str590[sizeof("MLOAD")];
      char stringpool_str597[sizeof("CLUSTER")];
      char stringpool_str605[sizeof("DESTRUCTOR")];
      char stringpool_str608[sizeof("MODIFIES")];
      char stringpool_str610[sizeof("NTILE")];
      char stringpool_str612[sizeof("DELAYED")];
      char stringpool_str632[sizeof("HANDLER")];
      char stringpool_str634[sizeof("CLUSTERED")];
      char stringpool_str644[sizeof("MIDDLEINT")];
      char stringpool_str652[sizeof("MATCHES")];
      char stringpool_str655[sizeof("STATISTICS")];
      char stringpool_str663[sizeof("DEL")];
      char stringpool_str665[sizeof("TITLE")];
      char stringpool_str667[sizeof("NONCLUSTERED")];
      char stringpool_str668[sizeof("SEL")];
      char stringpool_str669[sizeof("FREE")];
      char stringpool_str671[sizeof("FREEZE")];
      char stringpool_str678[sizeof("SSL")];
      char stringpool_str685[sizeof("MERGE")];
      char stringpool_str692[sizeof("DEGREES")];
      char stringpool_str698[sizeof("BIT")];
      char stringpool_str703[sizeof("CONTAINSTABLE")];
      char stringpool_str704[sizeof("HOLD")];
      char stringpool_str705[sizeof("MACRO")];
      char stringpool_str719[sizeof("RAISERROR")];
      char stringpool_str722[sizeof("CASCADE")];
      char stringpool_str723[sizeof("RESTRICT")];
      char stringpool_str728[sizeof("DISTINCT")];
      char stringpool_str737[sizeof("COLLATE")];
      char stringpool_str745[sizeof("TRACE")];
      char stringpool_str747[sizeof("REGR_R2")];
      char stringpool_str748[sizeof("FOR")];
      char stringpool_str753[sizeof("KURTOSIS")];
      char stringpool_str755[sizeof("FOUND")];
      char stringpool_str760[sizeof("NULLS")];
      char stringpool_str770[sizeof("NESTED_TABLE_ID")];
      char stringpool_str773[sizeof("CASCADED")];
      char stringpool_str777[sizeof("DISABLE")];
      char stringpool_str782[sizeof("LE")];
      char stringpool_str784[sizeof("RECURSIVE")];
      char stringpool_str792[sizeof("LT")];
      char stringpool_str800[sizeof("CCSID")];
      char stringpool_str803[sizeof("RESIGNAL")];
      char stringpool_str804[sizeof("SOME")];
      char stringpool_str805[sizeof("RANGE")];
      char stringpool_str812[sizeof("LN")];
      char stringpool_str814[sizeof("LESS")];
      char stringpool_str822[sizeof("MICROSECONDS")];
      char stringpool_str828[sizeof("DISABLED")];
      char stringpool_str831[sizeof("COLLID")];
      char stringpool_str832[sizeof("DECLARE")];
      char stringpool_str839[sizeof("TRANSLATE")];
      char stringpool_str849[sizeof("REAL")];
      char stringpool_str851[sizeof("MICROSECOND")];
      char stringpool_str863[sizeof("DEC")];
      char stringpool_str874[sizeof("DESC")];
      char stringpool_str883[sizeof("MINUTE_SECOND")];
      char stringpool_str908[sizeof("ONE")];
      char stringpool_str909[sizeof("DUAL")];
      char stringpool_str916[sizeof("REPEAT")];
      char stringpool_str918[sizeof("QUANTILE")];
      char stringpool_str922[sizeof("ON")];
      char stringpool_str940[sizeof("FIRST")];
      char stringpool_str950[sizeof("MAINTAINED")];
      char stringpool_str958[sizeof("RESULT_SET_LOCATOR")];
      char stringpool_str964[sizeof("REGR_INTERCEPT")];
      char stringpool_str965[sizeof("DEALLOCATE")];
      char stringpool_str969[sizeof("CEIL")];
      char stringpool_str971[sizeof("FUSION")];
      char stringpool_str973[sizeof("OUT")];
      char stringpool_str975[sizeof("LEAST")];
      char stringpool_str978[sizeof("NATIONAL")];
      char stringpool_str979[sizeof("LAST")];
      char stringpool_str982[sizeof("ET")];
      char stringpool_str985[sizeof("NAMES")];
      char stringpool_str989[sizeof("STATEMENT")];
      char stringpool_str991[sizeof("CALLED")];
      char stringpool_str992[sizeof("MATERIALIZED")];
      char stringpool_str1001[sizeof("ROLLFORWARD")];
      char stringpool_str1002[sizeof("BOOLEAN")];
      char stringpool_str1003[sizeof("SUCCEEDS")];
      char stringpool_str1006[sizeof("DOMAIN")];
      char stringpool_str1008[sizeof("ZEROFILL")];
      char stringpool_str1015[sizeof("REFERENCES")];
      char stringpool_str1018[sizeof("COALESCE")];
      char stringpool_str1022[sizeof("OR")];
      char stringpool_str1024[sizeof("COLLATION")];
      char stringpool_str1025[sizeof("NAMED")];
      char stringpool_str1030[sizeof("FLOOR")];
      char stringpool_str1035[sizeof("OUTER")];
      char stringpool_str1036[sizeof("MASTER_BIND")];
      char stringpool_str1040[sizeof("DEREF")];
      char stringpool_str1046[sizeof("CURRENT_LC_CTYPE")];
      char stringpool_str1047[sizeof("NOTNULL")];
      char stringpool_str1048[sizeof("DATEFORM")];
      char stringpool_str1056[sizeof("STATIC")];
      char stringpool_str1059[sizeof("LEAD")];
      char stringpool_str1065[sizeof("LINES")];
      char stringpool_str1074[sizeof("LOAD")];
      char stringpool_str1075[sizeof("ORDER")];
      char stringpool_str1079[sizeof("NULL")];
      char stringpool_str1086[sizeof("LINENO")];
      char stringpool_str1088[sizeof("END")];
      char stringpool_str1092[sizeof("CURRVAL")];
      char stringpool_str1093[sizeof("DEFERRED")];
      char stringpool_str1097[sizeof("SUSPEND")];
      char stringpool_str1107[sizeof("NATURAL")];
      char stringpool_str1109[sizeof("TIME")];
      char stringpool_str1111[sizeof("TRANSLATION")];
      char stringpool_str1114[sizeof("TERMINATE")];
      char stringpool_str1115[sizeof("FORCE")];
      char stringpool_str1117[sizeof("COLUMN_VALUE")];
      char stringpool_str1120[sizeof("FALSE")];
      char stringpool_str1127[sizeof("TRIGGER")];
      char stringpool_str1128[sizeof("STARTING")];
      char stringpool_str1130[sizeof("FLOAT")];
      char stringpool_str1136[sizeof("FLOAT8")];
      char stringpool_str1143[sizeof("MINUTE_MICROSECOND")];
      char stringpool_str1146[sizeof("FLOAT4")];
      char stringpool_str1150[sizeof("REGR_COUNT")];
      char stringpool_str1151[sizeof("FENCED")];
      char stringpool_str1165[sizeof("TERMINATED")];
      char stringpool_str1170[sizeof("DISCONNECT")];
      char stringpool_str1172[sizeof("COLLECT")];
      char stringpool_str1179[sizeof("RPAD")];
      char stringpool_str1181[sizeof("SCROLL")];
      char stringpool_str1182[sizeof("CURRENT_TIMESTAMP")];
      char stringpool_str1185[sizeof("ERROR")];
      char stringpool_str1186[sizeof("LINEAR")];
      char stringpool_str1189[sizeof("FILE")];
      char stringpool_str1192[sizeof("CAPTURE")];
      char stringpool_str1200[sizeof("COLLECTION")];
      char stringpool_str1211[sizeof("TRANSACTION")];
      char stringpool_str1213[sizeof("SECURITYAUDIT")];
      char stringpool_str1219[sizeof("CALL")];
      char stringpool_str1224[sizeof("CUME_DIST")];
      char stringpool_str1225[sizeof("ERASE")];
      char stringpool_str1231[sizeof("DEFINE")];
      char stringpool_str1246[sizeof("REPOVERRIDE")];
      char stringpool_str1260[sizeof("SPOOL")];
      char stringpool_str1287[sizeof("RANGE_N")];
      char stringpool_str1288[sizeof("SECOND_MICROSECOND")];
      char stringpool_str1293[sizeof("NEXT")];
      char stringpool_str1305[sizeof("BEGIN")];
      char stringpool_str1314[sizeof("ELSE")];
      char stringpool_str1317[sizeof("FOREIGN")];
      char stringpool_str1323[sizeof("OLD")];
      char stringpool_str1326[sizeof("FILTER")];
      char stringpool_str1331[sizeof("DIAGNOSTICS")];
      char stringpool_str1332[sizeof("RUNNING")];
      char stringpool_str1345[sizeof("SCOPE")];
      char stringpool_str1365[sizeof("SUCCESSFUL")];
      char stringpool_str1371[sizeof("IS")];
      char stringpool_str1373[sizeof("INT2")];
      char stringpool_str1374[sizeof("MEDIUMINT")];
      char stringpool_str1377[sizeof("INT")];
      char stringpool_str1380[sizeof("FINAL")];
      char stringpool_str1381[sizeof("IN")];
      char stringpool_str1383[sizeof("INT8")];
      char stringpool_str1387[sizeof("INS")];
      char stringpool_str1389[sizeof("SEPARATOR")];
      char stringpool_str1390[sizeof("INSERT")];
      char stringpool_str1393[sizeof("INT4")];
      char stringpool_str1398[sizeof("INTERSECT")];
      char stringpool_str1403[sizeof("INTO")];
      char stringpool_str1406[sizeof("LOCALE")];
      char stringpool_str1408[sizeof("FUNCTION")];
      char stringpool_str1409[sizeof("STRING_CS")];
      char stringpool_str1413[sizeof("INT3")];
      char stringpool_str1416[sizeof("CURRENT_DEFAULT_TRANSFORM_GROUP")];
      char stringpool_str1421[sizeof("BEFORE")];
      char stringpool_str1423[sizeof("INT1")];
      char stringpool_str1425[sizeof("LOG10")];
      char stringpool_str1426[sizeof("INTERSECTION")];
      char stringpool_str1430[sizeof("LOGON")];
      char stringpool_str1431[sizeof("ITERATE")];
      char stringpool_str1443[sizeof("TRAILING")];
      char stringpool_str1449[sizeof("INNER")];
      char stringpool_str1459[sizeof("INOUT")];
      char stringpool_str1460[sizeof("NOCOMPRESS")];
      char stringpool_str1469[sizeof("NORMALIZE")];
      char stringpool_str1476[sizeof("FORMAT")];
      char stringpool_str1484[sizeof("MAXEXTENTS")];
      char stringpool_str1490[sizeof("CURRENT_CATALOG")];
      char stringpool_str1492[sizeof("DEFAULT")];
      char stringpool_str1496[sizeof("ONLINE")];
      char stringpool_str1498[sizeof("LOCATORS")];
      char stringpool_str1505[sizeof("SPACE")];
      char stringpool_str1506[sizeof("INTEGER")];
      char stringpool_str1508[sizeof("SEMANTICSIMILARITYTABLE")];
      char stringpool_str1512[sizeof("READTEXT")];
      char stringpool_str1515[sizeof("SEMANTICSIMILARITYDETAILSTABLE")];
      char stringpool_str1521[sizeof("HOUR_MINUTE")];
      char stringpool_str1527[sizeof("DECIMAL")];
      char stringpool_str1528[sizeof("ROUND_UP")];
      char stringpool_str1540[sizeof("BTRIM")];
      char stringpool_str1541[sizeof("SIGNAL")];
      char stringpool_str1550[sizeof("DEFERRABLE")];
      char stringpool_str1552[sizeof("LOCATOR")];
      char stringpool_str1556[sizeof("BIGINT")];
      char stringpool_str1564[sizeof("RETURNING")];
      char stringpool_str1571[sizeof("HOUR_MICROSECOND")];
      char stringpool_str1573[sizeof("SMALLINT")];
      char stringpool_str1579[sizeof("KILL")];
      char stringpool_str1591[sizeof("ESCAPE")];
      char stringpool_str1594[sizeof("WRITE")];
      char stringpool_str1600[sizeof("MATCH_RECOGNIZE")];
      char stringpool_str1603[sizeof("DECFLOAT")];
      char stringpool_str1607[sizeof("SIMILAR")];
      char stringpool_str1608[sizeof("STOGROUP")];
      char stringpool_str1612[sizeof("CM")];
      char stringpool_str1613[sizeof("WRITETEXT")];
      char stringpool_str1615[sizeof("TIMEZONE_MINUTE")];
      char stringpool_str1616[sizeof("AT")];
      char stringpool_str1620[sizeof("INTEGERDATE")];
      char stringpool_str1621[sizeof("ROUND_FLOOR")];
      char stringpool_str1625[sizeof("LARGE")];
      char stringpool_str1626[sizeof("AS")];
      char stringpool_str1627[sizeof("CATALOG")];
      char stringpool_str1638[sizeof("SUM")];
      char stringpool_str1641[sizeof("COMMIT")];
      char stringpool_str1646[sizeof("ERRORTABLES")];
      char stringpool_str1657[sizeof("COMMENT")];
      char stringpool_str1659[sizeof("MSUM")];
      char stringpool_str1661[sizeof("INSTEAD")];
      char stringpool_str1664[sizeof("FULL")];
      char stringpool_str1667[sizeof("REPLACE")];
      char stringpool_str1669[sizeof("LOCALTIME")];
      char stringpool_str1672[sizeof("ARE")];
      char stringpool_str1675[sizeof("DESCRIPTOR")];
      char stringpool_str1684[sizeof("USE")];
      char stringpool_str1692[sizeof("ESCAPED")];
      char stringpool_str1694[sizeof("CSUM")];
      char stringpool_str1695[sizeof("DOUBLE")];
      char stringpool_str1697[sizeof("REVERT")];
      char stringpool_str1702[sizeof("REVOKE")];
      char stringpool_str1705[sizeof("SUBSET")];
      char stringpool_str1708[sizeof("WAIT")];
      char stringpool_str1717[sizeof("UNNEST")];
      char stringpool_str1722[sizeof("AND")];
      char stringpool_str1728[sizeof("CUBE")];
      char stringpool_str1733[sizeof("TIMEZONE_HOUR")];
      char stringpool_str1734[sizeof("END_FRAME")];
      char stringpool_str1735[sizeof("LOCAL")];
      char stringpool_str1738[sizeof("CORRESPONDING")];
      char stringpool_str1739[sizeof("FREESPACE")];
      char stringpool_str1740[sizeof("ZEROIFNULL")];
      char stringpool_str1742[sizeof("ROWS")];
      char stringpool_str1744[sizeof("ROWSET")];
      char stringpool_str1747[sizeof("COMPUTE")];
      char stringpool_str1750[sizeof("DIAGNOSTIC")];
      char stringpool_str1757[sizeof("ADD")];
      char stringpool_str1759[sizeof("RETRIEVE")];
      char stringpool_str1761[sizeof("RECONFIGURE")];
      char stringpool_str1763[sizeof("COMPRESS")];
      char stringpool_str1764[sizeof("LEFT")];
      char stringpool_str1770[sizeof("UNDO")];
      char stringpool_str1780[sizeof("COMPLETION")];
      char stringpool_str1791[sizeof("SYSDATE")];
      char stringpool_str1798[sizeof("EDITPROC")];
      char stringpool_str1802[sizeof("LATERAL")];
      char stringpool_str1804[sizeof("ATAN2")];
      char stringpool_str1806[sizeof("UNDER")];
      char stringpool_str1812[sizeof("INITIATE")];
      char stringpool_str1813[sizeof("ENCLOSED")];
      char stringpool_str1814[sizeof("OPEN")];
      char stringpool_str1815[sizeof("USER")];
      char stringpool_str1816[sizeof("RANDOM")];
      char stringpool_str1818[sizeof("ATAN")];
      char stringpool_str1822[sizeof("MLINREG")];
      char stringpool_str1825[sizeof("SUBSTR")];
      char stringpool_str1833[sizeof("CONVERT")];
      char stringpool_str1837[sizeof("TEXTSIZE")];
      char stringpool_str1841[sizeof("MSUBSTR")];
      char stringpool_str1843[sizeof("NUMPARTS")];
      char stringpool_str1851[sizeof("WAITFOR")];
      char stringpool_str1855[sizeof("SAVE")];
      char stringpool_str1856[sizeof("SAMPLE")];
      char stringpool_str1857[sizeof("CEILING")];
      char stringpool_str1861[sizeof("REFERENCING")];
      char stringpool_str1862[sizeof("XOR")];
      char stringpool_str1864[sizeof("DROP")];
      char stringpool_str1865[sizeof("CLASSIFIER")];
      char stringpool_str1868[sizeof("TOP")];
      char stringpool_str1871[sizeof("OUTPUT")];
      char stringpool_str1873[sizeof("ROUND_CEILING")];
      char stringpool_str1890[sizeof("LTRIM")];
      char stringpool_str1903[sizeof("ASIN")];
      char stringpool_str1905[sizeof("INSENSITIVE")];
      char stringpool_str1908[sizeof("STEPINFO")];
      char stringpool_str1910[sizeof("DBINFO")];
      char stringpool_str1911[sizeof("GE")];
      char stringpool_str1913[sizeof("DAYS")];
      char stringpool_str1914[sizeof("NOWAIT")];
      char stringpool_str1915[sizeof("REGR_SLOPE")];
      char stringpool_str1917[sizeof("SPATIAL")];
      char stringpool_str1919[sizeof("QUERY")];
      char stringpool_str1921[sizeof("GT")];
      char stringpool_str1922[sizeof("GET")];
      char stringpool_str1923[sizeof("ASSERTION")];
      char stringpool_str1933[sizeof("SQRT")];
      char stringpool_str1936[sizeof("SIMPLE")];
      char stringpool_str1939[sizeof("OMIT")];
      char stringpool_str1941[sizeof("GO")];
      char stringpool_str1942[sizeof("FREETEXT")];
      char stringpool_str1949[sizeof("LPAD")];
      char stringpool_str1953[sizeof("REF")];
      char stringpool_str1954[sizeof("TRIM")];
      char stringpool_str1956[sizeof("REQUEST")];
      char stringpool_str1963[sizeof("GOTO")];
      char stringpool_str1964[sizeof("MONTH")];
      char stringpool_str1973[sizeof("MAP")];
      char stringpool_str1974[sizeof("QUALIFIED")];
      char stringpool_str1976[sizeof("UNION")];
      char stringpool_str1982[sizeof("PER")];
      char stringpool_str1985[sizeof("IGNORE")];
      char stringpool_str1986[sizeof("PORTION")];
      char stringpool_str1987[sizeof("NUMERIC")];
      char stringpool_str1988[sizeof("INDICATOR")];
      char stringpool_str1989[sizeof("AUDIT")];
      char stringpool_str1993[sizeof("DETERMINISTIC")];
      char stringpool_str1998[sizeof("LOG")];
      char stringpool_str2000[sizeof("JSON")];
      char stringpool_str2001[sizeof("REQUIRE")];
      char stringpool_str2003[sizeof("ISOLATION")];
      char stringpool_str2008[sizeof("READ_WRITE")];
      char stringpool_str2009[sizeof("ALTER")];
      char stringpool_str2012[sizeof("MATCH_NUMBER")];
      char stringpool_str2013[sizeof("THEN")];
      char stringpool_str2014[sizeof("LONG")];
      char stringpool_str2023[sizeof("END_PARTITION")];
      char stringpool_str2024[sizeof("UNSIGNED")];
      char stringpool_str2028[sizeof("ROWID")];
      char stringpool_str2029[sizeof("UID")];
      char stringpool_str2030[sizeof("CURRENT_SERVER")];
      char stringpool_str2035[sizeof("METHOD")];
      char stringpool_str2039[sizeof("CURRENT_ROW")];
      char stringpool_str2040[sizeof("NULLIFZERO")];
      char stringpool_str2052[sizeof("PREORDER")];
      char stringpool_str2059[sizeof("STYLE")];
      char stringpool_str2064[sizeof("OPENDATASOURCE")];
      char stringpool_str2068[sizeof("KEYS")];
      char stringpool_str2072[sizeof("ANSIDATE")];
      char stringpool_str2073[sizeof("ACOS")];
      char stringpool_str2076[sizeof("OPTION")];
      char stringpool_str2078[sizeof("ASSOCIATE")];
      char stringpool_str2080[sizeof("LIMIT")];
      char stringpool_str2084[sizeof("SUBSCRIBER")];
      char stringpool_str2085[sizeof("GROUPS")];
      char stringpool_str2086[sizeof("TINYINT")];
      char stringpool_str2096[sizeof("ERRLVL")];
      char stringpool_str2101[sizeof("INCONSISTENT")];
      char stringpool_str2103[sizeof("PART")];
      char stringpool_str2107[sizeof("STARTUP")];
      char stringpool_str2109[sizeof("TRANSLATE_REGEX")];
      char stringpool_str2110[sizeof("ISNULL")];
      char stringpool_str2112[sizeof("ELEMENT")];
      char stringpool_str2114[sizeof("KEEP")];
      char stringpool_str2115[sizeof("UNDEFINED")];
      char stringpool_str2116[sizeof("REPLICATION")];
      char stringpool_str2126[sizeof("NEXTVAL")];
      char stringpool_str2127[sizeof("PAD")];
      char stringpool_str2135[sizeof("VOLUMES")];
      char stringpool_str2136[sizeof("PATTERN")];
      char stringpool_str2140[sizeof("SAVEPOINT")];
      char stringpool_str2141[sizeof("ANALYSE")];
      char stringpool_str2144[sizeof("OPERATION")];
      char stringpool_str2153[sizeof("BYTE")];
      char stringpool_str2154[sizeof("IDENTITY_INSERT")];
      char stringpool_str2155[sizeof("CHANGE")];
      char stringpool_str2158[sizeof("LAG")];
      char stringpool_str2159[sizeof("SHARE")];
      char stringpool_str2163[sizeof("SAMPLEID")];
      char stringpool_str2164[sizeof("BYTES")];
      char stringpool_str2165[sizeof("BETWEEN")];
      char stringpool_str2172[sizeof("GREATEST")];
      char stringpool_str2174[sizeof("GRANT")];
      char stringpool_str2176[sizeof("REVALIDATE")];
      char stringpool_str2177[sizeof("MINIMUM")];
      char stringpool_str2181[sizeof("BYTEINT")];
      char stringpool_str2182[sizeof("PASSWORD")];
      char stringpool_str2184[sizeof("TABLE")];
      char stringpool_str2187[sizeof("ALTERAND")];
      char stringpool_str2188[sizeof("THAN")];
      char stringpool_str2194[sizeof("PRINT")];
      char stringpool_str2195[sizeof("DISTRIBUTED")];
      char stringpool_str2209[sizeof("BROWSE")];
      char stringpool_str2210[sizeof("FILLFACTOR")];
      char stringpool_str2213[sizeof("GENERATED")];
      char stringpool_str2214[sizeof("CHARS")];
      char stringpool_str2218[sizeof("PSID")];
      char stringpool_str2220[sizeof("VARIANT")];
      char stringpool_str2227[sizeof("LEADING")];
      char stringpool_str2230[sizeof("PADDED")];
      char stringpool_str2241[sizeof("CONVERT_TABLE_HEADER")];
      char stringpool_str2242[sizeof("LOADING")];
      char stringpool_str2244[sizeof("FROM")];
      char stringpool_str2245[sizeof("JOIN")];
      char stringpool_str2246[sizeof("ROWCOUNT")];
      char stringpool_str2249[sizeof("JAR")];
      char stringpool_str2252[sizeof("CASESPECIFIC")];
      char stringpool_str2254[sizeof("PRIOR")];
      char stringpool_str2265[sizeof("MDIFF")];
      char stringpool_str2267[sizeof("ORGANIZATION")];
      char stringpool_str2268[sizeof("CHAR")];
      char stringpool_str2272[sizeof("OUTFILE")];
      char stringpool_str2273[sizeof("VALUE")];
      char stringpool_str2275[sizeof("PERIOD")];
      char stringpool_str2277[sizeof("FREETEXTTABLE")];
      char stringpool_str2281[sizeof("UNTIL")];
      char stringpool_str2294[sizeof("VALUES")];
      char stringpool_str2307[sizeof("DESCRIBE")];
      char stringpool_str2308[sizeof("ENCODING")];
      char stringpool_str2312[sizeof("VCAT")];
      char stringpool_str2328[sizeof("ROUND_DOWN")];
      char stringpool_str2330[sizeof("ACTION")];
      char stringpool_str2334[sizeof("INPUT")];
      char stringpool_str2345[sizeof("TRY_CONVERT")];
      char stringpool_str2353[sizeof("LANGUAGE")];
      char stringpool_str2354[sizeof("PARAMETERS")];
      char stringpool_str2360[sizeof("ERRORFILES")];
      char stringpool_str2366[sizeof("ANALYZE")];
      char stringpool_str2369[sizeof("HELP")];
      char stringpool_str2374[sizeof("INITIALIZE")];
      char stringpool_str2376[sizeof("PERCENT")];
      char stringpool_str2377[sizeof("SEQUENCE")];
      char stringpool_str2379[sizeof("ALIAS")];
      char stringpool_str2382[sizeof("SQLSTATE")];
      char stringpool_str2384[sizeof("MEDIUMTEXT")];
      char stringpool_str2388[sizeof("PLAN")];
      char stringpool_str2393[sizeof("PARTITION")];
      char stringpool_str2396[sizeof("REGEXP")];
      char stringpool_str2398[sizeof("INCLUSIVE")];
      char stringpool_str2399[sizeof("NCLOB")];
      char stringpool_str2408[sizeof("PARAMETER")];
      char stringpool_str2409[sizeof("PROTECTION")];
      char stringpool_str2412[sizeof("PRECEDES")];
      char stringpool_str2415[sizeof("ENCRYPTION")];
      char stringpool_str2422[sizeof("POSITION")];
      char stringpool_str2423[sizeof("VERSIONING")];
      char stringpool_str2430[sizeof("PARTITIONED")];
      char stringpool_str2435[sizeof("MASTER_SSL_VERIFY_SERVER_CERT")];
      char stringpool_str2445[sizeof("IDENTITYCOL")];
      char stringpool_str2449[sizeof("RELATIVE")];
      char stringpool_str2461[sizeof("USAGE")];
      char stringpool_str2466[sizeof("INITIAL")];
      char stringpool_str2467[sizeof("ALLOCATE")];
      char stringpool_str2471[sizeof("ROLLUP")];
      char stringpool_str2476[sizeof("ASUTIME")];
      char stringpool_str2477[sizeof("ASC")];
      char stringpool_str2480[sizeof("TBL_CS")];
      char stringpool_str2484[sizeof("DISTINCTROW")];
      char stringpool_str2486[sizeof("DISALLOW")];
      char stringpool_str2487[sizeof("SECURITY")];
      char stringpool_str2488[sizeof("PROCEDURE")];
      char stringpool_str2490[sizeof("ACCESS")];
      char stringpool_str2496[sizeof("ENDING")];
      char stringpool_str2497[sizeof("DAY_HOUR")];
      char stringpool_str2501[sizeof("JSON_TABLE")];
      char stringpool_str2503[sizeof("EXIT")];
      char stringpool_str2506[sizeof("DENSE_RANK")];
      char stringpool_str2509[sizeof("DICTIONARY")];
      char stringpool_str2510[sizeof("CARDINALITY")];
      char stringpool_str2511[sizeof("ACCOUNT")];
      char stringpool_str2515[sizeof("JSON_SERIALIZE")];
      char stringpool_str2517[sizeof("SQL")];
      char stringpool_str2525[sizeof("EXISTS")];
      char stringpool_str2529[sizeof("CYCLE")];
      char stringpool_str2533[sizeof("UC")];
      char stringpool_str2535[sizeof("MEMBER")];
      char stringpool_str2538[sizeof("OPTIMIZE")];
      char stringpool_str2539[sizeof("MATCH")];
      char stringpool_str2541[sizeof("RLIKE")];
      char stringpool_str2546[sizeof("SCHEMAS")];
      char stringpool_str2556[sizeof("USING")];
      char stringpool_str2558[sizeof("LOWER")];
      char stringpool_str2567[sizeof("SPECIFICTYPE")];
      char stringpool_str2577[sizeof("OPTIMIZATION")];
      char stringpool_str2589[sizeof("PURGE")];
      char stringpool_str2594[sizeof("LOOP")];
      char stringpool_str2595[sizeof("INFILE")];
      char stringpool_str2597[sizeof("ALL")];
      char stringpool_str2599[sizeof("NCHAR")];
      char stringpool_str2605[sizeof("NUMBER")];
      char stringpool_str2607[sizeof("FULLTEXT")];
      char stringpool_str2611[sizeof("VARCHAR2")];
      char stringpool_str2616[sizeof("BREAK")];
      char stringpool_str2624[sizeof("DAY_MINUTE")];
      char stringpool_str2627[sizeof("MLSLABEL")];
      char stringpool_str2631[sizeof("LEAVE")];
      char stringpool_str2634[sizeof("DUMP")];
      char stringpool_str2635[sizeof("OPTIMIZER_COSTS")];
      char stringpool_str2640[sizeof("SOUNDEX")];
      char stringpool_str2641[sizeof("VALIDATE")];
      char stringpool_str2646[sizeof("ARRAY_EXISTS")];
      char stringpool_str2648[sizeof("PRECISION")];
      char stringpool_str2649[sizeof("AFTER")];
      char stringpool_str2658[sizeof("INCREMENT")];
      char stringpool_str2661[sizeof("SQLEXCEPTION")];
      char stringpool_str2664[sizeof("UPD")];
      char stringpool_str2675[sizeof("VARCHAR")];
      char stringpool_str2676[sizeof("EXECUTE")];
      char stringpool_str2678[sizeof("TYPE")];
      char stringpool_str2680[sizeof("EXCEPT")];
      char stringpool_str2683[sizeof("INDEX")];
      char stringpool_str2687[sizeof("LONGTEXT")];
      char stringpool_str2691[sizeof("NULLIF")];
      char stringpool_str2695[sizeof("OVER")];
      char stringpool_str2699[sizeof("IDENTIFIED")];
      char stringpool_str2700[sizeof("SCHEMA")];
      char stringpool_str2709[sizeof("ADMIN")];
      char stringpool_str2719[sizeof("CHARACTERS")];
      char stringpool_str2721[sizeof("MAX")];
      char stringpool_str2723[sizeof("IO_AFTER_GTIDS")];
      char stringpool_str2725[sizeof("BEGIN_PARTITION")];
      char stringpool_str2728[sizeof("PERMANENT")];
      char stringpool_str2737[sizeof("END-EXEC")];
      char stringpool_str2739[sizeof("LOCALTIMESTAMP")];
      char stringpool_str2744[sizeof("UTC_DATE")];
      char stringpool_str2751[sizeof("EXTRACT")];
      char stringpool_str2766[sizeof("GENERAL")];
      char stringpool_str2767[sizeof("EXTERNAL")];
      char stringpool_str2771[sizeof("TSEQUAL")];
      char stringpool_str2773[sizeof("CHARACTER")];
      char stringpool_str2784[sizeof("RIGHT")];
      char stringpool_str2786[sizeof("LEVEL")];
      char stringpool_str2792[sizeof("UPDATE")];
      char stringpool_str2793[sizeof("PROC")];
      char stringpool_str2800[sizeof("RIGHTS")];
      char stringpool_str2806[sizeof("VOLATILE")];
      char stringpool_str2808[sizeof("PIECESIZE")];
      char stringpool_str2809[sizeof("DEPTH")];
      char stringpool_str2820[sizeof("SUBMULTISET")];
      char stringpool_str2827[sizeof("JSON_SCALAR")];
      char stringpool_str2829[sizeof("ROUND_HALF_EVEN")];
      char stringpool_str2832[sizeof("OF")];
      char stringpool_str2844[sizeof("DAY_SECOND")];
      char stringpool_str2853[sizeof("OBID")];
      char stringpool_str2856[sizeof("OFFSET")];
      char stringpool_str2857[sizeof("LISTAGG")];
      char stringpool_str2860[sizeof("VIRTUAL")];
      char stringpool_str2863[sizeof("UNPIVOT")];
      char stringpool_str2864[sizeof("HASHBUCKET")];
      char stringpool_str2869[sizeof("MINDEX")];
      char stringpool_str2871[sizeof("ENABLED")];
      char stringpool_str2872[sizeof("OFFSETS")];
      char stringpool_str2876[sizeof("PREPARE")];
      char stringpool_str2879[sizeof("ROUND_HALF_DOWN")];
      char stringpool_str2889[sizeof("DATABLOCKSIZE")];
      char stringpool_str2891[sizeof("BEGIN_FRAME")];
      char stringpool_str2893[sizeof("NTH_VALUE")];
      char stringpool_str2894[sizeof("LABEL")];
      char stringpool_str2896[sizeof("LAST_VALUE")];
      char stringpool_str2898[sizeof("DBCC")];
      char stringpool_str2931[sizeof("VARIADIC")];
      char stringpool_str2933[sizeof("JOURNAL")];
      char stringpool_str2936[sizeof("PARTIAL")];
      char stringpool_str2937[sizeof("GROUPING")];
      char stringpool_str2939[sizeof("OVERRIDE")];
      char stringpool_str2949[sizeof("FETCH")];
      char stringpool_str2954[sizeof("GROUP")];
      char stringpool_str2963[sizeof("EXCEPTION")];
      char stringpool_str2978[sizeof("ROWGUIDCOL")];
      char stringpool_str2980[sizeof("LIKE")];
      char stringpool_str2988[sizeof("SPECIFIC")];
      char stringpool_str3008[sizeof("UESCAPE")];
      char stringpool_str3009[sizeof("FIELDPROC")];
      char stringpool_str3024[sizeof("FASTEXPORT")];
      char stringpool_str3026[sizeof("DYNAMIC")];
      char stringpool_str3044[sizeof("HOLDLOCK")];
      char stringpool_str3050[sizeof("SYSTEM_USER")];
      char stringpool_str3065[sizeof("MAVG")];
      char stringpool_str3069[sizeof("ACCESSIBLE")];
      char stringpool_str3074[sizeof("TIMESTAMP")];
      char stringpool_str3084[sizeof("ISOBID")];
      char stringpool_str3090[sizeof("EQUALS")];
      char stringpool_str3093[sizeof("ROW_NUMBER")];
      char stringpool_str3098[sizeof("EXEC")];
      char stringpool_str3118[sizeof("XMLEXISTS")];
      char stringpool_str3144[sizeof("PERCENTILE_CONT")];
      char stringpool_str3146[sizeof("STDDEV_POP")];
      char stringpool_str3155[sizeof("MCHARACTERS")];
      char stringpool_str3157[sizeof("WLM")];
      char stringpool_str3175[sizeof("VARCHARACTER")];
      char stringpool_str3182[sizeof("OFFLINE")];
      char stringpool_str3189[sizeof("PERCENTILE_DISC")];
      char stringpool_str3194[sizeof("YEAR_MONTH")];
      char stringpool_str3207[sizeof("WHEN")];
      char stringpool_str3217[sizeof("LOGGING")];
      char stringpool_str3218[sizeof("SUBSTRING")];
      char stringpool_str3221[sizeof("REGR_SXX")];
      char stringpool_str3230[sizeof("SYSTEM_TIME")];
      char stringpool_str3231[sizeof("ABS")];
      char stringpool_str3234[sizeof("ABSENT")];
      char stringpool_str3236[sizeof("PROFILE")];
      char stringpool_str3239[sizeof("ROLLBACK")];
      char stringpool_str3242[sizeof("OPENXML")];
      char stringpool_str3243[sizeof("WHERE")];
      char stringpool_str3249[sizeof("SCRATCHPAD")];
      char stringpool_str3256[sizeof("ELSEIF")];
      char stringpool_str3261[sizeof("PCTFREE")];
      char stringpool_str3265[sizeof("TABLESAMPLE")];
      char stringpool_str3268[sizeof("EXCLUSIVE")];
      char stringpool_str3270[sizeof("ATOMIC")];
      char stringpool_str3276[sizeof("JSON_EXISTS")];
      char stringpool_str3278[sizeof("AVE")];
      char stringpool_str3281[sizeof("XMLCAST")];
      char stringpool_str3291[sizeof("IF")];
      char stringpool_str3298[sizeof("ABORT")];
      char stringpool_str3299[sizeof("ROWNUM")];
      char stringpool_str3300[sizeof("SYSTEM")];
      char stringpool_str3307[sizeof("STDDEV_SAMP")];
      char stringpool_str3309[sizeof("TRIM_ARRAY")];
      char stringpool_str3316[sizeof("ABSOLUTE")];
      char stringpool_str3324[sizeof("CHECKPOINT")];
      char stringpool_str3331[sizeof("TINYTEXT")];
      char stringpool_str3343[sizeof("ECHO")];
      char stringpool_str3346[sizeof("MAXIMUM")];
      char stringpool_str3348[sizeof("IMMEDIATE")];
      char stringpool_str3350[sizeof("ABORTSESSION")];
      char stringpool_str3367[sizeof("LONGBLOB")];
      char stringpool_str3379[sizeof("VERBOSE")];
      char stringpool_str3392[sizeof("FIRST_VALUE")];
      char stringpool_str3394[sizeof("CV")];
      char stringpool_str3396[sizeof("IDENTITY")];
      char stringpool_str3401[sizeof("JSON_TABLE_PRIMITIVE")];
      char stringpool_str3404[sizeof("NEW")];
      char stringpool_str3414[sizeof("ROW")];
      char stringpool_str3447[sizeof("DENY")];
      char stringpool_str3465[sizeof("DESTROY")];
      char stringpool_str3469[sizeof("WITHIN")];
      char stringpool_str3470[sizeof("SQLTEXT")];
      char stringpool_str3478[sizeof("PERM")];
      char stringpool_str3492[sizeof("STRAIGHT_JOIN")];
      char stringpool_str3494[sizeof("ORDINALITY")];
      char stringpool_str3508[sizeof("OLD_TABLE")];
      char stringpool_str3518[sizeof("OPENROWSET")];
      char stringpool_str3519[sizeof("TABLESPACE")];
      char stringpool_str3520[sizeof("WITHOUT")];
      char stringpool_str3521[sizeof("UPPER")];
      char stringpool_str3533[sizeof("MAXVALUE")];
      char stringpool_str3534[sizeof("DAY_MICROSECOND")];
      char stringpool_str3545[sizeof("INHERIT")];
      char stringpool_str3549[sizeof("VARBYTE")];
      char stringpool_str3553[sizeof("INTERVAL")];
      char stringpool_str3562[sizeof("HAVING")];
      char stringpool_str3570[sizeof("ASENSITIVE")];
      char stringpool_str3574[sizeof("RAW")];
      char stringpool_str3576[sizeof("PROPORTIONAL")];
      char stringpool_str3581[sizeof("PARTITIONING")];
      char stringpool_str3599[sizeof("UTC_TIME")];
      char stringpool_str3600[sizeof("DIV")];
      char stringpool_str3606[sizeof("DAY")];
      char stringpool_str3607[sizeof("CLOB")];
      char stringpool_str3609[sizeof("JSON_ARRAYAGG")];
      char stringpool_str3616[sizeof("JSON_OBJECT")];
      char stringpool_str3617[sizeof("STAY")];
      char stringpool_str3627[sizeof("POWER")];
      char stringpool_str3628[sizeof("ARRAY")];
      char stringpool_str3643[sizeof("SQL_BIG_RESULT")];
      char stringpool_str3653[sizeof("AGGREGATE")];
      char stringpool_str3664[sizeof("LOCKSIZE")];
      char stringpool_str3668[sizeof("PRESERVE")];
      char stringpool_str3671[sizeof("REGR_SXY")];
      char stringpool_str3688[sizeof("SQL_CALC_FOUND_ROWS")];
      char stringpool_str3709[sizeof("OVERLAPS")];
      char stringpool_str3711[sizeof("VALUE_OF")];
      char stringpool_str3713[sizeof("ATANH")];
      char stringpool_str3720[sizeof("SHUTDOWN")];
      char stringpool_str3743[sizeof("ARRAY_AGG")];
      char stringpool_str3746[sizeof("EXPLAIN")];
      char stringpool_str3750[sizeof("BUFFERPOOL")];
      char stringpool_str3761[sizeof("KEY")];
      char stringpool_str3763[sizeof("WHILE")];
      char stringpool_str3769[sizeof("MODIFY")];
      char stringpool_str3772[sizeof("ROUND_HALF_UP")];
      char stringpool_str3796[sizeof("UNIQUE")];
      char stringpool_str3797[sizeof("PTF")];
      char stringpool_str3798[sizeof("ASINH")];
      char stringpool_str3800[sizeof("PIVOT")];
      char stringpool_str3803[sizeof("OFF")];
      char stringpool_str3839[sizeof("GIVE")];
      char stringpool_str3852[sizeof("PRIVATE")];
      char stringpool_str3859[sizeof("OBJECT")];
      char stringpool_str3860[sizeof("BY")];
      char stringpool_str3875[sizeof("OBJECTS")];
      char stringpool_str3880[sizeof("ILIKE")];
      char stringpool_str3888[sizeof("SUBSTRING_REGEX")];
      char stringpool_str3891[sizeof("SEEK")];
      char stringpool_str3897[sizeof("COSH")];
      char stringpool_str3907[sizeof("JSON_VALUE")];
      char stringpool_str3940[sizeof("SQL_SMALL_RESULT")];
      char stringpool_str3946[sizeof("NEW_TABLE")];
      char stringpool_str3947[sizeof("AVERAGE")];
      char stringpool_str3951[sizeof("NO_WRITE_TO_BINLOG")];
      char stringpool_str3973[sizeof("ACOSH")];
      char stringpool_str3977[sizeof("ALLOW")];
      char stringpool_str3985[sizeof("SKIP")];
      char stringpool_str3987[sizeof("BLOB")];
      char stringpool_str3989[sizeof("FALLBACK")];
      char stringpool_str4006[sizeof("PLACING")];
      char stringpool_str4011[sizeof("TINYBLOB")];
      char stringpool_str4017[sizeof("EXP")];
      char stringpool_str4046[sizeof("OCCURRENCES_REGEX")];
      char stringpool_str4051[sizeof("COVAR_SAMP")];
      char stringpool_str4065[sizeof("UPPERCASE")];
      char stringpool_str4069[sizeof("SEARCH")];
      char stringpool_str4070[sizeof("UPDATETEXT")];
      char stringpool_str4073[sizeof("SEMANTICKEYPHRASETABLE")];
      char stringpool_str4077[sizeof("THRESHOLD")];
      char stringpool_str4080[sizeof("WIDTH_BUCKET")];
      char stringpool_str4082[sizeof("TANH")];
      char stringpool_str4086[sizeof("RANK")];
      char stringpool_str4095[sizeof("CONCURRENTLY")];
      char stringpool_str4097[sizeof("SINH")];
      char stringpool_str4113[sizeof("ADD_MONTHS")];
      char stringpool_str4120[sizeof("JSON_ARRAY")];
      char stringpool_str4121[sizeof("REGR_SYY")];
      char stringpool_str4126[sizeof("AUTHORIZATION")];
      char stringpool_str4130[sizeof("VARIABLE")];
      char stringpool_str4151[sizeof("DISK")];
      char stringpool_str4161[sizeof("GRAPHIC")];
      char stringpool_str4167[sizeof("AMP")];
      char stringpool_str4185[sizeof("VAR_POP")];
      char stringpool_str4190[sizeof("PRIVILEGES")];
      char stringpool_str4205[sizeof("AUX")];
      char stringpool_str4218[sizeof("SYMMETRIC")];
      char stringpool_str4236[sizeof("VAR_SAMP")];
      char stringpool_str4262[sizeof("HASH")];
      char stringpool_str4271[sizeof("PROGRAM")];
      char stringpool_str4272[sizeof("BOTH")];
      char stringpool_str4277[sizeof("VALIDPROC")];
      char stringpool_str4284[sizeof("OPTIONALLY")];
      char stringpool_str4293[sizeof("VARGRAPHIC")];
      char stringpool_str4297[sizeof("ANY_VALUE")];
      char stringpool_str4309[sizeof("BINARY")];
      char stringpool_str4311[sizeof("SYSTIMESTAMP")];
      char stringpool_str4315[sizeof("LIKE_REGEX")];
      char stringpool_str4333[sizeof("MEDIUMBLOB")];
      char stringpool_str4335[sizeof("CHECK")];
      char stringpool_str4364[sizeof("EMPTY")];
      char stringpool_str4379[sizeof("CHAR2HEXINT")];
      char stringpool_str4387[sizeof("COPY")];
      char stringpool_str4388[sizeof("LOCKING")];
      char stringpool_str4389[sizeof("JSON_OBJECTAGG")];
      char stringpool_str4425[sizeof("EVERY")];
      char stringpool_str4427[sizeof("PREVVAL")];
      char stringpool_str4433[sizeof("IO_BEFORE_GTIDS")];
      char stringpool_str4478[sizeof("AVG")];
      char stringpool_str4500[sizeof("BREADTH")];
      char stringpool_str4524[sizeof("GLOBAL")];
      char stringpool_str4532[sizeof("XMLNAMESPACES")];
      char stringpool_str4582[sizeof("SQLWARNING")];
      char stringpool_str4624[sizeof("CHARACTER_LENGTH")];
      char stringpool_str4667[sizeof("ONLY")];
      char stringpool_str4669[sizeof("POSTFIX")];
      char stringpool_str4670[sizeof("EQ")];
      char stringpool_str4693[sizeof("TRANSLATE_CHK")];
      char stringpool_str4702[sizeof("LC_CTYPE")];
      char stringpool_str4716[sizeof("BULK")];
      char stringpool_str4719[sizeof("PUBLIC")];
      char stringpool_str4727[sizeof("POSITION_REGEX")];
      char stringpool_str4762[sizeof("BACKUP")];
      char stringpool_str4808[sizeof("ASYMMETRIC")];
      char stringpool_str4869[sizeof("VARYING")];
      char stringpool_str4880[sizeof("REFRESH")];
      char stringpool_str4891[sizeof("ACCESS_LOCK")];
      char stringpool_str4916[sizeof("HASHAMP")];
      char stringpool_str4932[sizeof("WHENEVER")];
      char stringpool_str4976[sizeof("WINDOW")];
      char stringpool_str4990[sizeof("CURRENT_PATH")];
      char stringpool_str5002[sizeof("PACKAGE")];
      char stringpool_str5030[sizeof("SYNONYM")];
      char stringpool_str5040[sizeof("COVAR_POP")];
      char stringpool_str5045[sizeof("QUALIFY")];
      char stringpool_str5049[sizeof("CHAR_LENGTH")];
      char stringpool_str5050[sizeof("ANY")];
      char stringpool_str5058[sizeof("REGR_AVGX")];
      char stringpool_str5111[sizeof("LOCK")];
      char stringpool_str5135[sizeof("SUMMARY")];
      char stringpool_str5230[sizeof("WORK")];
      char stringpool_str5284[sizeof("UTC_TIMESTAMP")];
      char stringpool_str5302[sizeof("OPENQUERY")];
      char stringpool_str5319[sizeof("SHOW")];
      char stringpool_str5336[sizeof("SKEW")];
      char stringpool_str5338[sizeof("VIEW")];
      char stringpool_str5346[sizeof("WITH")];
      char stringpool_str5372[sizeof("UNKNOWN")];
      char stringpool_str5397[sizeof("EACH")];
      char stringpool_str5417[sizeof("TEMPORARY")];
      char stringpool_str5418[sizeof("PREFIX")];
      char stringpool_str5530[sizeof("FRAME_ROW")];
      char stringpool_str5566[sizeof("INITIALLY")];
      char stringpool_str5689[sizeof("JSON_QUERY")];
      char stringpool_str5718[sizeof("SECQTY")];
      char stringpool_str5721[sizeof("HIGH_PRIORITY")];
      char stringpool_str5755[sizeof("WITHIN_GROUP")];
      char stringpool_str5757[sizeof("HASHROW")];
      char stringpool_str5851[sizeof("PATH")];
      char stringpool_str5876[sizeof("LOCKMAX")];
      char stringpool_str5919[sizeof("UNLOCK")];
      char stringpool_str5950[sizeof("OCTET_LENGTH")];
      char stringpool_str5958[sizeof("REGR_AVGY")];
      char stringpool_str6145[sizeof("HASHBAKAMP")];
      char stringpool_str6236[sizeof("OVERLAY")];
      char stringpool_str6313[sizeof("ARRAY_MAX_CARDINALITY")];
      char stringpool_str6323[sizeof("PERCENT_RANK")];
      char stringpool_str6449[sizeof("PRIMARY")];
      char stringpool_str6688[sizeof("NOCHECK")];
      char stringpool_str6763[sizeof("LOW_PRIORITY")];
      char stringpool_str7060[sizeof("AUXILIARY")];
      char stringpool_str7164[sizeof("VARBINARY")];
      char stringpool_str7452[sizeof("PRIQTY")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "NE",
      "SET",
      "RET",
      "SS",
      "DO",
      "SETS",
      "ZONE",
      "NOT",
      "NO",
      "NONE",
      "CT",
      "CS",
      "MONTHS",
      "SESSION",
      "MODE",
      "COS",
      "RESUME",
      "RESTORE",
      "TO",
      "RESULT",
      "RESTART",
      "RUN",
      "MONSESSION",
      "STORES",
      "CONTENT",
      "RETURN",
      "ROUTINE",
      "RETURNS",
      "MOD",
      "CD",
      "CROSS",
      "MODULE",
      "COUNT",
      "SOURCE",
      "SETRESRATE",
      "CONTINUE",
      "DATE",
      "STATE",
      "STORED",
      "RENAME",
      "TRUE",
      "MONRESOURCE",
      "SETUSER",
      "SETSESSRATE",
      "STRUCTURE",
      "CASE",
      "CAST",
      "READS",
      "QUERYNO",
      "START",
      "CORR",
      "SESSION_USER",
      "CASE_N",
      "TAN",
      "DSSIZE",
      "CURRENT",
      "CURRENT_TIME",
      "READ",
      "CURRENT_TIMEZONE",
      "SIN",
      "HOST",
      "CREATE",
      "SENSITIVE",
      "MIN",
      "CONSTRUCTOR",
      "TREAT",
      "CURRENT_TRANSFORM_GROUP_FOR_TYPE",
      "CURSOR",
      "CURRENT_DATE",
      "DELETE",
      "TRAN",
      "CURRENT_ROLE",
      "SELECT",
      "RELEASE",
      "ROLE",
      "CONSTRAINT",
      "MINUTE",
      "CONSTRAINTS",
      "MINUS",
      "CONTAINS",
      "MINUTES",
      "TRUNCATE",
      "CLOSE",
      "RULE",
      "CLONE",
      "HOURS",
      "CONDITION",
      "BT",
      "MULTISET",
      "MONITOR",
      "CURRENT_SCHEMA",
      "HOUR",
      "CURRENT_USER",
      "YEARS",
      "HOUR_SECOND",
      "SECTION",
      "STANDARD",
      "SIZE",
      "COLUMN",
      "BUT",
      "CONCAT",
      "SECONDS",
      "RADIANS",
      "DATA",
      "RESOURCE",
      "CONNECT",
      "DOCUMENT",
      "YEAR",
      "NOAUDIT",
      "DATABASE",
      "SECOND",
      "DATABASES",
      "CONNECTION",
      "CLASS",
      "MLOAD",
      "CLUSTER",
      "DESTRUCTOR",
      "MODIFIES",
      "NTILE",
      "DELAYED",
      "HANDLER",
      "CLUSTERED",
      "MIDDLEINT",
      "MATCHES",
      "STATISTICS",
      "DEL",
      "TITLE",
      "NONCLUSTERED",
      "SEL",
      "FREE",
      "FREEZE",
      "SSL",
      "MERGE",
      "DEGREES",
      "BIT",
      "CONTAINSTABLE",
      "HOLD",
      "MACRO",
      "RAISERROR",
      "CASCADE",
      "RESTRICT",
      "DISTINCT",
      "COLLATE",
      "TRACE",
      "REGR_R2",
      "FOR",
      "KURTOSIS",
      "FOUND",
      "NULLS",
      "NESTED_TABLE_ID",
      "CASCADED",
      "DISABLE",
      "LE",
      "RECURSIVE",
      "LT",
      "CCSID",
      "RESIGNAL",
      "SOME",
      "RANGE",
      "LN",
      "LESS",
      "MICROSECONDS",
      "DISABLED",
      "COLLID",
      "DECLARE",
      "TRANSLATE",
      "REAL",
      "MICROSECOND",
      "DEC",
      "DESC",
      "MINUTE_SECOND",
      "ONE",
      "DUAL",
      "REPEAT",
      "QUANTILE",
      "ON",
      "FIRST",
      "MAINTAINED",
      "RESULT_SET_LOCATOR",
      "REGR_INTERCEPT",
      "DEALLOCATE",
      "CEIL",
      "FUSION",
      "OUT",
      "LEAST",
      "NATIONAL",
      "LAST",
      "ET",
      "NAMES",
      "STATEMENT",
      "CALLED",
      "MATERIALIZED",
      "ROLLFORWARD",
      "BOOLEAN",
      "SUCCEEDS",
      "DOMAIN",
      "ZEROFILL",
      "REFERENCES",
      "COALESCE",
      "OR",
      "COLLATION",
      "NAMED",
      "FLOOR",
      "OUTER",
      "MASTER_BIND",
      "DEREF",
      "CURRENT_LC_CTYPE",
      "NOTNULL",
      "DATEFORM",
      "STATIC",
      "LEAD",
      "LINES",
      "LOAD",
      "ORDER",
      "NULL",
      "LINENO",
      "END",
      "CURRVAL",
      "DEFERRED",
      "SUSPEND",
      "NATURAL",
      "TIME",
      "TRANSLATION",
      "TERMINATE",
      "FORCE",
      "COLUMN_VALUE",
      "FALSE",
      "TRIGGER",
      "STARTING",
      "FLOAT",
      "FLOAT8",
      "MINUTE_MICROSECOND",
      "FLOAT4",
      "REGR_COUNT",
      "FENCED",
      "TERMINATED",
      "DISCONNECT",
      "COLLECT",
      "RPAD",
      "SCROLL",
      "CURRENT_TIMESTAMP",
      "ERROR",
      "LINEAR",
      "FILE",
      "CAPTURE",
      "COLLECTION",
      "TRANSACTION",
      "SECURITYAUDIT",
      "CALL",
      "CUME_DIST",
      "ERASE",
      "DEFINE",
      "REPOVERRIDE",
      "SPOOL",
      "RANGE_N",
      "SECOND_MICROSECOND",
      "NEXT",
      "BEGIN",
      "ELSE",
      "FOREIGN",
      "OLD",
      "FILTER",
      "DIAGNOSTICS",
      "RUNNING",
      "SCOPE",
      "SUCCESSFUL",
      "IS",
      "INT2",
      "MEDIUMINT",
      "INT",
      "FINAL",
      "IN",
      "INT8",
      "INS",
      "SEPARATOR",
      "INSERT",
      "INT4",
      "INTERSECT",
      "INTO",
      "LOCALE",
      "FUNCTION",
      "STRING_CS",
      "INT3",
      "CURRENT_DEFAULT_TRANSFORM_GROUP",
      "BEFORE",
      "INT1",
      "LOG10",
      "INTERSECTION",
      "LOGON",
      "ITERATE",
      "TRAILING",
      "INNER",
      "INOUT",
      "NOCOMPRESS",
      "NORMALIZE",
      "FORMAT",
      "MAXEXTENTS",
      "CURRENT_CATALOG",
      "DEFAULT",
      "ONLINE",
      "LOCATORS",
      "SPACE",
      "INTEGER",
      "SEMANTICSIMILARITYTABLE",
      "READTEXT",
      "SEMANTICSIMILARITYDETAILSTABLE",
      "HOUR_MINUTE",
      "DECIMAL",
      "ROUND_UP",
      "BTRIM",
      "SIGNAL",
      "DEFERRABLE",
      "LOCATOR",
      "BIGINT",
      "RETURNING",
      "HOUR_MICROSECOND",
      "SMALLINT",
      "KILL",
      "ESCAPE",
      "WRITE",
      "MATCH_RECOGNIZE",
      "DECFLOAT",
      "SIMILAR",
      "STOGROUP",
      "CM",
      "WRITETEXT",
      "TIMEZONE_MINUTE",
      "AT",
      "INTEGERDATE",
      "ROUND_FLOOR",
      "LARGE",
      "AS",
      "CATALOG",
      "SUM",
      "COMMIT",
      "ERRORTABLES",
      "COMMENT",
      "MSUM",
      "INSTEAD",
      "FULL",
      "REPLACE",
      "LOCALTIME",
      "ARE",
      "DESCRIPTOR",
      "USE",
      "ESCAPED",
      "CSUM",
      "DOUBLE",
      "REVERT",
      "REVOKE",
      "SUBSET",
      "WAIT",
      "UNNEST",
      "AND",
      "CUBE",
      "TIMEZONE_HOUR",
      "END_FRAME",
      "LOCAL",
      "CORRESPONDING",
      "FREESPACE",
      "ZEROIFNULL",
      "ROWS",
      "ROWSET",
      "COMPUTE",
      "DIAGNOSTIC",
      "ADD",
      "RETRIEVE",
      "RECONFIGURE",
      "COMPRESS",
      "LEFT",
      "UNDO",
      "COMPLETION",
      "SYSDATE",
      "EDITPROC",
      "LATERAL",
      "ATAN2",
      "UNDER",
      "INITIATE",
      "ENCLOSED",
      "OPEN",
      "USER",
      "RANDOM",
      "ATAN",
      "MLINREG",
      "SUBSTR",
      "CONVERT",
      "TEXTSIZE",
      "MSUBSTR",
      "NUMPARTS",
      "WAITFOR",
      "SAVE",
      "SAMPLE",
      "CEILING",
      "REFERENCING",
      "XOR",
      "DROP",
      "CLASSIFIER",
      "TOP",
      "OUTPUT",
      "ROUND_CEILING",
      "LTRIM",
      "ASIN",
      "INSENSITIVE",
      "STEPINFO",
      "DBINFO",
      "GE",
      "DAYS",
      "NOWAIT",
      "REGR_SLOPE",
      "SPATIAL",
      "QUERY",
      "GT",
      "GET",
      "ASSERTION",
      "SQRT",
      "SIMPLE",
      "OMIT",
      "GO",
      "FREETEXT",
      "LPAD",
      "REF",
      "TRIM",
      "REQUEST",
      "GOTO",
      "MONTH",
      "MAP",
      "QUALIFIED",
      "UNION",
      "PER",
      "IGNORE",
      "PORTION",
      "NUMERIC",
      "INDICATOR",
      "AUDIT",
      "DETERMINISTIC",
      "LOG",
      "JSON",
      "REQUIRE",
      "ISOLATION",
      "READ_WRITE",
      "ALTER",
      "MATCH_NUMBER",
      "THEN",
      "LONG",
      "END_PARTITION",
      "UNSIGNED",
      "ROWID",
      "UID",
      "CURRENT_SERVER",
      "METHOD",
      "CURRENT_ROW",
      "NULLIFZERO",
      "PREORDER",
      "STYLE",
      "OPENDATASOURCE",
      "KEYS",
      "ANSIDATE",
      "ACOS",
      "OPTION",
      "ASSOCIATE",
      "LIMIT",
      "SUBSCRIBER",
      "GROUPS",
      "TINYINT",
      "ERRLVL",
      "INCONSISTENT",
      "PART",
      "STARTUP",
      "TRANSLATE_REGEX",
      "ISNULL",
      "ELEMENT",
      "KEEP",
      "UNDEFINED",
      "REPLICATION",
      "NEXTVAL",
      "PAD",
      "VOLUMES",
      "PATTERN",
      "SAVEPOINT",
      "ANALYSE",
      "OPERATION",
      "BYTE",
      "IDENTITY_INSERT",
      "CHANGE",
      "LAG",
      "SHARE",
      "SAMPLEID",
      "BYTES",
      "BETWEEN",
      "GREATEST",
      "GRANT",
      "REVALIDATE",
      "MINIMUM",
      "BYTEINT",
      "PASSWORD",
      "TABLE",
      "ALTERAND",
      "THAN",
      "PRINT",
      "DISTRIBUTED",
      "BROWSE",
      "FILLFACTOR",
      "GENERATED",
      "CHARS",
      "PSID",
      "VARIANT",
      "LEADING",
      "PADDED",
      "CONVERT_TABLE_HEADER",
      "LOADING",
      "FROM",
      "JOIN",
      "ROWCOUNT",
      "JAR",
      "CASESPECIFIC",
      "PRIOR",
      "MDIFF",
      "ORGANIZATION",
      "CHAR",
      "OUTFILE",
      "VALUE",
      "PERIOD",
      "FREETEXTTABLE",
      "UNTIL",
      "VALUES",
      "DESCRIBE",
      "ENCODING",
      "VCAT",
      "ROUND_DOWN",
      "ACTION",
      "INPUT",
      "TRY_CONVERT",
      "LANGUAGE",
      "PARAMETERS",
      "ERRORFILES",
      "ANALYZE",
      "HELP",
      "INITIALIZE",
      "PERCENT",
      "SEQUENCE",
      "ALIAS",
      "SQLSTATE",
      "MEDIUMTEXT",
      "PLAN",
      "PARTITION",
      "REGEXP",
      "INCLUSIVE",
      "NCLOB",
      "PARAMETER",
      "PROTECTION",
      "PRECEDES",
      "ENCRYPTION",
      "POSITION",
      "VERSIONING",
      "PARTITIONED",
      "MASTER_SSL_VERIFY_SERVER_CERT",
      "IDENTITYCOL",
      "RELATIVE",
      "USAGE",
      "INITIAL",
      "ALLOCATE",
      "ROLLUP",
      "ASUTIME",
      "ASC",
      "TBL_CS",
      "DISTINCTROW",
      "DISALLOW",
      "SECURITY",
      "PROCEDURE",
      "ACCESS",
      "ENDING",
      "DAY_HOUR",
      "JSON_TABLE",
      "EXIT",
      "DENSE_RANK",
      "DICTIONARY",
      "CARDINALITY",
      "ACCOUNT",
      "JSON_SERIALIZE",
      "SQL",
      "EXISTS",
      "CYCLE",
      "UC",
      "MEMBER",
      "OPTIMIZE",
      "MATCH",
      "RLIKE",
      "SCHEMAS",
      "USING",
      "LOWER",
      "SPECIFICTYPE",
      "OPTIMIZATION",
      "PURGE",
      "LOOP",
      "INFILE",
      "ALL",
      "NCHAR",
      "NUMBER",
      "FULLTEXT",
      "VARCHAR2",
      "BREAK",
      "DAY_MINUTE",
      "MLSLABEL",
      "LEAVE",
      "DUMP",
      "OPTIMIZER_COSTS",
      "SOUNDEX",
      "VALIDATE",
      "ARRAY_EXISTS",
      "PRECISION",
      "AFTER",
      "INCREMENT",
      "SQLEXCEPTION",
      "UPD",
      "VARCHAR",
      "EXECUTE",
      "TYPE",
      "EXCEPT",
      "INDEX",
      "LONGTEXT",
      "NULLIF",
      "OVER",
      "IDENTIFIED",
      "SCHEMA",
      "ADMIN",
      "CHARACTERS",
      "MAX",
      "IO_AFTER_GTIDS",
      "BEGIN_PARTITION",
      "PERMANENT",
      "END-EXEC",
      "LOCALTIMESTAMP",
      "UTC_DATE",
      "EXTRACT",
      "GENERAL",
      "EXTERNAL",
      "TSEQUAL",
      "CHARACTER",
      "RIGHT",
      "LEVEL",
      "UPDATE",
      "PROC",
      "RIGHTS",
      "VOLATILE",
      "PIECESIZE",
      "DEPTH",
      "SUBMULTISET",
      "JSON_SCALAR",
      "ROUND_HALF_EVEN",
      "OF",
      "DAY_SECOND",
      "OBID",
      "OFFSET",
      "LISTAGG",
      "VIRTUAL",
      "UNPIVOT",
      "HASHBUCKET",
      "MINDEX",
      "ENABLED",
      "OFFSETS",
      "PREPARE",
      "ROUND_HALF_DOWN",
      "DATABLOCKSIZE",
      "BEGIN_FRAME",
      "NTH_VALUE",
      "LABEL",
      "LAST_VALUE",
      "DBCC",
      "VARIADIC",
      "JOURNAL",
      "PARTIAL",
      "GROUPING",
      "OVERRIDE",
      "FETCH",
      "GROUP",
      "EXCEPTION",
      "ROWGUIDCOL",
      "LIKE",
      "SPECIFIC",
      "UESCAPE",
      "FIELDPROC",
      "FASTEXPORT",
      "DYNAMIC",
      "HOLDLOCK",
      "SYSTEM_USER",
      "MAVG",
      "ACCESSIBLE",
      "TIMESTAMP",
      "ISOBID",
      "EQUALS",
      "ROW_NUMBER",
      "EXEC",
      "XMLEXISTS",
      "PERCENTILE_CONT",
      "STDDEV_POP",
      "MCHARACTERS",
      "WLM",
      "VARCHARACTER",
      "OFFLINE",
      "PERCENTILE_DISC",
      "YEAR_MONTH",
      "WHEN",
      "LOGGING",
      "SUBSTRING",
      "REGR_SXX",
      "SYSTEM_TIME",
      "ABS",
      "ABSENT",
      "PROFILE",
      "ROLLBACK",
      "OPENXML",
      "WHERE",
      "SCRATCHPAD",
      "ELSEIF",
      "PCTFREE",
      "TABLESAMPLE",
      "EXCLUSIVE",
      "ATOMIC",
      "JSON_EXISTS",
      "AVE",
      "XMLCAST",
      "IF",
      "ABORT",
      "ROWNUM",
      "SYSTEM",
      "STDDEV_SAMP",
      "TRIM_ARRAY",
      "ABSOLUTE",
      "CHECKPOINT",
      "TINYTEXT",
      "ECHO",
      "MAXIMUM",
      "IMMEDIATE",
      "ABORTSESSION",
      "LONGBLOB",
      "VERBOSE",
      "FIRST_VALUE",
      "CV",
      "IDENTITY",
      "JSON_TABLE_PRIMITIVE",
      "NEW",
      "ROW",
      "DENY",
      "DESTROY",
      "WITHIN",
      "SQLTEXT",
      "PERM",
      "STRAIGHT_JOIN",
      "ORDINALITY",
      "OLD_TABLE",
      "OPENROWSET",
      "TABLESPACE",
      "WITHOUT",
      "UPPER",
      "MAXVALUE",
      "DAY_MICROSECOND",
      "INHERIT",
      "VARBYTE",
      "INTERVAL",
      "HAVING",
      "ASENSITIVE",
      "RAW",
      "PROPORTIONAL",
      "PARTITIONING",
      "UTC_TIME",
      "DIV",
      "DAY",
      "CLOB",
      "JSON_ARRAYAGG",
      "JSON_OBJECT",
      "STAY",
      "POWER",
      "ARRAY",
      "SQL_BIG_RESULT",
      "AGGREGATE",
      "LOCKSIZE",
      "PRESERVE",
      "REGR_SXY",
      "SQL_CALC_FOUND_ROWS",
      "OVERLAPS",
      "VALUE_OF",
      "ATANH",
      "SHUTDOWN",
      "ARRAY_AGG",
      "EXPLAIN",
      "BUFFERPOOL",
      "KEY",
      "WHILE",
      "MODIFY",
      "ROUND_HALF_UP",
      "UNIQUE",
      "PTF",
      "ASINH",
      "PIVOT",
      "OFF",
      "GIVE",
      "PRIVATE",
      "OBJECT",
      "BY",
      "OBJECTS",
      "ILIKE",
      "SUBSTRING_REGEX",
      "SEEK",
      "COSH",
      "JSON_VALUE",
      "SQL_SMALL_RESULT",
      "NEW_TABLE",
      "AVERAGE",
      "NO_WRITE_TO_BINLOG",
      "ACOSH",
      "ALLOW",
      "SKIP",
      "BLOB",
      "FALLBACK",
      "PLACING",
      "TINYBLOB",
      "EXP",
      "OCCURRENCES_REGEX",
      "COVAR_SAMP",
      "UPPERCASE",
      "SEARCH",
      "UPDATETEXT",
      "SEMANTICKEYPHRASETABLE",
      "THRESHOLD",
      "WIDTH_BUCKET",
      "TANH",
      "RANK",
      "CONCURRENTLY",
      "SINH",
      "ADD_MONTHS",
      "JSON_ARRAY",
      "REGR_SYY",
      "AUTHORIZATION",
      "VARIABLE",
      "DISK",
      "GRAPHIC",
      "AMP",
      "VAR_POP",
      "PRIVILEGES",
      "AUX",
      "SYMMETRIC",
      "VAR_SAMP",
      "HASH",
      "PROGRAM",
      "BOTH",
      "VALIDPROC",
      "OPTIONALLY",
      "VARGRAPHIC",
      "ANY_VALUE",
      "BINARY",
      "SYSTIMESTAMP",
      "LIKE_REGEX",
      "MEDIUMBLOB",
      "CHECK",
      "EMPTY",
      "CHAR2HEXINT",
      "COPY",
      "LOCKING",
      "JSON_OBJECTAGG",
      "EVERY",
      "PREVVAL",
      "IO_BEFORE_GTIDS",
      "AVG",
      "BREADTH",
      "GLOBAL",
      "XMLNAMESPACES",
      "SQLWARNING",
      "CHARACTER_LENGTH",
      "ONLY",
      "POSTFIX",
      "EQ",
      "TRANSLATE_CHK",
      "LC_CTYPE",
      "BULK",
      "PUBLIC",
      "POSITION_REGEX",
      "BACKUP",
      "ASYMMETRIC",
      "VARYING",
      "REFRESH",
      "ACCESS_LOCK",
      "HASHAMP",
      "WHENEVER",
      "WINDOW",
      "CURRENT_PATH",
      "PACKAGE",
      "SYNONYM",
      "COVAR_POP",
      "QUALIFY",
      "CHAR_LENGTH",
      "ANY",
      "REGR_AVGX",
      "LOCK",
      "SUMMARY",
      "WORK",
      "UTC_TIMESTAMP",
      "OPENQUERY",
      "SHOW",
      "SKEW",
      "VIEW",
      "WITH",
      "UNKNOWN",
      "EACH",
      "TEMPORARY",
      "PREFIX",
      "FRAME_ROW",
      "INITIALLY",
      "JSON_QUERY",
      "SECQTY",
      "HIGH_PRIORITY",
      "WITHIN_GROUP",
      "HASHROW",
      "PATH",
      "LOCKMAX",
      "UNLOCK",
      "OCTET_LENGTH",
      "REGR_AVGY",
      "HASHBAKAMP",
      "OVERLAY",
      "ARRAY_MAX_CARDINALITY",
      "PERCENT_RANK",
      "PRIMARY",
      "NOCHECK",
      "LOW_PRIORITY",
      "AUXILIARY",
      "VARBINARY",
      "PRIQTY"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str17,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str18,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str23,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str27,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str32,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str34,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str39,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str47,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str62,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str72,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str77,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str84,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str88,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str107,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str127,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str184,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str195,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str196,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str206,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str209,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str211,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str217,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str226,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str239,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str249,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str250,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str252,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str260,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str264,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str267,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str271,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str278,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str281,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str282,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str287,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str289,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str291,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str293,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str294,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str296,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str299,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str303,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str306,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str320,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str322,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str331,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str332,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str336,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str344,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str347,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str351,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str357,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str359,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str360,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str361,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str366,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str370,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str373,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str382,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str408,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str410,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str414,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str415,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str420,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str429,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str442,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str443,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str452,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str469,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str474,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str482,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str490,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str491,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str492,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str498,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str499,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str501,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str513,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str526,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str527,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str532,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str534,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str538,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str542,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str543,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str544,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str547,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str548,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str556,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str569,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str570,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str580,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str590,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str597,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str605,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str608,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str610,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str612,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str632,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str634,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str644,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str652,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str655,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str663,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str665,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str667,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str668,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str669,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str671,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str678,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str685,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str692,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str698,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str703,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str704,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str705,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str719,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str722,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str723,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str728,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str737,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str745,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str747,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str748,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str753,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str755,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str760,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str770,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str773,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str777,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str782,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str784,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str792,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str800,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str803,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str804,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str805,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str812,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str814,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str822,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str828,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str831,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str832,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str839,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str849,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str851,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str863,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str874,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str883,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str908,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str909,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str916,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str918,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str922,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str940,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str950,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str958,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str964,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str965,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str969,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str971,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str973,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str975,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str978,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str979,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str982,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str985,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str989,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str991,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str992,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1001,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1002,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1003,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1006,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1008,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1015,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1018,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1022,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1024,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1025,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1030,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1035,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1036,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1040,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1046,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1047,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1048,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1056,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1059,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1065,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1074,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1075,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1079,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1086,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1088,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1092,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1093,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1097,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1107,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1109,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1111,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1115,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1117,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1120,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1127,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1128,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1130,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1136,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1143,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1146,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1150,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1151,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1165,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1170,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1172,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1179,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1181,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1182,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1185,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1186,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1189,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1192,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1200,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1211,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1213,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1219,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1224,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1225,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1231,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1246,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1260,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1287,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1288,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1293,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1305,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1314,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1317,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1323,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1326,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1331,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1332,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1345,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1365,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1371,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1373,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1374,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1377,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1380,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1381,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1383,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1387,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1389,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1390,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1393,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1398,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1403,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1406,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1408,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1409,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1413,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1416,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1421,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1423,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1425,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1426,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1430,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1431,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1443,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1449,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1459,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1460,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1469,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1476,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1484,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1490,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1492,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1496,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1498,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1505,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1506,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1508,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1512,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1515,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1521,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1527,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1528,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1540,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1541,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1550,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1552,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1556,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1564,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1571,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1573,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1579,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1591,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1594,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1600,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1603,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1607,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1608,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1612,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1613,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1615,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1616,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1620,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1621,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1625,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1626,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1627,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1638,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1641,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1646,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1657,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1659,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1661,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1664,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1667,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1669,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1672,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1675,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1684,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1692,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1694,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1695,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1697,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1702,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1705,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1708,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1717,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1722,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1728,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1733,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1734,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1735,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1738,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1739,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1740,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1742,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1744,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1747,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1750,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1757,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1759,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1761,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1763,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1764,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1770,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1780,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1791,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1798,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1802,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1804,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1806,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1812,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1813,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1814,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1815,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1816,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1818,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1822,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1825,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1833,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1837,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1841,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1843,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1851,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1855,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1856,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1857,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1861,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1862,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1864,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1865,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1868,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1871,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1873,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1890,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1903,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1905,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1908,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1910,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1911,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1913,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1914,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1915,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1917,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1919,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1921,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1922,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1923,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1933,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1936,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1939,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1941,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1942,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1949,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1953,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1954,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1956,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1963,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1964,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1973,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1974,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1976,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1982,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1985,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1986,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1987,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1988,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1989,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1993,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1998,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2000,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2001,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2003,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2008,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2009,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2012,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2013,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2014,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2023,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2024,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2028,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2029,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2030,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2035,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2039,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2040,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2052,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2059,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2064,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2068,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2072,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2073,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2076,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2078,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2080,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2084,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2085,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2086,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2096,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2101,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2103,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2107,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2109,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2110,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2112,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2116,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2126,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2127,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2135,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2136,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2140,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2141,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2144,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2155,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2159,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2164,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2165,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2172,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2174,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2176,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2177,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2181,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2182,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2184,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2187,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2188,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2194,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2195,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2209,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2210,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2213,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2214,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2218,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2220,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2227,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2230,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2241,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2242,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2244,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2245,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2246,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2249,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2252,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2254,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2265,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2267,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2268,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2272,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2273,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2275,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2277,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2281,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2294,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2307,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2308,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2312,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2328,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2330,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2334,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2345,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2353,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2354,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2360,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2366,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2369,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2374,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2376,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2377,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2379,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2382,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2384,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2388,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2393,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2396,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2398,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2399,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2408,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2409,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2412,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2415,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2422,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2423,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2430,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2435,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2445,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2449,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2461,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2466,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2467,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2471,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2476,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2477,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2480,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2484,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2486,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2487,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2488,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2490,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2496,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2497,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2501,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2503,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2506,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2509,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2510,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2511,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2515,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2517,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2525,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2529,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2533,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2535,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2538,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2539,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2541,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2546,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2556,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2558,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2567,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2577,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2589,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2594,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2595,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2597,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2599,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2605,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2607,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2611,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2616,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2624,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2627,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2631,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2634,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2635,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2640,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2641,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2646,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2648,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2649,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2658,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2661,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2664,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2675,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2676,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2678,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2680,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2683,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2687,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2691,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2695,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2699,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2700,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2709,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2719,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2721,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2723,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2725,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2728,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2737,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2739,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2744,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2751,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2766,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2767,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2771,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2773,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2784,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2786,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2792,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2793,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2800,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2806,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2808,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2809,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2820,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2827,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2829,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2832,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2844,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2853,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2856,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2857,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2860,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2863,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2864,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2869,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2871,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2872,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2876,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2879,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2889,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2891,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2893,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2894,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2896,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2898,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2931,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2933,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2936,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2937,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2939,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2949,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2954,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2963,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2978,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2980,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2988,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3008,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3009,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3024,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3026,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3044,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3050,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3065,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3069,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3074,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3084,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3090,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3093,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3098,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3118,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3144,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3146,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3155,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3157,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3175,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3182,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3189,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3194,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3207,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3217,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3218,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3221,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3230,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3231,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3234,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3236,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3239,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3242,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3243,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3249,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3256,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3261,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3265,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3268,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3270,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3276,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3278,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3281,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3291,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3298,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3299,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3300,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3307,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3309,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3316,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3324,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3331,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3343,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3346,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3348,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3350,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3367,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3379,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3392,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3394,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3396,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3401,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3404,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3414,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3447,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3465,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3469,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3470,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3478,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3492,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3494,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3508,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3518,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3519,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3520,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3521,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3533,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3534,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3545,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3549,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3553,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3562,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3570,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3574,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3576,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3581,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3599,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3600,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3606,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3607,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3609,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3616,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3617,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3627,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3628,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3643,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3653,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3664,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3668,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3671,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3688,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3709,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3711,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3713,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3720,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3743,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3746,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3750,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3761,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3763,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3769,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3772,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3796,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3797,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3798,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3800,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3803,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3839,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3852,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3859,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3860,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3875,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3880,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3888,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3891,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3897,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3907,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3940,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3946,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3947,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3951,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3973,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3977,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3985,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3987,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3989,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4006,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4011,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4017,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4046,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4051,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4065,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4069,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4070,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4073,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4077,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4080,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4082,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4086,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4095,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4097,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4113,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4120,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4121,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4126,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4130,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4151,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4161,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4167,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4185,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4190,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4205,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4218,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4236,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4262,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4271,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4272,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4277,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4284,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4293,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4297,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4309,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4311,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4315,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4333,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4335,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4364,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4379,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4387,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4388,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4389,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4425,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4427,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4433,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4478,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4500,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4524,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4532,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4582,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4624,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4667,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4669,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4670,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4693,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4702,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4716,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4719,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4727,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4762,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4808,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4869,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4880,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4891,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4916,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4932,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4976,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4990,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5002,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5030,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5040,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5045,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5049,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5050,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5058,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5111,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5135,
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
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5230,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5284,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5302,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5319,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5336,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5338,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5346,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5372,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5397,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5417,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5418,
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
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5530,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5566,
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
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5689,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5718,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5721,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5755,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5757,
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
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5851,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5876,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5919,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5950,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5958,
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
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6145,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6236,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6313,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6323,
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
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6449,
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
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6688,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6763,
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
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7060,
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
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7164,
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
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str7452
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
