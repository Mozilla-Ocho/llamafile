/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_pascal_builtin.c llamafile/is_keyword_pascal_builtin.gperf  */
/* Computed positions: -k'1-6,8-9,12,$' */

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

#line 1 "llamafile/is_keyword_pascal_builtin.gperf"

#include <string.h>
#include <libc/str/tab.h>
#define GPERF_DOWNCASE

#define TOTAL_KEYWORDS 520
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 28
#define MIN_HASH_VALUE 10
#define MAX_HASH_VALUE 3675
/* maximum key range = 3666, duplicates = 0 */

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
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,    0,    0,
         0,    0,   20, 3676,   20, 3676,    5,    0, 3676, 3676,
      3676, 3676, 3676, 3676, 3676,   20,  990,   15,   35,    0,
        70,  535,  855,   60,  905,   50,  105,  390,   55,   30,
       335, 1011,    0,    5,    0,  210,  160,  975,  940,  465,
        15, 3676, 3676, 3676, 3676, 3676, 3676,   20,  990,   15,
        35,    0,   70,  535,  855,   60,  905,   50,  105,  390,
        55,   30,  335, 1011,    0,    5,    0,  210,  160,  975,
       940,  465,   15, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676, 3676,
      3676, 3676, 3676, 3676, 3676, 3676, 3676
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
        hval += asso_values[(unsigned char)str[7]];
      /*FALLTHROUGH*/
      case 7:
      case 6:
        hval += asso_values[(unsigned char)str[5]+1];
      /*FALLTHROUGH*/
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
        hval += asso_values[(unsigned char)str[1]];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const char *
is_keyword_pascal_builtin (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str10[sizeof("reset")];
      char stringpool_str19[sizeof("rect")];
      char stringpool_str30[sizeof("erase")];
      char stringpool_str38[sizeof("sec")];
      char stringpool_str44[sizeof("strtodate")];
      char stringpool_str48[sizeof("cot")];
      char stringpool_str53[sizeof("csc")];
      char stringpool_str58[sizeof("cos")];
      char stringpool_str59[sizeof("date")];
      char stringpool_str68[sizeof("dec")];
      char stringpool_str79[sizeof("strtocurr")];
      char stringpool_str91[sizeof("arccos")];
      char stringpool_str92[sizeof("arctan2")];
      char stringpool_str94[sizeof("read")];
      char stringpool_str96[sizeof("arcsec")];
      char stringpool_str101[sizeof("strend")];
      char stringpool_str103[sizeof("ord")];
      char stringpool_str109[sizeof("seek")];
      char stringpool_str111[sizeof("arccsc")];
      char stringpool_str118[sizeof("int")];
      char stringpool_str124[sizeof("frac")];
      char stringpool_str133[sizeof("tan")];
      char stringpool_str138[sizeof("odd")];
      char stringpool_str145[sizeof("strdispose")];
      char stringpool_str146[sizeof("arctan")];
      char stringpool_str148[sizeof("inc")];
      char stringpool_str153[sizeof("inttostr")];
      char stringpool_str160[sizeof("close")];
      char stringpool_str163[sizeof("strrscan")];
      char stringpool_str166[sizeof("forcedirectories")];
      char stringpool_str173[sizeof("eof")];
      char stringpool_str174[sizeof("createdir")];
      char stringpool_str176[sizeof("sincos")];
      char stringpool_str178[sizeof("sin")];
      char stringpool_str180[sizeof("cotan")];
      char stringpool_str187[sizeof("strtodatedef")];
      char stringpool_str190[sizeof("slice")];
      char stringpool_str191[sizeof("arcsin")];
      char stringpool_str197[sizeof("interestrate")];
      char stringpool_str198[sizeof("continue")];
      char stringpool_str201[sizeof("strlen")];
      char stringpool_str215[sizeof("decodedate")];
      char stringpool_str216[sizeof("delete")];
      char stringpool_str219[sizeof("randomize")];
      char stringpool_str222[sizeof("strtocurrdef")];
      char stringpool_str233[sizeof("diskfree")];
      char stringpool_str235[sizeof("encodedate")];
      char stringpool_str240[sizeof("floor")];
      char stringpool_str241[sizeof("addterminateproc")];
      char stringpool_str242[sizeof("strlfmt")];
      char stringpool_str246[sizeof("assert")];
      char stringpool_str249[sizeof("eoln")];
      char stringpool_str250[sizeof("radtocycle")];
      char stringpool_str251[sizeof("readln")];
      char stringpool_str255[sizeof("isnan")];
      char stringpool_str256[sizeof("strcat")];
      char stringpool_str263[sizeof("ioresult")];
      char stringpool_str264[sizeof("succ")];
      char stringpool_str278[sizeof("runerror")];
      char stringpool_str281[sizeof("randomrange")];
      char stringpool_str285[sizeof("filecreate")];
      char stringpool_str286[sizeof("teststreamformat")];
      char stringpool_str287[sizeof("initinheritedcomponent")];
      char stringpool_str289[sizeof("ceil")];
      char stringpool_str296[sizeof("arccot")];
      char stringpool_str298[sizeof("variance")];
      char stringpool_str299[sizeof("strtobool")];
      char stringpool_str300[sizeof("trunc")];
      char stringpool_str302[sizeof("setvariantmanager")];
      char stringpool_str311[sizeof("secant")];
      char stringpool_str325[sizeof("internalrateofreturn")];
      char stringpool_str336[sizeof("insert")];
      char stringpool_str337[sizeof("strtobooldef")];
      char stringpool_str341[sizeof("filesetattr")];
      char stringpool_str350[sizeof("freeandnil")];
      char stringpool_str351[sizeof("concat")];
      char stringpool_str353[sizeof("findnext")];
      char stringpool_str354[sizeof("olestrtostrvar")];
      char stringpool_str363[sizeof("setcurrentdir")];
      char stringpool_str370[sizeof("round")];
      char stringpool_str371[sizeof("strpas")];
      char stringpool_str376[sizeof("filesetdate")];
      char stringpool_str378[sizeof("varclear")];
      char stringpool_str381[sizeof("strpos")];
      char stringpool_str383[sizeof("fileread")];
      char stringpool_str385[sizeof("deletefile")];
      char stringpool_str390[sizeof("utf8encode")];
      char stringpool_str399[sizeof("datetostr")];
      char stringpool_str409[sizeof("pred")];
      char stringpool_str412[sizeof("loadstr")];
      char stringpool_str418[sizeof("fileseek")];
      char stringpool_str420[sizeof("createguid")];
      char stringpool_str423[sizeof("fillchar")];
      char stringpool_str426[sizeof("ensurerange")];
      char stringpool_str435[sizeof("startclassgroup")];
      char stringpool_str438[sizeof("strtodatetime")];
      char stringpool_str441[sizeof("ansistrscan")];
      char stringpool_str444[sizeof("linestart")];
      char stringpool_str446[sizeof("floattotext")];
      char stringpool_str447[sizeof("strcomp")];
      char stringpool_str450[sizeof("floattostr")];
      char stringpool_str451[sizeof("iszero")];
      char stringpool_str452[sizeof("include")];
      char stringpool_str457[sizeof("pi")];
      char stringpool_str465[sizeof("utf8decode")];
      char stringpool_str467[sizeof("seekeof")];
      char stringpool_str472[sizeof("dispose")];
      char stringpool_str476[sizeof("fmtstr")];
      char stringpool_str480[sizeof("disposestr")];
      char stringpool_str482[sizeof("ansistrrscan")];
      char stringpool_str485[sizeof("identtoint")];
      char stringpool_str488[sizeof("unicodetoutf8")];
      char stringpool_str490[sizeof("rmdir")];
      char stringpool_str493[sizeof("sametext")];
      char stringpool_str496[sizeof("raiselastoserror")];
      char stringpool_str501[sizeof("collectionsequal")];
      char stringpool_str502[sizeof("ansistrlower")];
      char stringpool_str505[sizeof("floattodatetime")];
      char stringpool_str508[sizeof("seekeoln")];
      char stringpool_str510[sizeof("ansistrlastchar")];
      char stringpool_str511[sizeof("strtodatetimedef")];
      char stringpool_str517[sizeof("trystrtotime")];
      char stringpool_str519[sizeof("floattotextfmt")];
      char stringpool_str521[sizeof("floattostrf")];
      char stringpool_str524[sizeof("mean")];
      char stringpool_str532[sizeof("trystrtocurr")];
      char stringpool_str535[sizeof("ansidequotedstr")];
      char stringpool_str540[sizeof("mkdir")];
      char stringpool_str541[sizeof("rename")];
      char stringpool_str546[sizeof("setprecisionmode")];
      char stringpool_str550[sizeof("netpresentvalue")];
      char stringpool_str551[sizeof("findresourcehinstance")];
      char stringpool_str552[sizeof("trystrtodate")];
      char stringpool_str554[sizeof("raiselastwin32error")];
      char stringpool_str556[sizeof("trystrtodatetime")];
      char stringpool_str558[sizeof("stralloc")];
      char stringpool_str560[sizeof("ansitoutf8")];
      char stringpool_str561[sizeof("replacedate")];
      char stringpool_str562[sizeof("presentvalue")];
      char stringpool_str563[sizeof("min")];
      char stringpool_str566[sizeof("replacetime")];
      char stringpool_str569[sizeof("currtostr")];
      char stringpool_str576[sizeof("trystrtoint")];
      char stringpool_str577[sizeof("roundto")];
      char stringpool_str578[sizeof("includetrailingpathdelimiter")];
      char stringpool_str584[sizeof("move")];
      char stringpool_str589[sizeof("interlockedexchange")];
      char stringpool_str591[sizeof("random")];
      char stringpool_str598[sizeof("ansilowercase")];
      char stringpool_str603[sizeof("tryencodedate")];
      char stringpool_str606[sizeof("ansilowercasefilename")];
      char stringpool_str608[sizeof("trystrtofloat")];
      char stringpool_str618[sizeof("trystrtoint64")];
      char stringpool_str625[sizeof("interlockeddecrement")];
      char stringpool_str627[sizeof("interlockedexchangeadd")];
      char stringpool_str630[sizeof("strtofloat")];
      char stringpool_str632[sizeof("setroundmode")];
      char stringpool_str633[sizeof("trimleft")];
      char stringpool_str639[sizeof("findclose")];
      char stringpool_str640[sizeof("currtostrf")];
      char stringpool_str641[sizeof("getdir")];
      char stringpool_str644[sizeof("strtotime")];
      char stringpool_str645[sizeof("decodetime")];
      char stringpool_str649[sizeof("findclass")];
      char stringpool_str650[sizeof("interlockedincrement")];
      char stringpool_str654[sizeof("fileclose")];
      char stringpool_str655[sizeof("pucs4chars")];
      char stringpool_str657[sizeof("gettime")];
      char stringpool_str660[sizeof("ispathdelimiter")];
      char stringpool_str661[sizeof("upcase")];
      char stringpool_str664[sizeof("assignstr")];
      char stringpool_str665[sizeof("encodetime")];
      char stringpool_str671[sizeof("floattocurr")];
      char stringpool_str672[sizeof("getlasterror")];
      char stringpool_str674[sizeof("log2")];
      char stringpool_str675[sizeof("log10")];
      char stringpool_str681[sizeof("strfmt")];
      char stringpool_str682[sizeof("getformatsettings")];
      char stringpool_str683[sizeof("degtorad")];
      char stringpool_str693[sizeof("meanandstddev")];
      char stringpool_str696[sizeof("futurevalue")];
      char stringpool_str698[sizeof("lastdelimiter")];
      char stringpool_str703[sizeof("strtofloatdef")];
      char stringpool_str708[sizeof("findclasshinstance")];
      char stringpool_str710[sizeof("renamefile")];
      char stringpool_str714[sizeof("sign")];
      char stringpool_str716[sizeof("divmod")];
      char stringpool_str719[sizeof("removedir")];
      char stringpool_str720[sizeof("fileexists")];
      char stringpool_str721[sizeof("sizeof")];
      char stringpool_str726[sizeof("format")];
      char stringpool_str731[sizeof("currentyear")];
      char stringpool_str733[sizeof("assigned")];
      char stringpool_str735[sizeof("ansistrpos")];
      char stringpool_str746[sizeof("ansistrcomp")];
      char stringpool_str750[sizeof("utf8toansi")];
      char stringpool_str754[sizeof("appendstr")];
      char stringpool_str765[sizeof("degtocycle")];
      char stringpool_str773[sizeof("paramstr")];
      char stringpool_str783[sizeof("utf8tounicode")];
      char stringpool_str784[sizeof("logn")];
      char stringpool_str785[sizeof("sleep")];
      char stringpool_str786[sizeof("append")];
      char stringpool_str787[sizeof("strtotimedef")];
      char stringpool_str788[sizeof("callterminateprocs")];
      char stringpool_str789[sizeof("olestrtostring")];
      char stringpool_str794[sizeof("timetostr")];
      char stringpool_str804[sizeof("closefile")];
      char stringpool_str806[sizeof("comparetext")];
      char stringpool_str810[sizeof("comparestr")];
      char stringpool_str814[sizeof("formatdatetime")];
      char stringpool_str820[sizeof("filesetreadonly")];
      char stringpool_str822[sizeof("ansipos")];
      char stringpool_str830[sizeof("assignfile")];
      char stringpool_str831[sizeof("outofmemoryerror")];
      char stringpool_str832[sizeof("getvariantmanager")];
      char stringpool_str838[sizeof("getlocaleformatsettings")];
      char stringpool_str844[sizeof("trim")];
      char stringpool_str845[sizeof("readcomponentresfile")];
      char stringpool_str846[sizeof("readcomponentres")];
      char stringpool_str848[sizeof("registerclass")];
      char stringpool_str849[sizeof("fileisreadonly")];
      char stringpool_str850[sizeof("registerclasses")];
      char stringpool_str853[sizeof("registerclassalias")];
      char stringpool_str860[sizeof("ansicomparetext")];
      char stringpool_str864[sizeof("ansicomparestr")];
      char stringpool_str865[sizeof("directoryexists")];
      char stringpool_str866[sizeof("formatfloat")];
      char stringpool_str869[sizeof("norm")];
      char stringpool_str871[sizeof("filegetattr")];
      char stringpool_str873[sizeof("chr")];
      char stringpool_str875[sizeof("cycletorad")];
      char stringpool_str890[sizeof("syserrormessage")];
      char stringpool_str893[sizeof("getcurrentdir")];
      char stringpool_str895[sizeof("decodedatefully")];
      char stringpool_str896[sizeof("registerintegerconsts")];
      char stringpool_str906[sizeof("filegetdate")];
      char stringpool_str912[sizeof("ansistrupper")];
      char stringpool_str917[sizeof("filepos")];
      char stringpool_str925[sizeof("interestpayment")];
      char stringpool_str927[sizeof("freemem")];
      char stringpool_str929[sizeof("uppercase")];
      char stringpool_str934[sizeof("ansicomparefilename")];
      char stringpool_str936[sizeof("sumint")];
      char stringpool_str937[sizeof("arcsech")];
      char stringpool_str938[sizeof("supports")];
      char stringpool_str940[sizeof("formatcurr")];
      char stringpool_str942[sizeof("arccosh")];
      char stringpool_str947[sizeof("arctanh")];
      char stringpool_str948[sizeof("strtoint")];
      char stringpool_str949[sizeof("registernoicon")];
      char stringpool_str952[sizeof("arccsch")];
      char stringpool_str957[sizeof("sumofsquares")];
      char stringpool_str958[sizeof("typeinfo")];
      char stringpool_str966[sizeof("minintvalue")];
      char stringpool_str970[sizeof("chdir")];
      char stringpool_str973[sizeof("datetimetostr")];
      char stringpool_str978[sizeof("periodpayment")];
      char stringpool_str979[sizeof("floattodecimal")];
      char stringpool_str984[sizeof("halt")];
      char stringpool_str986[sizeof("setmemorymanager")];
      char stringpool_str987[sizeof("comparevalue")];
      char stringpool_str989[sizeof("enumresourcemodules")];
      char stringpool_str990[sizeof("strtoint64")];
      char stringpool_str992[sizeof("arcsinh")];
      char stringpool_str993[sizeof("tryencodetime")];
      char stringpool_str998[sizeof("sum")];
      char stringpool_str1003[sizeof("unloadpackage")];
      char stringpool_str1004[sizeof("exit")];
      char stringpool_str1013[sizeof("tryfloattodatetime")];
      char stringpool_str1019[sizeof("sqr")];
      char stringpool_str1020[sizeof("sqrt")];
      char stringpool_str1023[sizeof("abs")];
      char stringpool_str1024[sizeof("gradtorad")];
      char stringpool_str1030[sizeof("systemtimetodatetime")];
      char stringpool_str1038[sizeof("datetimetofiledate")];
      char stringpool_str1039[sizeof("radtograd")];
      char stringpool_str1040[sizeof("write")];
      char stringpool_str1041[sizeof("msecstotimestamp")];
      char stringpool_str1043[sizeof("strtoint64def")];
      char stringpool_str1045[sizeof("abort")];
      char stringpool_str1046[sizeof("newstr")];
      char stringpool_str1049[sizeof("endthread")];
      char stringpool_str1051[sizeof("checksynchronize")];
      char stringpool_str1052[sizeof("inrange")];
      char stringpool_str1054[sizeof("stringtoolestr")];
      char stringpool_str1056[sizeof("strtointdef")];
      char stringpool_str1063[sizeof("cosecant")];
      char stringpool_str1068[sizeof("disksize")];
      char stringpool_str1072[sizeof("stringofchar")];
      char stringpool_str1074[sizeof("extractstrings")];
      char stringpool_str1075[sizeof("paramcount")];
      char stringpool_str1076[sizeof("getprecisionmode")];
      char stringpool_str1077[sizeof("strscan")];
      char stringpool_str1079[sizeof("formatbuf")];
      char stringpool_str1083[sizeof("minvalue")];
      char stringpool_str1088[sizeof("strlower")];
      char stringpool_str1093[sizeof("stricomp")];
      char stringpool_str1112[sizeof("ansistricomp")];
      char stringpool_str1115[sizeof("inttoident")];
      char stringpool_str1117[sizeof("fileage")];
      char stringpool_str1118[sizeof("allocmem")];
      char stringpool_str1122[sizeof("strlcat")];
      char stringpool_str1133[sizeof("vararrayredim")];
      char stringpool_str1138[sizeof("strlcomp")];
      char stringpool_str1139[sizeof("extractfileext")];
      char stringpool_str1147[sizeof("ansisametext")];
      char stringpool_str1149[sizeof("setstring")];
      char stringpool_str1151[sizeof("ansisamestr")];
      char stringpool_str1152[sizeof("arccoth")];
      char stringpool_str1153[sizeof("filesize")];
      char stringpool_str1155[sizeof("safeloadlibrary")];
      char stringpool_str1157[sizeof("ansistrlcomp")];
      char stringpool_str1159[sizeof("lowercase")];
      char stringpool_str1161[sizeof("widechartostrvar")];
      char stringpool_str1162[sizeof("getroundmode")];
      char stringpool_str1163[sizeof("radtodeg")];
      char stringpool_str1167[sizeof("ucs4stringtowidestring")];
      char stringpool_str1174[sizeof("extractfiledir")];
      char stringpool_str1175[sizeof("popnstddev")];
      char stringpool_str1176[sizeof("extractfiledrive")];
      char stringpool_str1181[sizeof("isdelimiter")];
      char stringpool_str1182[sizeof("getmodulefilename")];
      char stringpool_str1183[sizeof("registercomponents")];
      char stringpool_str1185[sizeof("randg")];
      char stringpool_str1190[sizeof("filesearch")];
      char stringpool_str1195[sizeof("extractfilename")];
      char stringpool_str1199[sizeof("tryfloattocurr")];
      char stringpool_str1200[sizeof("win32check")];
      char stringpool_str1209[sizeof("findfirst")];
      char stringpool_str1210[sizeof("widefmtstr")];
      char stringpool_str1216[sizeof("stddev")];
      char stringpool_str1218[sizeof("totalvariance")];
      char stringpool_str1223[sizeof("finalize")];
      char stringpool_str1224[sizeof("filewrite")];
      char stringpool_str1225[sizeof("isinfinite")];
      char stringpool_str1231[sizeof("isleapyear")];
      char stringpool_str1241[sizeof("countgenerations")];
      char stringpool_str1247[sizeof("payment")];
      char stringpool_str1248[sizeof("simpleroundto")];
      char stringpool_str1252[sizeof("rewrite")];
      char stringpool_str1257[sizeof("setlinebreakstyle")];
      char stringpool_str1258[sizeof("ansistrlicomp")];
      char stringpool_str1260[sizeof("quotedstr")];
      char stringpool_str1262[sizeof("ansilastchar")];
      char stringpool_str1269[sizeof("widecharlentostrvar")];
      char stringpool_str1274[sizeof("strlicomp")];
      char stringpool_str1277[sizeof("exclude")];
      char stringpool_str1278[sizeof("truncate")];
      char stringpool_str1279[sizeof("isvariantmanagerset")];
      char stringpool_str1293[sizeof("strecopy")];
      char stringpool_str1294[sizeof("blockread")];
      char stringpool_str1296[sizeof("slndepreciation")];
      char stringpool_str1298[sizeof("filedatetodatetime")];
      char stringpool_str1310[sizeof("initialize")];
      char stringpool_str1314[sizeof("copy")];
      char stringpool_str1324[sizeof("ansiquotedstr")];
      char stringpool_str1332[sizeof("findcmdlineswitch")];
      char stringpool_str1337[sizeof("initializepackage")];
      char stringpool_str1338[sizeof("stringreplace")];
      char stringpool_str1344[sizeof("getfileversion")];
      char stringpool_str1345[sizeof("power")];
      char stringpool_str1355[sizeof("reallocmem")];
      char stringpool_str1357[sizeof("comptodouble")];
      char stringpool_str1358[sizeof("momentskewkurtosis")];
      char stringpool_str1362[sizeof("isvalidident")];
      char stringpool_str1374[sizeof("datetimetotimestamp")];
      char stringpool_str1375[sizeof("equalrect")];
      char stringpool_str1376[sizeof("getmem")];
      char stringpool_str1379[sizeof("ansiuppercase")];
      char stringpool_str1387[sizeof("ansiuppercasefilename")];
      char stringpool_str1390[sizeof("cycletodeg")];
      char stringpool_str1391[sizeof("cycletograd")];
      char stringpool_str1394[sizeof("fileopen")];
      char stringpool_str1398[sizeof("strlcopy")];
      char stringpool_str1403[sizeof("excludetrailingpathdelimiter")];
      char stringpool_str1404[sizeof("poly")];
      char stringpool_str1406[sizeof("strnextchar")];
      char stringpool_str1407[sizeof("strmove")];
      char stringpool_str1408[sizeof("wraptext")];
      char stringpool_str1416[sizeof("gradtocycle")];
      char stringpool_str1421[sizeof("texttofloat")];
      char stringpool_str1422[sizeof("releaseexceptionobject")];
      char stringpool_str1428[sizeof("intpower")];
      char stringpool_str1429[sizeof("includetrailingbackslash")];
      char stringpool_str1436[sizeof("setexceptionmask")];
      char stringpool_str1440[sizeof("lnxp1")];
      char stringpool_str1444[sizeof("timestamptodatetime")];
      char stringpool_str1445[sizeof("datetimetosystemtime")];
      char stringpool_str1457[sizeof("samefilename")];
      char stringpool_str1463[sizeof("getmodulename")];
      char stringpool_str1478[sizeof("bytetype")];
      char stringpool_str1487[sizeof("writeln")];
      char stringpool_str1499[sizeof("booltostr")];
      char stringpool_str1511[sizeof("datetimetostring")];
      char stringpool_str1516[sizeof("getmemorymanager")];
      char stringpool_str1521[sizeof("adjustlinebreaks")];
      char stringpool_str1528[sizeof("widelowercase")];
      char stringpool_str1532[sizeof("doubledecliningbalance")];
      char stringpool_str1533[sizeof("strcopy")];
      char stringpool_str1542[sizeof("expanduncfilename")];
      char stringpool_str1549[sizeof("expandfilename")];
      char stringpool_str1553[sizeof("expandfilenamecase")];
      char stringpool_str1554[sizeof("degtograd")];
      char stringpool_str1556[sizeof("getpackagedescription")];
      char stringpool_str1569[sizeof("strupper")];
      char stringpool_str1579[sizeof("getpackageinfo")];
      char stringpool_str1580[sizeof("exceptaddr")];
      char stringpool_str1585[sizeof("comparemem")];
      char stringpool_str1596[sizeof("loadpackage")];
      char stringpool_str1600[sizeof("fmtloadstr")];
      char stringpool_str1606[sizeof("exceptionerrormessage")];
      char stringpool_str1613[sizeof("exp")];
      char stringpool_str1615[sizeof("finalizepackage")];
      char stringpool_str1617[sizeof("getenvironmentvariable")];
      char stringpool_str1618[sizeof("activateclassgroup")];
      char stringpool_str1622[sizeof("stringtoguid")];
      char stringpool_str1628[sizeof("strpcopy")];
      char stringpool_str1636[sizeof("syddepreciation")];
      char stringpool_str1644[sizeof("deallocatehwnd")];
      char stringpool_str1656[sizeof("ansiextractquotedstr")];
      char stringpool_str1663[sizeof("getclass")];
      char stringpool_str1664[sizeof("beep")];
      char stringpool_str1669[sizeof("dayofweek")];
      char stringpool_str1674[sizeof("swap")];
      char stringpool_str1682[sizeof("invalidpoint")];
      char stringpool_str1685[sizeof("frexp")];
      char stringpool_str1690[sizeof("hypot")];
      char stringpool_str1695[sizeof("sumsandsquares")];
      char stringpool_str1696[sizeof("widechartostring")];
      char stringpool_str1711[sizeof("enummodules")];
      char stringpool_str1717[sizeof("trystrtobool")];
      char stringpool_str1734[sizeof("sech")];
      char stringpool_str1749[sizeof("csch")];
      char stringpool_str1750[sizeof("settextbuf")];
      char stringpool_str1754[sizeof("strplcopy")];
      char stringpool_str1755[sizeof("ldexp")];
      char stringpool_str1758[sizeof("chartobytelen")];
      char stringpool_str1759[sizeof("coth")];
      char stringpool_str1761[sizeof("smallpoint")];
      char stringpool_str1763[sizeof("unregistermoduleclasses")];
      char stringpool_str1764[sizeof("cosh")];
      char stringpool_str1783[sizeof("readcomponentresex")];
      char stringpool_str1784[sizeof("samevalue")];
      char stringpool_str1788[sizeof("unregisterintegerconsts")];
      char stringpool_str1789[sizeof("tanh")];
      char stringpool_str1790[sizeof("widecomparetext")];
      char stringpool_str1794[sizeof("widecomparestr")];
      char stringpool_str1801[sizeof("timestamptomsecs")];
      char stringpool_str1804[sizeof("widecharlentostring")];
      char stringpool_str1811[sizeof("maxintvalue")];
      char stringpool_str1816[sizeof("writecomponentresfile")];
      char stringpool_str1817[sizeof("pointsequal")];
      char stringpool_str1825[sizeof("strbufsize")];
      char stringpool_str1830[sizeof("unregisterclass")];
      char stringpool_str1832[sizeof("unregisterclasses")];
      char stringpool_str1834[sizeof("sinh")];
      char stringpool_str1843[sizeof("registernonactivex")];
      char stringpool_str1849[sizeof("comptocurrency")];
      char stringpool_str1859[sizeof("setlength")];
      char stringpool_str1869[sizeof("extractrelativepath")];
      char stringpool_str1875[sizeof("extractshortpathname")];
      char stringpool_str1895[sizeof("wideformat")];
      char stringpool_str1896[sizeof("beginthread")];
      char stringpool_str1928[sizeof("maxvalue")];
      char stringpool_str1934[sizeof("languages")];
      char stringpool_str1942[sizeof("isuniqueglobalcomponentname")];
      char stringpool_str1958[sizeof("hextobin")];
      char stringpool_str1966[sizeof("getexceptionmask")];
      char stringpool_str1981[sizeof("strnew")];
      char stringpool_str1984[sizeof("set8087cw")];
      char stringpool_str1987[sizeof("widestringtoucs4string")];
      char stringpool_str1996[sizeof("stringtowidechar")];
      char stringpool_str1997[sizeof("popnvariance")];
      char stringpool_str2004[sizeof("gradtodeg")];
      char stringpool_str2008[sizeof("new")];
      char stringpool_str2013[sizeof("findhinstance")];
      char stringpool_str2038[sizeof("now")];
      char stringpool_str2040[sizeof("groupdescendantswith")];
      char stringpool_str2073[sizeof("ismemorymanagerset")];
      char stringpool_str2077[sizeof("widesametext")];
      char stringpool_str2081[sizeof("widesamestr")];
      char stringpool_str2083[sizeof("nextcharindex")];
      char stringpool_str2088[sizeof("low")];
      char stringpool_str2093[sizeof("inttohex")];
      char stringpool_str2105[sizeof("flush")];
      char stringpool_str2135[sizeof("numberofperiods")];
      char stringpool_str2146[sizeof("strbytetype")];
      char stringpool_str2167[sizeof("doubletocomp")];
      char stringpool_str2175[sizeof("objectresourcetotext")];
      char stringpool_str2178[sizeof("wideformatbuf")];
      char stringpool_str2200[sizeof("blockwrite")];
      char stringpool_str2202[sizeof("allocatehwnd")];
      char stringpool_str2204[sizeof("findglobalcomponent")];
      char stringpool_str2219[sizeof("trimright")];
      char stringpool_str2254[sizeof("excludetrailingbackslash")];
      char stringpool_str2257[sizeof("guidtostring")];
      char stringpool_str2293[sizeof("max")];
      char stringpool_str2296[sizeof("addexitproc")];
      char stringpool_str2298[sizeof("incmonth")];
      char stringpool_str2309[sizeof("wideuppercase")];
      char stringpool_str2330[sizeof("extractfilepath")];
      char stringpool_str2465[sizeof("charlength")];
      char stringpool_str2514[sizeof("get8087cw")];
      char stringpool_str2594[sizeof("incamonth")];
      char stringpool_str2618[sizeof("acquireexceptionobject")];
      char stringpool_str2668[sizeof("changefileext")];
      char stringpool_str2698[sizeof("uniquestring")];
      char stringpool_str2700[sizeof("chartobyteindex")];
      char stringpool_str2733[sizeof("bytetocharlen")];
      char stringpool_str2748[sizeof("objectbinarytotext")];
      char stringpool_str2763[sizeof("showexception")];
      char stringpool_str2838[sizeof("strcharlength")];
      char stringpool_str3067[sizeof("isequalguid")];
      char stringpool_str3083[sizeof("bintohex")];
      char stringpool_str3140[sizeof("objecttexttoresource")];
      char stringpool_str3164[sizeof("high")];
      char stringpool_str3407[sizeof("exceptobject")];
      char stringpool_str3603[sizeof("objecttexttobinary")];
      char stringpool_str3675[sizeof("bytetocharindex")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "reset",
      "rect",
      "erase",
      "sec",
      "strtodate",
      "cot",
      "csc",
      "cos",
      "date",
      "dec",
      "strtocurr",
      "arccos",
      "arctan2",
      "read",
      "arcsec",
      "strend",
      "ord",
      "seek",
      "arccsc",
      "int",
      "frac",
      "tan",
      "odd",
      "strdispose",
      "arctan",
      "inc",
      "inttostr",
      "close",
      "strrscan",
      "forcedirectories",
      "eof",
      "createdir",
      "sincos",
      "sin",
      "cotan",
      "strtodatedef",
      "slice",
      "arcsin",
      "interestrate",
      "continue",
      "strlen",
      "decodedate",
      "delete",
      "randomize",
      "strtocurrdef",
      "diskfree",
      "encodedate",
      "floor",
      "addterminateproc",
      "strlfmt",
      "assert",
      "eoln",
      "radtocycle",
      "readln",
      "isnan",
      "strcat",
      "ioresult",
      "succ",
      "runerror",
      "randomrange",
      "filecreate",
      "teststreamformat",
      "initinheritedcomponent",
      "ceil",
      "arccot",
      "variance",
      "strtobool",
      "trunc",
      "setvariantmanager",
      "secant",
      "internalrateofreturn",
      "insert",
      "strtobooldef",
      "filesetattr",
      "freeandnil",
      "concat",
      "findnext",
      "olestrtostrvar",
      "setcurrentdir",
      "round",
      "strpas",
      "filesetdate",
      "varclear",
      "strpos",
      "fileread",
      "deletefile",
      "utf8encode",
      "datetostr",
      "pred",
      "loadstr",
      "fileseek",
      "createguid",
      "fillchar",
      "ensurerange",
      "startclassgroup",
      "strtodatetime",
      "ansistrscan",
      "linestart",
      "floattotext",
      "strcomp",
      "floattostr",
      "iszero",
      "include",
      "pi",
      "utf8decode",
      "seekeof",
      "dispose",
      "fmtstr",
      "disposestr",
      "ansistrrscan",
      "identtoint",
      "unicodetoutf8",
      "rmdir",
      "sametext",
      "raiselastoserror",
      "collectionsequal",
      "ansistrlower",
      "floattodatetime",
      "seekeoln",
      "ansistrlastchar",
      "strtodatetimedef",
      "trystrtotime",
      "floattotextfmt",
      "floattostrf",
      "mean",
      "trystrtocurr",
      "ansidequotedstr",
      "mkdir",
      "rename",
      "setprecisionmode",
      "netpresentvalue",
      "findresourcehinstance",
      "trystrtodate",
      "raiselastwin32error",
      "trystrtodatetime",
      "stralloc",
      "ansitoutf8",
      "replacedate",
      "presentvalue",
      "min",
      "replacetime",
      "currtostr",
      "trystrtoint",
      "roundto",
      "includetrailingpathdelimiter",
      "move",
      "interlockedexchange",
      "random",
      "ansilowercase",
      "tryencodedate",
      "ansilowercasefilename",
      "trystrtofloat",
      "trystrtoint64",
      "interlockeddecrement",
      "interlockedexchangeadd",
      "strtofloat",
      "setroundmode",
      "trimleft",
      "findclose",
      "currtostrf",
      "getdir",
      "strtotime",
      "decodetime",
      "findclass",
      "interlockedincrement",
      "fileclose",
      "pucs4chars",
      "gettime",
      "ispathdelimiter",
      "upcase",
      "assignstr",
      "encodetime",
      "floattocurr",
      "getlasterror",
      "log2",
      "log10",
      "strfmt",
      "getformatsettings",
      "degtorad",
      "meanandstddev",
      "futurevalue",
      "lastdelimiter",
      "strtofloatdef",
      "findclasshinstance",
      "renamefile",
      "sign",
      "divmod",
      "removedir",
      "fileexists",
      "sizeof",
      "format",
      "currentyear",
      "assigned",
      "ansistrpos",
      "ansistrcomp",
      "utf8toansi",
      "appendstr",
      "degtocycle",
      "paramstr",
      "utf8tounicode",
      "logn",
      "sleep",
      "append",
      "strtotimedef",
      "callterminateprocs",
      "olestrtostring",
      "timetostr",
      "closefile",
      "comparetext",
      "comparestr",
      "formatdatetime",
      "filesetreadonly",
      "ansipos",
      "assignfile",
      "outofmemoryerror",
      "getvariantmanager",
      "getlocaleformatsettings",
      "trim",
      "readcomponentresfile",
      "readcomponentres",
      "registerclass",
      "fileisreadonly",
      "registerclasses",
      "registerclassalias",
      "ansicomparetext",
      "ansicomparestr",
      "directoryexists",
      "formatfloat",
      "norm",
      "filegetattr",
      "chr",
      "cycletorad",
      "syserrormessage",
      "getcurrentdir",
      "decodedatefully",
      "registerintegerconsts",
      "filegetdate",
      "ansistrupper",
      "filepos",
      "interestpayment",
      "freemem",
      "uppercase",
      "ansicomparefilename",
      "sumint",
      "arcsech",
      "supports",
      "formatcurr",
      "arccosh",
      "arctanh",
      "strtoint",
      "registernoicon",
      "arccsch",
      "sumofsquares",
      "typeinfo",
      "minintvalue",
      "chdir",
      "datetimetostr",
      "periodpayment",
      "floattodecimal",
      "halt",
      "setmemorymanager",
      "comparevalue",
      "enumresourcemodules",
      "strtoint64",
      "arcsinh",
      "tryencodetime",
      "sum",
      "unloadpackage",
      "exit",
      "tryfloattodatetime",
      "sqr",
      "sqrt",
      "abs",
      "gradtorad",
      "systemtimetodatetime",
      "datetimetofiledate",
      "radtograd",
      "write",
      "msecstotimestamp",
      "strtoint64def",
      "abort",
      "newstr",
      "endthread",
      "checksynchronize",
      "inrange",
      "stringtoolestr",
      "strtointdef",
      "cosecant",
      "disksize",
      "stringofchar",
      "extractstrings",
      "paramcount",
      "getprecisionmode",
      "strscan",
      "formatbuf",
      "minvalue",
      "strlower",
      "stricomp",
      "ansistricomp",
      "inttoident",
      "fileage",
      "allocmem",
      "strlcat",
      "vararrayredim",
      "strlcomp",
      "extractfileext",
      "ansisametext",
      "setstring",
      "ansisamestr",
      "arccoth",
      "filesize",
      "safeloadlibrary",
      "ansistrlcomp",
      "lowercase",
      "widechartostrvar",
      "getroundmode",
      "radtodeg",
      "ucs4stringtowidestring",
      "extractfiledir",
      "popnstddev",
      "extractfiledrive",
      "isdelimiter",
      "getmodulefilename",
      "registercomponents",
      "randg",
      "filesearch",
      "extractfilename",
      "tryfloattocurr",
      "win32check",
      "findfirst",
      "widefmtstr",
      "stddev",
      "totalvariance",
      "finalize",
      "filewrite",
      "isinfinite",
      "isleapyear",
      "countgenerations",
      "payment",
      "simpleroundto",
      "rewrite",
      "setlinebreakstyle",
      "ansistrlicomp",
      "quotedstr",
      "ansilastchar",
      "widecharlentostrvar",
      "strlicomp",
      "exclude",
      "truncate",
      "isvariantmanagerset",
      "strecopy",
      "blockread",
      "slndepreciation",
      "filedatetodatetime",
      "initialize",
      "copy",
      "ansiquotedstr",
      "findcmdlineswitch",
      "initializepackage",
      "stringreplace",
      "getfileversion",
      "power",
      "reallocmem",
      "comptodouble",
      "momentskewkurtosis",
      "isvalidident",
      "datetimetotimestamp",
      "equalrect",
      "getmem",
      "ansiuppercase",
      "ansiuppercasefilename",
      "cycletodeg",
      "cycletograd",
      "fileopen",
      "strlcopy",
      "excludetrailingpathdelimiter",
      "poly",
      "strnextchar",
      "strmove",
      "wraptext",
      "gradtocycle",
      "texttofloat",
      "releaseexceptionobject",
      "intpower",
      "includetrailingbackslash",
      "setexceptionmask",
      "lnxp1",
      "timestamptodatetime",
      "datetimetosystemtime",
      "samefilename",
      "getmodulename",
      "bytetype",
      "writeln",
      "booltostr",
      "datetimetostring",
      "getmemorymanager",
      "adjustlinebreaks",
      "widelowercase",
      "doubledecliningbalance",
      "strcopy",
      "expanduncfilename",
      "expandfilename",
      "expandfilenamecase",
      "degtograd",
      "getpackagedescription",
      "strupper",
      "getpackageinfo",
      "exceptaddr",
      "comparemem",
      "loadpackage",
      "fmtloadstr",
      "exceptionerrormessage",
      "exp",
      "finalizepackage",
      "getenvironmentvariable",
      "activateclassgroup",
      "stringtoguid",
      "strpcopy",
      "syddepreciation",
      "deallocatehwnd",
      "ansiextractquotedstr",
      "getclass",
      "beep",
      "dayofweek",
      "swap",
      "invalidpoint",
      "frexp",
      "hypot",
      "sumsandsquares",
      "widechartostring",
      "enummodules",
      "trystrtobool",
      "sech",
      "csch",
      "settextbuf",
      "strplcopy",
      "ldexp",
      "chartobytelen",
      "coth",
      "smallpoint",
      "unregistermoduleclasses",
      "cosh",
      "readcomponentresex",
      "samevalue",
      "unregisterintegerconsts",
      "tanh",
      "widecomparetext",
      "widecomparestr",
      "timestamptomsecs",
      "widecharlentostring",
      "maxintvalue",
      "writecomponentresfile",
      "pointsequal",
      "strbufsize",
      "unregisterclass",
      "unregisterclasses",
      "sinh",
      "registernonactivex",
      "comptocurrency",
      "setlength",
      "extractrelativepath",
      "extractshortpathname",
      "wideformat",
      "beginthread",
      "maxvalue",
      "languages",
      "isuniqueglobalcomponentname",
      "hextobin",
      "getexceptionmask",
      "strnew",
      "set8087cw",
      "widestringtoucs4string",
      "stringtowidechar",
      "popnvariance",
      "gradtodeg",
      "new",
      "findhinstance",
      "now",
      "groupdescendantswith",
      "ismemorymanagerset",
      "widesametext",
      "widesamestr",
      "nextcharindex",
      "low",
      "inttohex",
      "flush",
      "numberofperiods",
      "strbytetype",
      "doubletocomp",
      "objectresourcetotext",
      "wideformatbuf",
      "blockwrite",
      "allocatehwnd",
      "findglobalcomponent",
      "trimright",
      "excludetrailingbackslash",
      "guidtostring",
      "max",
      "addexitproc",
      "incmonth",
      "wideuppercase",
      "extractfilepath",
      "charlength",
      "get8087cw",
      "incamonth",
      "acquireexceptionobject",
      "changefileext",
      "uniquestring",
      "chartobyteindex",
      "bytetocharlen",
      "objectbinarytotext",
      "showexception",
      "strcharlength",
      "isequalguid",
      "bintohex",
      "objecttexttoresource",
      "high",
      "exceptobject",
      "objecttexttobinary",
      "bytetocharindex"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str30,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str48,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str53,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str58,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str59,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str68,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str79,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str103,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str111,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str145,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str187,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str197,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str198,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str201,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str215,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str222,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str233,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str235,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str240,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str241,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str242,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str246,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str249,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str250,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str251,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str255,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str256,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str263,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str264,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str278,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str281,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str285,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str286,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str287,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str289,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str296,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str298,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str299,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str300,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str302,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str311,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str325,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str336,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str337,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str341,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str350,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str351,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str353,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str354,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str363,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str370,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str371,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str376,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str378,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str381,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str383,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str385,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str390,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str399,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str409,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str412,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str418,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str420,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str423,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str426,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str435,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str438,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str441,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str444,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str446,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str447,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str450,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str451,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str452,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str457,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str465,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str467,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str472,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str476,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str480,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str482,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str485,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str488,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str490,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str493,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str496,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str501,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str502,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str505,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str508,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str510,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str511,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str517,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str519,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str521,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str524,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str532,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str535,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str540,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str541,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str546,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str550,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str551,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str552,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str554,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str556,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str558,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str560,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str561,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str562,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str563,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str566,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str569,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str576,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str577,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str578,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str584,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str589,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str591,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str598,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str603,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str606,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str608,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str618,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str625,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str627,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str630,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str632,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str633,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str639,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str640,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str641,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str644,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str645,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str649,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str650,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str654,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str655,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str657,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str660,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str661,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str664,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str665,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str671,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str672,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str674,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str675,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str681,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str682,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str683,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str693,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str696,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str698,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str703,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str708,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str710,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str714,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str716,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str719,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str720,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str721,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str726,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str731,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str733,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str735,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str746,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str750,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str754,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str765,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str773,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str783,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str784,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str785,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str786,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str787,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str788,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str789,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str794,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str804,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str806,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str810,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str814,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str820,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str822,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str830,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str831,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str832,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str838,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str844,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str845,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str846,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str848,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str849,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str850,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str853,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str860,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str864,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str865,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str866,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str869,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str871,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str873,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str875,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str890,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str893,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str895,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str896,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str906,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str912,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str917,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str925,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str927,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str929,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str934,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str936,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str937,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str938,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str940,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str942,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str947,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str948,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str949,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str952,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str957,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str958,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str966,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str970,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str973,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str978,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str979,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str984,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str986,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str987,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str989,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str990,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str992,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str993,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str998,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1003,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1004,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1013,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1019,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1020,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1023,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1024,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1030,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1038,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1039,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1040,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1041,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1043,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1045,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1046,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1049,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1051,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1052,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1054,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1056,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1063,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1068,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1072,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1074,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1075,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1076,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1077,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1079,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1083,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1088,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1093,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1112,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1115,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1117,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1118,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1122,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1133,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1138,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1139,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1147,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1149,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1151,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1152,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1153,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1155,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1157,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1159,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1161,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1162,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1163,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1167,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1174,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1175,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1176,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1181,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1182,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1183,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1185,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1190,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1195,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1199,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1200,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1209,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1210,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1216,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1218,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1223,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1224,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1225,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1231,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1241,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1247,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1248,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1252,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1257,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1258,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1260,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1262,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1269,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1274,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1277,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1278,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1279,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1293,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1294,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1296,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1298,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1310,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1314,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1324,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1332,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1337,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1338,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1344,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1345,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1355,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1357,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1358,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1362,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1374,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1375,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1376,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1379,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1387,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1390,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1391,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1394,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1398,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1403,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1404,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1406,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1407,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1408,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1416,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1421,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1422,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1428,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1429,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1436,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1440,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1444,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1445,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1457,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1463,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1478,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1487,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1499,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1511,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1516,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1521,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1528,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1532,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1533,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1542,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1549,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1553,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1554,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1556,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1569,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1579,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1580,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1585,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1596,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1600,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1606,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1613,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1615,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1617,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1618,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1622,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1628,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1636,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1644,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1656,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1663,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1664,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1669,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1674,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1682,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1685,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1690,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1695,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1696,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1711,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1717,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1734,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1749,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1750,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1754,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1755,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1758,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1759,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1761,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1763,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1764,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1783,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1784,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1788,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1789,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1790,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1794,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1801,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1804,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1811,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1816,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1817,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1825,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1830,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1832,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1834,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1843,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1849,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1859,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1869,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1875,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1895,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1896,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1928,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1934,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1942,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1958,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1966,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1981,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1984,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1987,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1996,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str1997,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2004,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2008,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2013,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2038,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2040,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2073,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2077,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2081,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2083,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2088,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2093,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2105,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2135,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2146,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2167,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2175,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2178,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2200,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2202,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2204,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2219,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2254,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2257,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2293,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2296,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2298,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2309,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2330,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2465,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2514,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2594,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2618,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2668,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2698,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2700,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2733,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2748,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2763,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str2838,
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
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3067,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3083,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3140,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3164,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3407,
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
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3603,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3675
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
