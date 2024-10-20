/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf llamafile/is_keyword_c_type.gperf  */
/* Computed positions: -k'1-2,4-6,8,10,12,17-18' */

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

#line 1 "llamafile/is_keyword_c_type.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 180
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 26
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 584
/* maximum key range = 582, duplicates = 0 */

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
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585,  60,
      110,  45, 180, 585,  30, 585, 165, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585,   0,   0,
        0, 585, 585,   0, 585, 585, 585, 585,   5, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585,   0, 585,   0, 105,  10,
      110,   5,  75, 220, 120,   0, 585, 185,  30,   5,
        5,  80,   0,   5,   0,  65,   0,  35,  15, 160,
      220, 160, 170, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585, 585, 585, 585, 585,
      585, 585, 585, 585, 585, 585
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[17]];
      /*FALLTHROUGH*/
      case 17:
        hval += asso_values[(unsigned char)str[16]];
      /*FALLTHROUGH*/
      case 16:
      case 15:
      case 14:
      case 13:
      case 12:
        hval += asso_values[(unsigned char)str[11]];
      /*FALLTHROUGH*/
      case 11:
      case 10:
        hval += asso_values[(unsigned char)str[9]];
      /*FALLTHROUGH*/
      case 9:
      case 8:
        hval += asso_values[(unsigned char)str[7]];
      /*FALLTHROUGH*/
      case 7:
      case 6:
        hval += asso_values[(unsigned char)str[5]];
      /*FALLTHROUGH*/
      case 5:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
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
is_keyword_c_type (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str3[sizeof("DIR")];
      char stringpool_str4[sizeof("FILE")];
      char stringpool_str5[sizeof("pid_t")];
      char stringpool_str10[sizeof("ino_t")];
      char stringpool_str11[sizeof("time_t")];
      char stringpool_str12[sizeof("timer_t")];
      char stringpool_str13[sizeof("intptr_t")];
      char stringpool_str14[sizeof("pthread_t")];
      char stringpool_str15[sizeof("mqd_t")];
      char stringpool_str16[sizeof("intN_t")];
      char stringpool_str19[sizeof("pthread_attr_t")];
      char stringpool_str22[sizeof("pthread_barrier_t")];
      char stringpool_str23[sizeof("pthread_key_t")];
      char stringpool_str24[sizeof("cc_t")];
      char stringpool_str25[sizeof("atomic_int")];
      char stringpool_str26[sizeof("pthread_barrierattr_t")];
      char stringpool_str27[sizeof("trace_attr_t")];
      char stringpool_str28[sizeof("pthread_spinlock_t")];
      char stringpool_str29[sizeof("pthread_once_t")];
      char stringpool_str30[sizeof("atomic_intptr_t")];
      char stringpool_str31[sizeof("atomic_ptrdiff_t")];
      char stringpool_str35[sizeof("mcontext_t")];
      char stringpool_str36[sizeof("atomic_char")];
      char stringpool_str37[sizeof("iconv_t")];
      char stringpool_str40[sizeof("uid_t")];
      char stringpool_str41[sizeof("rlim_t")];
      char stringpool_str43[sizeof("rlimit_t")];
      char stringpool_str44[sizeof("uintptr_t")];
      char stringpool_str49[sizeof("uintmax_t")];
      char stringpool_str50[sizeof("t_scalar_t")];
      char stringpool_str51[sizeof("trace_event_id_t")];
      char stringpool_str52[sizeof("trace_event_set_t")];
      char stringpool_str60[sizeof("pthread_mutex_t")];
      char stringpool_str61[sizeof("atomic_long")];
      char stringpool_str64[sizeof("pthread_mutexattr_t")];
      char stringpool_str65[sizeof("ucontext_t")];
      char stringpool_str66[sizeof("atomic_uint")];
      char stringpool_str71[sizeof("atomic_uintptr_t")];
      char stringpool_str75[sizeof("sem_t")];
      char stringpool_str76[sizeof("size_t")];
      char stringpool_str78[sizeof("uint_least8_t")];
      char stringpool_str80[sizeof("register_t")];
      char stringpool_str82[sizeof("lldiv_t")];
      char stringpool_str85[sizeof("atomic_char32_t")];
      char stringpool_str86[sizeof("t_uscalar_t")];
      char stringpool_str89[sizeof("cpu_set_t")];
      char stringpool_str91[sizeof("int_fast8_t")];
      char stringpool_str92[sizeof("sig_atomic_t")];
      char stringpool_str94[sizeof("in_port_t")];
      char stringpool_str96[sizeof("mode_t")];
      char stringpool_str97[sizeof("errno_t")];
      char stringpool_str98[sizeof("atomic_ullong")];
      char stringpool_str100[sizeof("atomic_char16_t")];
      char stringpool_str101[sizeof("fenv_t")];
      char stringpool_str102[sizeof("int16_t")];
      char stringpool_str104[sizeof("fexcept_t")];
      char stringpool_str106[sizeof("atomic_uint_least64_t")];
      char stringpool_str108[sizeof("atomic_int_fast8_t")];
      char stringpool_str109[sizeof("uint_least16_t")];
      char stringpool_str112[sizeof("float_t")];
      char stringpool_str114[sizeof("id_t")];
      char stringpool_str115[sizeof("div_t")];
      char stringpool_str116[sizeof("data_t")];
      char stringpool_str117[sizeof("va_list")];
      char stringpool_str118[sizeof("msglen_t")];
      char stringpool_str119[sizeof("mbstate_t")];
      char stringpool_str120[sizeof("dev_t")];
      char stringpool_str121[sizeof("atomic_uint_least32_t")];
      char stringpool_str122[sizeof("int_fast16_t")];
      char stringpool_str124[sizeof("msgqnum_t")];
      char stringpool_str133[sizeof("uint16_t")];
      char stringpool_str135[sizeof("trace_id_t")];
      char stringpool_str136[sizeof("atomic_uint_least16_t")];
      char stringpool_str139[sizeof("atomic_int_fast16_t")];
      char stringpool_str140[sizeof("regmatch_t")];
      char stringpool_str143[sizeof("sigset_t")];
      char stringpool_str144[sizeof("float64_t")];
      char stringpool_str146[sizeof("fpos_t")];
      char stringpool_str148[sizeof("int_least64_t")];
      char stringpool_str151[sizeof("nfds_t")];
      char stringpool_str153[sizeof("locale_t")];
      char stringpool_str154[sizeof("siginfo_t")];
      char stringpool_str155[sizeof("atomic_int_least16_t")];
      char stringpool_str158[sizeof("blkcnt_t")];
      char stringpool_str159[sizeof("float32_t")];
      char stringpool_str160[sizeof("off_t")];
      char stringpool_str161[sizeof("ldiv_t")];
      char stringpool_str163[sizeof("int_least32_t")];
      char stringpool_str165[sizeof("atomic_uint_fast16_t")];
      char stringpool_str166[sizeof("wint_t")];
      char stringpool_str167[sizeof("int32_t")];
      char stringpool_str168[sizeof("timed_mutex_t")];
      char stringpool_str172[sizeof("atomic_short")];
      char stringpool_str174[sizeof("float16_t")];
      char stringpool_str176[sizeof("int8_t")];
      char stringpool_str177[sizeof("wchar_t")];
      char stringpool_str178[sizeof("int_least16_t")];
      char stringpool_str182[sizeof("atomic_uchar")];
      char stringpool_str183[sizeof("atomic_ushort")];
      char stringpool_str184[sizeof("wctrans_t")];
      char stringpool_str186[sizeof("sa_family_t")];
      char stringpool_str187[sizeof("speed_t")];
      char stringpool_str188[sizeof("ushort_t")];
      char stringpool_str189[sizeof("uint_least32_t")];
      char stringpool_str190[sizeof("fsfilcnt_t")];
      char stringpool_str191[sizeof("loff_t")];
      char stringpool_str193[sizeof("shmatt_t")];
      char stringpool_str194[sizeof("ptrdiff_t")];
      char stringpool_str195[sizeof("key_t")];
      char stringpool_str198[sizeof("uint32_t")];
      char stringpool_str201[sizeof("semaphore_t")];
      char stringpool_str202[sizeof("int_fast32_t")];
      char stringpool_str207[sizeof("uint8_t")];
      char stringpool_str209[sizeof("pthread_cond_t")];
      char stringpool_str211[sizeof("atomic_bool")];
      char stringpool_str212[sizeof("atomic_schar")];
      char stringpool_str213[sizeof("pthread_condattr_t")];
      char stringpool_str214[sizeof("uint128_t")];
      char stringpool_str218[sizeof("uint_fast64_t")];
      char stringpool_str219[sizeof("atomic_int_fast32_t")];
      char stringpool_str220[sizeof("atomic_int_least32_t")];
      char stringpool_str222[sizeof("int64_t")];
      char stringpool_str225[sizeof("gid_t")];
      char stringpool_str228[sizeof("char16_t")];
      char stringpool_str229[sizeof("atomic_int_least8_t")];
      char stringpool_str230[sizeof("atomic_uint_fast32_t")];
      char stringpool_str232[sizeof("nlink_t")];
      char stringpool_str233[sizeof("uint_fast32_t")];
      char stringpool_str234[sizeof("in_addr_t")];
      char stringpool_str236[sizeof("thrd_t")];
      char stringpool_str237[sizeof("regex_t")];
      char stringpool_str238[sizeof("intmax_t")];
      char stringpool_str239[sizeof("atomic_uint_fast8_t")];
      char stringpool_str240[sizeof("atomic_uint_least8_t")];
      char stringpool_str242[sizeof("clock_t")];
      char stringpool_str243[sizeof("regoff_t")];
      char stringpool_str244[sizeof("clockid_t")];
      char stringpool_str248[sizeof("uint_fast16_t")];
      char stringpool_str253[sizeof("uint64_t")];
      char stringpool_str258[sizeof("atomic_bool32")];
      char stringpool_str259[sizeof("uint_least64_t")];
      char stringpool_str261[sizeof("pthread_rwlock_t")];
      char stringpool_str263[sizeof("atomic_size_t")];
      char stringpool_str265[sizeof("pthread_rwlockattr_t")];
      char stringpool_str266[sizeof("max_align_t")];
      char stringpool_str267[sizeof("stack_t")];
      char stringpool_str268[sizeof("tcflag_t")];
      char stringpool_str270[sizeof("useconds_t")];
      char stringpool_str272[sizeof("int_fast64_t")];
      char stringpool_str275[sizeof("atomic_int_least64_t")];
      char stringpool_str278[sizeof("syscall_arg_t")];
      char stringpool_str282[sizeof("int_least8_t")];
      char stringpool_str283[sizeof("idtype_t")];
      char stringpool_str285[sizeof("atomic_uint_fast64_t")];
      char stringpool_str289[sizeof("atomic_int_fast64_t")];
      char stringpool_str293[sizeof("char32_t")];
      char stringpool_str300[sizeof("bfloat16_t")];
      char stringpool_str309[sizeof("atomic_wchar_t")];
      char stringpool_str312[sizeof("ssize_t")];
      char stringpool_str313[sizeof("ssizet_t")];
      char stringpool_str316[sizeof("suseconds_t")];
      char stringpool_str328[sizeof("posix_trace_attr_t")];
      char stringpool_str333[sizeof("uint24_t")];
      char stringpool_str338[sizeof("double_t")];
      char stringpool_str340[sizeof("float128_t")];
      char stringpool_str343[sizeof("wctype_t")];
      char stringpool_str344[sizeof("imaxdiv_t")];
      char stringpool_str348[sizeof("int128_t")];
      char stringpool_str352[sizeof("uint_fast8_t")];
      char stringpool_str357[sizeof("atomic_llong")];
      char stringpool_str361[sizeof("glob_t")];
      char stringpool_str362[sizeof("atomic_ulong")];
      char stringpool_str364[sizeof("timed_thread_t")];
      char stringpool_str374[sizeof("socklen_t")];
      char stringpool_str375[sizeof("fsblkcnt_t")];
      char stringpool_str376[sizeof("bool32")];
      char stringpool_str379[sizeof("blksize_t")];
      char stringpool_str477[sizeof("posix_spawnattr_t")];
      char stringpool_str486[sizeof("posix_spawn_file_actions_t")];
      char stringpool_str584[sizeof("wordexp_t")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "DIR",
      "FILE",
      "pid_t",
      "ino_t",
      "time_t",
      "timer_t",
      "intptr_t",
      "pthread_t",
      "mqd_t",
      "intN_t",
      "pthread_attr_t",
      "pthread_barrier_t",
      "pthread_key_t",
      "cc_t",
      "atomic_int",
      "pthread_barrierattr_t",
      "trace_attr_t",
      "pthread_spinlock_t",
      "pthread_once_t",
      "atomic_intptr_t",
      "atomic_ptrdiff_t",
      "mcontext_t",
      "atomic_char",
      "iconv_t",
      "uid_t",
      "rlim_t",
      "rlimit_t",
      "uintptr_t",
      "uintmax_t",
      "t_scalar_t",
      "trace_event_id_t",
      "trace_event_set_t",
      "pthread_mutex_t",
      "atomic_long",
      "pthread_mutexattr_t",
      "ucontext_t",
      "atomic_uint",
      "atomic_uintptr_t",
      "sem_t",
      "size_t",
      "uint_least8_t",
      "register_t",
      "lldiv_t",
      "atomic_char32_t",
      "t_uscalar_t",
      "cpu_set_t",
      "int_fast8_t",
      "sig_atomic_t",
      "in_port_t",
      "mode_t",
      "errno_t",
      "atomic_ullong",
      "atomic_char16_t",
      "fenv_t",
      "int16_t",
      "fexcept_t",
      "atomic_uint_least64_t",
      "atomic_int_fast8_t",
      "uint_least16_t",
      "float_t",
      "id_t",
      "div_t",
      "data_t",
      "va_list",
      "msglen_t",
      "mbstate_t",
      "dev_t",
      "atomic_uint_least32_t",
      "int_fast16_t",
      "msgqnum_t",
      "uint16_t",
      "trace_id_t",
      "atomic_uint_least16_t",
      "atomic_int_fast16_t",
      "regmatch_t",
      "sigset_t",
      "float64_t",
      "fpos_t",
      "int_least64_t",
      "nfds_t",
      "locale_t",
      "siginfo_t",
      "atomic_int_least16_t",
      "blkcnt_t",
      "float32_t",
      "off_t",
      "ldiv_t",
      "int_least32_t",
      "atomic_uint_fast16_t",
      "wint_t",
      "int32_t",
      "timed_mutex_t",
      "atomic_short",
      "float16_t",
      "int8_t",
      "wchar_t",
      "int_least16_t",
      "atomic_uchar",
      "atomic_ushort",
      "wctrans_t",
      "sa_family_t",
      "speed_t",
      "ushort_t",
      "uint_least32_t",
      "fsfilcnt_t",
      "loff_t",
      "shmatt_t",
      "ptrdiff_t",
      "key_t",
      "uint32_t",
      "semaphore_t",
      "int_fast32_t",
      "uint8_t",
      "pthread_cond_t",
      "atomic_bool",
      "atomic_schar",
      "pthread_condattr_t",
      "uint128_t",
      "uint_fast64_t",
      "atomic_int_fast32_t",
      "atomic_int_least32_t",
      "int64_t",
      "gid_t",
      "char16_t",
      "atomic_int_least8_t",
      "atomic_uint_fast32_t",
      "nlink_t",
      "uint_fast32_t",
      "in_addr_t",
      "thrd_t",
      "regex_t",
      "intmax_t",
      "atomic_uint_fast8_t",
      "atomic_uint_least8_t",
      "clock_t",
      "regoff_t",
      "clockid_t",
      "uint_fast16_t",
      "uint64_t",
      "atomic_bool32",
      "uint_least64_t",
      "pthread_rwlock_t",
      "atomic_size_t",
      "pthread_rwlockattr_t",
      "max_align_t",
      "stack_t",
      "tcflag_t",
      "useconds_t",
      "int_fast64_t",
      "atomic_int_least64_t",
      "syscall_arg_t",
      "int_least8_t",
      "idtype_t",
      "atomic_uint_fast64_t",
      "atomic_int_fast64_t",
      "char32_t",
      "bfloat16_t",
      "atomic_wchar_t",
      "ssize_t",
      "ssizet_t",
      "suseconds_t",
      "posix_trace_attr_t",
      "uint24_t",
      "double_t",
      "float128_t",
      "wctype_t",
      "imaxdiv_t",
      "int128_t",
      "uint_fast8_t",
      "atomic_llong",
      "glob_t",
      "atomic_ulong",
      "timed_thread_t",
      "socklen_t",
      "fsblkcnt_t",
      "bool32",
      "blksize_t",
      "posix_spawnattr_t",
      "posix_spawn_file_actions_t",
      "wordexp_t"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str3,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str10,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str11,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str13,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str14,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str15,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str16,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str19,
      -1, -1,
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
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str35,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str36,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str37,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str51,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str52,
      -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str60,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str61,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str64,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str65,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str66,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str71,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str75,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str76,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str78,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str80,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str82,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str85,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str100,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str102,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str109,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str114,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str115,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str117,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str133,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str135,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str136,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str139,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str144,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str155,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str158,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str161,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str165,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str176,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str177,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str178,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str182,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str184,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str187,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str189,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str193,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str194,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str195,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str198,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str201,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str202,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str207,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str209,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str211,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str212,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str213,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str219,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str220,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str222,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str225,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str228,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str229,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str230,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str232,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str233,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str234,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str236,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str237,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str238,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str239,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str240,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str242,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str243,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str244,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str248,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str253,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str258,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str259,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str261,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str263,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str265,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str266,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str267,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str268,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str270,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str272,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str275,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str278,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str282,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str283,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str285,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str289,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str293,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str300,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str309,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str312,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str313,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str316,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str328,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str333,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str338,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str340,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str343,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str344,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str348,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str352,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str357,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str361,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str362,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str364,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str374,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str375,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str376,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str379,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str477,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str486,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str584
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
