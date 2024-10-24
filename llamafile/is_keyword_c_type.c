/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_c_type.c llamafile/is_keyword_c_type.gperf  */
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

#define TOTAL_KEYWORDS 192
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 26
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 536
/* maximum key range = 534, duplicates = 0 */

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
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537,  90,
      115, 145, 215,  30,  50, 537, 105, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537,   0,   0,
        0, 537, 537,   0, 537, 537, 537, 537,   5, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537,   0, 537,   0, 160,  10,
      115,   5,  75, 205,  75,   0, 537, 240,  30,   5,
        5,  80,   0,   5,   0,  65,   0,  35,  20, 210,
       20, 130,  20, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537, 537, 537, 537, 537,
      537, 537, 537, 537, 537, 537
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
      char stringpool_str37[sizeof("regex_t")];
      char stringpool_str38[sizeof("intmax_t")];
      char stringpool_str40[sizeof("uid_t")];
      char stringpool_str41[sizeof("rlim_t")];
      char stringpool_str42[sizeof("iconv_t")];
      char stringpool_str43[sizeof("rlimit_t")];
      char stringpool_str44[sizeof("uintptr_t")];
      char stringpool_str46[sizeof("axdx_t")];
      char stringpool_str49[sizeof("uintmax_t")];
      char stringpool_str50[sizeof("t_scalar_t")];
      char stringpool_str56[sizeof("trace_event_id_t")];
      char stringpool_str57[sizeof("trace_event_set_t")];
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
      char stringpool_str86[sizeof("t_uscalar_t")];
      char stringpool_str87[sizeof("lldiv_t")];
      char stringpool_str89[sizeof("cpu_set_t")];
      char stringpool_str91[sizeof("int_fast8_t")];
      char stringpool_str92[sizeof("sig_atomic_t")];
      char stringpool_str94[sizeof("in_port_t")];
      char stringpool_str95[sizeof("regmatch_t")];
      char stringpool_str96[sizeof("mode_t")];
      char stringpool_str97[sizeof("errno_t")];
      char stringpool_str98[sizeof("atomic_ullong")];
      char stringpool_str101[sizeof("atomic_flag")];
      char stringpool_str104[sizeof("fexcept_t")];
      char stringpool_str106[sizeof("fenv_t")];
      char stringpool_str108[sizeof("atomic_int_fast8_t")];
      char stringpool_str112[sizeof("float_t")];
      char stringpool_str113[sizeof("atomic_size_t")];
      char stringpool_str116[sizeof("int8_t")];
      char stringpool_str118[sizeof("msglen_t")];
      char stringpool_str119[sizeof("id_t")];
      char stringpool_str120[sizeof("div_t")];
      char stringpool_str121[sizeof("data_t")];
      char stringpool_str122[sizeof("va_list")];
      char stringpool_str124[sizeof("msgqnum_t")];
      char stringpool_str125[sizeof("dev_t")];
      char stringpool_str126[sizeof("atomic_uint_least64_t")];
      char stringpool_str128[sizeof("posix_trace_attr_t")];
      char stringpool_str129[sizeof("uint_least16_t")];
      char stringpool_str130[sizeof("atomic_char16_t")];
      char stringpool_str137[sizeof("atomic_uchar")];
      char stringpool_str138[sizeof("atomic_ushort")];
      char stringpool_str140[sizeof("trace_id_t")];
      char stringpool_str142[sizeof("int_fast16_t")];
      char stringpool_str143[sizeof("sigset_t")];
      char stringpool_str146[sizeof("fpos_t")];
      char stringpool_str147[sizeof("uint8_t")];
      char stringpool_str148[sizeof("shmatt_t")];
      char stringpool_str149[sizeof("imaxdiv_t")];
      char stringpool_str151[sizeof("nfds_t")];
      char stringpool_str152[sizeof("int16_t")];
      char stringpool_str153[sizeof("locale_t")];
      char stringpool_str154[sizeof("siginfo_t")];
      char stringpool_str156[sizeof("semaphore_t")];
      char stringpool_str159[sizeof("atomic_int_fast16_t")];
      char stringpool_str160[sizeof("off_t")];
      char stringpool_str162[sizeof("ssize_t")];
      char stringpool_str163[sizeof("ssizet_t")];
      char stringpool_str164[sizeof("float64_t")];
      char stringpool_str166[sizeof("atomic_uint_least16_t")];
      char stringpool_str167[sizeof("atomic_schar")];
      char stringpool_str168[sizeof("int_least64_t")];
      char stringpool_str169[sizeof("atomic_int_least8_t")];
      char stringpool_str171[sizeof("ldiv_t")];
      char stringpool_str172[sizeof("atomic_short")];
      char stringpool_str173[sizeof("timed_mutex_t")];
      char stringpool_str174[sizeof("mbstate_t")];
      char stringpool_str179[sizeof("atomic_uint_fast8_t")];
      char stringpool_str180[sizeof("atomic_uint_least8_t")];
      char stringpool_str183[sizeof("uint16_t")];
      char stringpool_str185[sizeof("atomic_char32_t")];
      char stringpool_str186[sizeof("sa_family_t")];
      char stringpool_str188[sizeof("ushort_t")];
      char stringpool_str190[sizeof("fsfilcnt_t")];
      char stringpool_str191[sizeof("loff_t")];
      char stringpool_str192[sizeof("speed_t")];
      char stringpool_str194[sizeof("uint_least32_t")];
      char stringpool_str196[sizeof("thrd_t")];
      char stringpool_str199[sizeof("ptrdiff_t")];
      char stringpool_str201[sizeof("__m256")];
      char stringpool_str202[sizeof("__m256i")];
      char stringpool_str204[sizeof("float16_t")];
      char stringpool_str205[sizeof("atomic_int_least16_t")];
      char stringpool_str207[sizeof("int_fast32_t")];
      char stringpool_str208[sizeof("int_least16_t")];
      char stringpool_str210[sizeof("gid_t")];
      char stringpool_str213[sizeof("blkcnt_t")];
      char stringpool_str214[sizeof("pthread_cond_t")];
      char stringpool_str215[sizeof("atomic_uint_fast16_t")];
      char stringpool_str216[sizeof("wint_t")];
      char stringpool_str218[sizeof("pthread_condattr_t")];
      char stringpool_str221[sizeof("atomic_uint_least32_t")];
      char stringpool_str222[sizeof("int_least8_t")];
      char stringpool_str224[sizeof("atomic_int_fast32_t")];
      char stringpool_str226[sizeof("float16x8_t")];
      char stringpool_str227[sizeof("wchar_t")];
      char stringpool_str233[sizeof("char16_t")];
      char stringpool_str234[sizeof("wctrans_t")];
      char stringpool_str238[sizeof("uint_fast64_t")];
      char stringpool_str241[sizeof("__m512")];
      char stringpool_str242[sizeof("__m512i")];
      char stringpool_str243[sizeof("regoff_t")];
      char stringpool_str244[sizeof("in_addr_t")];
      char stringpool_str248[sizeof("syscall_arg_t")];
      char stringpool_str249[sizeof("uint128_t")];
      char stringpool_str250[sizeof("key_t")];
      char stringpool_str251[sizeof("max_align_t")];
      char stringpool_str253[sizeof("tcflag_t")];
      char stringpool_str258[sizeof("idtype_t")];
      char stringpool_str259[sizeof("float32_t")];
      char stringpool_str263[sizeof("int_least32_t")];
      char stringpool_str264[sizeof("dim3")];
      char stringpool_str266[sizeof("atomic_bool")];
      char stringpool_str270[sizeof("useconds_t")];
      char stringpool_str272[sizeof("int32_t")];
      char stringpool_str277[sizeof("int64_t")];
      char stringpool_str278[sizeof("uint_fast16_t")];
      char stringpool_str281[sizeof("float32x4_t")];
      char stringpool_str284[sizeof("blksize_t")];
      char stringpool_str287[sizeof("nlink_t")];
      char stringpool_str292[sizeof("uint_fast8_t")];
      char stringpool_str294[sizeof("uint_least64_t")];
      char stringpool_str297[sizeof("clock_t")];
      char stringpool_str299[sizeof("clockid_t")];
      char stringpool_str303[sizeof("uint32_t")];
      char stringpool_str307[sizeof("int_fast64_t")];
      char stringpool_str308[sizeof("uint64_t")];
      char stringpool_str310[sizeof("float128_t")];
      char stringpool_str311[sizeof("pthread_rwlock_t")];
      char stringpool_str314[sizeof("atomic_wchar_t")];
      char stringpool_str315[sizeof("pthread_rwlockattr_t")];
      char stringpool_str316[sizeof("__m128")];
      char stringpool_str317[sizeof("__m128i")];
      char stringpool_str318[sizeof("__m512bh")];
      char stringpool_str321[sizeof("suseconds_t")];
      char stringpool_str322[sizeof("stack_t")];
      char stringpool_str323[sizeof("int128_t")];
      char stringpool_str324[sizeof("atomic_int_fast64_t")];
      char stringpool_str325[sizeof("atomic_int_least32_t")];
      char stringpool_str327[sizeof("posix_spawnattr_t")];
      char stringpool_str329[sizeof("timed_thread_t")];
      char stringpool_str330[sizeof("atomic_int_least64_t")];
      char stringpool_str333[sizeof("uint_fast32_t")];
      char stringpool_str335[sizeof("atomic_uint_fast32_t")];
      char stringpool_str336[sizeof("posix_spawn_file_actions_t")];
      char stringpool_str340[sizeof("atomic_uint_fast64_t")];
      char stringpool_str342[sizeof("atomic_llong")];
      char stringpool_str347[sizeof("atomic_ulong")];
      char stringpool_str353[sizeof("char32_t")];
      char stringpool_str363[sizeof("wctype_t")];
      char stringpool_str373[sizeof("uint24_t")];
      char stringpool_str375[sizeof("bfloat16_t")];
      char stringpool_str398[sizeof("double_t")];
      char stringpool_str401[sizeof("glob_t")];
      char stringpool_str413[sizeof("atomic_bool32")];
      char stringpool_str429[sizeof("socklen_t")];
      char stringpool_str430[sizeof("fsblkcnt_t")];
      char stringpool_str439[sizeof("wordexp_t")];
      char stringpool_str536[sizeof("bool32")];
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
      "regex_t",
      "intmax_t",
      "uid_t",
      "rlim_t",
      "iconv_t",
      "rlimit_t",
      "uintptr_t",
      "axdx_t",
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
      "t_uscalar_t",
      "lldiv_t",
      "cpu_set_t",
      "int_fast8_t",
      "sig_atomic_t",
      "in_port_t",
      "regmatch_t",
      "mode_t",
      "errno_t",
      "atomic_ullong",
      "atomic_flag",
      "fexcept_t",
      "fenv_t",
      "atomic_int_fast8_t",
      "float_t",
      "atomic_size_t",
      "int8_t",
      "msglen_t",
      "id_t",
      "div_t",
      "data_t",
      "va_list",
      "msgqnum_t",
      "dev_t",
      "atomic_uint_least64_t",
      "posix_trace_attr_t",
      "uint_least16_t",
      "atomic_char16_t",
      "atomic_uchar",
      "atomic_ushort",
      "trace_id_t",
      "int_fast16_t",
      "sigset_t",
      "fpos_t",
      "uint8_t",
      "shmatt_t",
      "imaxdiv_t",
      "nfds_t",
      "int16_t",
      "locale_t",
      "siginfo_t",
      "semaphore_t",
      "atomic_int_fast16_t",
      "off_t",
      "ssize_t",
      "ssizet_t",
      "float64_t",
      "atomic_uint_least16_t",
      "atomic_schar",
      "int_least64_t",
      "atomic_int_least8_t",
      "ldiv_t",
      "atomic_short",
      "timed_mutex_t",
      "mbstate_t",
      "atomic_uint_fast8_t",
      "atomic_uint_least8_t",
      "uint16_t",
      "atomic_char32_t",
      "sa_family_t",
      "ushort_t",
      "fsfilcnt_t",
      "loff_t",
      "speed_t",
      "uint_least32_t",
      "thrd_t",
      "ptrdiff_t",
      "__m256",
      "__m256i",
      "float16_t",
      "atomic_int_least16_t",
      "int_fast32_t",
      "int_least16_t",
      "gid_t",
      "blkcnt_t",
      "pthread_cond_t",
      "atomic_uint_fast16_t",
      "wint_t",
      "pthread_condattr_t",
      "atomic_uint_least32_t",
      "int_least8_t",
      "atomic_int_fast32_t",
      "float16x8_t",
      "wchar_t",
      "char16_t",
      "wctrans_t",
      "uint_fast64_t",
      "__m512",
      "__m512i",
      "regoff_t",
      "in_addr_t",
      "syscall_arg_t",
      "uint128_t",
      "key_t",
      "max_align_t",
      "tcflag_t",
      "idtype_t",
      "float32_t",
      "int_least32_t",
      "dim3",
      "atomic_bool",
      "useconds_t",
      "int32_t",
      "int64_t",
      "uint_fast16_t",
      "float32x4_t",
      "blksize_t",
      "nlink_t",
      "uint_fast8_t",
      "uint_least64_t",
      "clock_t",
      "clockid_t",
      "uint32_t",
      "int_fast64_t",
      "uint64_t",
      "float128_t",
      "pthread_rwlock_t",
      "atomic_wchar_t",
      "pthread_rwlockattr_t",
      "__m128",
      "__m128i",
      "__m512bh",
      "suseconds_t",
      "stack_t",
      "int128_t",
      "atomic_int_fast64_t",
      "atomic_int_least32_t",
      "posix_spawnattr_t",
      "timed_thread_t",
      "atomic_int_least64_t",
      "uint_fast32_t",
      "atomic_uint_fast32_t",
      "posix_spawn_file_actions_t",
      "atomic_uint_fast64_t",
      "atomic_llong",
      "atomic_ulong",
      "char32_t",
      "wctype_t",
      "uint24_t",
      "bfloat16_t",
      "double_t",
      "glob_t",
      "atomic_bool32",
      "socklen_t",
      "fsblkcnt_t",
      "wordexp_t",
      "bool32"
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str38,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str40,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str41,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str42,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str43,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str44,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str46,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str49,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str50,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str56,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str57,
      -1, -1,
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
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str86,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str87,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str89,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str91,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str92,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str94,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str95,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str96,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str97,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str98,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str101,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str104,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str106,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str108,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str112,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str113,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str116,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str118,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str119,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str120,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str121,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str122,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str124,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str125,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str126,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str128,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str129,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str130,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str137,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str138,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str140,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str142,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str143,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str146,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str147,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str148,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str149,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str151,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str152,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str153,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str154,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str156,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str159,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str160,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str162,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str163,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str164,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str166,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str167,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str168,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str169,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str171,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str172,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str173,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str174,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str179,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str180,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str183,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str185,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str186,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str188,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str190,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str191,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str192,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str194,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str196,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str199,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str201,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str202,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str204,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str205,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str207,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str208,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str210,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str213,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str214,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str215,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str216,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str218,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str221,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str222,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str224,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str226,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str227,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str233,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str234,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str238,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str241,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str242,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str243,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str244,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str248,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str249,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str250,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str251,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str253,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str258,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str259,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str263,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str264,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str266,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str270,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str272,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str277,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str278,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str281,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str284,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str287,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str292,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str294,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str297,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str299,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str303,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str307,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str308,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str310,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str311,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str314,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str315,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str316,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str317,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str318,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str321,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str322,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str323,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str324,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str325,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str327,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str329,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str330,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str333,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str335,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str336,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str340,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str342,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str347,
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str353,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str363,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str373,
      -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str375,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str398,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str401,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str413,
      -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str429,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str430,
      -1, -1, -1, -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str439,
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
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str536
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
