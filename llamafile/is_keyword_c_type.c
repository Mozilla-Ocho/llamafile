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

#define TOTAL_KEYWORDS 179
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
  static const char * const wordlist[] =
    {
      "", "", "",
      "DIR",
      "FILE",
      "pid_t",
      "", "", "", "",
      "ino_t",
      "time_t",
      "timer_t",
      "intptr_t",
      "pthread_t",
      "mqd_t",
      "intN_t",
      "", "",
      "pthread_attr_t",
      "", "",
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
      "", "", "",
      "mcontext_t",
      "atomic_char",
      "iconv_t",
      "", "",
      "uid_t",
      "rlim_t",
      "",
      "rlimit_t",
      "uintptr_t",
      "", "", "", "",
      "uintmax_t",
      "t_scalar_t",
      "trace_event_id_t",
      "trace_event_set_t",
      "", "", "", "", "", "", "",
      "pthread_mutex_t",
      "atomic_long",
      "", "",
      "pthread_mutexattr_t",
      "ucontext_t",
      "atomic_uint",
      "", "", "", "",
      "atomic_uintptr_t",
      "", "", "",
      "sem_t",
      "size_t",
      "",
      "uint_least8_t",
      "",
      "register_t",
      "",
      "lldiv_t",
      "", "",
      "atomic_char32_t",
      "t_uscalar_t",
      "", "",
      "cpu_set_t",
      "",
      "int_fast8_t",
      "sig_atomic_t",
      "",
      "in_port_t",
      "",
      "mode_t",
      "errno_t",
      "atomic_ullong",
      "",
      "atomic_char16_t",
      "fenv_t",
      "int16_t",
      "",
      "fexcept_t",
      "",
      "atomic_uint_least64_t",
      "",
      "atomic_int_fast8_t",
      "uint_least16_t",
      "", "",
      "float_t",
      "",
      "id_t",
      "div_t",
      "data_t",
      "",
      "msglen_t",
      "mbstate_t",
      "dev_t",
      "atomic_uint_least32_t",
      "int_fast16_t",
      "",
      "msgqnum_t",
      "", "", "", "", "", "", "", "",
      "uint16_t",
      "",
      "trace_id_t",
      "atomic_uint_least16_t",
      "", "",
      "atomic_int_fast16_t",
      "regmatch_t",
      "", "",
      "sigset_t",
      "float64_t",
      "",
      "fpos_t",
      "",
      "int_least64_t",
      "", "",
      "nfds_t",
      "",
      "locale_t",
      "siginfo_t",
      "atomic_int_least16_t",
      "", "",
      "blkcnt_t",
      "float32_t",
      "off_t",
      "ldiv_t",
      "",
      "int_least32_t",
      "",
      "atomic_uint_fast16_t",
      "wint_t",
      "int32_t",
      "timed_mutex_t",
      "", "", "",
      "atomic_short",
      "",
      "float16_t",
      "",
      "int8_t",
      "wchar_t",
      "int_least16_t",
      "", "", "",
      "atomic_uchar",
      "atomic_ushort",
      "wctrans_t",
      "",
      "sa_family_t",
      "speed_t",
      "ushort_t",
      "uint_least32_t",
      "fsfilcnt_t",
      "loff_t",
      "",
      "shmatt_t",
      "ptrdiff_t",
      "key_t",
      "", "",
      "uint32_t",
      "", "",
      "semaphore_t",
      "int_fast32_t",
      "", "", "", "",
      "uint8_t",
      "",
      "pthread_cond_t",
      "",
      "atomic_bool",
      "atomic_schar",
      "pthread_condattr_t",
      "uint128_t",
      "", "", "",
      "uint_fast64_t",
      "atomic_int_fast32_t",
      "atomic_int_least32_t",
      "",
      "int64_t",
      "", "",
      "gid_t",
      "", "",
      "char16_t",
      "atomic_int_least8_t",
      "atomic_uint_fast32_t",
      "",
      "nlink_t",
      "uint_fast32_t",
      "in_addr_t",
      "",
      "thrd_t",
      "regex_t",
      "intmax_t",
      "atomic_uint_fast8_t",
      "atomic_uint_least8_t",
      "",
      "clock_t",
      "regoff_t",
      "clockid_t",
      "", "", "",
      "uint_fast16_t",
      "", "", "", "",
      "uint64_t",
      "", "", "", "",
      "atomic_bool32",
      "uint_least64_t",
      "",
      "pthread_rwlock_t",
      "",
      "atomic_size_t",
      "",
      "pthread_rwlockattr_t",
      "max_align_t",
      "stack_t",
      "tcflag_t",
      "",
      "useconds_t",
      "",
      "int_fast64_t",
      "", "",
      "atomic_int_least64_t",
      "", "",
      "syscall_arg_t",
      "", "", "",
      "int_least8_t",
      "idtype_t",
      "",
      "atomic_uint_fast64_t",
      "", "", "",
      "atomic_int_fast64_t",
      "", "", "",
      "char32_t",
      "", "", "", "", "", "",
      "bfloat16_t",
      "", "", "", "", "", "", "", "",
      "atomic_wchar_t",
      "", "",
      "ssize_t",
      "ssizet_t",
      "", "",
      "suseconds_t",
      "", "", "", "", "", "", "", "", "",
      "", "",
      "posix_trace_attr_t",
      "", "", "", "",
      "uint24_t",
      "", "", "", "",
      "double_t",
      "",
      "float128_t",
      "", "",
      "wctype_t",
      "imaxdiv_t",
      "", "", "",
      "int128_t",
      "", "", "",
      "uint_fast8_t",
      "", "", "", "",
      "atomic_llong",
      "", "", "",
      "glob_t",
      "atomic_ulong",
      "",
      "timed_thread_t",
      "", "", "", "", "", "", "", "", "",
      "socklen_t",
      "fsblkcnt_t",
      "bool32",
      "", "",
      "blksize_t",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "",
      "posix_spawnattr_t",
      "", "", "", "", "", "", "", "",
      "posix_spawn_file_actions_t",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "", "", "",
      "", "", "", "", "", "", "",
      "wordexp_t"
    };

  if (len <= MAX_WORD_LENGTH && len >= MIN_WORD_LENGTH)
    {
      register unsigned int key = hash (str, len);

      if (key <= MAX_HASH_VALUE)
        {
          register const char *s = wordlist[key];

          if (*str == *s && !strncmp (str + 1, s + 1, len - 1) && s[len] == '\0')
            return s;
        }
    }
  return 0;
}
