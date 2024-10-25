/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_ld_warning.c llamafile/is_keyword_ld_warning.gperf  */
/* Computed positions: -k'' */

#line 1 "llamafile/is_keyword_ld_warning.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 3
#define MIN_WORD_LENGTH 5
#define MAX_WORD_LENGTH 12
#define MIN_HASH_VALUE 5
#define MAX_HASH_VALUE 12
/* maximum key range = 8, duplicates = 0 */

#ifdef __GNUC__
__inline
#else
#ifdef __cplusplus
inline
#endif
#endif
/*ARGSUSED*/
static unsigned int
hash (register const char *str, register size_t len)
{
  return len;
}

const char *
is_keyword_ld_warning (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str5[sizeof(":NONE")];
      char stringpool_str9[sizeof("/DISCARD/")];
      char stringpool_str12[sizeof("EXCLUDE_FILE")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      ":NONE",
      "/DISCARD/",
      "EXCLUDE_FILE"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str9,
      -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str12
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
