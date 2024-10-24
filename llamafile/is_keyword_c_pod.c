/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf --output-file=llamafile/is_keyword_c_pod.c llamafile/is_keyword_c_pod.gperf  */
/* Computed positions: -k'' */

#line 1 "llamafile/is_keyword_c_pod.gperf"

#include <string.h>

#define TOTAL_KEYWORDS 3
#define MIN_WORD_LENGTH 4
#define MAX_WORD_LENGTH 6
#define MIN_HASH_VALUE 4
#define MAX_HASH_VALUE 6
/* maximum key range = 3, duplicates = 0 */

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
is_keyword_c_pod (register const char *str, register size_t len)
{
  struct stringpool_t
    {
      char stringpool_str4[sizeof("enum")];
      char stringpool_str5[sizeof("union")];
      char stringpool_str6[sizeof("struct")];
    };
  static const struct stringpool_t stringpool_contents =
    {
      "enum",
      "union",
      "struct"
    };
  #define stringpool ((const char *) &stringpool_contents)
  static const int wordlist[] =
    {
      -1, -1, -1, -1,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str4,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str5,
      (int)(size_t)&((struct stringpool_t *)0)->stringpool_str6
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
