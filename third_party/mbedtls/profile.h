#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_PROFILE_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_PROFILE_H_
#include <libc/intrin/safemacros.h>
#include <libc/log/log.h>
#include <libc/nexgen32e/bench.h>
#include <libc/nexgen32e/rdtsc.h>
#if 1

#define START() \
  {             \
    volatile uint64_t Time = __startbench()
#define STOP(x)                                      \
  fprintf(stderr, "PROFILE %,10ldc %s\n",            \
          unsignedsubtract(__endbench(), Time), #x); \
  }

#define PROFILE(x) \
  ({               \
    typeof(x) Res; \
    START();       \
    Res = (x);     \
    STOP(x);       \
    Res;           \
  })
#define PROFILS(x) \
  do {             \
    START();       \
    x;             \
    STOP(x);       \
  } while (0)
#define PRINT() \
  fprintf(stderr, "PRINT %s called by %s\n", __FUNCTION__, GetCallerName(0))

#else

#define PRINT()    ((void)0)
#define PROFILE(x) x
#define PROFILS(x) x
#define START()    ((void)0)
#define STOP(x)    ((void)0)

#endif
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_PROFILE_H_ */
