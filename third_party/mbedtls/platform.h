#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_PLATFORM_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_PLATFORM_H_
#include <libc/assert.h>
#include <libc/calls/calls.h>
#include <libc/intrin/likely.h>
#include <libc/mem/mem.h>
#include <libc/runtime/runtime.h>
#include <libc/stdio/stdio.h>
#include "third_party/mbedtls/config.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_EXIT_SUCCESS 0
#define MBEDTLS_EXIT_FAILURE 1

#define mbedtls_free              free
#define mbedtls_calloc            calloc
#define mbedtls_snprintf          snprintf
#define mbedtls_vsnprintf         vsnprintf
#define mbedtls_exit              exit
#define mbedtls_time_t            int64_t
#define mbedtls_time              time
#define mbedtls_platform_gmtime_r gmtime_r

#define mbedtls_fprintf(...) ((void)0)
#define mbedtls_printf(...)  ((void)0)

#ifdef MBEDTLS_CHECK_PARAMS
#define MBEDTLS_PARAM_FAILED(cond) \
  mbedtls_param_failed(#cond, __FILE__, __LINE__)
#else
#define MBEDTLS_PARAM_FAILED(cond) __builtin_unreachable()
#endif

#define MBEDTLS_INTERNAL_VALIDATE_RET(cond, ret) \
  do {                                           \
    if (UNLIKELY(!(cond))) {                     \
      MBEDTLS_PARAM_FAILED(cond);                \
      return ret;                                \
    }                                            \
  } while (0)

#define MBEDTLS_INTERNAL_VALIDATE(cond) \
  do {                                  \
    if (UNLIKELY(!(cond))) {            \
      MBEDTLS_PARAM_FAILED(cond);       \
      return;                           \
    }                                   \
  } while (0)

#if IsModeDbg()
#define MBEDTLS_ASSERT(EXPR) \
  ((void)((EXPR) || (__assert_fail(#EXPR, __FILE__, __LINE__), 0)))
#else
#define MBEDTLS_ASSERT(EXPR) unassert(EXPR)
#endif

typedef struct mbedtls_platform_context {
  char dummy;
} mbedtls_platform_context;

void mbedtls_platform_zeroize(void *, size_t);
int mbedtls_platform_setup(mbedtls_platform_context *);
void mbedtls_platform_teardown(mbedtls_platform_context *);
void mbedtls_param_failed(const char *, const char *, int) relegated;

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_PLATFORM_H_ */
