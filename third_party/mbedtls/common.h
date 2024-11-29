#ifndef MBEDTLS_LIBRARY_COMMON_H
#define MBEDTLS_LIBRARY_COMMON_H
#include "third_party/mbedtls/config.h"

#ifdef MBEDTLS_TEST_HOOKS
#define MBEDTLS_STATIC_TESTABLE
#else
#define MBEDTLS_STATIC_TESTABLE static
#endif

#endif /* MBEDTLS_LIBRARY_COMMON_H */
