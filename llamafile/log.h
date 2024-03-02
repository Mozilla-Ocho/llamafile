#ifndef LLAMAFILE_LOG_H_
#define LLAMAFILE_LOG_H_
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

extern bool FLAG_log_disable;

void tinylog(const char *, ...);

#define tinylog(...) (void)(!FLAG_log_disable && (tinylog(__VA_ARGS__), 0))
#define tinylogf(...) (void)(!FLAG_log_disable && (fprintf(stderr, __VA_ARGS__), 0))

#ifdef __cplusplus
}
#endif
#endif /* LLAMAFILE_LOG_H_ */
