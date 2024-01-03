#ifndef LLAMAFILE_X_H_
#define LLAMAFILE_X_H_
#include <stdarg.h>
#ifdef __cplusplus
extern "C" {
#endif

char *xasprintf(const char *, ...);
char *xvasprintf(const char *, va_list);

#ifdef __cplusplus
}
#endif
#endif /* LLAMAFILE_X_H_ */
