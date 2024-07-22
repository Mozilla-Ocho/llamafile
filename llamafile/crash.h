#ifndef LLAMAFILE_CRASH_H_
#define LLAMAFILE_CRASH_H_
#include <signal.h>
#ifdef __cplusplus
extern "C" {
#endif

struct StackFrame;

char *describe_crash(char *, size_t, int, siginfo_t *, void *);
char *describe_backtrace(char *, size_t, const struct StackFrame *);

#ifdef __cplusplus
}
#endif
#endif /* LLAMAFILE_CRASH_H_ */
