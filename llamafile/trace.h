#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void llamafile_trace_set_pid(int);
void llamafile_trace_set_tid(int);
void llamafile_trace_begin(const char *);
void llamafile_trace_end(const char *);

#ifdef __cplusplus
}
#endif
