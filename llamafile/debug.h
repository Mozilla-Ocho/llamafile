#ifndef LLAMAFILE_DEBUG_H_
#define LLAMAFILE_DEBUG_H_
#include <threads.h>
#ifdef __cplusplus
extern "C" {
#endif

struct StackFrame;
struct ggml_cgraph;
int llamafile_trapping_enabled(int);
void llamafile_trapping_restore(void);
void ShowBacktrace(int, const struct StackFrame *);
extern const struct ggml_cgraph *llamafile_debug_graph;
extern thread_local int llamafile_debug_op_index;

#ifdef __cplusplus
}
#endif
#endif /* LLAMAFILE_DEBUG_H_ */
