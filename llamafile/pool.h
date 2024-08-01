#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct llamafile_task *llamafile_task_t;

errno_t llamafile_task_create(llamafile_task_t *, void *(*)(void *), void *);
errno_t llamafile_task_join(llamafile_task_t, void **);
errno_t llamafile_task_cancel(llamafile_task_t);
void llamafile_task_shutdown(void);

#ifdef __cplusplus
}
#endif
