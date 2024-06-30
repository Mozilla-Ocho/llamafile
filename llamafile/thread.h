#ifndef LLAMAFILE_THREAD_H_
#define LLAMAFILE_THREAD_H_
#include <pthread.h>
#ifdef __cplusplus
extern "C" {
#endif

errno_t llamafile_thread_create(pthread_t *thread, const pthread_attr_t *attr,
                                void *(*start_routine)(void *), void *arg);

#ifdef __cplusplus
}
#endif
#endif /* LLAMAFILE_THREAD_H_ */
