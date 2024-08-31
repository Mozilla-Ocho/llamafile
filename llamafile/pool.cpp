// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pool.h"

#include <assert.h>
#include <cosmo.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <unistd.h>

#include "threadlocal.h"

struct llamafile_thread;
static void llamafile_thread_canceled(llamafile_thread *);
static ThreadLocal<llamafile_thread> g_key(llamafile_thread_canceled);

struct llamafile_task {
    _Atomic(pthread_t) th = -1;
    pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    void *(*func)(void *);
    void *arg;
    void *res;
};

struct llamafile_thread {
    _Atomic(pthread_t) th = -1;
    pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    llamafile_task *task;
    llamafile_thread *next;
};

static atomic_int g_active;
static atomic_uintptr_t g_idle;

#define MASQUE 0x00fffffffffffff0
#define PTR(x) ((uintptr_t)(x) & MASQUE)
#define TAG(x) ROL((uintptr_t)(x) & ~MASQUE, 8)
#define ABA(p, t) ((uintptr_t)(p) | (ROR((uintptr_t)(t), 8) & ~MASQUE))
#define ROL(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
#define ROR(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

static void idle_push(llamafile_thread *thread) {
    uintptr_t tip;
    unassert(!TAG(thread));
    tip = atomic_load_explicit(&g_idle, memory_order_relaxed);
    for (;;) {
        thread->next = (llamafile_thread *)PTR(tip);
        if (atomic_compare_exchange_weak_explicit(&g_idle, &tip, ABA(thread, TAG(tip) + 1),
                                                  memory_order_release, memory_order_relaxed))
            break;
    }
}

static llamafile_thread *idle_pop(void) {
    uintptr_t tip;
    llamafile_thread *thread;
    tip = atomic_load_explicit(&g_idle, memory_order_relaxed);
    while ((thread = (llamafile_thread *)PTR(tip)))
        if (atomic_compare_exchange_weak_explicit(&g_idle, &tip, ABA(thread->next, TAG(tip) + 1),
                                                  memory_order_acquire, memory_order_relaxed))
            break;
    return thread;
}

static void cancel_task(llamafile_task *task) {
    pthread_mutex_lock(&task->mu);
    task->res = PTHREAD_CANCELED;
    atomic_store_explicit(&task->th, 0, memory_order_release);
    pthread_cond_signal(&task->cv);
    pthread_mutex_unlock(&task->mu);
}

static void llamafile_thread_canceled(llamafile_thread *thread) {
    atomic_store_explicit(&thread->th, 0, memory_order_release);
    cancel_task(thread->task);
    delete thread;
    --g_active;
}

static void *llamafile_thread_worker(void *arg) {
    errno_t err;
    llamafile_thread *thread = (llamafile_thread *)arg;

    ++g_active;
    g_key.set(thread);
    do {
        void *res = thread->task->func(thread->task->arg);
        pthread_setcancelstate(PTHREAD_CANCEL_MASKED, 0);

        for (;;)
            if (atomic_load_explicit(&thread->th, memory_order_acquire) != -1)
                if (atomic_load_explicit(&thread->task->th, memory_order_acquire) != -1)
                    break;

        pthread_mutex_lock(&thread->task->mu);
        thread->task->res = res;
        atomic_store_explicit(&thread->task->th, 0, memory_order_release);
        pthread_cond_signal(&thread->task->cv);
        pthread_mutex_unlock(&thread->task->mu);

        pthread_mutex_lock(&thread->mu);
        thread->task = nullptr;
        idle_push(thread);
        while (!thread->task) {
            err = pthread_cond_wait(&thread->cv, &thread->mu);
            if (err == ECANCELED)
                break;
        }
        pthread_mutex_unlock(&thread->mu);
        pthread_setcancelstate(PTHREAD_CANCEL_DEFERRED, 0);
    } while (err != ECANCELED);

    if (thread->task)
        cancel_task(thread->task);

    atomic_store_explicit(&thread->th, 0, memory_order_release);
    g_key.set(nullptr);
    delete thread;
    --g_active;

    return 0;
}

static errno_t llamafile_thread_create(llamafile_task *task) {
    llamafile_thread *thread = new llamafile_thread;
    thread->task = task;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 128 * 1024);
    pthread_attr_setguardsize(&attr, sysconf(_SC_PAGESIZE));
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setsigaltstacksize_np(&attr, sysconf(_SC_MINSIGSTKSZ) + 32768);
    errno_t err = pthread_create((pthread_t *)&thread->th, &attr, llamafile_thread_worker, thread);
    pthread_attr_destroy(&attr);
    if (!err) {
        atomic_store_explicit(&task->th, atomic_load_explicit(&thread->th, memory_order_relaxed),
                              memory_order_release);
    } else {
        delete thread;
    }
    return err;
}

errno_t llamafile_task_create(llamafile_task **out_task, void *(*func)(void *), void *arg) {
    llamafile_task *task = new llamafile_task;
    task->func = func;
    task->arg = arg;
    errno_t err;
    llamafile_thread *thread;
    if ((thread = idle_pop())) {
        pthread_mutex_lock(&thread->mu);
        atomic_store_explicit(&task->th, atomic_load_explicit(&thread->th, memory_order_relaxed),
                              memory_order_release);
        thread->task = task;
        pthread_cond_signal(&thread->cv);
        pthread_mutex_unlock(&thread->mu);
        err = 0;
    } else {
        err = llamafile_thread_create(task);
    }
    if (!err) {
        *out_task = task;
    } else {
        delete task;
    }
    return err;
}

static void unlock_mutex(void *arg) {
    pthread_mutex_t *mu = (pthread_mutex_t *)arg;
    pthread_mutex_unlock(mu);
}

errno_t llamafile_task_join(llamafile_task *task, void **out_res) {
    pthread_cleanup_push(unlock_mutex, &task->mu);
    pthread_mutex_lock(&task->mu);
    while (atomic_load_explicit(&task->th, memory_order_acquire))
        pthread_cond_wait(&task->cv, &task->mu);
    pthread_cleanup_pop(true);
    if (out_res)
        *out_res = task->res;
    delete task;
    return 0;
}

errno_t llamafile_task_cancel(llamafile_task *task) {
    pthread_t th;
    errno_t err = 0;
    if ((th = atomic_load_explicit(&task->th, memory_order_acquire))) {
        err = pthread_cancel(th);
    } else {
        err = ESRCH;
    }
    return err;
}

void llamafile_task_shutdown(void) {
    pthread_t th;
    int backoff = 0;
    llamafile_thread *thread;
    for (;;) {
        while ((thread = idle_pop()))
            if ((th = atomic_load_explicit(&thread->th, memory_order_acquire)))
                pthread_cancel(th);
        if (!g_active)
            break;
        backoff = pthread_delay_np(&g_idle, backoff);
    }
}

static struct llamafile_tasks {
    ~llamafile_tasks(void) {
        llamafile_task_shutdown();
    }
} g_tasks;
