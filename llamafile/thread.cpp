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

#include "thread.h"

#include <stdatomic.h>
#include <stdlib.h>

#include "dll3.h"
#include "lockable.h"
#include "refcounted.h"
#include "threadlocal.h"

struct Parent : public Lockable, public RefCounted<Parent> {
    Dll3 *children;

    Parent() : children(nullptr) {
    }

    ~Parent() {
        if (children)
            __builtin_trap();
    }
};

struct Child : public RefCounted<Child> {
    pthread_t thread;
    int detach_state;
    _Atomic(Parent *) parent;
    void *(*start_routine)(void *);
    void *arg;
    Dll3 elem; // owned by parent lock

    Child(Parent *p, int detach_state, void *(*start_routine)(void *), void *arg)
        : thread(0),
          detach_state(detach_state),
          parent(ATOMIC_VAR_INIT(p->ref())),
          start_routine(start_routine),
          arg(arg) {
        dll3_init(&elem, this);
        p->lock();
        dll3_make_last(&p->children, &elem);
        p->unlock();
    }

    ~Child() {
        unregister();
    }

    void unregister() {
        Parent *p;
        if ((p = atomic_exchange(&parent, nullptr))) {
            p->lock();
            dll3_remove(&p->children, &elem);
            p->unlock();
            p->unref(); // remove child's reference to parent
            unref(); // remove parent's reference to child
        }
    }
};

static void on_child_exit(Child *child) {
    child->unregister();
    child->unref(); // child frees self
}

static ThreadLocal<Child> g_child(on_child_exit);

static void on_parent_exit(Parent *parent) {
    Dll3 *e;
    parent->lock();
    while ((e = dll3_first(parent->children))) {
        Child *child = (Child *)e->container;
        if ((parent = atomic_exchange(&child->parent, nullptr))) {
            dll3_remove(&parent->children, e);
            parent->unlock();
            parent->unref(); // remove child's reference to parent
            pthread_cancel(child->thread);
            if (child->detach_state == PTHREAD_CREATE_JOINABLE)
                if (pthread_join(child->thread, 0))
                    __builtin_trap();
            child->unref(); // remove parent's reference to child
            parent->lock();
        }
    }
    parent->unlock();
    parent->unref(); // parent frees self
}

static ThreadLocal<Parent> g_parent(on_parent_exit);

static void *llamafile_thread(void *arg) {
    Child *child;
    g_child.set((child = (Child *)arg));
    return child->start_routine(child->arg);
}

/**
 * Creates managed thread.
 *
 * Threads created by this function will be canceled automatically, when
 * the current thread exits or is itself canceled by another thread. The
 * killing hierarchy works for both detached and joinable threads. It is
 * a good idea for threads to put themselves in asynchronous cancelation
 * mode whilst performing mathematical operations. Threads that block on
 * i/o or semaphores should use the default deferred cancelation instead
 * since it helps avoid memory leaks and other badness.
 *
 * @return 0 on success, or errno on error
 */
errno_t llamafile_thread_create(pthread_t *thread, const pthread_attr_t *attr,
                                void *(*start_routine)(void *), void *arg) {
    errno_t err;
    Parent *parent;
    int detach_state = PTHREAD_CREATE_JOINABLE;
    if (attr)
        pthread_attr_getdetachstate(attr, &detach_state);
    if (!(parent = g_parent.get()))
        g_parent.set((parent = new Parent));
    Child *child = new Child(parent, detach_state, start_routine, arg);
    child->ref(); // acquire ownership for parent destructor
    child->ref(); // ensure ownership for rest of this function
    err = pthread_create(&child->thread, attr, llamafile_thread, child);
    if (!err) {
        *thread = child->thread;
    } else {
        child->unref(); // release reference for parent destructor
        child->unref(); // release reference for child thread
    }
    child->unref(); // release reference for this function
    return err;
}
