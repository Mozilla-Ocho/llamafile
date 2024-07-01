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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Double linked list.
 *
 * This is the same thing as `Dll` except it does not need `offsetof()`,
 * which can't be used for non-POD C++ object types.
 */
struct Dll3 {
    struct Dll3 *next;
    struct Dll3 *prev;
    void *container;
};

static inline void dll3_init(struct Dll3 *e, void *container) {
    e->next = e;
    e->prev = e;
    e->container = container;
}

static inline int dll3_is_alone(struct Dll3 *e) {
    return e->next == e && e->prev == e;
}

static inline int dll3_is_empty(struct Dll3 *list) {
    return !list;
}

static inline struct Dll3 *dll3_last(struct Dll3 *list) {
    return list;
}

static inline struct Dll3 *dll3_first(struct Dll3 *list) {
    struct Dll3 *first = 0;
    if (list)
        first = list->next;
    return first;
}

static inline struct Dll3 *dll3_next(struct Dll3 *list, struct Dll3 *e) {
    struct Dll3 *next = 0;
    if (e != list)
        next = e->next;
    return next;
}

static inline struct Dll3 *dll3_prev(struct Dll3 *list, struct Dll3 *e) {
    struct Dll3 *prev = 0;
    if (e != list->next)
        prev = e->prev;
    return prev;
}

void dll3_remove(struct Dll3 **, struct Dll3 *);
void dll3_make_last(struct Dll3 **, struct Dll3 *);
void dll3_make_first(struct Dll3 **, struct Dll3 *);
void dll3_splice_after(struct Dll3 *, struct Dll3 *);

#ifdef __cplusplus
}
#endif
