// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#include "dll3.h"

/**
 * Makes `succ` and its successors come after `elem`.
 *
 * It's required that `elem` and `succ` aren't part of the same list.
 */
void dll3_splice_after(struct Dll3 *elem, struct Dll3 *succ) {
    struct Dll3 *tmp1, *tmp2;
    tmp1 = elem->next;
    tmp2 = succ->prev;
    elem->next = succ;
    succ->prev = elem;
    tmp2->next = tmp1;
    tmp1->prev = tmp2;
}

/**
 * Removes item from doubly-linked list.
 *
 * @param list is a doubly-linked list, where `!*list` means empty
 */
void dll3_remove(struct Dll3 **list, struct Dll3 *elem) {
    if (*list == elem) {
        if ((*list)->prev == *list) {
            *list = 0;
        } else {
            *list = (*list)->prev;
        }
    }
    elem->next->prev = elem->prev;
    elem->prev->next = elem->next;
    elem->next = elem;
    elem->prev = elem;
}

/**
 * Inserts items into list, at the beginning.
 *
 * The resulting list will start with `elem`, followed by other items in
 * `elem`, followed by the items previously in `*list`.
 *
 * @param list is a doubly-linked list, where `!*list` means empty
 * @param elem must not be a member of `list`, or null for no-op
 */
void dll3_make_first(struct Dll3 **list, struct Dll3 *elem) {
    if (elem) {
        if (!*list) {
            *list = elem->prev;
        } else {
            dll3_splice_after(*list, elem);
        }
    }
}

/**
 * Inserts items into list, at the end.
 *
 * The resulting `*list` will end with `elem`, preceded by the other
 * items in `elem`, preceded by the items previously in `*list`.
 *
 * @param list is a doubly-linked list, where `!*list` means empty
 * @param elem must not be a member of `list`, or null for no-op
 */
void dll3_make_last(struct Dll3 **list, struct Dll3 *elem) {
    if (elem) {
        dll3_make_first(list, elem->next);
        *list = elem;
    }
}
