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

#include "core_manager.h"

#include <assert.h>

#include "llama.cpp/cores.h"

CoreManager g_core_manager;

CoreManager::CoreManager()
    : used_(0),
      total_(cpu_get_num_math()),
      cv_(PTHREAD_COND_INITIALIZER),
      mu_(PTHREAD_MUTEX_INITIALIZER) {
}

static void unlock_mutex(void *arg) {
    pthread_mutex_t *mu = (pthread_mutex_t *)arg;
    pthread_mutex_unlock(mu);
}

int CoreManager::acquire(int need, int greed) {
    npassert(need >= 1);
    npassert(greed >= need);

    int got = 0;

    while (got < need) {
        pthread_mutex_lock(&mu_);
        pthread_cleanup_push(unlock_mutex, &mu_);
        if (used_ < total_) {
            ++got;
            ++used_;
        } else {
            pthread_cond_wait(&cv_, &mu_);
        }
        pthread_cleanup_pop(true);
    }

    while (got < greed) {
        errno_t err;
        if (pthread_mutex_trylock(&mu_))
            break;
        if (used_ < total_) {
            ++got;
            ++used_;
        } else {
            greed = got;
        }
        pthread_mutex_unlock(&mu_);
    }

    return got;
}

void CoreManager::release(int count) {
    bool ok;
    pthread_mutex_lock(&mu_);
    if ((used_ -= count) >= 0) {
        ok = true;
    } else {
        ok = false;
        used_ = 0;
    }
    pthread_cond_signal(&cv_);
    pthread_mutex_unlock(&mu_);
    npassert(ok);
}
