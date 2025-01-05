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
#include <atomic>
#include <cosmo.h>
#include <pthread.h>

struct llama_model;

namespace lf {
namespace server {

struct Slots;

struct Server
{
    Server(int, Slots*, llama_model*);
    ~Server();

    int accept(unsigned*);
    errno_t spawn();
    void terminate();
    void shutdown();
    int close();
    void run();
    void lock();
    void unlock();
    void signal();
    void wait();

    int fd;
    Slots* slots_;
    llama_model* model_;
    Dll* idle_workers = nullptr;
    Dll* active_workers = nullptr;
    pthread_cond_t cond_ = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t lock_ = PTHREAD_MUTEX_INITIALIZER;
    std::atomic_int worker_count = ATOMIC_VAR_INIT(0);
    std::atomic_bool terminated = ATOMIC_VAR_INIT(false);
};

extern Server* g_server;

int
create_listening_socket(const char*, unsigned*, int*);

} // namespace server
} // namespace lf
