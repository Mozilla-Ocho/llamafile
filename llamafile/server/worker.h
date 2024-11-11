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
#include "client.h"
#include <cosmo.h>
#include <pthread.h>

#define WORKER(e) DLL_CONTAINER(Worker, elem_, e)

struct llama_model;

namespace lf {
namespace server {

struct Server;

struct Worker
{
    Server* server_;
    Dll elem_;
    pthread_t th_ = 0;
    bool working_ = false;
    Client client_;

    explicit Worker(Server*, llama_model*);
    void run();
    void begin();
    void handle();
    void end();
    void deprioritize();
    void retire();
    void kill();
};

} // namespace server
} // namespace lf
