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

#include "worker.h"

#include <assert.h>
#include <exception>
#include <pthread.h>

#include "client.h"
#include "llamafile/llamafile.h"
#include "log.h"
#include "signals.h"

Worker::Worker(Server* server) : server(server)
{
    dll_init(&elem);
}

void
Worker::kill()
{
    pthread_cancel(th);
}

void
Worker::begin()
{
    unassert(!working);
    server->lock();
    dll_remove(&server->idle_workers, &elem);
    if (dll_is_empty(server->idle_workers)) {
        Dll* slowbro;
        if ((slowbro = dll_last(server->active_workers))) {
            LOG("all threads active! dropping oldest client");
            WORKER(slowbro)->kill();
        }
    }
    working = true;
    dll_make_first(&server->active_workers, &elem);
    server->unlock();
}

void
Worker::end()
{
    unassert(working);
    server->lock();
    dll_remove(&server->active_workers, &elem);
    working = false;
    dll_make_first(&server->idle_workers, &elem);
    server->unlock();
}

void
Worker::retire()
{
    server->lock();
    if (working)
        dll_remove(&server->active_workers, &elem);
    else
        dll_remove(&server->idle_workers, &elem);
    server->worker_count.fetch_sub(1, std::memory_order_acq_rel);
    server->signal();
    server->unlock();
    delete this;
}

void
Worker::handle(void)
{
    if ((client.fd = server->accept()) == -1) {
        LOG("accept returned %m");
        return;
    }

    begin();
    pthread_cleanup_push(
      [](void* arg) {
          Worker* worker = (Worker*)arg;
          worker->client.close();
          worker->end();
      },
      this);

    try {
        client.run();
    } catch (const std::exception& e) {
        LOG("caught %s", e.what());
    } catch (...) {
        LOG("caught unknown exception");
    }

    pthread_cleanup_pop(true);
}

void
Worker::run()
{
    server->lock();
    dll_make_first(&server->idle_workers, &elem);
    server->worker_count.fetch_add(1, std::memory_order_acq_rel);
    server->unlock();

    pthread_cleanup_push(
      [](void* arg) {
          Worker* worker = (Worker*)arg;
          worker->retire();
      },
      this);

    while (!server->terminated.load(std::memory_order_acquire)) {
        sigset_t mask;
        sigemptyset(&mask);
        sigaddset(&mask, SIGHUP);
        sigaddset(&mask, SIGINT);
        sigaddset(&mask, SIGQUIT);
        sigaddset(&mask, SIGTERM);
        sigaddset(&mask, SIGUSR1);
        sigaddset(&mask, SIGALRM);
        pthread_sigmask(SIG_SETMASK, &mask, 0);
        handle();
    }

    pthread_cleanup_pop(true);
}
