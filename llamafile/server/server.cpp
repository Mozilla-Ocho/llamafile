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

#include "server.h"
#include "llamafile/crash.h"
#include "llamafile/llamafile.h"
#include "llamafile/server/log.h"
#include "llamafile/server/server.h"
#include "llamafile/server/slots.h"
#include "llamafile/server/worker.h"
#include <cassert>
#include <cstdio>
#include <ctime>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace lf {
namespace server {

Server::Server(int fd, Slots* slots, llama_model* model)
  : fd(fd), slots_(slots), model_(model)
{
}

Server::~Server()
{
    npassert(fd == -1);
    npassert(!worker_count.load(std::memory_order_relaxed));
    npassert(dll_is_empty(active_workers));
    npassert(dll_is_empty(idle_workers));
    pthread_mutex_destroy(&lock_);
    pthread_cond_destroy(&cond_);
}

void
Server::lock()
{
    pthread_mutex_lock(&lock_);
}

void
Server::signal()
{
    pthread_cond_signal(&cond_);
}

void
Server::wait()
{
    pthread_cond_wait(&cond_, &lock_);
}

void
Server::unlock()
{
    pthread_mutex_unlock(&lock_);
}

void
Server::terminate()
{
    terminated.store(true, std::memory_order_release);
    signal();
}

int
Server::close()
{
    int rc = 0;
    if (fd != -1) {
        rc = ::close(fd);
        fd = -1;
    }
    return rc;
}

void*
worker_thread(void* arg)
{
    Worker* worker = (Worker*)arg;
    worker->run();
    return 0;
}

errno_t
Server::spawn()
{
    errno_t err;
    Worker* worker;
    pthread_attr_t attr;
    worker = new Worker(this, model_);
    pthread_attr_init(&attr);
    pthread_attr_setguardsize(&attr, sysconf(_SC_PAGESIZE));
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setsigaltstacksize_np(&attr, sysconf(_SC_MINSIGSTKSZ) + 16384);
    if ((err = pthread_create(&worker->th_, &attr, worker_thread, worker)))
        delete worker;
    pthread_attr_destroy(&attr);
    return err;
}

int
Server::accept(unsigned* out_ip)
{
    // accept connection
    sockaddr_in clientaddr;
    set_thread_name("listen");
    uint32_t clientsize = sizeof(clientaddr);
    int clifd = ::accept(fd, (sockaddr*)&clientaddr, &clientsize);
    if (clifd == -1)
        return -1;

    // set name
    char name[17];
    int port = ntohs(clientaddr.sin_port);
    unsigned ip = ntohl(clientaddr.sin_addr.s_addr);
    if (ip == 0x7f000001) {
        snprintf(name, sizeof(name), "%hu", port);
    } else {
        snprintf(name,
                 sizeof(name),
                 "%hhu.%hhu.%hhu.%hhu",
                 ip >> 24,
                 ip >> 16,
                 ip >> 8,
                 ip);
    }
    set_thread_name(name);

    // keep sockets open
    if (FLAG_keepalive > 0) {
        int yes = 1;
        int secs = FLAG_keepalive;
        setsockopt(clifd, SOL_SOCKET, SO_KEEPALIVE, &yes, sizeof(yes));
        setsockopt(clifd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        setsockopt(clifd, IPPROTO_TCP, TCP_KEEPIDLE, &secs, sizeof(secs));
        setsockopt(clifd, IPPROTO_TCP, TCP_KEEPINTVL, &secs, sizeof(secs));
    }

    if (FLAG_verbose >= 2)
        SLOG("accept");
    if (out_ip)
        *out_ip = ip;
    return clifd;
}

void
Server::run()
{
    while (!terminated.load(std::memory_order_acquire)) {
        lock();
        if (!terminated.load(std::memory_order_acquire))
            wait();
        unlock();
        if (terminated.load(std::memory_order_acquire))
            break;
        int missing =
          FLAG_workers - worker_count.load(std::memory_order_acquire);
        for (int i = 0; i < missing; ++i)
            spawn();
    }
}

void
Server::shutdown()
{
    // on windows this is the only way accept() can be canceled
    if (IsWindows())
        close();

    // kill workers
    lock();
    for (Dll* e = dll_first(idle_workers); e; e = dll_next(idle_workers, e))
        WORKER(e)->kill();
    for (Dll* e = dll_first(active_workers); e; e = dll_next(active_workers, e))
        WORKER(e)->kill();
    unlock();

    // wait for workers to die
    while (worker_count.load(std::memory_order_acquire) > 0) {
        lock();
        if (worker_count.load(std::memory_order_acquire) > 0)
            wait();
        unlock();
    }
}

} // namespace server
} // namespace lf
