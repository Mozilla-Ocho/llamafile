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

#include "net.h"
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

namespace lf {

static bool Tune(int fd, int a, int b, int x) {
    if (!b)
        return false;
    return setsockopt(fd, a, b, &x, sizeof(x)) != -1;
}

/**
 * Returns new socket with modern goodness enabled.
 */
int socket(int family, int type, int protocol, bool isserver, const struct timeval *timeout) {
    int e, fd;
    if ((fd = ::socket(family, type, protocol)) != -1) {
        e = errno;
        if (isserver) {
            Tune(fd, SOL_TCP, TCP_FASTOPEN, 100);
            Tune(fd, SOL_SOCKET, SO_REUSEADDR, 1);
        } else {
            Tune(fd, SOL_TCP, TCP_FASTOPEN_CONNECT, 1);
        }
        errno = e;
        if (!Tune(fd, SOL_TCP, TCP_QUICKACK, 1)) {
            e = errno;
            Tune(fd, SOL_TCP, TCP_NODELAY, 1);
            errno = e;
        }
        if (timeout) {
            e = errno;
            if (timeout->tv_sec < 0) {
                Tune(fd, SOL_SOCKET, SO_KEEPALIVE, 1);
                Tune(fd, SOL_TCP, TCP_KEEPIDLE, -timeout->tv_sec);
                Tune(fd, SOL_TCP, TCP_KEEPINTVL, -timeout->tv_sec);
            } else {
                setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, timeout, sizeof(*timeout));
                setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, timeout, sizeof(*timeout));
            }
            errno = e;
        }
    }
    return fd;
}

} // namespace lf
