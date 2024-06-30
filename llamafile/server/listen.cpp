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

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#include "log.h"

void
print_listening_url(unsigned ip, int port)
{
    LOG("listen http://%hhu.%hhu.%hhu.%hhu:%hu",
        ip >> 24,
        ip >> 16,
        ip >> 8,
        ip,
        port);
}

int
create_listening_socket(const char* hostport)
{
    // parse hostname:port
    char* p;
    char* host;
    char* port;
    char addr[128];
    strlcpy(addr, hostport, sizeof(addr));
    if ((p = strrchr(addr, ':'))) {
        *p = '\0';
        host = addr;
        port = p + 1;
    } else {
        host = NULL;
        port = addr;
    }

    // turn listen address names into numbers
    int status;
    struct addrinfo* ai;
    struct addrinfo hints = {
        .ai_family = AF_INET,
        .ai_socktype = SOCK_STREAM,
        .ai_protocol = IPPROTO_TCP,
    };
    if ((status = getaddrinfo(host, port, &hints, &ai))) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(status));
        exit(1);
    }

    // create socket
    int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd == -1) {
        perror(hostport);
        exit(1);
    }

    // these fail on some platforms but it's harmless
    int yes = 1;
    int qlen = 5;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    setsockopt(fd, IPPROTO_TCP, TCP_FASTOPEN, &qlen, sizeof(qlen));

    // bind the socket
    if (bind(fd, (struct sockaddr*)ai->ai_addr, ai->ai_addrlen) == -1) {
        perror(hostport);
        exit(1);
    }

    // listen for connections
    if (listen(fd, SOMAXCONN)) {
        perror(hostport);
        exit(1);
    }

    // print listening urls
    if (getsockname(fd, (struct sockaddr*)ai->ai_addr, &ai->ai_addrlen)) {
        perror(hostport);
        exit(1);
    }
    struct sockaddr_in* in = (struct sockaddr_in*)ai->ai_addr;
    if (ntohl(in->sin_addr.s_addr) == INADDR_ANY) {
        int i;
        uint32_t* hostips;
        for (hostips = GetHostIps(), i = 0; hostips[i]; ++i)
            print_listening_url(hostips[i], ntohs(in->sin_port));
        free(hostips);
    } else {
        print_listening_url(ntohl(in->sin_addr.s_addr), ntohs(in->sin_port));
    }

    freeaddrinfo(ai);
    return fd;
}
