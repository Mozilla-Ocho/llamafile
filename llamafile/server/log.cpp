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

#include "log.h"
#include "time.h"
#include <cstring>
#include <pthread.h>

namespace lf {
namespace server {

bool g_log_disable;

static thread_local char g_thread_name[128];

const char*
get_thread_name(void)
{
    return g_thread_name;
}

void
set_thread_name(const char* name)
{
    char shortened[16];
    strlcpy(shortened, name, sizeof(shortened));
    pthread_setname_np(pthread_self(), shortened);
    strlcpy(g_thread_name, name, sizeof(g_thread_name));
}

char*
get_log_timestamp(void)
{
    tm tm;
    int x;
    timespec ts;
    thread_local static long last;
    thread_local static char s[27];
    clock_gettime(0, &ts);
    if (ts.tv_sec != last) {
        localtime_lockless(ts.tv_sec, &tm);
        x = tm.tm_year + 1900;
        s[0] = '0' + x / 1000;
        s[1] = '0' + x / 100 % 10;
        s[2] = '0' + x / 10 % 10;
        s[3] = '0' + x % 10;
        s[4] = '-';
        x = tm.tm_mon + 1;
        s[5] = '0' + x / 10;
        s[6] = '0' + x % 10;
        s[7] = '-';
        x = tm.tm_mday;
        s[8] = '0' + x / 10;
        s[9] = '0' + x % 10;
        s[10] = 'T';
        x = tm.tm_hour;
        s[11] = '0' + x / 10;
        s[12] = '0' + x % 10;
        s[13] = ':';
        x = tm.tm_min;
        s[14] = '0' + x / 10;
        s[15] = '0' + x % 10;
        s[16] = ':';
        x = tm.tm_sec;
        s[17] = '0' + x / 10;
        s[18] = '0' + x % 10;
        s[19] = '.';
        s[26] = 0;
        last = ts.tv_sec;
    }
    x = ts.tv_nsec;
    s[20] = '0' + x / 100000000;
    s[21] = '0' + x / 10000000 % 10;
    s[22] = '0' + x / 1000000 % 10;
    s[23] = '0' + x / 100000 % 10;
    s[24] = '0' + x / 10000 % 10;
    s[25] = '0' + x / 1000 % 10;
    return s;
}

} // namespace server
} // namespace lf
