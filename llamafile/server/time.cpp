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

#include "time.h"
#include "llamafile/crash.h"
#include "llamafile/server/log.h"
#include <atomic>
#include <csignal>
#include <pthread.h>
#include <unistd.h>

//
// lockless implementation of gmtime_r() and localtime_r()
//

namespace lf {
namespace server {

struct Clock
{
    std::atomic_uint roll;
    std::atomic_ulong time;
    std::atomic_ulong date;
};

static Clock g_clck[2];
static pthread_t g_time_thread;

static void
set_clck(Clock* clck, long time, long date)
{
    unsigned long roll;
    roll = clck->roll.fetch_add(1, std::memory_order_relaxed);
    time &= 0xffffffffffff;
    date &= 0xffffffffffff;
    time |= roll << 48;
    date |= roll << 48;
    clck->time.store(time, std::memory_order_relaxed);
    clck->date.store(date, std::memory_order_relaxed);
}

static void
get_clck(Clock* clck, long* out_time, long* out_date)
{
    long time, date;
    do {
        time = clck->time.load(std::memory_order_relaxed);
        date = clck->date.load(std::memory_order_relaxed);
    } while ((time >> 48) != (date >> 48));
    *out_date = date & 0xffffffffffff;
    *out_time = time & 0xffffffffffff;
}

static long
encode_date(const tm* tm)
{
    long date;
    date = tm->tm_year;
    date <<= 4;
    date |= tm->tm_isdst == 1;
    date <<= 1;
    date |= tm->tm_mon;
    date <<= 5;
    date |= tm->tm_mday;
    date <<= 3;
    date |= tm->tm_wday;
    date <<= 5;
    date |= tm->tm_hour;
    date <<= 6;
    date |= tm->tm_min;
    date <<= 6;
    date |= tm->tm_sec;
    return date;
}

static void
decode_date(long date, tm* tm)
{
    tm->tm_sec = date & 63;
    date >>= 6;
    tm->tm_min = date & 63;
    date >>= 6;
    tm->tm_hour = date & 31;
    date >>= 5;
    tm->tm_wday = date & 7;
    date >>= 3;
    tm->tm_mday = date & 31;
    date >>= 5;
    tm->tm_mon = date & 15;
    date >>= 4;
    tm->tm_isdst = date & 1;
    date >>= 1;
    tm->tm_year = date;
    tm->tm_gmtoff = 0; // unsupported
    tm->tm_zone = 0; // unsupported
    tm->tm_yday = 0; // unsupported
}

static void
update_time()
{
    tm tm;
    timespec ts;
    clock_gettime(0, &ts);
    gmtime_r(&ts.tv_sec, &tm);
    set_clck(&g_clck[0], ts.tv_sec, encode_date(&tm));
    localtime_r(&ts.tv_sec, &tm);
    set_clck(&g_clck[1], ts.tv_sec, encode_date(&tm));
}

static void*
time_worker(void* arg)
{
    sigset_t ss;
    sigemptyset(&ss);
    sigaddset(&ss, SIGHUP);
    sigaddset(&ss, SIGINT);
    sigaddset(&ss, SIGQUIT);
    sigaddset(&ss, SIGTERM);
    sigaddset(&ss, SIGUSR1);
    sigaddset(&ss, SIGALRM);
    pthread_sigmask(SIG_SETMASK, &ss, 0);
    set_thread_name("localtime");
    for (;;) {
        sleep(10);
        update_time();
    }
    return nullptr;
}

void
time_init()
{
    update_time();
    if (pthread_create(&g_time_thread, 0, time_worker, 0))
        __builtin_trap();
}

void
time_destroy()
{
    pthread_cancel(g_time_thread);
    if (pthread_join(g_time_thread, 0))
        __builtin_trap();
}

static const char kMonDays[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
};

static void
time_lockless(Clock* clck, long now, tm* tm)
{
    long time, date, since;
    get_clck(clck, &time, &date);
    decode_date(date, tm);
    since = now - time;
    since = since < 60 ? since : 60;
    for (; since > 0; --since) {
        if (++tm->tm_sec >= 60) {
            tm->tm_sec = 0;
            if (++tm->tm_min >= 60) {
                tm->tm_min = 0;
                if (++tm->tm_hour >= 24) {
                    tm->tm_hour = 0;
                    if (++tm->tm_mday >= 7)
                        tm->tm_mday = 0;
                    if (++tm->tm_mday > kMonDays[!!tm->tm_isdst][tm->tm_mon]) {
                        tm->tm_mday = 1;
                        if (++tm->tm_mon >= 12) {
                            tm->tm_mon = 0;
                            ++tm->tm_year;
                        }
                    }
                }
            }
        }
    }
}

void
gmtime_lockless(long now, tm* tm)
{
    time_lockless(&g_clck[0], now, tm);
}

void
localtime_lockless(long now, tm* tm)
{
    time_lockless(&g_clck[1], now, tm);
}

} // namespace server
} // namespace lf
