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

#include "llamafile.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// cpu governor for llamafile-bench
//
// benchmarks can be negatively impacted when the operating system or
// motherboard throttles the cpu due to high temperature. we can say:
//
//     echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
//     export LLAMAFILE_TEMPERATURE_FILE=/sys/class/hwmon/hwmon3/temp1_input
//     export LLAMAFILE_TEMPERATURE_MAX=60000  # 10° beneath mobo's governor
//     o//llama.cpp/llama-bench/llama-bench -m model1.gguf -m model2.gguf
//
// to ask llama-bench to wait until the temperature drops to 60°C before
// running a benchmark. this number should be lower than the motherboard
// threshold, which jart configures to 80°C. the actual path of your cpu
// temperature sensor may vary.

static int llamafile_read_temperature(void) {
    int fd;
    const char *path;
    char buf[13] = {0};
    if (!(path = getenv("LLAMAFILE_TEMPERATURE_FILE")))
        return -1;
    if ((fd = open(path, O_RDONLY)) == -1) {
        perror(path);
        exit(1);
    }
    pread(fd, buf, 12, 0);
    close(fd);
    return atoi(buf);
}

static int llamafile_govern_threshold(void) {
    int temp;
    const char *s;
    if (!(s = getenv("LLAMAFILE_TEMPERATURE_MAX")))
        return -1;
    temp = atoi(s);
    if (temp <= 0) {
        errno = EINVAL;
        perror("LLAMAFILE_TEMPERATURE_MAX");
        exit(1);
    }
    return temp;
}

void llamafile_govern(void) {
    int max, cur;
    int delay = 1;
    for (;;) {
        if ((max = llamafile_govern_threshold()) == -1)
            return;
        if ((cur = llamafile_read_temperature()) == -1)
            return;
        if (cur <= max)
            return;
        usleep(delay);
        if (delay < 16384)
            delay <<= 1;
    }
}
