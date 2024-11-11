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

#include "trust.h"

#include <string>

int main(int argc, char *argv[]) {
    cidr in;

    if (!parse_cidr("10.10.10.1/24", &in))
        return 1;
    if (in.ip != 0x0a0a0a01)
        return 2;
    if (in.bits != 24)
        return 3;
    if (in.matches(parse_ip("10.10.11.1")))
        return 3;
    if (!in.matches(parse_ip("10.10.10.1")))
        return 3;
    if (!in.matches(parse_ip("  10.10.10.255  ")))
        return 3;

    if (!parse_cidr("168430081/1", &in))
        return 4;
    if (in.ip != 0x0a0a0a01)
        return 5;
    if (in.bits != 1)
        return 6;

    if (parse_cidr("", 0))
        return 77;
    if (parse_cidr(" ", 0))
        return 78;
    if (parse_cidr("10.10.10.1/", 0))
        return 7;
    if (parse_cidr("10.10.10.1/a", 0))
        return 8;
    if (parse_cidr("10.10.10.1/33", 0))
        return 9;
    if (parse_cidr("10.10.10.a", 0))
        return 10;

    if (!parse_cidr("10.10.10.255", &in))
        return 11;
    if (in.ip != 0x0a0a0aff)
        return 12;
    if (in.bits != 32)
        return 13;
    if (in.matches(parse_ip("10.10.10.254")))
        return 14;
    if (!in.matches(parse_ip("10.10.10.255")))
        return 15;
}
