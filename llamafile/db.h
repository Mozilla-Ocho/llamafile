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
#include "json.h"
#include <__fwd/string.h>

struct sqlite3;

namespace lf {
namespace db {

sqlite3 *open();
void close(sqlite3 *);
int64_t add_chat(sqlite3 *, const std::string &, const std::string &);
int64_t add_message(sqlite3 *, int64_t, const std::string &, const std::string &, double, double,
                    double, double);
bool update_title(sqlite3 *, int64_t, const std::string &);
bool delete_message(sqlite3 *, int64_t);
jt::Json get_chat(sqlite3 *, int64_t);
jt::Json get_chats(sqlite3 *);
jt::Json get_message(sqlite3 *, int64_t);
jt::Json get_messages(sqlite3 *, int64_t);

} // namespace db
} // namespace lf
