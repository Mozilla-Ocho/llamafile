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

#include "client.h"
#include "llama.cpp/llama.h"
#include "llamafile/json.h"
#include "llamafile/llamafile.h"
#include "llamafile/string.h"
#include <ctime>

using jt::Json;

namespace lf {
namespace server {

// Use it as reported model creation time
static const time_t model_creation_time = time(0);

bool
Client::v1_models()
{
    jt::Json json;
    json["object"] = "list";
    Json& model = json["data"][0];
    model["id"] = stripext(basename(FLAG_model));
    model["object"] = "model";
    model["created"] = model_creation_time;
    model["owned_by"] = "llamafile";
    char* p = append_http_response_message(obuf_.p, 200);
    p = stpcpy(p, "Content-Type: application/json\r\n");
    return send_response(obuf_.p, p, json.toString());
}
    
} // namespace server
} // namespace lf