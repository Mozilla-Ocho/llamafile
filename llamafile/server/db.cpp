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

#include "llamafile/db.h"
#include "client.h"
#include "llama.cpp/llama.h"
#include "llamafile/llamafile.h"
#include "llamafile/string.h"
#include <string>

namespace lf {
namespace server {

bool
Client::db_chats()
{
    if (msg_.method == kHttpGet) {
        sqlite3* db = db::open();
        if (!db)
            return send_error(500, "db::open failed");
        jt::Json json = db::get_chats(db);
        db::close(db);
        dump_ = json.toStringPretty();
        dump_ += '\n';
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, dump_);
    } else if (msg_.method == kHttpPut) {
        if (!HasHeader(kHttpContentType) ||
            !IsMimeType(HeaderData(kHttpContentType),
                        HeaderLength(kHttpContentType),
                        "application/json")) {
            return send_error(501, "Content Type Not Implemented");
        }
        if (!read_payload())
            return false;
        auto [status, json] = jt::Json::parse(std::string(payload_));
        if (status != jt::Json::success)
            return send_error(400, jt::Json::StatusToString(status));
        if (!json.isObject())
            return send_error(400, "JSON body must be an object");
        if (!json["title"].isString())
            return send_error(400, "title must be a string");
        sqlite3* db = db::open();
        if (!db)
            return send_error(500, "db::open failed");
        int64_t chat_id =
          db::add_chat(db, FLAG_model, json["title"].getString());
        if (chat_id == -1) {
            db::close(db);
            return send_error(500, "db::add_chat failed");
        }
        jt::Json json2 = db::get_chat(db, chat_id);
        db::close(db);
        dump_ = json2.toStringPretty();
        dump_ += '\n';
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, dump_);
    } else {
        return send_error(405);
    }
}

bool
Client::db_chat(int64_t id)
{
    if (msg_.method == kHttpGet) {
        sqlite3* db = db::open();
        if (!db)
            return send_error(500, "db::open failed");
        jt::Json json = db::get_chat(db, id);
        db::close(db);
        dump_ = json.toStringPretty();
        dump_ += '\n';
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, dump_);
    } else if (msg_.method == kHttpPut) {
        if (!HasHeader(kHttpContentType) ||
            !IsMimeType(HeaderData(kHttpContentType),
                        HeaderLength(kHttpContentType),
                        "application/json")) {
            return send_error(501, "Content Type Not Implemented");
        }
        if (!read_payload())
            return false;
        auto [status, json] = jt::Json::parse(std::string(payload_));
        if (status != jt::Json::success)
            return send_error(400, jt::Json::StatusToString(status));
        if (!json.isObject())
            return send_error(400, "JSON body must be an object");
        if (!json["title"].isString())
            return send_error(400, "title must be a string");
        sqlite3* db = db::open();
        if (!db)
            return send_error(500, "db::open failed");
        if (!db::update_title(db, id, json["title"].getString())) {
            db::close(db);
            return send_error(500, "db::update_title failed");
        }
        jt::Json json2 = db::get_chat(db, id);
        db::close(db);
        dump_ = json2.toStringPretty();
        dump_ += '\n';
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, dump_);
    } else {
        return send_error(405);
    }
}

bool
Client::db_messages(int64_t chat_id)
{
    if (msg_.method == kHttpGet) {
        sqlite3* db = db::open();
        if (!db)
            return send_error(500, "db::open failed");
        jt::Json json = db::get_messages(db, chat_id);
        db::close(db);
        dump_ = json.toStringPretty();
        dump_ += '\n';
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, dump_);
    } else if (msg_.method == kHttpPut) {
        if (!HasHeader(kHttpContentType) ||
            !IsMimeType(HeaderData(kHttpContentType),
                        HeaderLength(kHttpContentType),
                        "application/json")) {
            return send_error(501, "Content Type Not Implemented");
        }
        if (!read_payload())
            return false;
        auto [status, json] = jt::Json::parse(std::string(payload_));
        if (status != jt::Json::success)
            return send_error(400, jt::Json::StatusToString(status));
        if (!json.isObject())
            return send_error(400, "JSON body must be an object");
        if (!json["role"].isString())
            return send_error(400, "role must be a string");
        if (!json["content"].isString())
            return send_error(400, "content must be a string");
        if (!json["temperature"].isNumber())
            return send_error(400, "temperature must be a number");
        if (!json["top_p"].isNumber())
            return send_error(400, "top_p must be a number");
        if (!json["presence_penalty"].isNumber())
            return send_error(400, "presence_penalty must be a number");
        if (!json["frequency_penalty"].isNumber())
            return send_error(400, "frequency_penalty must be a number");
        sqlite3* db = db::open();
        if (!db)
            return send_error(500, "db::open failed");
        int64_t chat_id =
          db::add_message(db,
                          chat_id,
                          json["role"].getString(),
                          json["content"].getString(),
                          json["temperature"].getNumber(),
                          json["top_p"].getNumber(),
                          json["presence_penalty"].getNumber(),
                          json["frequency_penalty"].getNumber());
        if (chat_id == -1) {
            db::close(db);
            return send_error(500, "db::add_chat failed");
        }
        jt::Json json2 = db::get_chat(db, chat_id);
        db::close(db);
        dump_ = json2.toStringPretty();
        dump_ += '\n';
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, dump_);
    } else {
        return send_error(405);
    }
}

bool
Client::db_message(int64_t id)
{
    if (msg_.method == kHttpGet) {
        sqlite3* db = db::open();
        if (!db)
            return send_error(500, "db::open failed");
        jt::Json json = db::get_message(db, id);
        db::close(db);
        dump_ = json.toStringPretty();
        dump_ += '\n';
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        return send_response(obuf_.p, p, dump_);
    } else if (msg_.method == kHttpDelete) {
        sqlite3* db = db::open();
        if (!db)
            return send_error(500, "db::open failed");
        if (!db::delete_message(db, id)) {
            db::close(db);
            return send_error(500, "db::delete_message failed");
        }
        db::close(db);
        char* p = append_http_response_message(obuf_.p, 200);
        return send_response(obuf_.p, p, "");
    } else {
        return send_error(405);
    }
}

} // namespace server
} // namespace lf
