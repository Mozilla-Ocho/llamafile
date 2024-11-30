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

#include "db.h"
#include "llamafile/json.h"
#include "llamafile/llamafile.h"
#include "third_party/sqlite/sqlite3.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

__static_yoink("llamafile/schema.sql");

#define SCHEMA_VERSION 1

namespace lf {
namespace db {

static bool table_exists(sqlite3 *db, const char *table_name) {
    const char *query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    if (sqlite3_bind_text(stmt, 1, table_name, -1, SQLITE_STATIC) != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return false;
    }
    bool exists = sqlite3_step(stmt) == SQLITE_ROW;
    sqlite3_finalize(stmt);
    return exists;
}

static bool init_schema(sqlite3 *db) {
    FILE *f = fopen("/zip/llamafile/schema.sql", "r");
    if (!f)
        return false;
    std::string schema;
    int c;
    while ((c = fgetc(f)) != EOF)
        schema += c;
    fclose(f);
    char *errmsg = nullptr;
    int rc = sqlite3_exec(db, schema.c_str(), nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        if (errmsg) {
            fprintf(stderr, "SQL error: %s\n", errmsg);
            sqlite3_free(errmsg);
        }
        return false;
    }
    return true;
}

static sqlite3 *open_impl() {
    std::string path;
    if (FLAG_db) {
        path = FLAG_db;
    } else {
        const char *home = getenv("HOME");
        if (home) {
            path = std::string(home) + "/.llamafile/llamafile.sqlite3";
        } else {
            path = "llamafile.sqlite3";
        }
    }
    sqlite3 *db;
    int rc = sqlite3_open(path.c_str(), &db);
    if (rc) {
        fprintf(stderr, "%s: can't open database: %s\n", path.c_str(), sqlite3_errmsg(db));
        return nullptr;
    }
    char *errmsg = nullptr;
    if (sqlite3_exec(db, FLAG_db_startup_sql, nullptr, nullptr, &errmsg) != SQLITE_OK) {
        fprintf(stderr, "%s: failed to execute startup SQL (%s) because: %s", path.c_str(),
                FLAG_db_startup_sql, errmsg);
        sqlite3_free(errmsg);
        sqlite3_close(db);
        return nullptr;
    }
    if (!table_exists(db, "metadata") && !init_schema(db)) {
        fprintf(stderr, "%s: failed to initialize database schema\n", path.c_str());
        sqlite3_close(db);
        return nullptr;
    }
    return db;
}

sqlite3 *open() {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    sqlite3 *res = open_impl();
    pthread_setcancelstate(cs, 0);
    return res;
}

void close(sqlite3 *db) {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    sqlite3_close(db);
    pthread_setcancelstate(cs, 0);
}

static int64_t add_chat_impl(sqlite3 *db, const std::string &model, const std::string &title) {
    const char *query = "INSERT INTO chats (model, title) VALUES (?, ?);";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        return -1;
    }
    if (sqlite3_bind_text(stmt, 1, model.data(), model.size(), SQLITE_STATIC) != SQLITE_OK ||
        sqlite3_bind_text(stmt, 2, title.data(), title.size(), SQLITE_STATIC) != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return -1;
    }
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);
    return sqlite3_last_insert_rowid(db);
}

int64_t add_chat(sqlite3 *db, const std::string &model, const std::string &title) {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    int64_t res = add_chat_impl(db, model, title);
    pthread_setcancelstate(cs, 0);
    return res;
}

static int64_t add_message_impl(sqlite3 *db, int64_t chat_id, const std::string &role,
                                const std::string &content, double temperature, double top_p,
                                double presence_penalty, double frequency_penalty) {
    const char *query = "INSERT INTO messages (chat_id, role, content, temperature, "
                        "top_p, presence_penalty, frequency_penalty) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?);";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        return -1;
    }
    if (sqlite3_bind_int64(stmt, 1, chat_id) != SQLITE_OK ||
        sqlite3_bind_text(stmt, 2, role.data(), role.size(), SQLITE_STATIC) != SQLITE_OK ||
        sqlite3_bind_text(stmt, 3, content.data(), content.size(), SQLITE_STATIC) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 4, temperature) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 5, top_p) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 6, presence_penalty) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 7, frequency_penalty) != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return -1;
    }
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);
    return sqlite3_last_insert_rowid(db);
}

int64_t add_message(sqlite3 *db, int64_t chat_id, const std::string &role,
                    const std::string &content, double temperature, double top_p,
                    double presence_penalty, double frequency_penalty) {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    int64_t res = add_message_impl(db, chat_id, role, content, temperature, top_p, presence_penalty,
                                   frequency_penalty);
    pthread_setcancelstate(cs, 0);
    return res;
}

static bool update_title_impl(sqlite3 *db, int64_t chat_id, const std::string &title) {
    const char *query = "UPDATE chats SET title = ? WHERE id = ?;";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    if (sqlite3_bind_text(stmt, 1, title.data(), title.size(), SQLITE_STATIC) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 2, chat_id) != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return false;
    }
    bool success = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return success;
}

bool update_title(sqlite3 *db, int64_t chat_id, const std::string &title) {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    bool res = update_title_impl(db, chat_id, title);
    pthread_setcancelstate(cs, 0);
    return res;
}

static bool delete_message_impl(sqlite3 *db, int64_t message_id) {
    const char *query = "DELETE FROM messages WHERE id = ?;";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    if (sqlite3_bind_int64(stmt, 1, message_id) != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return false;
    }
    bool success = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return success;
}

bool delete_message(sqlite3 *db, int64_t message_id) {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    bool res = delete_message_impl(db, message_id);
    pthread_setcancelstate(cs, 0);
    return res;
}

static jt::Json get_chats_impl(sqlite3 *db) {
    const char *query = "SELECT id, created_at, model, title FROM chats ORDER BY created_at DESC;";
    sqlite3_stmt *stmt;
    jt::Json result;
    result.setArray();
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        return result;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        jt::Json chat;
        chat.setObject();
        chat["id"] = sqlite3_column_int64(stmt, 0);
        chat["created_at"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
        chat["model"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
        chat["title"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 3));
        result.getArray().push_back(std::move(chat));
    }
    sqlite3_finalize(stmt);
    return result;
}

jt::Json get_chats(sqlite3 *db) {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    jt::Json res = get_chats_impl(db);
    pthread_setcancelstate(cs, 0);
    return res;
}

static jt::Json get_messages_impl(sqlite3 *db, int64_t chat_id) {
    const char *query = "SELECT id, created_at, role, content, temperature, top_p, "
                        "presence_penalty, frequency_penalty "
                        "FROM messages "
                        "WHERE chat_id = ? "
                        "ORDER BY created_at DESC;";
    sqlite3_stmt *stmt;
    jt::Json result;
    result.setArray();
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        return result;
    }
    if (sqlite3_bind_int64(stmt, 1, chat_id) != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return result;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        jt::Json msg;
        msg.setObject();
        msg["id"] = sqlite3_column_int64(stmt, 0);
        msg["created_at"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
        msg["role"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
        msg["content"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 3));
        msg["temperature"] = sqlite3_column_double(stmt, 4);
        msg["top_p"] = sqlite3_column_double(stmt, 5);
        msg["presence_penalty"] = sqlite3_column_double(stmt, 6);
        msg["frequency_penalty"] = sqlite3_column_double(stmt, 7);
        result.getArray().push_back(std::move(msg));
    }
    sqlite3_finalize(stmt);
    return result;
}

jt::Json get_messages(sqlite3 *db, int64_t chat_id) {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    jt::Json res = get_messages_impl(db, chat_id);
    pthread_setcancelstate(cs, 0);
    return res;
}

static jt::Json get_chat_impl(sqlite3 *db, int64_t chat_id) {
    const char *query = "SELECT id, created_at, model, title FROM chats WHERE id = ?;";
    sqlite3_stmt *stmt;
    jt::Json result;
    result.setObject();
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        return result;
    }
    if (sqlite3_bind_int64(stmt, 1, chat_id) != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return result;
    }
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result["id"] = sqlite3_column_int64(stmt, 0);
        result["created_at"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
        result["model"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
        result["title"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 3));
    }
    sqlite3_finalize(stmt);
    return result;
}

jt::Json get_chat(sqlite3 *db, int64_t chat_id) {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    jt::Json res = get_chat_impl(db, chat_id);
    pthread_setcancelstate(cs, 0);
    return res;
}

static jt::Json get_message_impl(sqlite3 *db, int64_t message_id) {
    const char *query = "SELECT id, created_at, chat_id, role, content, temperature, top_p, "
                        "presence_penalty, frequency_penalty "
                        "FROM messages WHERE id = ?"
                        "ORDER BY created_at ASC;";
    sqlite3_stmt *stmt;
    jt::Json result;
    result.setObject();
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        return result;
    }
    if (sqlite3_bind_int64(stmt, 1, message_id) != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return result;
    }
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result["id"] = sqlite3_column_int64(stmt, 0);
        result["created_at"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
        result["chat_id"] = sqlite3_column_int64(stmt, 2);
        result["role"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 3));
        result["content"] = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 4));
        result["temperature"] = sqlite3_column_double(stmt, 5);
        result["top_p"] = sqlite3_column_double(stmt, 6);
        result["presence_penalty"] = sqlite3_column_double(stmt, 7);
        result["frequency_penalty"] = sqlite3_column_double(stmt, 8);
    }
    sqlite3_finalize(stmt);
    return result;
}

jt::Json get_message(sqlite3 *db, int64_t message_id) {
    int cs;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
    jt::Json res = get_message_impl(db, message_id);
    pthread_setcancelstate(cs, 0);
    return res;
}

} // namespace db
} // namespace lf
