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
#include <stdio.h>
#include <string>

__static_yoink("llamafile/schema.sql");

#define SCHEMA_VERSION 1

namespace llamafile {
namespace db {

static bool table_exists(sqlite3* db, const char* table_name) {
    const char* query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;";
    sqlite3_stmt* stmt;
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

static bool init_schema(sqlite3* db) {
    FILE* f = fopen("/zip/llamafile/schema.sql", "r");
    if (!f)
        return false;
    std::string schema;
    int c;
    while ((c = fgetc(f)) != EOF)
        schema += c;
    fclose(f);
    char* errmsg = nullptr;
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

sqlite3* open(const char* path) {
    sqlite3* db;
    int rc = sqlite3_open(path, &db);
    if (rc) {
        fprintf(stderr, "%s: can't open database: %s\n", path, sqlite3_errmsg(db));
        return nullptr;
    }
    char* errmsg = nullptr;
    if (sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
        fprintf(stderr, "Failed to set journal mode to WAL: %s\n", errmsg);
        sqlite3_free(errmsg);
        sqlite3_close(db);
        return nullptr;
    }
    if (sqlite3_exec(db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
        fprintf(stderr, "Failed to set synchronous to NORMAL: %s\n", errmsg);
        sqlite3_free(errmsg);
        sqlite3_close(db);
        return nullptr;
    }
    if (!table_exists(db, "metadata") && !init_schema(db)) {
        fprintf(stderr, "%s: failed to initialize database schema\n", path);
        sqlite3_close(db);
        return nullptr;
    }
    return db;
}

void close(sqlite3* db) {
    sqlite3_close(db);
}

} // namespace db
} // namespace llamafile
