// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include "llama.cpp/llama.h"
#include "llamafile/version.h"
#include "llama.cpp/embedr/sqlite3.h"
#include "llama.cpp/embedr/sqlite-vec.h"
#include "llama.cpp/embedr/shell.h"
#include "string.h"
int main(int argc, char ** argv) {
    int rc;
    sqlite3* db;
    sqlite3_stmt* stmt;
    rc = sqlite3_auto_extension((void (*)())sqlite3_vec_init);

    if(argc > 1 &&  (strcmp(argv[1], "sh") == 0)) {
      return mn(argc, argv);
    }
    printf("%d\n", argc);
    printf("llamafile-embed %s, SQLite %s, sqlite-vec=%s, %d\n", LLAMAFILE_VERSION_STRING, sqlite3_version, SQLITE_VEC_VERSION, LLAMA_FTYPE_MOSTLY_Q4_1);

    rc = sqlite3_open(":memory:", &db);
    if(rc != SQLITE_OK) {
      printf("x\n");
      return 1;
    }

    rc = sqlite3_prepare_v2(db, "select vec_version()", -1, &stmt, NULL);
    if(rc != SQLITE_OK) {
      printf("a\n");
      return 1;
    }
    rc = sqlite3_step(stmt);
    if(rc != SQLITE_ROW) {
      printf("b\n");
      sqlite3_finalize(stmt);
      return 1;
    }
    printf("x=%s\n", sqlite3_column_text(stmt, 0));

    sqlite3_finalize(stmt);


    return 0;
}
