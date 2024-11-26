// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include "llama.cpp/llama.h"
#include "llamafile/version.h"
#include "llama.cpp/embedfile/embedfile.h"
#include "llama.cpp/embedfile/sqlite3.h"
#include "llama.cpp/embedfile/sqlite-vec.h"
#include "llama.cpp/embedfile/sqlite-lembed.h"
#include "llama.cpp/embedfile/sqlite-csv.h"
#include "llama.cpp/embedfile/sqlite-lines.h"
#include "llama.cpp/embedfile/shell.h"
#include <string.h>

#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cosmo.h>


int64_t time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

char * EMBEDFILE_MODEL = NULL;

void embedfile_version(sqlite3_context * context, int argc, sqlite3_value **value) {
  sqlite3_result_text(context, EMBEDFILE_VERSION, -1, SQLITE_STATIC);
}

int embedfile_sqlite3_init(sqlite3 * db) {
  int rc;

  rc = sqlite3_vec_init(db, NULL, NULL); assert(rc == SQLITE_OK);
  rc = sqlite3_lembed_init(db, NULL, NULL); assert(rc == SQLITE_OK);
  rc = sqlite3_csv_init(db, NULL, NULL); assert(rc == SQLITE_OK);
  rc = sqlite3_lines_init(db, NULL, NULL); assert(rc == SQLITE_OK);
  rc = sqlite3_create_function_v2(db, "embedfile_version",0, SQLITE_DETERMINISTIC | SQLITE_UTF8, NULL, embedfile_version, NULL, NULL, NULL); assert(rc == SQLITE_OK);

  if(!EMBEDFILE_MODEL) {
    return SQLITE_OK;
  }
  sqlite3_stmt * stmt;
  rc = sqlite3_prepare_v2(db, "insert into temp.lembed_models(model) values (?)", -1, &stmt, NULL);
  if(rc != SQLITE_OK) {
    assert(rc == SQLITE_OK);
    return rc;
  }
  sqlite3_bind_text(stmt, 1, EMBEDFILE_MODEL, -1, SQLITE_STATIC);
  sqlite3_step(stmt);
  rc = sqlite3_finalize(stmt);
  assert(rc == SQLITE_OK);

  return rc;
}

int table_exists(sqlite3 * db, const char * table) {
  int rc;
  sqlite3_stmt * stmt;
  rc = sqlite3_prepare_v2(db, "select ? in (select name from pragma_table_list)", -1, &stmt, NULL);
  assert(rc == SQLITE_OK);
  sqlite3_bind_text(stmt, 1, table, strlen(table), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  assert(rc == SQLITE_ROW);
  int result = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return result;
}

#define BAR_WIDTH 20
void print_progress_bar(long long nEmbed, long long nTotal, long long elapsed_ms) {
    float progress = (float)nEmbed / nTotal;
    int bar_fill = (int)(progress * BAR_WIDTH);

    long long remaining = nTotal - nEmbed;
    float rate = (float)nEmbed / (elapsed_ms / 1000.0);
    long long remaining_time = (rate > 0) ? remaining / rate : 0;

    printf("\r%3d%%|", (int)(progress * 100));
    for (int i = 0; i < BAR_WIDTH; i++) {
        if (i < bar_fill)
            printf("█");
        else
            printf(" ");
    }
    printf("| %lld/%lld [%02lld:%02lld<%02lld:%02lld, %.0f/s]",
           nEmbed, nTotal,
           elapsed_ms / 1000 / 60, elapsed_ms / 1000 % 60,
           remaining_time / 60, remaining_time % 60,
           rate);

    fflush(stdout);
}

int default_model_dimensions(sqlite3 * db, int64_t * dimensions) {
  int rc;
  sqlite3_stmt * stmt;
  rc = sqlite3_prepare_v2(db, "select dimensions from lembed_models where name = ?", -1, &stmt, NULL);
  assert(rc == SQLITE_OK);

  sqlite3_bind_text(stmt, 1, "default", -1, SQLITE_STATIC);

  rc = sqlite3_step(stmt);
  assert(rc == SQLITE_ROW);
  *dimensions = sqlite3_column_int64(stmt, 0);
  sqlite3_finalize(stmt);

  return SQLITE_OK;
}

int cmd_index(char * filename, char * target_column) {
  int rc;
  sqlite3* db = NULL;
  sqlite3_stmt* stmt = NULL;
  char * zDbPath = sqlite3_mprintf("%s.db", filename);
  assert(zDbPath);

  rc = sqlite3_open(zDbPath, &db);
  assert(rc == SQLITE_OK);

  rc = sqlite3_exec(db, "PRAGMA page_size=16384;", NULL, NULL, NULL);
  assert(rc == SQLITE_OK);

  rc = embedfile_sqlite3_init(db);
  assert(rc == SQLITE_OK);

  if(sqlite3_strlike("%.csv", filename, 0) == 0) {
    const char * zSql;

    rc = sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);
    assert(rc == SQLITE_OK);

    zSql = sqlite3_mprintf(
      "CREATE VIRTUAL TABLE temp.source USING csv(filename=\"%w\", header=yes)",
      filename
    );
    assert(zSql);
    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
    assert(rc == SQLITE_OK);
    rc = sqlite3_step(stmt);
    assert(rc == SQLITE_DONE);
    sqlite3_finalize(stmt);

    int64_t dimensions;
    rc = default_model_dimensions(db, &dimensions);

    rc = sqlite3_exec(db, "CREATE TABLE source AS SELECT * FROM temp.source;", NULL, NULL, NULL);
    assert(rc == SQLITE_OK);

    zSql = sqlite3_mprintf(
      "CREATE VIRTUAL TABLE vec_source USING vec0(embedding float[%lld])",
      dimensions
    );
    assert(zSql);
    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
    assert(rc == SQLITE_OK);
    rc = sqlite3_step(stmt);
    assert(rc == SQLITE_DONE);
    sqlite3_finalize(stmt);

      int64_t nTotal;
    {
      sqlite3_stmt * stmt;
      rc = sqlite3_prepare_v2(db, "SELECT count(*) FROM source", -1, &stmt, NULL);
      assert(rc == SQLITE_OK);
      rc = sqlite3_step(stmt);
      assert(rc == SQLITE_ROW);
      nTotal = sqlite3_column_int64(stmt, 0);
      sqlite3_finalize(stmt);
    }

    int64_t nRemaining = nTotal;


    zSql = sqlite3_mprintf(
      " \
        WITH chunk AS ( \
          SELECT \
            source.rowid, \
            lembed(source.\"%w\") AS embedding \
          FROM source \
          WHERE source.rowid NOT IN (select rowid from vec_source) \
          LIMIT 256 \
        ) \
        INSERT INTO vec_source(rowid, embedding) \
        SELECT rowid, embedding FROM chunk \
        RETURNING rowid; \
      ",
      target_column
    );
    assert(zSql);

    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
    assert(rc == SQLITE_OK);

    int64_t nEmbed = 0;
    int64_t t0 = time_ms();

    while(1){
      sqlite3_reset(stmt);

      int nChunkEmbed = 0;
      while(1) {
        rc = sqlite3_step(stmt);
        if(rc == SQLITE_DONE) {
          break;
        }
        assert(rc == SQLITE_ROW);
        nChunkEmbed++;
      }
      if(nChunkEmbed == 0) {
        break;
      }
      nEmbed += nChunkEmbed;
      nRemaining -= nChunkEmbed;
      print_progress_bar(nEmbed, nTotal, time_ms() - t0);
    }
  }
  else {
    printf("Unknown filetype\n");
  }

  rc = sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
  assert(rc == SQLITE_OK);

  sqlite3_free(zDbPath);
  sqlite3_close(db);
  return SQLITE_OK;
}

int cmd_backfill(char * dbPath, char * table, char * column) {
  int rc;
  sqlite3* db;
  rc = sqlite3_open(dbPath, &db);
  if(rc != SQLITE_OK) {
    fprintf(stderr, "could not open database");
    return rc;
  }

  rc = embedfile_sqlite3_init(db);
  assert(rc == SQLITE_OK);

  rc = sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);
  assert(rc == SQLITE_OK);

  const char *tableEmbeddings = sqlite3_mprintf("%s_embeddings", table);
  assert(tableEmbeddings);

  if(!(table_exists(db, tableEmbeddings))) {
    const char * zSql = sqlite3_mprintf(
      "CREATE TABLE \"%w\"(rowid INTEGER PRIMARY KEY, embedding BLOB);"
      "INSERT INTO \"%w\"(rowid) SELECT rowid FROM \"%w\";"
      "CREATE INDEX \"idx_%w\" ON \"%w\"(embedding) WHERE embedding IS NULL;",
      tableEmbeddings,
      tableEmbeddings,
      table,
      tableEmbeddings,
      tableEmbeddings
    );
    rc = sqlite3_exec(db, zSql, NULL, NULL, NULL);
    sqlite3_free((void *) zSql);
    assert(rc == SQLITE_OK);
  }


  int64_t nTotal;
  {
    sqlite3_stmt * stmt;
    const char * zSql = sqlite3_mprintf("SELECT count(*) FROM \"%w\" WHERE embedding IS NULL", tableEmbeddings);
    assert(zSql);
    rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
    assert(rc == SQLITE_OK);
    rc = sqlite3_step(stmt);
    assert(rc == SQLITE_ROW);
    nTotal = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
  }

  int64_t nRemaining = nTotal;


  sqlite3_stmt * stmt;
  const char * zSql = sqlite3_mprintf(
    " \
      WITH chunk AS ( \
        SELECT \
          e.rowid, \
          lembed(\"%w\") AS embedding \
        FROM \"%w\" AS e \
        LEFT JOIN \"%w\" AS src ON src.rowid = e.rowid \
        WHERE e.embedding IS NULL \
        LIMIT ? \
      ) \
      UPDATE \"%w\" AS e \
      SET embedding = chunk.embedding \
      FROM chunk \
      WHERE e.rowid = chunk.rowid \
      RETURNING rowid \
    ",
    column,
    tableEmbeddings,
    table,
    tableEmbeddings
  );
  assert(zSql);

  rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
  sqlite3_free((void *) zSql);
  assert(rc == SQLITE_OK);

  sqlite3_bind_int(stmt, 1, 16);

  int64_t nEmbed = 0;
  int64_t t0 = time_ms();

  while(1){
    sqlite3_reset(stmt);

    int nChunkEmbed = 0;
    while(1) {
      rc = sqlite3_step(stmt);
      if(rc == SQLITE_DONE) {
        break;
      }
      assert(rc == SQLITE_ROW);
      nChunkEmbed++;
    }
    if(nChunkEmbed == 0) {
      break;
    }
    nEmbed += nChunkEmbed;
    nRemaining -= nChunkEmbed;
    print_progress_bar(nEmbed, nTotal, time_ms() - t0);
  }




  rc = sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
  assert(rc == SQLITE_OK);

  sqlite3_free((void *) tableEmbeddings);
  sqlite3_close(db);
  return 0;
}

int cmd_embed(char * source) {
  int rc;
  sqlite3* db;
  sqlite3_stmt * stmt;

  rc = sqlite3_open(":memory:", &db);
  assert(rc == SQLITE_OK);

  rc = embedfile_sqlite3_init(db);
  assert(rc == SQLITE_OK);

  rc = sqlite3_prepare_v2(db, "select vec_to_json(lembed(?))", -1, &stmt, NULL);
  assert(rc == SQLITE_OK);

  sqlite3_bind_text(stmt, 1, source, strlen(source), SQLITE_STATIC);

  rc = sqlite3_step(stmt);
  assert(rc == SQLITE_ROW);

  printf("%.*s", sqlite3_column_bytes(stmt, 0), sqlite3_column_text(stmt, 0));

  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return 0;
}


int cmd_sh(int argc, char * argv[]) {
  return mn(argc, argv);
}

int main(int argc, char ** argv) {
    int rc;
    sqlite3* db;
    sqlite3_stmt* stmt;

    FLAG_log_disable = 1;
    argc = cosmo_args("/zip/.args", &argv);
    FLAGS_READY = 1;

    char * modelPath = NULL;
    for(int i = 1; i < argc; i++) {
      char * arg = argv[i];
      if(sqlite3_stricmp(arg, "--model") == 0 || sqlite3_stricmp(arg, "-m") == 0) {
        assert(++i <= argc);
        EMBEDFILE_MODEL = argv[i];
      }
      else if(sqlite3_stricmp(arg, "--version") == 0 || sqlite3_stricmp(arg, "-v") == 0) {
        fprintf(stderr,
          "embedfile %s, llamafile %s, SQLite %s, sqlite-vec=%s, sqlite-lembed=%s\n",
          EMBEDFILE_VERSION,
          LLAMAFILE_VERSION_STRING,
          sqlite3_version,
          SQLITE_VEC_VERSION,
          SQLITE_LEMBED_VERSION
        );
        return 0;
      }
      else if(sqlite3_stricmp(arg, "sh") == 0) {
        return cmd_sh(argc-i, argv+i);
      }
      else if(sqlite3_stricmp(arg, "embed") == 0) {
        assert(i + 2 == argc);
        return cmd_embed(argv[i+1]);
      }
      else if(sqlite3_stricmp(arg, "backfill") == 0) {
        assert(i + 4 == argc);
        char * dbpath = argv[i+1];
        char * table = argv[i+2];
        char * column = argv[i+3];
        return cmd_backfill(dbpath, table, column);
      }
      else if(sqlite3_stricmp(arg, "index") == 0) {
        assert(i + 3 == argc);
        char * path = argv[i+1];
        char * column = argv[i+2];
        return cmd_index(path, column);
      }
      else {
        printf("Unknown arg %s\n", arg);
        return 1;
      }
    }

    return 0;
}

