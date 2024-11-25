//#include "sqlite3ext.h"
#include "sqlite-lines.h"
#include "sqlite3.h"
///SQLITE_EXTENSION_INIT1

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma region sqlite - lines meta scalar functions

// TODO is this deterministic?
static void linesVersionFunc(sqlite3_context *context, int argc,
                             sqlite3_value **argv) {
  sqlite3_result_text(context, sqlite3_user_data(context), -1, SQLITE_STATIC);
}

// TODO is this deterministic?
static void linesDebugFunc(sqlite3_context *context, int argc,
                           sqlite3_value **arg) {
  sqlite3_result_text(context, sqlite3_user_data(context), -1, SQLITE_STATIC);
}

#pragma endregion

#pragma region lines() and lines_read() table functions

typedef struct lines_cursor lines_cursor;
struct lines_cursor {
  sqlite3_vtab_cursor base; /* Base class - must be first */
  // File pointer of the file being "read" (or in memory file for lines())
  FILE *fp;
  // length of current line
  size_t curLineLength;
  char *curLineContents;
  size_t curLineLen;
  char delim;
  int idxNum;
  int rowid_eq_yielded;
  // either the path to the file being read (lines_read()),
  // or the contents of the "document" (lines())
  char *in;
  sqlite3_int64 iRowid; /* The rowid */
};

#define LINES_READ_COLUMN_ROWID -1
#define LINES_READ_COLUMN_LINE 0
#define LINES_READ_COLUMN_PATH 1
#define LINES_READ_COLUMN_DELIM 2

#define LINES_IDXNUM_FULL 1
#define LINES_IDXNUM_ROWID_EQ 2

#define LINES_IDXSTR_PATH 'P'
#define LINES_IDXSTR_DELIMITER 'D'
#define LINES_IDXSTR_ROWID 'R'
#define LINES_IDXSTR_LENGTH 3
/*
** The linesReadConnect() method is invoked to create a new
** lines_vtab that describes the lines_read virtual table.
**
** Think of this routine as the constructor for lines_vtab objects.
**
** All this routine needs to do is:
**
**    (1) Allocate the lines_vtab object and initialize all fields.
**
**    (2) Tell SQLite (via the sqlite3_declare_vtab() interface) what the
**        result set of queries against lines_read will look like.
*/
static int linesConnect(sqlite3 *db, void *pUnused, int argcUnused,
                        const char *const *argvUnused, sqlite3_vtab **ppVtab,
                        char **pzErrUnused) {
  sqlite3_vtab *pNew;
  int rc;
  (void)pUnused;
  (void)argcUnused;
  (void)argvUnused;
  (void)pzErrUnused;
  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(line text,"
                                "document hidden, delimiter hidden)");
  if (rc == SQLITE_OK) {
    pNew = *ppVtab = sqlite3_malloc(sizeof(*pNew));
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
    sqlite3_vtab_config(db, SQLITE_VTAB_INNOCUOUS);
  }
  return rc;
}

/*
** This method is the destructor for lines_cursor objects.
*/
static int linesDisconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

/*
** Constructor for a new lines_cursor object.
*/
static int linesOpen(sqlite3_vtab *pUnused, sqlite3_vtab_cursor **ppCursor) {
  lines_cursor *pCur;
  (void)pUnused;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

/*
** Destructor for a lines_cursor.
*/
static int linesClose(sqlite3_vtab_cursor *cur) {
  lines_cursor *pCur = (lines_cursor *)cur;
  if (pCur->curLineContents != NULL)
    free(pCur->curLineContents);
  if (pCur->fp != NULL)
    fclose(pCur->fp);
  sqlite3_free(cur);
  return SQLITE_OK;
}

/*
** Advance a lines_cursor to its next row of output.
*/
static int linesNext(sqlite3_vtab_cursor *cur) {
  lines_cursor *pCur = (lines_cursor *)cur;
  pCur->iRowid++;
  pCur->curLineLength = getdelim(&pCur->curLineContents, &pCur->curLineLen,
                                 pCur->delim, pCur->fp);
  return SQLITE_OK;
}

/*
** Return TRUE if the cursor has been moved off of the last
** row of output.
*/
static int linesEof(sqlite3_vtab_cursor *cur) {
  lines_cursor *pCur = (lines_cursor *)cur;
  if (pCur->idxNum == LINES_IDXNUM_ROWID_EQ) {
    if (pCur->rowid_eq_yielded)
      return 1;
    pCur->rowid_eq_yielded = 1;
    return 0;
  }
  return pCur->curLineLength == -1;
}

/*
** Return values of columns for the row at which the lines_cursor
** is currently pointing.
*/
static int
linesColumn(sqlite3_vtab_cursor *cur, /* The cursor */
            sqlite3_context *ctx, /* First argument to sqlite3_result_...() */
            int i                 /* Which column to return */
) {
  lines_cursor *pCur = (lines_cursor *)cur;
  sqlite3_int64 x = 0;
  switch (i) {
  case LINES_READ_COLUMN_LINE: {
    // If the line ends in the delimiter character, then shave it off.
    // If the delimter is '\n' and the line ends with '\r\n', then also
    // shave  off that '\r', to support CRLF files.
    int trim = 0;
    if (pCur->curLineLength > 0 &&
        pCur->curLineContents[pCur->curLineLength - 1] == pCur->delim) {
      if (pCur->curLineLength > 1 &&
          pCur->curLineContents[pCur->curLineLength - 1] == '\n' &&
          pCur->curLineContents[pCur->curLineLength - 2] == '\r')
        trim = 2;
      else
        trim = 1;
    }
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    int mxBlob = sqlite3_limit(db, SQLITE_LIMIT_LENGTH, -1);
    if (pCur->curLineLength > mxBlob) {
      sqlite3_result_error_code(ctx, SQLITE_TOOBIG);
      sqlite3_result_error(
          ctx,
          sqlite3_mprintf(
              "line %d has a size of %d bytes, but SQLITE_LIMIT_LENGTH is %d",
              pCur->iRowid, pCur->curLineLength, mxBlob),
          -1);
      return SQLITE_ERROR;
    }
    sqlite3_result_text(ctx, pCur->curLineContents, pCur->curLineLength - trim,
                        SQLITE_TRANSIENT);
    break;
  }
  case LINES_READ_COLUMN_DELIM: {
    sqlite3_result_text(ctx, &pCur->delim, 1, SQLITE_TRANSIENT);
    break;
  }
  case LINES_READ_COLUMN_PATH: {
    sqlite3_result_text(ctx, pCur->in, -1, SQLITE_TRANSIENT);
    break;
  }
  }
  return SQLITE_OK;
}

/*
** Return the rowid for the current row. In this implementation, the
** first row returned is assigned rowid value 1, and each subsequent
** row a value 1 more than that of the previous.
*/
static int linesRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  lines_cursor *pCur = (lines_cursor *)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}

/*
** SQLite will invoke this method one or more times while planning a query
** that uses the lines_read virtual table.  This routine needs to create
** a query plan for each invocation and compute an estimated cost for that
** plan.
*/
/*
  Every query plan for lines() or lines_read() will use idxNum and idxStr.

  idxNum options:
    LINES_IDXNUM_FULL: "do a full scan", ie read all lines from file/document
    LINES_IDXNUM_ROWID_EQ: Only read a single line, defined by a "rowid = :x"
  constraint

  idxStr is a 3-character string that denotes which argv option cooresponds
  to which column constraint. The i-th character in the string cooresponds
  to the i-th argv option in the xFilter functions.

  idxStr character options:
    LINES_IDXSTR_PATH: argv[i] is text to the path of the file or the document
  itself LINES_IDXSTR_DELIMITER: argv[i] will be text of delimiter to use
    LINES_IDXSTR_ROWID: argv[i] is integer of rowid to filter to, with
  LINES_IDXNUM_ROWID_EQ

*/

static int linesBestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
  int hasPath = 0;
  int hasDelim = 0;
  int hasRowidEq = 0;
  int argv = 1;

  pIdxInfo->idxStr = sqlite3_mprintf("000");

  if (pIdxInfo->idxStr == NULL) {
    pVTab->zErrMsg = sqlite3_mprintf("unable to allocate memory for idxStr");
    return SQLITE_NOMEM;
  }

  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *pCons = &pIdxInfo->aConstraint[i];
#ifdef SQLITE_LINES_DEBUG
    printf("i=%d iColumn=%d, op=%d, usable=%d\n", i, pCons->iColumn, pCons->op,
           pCons->usable);
#endif
    switch (pCons->iColumn) {
    case LINES_READ_COLUMN_ROWID: {
      if (pCons->op == SQLITE_INDEX_CONSTRAINT_EQ && pCons->usable) {
        hasRowidEq = 1;
        pIdxInfo->aConstraintUsage[i].argvIndex = argv;
        pIdxInfo->aConstraintUsage[i].omit = 1;
        pIdxInfo->idxStr[argv - 1] = LINES_IDXSTR_ROWID;
        argv++;
      }
      break;
    }
    case LINES_READ_COLUMN_PATH: {
      if (!hasPath && !pCons->usable || pCons->op != SQLITE_INDEX_CONSTRAINT_EQ)
        return SQLITE_CONSTRAINT;
      hasPath = 1;
      pIdxInfo->aConstraintUsage[i].argvIndex = argv;
      pIdxInfo->aConstraintUsage[i].omit = 1;
      pIdxInfo->idxStr[argv - 1] = LINES_IDXSTR_PATH;
      argv++;
      break;
    }
    case LINES_READ_COLUMN_DELIM: {
      if (!pCons->usable || pCons->op != SQLITE_INDEX_CONSTRAINT_EQ)
        return SQLITE_CONSTRAINT;
      hasDelim = 1;
      pIdxInfo->aConstraintUsage[i].argvIndex = argv;
      pIdxInfo->aConstraintUsage[i].omit = 1;
      pIdxInfo->idxStr[argv - 1] = LINES_IDXSTR_DELIMITER;
      argv++;
      break;
    }
    }
  }
  if (!hasPath) {
    pVTab->zErrMsg = sqlite3_mprintf("path argument is required");
    return SQLITE_ERROR;
  }
  if (hasRowidEq) {
    pIdxInfo->idxNum = LINES_IDXNUM_ROWID_EQ;
    pIdxInfo->estimatedCost = (double)1;
    pIdxInfo->estimatedRows = 1;
    // pIdxInfo->idxFlags |= SQLITE_INDEX_SCAN_UNIQUE;
    return SQLITE_OK;
  }
  pIdxInfo->idxNum = LINES_IDXNUM_FULL;
  pIdxInfo->needToFreeIdxStr = 1;
  pIdxInfo->estimatedCost = (double)100000;
  pIdxInfo->estimatedRows = 100000;

  return SQLITE_OK;
}

/*
** This method is called to "rewind" the lines_cursor object back
** to the first row of output.  This method is always called at least
** once prior to any call to xColumn() or xRowid() or xEof().
**
** This routine should initialize the cursor and position it so that it
** is pointing at the first row, or pointing off the end of the table
** (so that xEof() will return true) if the table is empty.
*/
static int linesFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                       const char *idxStr, int argc, sqlite3_value **argv) {
  int targetRowid;
  char delim = '\n';
  lines_cursor *pCur = (lines_cursor *)pVtabCursor;
  if (pCur->fp != NULL) {
    fclose(pCur->fp);
  }
  if (pCur->curLineContents != NULL)
    free(pCur->curLineContents);

  for (int i = 0; i < LINES_IDXSTR_LENGTH; i++) {
    switch (idxStr[i]) {
    case LINES_IDXSTR_ROWID: {
      targetRowid = sqlite3_value_int64(argv[i]);
      break;
    }
    case LINES_IDXSTR_PATH: {
      int nByte = sqlite3_value_bytes(argv[i]);
      void *pData = (void *)sqlite3_value_blob(argv[i]);
      int errnum;
      pCur->fp = fmemopen(pData, nByte, "r");
      if (pCur->fp == NULL) {
        int errnum;
        errnum = errno;
        pVtabCursor->pVtab->zErrMsg = sqlite3_mprintf(
            "Error reading document, size=%d: %s", nByte, strerror(errnum));
        return SQLITE_ERROR;
      }
      break;
    }
    case LINES_IDXSTR_DELIMITER: {
      int nByte = sqlite3_value_bytes(argv[i]);
      if (nByte != 1) {
        pVtabCursor->pVtab->zErrMsg = sqlite3_mprintf(
            "Delimiter must be 1 character long, got %d characters", nByte);
        return SQLITE_ERROR;
      }
      const char *s = (const char *)sqlite3_value_text(argv[i]);
      delim = s[0];
      break;
    }
    }
  }

  pCur->curLineContents = 0;
  pCur->curLineLength =
      getdelim(&pCur->curLineContents, &pCur->curLineLen, delim, pCur->fp);
  pCur->iRowid = 1;
  pCur->delim = delim;
  pCur->idxNum = idxNum;
  pCur->in = "";

  if (pCur->idxNum == LINES_IDXNUM_ROWID_EQ) {
    pCur->rowid_eq_yielded = 0;
    while (pCur->iRowid < targetRowid && pCur->curLineLength >= 0) {
      pCur->curLineLength =
          getdelim(&pCur->curLineContents, &pCur->curLineLen, delim, pCur->fp);
      pCur->iRowid++;
    }
  }
  return SQLITE_OK;
}

static int linesReadFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                           const char *idxStr, int argc, sqlite3_value **argv) {
  int targetRowid;
  char delim = '\n';

  lines_cursor *pCur = (lines_cursor *)pVtabCursor;
  if (pCur->fp != NULL) {
    fclose(pCur->fp);
  }
  if (pCur->curLineContents != NULL)
    free(pCur->curLineContents);

  for (int i = 0; i < LINES_IDXSTR_LENGTH; i++) {
    switch (idxStr[i]) {
    case LINES_IDXSTR_ROWID: {
      targetRowid = sqlite3_value_int64(argv[i]);
      break;
    }
    case LINES_IDXSTR_PATH: {
      if (sqlite3_value_type(argv[i]) == SQLITE_NULL) {
        pVtabCursor->pVtab->zErrMsg = sqlite3_mprintf("path is null");
        return SQLITE_ERROR;
      }
      char *path = (char *)sqlite3_value_text(argv[i]);
      // TODO should we free this later?
      pCur->in = (char *)path;

      int errnum;
      pCur->fp = fopen(path, "r");
      if (pCur->fp == NULL) {
        int errnum;
        errnum = errno;
        pVtabCursor->pVtab->zErrMsg =
            sqlite3_mprintf("Error reading %s: %s", path, strerror(errnum));
        return SQLITE_ERROR;
      }
      break;
    }
    case LINES_IDXSTR_DELIMITER: {
      int nByte = sqlite3_value_bytes(argv[i]);
      if (nByte != 1) {
        pVtabCursor->pVtab->zErrMsg = sqlite3_mprintf(
            "Delimiter must be 1 character long, got %d characters", nByte);
        return SQLITE_ERROR;
      }
      const char *s = (const char *)sqlite3_value_text(argv[i]);
      delim = s[0];
      break;
    }
    }
  }

  pCur->curLineContents = 0;
  pCur->curLineLength =
      getdelim(&pCur->curLineContents, &pCur->curLineLen, delim, pCur->fp);
  pCur->iRowid = 1;
  pCur->delim = delim;
  pCur->idxNum = idxNum;

  if (pCur->idxNum == LINES_IDXNUM_ROWID_EQ) {
    pCur->rowid_eq_yielded = 0;
    while (pCur->iRowid < targetRowid && pCur->curLineLength >= 0) {
      pCur->curLineLength =
          getdelim(&pCur->curLineContents, &pCur->curLineLen, delim, pCur->fp);
      pCur->iRowid++;
    }
  }
  return SQLITE_OK;
}

static int linesReadConnect(sqlite3 *db, void *pUnused, int argcUnused,
                            const char *const *argvUnused,
                            sqlite3_vtab **ppVtab, char **pzErrUnused) {
  sqlite3_vtab *pNew;
  int rc;
  (void)pUnused;
  (void)argcUnused;
  (void)argvUnused;
  (void)pzErrUnused;
  // only difference is schema, uses "path" instead of "document"
  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(line text,"
                                "path hidden, delimiter hidden)");
  if (rc == SQLITE_OK) {
    pNew = *ppVtab = sqlite3_malloc(sizeof(*pNew));
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
    sqlite3_vtab_config(db, SQLITE_VTAB_INNOCUOUS);
  }
  return rc;
}

static sqlite3_module linesModule = {
    0,               /* iVersion */
    0,               /* xCreate */
    linesConnect,    /* xConnect */
    linesBestIndex,  /* xBestIndex */
    linesDisconnect, /* xDisconnect */
    0,               /* xDestroy */
    linesOpen,       /* xOpen - open a cursor */
    linesClose,      /* xClose - close a cursor */
    linesFilter,     /* xFilter - configure scan constraints */
    linesNext,       /* xNext - advance a cursor */
    linesEof,        /* xEof - check for end of scan */
    linesColumn,     /* xColumn - read data */
    linesRowid,      /* xRowid - read data */
    0,               /* xUpdate */
    0,               /* xBegin */
    0,               /* xSync */
    0,               /* xCommit */
    0,               /* xRollback */
    0,               /* xFindMethod */
    0,               /* xRename */
    0,               /* xSavepoint */
    0,               /* xRelease */
    0,               /* xRollbackTo */
    0                /* xShadowName */
};

static sqlite3_module linesReadModule = {
    0,                /* iVersion */
    0,                /* xCreate */
    linesReadConnect, /* xConnect */
    linesBestIndex,   /* xBestIndex */
    linesDisconnect,  /* xDisconnect */
    0,                /* xDestroy */
    linesOpen,        /* xOpen - open a cursor */
    linesClose,       /* xClose - close a cursor */
    linesReadFilter,  /* xFilter - configure scan constraints */
    linesNext,        /* xNext - advance a cursor */
    linesEof,         /* xEof - check for end of scan */
    linesColumn,      /* xColumn - read data */
    linesRowid,       /* xRowid - read data */
    0,                /* xUpdate */
    0,                /* xBegin */
    0,                /* xSync */
    0,                /* xCommit */
    0,                /* xRollback */
    0,                /* xFindMethod */
    0,                /* xRename */
    0,                /* xSavepoint */
    0,                /* xRelease */
    0,                /* xRollbackTo */
    0                 /* xShadowName */
};

#pragma endregion

#pragma region entry points

#ifdef _WIN32
__declspec(dllexport)
#endif
    int sqlite3_lines_init(sqlite3 *db, char **pzErrMsg,
                           const sqlite3_api_routines *pApi) {
  int rc = SQLITE_OK;
  //SQLITE_EXTENSION_INIT2(pApi);

  (void)pzErrMsg; /* Unused parameter */
  int flags = SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC;
  const char *debug = sqlite3_mprintf(
      "Version: %s\nDate: %s\nSource: %s", SQLITE_LINES_VERSION,
      SQLITE_LINES_DATE, SQLITE_LINES_SOURCE);

  if (rc == SQLITE_OK)
    rc = sqlite3_create_function_v2(db, "lines_version", 0, flags,
                                    (void *)SQLITE_LINES_VERSION,
                                    linesVersionFunc, 0, 0, 0);
  if (rc == SQLITE_OK)
    rc = sqlite3_create_function_v2(db, "lines_debug", 0, flags, (void *)debug,
                                    linesDebugFunc, 0, 0, sqlite3_free);

  if (rc == SQLITE_OK)
    rc = sqlite3_create_module(db, "lines", &linesModule, 0);
  if (rc == SQLITE_OK)
    rc = sqlite3_create_module(db, "lines_read", &linesReadModule, 0);
  return rc;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int sqlite3_lines_no_read_init(sqlite3 *db, char **pzErrMsg,
                                   const sqlite3_api_routines *pApi) {
  int rc = SQLITE_OK;
  //SQLITE_EXTENSION_INIT2(pApi);

  (void)pzErrMsg; /* Unused parameter */
  int flags = SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC;

  const char *debug = sqlite3_mprintf(
      "Version: %s\nDate: %s\nSource: %s\nNO FILESYSTEM", SQLITE_LINES_VERSION,
      SQLITE_LINES_DATE, SQLITE_LINES_SOURCE);

  if (rc == SQLITE_OK)
    rc = sqlite3_create_function_v2(db, "lines_version", 0, flags,
                                    (void *)SQLITE_LINES_VERSION,
                                    linesVersionFunc, 0, 0, 0);
  if (rc == SQLITE_OK)
    rc = sqlite3_create_function_v2(db, "lines_debug", 0, flags, (void *)debug,
                                    linesDebugFunc, 0, 0, sqlite3_free);

  if (rc == SQLITE_OK)
    rc = sqlite3_create_module(db, "lines", &linesModule, 0);
  return rc;
}

#pragma endregion
