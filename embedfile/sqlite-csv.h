#ifndef SQLITE_CSV_H
#define SQLITE_CSV_H

//#include "sqlite3ext.h"
#include "third_party/sqlite/sqlite3.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_csv_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);

#ifdef __cplusplus
}  /* end of the 'extern "C"' block */
#endif

#endif /* ifndef SQLITE__H */
