#include "sqlite3.h"
//#include "sqlite3ext.h"

#define SQLITE_LINES_VERSION "TODO"
#define SQLITE_LINES_DATE "TODO"
#define SQLITE_LINES_SOURCE "TODO"

#ifdef SQLITE_LINES_ENTRYPOINT
int SQLITE_LINES_ENTRYPOINT(
#else
int sqlite3_lines_init(
#endif
    sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);
