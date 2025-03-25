
#ifndef SHELL_H
#define SHELL_H

#include "third_party/sqlite/sqlite3.h"

#ifdef __cplusplus
extern "C" {
#endif

int mn( int argc, char * argv[]);
int sqlite3_fileio_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);
#ifdef __cplusplus
}  /* end of the 'extern "C"' block */
#endif

#endif /* ifndef SHELL_H */
