#ifndef SQLITE_LEMBED_H
#define SQLITE_LEMBED_H

//#include "sqlite3ext.h"
#include "sqlite3.h"
#include "llama.cpp/llama.h"

#define SQLITE_LEMBED_VERSION "v0.0.1-alpha.8"
#define SQLITE_LEMBED_DATE "2024-09-29T16:36:54Z-0700"
#define SQLITE_LEMBED_SOURCE "23fe65121d9a440bccc5f46ff89e33f81d02fcb4"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_lembed_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);

#ifdef __cplusplus
}  /* end of the 'extern "C"' block */
#endif

#endif /* ifndef SQLITE_LEMBED_H */
