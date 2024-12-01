#ifndef QUICKJS_SQLITE_H
#define QUICKJS_SQLITE_H

#include "third_party/quickjs/quickjs.h"

#ifdef __cplusplus
extern "C" {
#endif

JSModuleDef *js_init_module_sqlite(JSContext *ctx, const char *module_name);

#ifdef __cplusplus
}  /* end of the 'extern "C"' block */
#endif

#endif /* ifndef QUICKJS_SQLITE_H */
