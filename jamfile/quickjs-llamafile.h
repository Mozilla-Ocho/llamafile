#ifndef QUICKJS_LLAMAFILE_H
#define QUICKJS_LLAMAFILE_H

#include "third_party/quickjs/quickjs.h"

#ifdef __cplusplus
extern "C" {
#endif

JSModuleDef *js_init_module_llamafile(JSContext *ctx, const char *module_name, char *, char * );

#ifdef __cplusplus
}  /* end of the 'extern "C"' block */
#endif

#endif /* ifndef QUICKJS_LLAMAFILE_H */
