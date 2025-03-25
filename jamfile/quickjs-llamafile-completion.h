#ifndef QUICKJS_COMPLETIONMODEL_H
#define QUICKJS_COMPLETIONMODEL_H

#include "third_party/quickjs/quickjs.h"

#ifdef __cplusplus
extern "C" {
#endif

int js_llamafile_init_completion_model(JSContext *ctx, JSModuleDef *m, char * default_model);

#ifdef __cplusplus
}  /* end of the 'extern "C"' block */
#endif

#endif /* ifndef QUICKJS_COMPLETIONMODEL_H */
