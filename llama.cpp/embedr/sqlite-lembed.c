#include "sqlite-lembed.h"
#include "llama.cpp/llama.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sqlite3.h"
//#include "sqlite3ext.h"
//SQLITE_EXTENSION_INIT1

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(X) (void)(X)
#endif

#define SQLITE_VEC_FLOAT32_SUBTYPE 223

void dummy_log(enum ggml_log_level level, const char *text, void *user_data) {}

static void normalize(float *vec, float *out, int n) {
  double norm = 0;
  for (int i = 0; i < n; i++) {
    norm += vec[i] * vec[i];
  }
  norm = sqrt(norm);
  for (int i = 0; i < n; i++) {
    out[i] = vec[i] / norm;
  }
}

#define LEMBED_TOKEN_SUBTYPE 116 // ascii 't'

int tokenize(struct llama_model *model, const char *input, size_t input_length,
             int *token_count, llama_token **tokens) {
  int input_token_count_estimate =
      llama_tokenize(model, input, input_length, NULL, 0, true, true);
  if (input_token_count_estimate >= 0) {
    return SQLITE_ERROR;
  }
  *tokens =
      sqlite3_malloc(sizeof(llama_token) * abs(input_token_count_estimate));
  if (!(*tokens)) {
    return SQLITE_NOMEM;
  }
  int input_token_count =
      llama_tokenize(model, input, input_length, *tokens,
                     abs(input_token_count_estimate), true, true);
  if (input_token_count != abs(input_token_count_estimate)) {
    sqlite3_free(*tokens);
    return SQLITE_ERROR;
  }

  *token_count = input_token_count;
  return SQLITE_OK;
}


int embed_single(struct llama_context *context,
                 const char *input, size_t input_length,
                 /** Output float embedding */
                 float **out_embedding,
                 /** Output embedding length (n dimensions) */
                 int *out_dimensions,
                 char ** errmsg) {
  struct llama_model * model = (struct llama_model *) llama_get_model(context);

  int n_ctx_train = llama_n_ctx_train(model);
  int n_ctx = llama_n_ctx(context);

  int dimensions = llama_n_embd(model);
  float *output_embedding = sqlite3_malloc(sizeof(float) * dimensions);
  if(!output_embedding) {
    return SQLITE_NOMEM;
  }

  llama_token *tokens;
  int token_count;
  int rc = tokenize(model, input, input_length, &token_count, &tokens);
  if(rc != SQLITE_OK) {
    // TODO error message
    *errmsg = sqlite3_mprintf("Could not tokenize input.");
    return rc;
  }

  if(token_count > n_ctx) {
    *errmsg = sqlite3_mprintf("Input too long, provided %lld tokens, but model has context size of %lld", (int64_t) token_count, (int64_t) n_ctx);
    return SQLITE_ERROR;
  }

  struct llama_batch batch = llama_batch_init(n_ctx, 0, 1);

  int seq_id = 0;
  // llama_batch_add(batch, tokens, 0, )
  for (int i = 0; i < token_count; i++) {
    batch.token[batch.n_tokens] = tokens[i];
    batch.pos[batch.n_tokens] = i;

    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = seq_id;

    batch.logits[batch.n_tokens] = i == (token_count - 1);
    batch.n_tokens++;
  }

  llama_kv_cache_clear(context); // KV not needed for embeddings?
  rc = llama_decode(context, batch);
  if(rc != 0) {
    sqlite3_free(output_embedding);
    llama_batch_free(batch);
    *errmsg = sqlite3_mprintf("Could not decode batch");
    return SQLITE_ERROR;
  }

  float * source_embedding;
  if(llama_pooling_type(context) == LLAMA_POOLING_TYPE_NONE) {
    source_embedding = llama_get_embeddings(context);
  }
  else {
    source_embedding = llama_get_embeddings_seq(context, batch.seq_id[0][0]);
  }
  if(!source_embedding) {
    sqlite3_free(output_embedding);
    llama_batch_free(batch);
    *errmsg = sqlite3_mprintf("Could not find embedding");
    return SQLITE_ERROR;
  }

  normalize(source_embedding, output_embedding, dimensions);
  llama_batch_free(batch);

  *out_dimensions = dimensions;
  *out_embedding = output_embedding;
  return SQLITE_OK;
}

typedef struct ApiModel ApiModel;
struct ApiModel {
  char *name;
  struct llama_model *model;
  struct llama_context *context;
};

#define MAX_MODELS 16
struct Api {
  int default_index;
  ApiModel models[MAX_MODELS];
};

void api_free(void *p) {
  struct Api *a = (struct Api *)p;
  llama_backend_free();
  sqlite3_free(a);
}

typedef struct lembed_model_options lembed_model_options;
struct lembed_model_options {
  int32_t n_gpu_layers;

  int8_t defined[1];
};
static char *POINTER_NAME_MODEL = "lembed_model";
static char *POINTER_NAME_MODEL_OPTIONS = "lembed_model_options";

static void lembed_model_options_(sqlite3_context *context, int argc,
                                  sqlite3_value **argv) {

  if(argc % 2 == 0) {
    sqlite3_result_error(context, "an even number of arguments are required in lembed_model_options, key-value pairs", -1);
    return;
  }
  lembed_model_options *o = sqlite3_malloc(sizeof(lembed_model_options));
  if(!o) {
    sqlite3_result_error_nomem(context);
    return;
  }
  memset(o, 0, sizeof(*o));

  for (int i = 0; i < argc; i += 2) {
    sqlite3_value *key = argv[i];
    sqlite3_value *value = argv[i + 1];
    if(sqlite3_value_type(key) != SQLITE_TEXT) {
      char * errmsg = sqlite3_mprintf("Expected string key at index %d", i);
      sqlite3_result_error(context, errmsg, -1);
      sqlite3_free(errmsg);
      sqlite3_free(o);
      return;
    }
    const char *k = (const char *)sqlite3_value_text(key);
    if (sqlite3_stricmp(k, "n_gpu_layers") == 0) {
      o->n_gpu_layers = sqlite3_value_int(value);
      o->defined[0] = 1;
    } else {
      char * errmsg = sqlite3_mprintf("Unknown model option '%s'", k);
      sqlite3_result_error(context, errmsg, -1);
      sqlite3_free(errmsg);
      sqlite3_free(o);
      return;
    }
  }
  sqlite3_result_pointer(context, o, POINTER_NAME_MODEL_OPTIONS, sqlite3_free);
}

typedef struct lembed_context_options lembed_context_options;
struct lembed_context_options {
  uint32_t seed;
  uint32_t n_ctx;
  enum llama_rope_scaling_type rope_scaling_type;
  float rope_freq_scale;

  int8_t defined[4];
};
static char *POINTER_NAME_CONTEXT_OPTIONS = "lembed_context_options";

static void lembed_context_options_(sqlite3_context *context, int argc,
                                    sqlite3_value **argv) {
  if(argc % 2 == 0) {
    sqlite3_result_error(context, "an even number of arguments are required in lembed_context_options, key-value pairs", -1);
    return;
  }
  lembed_context_options *o = sqlite3_malloc(sizeof(lembed_context_options));
  if(!o) {
    sqlite3_result_error_nomem(context);
    return;
  }
  memset(o, 0, sizeof(*o));

  for (int i = 0; i < argc; i += 2) {
    sqlite3_value *key = argv[i];
    sqlite3_value *value = argv[i + 1];
    if(sqlite3_value_type(key) != SQLITE_TEXT) {
      char * errmsg = sqlite3_mprintf("Expected string value at index %d", i+1);
      sqlite3_result_error(context, errmsg, -1);
      sqlite3_free(errmsg);
      return;
    }
    const char *k = (const char *)sqlite3_value_text(key);
    if (sqlite3_stricmp("seed", k) == 0) {
      sqlite3_int64 v = sqlite3_value_int64(value);
      if(v < 0) {
        sqlite3_result_error(context, "Expected positive value for seed", -1);
        sqlite3_free(o);
        return;
      }
      o->seed = v;
      o->defined[0] = 1;
    } else if (sqlite3_stricmp("n_ctx", k) == 0) {
      sqlite3_int64 v = sqlite3_value_int64(value);
      if(v < 0) {
        sqlite3_result_error(context, "Expected positive value for n_ctx", -1);
        sqlite3_free(o);
        return;
      }
      o->n_ctx = v;
      o->defined[1] = 1;
    } else if (sqlite3_stricmp("rope_scaling_type", k) == 0) {
      const char *v = (const char *)sqlite3_value_text(value);
      if (sqlite3_stricmp(v, "none")) {
        o->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
      } else if (sqlite3_stricmp(v, "linear")) {
        o->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
      } else if (sqlite3_stricmp(v, "yarn")) {
        o->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
      } else {
        abort();
      }

      o->defined[2] = 1;
    } else if (sqlite3_stricmp(k, "rope_freq_scale") == 0) {
      o->rope_freq_scale = sqlite3_value_double(value);
      o->defined[3] = 1;
    } else {
      abort();
    }
  }
  sqlite3_result_pointer(context, o, POINTER_NAME_CONTEXT_OPTIONS,
                         sqlite3_free);
}
static char *POINTER_NAME_MODEL_PATH = "lembed_model_path";

static void lembed_model_from_file(sqlite3_context *context, int argc,
                                   sqlite3_value **argv) {
  sqlite3_result_pointer(context,
                         sqlite3_mprintf("%.*s", sqlite3_value_bytes(argv[0]),
                                         sqlite3_value_text(argv[0])),
                         POINTER_NAME_MODEL_PATH, sqlite3_free);
}


static void _static_text_func(sqlite3_context *context, int argc,
                              sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  sqlite3_result_text(context, sqlite3_user_data(context), -1, SQLITE_STATIC);
}

int api_model_from_name(struct Api *api, const char *name, int name_length,
                        struct llama_model **model,
                        struct llama_context **context) {
  for (int i = 0; i < MAX_MODELS; i++) {
    if (!api->models[i].name)
      continue;
    if (strncmp(api->models[i].name, name, name_length) == 0) {
      *model = api->models[i].model;
      if (context)
        *context = api->models[i].context;
      return SQLITE_OK;
    }
  }
  return SQLITE_ERROR;
}
static void lembed(sqlite3_context *context, int argc, sqlite3_value **argv) {
  struct llama_model *model;
  struct llama_context *ctx;
  int rc;
  const char * input;
  sqlite3_int64 input_len;
  if(argc == 1) {
    input = (const char *)sqlite3_value_text(argv[0]);
    input_len = sqlite3_value_bytes(argv[0]);
    rc = api_model_from_name((struct Api *)sqlite3_user_data(context), "default", strlen("default"), &model, &ctx);
    if(rc != SQLITE_OK) {
      sqlite3_result_error(context, "No default model has been registered yet with lembed_models", -1);
      return;
    }
  }else {
    input = (const char *)sqlite3_value_text(argv[1]);
    input_len = sqlite3_value_bytes(argv[1]);
    rc = api_model_from_name((struct Api *)sqlite3_user_data(context),
                               (const char *)sqlite3_value_text(argv[0]),
                               sqlite3_value_bytes(argv[0]), &model, &ctx);

    if(rc != SQLITE_OK) {
      char * zSql = sqlite3_mprintf("Unknown model name '%s'. Was it registered with lembed_models?", sqlite3_value_text(argv[0]));
      sqlite3_result_error(context, zSql, -1);
      sqlite3_free(zSql);
      return;
    }
  }

  int dimensions;
  float *embedding;
  char * errmsg;
  rc = embed_single(ctx, input, input_len, &embedding, &dimensions, &errmsg);
  if(rc != SQLITE_OK) {
    sqlite3_result_error(context, sqlite3_mprintf("Error generating embedding: %z", errmsg), -1);
    return;
  }
  sqlite3_result_blob(context, embedding, sizeof(float) * dimensions, sqlite3_free);
  sqlite3_result_subtype(context, SQLITE_VEC_FLOAT32_SUBTYPE);
}

static void lembed_tokenize_json(sqlite3_context *context, int argc,
                                 sqlite3_value **argv) {
  int rc;
  struct llama_model *model;
  struct llama_context *ctx;
  const char *input;
  sqlite3_int64 input_len;

  if(argc == 1) {
    input = (const char *)sqlite3_value_text(argv[0]);
    input_len = sqlite3_value_bytes(argv[0]);
    rc = api_model_from_name((struct Api *)sqlite3_user_data(context), "default", strlen("default"), &model, &ctx);
    if(rc != SQLITE_OK) {
      sqlite3_result_error(context, "No default model has been registered yet with lembed_models", -1);
      return;
    }
  }else {
    input = (const char *)sqlite3_value_text(argv[1]);
    input_len = sqlite3_value_bytes(argv[1]);
    rc = api_model_from_name((struct Api *)sqlite3_user_data(context),
                               (const char *)sqlite3_value_text(argv[0]),
                               sqlite3_value_bytes(argv[0]), &model, &ctx);

    if(rc != SQLITE_OK) {
      char * zSql = sqlite3_mprintf("Unknown model name '%s'. Was it registered with lembed_models?", sqlite3_value_text(argv[0]));
      sqlite3_result_error(context, zSql, -1);
      sqlite3_free(zSql);
      return;
    }
  }

  int token_count;
  llama_token *tokens;
  rc = tokenize(model, input, input_len, &token_count, &tokens);
  if(rc != SQLITE_OK) {
    sqlite3_result_error(context, "Failed to tokenize input", -1);
    return;
  }

  sqlite3_str *s = sqlite3_str_new(NULL);
  sqlite3_str_appendchar(s, 1, '[');
  for (int i = 0; i < token_count; i++) {
    if (i != 0) {
      sqlite3_str_appendchar(s, 1, ',');
    }
    sqlite3_str_appendf(s, "%d", tokens[i]);
  }
  sqlite3_str_appendchar(s, 1, ']');
  char *result = sqlite3_str_finish(s);
  if(!result) {
    sqlite3_result_error_nomem(context);
  }else {
    sqlite3_result_text(context, result, -1, sqlite3_free);
  }
}

static void lembed_token_score(sqlite3_context *context, int argc,
                               sqlite3_value **argv) {
  struct llama_model *model;
  int rc = api_model_from_name((struct Api *)sqlite3_user_data(context),
                               (const char *)sqlite3_value_text(argv[0]),
                               sqlite3_value_bytes(argv[0]), &model, NULL);

  int32_t token = sqlite3_value_int(argv[1]);

  float score = llama_token_get_score(model, token);
  sqlite3_result_double(context, score);
}
static void lembed_token_to_piece_(sqlite3_context *context, int argc,
                                   sqlite3_value **argv) {
  struct llama_model *model;
  int rc = api_model_from_name((struct Api *)sqlite3_user_data(context),
                               (const char *)sqlite3_value_text(argv[0]),
                               sqlite3_value_bytes(argv[0]), &model, NULL);

  int32_t token = sqlite3_value_int(argv[1]);
#define BUFLEN 256
  char buf[BUFLEN];
  int n = llama_token_to_piece(model, token, buf, BUFLEN, 0, false);
  if (n) {
    sqlite3_result_text(context, buf, n, SQLITE_TRANSIENT);
  } else {
    sqlite3_result_null(context);
  }
}

static void _noop(sqlite3_context *context, int argc, sqlite3_value **argv) {}
static void ggml_test(sqlite3_context *context, int argc,
                      sqlite3_value **argv) {
  sqlite3_result_int64(context, ggml_cpu_has_avx());
}


void lembed_vtab_set_error(sqlite3_vtab *pVTab, const char *zFormat, ...) {
  va_list args;
  sqlite3_free(pVTab->zErrMsg);
  va_start(args, zFormat);
  pVTab->zErrMsg = sqlite3_vmprintf(zFormat, args);
  va_end(args);
}

#pragma region lembed_models() table function

typedef struct lembed_models_vtab lembed_models_vtab;
struct lembed_models_vtab {
  sqlite3_vtab base;
  struct Api *api;
};

typedef struct lembed_models_cursor lembed_models_cursor;
struct lembed_models_cursor {
  sqlite3_vtab_cursor base;
  sqlite3_int64 iRowid;
};

static int lembed_modelsConnect(sqlite3 *db, void *pAux, int argc,
                                const char *const *argv, sqlite3_vtab **ppVtab,
                                char **pzErr) {
  lembed_models_vtab *pNew;
  int rc;
  if (strcmp(argv[1], "temp") != 0) {
    // return SQLITE_ERROR;
  }
#define LEMBED_MODELS_NAME            0
#define LEMBED_MODELS_MODEL           1
#define LEMBED_MODELS_SIZE            2
#define LEMBED_MODELS_DIMENSIONS      3
#define LEMBED_MODELS_N_CTX           4
#define LEMBED_MODELS_POOLING_TYPE    5
#define LEMBED_MODELS_MODEL_OPTIONS   6
#define LEMBED_MODELS_CONTEXT_OPTIONS 7
  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(name, model, size, dimensions, n_ctx, pooling_type, model_options "
                                "hidden, context_options hidden)");
  if (rc == SQLITE_OK) {
    pNew = sqlite3_malloc(sizeof(*pNew));
    *ppVtab = (sqlite3_vtab *)pNew;
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
    pNew->api = pAux;
  }
  return rc;
}

static int lembed_modelsDisconnect(sqlite3_vtab *pVtab) {
  lembed_models_vtab *p = (lembed_models_vtab *)pVtab;
  sqlite3_free(p);
  return SQLITE_OK;
}

#define POINTER_SUBTYPE 112

static int lembed_modelsUpdate(sqlite3_vtab *pVTab, int argc,
                               sqlite3_value **argv, sqlite_int64 *pRowid) {
  lembed_models_vtab *p = (lembed_models_vtab *)pVTab;
  // DELETE operation
  if (argc == 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    return SQLITE_ERROR;
  }
  // INSERT operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    sqlite3_value **columnValues = &argv[2];
    const char *key;
    if(sqlite3_value_type(columnValues[LEMBED_MODELS_NAME]) == SQLITE_NULL) {
      key = "default";
    }else {
      key = (const char *)sqlite3_value_text(columnValues[LEMBED_MODELS_NAME]);
    }

    int idx = -1;
    for (int i = 0; i < MAX_MODELS; i++) {
      if (!p->api->models[i].name) {
        p->api->models[i].name = sqlite3_mprintf("%s", key);
        idx = i;
        break;
      }
    }
    if (idx < 0)
      abort();


    const char *modelPath;
    if(sqlite3_value_subtype(columnValues[LEMBED_MODELS_MODEL]) == POINTER_SUBTYPE) {
      modelPath = sqlite3_value_pointer(columnValues[LEMBED_MODELS_MODEL], POINTER_NAME_MODEL_PATH);
    }
    else if (sqlite3_value_type(columnValues[LEMBED_MODELS_MODEL]) == SQLITE_TEXT) {
      modelPath = sqlite3_value_text(columnValues[LEMBED_MODELS_MODEL]);
    }
    if(!modelPath) {
      lembed_vtab_set_error(pVTab, "Could not resolve model path");
      return SQLITE_ERROR;
    }

    lembed_model_options *modelOptions = NULL;
    if (sqlite3_value_subtype(columnValues[LEMBED_MODELS_MODEL_OPTIONS]) ==
        POINTER_SUBTYPE) {
      modelOptions =
          sqlite3_value_pointer(columnValues[LEMBED_MODELS_MODEL_OPTIONS],
                                POINTER_NAME_MODEL_OPTIONS);
    }

    lembed_context_options *contextOptions = NULL;
    if (sqlite3_value_subtype(columnValues[LEMBED_MODELS_CONTEXT_OPTIONS]) ==
        POINTER_SUBTYPE) {
      contextOptions =
          sqlite3_value_pointer(columnValues[LEMBED_MODELS_CONTEXT_OPTIONS],
                                POINTER_NAME_CONTEXT_OPTIONS);
    }

    struct llama_model *model;
    struct llama_model_params mparams = llama_model_default_params();
    if (modelOptions && modelOptions->defined[0]) {
      mparams.n_gpu_layers = modelOptions->n_gpu_layers;
    }

    model = llama_load_model_from_file(modelPath, mparams);
    if (!model) {
      return SQLITE_ERROR;
    }

    struct llama_context *ctx;
    struct llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = 1;
    //cparams.n_ubatch = cparams.n_batch = 4096;
    if (contextOptions) {
      if (contextOptions->defined[0]) {
        cparams.seed = contextOptions->seed;
      }
      if (contextOptions->defined[1]) {
        cparams.n_ctx = contextOptions->n_ctx;
      }
      if (contextOptions->defined[2]) {
        cparams.rope_scaling_type = contextOptions->rope_scaling_type;
      }
      if (contextOptions->defined[3]) {
        cparams.rope_freq_scale = contextOptions->rope_freq_scale;
      }
    }

    ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
      llama_free_model(model);
      return SQLITE_ERROR;
    }
    p->api->models[idx].model = model;
    p->api->models[idx].context = ctx;
    return SQLITE_OK;
  }
  // UPDATE operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    if ((sqlite3_value_type(argv[0]) == SQLITE_INTEGER) &&
        (sqlite3_value_type(argv[1]) == SQLITE_INTEGER) &&
        (sqlite3_value_int64(argv[0]) == sqlite3_value_int64(argv[1]))) {
      return SQLITE_ERROR;
    }

    return SQLITE_ERROR;
  }
  return SQLITE_ERROR;
}

static int lembed_modelsOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  lembed_models_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int lembed_modelsClose(sqlite3_vtab_cursor *cur) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int lembed_modelsBestIndex(sqlite3_vtab *pVTab,
                                  sqlite3_index_info *pIdxInfo) {
  pIdxInfo->idxNum = 1;
  pIdxInfo->estimatedCost = (double)10;
  pIdxInfo->estimatedRows = 10;
  return SQLITE_OK;
}

static int lembed_modelsNext(sqlite3_vtab_cursor *cur);
static int lembed_modelsFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                               const char *idxStr, int argc,
                               sqlite3_value **argv) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)pVtabCursor;
  struct Api *api = ((lembed_models_vtab *)pVtabCursor->pVtab)->api;
  pCur->iRowid = -1;
  lembed_modelsNext(pVtabCursor);
  return SQLITE_OK;
}

static int lembed_modelsRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}

static int lembed_modelsNext(sqlite3_vtab_cursor *cur) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  lembed_models_vtab *p = (lembed_models_vtab *)pCur->base.pVtab;
  pCur->iRowid++;
  while (pCur->iRowid < MAX_MODELS) {
    if (p->api->models[pCur->iRowid].name) {
      return SQLITE_OK;
    }
    pCur->iRowid++;
  }
  return SQLITE_OK;
}

static int lembed_modelsEof(sqlite3_vtab_cursor *cur) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  return pCur->iRowid >= MAX_MODELS;
}

static int lembed_modelsColumn(sqlite3_vtab_cursor *cur,
                               sqlite3_context *context, int i) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  lembed_models_vtab *p = (lembed_models_vtab *)cur->pVtab;
  switch (i) {
  case LEMBED_MODELS_NAME:
    sqlite3_result_text(context, p->api->models[pCur->iRowid].name, -1,
                        SQLITE_TRANSIENT);
    break;
  case LEMBED_MODELS_SIZE:
    sqlite3_result_int64(context, llama_model_size(p->api->models[pCur->iRowid].model));
    break;
  case LEMBED_MODELS_DIMENSIONS:
    sqlite3_result_int64(context, llama_n_embd(p->api->models[pCur->iRowid].model));
    break;
  case LEMBED_MODELS_N_CTX:
    sqlite3_result_int64(context, llama_n_ctx(p->api->models[pCur->iRowid].context));
    break;
  case LEMBED_MODELS_POOLING_TYPE: {
      switch(llama_pooling_type(p->api->models[pCur->iRowid].context)) {
        case LLAMA_POOLING_TYPE_NONE: {
          sqlite3_result_text(context, "none", -1, SQLITE_STATIC);
          break;
        }
        case LLAMA_POOLING_TYPE_MEAN: {
          sqlite3_result_text(context, "mean", -1, SQLITE_STATIC);
          break;
        }
        case LLAMA_POOLING_TYPE_CLS: {
          sqlite3_result_text(context, "cls", -1, SQLITE_STATIC);
          break;
        }
        case LLAMA_POOLING_TYPE_UNSPECIFIED: {
          sqlite3_result_text(context, "unspecified", -1, SQLITE_STATIC);
          break;
        }
      }
      break;
  }

  case LEMBED_MODELS_MODEL:
    sqlite3_result_pointer(context, p->api->models[pCur->iRowid].model,
                           POINTER_NAME_MODEL, NULL);
    break;
  }
  return SQLITE_OK;
}

static sqlite3_module lembed_modelsModule = {
    /* iVersion    */ 3,
    /* xCreate     */ 0,
    /* xConnect    */ lembed_modelsConnect,
    /* xBestIndex  */ lembed_modelsBestIndex,
    /* xDisconnect */ lembed_modelsDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ lembed_modelsOpen,
    /* xClose      */ lembed_modelsClose,
    /* xFilter     */ lembed_modelsFilter,
    /* xNext       */ lembed_modelsNext,
    /* xEof        */ lembed_modelsEof,
    /* xColumn     */ lembed_modelsColumn,
    /* xRowid      */ lembed_modelsRowid,
    /* xUpdate     */ lembed_modelsUpdate,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0};
#pragma endregion

#pragma region lembed_batch


struct Array {
  size_t element_size;
  size_t length;
  size_t capacity;
  void *z;
};

/**
 * @brief Initial an array with the given element size and capacity.
 *
 * @param array
 * @param element_size
 * @param init_capacity
 * @return SQLITE_OK on success, error code on failure. Only error is
 * SQLITE_NOMEM
 */
int lembed_array_init(struct Array *array, size_t element_size, size_t init_capacity) {
  int sz = element_size * init_capacity;
  void *z = sqlite3_malloc(sz);
  if (!z) {
    return SQLITE_NOMEM;
  }
  memset(z, 0, sz);

  array->element_size = element_size;
  array->length = 0;
  array->capacity = init_capacity;
  array->z = z;
  return SQLITE_OK;
}

int lembed_array_append(struct Array *array, const void *element) {
  if (array->length == array->capacity) {
    size_t new_capacity = array->capacity * 2 + 100;
    void *z = sqlite3_realloc64(array->z, array->element_size * new_capacity);
    if (z) {
      array->capacity = new_capacity;
      array->z = z;
    } else {
      return SQLITE_NOMEM;
    }
  }
  memcpy(&((unsigned char *)array->z)[array->length * array->element_size],
         element, array->element_size);
  array->length++;
  return SQLITE_OK;
}

void lembed_array_cleanup(struct Array *array) {
  if (!array)
    return;
  array->element_size = 0;
  array->length = 0;
  array->capacity = 0;
  sqlite3_free(array->z);
  array->z = NULL;
}

typedef struct lembed_batch_vtab lembed_batch_vtab;
struct lembed_batch_vtab {
  sqlite3_vtab base;
  sqlite3 * db;
  struct Api * api;
};

typedef struct lembed_batch_cursor lembed_batch_cursor;
struct lembed_batch_cursor {
  sqlite3_vtab_cursor base;
  struct Api * api;
  struct llama_context *lctx;
  sqlite3_int64 iRowid;
  sqlite3_stmt * stmt;
  int dimensions;
  int eof;
  int stmtRc;


  int batchIdx;
  int batchSize;
  struct Array contentsArray;
  struct Array contentLengthsArray;
  float * embeddings;
};


static int lembed_batchConnect(
  sqlite3 *db,
  void *pAux,
  int argc, const char *const*argv,
  sqlite3_vtab **ppVtab,
  char **pzErr
){
  lembed_batch_vtab *pNew;
  int rc;

  rc = sqlite3_declare_vtab(db,
           "CREATE TABLE x(contents,embedding, model hidden, input hidden)"
       );
#define LEMBED_BATCH_CONTENTS  0
#define LEMBED_BATCH_EMBEDDING 1
#define LEMBED_BATCH_MODEL     2
#define LEMBED_BATCH_INPUT     3
  if( rc==SQLITE_OK ){
    pNew = sqlite3_malloc( sizeof(*pNew) );
    *ppVtab = (sqlite3_vtab*)pNew;
    if( pNew==0 ) return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
  }
  rc = sqlite3_open(":memory:", &pNew->db);
  pNew->api = pAux;
  return rc;
}

static int lembed_batchDisconnect(sqlite3_vtab *pVtab){
  lembed_batch_vtab *p = (lembed_batch_vtab*)pVtab;
  sqlite3_close(p->db);
  sqlite3_free(p);
  return SQLITE_OK;
}

static int lembed_batchOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor){
  lembed_batch_cursor *pCur;
  pCur = sqlite3_malloc( sizeof(*pCur) );
  if( pCur==0 ) return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  pCur->api = ( (lembed_batch_vtab *) p)->api;
  int rc = sqlite3_prepare_v2(
    ( (lembed_batch_vtab *) p)->db,
    "select json_extract(value, '$.contents') from json_each(?)",
    -1,
    &pCur->stmt,
    NULL
  );
  assert(rc == SQLITE_OK);
  return rc;
}

static int lembed_batchClose(sqlite3_vtab_cursor *cur){
  lembed_batch_cursor *pCur = (lembed_batch_cursor*)cur;
  sqlite3_finalize(pCur->stmt);
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int lembed_batchBestIndex(
  sqlite3_vtab *pVTab,
  sqlite3_index_info *pIdxInfo
){
  int hasSource = 0;

  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *pCons = &pIdxInfo->aConstraint[i];
    switch (pCons->iColumn) {
    case LEMBED_BATCH_MODEL: {
      if (!hasSource && !pCons->usable ||
          pCons->op != SQLITE_INDEX_CONSTRAINT_EQ)
        return SQLITE_CONSTRAINT;
      hasSource = 1;
      pIdxInfo->aConstraintUsage[i].argvIndex = 1;
      pIdxInfo->aConstraintUsage[i].omit = 1;
      break;
    }
    }
  }
  if (!hasSource) {
    pVTab->zErrMsg = sqlite3_mprintf("source argument is required");
    return SQLITE_ERROR;
  }

  pIdxInfo->estimatedCost = (double)10;
  pIdxInfo->estimatedRows = 10;
  return SQLITE_OK;
}

// SQLITE_ROW: embed some, stmt has more
// SQLITE_DONE: done after this chunk
// else: error
int embed_batch(
  lembed_batch_cursor *pCur
  ) {
  uint32_t n_batch = llama_n_ctx(pCur->lctx);
  struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
  int nprocessed = 0;
  int rc;

  while(1) {
    if(pCur->stmtRc == SQLITE_DONE) {
      pCur->eof = 1;
      break;
    }
    assert(pCur->stmtRc == SQLITE_ROW);

    char * s = (char *) sqlite3_column_text(pCur->stmt, 0);
    int len = sqlite3_column_bytes(pCur->stmt, 0);

    int input_token_count_estimate = llama_tokenize(llama_get_model(pCur->lctx), s, len, NULL, 0, true, true);
    assert(input_token_count_estimate < 0);
    llama_token *tokens = sqlite3_malloc(sizeof(llama_token) * abs(input_token_count_estimate));
    assert(tokens);

    int input_token_count = llama_tokenize(llama_get_model(pCur->lctx), s, len, tokens, abs(input_token_count_estimate), true, true);
    assert(input_token_count == abs(input_token_count_estimate));

    if (batch.n_tokens + input_token_count > n_batch) {
      assert(nprocessed>0);
      sqlite3_free(tokens);
      break;
    }

    for (size_t i = 0; i < input_token_count; i++) {
        batch.token   [batch.n_tokens] = tokens[i];
        batch.pos     [batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = nprocessed;
        batch.logits  [batch.n_tokens] = i == (input_token_count - 1);
        batch.n_tokens++;
    }
    sqlite3_free(tokens);
    nprocessed += 1;
    char * zCopy = sqlite3_mprintf("%.*s", len, s);
    assert(zCopy);
    lembed_array_append(&pCur->contentsArray, &zCopy) == SQLITE_OK;//assert();
    lembed_array_append(&pCur->contentLengthsArray, &len) == SQLITE_OK;//assert();
    pCur->stmtRc = sqlite3_step(pCur->stmt);
  }
  if(nprocessed==0) {
    pCur->batchSize = 0;
    pCur->batchIdx = 0;
    return SQLITE_DONE;
  }
  printf("nprocessed=%d\n", nprocessed);

  float * embeddings = sqlite3_malloc(pCur->dimensions * sizeof(float) * nprocessed);
  assert(embeddings);
  memset(embeddings, 0, pCur->dimensions * sizeof(float) * nprocessed);

  llama_kv_cache_clear(pCur->lctx);
  rc = llama_decode(pCur->lctx, batch);
  assert(rc >= 0 );
  for (int i = 0; i < batch.n_tokens; i++) {
    if (!batch.logits[i]) {
        continue;
    }

    float * embd = llama_get_embeddings_seq(pCur->lctx, batch.seq_id[i][0]);
    assert(embd);
    float * out = embeddings + batch.seq_id[i][0] * pCur->dimensions;
    normalize(embd, out, pCur->dimensions);
  }

  llama_batch_free(batch);
  pCur->embeddings = embeddings;
  pCur->batchSize = nprocessed;
  pCur->batchIdx = 0;
  return SQLITE_ROW;
}
static int lembed_batchFilter(
  sqlite3_vtab_cursor *pVtabCursor,
  int idxNum, const char *idxStr,
  int argc, sqlite3_value **argv
){
  int rc;
  lembed_batch_cursor *pCur = (lembed_batch_cursor *)pVtabCursor;
  sqlite3_reset(pCur->stmt);
  sqlite3_clear_bindings(pCur->stmt);
  sqlite3_bind_text(pCur->stmt, 1, sqlite3_value_text(argv[0]), sqlite3_value_bytes(argv[0]), SQLITE_TRANSIENT);
  pCur->stmtRc = sqlite3_step(pCur->stmt);
  assert(pCur->stmtRc == SQLITE_ROW || pCur->stmtRc == SQLITE_DONE);

  struct llama_model *model;
  rc = api_model_from_name(pCur->api, "default", strlen("default"), &model, &pCur->lctx);
  if(rc != SQLITE_OK) {
    return SQLITE_ERROR;
  }
  pCur->dimensions = llama_n_embd(model);
  for(int i = 0; i < pCur->batchSize; i++) {
    sqlite3_free(((char **)pCur->contentsArray.z)[i]);
  }
  lembed_array_cleanup(&pCur->contentsArray);
  lembed_array_cleanup(&pCur->contentLengthsArray);
  if(pCur->embeddings) {
    sqlite3_free(pCur->embeddings);
    pCur->embeddings = NULL;
  }
  rc = lembed_array_init(&pCur->contentsArray, sizeof(char *), 32);
  assert(rc == SQLITE_OK);
  rc = lembed_array_init(&pCur->contentLengthsArray, sizeof(int), 32);
  assert(rc == SQLITE_OK);
  pCur->iRowid = 0;
  pCur->eof = 0;

  rc = embed_batch(pCur);
  assert(rc == SQLITE_ROW || rc == SQLITE_DONE);
  return SQLITE_OK;
}

static int lembed_batchEof(sqlite3_vtab_cursor *cur){
  lembed_batch_cursor *pCur = (lembed_batch_cursor*)cur;
  return (pCur->batchIdx >= pCur->batchSize) &&  pCur->eof;
}


static int lembed_batchNext(sqlite3_vtab_cursor *cur){
  lembed_batch_cursor *pCur = (lembed_batch_cursor*)cur;
  pCur->iRowid++;
  pCur->batchIdx++;
  if(pCur->batchIdx >= pCur->batchSize) {
    int rc;
    for(int i = 0; i < pCur->batchSize; i++) {
      sqlite3_free(((char **)pCur->contentsArray.z)[i]);
    }
    lembed_array_cleanup(&pCur->contentsArray);
    lembed_array_cleanup(&pCur->contentLengthsArray);
    if(pCur->embeddings) {
      sqlite3_free(pCur->embeddings);
      pCur->embeddings = NULL;
    }
    rc = lembed_array_init(&pCur->contentsArray, sizeof(char *), 32);
    assert(rc == SQLITE_OK);
    rc = lembed_array_init(&pCur->contentLengthsArray, sizeof(int), 32);
    assert(rc == SQLITE_OK);
    rc = embed_batch(pCur);
    assert(rc == SQLITE_ROW || rc == SQLITE_DONE);
  }
  return SQLITE_OK;
}

static int lembed_batchRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid){
  lembed_batch_cursor *pCur = (lembed_batch_cursor*)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}


static int lembed_batchColumn(
  sqlite3_vtab_cursor *cur,
  sqlite3_context *context,
  int i
){
  lembed_batch_cursor *pCur = (lembed_batch_cursor*)cur;
  switch( i ){
    case LEMBED_BATCH_CONTENTS:
      sqlite3_result_text(
        context,
        ((char **)pCur->contentsArray.z)[pCur->batchIdx],
        ((int *) pCur->contentLengthsArray.z)[pCur->batchIdx],
        SQLITE_TRANSIENT
      );
      break;
    case LEMBED_BATCH_EMBEDDING:
      sqlite3_result_blob(
        context,
        pCur->embeddings + (pCur->dimensions * pCur->batchIdx),
        sizeof(float) * pCur->dimensions,
        SQLITE_TRANSIENT
      );
      sqlite3_result_subtype(context, SQLITE_VEC_FLOAT32_SUBTYPE);
      break;
    default:
      sqlite3_result_null(context);
  }
  return SQLITE_OK;
}

/*
** This following structure defines all the methods for the
** virtual table.
*/
static sqlite3_module lembed_batchModule = {
  /* iVersion    */ 3,
  /* xCreate     */ 0,
  /* xConnect    */ lembed_batchConnect,
  /* xBestIndex  */ lembed_batchBestIndex,
  /* xDisconnect */ lembed_batchDisconnect,
  /* xDestroy    */ 0,
  /* xOpen       */ lembed_batchOpen,
  /* xClose      */ lembed_batchClose,
  /* xFilter     */ lembed_batchFilter,
  /* xNext       */ lembed_batchNext,
  /* xEof        */ lembed_batchEof,
  /* xColumn     */ lembed_batchColumn,
  /* xRowid      */ lembed_batchRowid,
  /* xUpdate     */ 0,
  /* xBegin      */ 0,
  /* xSync       */ 0,
  /* xCommit     */ 0,
  /* xRollback   */ 0,
  /* xFindMethod */ 0,
  /* xRename     */ 0,
  /* xSavepoint  */ 0,
  /* xRelease    */ 0,
  /* xRollbackTo */ 0,
  /* xShadowName */ 0,
  /* xIntegrity  */ 0
};
#pragma endregion

#ifndef SQLITE_SUBTYPE
#define SQLITE_SUBTYPE 0x000100000
#endif

#ifndef SQLITE_RESULT_SUBTYPE
#define SQLITE_RESULT_SUBTYPE 0x001000000
#endif

#define SQLITE_LEMBED_DEBUG_STRING                                                \
  "Version: " SQLITE_LEMBED_VERSION "\n"                                          \
  "Date: " SQLITE_LEMBED_DATE "\n"                                                \
  "Commit: " SQLITE_LEMBED_SOURCE "\n"                                            \


#define DEFAULT_FLAGS (SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC)

#ifdef _WIN32
__declspec(dllexport)
#endif
    int sqlite3_lembed_init(sqlite3 *db, char **pzErrMsg,
                            const sqlite3_api_routines *pApi) {
  //SQLITE_EXTENSION_INIT2(pApi);

  llama_backend_init();
  llama_log_set(dummy_log, NULL);

  struct Api *a = sqlite3_malloc(sizeof(struct Api));
  if(!a) {
    return SQLITE_NOMEM;
  }
  memset(a, 0, sizeof(*a));

  int rc = SQLITE_OK;

  static const struct {
    char *zFName;
    void (*xFunc)(sqlite3_context *, int, sqlite3_value **);
    int nArg;
    int flags;
    void *p;
  } aFunc[] = {
      // clang-format off
    {"lembed_version", _static_text_func, 0, DEFAULT_FLAGS,  SQLITE_LEMBED_VERSION },
    {"lembed_debug",   _static_text_func, 0, DEFAULT_FLAGS,  SQLITE_LEMBED_DEBUG_STRING }
    // clang-format on
  };

  for (unsigned long i = 0;i < sizeof(aFunc) / sizeof(aFunc[0]) && rc == SQLITE_OK; i++) {
    rc = sqlite3_create_function_v2(db, aFunc[i].zFName, aFunc[i].nArg, aFunc[i].flags, aFunc[i].p, aFunc[i].xFunc, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
      *pzErrMsg = sqlite3_mprintf("Error creating function %s: %s",
                                  aFunc[i].zFName, sqlite3_errmsg(db));
      return rc;
    }
  }

  static const struct {
    char *zFName;
    void (*xFunc)(sqlite3_context *, int, sqlite3_value **);
    int nArg;
  } aFuncApi[] = {
      // clang-format off
    {"lembed",                 lembed,                    1},
    {"lembed",                 lembed,                    2},
    {"lembed_tokenize_json",   lembed_tokenize_json,      1},
    {"lembed_tokenize_json",   lembed_tokenize_json,      2},
    {"lembed_token_score",     lembed_token_score,        2},
    {"lembed_token_to_piece",  lembed_token_to_piece_,    2},
    {"lembed_model_from_file", lembed_model_from_file,    1},
    {"lembed_model_options",   lembed_model_options_,     -1},
    {"lembed_context_options", lembed_context_options_,   -1},
    // clang-format on
  };
  for (unsigned long i = 0;i < sizeof(aFuncApi) / sizeof(aFuncApi[0]) && rc == SQLITE_OK; i++) {
    rc = sqlite3_create_function_v2(db, aFuncApi[i].zFName, aFuncApi[i].nArg, DEFAULT_FLAGS, a, aFuncApi[i].xFunc, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
      *pzErrMsg = sqlite3_mprintf("Error creating function %s: %s",
                                  aFuncApi[i].zFName, sqlite3_errmsg(db));
      return rc;
    }
  }

  sqlite3_create_function_v2(db, "_lembed_api", 0, 0, a, _noop, NULL, NULL, api_free);

  sqlite3_create_module_v2(db, "lembed_models", &lembed_modelsModule, a, NULL);
  sqlite3_create_module_v2(db, "lembed_batch",  &lembed_batchModule,  a, NULL);
  return SQLITE_OK;
}
