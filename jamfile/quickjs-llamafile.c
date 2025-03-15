#include "third_party/quickjs/quickjs.h"
#include "third_party/quickjs/cutils.h"
#include <assert.h>
#include "llama.cpp/llama.h"
#include "jamfile/jamfile.h"
#include "jamfile/quickjs-llamafile.h"
#include "jamfile/quickjs-llamafile-completion.h"

static char * DEFAULT_EMBEDDING_MODEL;
static char * DEFAULT_COMPLETION_MODEL1;
static JSClassID jama_text_embedding_model_class_id;

static void js_text_embedding_model_finalizer(JSRuntime *rt, JSValue val) {
  struct llama_context *context = JS_GetOpaque(val, jama_text_embedding_model_class_id);
  struct llama_model *model = llama_get_model(context);
  if(context) {
    llama_free(context);
  }
  if(model) {
    llama_free_model(model);
  }
}

static JSClassDef jama_text_embedding_model_class = {
    "EmbeddingFile",
    .finalizer = js_text_embedding_model_finalizer,
};

int _llama_tokenize(struct llama_model *model, const char *input, size_t input_length, int *token_count, llama_token **tokens) {
  int input_token_count_estimate =
      llama_tokenize(model, input, input_length, NULL, 0, true, true);
  if (input_token_count_estimate >= 0) {
    return 1;
  }
  *tokens =
      malloc(sizeof(llama_token) * abs(input_token_count_estimate));
  if (!(*tokens)) {
    return 1;
  }
  int input_token_count =
      llama_tokenize(model, input, input_length, *tokens,
                     abs(input_token_count_estimate), true, true);
  if (input_token_count != abs(input_token_count_estimate)) {
    free(*tokens);
    return 1;
  }

  *token_count = input_token_count;
  return 0;
}

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

static JSValue embedding_model_embed(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  struct llama_context * context = (struct llama_context *) JS_GetOpaque2(ctx, this_val, jama_text_embedding_model_class_id);
  const char * input;
  size_t inputLength;
  JSValue rv;

  input = JS_ToCStringLen(ctx, &inputLength, argv[0]);


  struct llama_model * model = (struct llama_model *) llama_get_model(context);

  int n_ctx_train = llama_n_ctx_train(model);
  int n_ctx = llama_n_ctx(context);

  int dimensions = llama_n_embd(model);
  float *output_embedding = NULL;

  output_embedding = malloc(sizeof(float) * dimensions);
  if(!output_embedding) {
    rv = JS_EXCEPTION;
    goto done;
  }

  llama_token *tokens;
  int token_count;
  int rc = _llama_tokenize(model, input, inputLength, &token_count, &tokens);
  if(rc) {
    rv = JS_ThrowPlainError(ctx, "Could not tokenize input");
    goto done;
  }

  if(token_count > n_ctx_train) {
    rv = JS_ThrowPlainError(ctx, "Input token count larger than context size, %d tokens vs %d context size", token_count, n_ctx_train);
    goto done;
  }

  struct llama_batch batch = llama_batch_init(n_ctx_train, 0, 1);

  int seq_id = 0;
  for (int i = 0; i < token_count; i++) {
    batch.token[batch.n_tokens] = tokens[i];
    batch.pos[batch.n_tokens] = i;

    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = seq_id;

    batch.logits[batch.n_tokens] = i == (token_count - 1);
    batch.n_tokens++;
  }

  llama_kv_cache_clear(context);
  rc = llama_decode(context, batch);
  if(rc != 0) {
    free(output_embedding);
    llama_batch_free(batch);
    rv = JS_EXCEPTION;
    goto done;
  }

  float * source_embedding;
  if(llama_pooling_type(context) == LLAMA_POOLING_TYPE_NONE) {
    source_embedding = llama_get_embeddings(context);
  }
  else {
    source_embedding = llama_get_embeddings_seq(context, batch.seq_id[0][0]);
  }
  if(!source_embedding) {
    free(output_embedding);
    llama_batch_free(batch);
    rv = JS_EXCEPTION;
    goto done;
  }

  normalize(source_embedding, output_embedding, dimensions);
  llama_batch_free(batch);

  JSValue embeddingArrayBuffer = JS_NewArrayBufferCopy(ctx, (const uint8_t*) output_embedding, sizeof(float) * dimensions);
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue Float32Array = JS_GetPropertyStr(ctx, global, "Float32Array");
  rv = JS_CallConstructor(ctx, Float32Array, 1, &embeddingArrayBuffer);
  JS_FreeValue(ctx, embeddingArrayBuffer);
  JS_FreeValue(ctx, Float32Array);
  JS_FreeValue(ctx, global);




  done:
  JS_FreeCString(ctx, input);
  free(output_embedding);
  free(tokens);
  return rv;
}

int _instance_of_int32_array(JSContext *ctx, JSValue v) {
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue Int32Array = JS_GetPropertyStr(ctx, global, "Int32Array");
  int ret = JS_IsInstanceOf(ctx, v, Int32Array);
  JS_FreeValue(ctx, Int32Array);
  JS_FreeValue(ctx, global);
  return ret;
}

static JSValue embedding_model_tokenize(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  JSValue rv;
  struct llama_context * context = (struct llama_context *) JS_GetOpaque2(ctx, this_val, jama_text_embedding_model_class_id);
  struct llama_model * model = (struct llama_model *) llama_get_model(context);

  llama_token *tokens = NULL;
  int token_count;

  if(JS_IsString(argv[0])) {
    const char * input;
    size_t inputLength;

    input = JS_ToCStringLen(ctx, &inputLength, argv[0]);
    int rc = _llama_tokenize(model, input, inputLength, &token_count, &tokens);
    JS_FreeCString(ctx, input);

    if(rc) {
      rv = JS_EXCEPTION;
      goto done;
    }
  }
  else if(_instance_of_int32_array(ctx, argv[0])) {
    size_t unused;
    JSValue vBuf = JS_GetTypedArrayBuffer(ctx, argv[0], &unused, &unused, &unused);
    assert(!JS_IsException(vBuf));
    size_t bufLen;
    uint8_t * buf = JS_GetArrayBuffer(ctx, &bufLen, vBuf);
    JS_FreeValue(ctx, vBuf);
    tokens = buf;
  }
  else {
    rv = JS_ThrowPlainError(ctx, "Unknown input type to tokenize()");
    goto done;
  }



  JSValue tokensArrayBuffer = JS_NewArrayBufferCopy(ctx, (const uint8_t*) tokens, sizeof(llama_token) * token_count);
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue Int32Array = JS_GetPropertyStr(ctx, global, "Int32Array");
  rv = JS_CallConstructor(ctx, Int32Array, 1, &tokensArrayBuffer);
  JS_FreeValue(ctx, tokensArrayBuffer);
  JS_FreeValue(ctx, Int32Array);
  JS_FreeValue(ctx, global);

  done:
  free(tokens);
  return rv;
}

static JSValue embedding_model_detokenize(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  JSValue rv;
  struct llama_context * context = (struct llama_context *) JS_GetOpaque2(ctx, this_val, jama_text_embedding_model_class_id);
  struct llama_model * model = (struct llama_model *) llama_get_model(context);

  if(!_instance_of_int32_array(ctx, argv[0])) {
    return JS_EXCEPTION;
  }

  size_t unused;
  JSValue vBuf = JS_GetTypedArrayBuffer(ctx, argv[0], &unused, &unused, &unused);
  assert(!JS_IsException(vBuf));
  size_t bufLen;
  llama_token * tokens = JS_GetArrayBuffer(ctx, &bufLen, vBuf);
  JS_FreeValue(ctx, vBuf);


  int token_count = bufLen / sizeof(llama_token);

  int32_t needed = llama_detokenize(model, tokens, token_count, NULL, 0, 1, 1);
  assert(needed < 0);
  char * sBuf = malloc(abs(needed));
  int x = llama_detokenize(model, tokens, token_count, sBuf, abs(needed), 1, 1);
  //assert(x == 0);

  rv = JS_NewStringLen(ctx, sBuf, abs(needed));

  done:
  return rv;
}

static JSValue embedding_model_dimensions_getter(JSContext *ctx, JSValue this_val) {
  struct llama_context * context = (struct llama_context *) JS_GetOpaque2(ctx, this_val, jama_text_embedding_model_class_id);
  struct llama_model * model = (struct llama_model *) llama_get_model(context);
  return JS_NewInt32(ctx, llama_n_embd(model));
}

static const JSCFunctionListEntry js_embedding_model_proto_funcs[] = {
  JS_CFUNC_DEF("embed", 1, embedding_model_embed ),
  JS_CFUNC_DEF("tokenize", 1, embedding_model_tokenize ),
  JS_CFUNC_DEF("detokenize", 1, embedding_model_detokenize ),
  JS_CGETSET_DEF("dimensions", embedding_model_dimensions_getter, NULL)
};


static JSValue text_embedding_model_constructor(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  JSValue rv = JS_UNDEFINED;
  struct llama_model *model;
  struct llama_model_params mparams = llama_model_default_params();



  if(JS_IsString(argv[0])) {
    const char * modelPath = NULL;
    size_t modelPathLength;
    modelPath = JS_ToCStringLen(ctx, &modelPathLength, argv[0]);
    model = llama_load_model_from_file(modelPath, mparams);
    JS_FreeCString(ctx, modelPath);
  }
  else if(DEFAULT_EMBEDDING_MODEL) {
    model = llama_load_model_from_file(DEFAULT_EMBEDDING_MODEL, mparams);
  }


  if (!model) {
    return JS_ThrowPlainError(ctx, "Could not load model file");
  }

  struct llama_context * context;
  struct llama_context_params cparams = llama_context_default_params();
  cparams.embeddings = 1;
  cparams.n_ctx = llama_n_ctx_train(model);
  cparams.n_ubatch = cparams.n_batch = cparams.n_ctx; // ?
  context = llama_new_context_with_model(model, cparams);
  if (!context) {
    llama_free_model(model);
    return JS_ThrowPlainError(ctx, "Could not create context from model");
  }

  JSValue proto = JS_GetClassProto(ctx, jama_text_embedding_model_class_id);
  JSValue obj = JS_NewObjectProtoClass(ctx, proto, jama_text_embedding_model_class_id);
  JS_FreeValue(ctx, proto);

  JS_SetOpaque(obj, context);
  return obj;

}


static int js_llamafile_init(JSContext *ctx, JSModuleDef *m)
{
    JSRuntime *rt = JS_GetRuntime(ctx);

    JS_NewClassID(JS_GetRuntime(ctx), &jama_text_embedding_model_class_id);
    JS_NewClass(rt, jama_text_embedding_model_class_id, &jama_text_embedding_model_class);
    JSValue proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, proto, js_embedding_model_proto_funcs, countof(js_embedding_model_proto_funcs));

    JSValue obj = JS_NewCFunction2(ctx, text_embedding_model_constructor, "TextEmbeddingModel", 1, JS_CFUNC_constructor, 0);
    JS_SetConstructor(ctx, obj, proto);
    JS_SetClassProto(ctx, jama_text_embedding_model_class_id, proto);
    JS_SetModuleExport(ctx, m, "TextEmbeddingModel", obj);

    return js_llamafile_init_completion_model(ctx, m, DEFAULT_COMPLETION_MODEL1);
}

JSModuleDef *js_init_module_llamafile(JSContext *ctx, const char *module_name, char * default_embedding, char * default_completion)

{
    if(default_embedding) {
      DEFAULT_EMBEDDING_MODEL = default_embedding;
    }
    if(default_completion) {
      DEFAULT_COMPLETION_MODEL1 = default_completion;
    }
    JSModuleDef *m;
    m = JS_NewCModule(ctx, module_name, js_llamafile_init);
    if (!m)
        return NULL;
    JS_AddModuleExport(ctx, m, "TextEmbeddingModel");
    JS_AddModuleExport(ctx, m, "CompletionModel");
    return m;
}
