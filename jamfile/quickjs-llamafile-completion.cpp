#include "third_party/quickjs/quickjs.h"
//#include "third_party/quickjs/cutils.h"
#include <assert.h>
#include "llama.cpp/llama.h"
#include "llama.cpp/common.h"
#include "llama.cpp/sampling.h"

static char * DEFAULT_COMPLETION_MODEL;
static JSClassID jama_completionmodel_class_id;

static void js_completionmodel_finalizer(JSRuntime *rt, JSValue val) {
  struct llama_context *context = (struct llama_context *) JS_GetOpaque(val, jama_completionmodel_class_id);
  if(context) {
    llama_free(context);
  }
}

static JSClassDef jama_completionmodel_class = {
    "CompletionModel",
    .finalizer = js_completionmodel_finalizer,
};



int _llama_tokenizex(struct llama_model *model, const char *input, size_t input_length, int *token_count, llama_token **tokens) {
  int input_token_count_estimate =
      llama_tokenize(model, input, input_length, NULL, 0, true, true);
  if (input_token_count_estimate >= 0) {
    return 1;
  }
  *tokens =
      (llama_token *) malloc(sizeof(llama_token) * abs(input_token_count_estimate));
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

static bool eval_tokens(struct llama_context *ctx_llama, std::vector<llama_token> tokens,
                        int n_batch, int *n_past) {
    int N = (int)tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch)
            n_eval = n_batch;
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0)))
            return false; // probably ran out of context
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context *ctx_llama, int id, int *n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context *ctx_llama, const char *str, int n_batch, int *n_past,
                        bool add_bos) {
    std::string str2 = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos);
    return eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
}

static JSValue completion_model_complete(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  struct llama_context * context = (struct llama_context *) JS_GetOpaque2(ctx, this_val, jama_completionmodel_class_id);
  JSValue parse = JS_UNDEFINED;
  const char * input;
  size_t inputLength;
  JSValue rv;

  input = JS_ToCStringLen(ctx, &inputLength, argv[0]);
  struct llama_model * model = (struct llama_model *) llama_get_model(context);
  int token_count;

  struct llama_sampling_context *ctx_sampling;

  int n_past = 0;
  gpt_params params;
  params.n_ctx = 0;
  bool add_bos;

  //int rc = gpt_params_parse(0, NULL, params);

  struct llama_sampling_params x;// = params.sparams;
    std::string result = "";


  if(!JS_IsUndefined(argv[1])) {
    JSValue schema = JS_GetPropertyStr(ctx, argv[1], "schema");
    JSValue temperature = JS_GetPropertyStr(ctx, argv[1], "temperature");
    parse = JS_GetPropertyStr(ctx, argv[1], "parse");
    
    if(!JS_IsUndefined(schema)) {
      if(JS_IsObject(schema)) {
        JSValue stringified = JS_JSONStringify(ctx, schema, JS_UNDEFINED, JS_UNDEFINED);
        assert(!JS_IsException(stringified));
        size_t sLen;
        const char * s = JS_ToCStringLen(ctx, &sLen, stringified);
        x.grammar = json_schema_string_to_grammar(s);
        JS_FreeValue(ctx, stringified);
      }
      else {
        const char * g = JS_ToCString(ctx, schema);
        x.grammar = std::string(g);
        JS_FreeCString(ctx, g);
      }
      
    }

    if(!JS_IsUndefined(temperature)) {
      double temperatureValue;
      int rc = JS_ToFloat64(ctx, &temperatureValue, temperature);
      assert(!rc);
      x.temp = temperatureValue;
    }
    JS_FreeValue(ctx, schema);
    JS_FreeValue(ctx, temperature);
  }
  llama_set_rng_seed(context, 0xdd);


  add_bos = llama_should_add_bos_token(model);
  eval_string(context, input, 2048, &n_past, add_bos);

  ctx_sampling = llama_sampling_init(x);


  for (;;) {
      llama_token id = llama_sampling_sample(ctx_sampling, context, NULL);
      llama_sampling_accept(ctx_sampling, context, id, true);
      if (llama_token_is_eog(model, id))
          break;
      result += llama_token_to_piece(context, id);
      if (!eval_id(context, id, &n_past))
          break;
  }

  rv = JS_NewStringLen(ctx, result.c_str(), result.size());

  llama_sampling_free(ctx_sampling);
  llama_kv_cache_clear(context);

  if(!JS_IsUndefined(parse)) {
    rv = JS_Call(ctx, parse, JS_UNDEFINED, 1, &rv);
  }

  done:
  JS_FreeCString(ctx, input);
  return rv;
}


int _instance_of_int32_arrayx(JSContext *ctx, JSValue v) {
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue Int32Array = JS_GetPropertyStr(ctx, global, "Int32Array");
  int ret = JS_IsInstanceOf(ctx, v, Int32Array);
  JS_FreeValue(ctx, Int32Array);
  JS_FreeValue(ctx, global);
  return ret;
}

static JSValue completion_model_tokenize(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  JSValue rv;
  struct llama_context * context = (struct llama_context *) JS_GetOpaque2(ctx, this_val, jama_completionmodel_class_id);
  struct llama_model * model = (struct llama_model *) llama_get_model(context);

  llama_token *tokens = NULL;
  int token_count;
  JSValue tokensArrayBuffer;
  JSValue global;
  JSValue Int32Array;

  if(JS_IsString(argv[0])) {
    const char * input;
    size_t inputLength;

    input = JS_ToCStringLen(ctx, &inputLength, argv[0]);
    int rc = _llama_tokenizex(model, input, inputLength, &token_count, &tokens);
    JS_FreeCString(ctx, input);

    if(rc) {
      rv = JS_EXCEPTION;
      goto done;
    }
  }
  else if(_instance_of_int32_arrayx(ctx, argv[0])) {
    size_t unused;
    JSValue vBuf = JS_GetTypedArrayBuffer(ctx, argv[0], &unused, &unused, &unused);
    assert(!JS_IsException(vBuf));
    size_t bufLen;
    uint8_t * buf = JS_GetArrayBuffer(ctx, &bufLen, vBuf);
    JS_FreeValue(ctx, vBuf);
    tokens = (llama_token *) buf;
  }
  else {
    rv = JS_ThrowPlainError(ctx, "Unknown input type to tokenize()");
    goto done;
  }



  tokensArrayBuffer = JS_NewArrayBufferCopy(ctx, (const uint8_t*) tokens, sizeof(llama_token) * token_count);
  global = JS_GetGlobalObject(ctx);
  Int32Array = JS_GetPropertyStr(ctx, global, "Int32Array");
  rv = JS_CallConstructor(ctx, Int32Array, 1, &tokensArrayBuffer);
  JS_FreeValue(ctx, tokensArrayBuffer);
  JS_FreeValue(ctx, Int32Array);
  JS_FreeValue(ctx, global);

  done:
  free(tokens);
  return rv;
}

static JSValue completion_model_detokenize(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  JSValue rv;
  struct llama_context * context = (struct llama_context *) JS_GetOpaque2(ctx, this_val, jama_completionmodel_class_id);
  struct llama_model * model = (struct llama_model *) llama_get_model(context);

  if(!_instance_of_int32_arrayx(ctx, argv[0])) {
    return JS_EXCEPTION;
  }

  size_t unused;
  JSValue vBuf = JS_GetTypedArrayBuffer(ctx, argv[0], &unused, &unused, &unused);
  assert(!JS_IsException(vBuf));
  size_t bufLen;
  llama_token * tokens = (llama_token *) JS_GetArrayBuffer(ctx, &bufLen, vBuf);
  JS_FreeValue(ctx, vBuf);


  int token_count = bufLen / sizeof(llama_token);

  int32_t needed = llama_detokenize(model, tokens, token_count, NULL, 0, 1, 1);
  assert(needed < 0);
  char * sBuf = (char *) malloc(abs(needed));
  int x = llama_detokenize(model, tokens, token_count, sBuf, abs(needed), 1, 1);
  //assert(x == 0);

  rv = JS_NewStringLen(ctx, sBuf, abs(needed));

  done:
  return rv;
}


static const JSCFunctionListEntry js_completionmodel_database_proto_funcs[] = {
  JS_CFUNC_DEF("complete", 2, completion_model_complete ),
  JS_CFUNC_DEF("tokenize", 1, completion_model_tokenize ),
  JS_CFUNC_DEF("detokenize", 1, completion_model_detokenize ),

};


static JSValue completionmodel_constructor(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  JSValue rv = JS_UNDEFINED;
  struct llama_model *model = NULL;
  struct llama_model_params mparams = llama_model_default_params();



  if(JS_IsString(argv[0])) {
    const char * modelPath = NULL;
    size_t modelPathLength;
    modelPath = JS_ToCStringLen(ctx, &modelPathLength, argv[0]);
    model = llama_load_model_from_file(modelPath, mparams);
    JS_FreeCString(ctx, modelPath);
  }
  else if(DEFAULT_COMPLETION_MODEL) {
    model = llama_load_model_from_file(DEFAULT_COMPLETION_MODEL, mparams);
  }

  if (!model) {
    return JS_ThrowPlainError(ctx, "Could not load model file");
  }

  struct llama_context * context;
  struct llama_context_params cparams = llama_context_default_params();
  context = llama_new_context_with_model(model, cparams);
  if (!context) {
    llama_free_model(model);
    return JS_ThrowPlainError(ctx, "Could not create context from model");
  }

  JSValue proto = JS_GetClassProto(ctx, jama_completionmodel_class_id);
  JSValue obj = JS_NewObjectProtoClass(ctx, proto, jama_completionmodel_class_id);
  JS_FreeValue(ctx, proto);

  JS_SetOpaque(obj, context);
  return obj;

}



extern "C" {

int js_llamafile_init_completion_model(JSContext *ctx, JSModuleDef *m, char * default_model)
{
    DEFAULT_COMPLETION_MODEL = default_model;
    JSRuntime *rt = JS_GetRuntime(ctx);

    JS_NewClassID(JS_GetRuntime(ctx), &jama_completionmodel_class_id);
    JS_NewClass(rt, jama_completionmodel_class_id, &jama_completionmodel_class);
    JSValue proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(
      ctx,
      proto,
      js_completionmodel_database_proto_funcs,
      (sizeof(js_completionmodel_database_proto_funcs) 
        / sizeof((js_completionmodel_database_proto_funcs)[0]))
    );

    JSValue obj = JS_NewCFunction2(ctx, completionmodel_constructor, "CompletionModel", 1, JS_CFUNC_constructor, 0);
    JS_SetConstructor(ctx, obj, proto);
    JS_SetClassProto(ctx, jama_completionmodel_class_id, proto);

    return JS_SetModuleExport(ctx, m, "CompletionModel", obj);
}

//JSModuleDef *js_init_module_completionmodel(JSContext *ctx, const char *module_name)

/*{
    JSModuleDef *m;
    m = JS_NewCModule(ctx, module_name, js_completionmodel_init);
    if (!m)
        return NULL;
    JS_AddModuleExport(ctx, m, "CompletionModel");
    return m;
}
*/
}
