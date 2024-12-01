#include "third_party/sqlite/sqlite3.h"
#include "third_party/quickjs/quickjs.h"
#include "third_party/quickjs/cutils.h"
#include "embedfile/sqlite-vec.h"
#include "embedfile/sqlite-csv.h"
#include <assert.h>


// escapeIdentifier(id): string
static JSValue js_escape_identifier(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  JSValue rv = JS_UNDEFINED;
  char * escaped = NULL;

  const char * identifier = JS_ToCString(ctx, argv[0]);

  if (!identifier)
      return JS_EXCEPTION;

  sqlite3_str * s = sqlite3_str_new(NULL);
  sqlite3_str_appendf(s, "%w", identifier);
  int len = sqlite3_str_length(s);
  escaped = sqlite3_str_finish(s);
  if(!escaped) {
    rv = JS_EXCEPTION;
    goto done;
  }
  rv = JS_NewStringLen(ctx, escaped, len);

  done:
  sqlite3_free(escaped);
  JS_FreeCString(ctx, identifier);
  return rv;
}

static const JSCFunctionListEntry js_sqlite_funcs[] = {
  JS_PROP_STRING_DEF("SQLITE_VERSION", SQLITE_VERSION, 0),
  JS_CFUNC_DEF("escapeIdentifier", 1, js_escape_identifier)

};

#define JSON_SUBTYPE 74

JSValue column_value_to_js(JSContext *ctx, sqlite3_stmt * stmt, int idx) {
  switch(sqlite3_column_type(stmt, idx)) {
    case SQLITE_INTEGER: {
      // TODO big int?
      return JS_NewInt64(ctx, sqlite3_column_int64(stmt, idx));
    }
    case SQLITE_FLOAT: {
      return JS_NewFloat64(ctx, sqlite3_column_double(stmt, idx));
    }
    case SQLITE_TEXT: {
      if(sqlite3_value_subtype(sqlite3_column_value(stmt, idx)) == JSON_SUBTYPE) {
        return JS_ParseJSON(ctx, (const char *) sqlite3_column_text(stmt, idx), sqlite3_column_bytes(stmt, idx), NULL);
      }
      return JS_NewStringLen(ctx, (const char *) sqlite3_column_text(stmt, idx), sqlite3_column_bytes(stmt, idx));
    }
    case SQLITE_BLOB: {
      sqlite3_value * v = sqlite3_column_value(stmt, idx);
      switch(sqlite3_value_subtype(v)) {

        // sqlite-vec float32 vector
        case 223: {
          JSValue rv;
          // TODO assert % 4
          JSValue embeddingArrayBuffer = JS_NewArrayBufferCopy(ctx, (const uint8_t*) sqlite3_column_blob(stmt, idx), sqlite3_column_bytes(stmt, idx));
          JSValue global = JS_GetGlobalObject(ctx);
          JSValue Float32Array = JS_GetPropertyStr(ctx, global, "Float32Array");
          rv = JS_CallConstructor(ctx, Float32Array, 1, &embeddingArrayBuffer);
          JS_FreeValue(ctx, embeddingArrayBuffer);
          JS_FreeValue(ctx, Float32Array);
          JS_FreeValue(ctx, global);
          return rv;
        }

        // sqlite-vec bit vector
        case 224: {
          JSValue rv;
          JSValue embeddingArrayBuffer = JS_NewArrayBufferCopy(ctx, (const uint8_t*) sqlite3_column_blob(stmt, idx), sqlite3_column_bytes(stmt, idx));
          JSValue global = JS_GetGlobalObject(ctx);
          JSValue Uint8Array = JS_GetPropertyStr(ctx, global, "Uint8Array");
          rv = JS_CallConstructor(ctx, Uint8Array, 1, &embeddingArrayBuffer);
          JS_FreeValue(ctx, embeddingArrayBuffer);
          JS_FreeValue(ctx, Uint8Array);
          JS_FreeValue(ctx, global);
          return rv;
        }

        // sqlite-vec int8 vector
        case 225: {
          JSValue rv;
          JSValue embeddingArrayBuffer = JS_NewArrayBufferCopy(ctx, (const uint8_t*) sqlite3_column_blob(stmt, idx), sqlite3_column_bytes(stmt, idx));
          JSValue global = JS_GetGlobalObject(ctx);
          JSValue Int8Array = JS_GetPropertyStr(ctx, global, "Int8Array");
          rv = JS_CallConstructor(ctx, Int8Array, 1, &embeddingArrayBuffer);
          JS_FreeValue(ctx, embeddingArrayBuffer);
          JS_FreeValue(ctx, Int8Array);
          JS_FreeValue(ctx, global);
          return rv;
        }

        default:
        return JS_NewUint8ArrayCopy(ctx, sqlite3_column_blob(stmt, idx), sqlite3_column_bytes(stmt, idx));
      }
    }
    case SQLITE_NULL: {
      return JS_NULL;
    }
  }
  return JS_UNDEFINED;
}
static JSClassID jama_sqlite_class_id;

static void js_sqlite_database_finalizer(JSRuntime *rt, JSValue val) {
  sqlite3 *db = JS_GetOpaque(val, jama_sqlite_class_id);
  if(db) {
    sqlite3_close(db);
  }
}

static JSClassDef jama_sqlite_class = {
    "Database",
    .finalizer = js_sqlite_database_finalizer,
};

int js_is_typed_array(JSContext * ctx, JSValue v) {
    int ret = FALSE;
    int isView;
    JSValue global = JS_GetGlobalObject(ctx);
    JSValue ArrayBuffer = JS_GetPropertyStr(ctx, global, "ArrayBuffer");
    JSValue ArrayBuffer_isView = JS_GetPropertyStr(ctx, ArrayBuffer, "isView");
    JSValue x = JS_Call(ctx, ArrayBuffer_isView, JS_UNDEFINED, 1, &v);
    if(JS_IsBool(x)) {
      isView = JS_ToBool(ctx, x);
    }
    JS_FreeValue(ctx, x);
    JS_FreeValue(ctx, ArrayBuffer_isView);
    JS_FreeValue(ctx, ArrayBuffer);

    if(!isView) {
      ret = FALSE;
      goto done;
    }

    JSValue DataView = JS_GetPropertyStr(ctx, global, "DataView");
    ret = !JS_IsInstanceOf(ctx, v, DataView);
    JS_FreeValue(ctx, DataView);

    done:
      JS_FreeValue(ctx, global);
      return ret;
}

int js_is_date(JSContext * ctx, JSValue v) {
    JSValue global = JS_GetGlobalObject(ctx);
    JSValue Date = JS_GetPropertyStr(ctx, global, "Date");
    int ret = JS_IsInstanceOf(ctx, v, Date);
    JS_FreeValue(ctx, Date);
    JS_FreeValue(ctx, global);
    return ret;
}

static void js_dump_objx(JSContext *ctx, FILE *f, JSValue val)
{
    const char *str;

    str = JS_ToCString(ctx, val);
    if (str) {
        fprintf(f, "%s\n", str);
        JS_FreeCString(ctx, str);
    } else {
        fprintf(f, "[exception]\n");
    }
}

void js_std_dump_error1x(JSContext *ctx, JSValue exception_val)
{
    JSValue val;
    BOOL is_error;

    is_error = JS_IsError(ctx, exception_val);
    js_dump_objx(ctx, stderr, exception_val);
    if (is_error) {
        val = JS_GetPropertyStr(ctx, exception_val, "stack");
        if (!JS_IsUndefined(val)) {
            js_dump_objx(ctx, stderr, val);
        }
        JS_FreeValue(ctx, val);
    }
}

JSValue _bind_js_value(JSContext *ctx, JSValue v, sqlite3_stmt * stmt, int32_t idx) {
  if(JS_IsBool(v)) {
    sqlite3_bind_int(stmt, idx, JS_ToBool(ctx, v));
    return JS_UNDEFINED;
  }

  if(JS_IsNull(v) || JS_IsUndefined(v)) {
    sqlite3_bind_null(stmt, idx);
    return JS_UNDEFINED;

  }

  if(JS_IsNumber(v)) {
    switch(JS_VALUE_GET_NORM_TAG(v)) {
      case JS_TAG_INT: {
        sqlite3_bind_int(stmt, idx, JS_VALUE_GET_INT(v));
        break;
      }
      case JS_TAG_FLOAT64: {
        sqlite3_bind_double(stmt, idx, JS_VALUE_GET_FLOAT64(v));
        break;
      }
      default: {
        return JS_ThrowPlainError(ctx, "Unknown number type");
      }
    }
    return JS_UNDEFINED;
  }

  if(JS_IsString(v)) {
    size_t sLen;
    const char * s = JS_ToCStringLen(ctx, &sLen, v);
    sqlite3_bind_text(stmt, idx, s, sLen, SQLITE_TRANSIENT);
    JS_FreeCString(ctx, s);
    return JS_UNDEFINED;
  }

  if( JS_IsUint8Array(v)) {
    size_t bufLen;
    uint8_t * buf = JS_GetUint8Array(ctx, &bufLen, v);
    if(!buf) {
      return JS_EXCEPTION;
    }
    sqlite3_bind_blob(stmt, idx, buf, bufLen, SQLITE_TRANSIENT);
    return JS_UNDEFINED;
  }

  if (js_is_typed_array(ctx, v)) {
    size_t unused;
    JSValue vBuf = JS_GetTypedArrayBuffer(ctx, v, &unused, &unused, &unused);
    assert(!JS_IsException(vBuf));
    size_t bufLen;
    uint8_t * buf = JS_GetArrayBuffer(ctx, &bufLen, vBuf);
    sqlite3_bind_blob(stmt, idx, buf, bufLen, SQLITE_TRANSIENT);
    JS_FreeValue(ctx, vBuf);
    return JS_UNDEFINED;
  }

  if (js_is_date(ctx, v)) {
    JSValue toISOString = JS_GetPropertyStr(ctx, v, "toISOString");
    assert(JS_IsFunction(ctx, toISOString ));
    JSValue ret = JS_Call(ctx, toISOString, v, 0, NULL);
    JS_FreeValue(ctx, toISOString);
    assert(JS_IsString(ret));

    size_t sLen;
    const char * s = JS_ToCStringLen(ctx, &sLen, ret);
    sqlite3_bind_text(stmt, idx, s, sLen, SQLITE_TRANSIENT);
    JS_FreeCString(ctx, s);
    return JS_UNDEFINED;
  }

  if(JS_IsObject(v) || JS_IsArray(ctx, v)) {
    JSValue stringified = JS_JSONStringify(ctx, v, JS_UNDEFINED, JS_UNDEFINED);
    if(JS_IsException(stringified)) {
      return JS_EXCEPTION;
    }
    size_t sLen;
    const char * s = JS_ToCStringLen(ctx, &sLen, stringified);

    sqlite3_bind_text(stmt, idx, s, sLen, SQLITE_TRANSIENT);
    JS_FreeCString(ctx, s);
    return JS_UNDEFINED;
  }

  // TODO when does this happen
  return JS_EXCEPTION;
}

JSValue apply_parameters(JSContext *ctx, JSValue parameters, sqlite3_stmt * stmt) {
  JSValue rv = JS_UNDEFINED;

  if(JS_IsArray(ctx, parameters)) {
    int64_t n;
    if (JS_GetLength(ctx, parameters, &n)) {
      rv = JS_EXCEPTION;
      goto done;
    }
    for(int i = 0; i < n; i++) {
      JSValue v = JS_GetPropertyInt64(ctx, parameters, i);
      if (JS_IsException(v)) {
        rv = JS_EXCEPTION;
        goto done;
      }

      JSValue rv = _bind_js_value(ctx, v, stmt, i + 1);
      JS_FreeValue(ctx, v);

      if(JS_IsException(rv)) {
        goto done;
      }

    }
  }
  else {
    rv = JS_ThrowPlainError(ctx, "Unknown parameter type");
  }

  done:
    return rv;
}
static JSValue db_query_value(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  sqlite3 * db = (sqlite3 * ) JS_GetOpaque2(ctx, this_val, jama_sqlite_class_id);
  sqlite3_stmt * stmt = NULL;
  size_t sqlLength;
  const char * sql;
  JSValue rv;

  sql = JS_ToCStringLen(ctx, &sqlLength, argv[0]);

  int rc = sqlite3_prepare_v2(db, sql, sqlLength, &stmt, NULL);
  if(rc != SQLITE_OK) {
    rv = JS_EXCEPTION;
    goto done;
  }
  if(!(sqlite3_stmt_readonly(stmt))) {
    rv =  JS_ThrowPlainError(ctx, "SQL in queryValue() must be a read-only script, ie SELECT only. ");
    goto done;
  }

  if(!JS_IsUndefined(argv[1])) {
    rv = apply_parameters(ctx, argv[1], stmt);
    if(JS_IsException(rv)) {
      goto done;
    }
  }

  rc = sqlite3_step(stmt);
  if(rc != SQLITE_ROW) {
    rv = JS_EXCEPTION;
    goto done;
  }

  rv = column_value_to_js(ctx, stmt, 0);

  done:
  JS_FreeCString(ctx, sql);
  sqlite3_finalize(stmt);
  return rv;
}

static JSValue db_query_row(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  sqlite3 * db = (sqlite3 * ) JS_GetOpaque2(ctx, this_val, jama_sqlite_class_id);
  sqlite3_stmt * stmt = NULL;
  size_t sqlLength;
  const char * sql;
  JSValue rv;

  sql = JS_ToCStringLen(ctx, &sqlLength, argv[0]);

  int rc = sqlite3_prepare_v2(db, sql, sqlLength, &stmt, NULL);
  if(rc != SQLITE_OK) {
    rv = JS_EXCEPTION;
    goto done;
  }
  if(!(sqlite3_stmt_readonly(stmt))) {
    rv =  JS_ThrowPlainError(ctx, "SQL in queryRow() must be a read-only script, ie SELECT only. ");
    goto done;
  }

  rc = sqlite3_step(stmt);
  if(rc != SQLITE_ROW) {
    rv = JS_EXCEPTION;
    goto done;
  }

  JSValue row = JS_NewObject(ctx);

  for(int i = 0; i < sqlite3_column_count(stmt); i++) {
    JSValue v = column_value_to_js(ctx, stmt, i);
    JS_DefinePropertyValueStr(ctx, row, sqlite3_column_name(stmt, i), v, JS_PROP_C_W_E);
  }
  rv = row;

  done:
  JS_FreeCString(ctx, sql);
  sqlite3_finalize(stmt);
  return rv;
}

static JSValue  db_execute(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  sqlite3 * db = (sqlite3 * ) JS_GetOpaque2(ctx, this_val, jama_sqlite_class_id);
  sqlite3_stmt * stmt = NULL;
  size_t sqlLength;
  const char * sql;
  JSValue rv;

  sql = JS_ToCStringLen(ctx, &sqlLength, argv[0]);

  // TODO Can it handle comments?
  int rc = sqlite3_prepare_v2(db, sql, sqlLength, &stmt, NULL);
  if(rc != SQLITE_OK) {
    rv = JS_ThrowPlainError(ctx, "error preparing statement: %s", sqlite3_errmsg(db));
    goto done;
  }

  if(!JS_IsUndefined(argv[1])) {
    rv = apply_parameters(ctx, argv[1], stmt);
    if(JS_IsException(rv)) {
      goto done;
    }
  }

  while(1) {
    rc = sqlite3_step(stmt);
    if(rc == SQLITE_DONE) {
      break;
    }
    if(rc != SQLITE_ROW) {
      rv = JS_ThrowPlainError(ctx, "Error executing statement: %s", sqlite3_errmsg(db));
      goto done;
    }
  }

  rv = JS_TRUE;

  done:
  JS_FreeCString(ctx, sql);
  sqlite3_finalize(stmt);
  return rv;
}

static JSValue db_execute_script(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  sqlite3 * db = (sqlite3 * ) JS_GetOpaque2(ctx, this_val, jama_sqlite_class_id);
  sqlite3_stmt * stmt = NULL;
  const char * extra;

  size_t sqlLength;
  const char * sql;
  JSValue rv;

  sql = JS_ToCStringLen(ctx, &sqlLength, argv[0]);
  const char * curr = sql;

  // TODO is the extra - curr math wrong? Could this overflow? Can it handle comments?
  while(1) {
    stmt = NULL;
    int rc = sqlite3_prepare_v2(db, curr, sqlLength, &stmt, &extra);
    if(rc != SQLITE_OK) {
      rv = JS_ThrowPlainError(ctx, "Error preparing SQL statement");
      goto done;
    }
    if(!stmt) {
      break;
    }

    while(1) {
      rc = sqlite3_step(stmt);
      if(rc == SQLITE_DONE) {
        break;
      }
      if(rc != SQLITE_ROW) {
        rv = JS_EXCEPTION;
        goto done;
      }
    }
    sqlite3_finalize(stmt);
    stmt = NULL;
    curr = extra;
    sqlLength = sqlLength - (extra-curr);
  }


  rv = JS_TRUE;

  done:
  JS_FreeCString(ctx, sql);
  sqlite3_finalize(stmt);
  return rv;
}

static JSValue db_query_all(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  sqlite3 * db = (sqlite3 * ) JS_GetOpaque2(ctx, this_val, jama_sqlite_class_id);
  sqlite3_stmt * stmt = NULL;
  size_t sqlLength;
  const char * sql;
  JSValue rv;

  sql = JS_ToCStringLen(ctx, &sqlLength, argv[0]);

  int rc = sqlite3_prepare_v2(db, sql, sqlLength, &stmt, NULL);
  if(rc != SQLITE_OK) {
    rv = JS_EXCEPTION;
    goto done;
  }
  if(!(sqlite3_stmt_readonly(stmt))) {
    rv =  JS_ThrowPlainError(ctx, "SQL in queryAll() must be a read-only script, ie SELECT only. ");
    goto done;
  }

  if(!JS_IsUndefined(argv[1])) {
    rv = apply_parameters(ctx, argv[1], stmt);
    if(JS_IsException(rv)) {
      goto done;
    }
  }

  JSValue array = JS_NewArray(ctx);
  int i = 0;
  while(1) {
    rc = sqlite3_step(stmt);
    if(rc == SQLITE_DONE) {
      break;
    }
    if(rc != SQLITE_ROW) {
      rv = JS_EXCEPTION;
      goto done;
    }
    JSValue row = JS_NewObject(ctx);

    for(int i = 0; i < sqlite3_column_count(stmt); i++) {
      JSValue v = column_value_to_js(ctx, stmt, i);
      JS_DefinePropertyValueStr(ctx, row, sqlite3_column_name(stmt, i), v, JS_PROP_C_W_E);
      ;
    }

    JS_DefinePropertyValueUint32(ctx, array, i, row, JS_PROP_C_W_E);
    i++;
  }

  rv = array;


  done:
  JS_FreeCString(ctx, sql);
  sqlite3_finalize(stmt);
  return rv;
}


struct u {
  JSContext * ctx;
  JSValue callback;
};

static void _update_hook(void * p, int operation,char const * db, char const * table,sqlite3_int64 rowid) {
  struct u * update = (struct u *) p;
  JSContext *ctx = update->ctx;
  JSValue callback = JS_DupValue(ctx, update->callback);

  char * opName = "";
  switch(operation) {
    case SQLITE_INSERT:
      opName = "insert";
      break;
    case SQLITE_DELETE:
      opName = "delete";
      break;
    case SQLITE_UPDATE:
      opName = "update";
      break;
  }

  JSValue argv[] = {
    JS_NewString(ctx, opName),
    JS_NewString(ctx, db),
    JS_NewString(ctx, table),
    JS_NewInt64(ctx, rowid)
  };

  // ignore return value for now?
  JSValue rv = JS_Call(update->ctx, callback, JS_UNDEFINED, countof(argv), argv);

  for(int i = 0; i < countof(argv); i++) {
    JS_FreeValue(ctx, argv[i]);
  }
  JS_FreeValue(ctx, callback);
  JS_FreeValue(ctx, rv);
}

static JSValue db_update_hook(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  sqlite3 * db = (sqlite3 * ) JS_GetOpaque2(ctx, this_val, jama_sqlite_class_id);
  struct u  * ux;

  JSValue func = argv[0];

  if(JS_IsUndefined(func)) {
    struct u *prev = (struct u  *) sqlite3_update_hook(db, NULL, NULL);
    if(prev) {
      JS_FreeValue(ctx, prev->callback);
    }
    return JS_UNDEFINED;
  }
  if (!JS_IsFunction(ctx, func))
      return JS_ThrowTypeError(ctx, "not a function");

  ux = js_mallocz(ctx, sizeof(ux));
  assert(ux);

  ux->ctx = ctx;
  ux->callback = JS_DupValue(ctx, func);
  void * prev = sqlite3_update_hook(db, _update_hook, ux);
  return JS_UNDEFINED;
}

static const JSCFunctionListEntry js_sqlite_database_proto_funcs[] = {
  
  JS_CFUNC_DEF("queryAll", 2, db_query_all ),
  JS_CFUNC_DEF("queryRow", 1, db_query_row ),
  JS_CFUNC_DEF("queryValue", 2, db_query_value ),

  JS_CFUNC_DEF("execute", 2, db_execute ),
  JS_CFUNC_DEF("executeScript", 1, db_execute_script ),

  JS_CFUNC_DEF("updateHook", 1, db_update_hook ),

};


static JSValue database_create(JSContext *ctx, JSValue this_val, int argc, JSValue *argv) {
  int rc;
  JSValue proto = JS_GetClassProto(ctx, jama_sqlite_class_id);
  JSValue obj = JS_NewObjectProtoClass(ctx, proto, jama_sqlite_class_id);
  JS_FreeValue(ctx, proto);
  sqlite3* db;

  if(JS_IsString(argv[0])) {
    const char * dbPath = JS_ToCString(ctx, argv[0]);
    rc = sqlite3_open(dbPath, &db);
    JS_FreeCString(ctx, dbPath);
  }
  else {
    rc = sqlite3_open(":memory:", &db);
  }

  assert(rc == SQLITE_OK);
  sqlite3_vec_init(db, NULL, NULL);
  sqlite3_csv_init(db, NULL, NULL);
  
  JS_SetOpaque(obj, db);
  return obj;
}


static int js_sqlite_init(JSContext *ctx, JSModuleDef *m)
{
    JSRuntime *rt = JS_GetRuntime(ctx);

    JS_NewClassID(JS_GetRuntime(ctx), &jama_sqlite_class_id);
    JS_NewClass(rt, jama_sqlite_class_id, &jama_sqlite_class);
    JSValue proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, proto, js_sqlite_database_proto_funcs, countof(js_sqlite_database_proto_funcs));

    JSValue obj = JS_NewCFunction2(ctx, database_create, "Database", 1, JS_CFUNC_constructor, 0);
    JS_SetConstructor(ctx, obj, proto);

    JS_SetClassProto(ctx, jama_sqlite_class_id, proto);

    JS_SetModuleExport(ctx, m, "Database", obj);
    return JS_SetModuleExportList(ctx, m, js_sqlite_funcs, countof(js_sqlite_funcs));
}

JSModuleDef *js_init_module_sqlite(JSContext *ctx, const char *module_name)
{
    JSModuleDef *m;
    m = JS_NewCModule(ctx, module_name, js_sqlite_init);
    if (!m)
        return NULL;
    JS_AddModuleExportList(ctx, m, js_sqlite_funcs, countof(js_sqlite_funcs));
    JS_AddModuleExport(ctx, m, "Database");
    return m;
}
