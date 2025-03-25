/*
 * QuickJS Javascript Engine
 *
 * Copyright (c) 2017-2021 Fabrice Bellard
 * Copyright (c) 2017-2021 Charlie Gordon
 * Copyright (c) 2023 Ben Noordhuis
 * Copyright (c) 2023 Saúl Ibarra Corretgé
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef QUICKJS_H
#define QUICKJS_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define js_force_inline       inline __attribute__((always_inline))
#define __js_printf_like(f, a)   __attribute__((format(printf, f, a)))
#define JS_EXTERN __attribute__((visibility("default")))
#else
#define js_force_inline  inline
#define __js_printf_like(a, b)
#define JS_EXTERN /* nothing */
#endif

#define JS_BOOL int

typedef struct JSRuntime JSRuntime;
typedef struct JSContext JSContext;
typedef struct JSObject JSObject;
typedef struct JSClass JSClass;
typedef uint32_t JSClassID;
typedef uint32_t JSAtom;

/* Unless documented otherwise, C string pointers (`char *` or `const char *`)
   are assumed to verify these constraints:
   - unless a length is passed separately, the string has a null terminator
   - string contents is either pure ASCII or is UTF-8 encoded.
 */

/* Overridable purely for testing purposes; don't touch. */
#ifndef JS_NAN_BOXING
#if INTPTR_MAX < INT64_MAX
#define JS_NAN_BOXING 1 /* Use NAN boxing for 32bit builds. */
#endif
#endif

enum {
    /* all tags with a reference count are negative */
    JS_TAG_FIRST       = -9, /* first negative tag */
    JS_TAG_BIG_INT     = -9,
    JS_TAG_SYMBOL      = -8,
    JS_TAG_STRING      = -7,
    JS_TAG_MODULE      = -3, /* used internally */
    JS_TAG_FUNCTION_BYTECODE = -2, /* used internally */
    JS_TAG_OBJECT      = -1,

    JS_TAG_INT         = 0,
    JS_TAG_BOOL        = 1,
    JS_TAG_NULL        = 2,
    JS_TAG_UNDEFINED   = 3,
    JS_TAG_UNINITIALIZED = 4,
    JS_TAG_CATCH_OFFSET = 5,
    JS_TAG_EXCEPTION   = 6,
    JS_TAG_FLOAT64     = 7,
    /* any larger tag is FLOAT64 if JS_NAN_BOXING */
};

#define JS_FLOAT64_NAN NAN
#define JSValueConst JSValue /* For backwards compatibility. */

#if defined(JS_NAN_BOXING) && JS_NAN_BOXING

typedef uint64_t JSValue;

#define JS_VALUE_GET_TAG(v) (int)((v) >> 32)
#define JS_VALUE_GET_INT(v) (int)(v)
#define JS_VALUE_GET_BOOL(v) (int)(v)
#define JS_VALUE_GET_PTR(v) (void *)(intptr_t)(v)

#define JS_MKVAL(tag, val) (((uint64_t)(tag) << 32) | (uint32_t)(val))
#define JS_MKPTR(tag, ptr) (((uint64_t)(tag) << 32) | (uintptr_t)(ptr))

#define JS_FLOAT64_TAG_ADDEND (0x7ff80000 - JS_TAG_FIRST + 1) /* quiet NaN encoding */

static inline double JS_VALUE_GET_FLOAT64(JSValue v)
{
    union {
        JSValue v;
        double d;
    } u;
    u.v = v;
    u.v += (uint64_t)JS_FLOAT64_TAG_ADDEND << 32;
    return u.d;
}

#define JS_NAN (0x7ff8000000000000 - ((uint64_t)JS_FLOAT64_TAG_ADDEND << 32))

static inline JSValue __JS_NewFloat64(double d)
{
    union {
        double d;
        uint64_t u64;
    } u;
    JSValue v;
    u.d = d;
    /* normalize NaN */
    if ((u.u64 & 0x7fffffffffffffff) > 0x7ff0000000000000)
        v = JS_NAN;
    else
        v = u.u64 - ((uint64_t)JS_FLOAT64_TAG_ADDEND << 32);
    return v;
}

#define JS_TAG_IS_FLOAT64(tag) ((unsigned)((tag) - JS_TAG_FIRST) >= (JS_TAG_FLOAT64 - JS_TAG_FIRST))

/* same as JS_VALUE_GET_TAG, but return JS_TAG_FLOAT64 with NaN boxing */
static inline int JS_VALUE_GET_NORM_TAG(JSValue v)
{
    uint32_t tag;
    tag = JS_VALUE_GET_TAG(v);
    if (JS_TAG_IS_FLOAT64(tag))
        return JS_TAG_FLOAT64;
    else
        return tag;
}

static inline JS_BOOL JS_VALUE_IS_NAN(JSValue v)
{
    uint32_t tag;
    tag = JS_VALUE_GET_TAG(v);
    return tag == (JS_NAN >> 32);
}

#else /* !JS_NAN_BOXING */

typedef union JSValueUnion {
    int32_t int32;
    double float64;
    void *ptr;
} JSValueUnion;

typedef struct JSValue {
    JSValueUnion u;
    int64_t tag;
} JSValue;

#define JS_VALUE_GET_TAG(v) ((int32_t)(v).tag)
/* same as JS_VALUE_GET_TAG, but return JS_TAG_FLOAT64 with NaN boxing */
#define JS_VALUE_GET_NORM_TAG(v) JS_VALUE_GET_TAG(v)
#define JS_VALUE_GET_INT(v) ((v).u.int32)
#define JS_VALUE_GET_BOOL(v) ((v).u.int32)
#define JS_VALUE_GET_FLOAT64(v) ((v).u.float64)
#define JS_VALUE_GET_PTR(v) ((v).u.ptr)

/* msvc doesn't understand designated initializers without /std:c++20 */
#ifdef __cplusplus
static inline JSValue JS_MKPTR(int64_t tag, void *ptr)
{
    JSValue v;
    v.u.ptr = ptr;
    v.tag = tag;
    return v;
}
static inline JSValue JS_MKVAL(int64_t tag, int32_t int32)
{
    JSValue v;
    v.u.int32 = int32;
    v.tag = tag;
    return v;
}
static inline JSValue JS_MKNAN(void)
{
    JSValue v;
    v.u.float64 = JS_FLOAT64_NAN;
    v.tag = JS_TAG_FLOAT64;
    return v;
}
/* provide as macros for consistency and backward compat reasons */
#define JS_MKPTR(tag, ptr) JS_MKPTR(tag, ptr)
#define JS_MKVAL(tag, val) JS_MKVAL(tag, val)
#define JS_NAN             JS_MKNAN() /* alas, not a constant expression */
#else
#define JS_MKPTR(tag, p)   (JSValue){ (JSValueUnion){ .ptr = p }, tag }
#define JS_MKVAL(tag, val) (JSValue){ (JSValueUnion){ .int32 = val }, tag }
#define JS_NAN             (JSValue){ (JSValueUnion){ .float64 = JS_FLOAT64_NAN }, JS_TAG_FLOAT64 }
#endif

#define JS_TAG_IS_FLOAT64(tag) ((unsigned)(tag) == JS_TAG_FLOAT64)

static inline JSValue __JS_NewFloat64(double d)
{
    JSValue v;
    v.tag = JS_TAG_FLOAT64;
    v.u.float64 = d;
    return v;
}

static inline JS_BOOL JS_VALUE_IS_NAN(JSValue v)
{
    union {
        double d;
        uint64_t u64;
    } u;
    if (v.tag != JS_TAG_FLOAT64)
        return 0;
    u.d = v.u.float64;
    return (u.u64 & 0x7fffffffffffffff) > 0x7ff0000000000000;
}

#endif /* !JS_NAN_BOXING */

#define JS_VALUE_IS_BOTH_INT(v1, v2) ((JS_VALUE_GET_TAG(v1) | JS_VALUE_GET_TAG(v2)) == 0)
#define JS_VALUE_IS_BOTH_FLOAT(v1, v2) (JS_TAG_IS_FLOAT64(JS_VALUE_GET_TAG(v1)) && JS_TAG_IS_FLOAT64(JS_VALUE_GET_TAG(v2)))

#define JS_VALUE_GET_OBJ(v) ((JSObject *)JS_VALUE_GET_PTR(v))
#define JS_VALUE_HAS_REF_COUNT(v) ((unsigned)JS_VALUE_GET_TAG(v) >= (unsigned)JS_TAG_FIRST)

/* special values */
#define JS_NULL      JS_MKVAL(JS_TAG_NULL, 0)
#define JS_UNDEFINED JS_MKVAL(JS_TAG_UNDEFINED, 0)
#define JS_FALSE     JS_MKVAL(JS_TAG_BOOL, 0)
#define JS_TRUE      JS_MKVAL(JS_TAG_BOOL, 1)
#define JS_EXCEPTION JS_MKVAL(JS_TAG_EXCEPTION, 0)
#define JS_UNINITIALIZED JS_MKVAL(JS_TAG_UNINITIALIZED, 0)

/* flags for object properties */
#define JS_PROP_CONFIGURABLE  (1 << 0)
#define JS_PROP_WRITABLE      (1 << 1)
#define JS_PROP_ENUMERABLE    (1 << 2)
#define JS_PROP_C_W_E         (JS_PROP_CONFIGURABLE | JS_PROP_WRITABLE | JS_PROP_ENUMERABLE)
#define JS_PROP_LENGTH        (1 << 3) /* used internally in Arrays */
#define JS_PROP_TMASK         (3 << 4) /* mask for NORMAL, GETSET, VARREF, AUTOINIT */
#define JS_PROP_NORMAL         (0 << 4)
#define JS_PROP_GETSET         (1 << 4)
#define JS_PROP_VARREF         (2 << 4) /* used internally */
#define JS_PROP_AUTOINIT       (3 << 4) /* used internally */

/* flags for JS_DefineProperty */
#define JS_PROP_HAS_SHIFT        8
#define JS_PROP_HAS_CONFIGURABLE (1 << 8)
#define JS_PROP_HAS_WRITABLE     (1 << 9)
#define JS_PROP_HAS_ENUMERABLE   (1 << 10)
#define JS_PROP_HAS_GET          (1 << 11)
#define JS_PROP_HAS_SET          (1 << 12)
#define JS_PROP_HAS_VALUE        (1 << 13)

/* throw an exception if false would be returned
   (JS_DefineProperty/JS_SetProperty) */
#define JS_PROP_THROW            (1 << 14)
/* throw an exception if false would be returned in strict mode
   (JS_SetProperty) */
#define JS_PROP_THROW_STRICT     (1 << 15)

#define JS_PROP_NO_ADD           (1 << 16) /* internal use */
#define JS_PROP_NO_EXOTIC        (1 << 17) /* internal use */
#define JS_PROP_DEFINE_PROPERTY  (1 << 18) /* internal use */
#define JS_PROP_REFLECT_DEFINE_PROPERTY (1 << 19) /* internal use */

#if defined(__wasi__)
#define JS_DEFAULT_STACK_SIZE 0
#else
#define JS_DEFAULT_STACK_SIZE (256 * 1024)
#endif

/* JS_Eval() flags */
#define JS_EVAL_TYPE_GLOBAL   (0 << 0) /* global code (default) */
#define JS_EVAL_TYPE_MODULE   (1 << 0) /* module code */
#define JS_EVAL_TYPE_DIRECT   (2 << 0) /* direct call (internal use) */
#define JS_EVAL_TYPE_INDIRECT (3 << 0) /* indirect call (internal use) */
#define JS_EVAL_TYPE_MASK     (3 << 0)

#define JS_EVAL_FLAG_STRICT   (1 << 3) /* force 'strict' mode */
#define JS_EVAL_FLAG_UNUSED   (1 << 4) /* unused */
/* compile but do not run. The result is an object with a
   JS_TAG_FUNCTION_BYTECODE or JS_TAG_MODULE tag. It can be executed
   with JS_EvalFunction(). */
#define JS_EVAL_FLAG_COMPILE_ONLY (1 << 5)
/* don't include the stack frames before this eval in the Error() backtraces */
#define JS_EVAL_FLAG_BACKTRACE_BARRIER (1 << 6)
/* allow top-level await in normal script. JS_Eval() returns a
   promise. Only allowed with JS_EVAL_TYPE_GLOBAL */
#define JS_EVAL_FLAG_ASYNC (1 << 7)

typedef JSValue JSCFunction(JSContext *ctx, JSValue this_val, int argc, JSValue *argv);
typedef JSValue JSCFunctionMagic(JSContext *ctx, JSValue this_val, int argc, JSValue *argv, int magic);
typedef JSValue JSCFunctionData(JSContext *ctx, JSValue this_val, int argc, JSValue *argv, int magic, JSValue *func_data);

typedef struct JSMallocFunctions {
    void *(*js_calloc)(void *opaque, size_t count, size_t size);
    void *(*js_malloc)(void *opaque, size_t size);
    void (*js_free)(void *opaque, void *ptr);
    void *(*js_realloc)(void *opaque, void *ptr, size_t size);
    size_t (*js_malloc_usable_size)(const void *ptr);
} JSMallocFunctions;

// Finalizers run in LIFO order at the very end of JS_FreeRuntime.
// Intended for cleanup of associated resources; the runtime itself
// is no longer usable.
typedef void JSRuntimeFinalizer(JSRuntime *rt, void *arg);

typedef struct JSGCObjectHeader JSGCObjectHeader;

JS_EXTERN JSRuntime *JS_NewRuntime(void);
/* info lifetime must exceed that of rt */
JS_EXTERN void JS_SetRuntimeInfo(JSRuntime *rt, const char *info);
/* use 0 to disable memory limit */
JS_EXTERN void JS_SetMemoryLimit(JSRuntime *rt, size_t limit);
JS_EXTERN void JS_SetDumpFlags(JSRuntime *rt, uint64_t flags);
JS_EXTERN size_t JS_GetGCThreshold(JSRuntime *rt);
JS_EXTERN void JS_SetGCThreshold(JSRuntime *rt, size_t gc_threshold);
/* use 0 to disable maximum stack size check */
JS_EXTERN void JS_SetMaxStackSize(JSRuntime *rt, size_t stack_size);
/* should be called when changing thread to update the stack top value
   used to check stack overflow. */
JS_EXTERN void JS_UpdateStackTop(JSRuntime *rt);
JS_EXTERN JSRuntime *JS_NewRuntime2(const JSMallocFunctions *mf, void *opaque);
JS_EXTERN void JS_FreeRuntime(JSRuntime *rt);
JS_EXTERN void *JS_GetRuntimeOpaque(JSRuntime *rt);
JS_EXTERN void JS_SetRuntimeOpaque(JSRuntime *rt, void *opaque);
JS_EXTERN int JS_AddRuntimeFinalizer(JSRuntime *rt,
                                     JSRuntimeFinalizer *finalizer, void *arg);
typedef void JS_MarkFunc(JSRuntime *rt, JSGCObjectHeader *gp);
JS_EXTERN void JS_MarkValue(JSRuntime *rt, JSValue val, JS_MarkFunc *mark_func);
JS_EXTERN void JS_RunGC(JSRuntime *rt);
JS_EXTERN JS_BOOL JS_IsLiveObject(JSRuntime *rt, JSValue obj);

JS_EXTERN JSContext *JS_NewContext(JSRuntime *rt);
JS_EXTERN void JS_FreeContext(JSContext *s);
JS_EXTERN JSContext *JS_DupContext(JSContext *ctx);
JS_EXTERN void *JS_GetContextOpaque(JSContext *ctx);
JS_EXTERN void JS_SetContextOpaque(JSContext *ctx, void *opaque);
JS_EXTERN JSRuntime *JS_GetRuntime(JSContext *ctx);
JS_EXTERN void JS_SetClassProto(JSContext *ctx, JSClassID class_id, JSValue obj);
JS_EXTERN JSValue JS_GetClassProto(JSContext *ctx, JSClassID class_id);
JS_EXTERN JSValue JS_GetFunctionProto(JSContext *ctx);

/* the following functions are used to select the intrinsic object to
   save memory */
JS_EXTERN JSContext *JS_NewContextRaw(JSRuntime *rt);
JS_EXTERN void JS_AddIntrinsicBaseObjects(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicDate(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicEval(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicRegExpCompiler(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicRegExp(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicJSON(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicProxy(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicMapSet(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicTypedArrays(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicPromise(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicBigInt(JSContext *ctx);
JS_EXTERN void JS_AddIntrinsicWeakRef(JSContext *ctx);
JS_EXTERN void JS_AddPerformance(JSContext *ctx);

/* for equality comparisons and sameness */
JS_EXTERN int JS_IsEqual(JSContext *ctx, JSValue op1, JSValue op2);
JS_EXTERN JS_BOOL JS_IsStrictEqual(JSContext *ctx, JSValue op1, JSValue op2);
JS_EXTERN JS_BOOL JS_IsSameValue(JSContext *ctx, JSValue op1, JSValue op2);
/* Similar to same-value equality, but +0 and -0 are considered equal. */
JS_EXTERN JS_BOOL JS_IsSameValueZero(JSContext *ctx, JSValue op1, JSValue op2);

/* Only used for running 262 tests. TODO(saghul) add build time flag. */
JS_EXTERN JSValue js_string_codePointRange(JSContext *ctx, JSValue this_val,
                                 int argc, JSValue *argv);

JS_EXTERN void *js_calloc_rt(JSRuntime *rt, size_t count, size_t size);
JS_EXTERN void *js_malloc_rt(JSRuntime *rt, size_t size);
JS_EXTERN void js_free_rt(JSRuntime *rt, void *ptr);
JS_EXTERN void *js_realloc_rt(JSRuntime *rt, void *ptr, size_t size);
JS_EXTERN size_t js_malloc_usable_size_rt(JSRuntime *rt, const void *ptr);
JS_EXTERN void *js_mallocz_rt(JSRuntime *rt, size_t size);

JS_EXTERN void *js_calloc(JSContext *ctx, size_t count, size_t size);
JS_EXTERN void *js_malloc(JSContext *ctx, size_t size);
JS_EXTERN void js_free(JSContext *ctx, void *ptr);
JS_EXTERN void *js_realloc(JSContext *ctx, void *ptr, size_t size);
JS_EXTERN size_t js_malloc_usable_size(JSContext *ctx, const void *ptr);
JS_EXTERN void *js_realloc2(JSContext *ctx, void *ptr, size_t size, size_t *pslack);
JS_EXTERN void *js_mallocz(JSContext *ctx, size_t size);
JS_EXTERN char *js_strdup(JSContext *ctx, const char *str);
JS_EXTERN char *js_strndup(JSContext *ctx, const char *s, size_t n);

typedef struct JSMemoryUsage {
    int64_t malloc_size, malloc_limit, memory_used_size;
    int64_t malloc_count;
    int64_t memory_used_count;
    int64_t atom_count, atom_size;
    int64_t str_count, str_size;
    int64_t obj_count, obj_size;
    int64_t prop_count, prop_size;
    int64_t shape_count, shape_size;
    int64_t js_func_count, js_func_size, js_func_code_size;
    int64_t js_func_pc2line_count, js_func_pc2line_size;
    int64_t c_func_count, array_count;
    int64_t fast_array_count, fast_array_elements;
    int64_t binary_object_count, binary_object_size;
} JSMemoryUsage;

JS_EXTERN void JS_ComputeMemoryUsage(JSRuntime *rt, JSMemoryUsage *s);
JS_EXTERN void JS_DumpMemoryUsage(FILE *fp, const JSMemoryUsage *s, JSRuntime *rt);

/* atom support */
#define JS_ATOM_NULL 0

JS_EXTERN JSAtom JS_NewAtomLen(JSContext *ctx, const char *str, size_t len);
JS_EXTERN JSAtom JS_NewAtom(JSContext *ctx, const char *str);
JS_EXTERN JSAtom JS_NewAtomUInt32(JSContext *ctx, uint32_t n);
JS_EXTERN JSAtom JS_DupAtom(JSContext *ctx, JSAtom v);
JS_EXTERN void JS_FreeAtom(JSContext *ctx, JSAtom v);
JS_EXTERN void JS_FreeAtomRT(JSRuntime *rt, JSAtom v);
JS_EXTERN JSValue JS_AtomToValue(JSContext *ctx, JSAtom atom);
JS_EXTERN JSValue JS_AtomToString(JSContext *ctx, JSAtom atom);
JS_EXTERN const char *JS_AtomToCString(JSContext *ctx, JSAtom atom);
JS_EXTERN JSAtom JS_ValueToAtom(JSContext *ctx, JSValue val);

/* object class support */

typedef struct JSPropertyEnum {
    JS_BOOL is_enumerable;
    JSAtom atom;
} JSPropertyEnum;

typedef struct JSPropertyDescriptor {
    int flags;
    JSValue value;
    JSValue getter;
    JSValue setter;
} JSPropertyDescriptor;

typedef struct JSClassExoticMethods {
    /* Return -1 if exception (can only happen in case of Proxy object),
       FALSE if the property does not exists, TRUE if it exists. If 1 is
       returned, the property descriptor 'desc' is filled if != NULL. */
    int (*get_own_property)(JSContext *ctx, JSPropertyDescriptor *desc,
                             JSValue obj, JSAtom prop);
    /* '*ptab' should hold the '*plen' property keys. Return 0 if OK,
       -1 if exception. The 'is_enumerable' field is ignored.
    */
    int (*get_own_property_names)(JSContext *ctx, JSPropertyEnum **ptab,
                                  uint32_t *plen,
                                  JSValue obj);
    /* return < 0 if exception, or TRUE/FALSE */
    int (*delete_property)(JSContext *ctx, JSValue obj, JSAtom prop);
    /* return < 0 if exception or TRUE/FALSE */
    int (*define_own_property)(JSContext *ctx, JSValue this_obj,
                               JSAtom prop, JSValue val,
                               JSValue getter, JSValue setter,
                               int flags);
    /* The following methods can be emulated with the previous ones,
       so they are usually not needed */
    /* return < 0 if exception or TRUE/FALSE */
    int (*has_property)(JSContext *ctx, JSValue obj, JSAtom atom);
    JSValue (*get_property)(JSContext *ctx, JSValue obj, JSAtom atom,
                            JSValue receiver);
    /* return < 0 if exception or TRUE/FALSE */
    int (*set_property)(JSContext *ctx, JSValue obj, JSAtom atom,
                        JSValue value, JSValue receiver, int flags);
} JSClassExoticMethods;

typedef void JSClassFinalizer(JSRuntime *rt, JSValue val);
typedef void JSClassGCMark(JSRuntime *rt, JSValue val,
                           JS_MarkFunc *mark_func);
#define JS_CALL_FLAG_CONSTRUCTOR (1 << 0)
typedef JSValue JSClassCall(JSContext *ctx, JSValue func_obj,
                            JSValue this_val, int argc, JSValue *argv,
                            int flags);

typedef struct JSClassDef {
    const char *class_name; /* pure ASCII only! */
    JSClassFinalizer *finalizer;
    JSClassGCMark *gc_mark;
    /* if call != NULL, the object is a function. If (flags &
       JS_CALL_FLAG_CONSTRUCTOR) != 0, the function is called as a
       constructor. In this case, 'this_val' is new.target. A
       constructor call only happens if the object constructor bit is
       set (see JS_SetConstructorBit()). */
    JSClassCall *call;
    /* XXX: suppress this indirection ? It is here only to save memory
       because only a few classes need these methods */
    JSClassExoticMethods *exotic;
} JSClassDef;

#define JS_INVALID_CLASS_ID 0
JS_EXTERN JSClassID JS_NewClassID(JSRuntime *rt, JSClassID *pclass_id);
/* Returns the class ID if `v` is an object, otherwise returns JS_INVALID_CLASS_ID. */
JS_EXTERN JSClassID JS_GetClassID(JSValue v);
JS_EXTERN int JS_NewClass(JSRuntime *rt, JSClassID class_id, const JSClassDef *class_def);
JS_EXTERN int JS_IsRegisteredClass(JSRuntime *rt, JSClassID class_id);

/* value handling */

static js_force_inline JSValue JS_NewBool(JSContext *ctx, JS_BOOL val)
{
    (void)&ctx;
    return JS_MKVAL(JS_TAG_BOOL, (val != 0));
}

static js_force_inline JSValue JS_NewInt32(JSContext *ctx, int32_t val)
{
    (void)&ctx;
    return JS_MKVAL(JS_TAG_INT, val);
}

static js_force_inline JSValue JS_NewFloat64(JSContext *ctx, double val)
{
    (void)&ctx;
    return __JS_NewFloat64(val);
}

static js_force_inline JSValue JS_NewCatchOffset(JSContext *ctx, int32_t val)
{
    (void)&ctx;
    return JS_MKVAL(JS_TAG_CATCH_OFFSET, val);
}

static js_force_inline JSValue JS_NewInt64(JSContext *ctx, int64_t val)
{
    JSValue v;
    if (val >= INT32_MIN && val <= INT32_MAX) {
        v = JS_NewInt32(ctx, (int32_t)val);
    } else {
        v = JS_NewFloat64(ctx, (double)val);
    }
    return v;
}

static js_force_inline JSValue JS_NewUint32(JSContext *ctx, uint32_t val)
{
    JSValue v;
    if (val <= INT32_MAX) {
        v = JS_NewInt32(ctx, (int32_t)val);
    } else {
        v = JS_NewFloat64(ctx, (double)val);
    }
    return v;
}

JS_EXTERN JSValue JS_NewNumber(JSContext *ctx, double d);
JS_EXTERN JSValue JS_NewBigInt64(JSContext *ctx, int64_t v);
JS_EXTERN JSValue JS_NewBigUint64(JSContext *ctx, uint64_t v);

static inline JS_BOOL JS_IsNumber(JSValue v)
{
    int tag = JS_VALUE_GET_TAG(v);
    return tag == JS_TAG_INT || JS_TAG_IS_FLOAT64(tag);
}

static inline JS_BOOL JS_IsBigInt(JSContext *ctx, JSValue v)
{
    (void)&ctx;
    return JS_VALUE_GET_TAG(v) == JS_TAG_BIG_INT;
}

static inline JS_BOOL JS_IsBool(JSValue v)
{
    return JS_VALUE_GET_TAG(v) == JS_TAG_BOOL;
}

static inline JS_BOOL JS_IsNull(JSValue v)
{
    return JS_VALUE_GET_TAG(v) == JS_TAG_NULL;
}

static inline JS_BOOL JS_IsUndefined(JSValue v)
{
    return JS_VALUE_GET_TAG(v) == JS_TAG_UNDEFINED;
}

static inline JS_BOOL JS_IsException(JSValue v)
{
    return JS_VALUE_GET_TAG(v) == JS_TAG_EXCEPTION;
}

static inline JS_BOOL JS_IsUninitialized(JSValue v)
{
    return JS_VALUE_GET_TAG(v) == JS_TAG_UNINITIALIZED;
}

static inline JS_BOOL JS_IsString(JSValue v)
{
    return JS_VALUE_GET_TAG(v) == JS_TAG_STRING;
}

static inline JS_BOOL JS_IsSymbol(JSValue v)
{
    return JS_VALUE_GET_TAG(v) == JS_TAG_SYMBOL;
}

static inline JS_BOOL JS_IsObject(JSValue v)
{
    return JS_VALUE_GET_TAG(v) == JS_TAG_OBJECT;
}

JS_EXTERN JSValue JS_Throw(JSContext *ctx, JSValue obj);
JS_EXTERN JSValue JS_GetException(JSContext *ctx);
JS_BOOL JS_HasException(JSContext *ctx);
JS_EXTERN JS_BOOL JS_IsError(JSContext *ctx, JSValue val);
JS_EXTERN void JS_ResetUncatchableError(JSContext *ctx);
JS_EXTERN JSValue JS_NewError(JSContext *ctx);
JS_EXTERN JSValue __js_printf_like(2, 3) JS_ThrowPlainError(JSContext *ctx, const char *fmt, ...);
JS_EXTERN JSValue __js_printf_like(2, 3) JS_ThrowSyntaxError(JSContext *ctx, const char *fmt, ...);
JS_EXTERN JSValue __js_printf_like(2, 3) JS_ThrowTypeError(JSContext *ctx, const char *fmt, ...);
JS_EXTERN JSValue __js_printf_like(2, 3) JS_ThrowReferenceError(JSContext *ctx, const char *fmt, ...);
JS_EXTERN JSValue __js_printf_like(2, 3) JS_ThrowRangeError(JSContext *ctx, const char *fmt, ...);
JS_EXTERN JSValue __js_printf_like(2, 3) JS_ThrowInternalError(JSContext *ctx, const char *fmt, ...);
JS_EXTERN JSValue JS_ThrowOutOfMemory(JSContext *ctx);
JS_EXTERN void JS_FreeValue(JSContext *ctx, JSValue v);
JS_EXTERN void JS_FreeValueRT(JSRuntime *rt, JSValue v);
JS_EXTERN JSValue JS_DupValue(JSContext *ctx, JSValue v);
JS_EXTERN JSValue JS_DupValueRT(JSRuntime *rt, JSValue v);
JS_EXTERN int JS_ToBool(JSContext *ctx, JSValue val); /* return -1 for JS_EXCEPTION */
JS_EXTERN int JS_ToInt32(JSContext *ctx, int32_t *pres, JSValue val);
static inline int JS_ToUint32(JSContext *ctx, uint32_t *pres, JSValue val)
{
    return JS_ToInt32(ctx, (int32_t*)pres, val);
}
JS_EXTERN int JS_ToInt64(JSContext *ctx, int64_t *pres, JSValue val);
JS_EXTERN int JS_ToIndex(JSContext *ctx, uint64_t *plen, JSValue val);
JS_EXTERN int JS_ToFloat64(JSContext *ctx, double *pres, JSValue val);
/* return an exception if 'val' is a Number */
JS_EXTERN int JS_ToBigInt64(JSContext *ctx, int64_t *pres, JSValue val);
JS_EXTERN int JS_ToBigUint64(JSContext *ctx, uint64_t *pres, JSValue val);
/* same as JS_ToInt64() but allow BigInt */
JS_EXTERN int JS_ToInt64Ext(JSContext *ctx, int64_t *pres, JSValue val);

JS_EXTERN JSValue JS_NewStringLen(JSContext *ctx, const char *str1, size_t len1);
static inline JSValue JS_NewString(JSContext *ctx, const char *str) {
    return JS_NewStringLen(ctx, str, strlen(str));
}
JS_EXTERN JSValue JS_NewAtomString(JSContext *ctx, const char *str);
JS_EXTERN JSValue JS_ToString(JSContext *ctx, JSValue val);
JS_EXTERN JSValue JS_ToPropertyKey(JSContext *ctx, JSValue val);
JS_EXTERN const char *JS_ToCStringLen2(JSContext *ctx, size_t *plen, JSValue val1, JS_BOOL cesu8);
static inline const char *JS_ToCStringLen(JSContext *ctx, size_t *plen, JSValue val1)
{
    return JS_ToCStringLen2(ctx, plen, val1, 0);
}
static inline const char *JS_ToCString(JSContext *ctx, JSValue val1)
{
    return JS_ToCStringLen2(ctx, NULL, val1, 0);
}
JS_EXTERN void JS_FreeCString(JSContext *ctx, const char *ptr);

JS_EXTERN JSValue JS_NewObjectProtoClass(JSContext *ctx, JSValue proto, JSClassID class_id);
JS_EXTERN JSValue JS_NewObjectClass(JSContext *ctx, int class_id);
JS_EXTERN JSValue JS_NewObjectProto(JSContext *ctx, JSValue proto);
JS_EXTERN JSValue JS_NewObject(JSContext *ctx);

JS_EXTERN JS_BOOL JS_IsFunction(JSContext* ctx, JSValue val);
JS_EXTERN JS_BOOL JS_IsConstructor(JSContext* ctx, JSValue val);
JS_EXTERN JS_BOOL JS_SetConstructorBit(JSContext *ctx, JSValue func_obj, JS_BOOL val);

JS_EXTERN JSValue JS_NewArray(JSContext *ctx);
JS_EXTERN int JS_IsArray(JSContext *ctx, JSValue val);

JS_EXTERN JSValue JS_NewDate(JSContext *ctx, double epoch_ms);

JS_EXTERN JSValue JS_GetProperty(JSContext *ctx, JSValue this_obj, JSAtom prop);
JS_EXTERN JSValue JS_GetPropertyUint32(JSContext *ctx, JSValue this_obj,
                                       uint32_t idx);
JS_EXTERN JSValue JS_GetPropertyInt64(JSContext *ctx, JSValue this_obj,
                                      int64_t idx);
JS_EXTERN JSValue JS_GetPropertyStr(JSContext *ctx, JSValue this_obj,
                                    const char *prop);

JS_EXTERN int JS_SetProperty(JSContext *ctx, JSValue this_obj,
                             JSAtom prop, JSValue val);
JS_EXTERN int JS_SetPropertyUint32(JSContext *ctx, JSValue this_obj,
                                   uint32_t idx, JSValue val);
JS_EXTERN int JS_SetPropertyInt64(JSContext *ctx, JSValue this_obj,
                                  int64_t idx, JSValue val);
JS_EXTERN int JS_SetPropertyStr(JSContext *ctx, JSValue this_obj,
                                const char *prop, JSValue val);
JS_EXTERN int JS_HasProperty(JSContext *ctx, JSValue this_obj, JSAtom prop);
JS_EXTERN int JS_IsExtensible(JSContext *ctx, JSValue obj);
JS_EXTERN int JS_PreventExtensions(JSContext *ctx, JSValue obj);
JS_EXTERN int JS_DeleteProperty(JSContext *ctx, JSValue obj, JSAtom prop, int flags);
JS_EXTERN int JS_SetPrototype(JSContext *ctx, JSValue obj, JSValue proto_val);
JS_EXTERN JSValue JS_GetPrototype(JSContext *ctx, JSValue val);
JS_EXTERN int JS_GetLength(JSContext *ctx, JSValue obj, int64_t *pres);
JS_EXTERN int JS_SetLength(JSContext *ctx, JSValue obj, int64_t len);

#define JS_GPN_STRING_MASK  (1 << 0)
#define JS_GPN_SYMBOL_MASK  (1 << 1)
#define JS_GPN_PRIVATE_MASK (1 << 2)
/* only include the enumerable properties */
#define JS_GPN_ENUM_ONLY    (1 << 4)
/* set theJSPropertyEnum.is_enumerable field */
#define JS_GPN_SET_ENUM     (1 << 5)

JS_EXTERN int JS_GetOwnPropertyNames(JSContext *ctx, JSPropertyEnum **ptab,
                                     uint32_t *plen, JSValue obj, int flags);
JS_EXTERN int JS_GetOwnProperty(JSContext *ctx, JSPropertyDescriptor *desc,
                                JSValue obj, JSAtom prop);
JS_EXTERN void JS_FreePropertyEnum(JSContext *ctx, JSPropertyEnum *tab,
                                   uint32_t len);

JS_EXTERN JSValue JS_Call(JSContext *ctx, JSValue func_obj, JSValue this_obj,
                          int argc, JSValue *argv);
JS_EXTERN JSValue JS_Invoke(JSContext *ctx, JSValue this_val, JSAtom atom,
                            int argc, JSValue *argv);
JS_EXTERN JSValue JS_CallConstructor(JSContext *ctx, JSValue func_obj,
                                     int argc, JSValue *argv);
JS_EXTERN JSValue JS_CallConstructor2(JSContext *ctx, JSValue func_obj,
                                      JSValue new_target,
                                      int argc, JSValue *argv);
/* Try to detect if the input is a module. Returns TRUE if parsing the input
 * as a module produces no syntax errors. It's a naive approach that is not
 * wholly infallible: non-strict classic scripts may _parse_ okay as a module
 * but not _execute_ as one (different runtime semantics.) Use with caution.
 * |input| can be either ASCII or UTF-8 encoded source code.
 */
JS_EXTERN JS_BOOL JS_DetectModule(const char *input, size_t input_len);
/* 'input' must be zero terminated i.e. input[input_len] = '\0'. */
JS_EXTERN JSValue JS_Eval(JSContext *ctx, const char *input, size_t input_len,
                          const char *filename, int eval_flags);
/* same as JS_Eval() but with an explicit 'this_obj' parameter */
JS_EXTERN JSValue JS_EvalThis(JSContext *ctx, JSValue this_obj,
                              const char *input, size_t input_len,
                              const char *filename, int eval_flags);
JS_EXTERN JSValue JS_GetGlobalObject(JSContext *ctx);
JS_EXTERN int JS_IsInstanceOf(JSContext *ctx, JSValue val, JSValue obj);
JS_EXTERN int JS_DefineProperty(JSContext *ctx, JSValue this_obj,
                                JSAtom prop, JSValue val,
                                JSValue getter, JSValue setter, int flags);
JS_EXTERN int JS_DefinePropertyValue(JSContext *ctx, JSValue this_obj,
                                     JSAtom prop, JSValue val, int flags);
JS_EXTERN int JS_DefinePropertyValueUint32(JSContext *ctx, JSValue this_obj,
                                           uint32_t idx, JSValue val, int flags);
JS_EXTERN int JS_DefinePropertyValueStr(JSContext *ctx, JSValue this_obj,
                                        const char *prop, JSValue val, int flags);
JS_EXTERN int JS_DefinePropertyGetSet(JSContext *ctx, JSValue this_obj,
                                      JSAtom prop, JSValue getter, JSValue setter,
                                      int flags);
JS_EXTERN JS_BOOL JS_SetOpaque(JSValue obj, void *opaque);
JS_EXTERN void *JS_GetOpaque(JSValue obj, JSClassID class_id);
JS_EXTERN void *JS_GetOpaque2(JSContext *ctx, JSValue obj, JSClassID class_id);
JS_EXTERN void *JS_GetAnyOpaque(JSValue obj, JSClassID *class_id);

/* 'buf' must be zero terminated i.e. buf[buf_len] = '\0'. */
JS_EXTERN JSValue JS_ParseJSON(JSContext *ctx, const char *buf, size_t buf_len,
                               const char *filename);
JS_EXTERN JSValue JS_JSONStringify(JSContext *ctx, JSValue obj,
                                   JSValue replacer, JSValue space0);

typedef void JSFreeArrayBufferDataFunc(JSRuntime *rt, void *opaque, void *ptr);
JS_EXTERN JSValue JS_NewArrayBuffer(JSContext *ctx, uint8_t *buf, size_t len,
                                    JSFreeArrayBufferDataFunc *free_func, void *opaque,
                                    JS_BOOL is_shared);
JS_EXTERN JSValue JS_NewArrayBufferCopy(JSContext *ctx, const uint8_t *buf, size_t len);
JS_EXTERN void JS_DetachArrayBuffer(JSContext *ctx, JSValue obj);
JS_EXTERN uint8_t *JS_GetArrayBuffer(JSContext *ctx, size_t *psize, JSValue obj);
JS_EXTERN JS_BOOL JS_IsArrayBuffer(JSValue obj);
JS_EXTERN uint8_t *JS_GetUint8Array(JSContext *ctx, size_t *psize, JSValue obj);
JS_EXTERN JSValue JS_GetTypedArrayBuffer(JSContext *ctx, JSValue obj,
                                         size_t *pbyte_offset,
                                         size_t *pbyte_length,
                                         size_t *pbytes_per_element);
JS_EXTERN JSValue JS_NewUint8Array(JSContext *ctx, uint8_t *buf, size_t len,
                                   JSFreeArrayBufferDataFunc *free_func, void *opaque,
                                   JS_BOOL is_shared);
JS_EXTERN JS_BOOL JS_IsUint8Array(JSValue obj);
JS_EXTERN JSValue JS_NewUint8ArrayCopy(JSContext *ctx, const uint8_t *buf, size_t len);
typedef struct {
    void *(*sab_alloc)(void *opaque, size_t size);
    void (*sab_free)(void *opaque, void *ptr);
    void (*sab_dup)(void *opaque, void *ptr);
    void *sab_opaque;
} JSSharedArrayBufferFunctions;
JS_EXTERN void JS_SetSharedArrayBufferFunctions(JSRuntime *rt, const JSSharedArrayBufferFunctions *sf);

typedef enum JSPromiseStateEnum {
    JS_PROMISE_PENDING,
    JS_PROMISE_FULFILLED,
    JS_PROMISE_REJECTED,
} JSPromiseStateEnum;

JS_EXTERN JSValue JS_NewPromiseCapability(JSContext *ctx, JSValue *resolving_funcs);
JS_EXTERN JSPromiseStateEnum JS_PromiseState(JSContext *ctx, JSValue promise);
JS_EXTERN JSValue JS_PromiseResult(JSContext *ctx, JSValue promise);

JS_EXTERN JSValue JS_NewSymbol(JSContext *ctx, const char *description, JS_BOOL is_global);

/* is_handled = TRUE means that the rejection is handled */
typedef void JSHostPromiseRejectionTracker(JSContext *ctx, JSValue promise,
                                           JSValue reason,
                                           JS_BOOL is_handled, void *opaque);
JS_EXTERN void JS_SetHostPromiseRejectionTracker(JSRuntime *rt, JSHostPromiseRejectionTracker *cb, void *opaque);

/* return != 0 if the JS code needs to be interrupted */
typedef int JSInterruptHandler(JSRuntime *rt, void *opaque);
JS_EXTERN void JS_SetInterruptHandler(JSRuntime *rt, JSInterruptHandler *cb, void *opaque);
/* if can_block is TRUE, Atomics.wait() can be used */
JS_EXTERN void JS_SetCanBlock(JSRuntime *rt, JS_BOOL can_block);
/* set the [IsHTMLDDA] internal slot */
JS_EXTERN void JS_SetIsHTMLDDA(JSContext *ctx, JSValue obj);

typedef struct JSModuleDef JSModuleDef;

/* return the module specifier (allocated with js_malloc()) or NULL if
   exception */
typedef char *JSModuleNormalizeFunc(JSContext *ctx,
                                    const char *module_base_name,
                                    const char *module_name, void *opaque);
typedef JSModuleDef *JSModuleLoaderFunc(JSContext *ctx,
                                        const char *module_name, void *opaque);

/* module_normalize = NULL is allowed and invokes the default module
   filename normalizer */
JS_EXTERN void JS_SetModuleLoaderFunc(JSRuntime *rt,
                                      JSModuleNormalizeFunc *module_normalize,
                                      JSModuleLoaderFunc *module_loader, void *opaque);
/* return the import.meta object of a module */
JS_EXTERN JSValue JS_GetImportMeta(JSContext *ctx, JSModuleDef *m);
JS_EXTERN JSAtom JS_GetModuleName(JSContext *ctx, JSModuleDef *m);
JSValue JS_GetModuleNamespace(JSContext *ctx, JSModuleDef *m);

/* JS Job support */

typedef JSValue JSJobFunc(JSContext *ctx, int argc, JSValue *argv);
JS_EXTERN int JS_EnqueueJob(JSContext *ctx, JSJobFunc *job_func, int argc, JSValue *argv);

JS_EXTERN JS_BOOL JS_IsJobPending(JSRuntime *rt);
JS_EXTERN int JS_ExecutePendingJob(JSRuntime *rt, JSContext **pctx);

/* Structure to retrieve (de)serialized SharedArrayBuffer objects. */
typedef struct JSSABTab {
    uint8_t **tab;
    size_t len;
} JSSABTab;

/* Object Writer/Reader (currently only used to handle precompiled code) */
#define JS_WRITE_OBJ_BYTECODE  (1 << 0) /* allow function/module */
#define JS_WRITE_OBJ_BSWAP     (0)      /* byte swapped output (obsolete, handled transparently) */
#define JS_WRITE_OBJ_SAB       (1 << 2) /* allow SharedArrayBuffer */
#define JS_WRITE_OBJ_REFERENCE (1 << 3) /* allow object references to encode arbitrary object graph */
#define JS_WRITE_OBJ_STRIP_SOURCE  (1 << 4) /* do not write source code information */
#define JS_WRITE_OBJ_STRIP_DEBUG   (1 << 5) /* do not write debug information */
JS_EXTERN uint8_t *JS_WriteObject(JSContext *ctx, size_t *psize, JSValue obj, int flags);
JS_EXTERN uint8_t *JS_WriteObject2(JSContext *ctx, size_t *psize, JSValue obj,
                                   int flags, JSSABTab *psab_tab);

#define JS_READ_OBJ_BYTECODE  (1 << 0) /* allow function/module */
#define JS_READ_OBJ_ROM_DATA  (0)      /* avoid duplicating 'buf' data (obsolete, broken by ICs) */
#define JS_READ_OBJ_SAB       (1 << 2) /* allow SharedArrayBuffer */
#define JS_READ_OBJ_REFERENCE (1 << 3) /* allow object references */
JS_EXTERN JSValue JS_ReadObject(JSContext *ctx, const uint8_t *buf, size_t buf_len, int flags);
JS_EXTERN JSValue JS_ReadObject2(JSContext *ctx, const uint8_t *buf, size_t buf_len,
                                 int flags, JSSABTab *psab_tab);
/* instantiate and evaluate a bytecode function. Only used when
   reading a script or module with JS_ReadObject() */
JS_EXTERN JSValue JS_EvalFunction(JSContext *ctx, JSValue fun_obj);
/* load the dependencies of the module 'obj'. Useful when JS_ReadObject()
   returns a module. */
JS_EXTERN int JS_ResolveModule(JSContext *ctx, JSValue obj);

/* only exported for os.Worker() */
JS_EXTERN JSAtom JS_GetScriptOrModuleName(JSContext *ctx, int n_stack_levels);
/* only exported for os.Worker() */
JS_EXTERN JSValue JS_LoadModule(JSContext *ctx, const char *basename,
                      const char *filename);

/* C function definition */
typedef enum JSCFunctionEnum {  /* XXX: should rename for namespace isolation */
    JS_CFUNC_generic,
    JS_CFUNC_generic_magic,
    JS_CFUNC_constructor,
    JS_CFUNC_constructor_magic,
    JS_CFUNC_constructor_or_func,
    JS_CFUNC_constructor_or_func_magic,
    JS_CFUNC_f_f,
    JS_CFUNC_f_f_f,
    JS_CFUNC_getter,
    JS_CFUNC_setter,
    JS_CFUNC_getter_magic,
    JS_CFUNC_setter_magic,
    JS_CFUNC_iterator_next,
} JSCFunctionEnum;

typedef union JSCFunctionType {
    JSCFunction *generic;
    JSValue (*generic_magic)(JSContext *ctx, JSValue this_val, int argc, JSValue *argv, int magic);
    JSCFunction *constructor;
    JSValue (*constructor_magic)(JSContext *ctx, JSValue new_target, int argc, JSValue *argv, int magic);
    JSCFunction *constructor_or_func;
    double (*f_f)(double);
    double (*f_f_f)(double, double);
    JSValue (*getter)(JSContext *ctx, JSValue this_val);
    JSValue (*setter)(JSContext *ctx, JSValue this_val, JSValue val);
    JSValue (*getter_magic)(JSContext *ctx, JSValue this_val, int magic);
    JSValue (*setter_magic)(JSContext *ctx, JSValue this_val, JSValue val, int magic);
    JSValue (*iterator_next)(JSContext *ctx, JSValue this_val,
                             int argc, JSValue *argv, int *pdone, int magic);
} JSCFunctionType;

JS_EXTERN JSValue JS_NewCFunction2(JSContext *ctx, JSCFunction *func,
                                   const char *name,
                                   int length, JSCFunctionEnum cproto, int magic);
JS_EXTERN JSValue JS_NewCFunctionData(JSContext *ctx, JSCFunctionData *func,
                                      int length, int magic, int data_len,
                                      JSValue *data);

static inline JSValue JS_NewCFunction(JSContext *ctx, JSCFunction *func, const char *name,
                                      int length)
{
    return JS_NewCFunction2(ctx, func, name, length, JS_CFUNC_generic, 0);
}

static inline JSValue JS_NewCFunctionMagic(JSContext *ctx, JSCFunctionMagic *func,
                                           const char *name,
                                           int length, JSCFunctionEnum cproto, int magic)
{
    /* Used to squelch a -Wcast-function-type warning. */
    JSCFunctionType ft;
    ft.generic_magic = func;
    return JS_NewCFunction2(ctx, ft.generic, name, length, cproto, magic);
}
JS_EXTERN void JS_SetConstructor(JSContext *ctx, JSValue func_obj,
                                 JSValue proto);

/* C property definition */

typedef struct JSCFunctionListEntry {
    const char *name;       /* pure ASCII or UTF-8 encoded */
    uint8_t prop_flags;
    uint8_t def_type;
    int16_t magic;
    union {
        struct {
            uint8_t length; /* XXX: should move outside union */
            uint8_t cproto; /* XXX: should move outside union */
            JSCFunctionType cfunc;
        } func;
        struct {
            JSCFunctionType get;
            JSCFunctionType set;
        } getset;
        struct {
            const char *name;
            int base;
        } alias;
        struct {
            const struct JSCFunctionListEntry *tab;
            int len;
        } prop_list;
        const char *str;    /* pure ASCII or UTF-8 encoded */
        int32_t i32;
        int64_t i64;
        double f64;
    } u;
} JSCFunctionListEntry;

#define JS_DEF_CFUNC          0
#define JS_DEF_CGETSET        1
#define JS_DEF_CGETSET_MAGIC  2
#define JS_DEF_PROP_STRING    3
#define JS_DEF_PROP_INT32     4
#define JS_DEF_PROP_INT64     5
#define JS_DEF_PROP_DOUBLE    6
#define JS_DEF_PROP_UNDEFINED 7
#define JS_DEF_OBJECT         8
#define JS_DEF_ALIAS          9

/* Note: c++ does not like nested designators */
#define JS_CFUNC_DEF(name, length, func1) { name, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE, JS_DEF_CFUNC, 0, { .func = { length, JS_CFUNC_generic, { .generic = func1 } } } }
#define JS_CFUNC_DEF2(name, length, func1, prop_flags) { name, prop_flags, JS_DEF_CFUNC, 0, { .func = { length, JS_CFUNC_generic, { .generic = func1 } } } }
#define JS_CFUNC_MAGIC_DEF(name, length, func1, magic) { name, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE, JS_DEF_CFUNC, magic, { .func = { length, JS_CFUNC_generic_magic, { .generic_magic = func1 } } } }
#define JS_CFUNC_SPECIAL_DEF(name, length, cproto, func1) { name, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE, JS_DEF_CFUNC, 0, { .func = { length, JS_CFUNC_ ## cproto, { .cproto = func1 } } } }
#define JS_ITERATOR_NEXT_DEF(name, length, func1, magic) { name, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE, JS_DEF_CFUNC, magic, { .func = { length, JS_CFUNC_iterator_next, { .iterator_next = func1 } } } }
#define JS_CGETSET_DEF(name, fgetter, fsetter) { name, JS_PROP_CONFIGURABLE, JS_DEF_CGETSET, 0, { .getset = { .get = { .getter = fgetter }, .set = { .setter = fsetter } } } }
#define JS_CGETSET_DEF2(name, fgetter, fsetter, prop_flags) { name, prop_flags, JS_DEF_CGETSET, 0, { .getset = { .get = { .getter = fgetter }, .set = { .setter = fsetter } } } }
#define JS_CGETSET_MAGIC_DEF(name, fgetter, fsetter, magic) { name, JS_PROP_CONFIGURABLE, JS_DEF_CGETSET_MAGIC, magic, { .getset = { .get = { .getter_magic = fgetter }, .set = { .setter_magic = fsetter } } } }
#define JS_PROP_STRING_DEF(name, cstr, prop_flags) { name, prop_flags, JS_DEF_PROP_STRING, 0, { .str = cstr } }
#define JS_PROP_INT32_DEF(name, val, prop_flags) { name, prop_flags, JS_DEF_PROP_INT32, 0, { .i32 = val } }
#define JS_PROP_INT64_DEF(name, val, prop_flags) { name, prop_flags, JS_DEF_PROP_INT64, 0, { .i64 = val } }
#define JS_PROP_DOUBLE_DEF(name, val, prop_flags) { name, prop_flags, JS_DEF_PROP_DOUBLE, 0, { .f64 = val } }
#define JS_PROP_UNDEFINED_DEF(name, prop_flags) { name, prop_flags, JS_DEF_PROP_UNDEFINED, 0, { .i32 = 0 } }
#define JS_OBJECT_DEF(name, tab, len, prop_flags) { name, prop_flags, JS_DEF_OBJECT, 0, { .prop_list = { tab, len } } }
#define JS_ALIAS_DEF(name, from) { name, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE, JS_DEF_ALIAS, 0, { .alias = { from, -1 } } }
#define JS_ALIAS_BASE_DEF(name, from, base) { name, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE, JS_DEF_ALIAS, 0, { .alias = { from, base } } }

JS_EXTERN void JS_SetPropertyFunctionList(JSContext *ctx, JSValue obj,
                                          const JSCFunctionListEntry *tab,
                                          int len);

/* C module definition */

typedef int JSModuleInitFunc(JSContext *ctx, JSModuleDef *m);

JS_EXTERN JSModuleDef *JS_NewCModule(JSContext *ctx, const char *name_str,
                                     JSModuleInitFunc *func);
/* can only be called before the module is instantiated */
JS_EXTERN int JS_AddModuleExport(JSContext *ctx, JSModuleDef *m, const char *name_str);
JS_EXTERN int JS_AddModuleExportList(JSContext *ctx, JSModuleDef *m,
                                      const JSCFunctionListEntry *tab, int len);
/* can only be called after the module is instantiated */
JS_EXTERN int JS_SetModuleExport(JSContext *ctx, JSModuleDef *m, const char *export_name,
                                 JSValue val);
JS_EXTERN int JS_SetModuleExportList(JSContext *ctx, JSModuleDef *m,
                                     const JSCFunctionListEntry *tab, int len);

/* Version */

#define QJS_VERSION_MAJOR 0
#define QJS_VERSION_MINOR 7
#define QJS_VERSION_PATCH 0
#define QJS_VERSION_SUFFIX ""

JS_EXTERN const char* JS_GetVersion(void);

/* Integration point for quickjs-libc.c, not for public use. */
JS_EXTERN uintptr_t js_std_cmd(int cmd, ...);

#undef JS_EXTERN
#undef js_force_inline
#undef __js_printf_like

#ifdef __cplusplus
} /* extern "C" { */
#endif

#endif /* QUICKJS_H */
