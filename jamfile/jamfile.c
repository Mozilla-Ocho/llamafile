// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#include "third_party/quickjs/cutils.h"
#include "third_party/quickjs/quickjs.h"
#include "third_party/quickjs/quickjs-libc.h"

#include "llama.cpp/llama.h"
#include "llamafile/version.h"
#include "third_party/sqlite/sqlite3.h"
#include <string.h>

#include <assert.h>
#include <cosmo.h>
#include <stdlib.h>
#include <time.h>


#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#if !defined(_MSC_VER)
#include <unistd.h>
#endif
#include <errno.h>
#include <fcntl.h>
#include <time.h>

#include "jamfile/jamfile.h"
#include "third_party/sqlite/sqlite3.h"

#include "jamfile/quickjs-sqlite.h"
#include "jamfile/quickjs-llamafile.h"
#include "jamfile/quickjs-llamafile-completion.h"
#include "llamafile/llamafile.h"

char *JAMFILE_DEFAULT_EMBEDDING_MODEL = NULL;
char *JAMFILE_DEFAULT_COMPLETION_MODEL = NULL;

// raw jamfile.d.ts contents, for the `jamfile types`
extern const unsigned char jamfile_jamfile_d_ts[];
extern const unsigned int jamfile_jamfile_d_ts_len;

// pre-compiled JS bytecode for the interactive REPL, adapted from quickjs project
extern const uint8_t qjsc_repl[];
extern const uint32_t qjsc_repl_size;

// pre-compiled JS bytecode for jamfile:colors
extern const uint8_t qjsc_colors[];
extern const uint32_t qjsc_colors_size;

// pre-compiled JS bytecode for jamfile:cli
extern const uint8_t qjsc_cli[];
extern const uint32_t qjsc_cli_size;

// pre-compiled JS bytecode for jamfile:zod
extern const uint8_t qjsc_zod[];
extern const uint32_t qjsc_zod_size;

// pre-compiled JS bytecode for jamfile:fmt
extern const uint8_t qjsc_fmt[];
extern const uint32_t qjsc_fmt_size;

// pre-compiled JS bytecode for jamfile:assert
extern const uint8_t qjsc_assert[];
extern const uint32_t qjsc_assert_size;

// pre-compiled JS bytecode for jamfile:toml
extern const uint8_t qjsc_toml[];
extern const uint32_t qjsc_toml_size;

// pre-compiled JS bytecode for jamfile:yaml
extern const uint8_t qjsc_yaml[];
extern const uint32_t qjsc_yaml_size;

// pre-compiled JS bytecode for jamfile:frontmatter
extern const uint8_t qjsc_frontmatter[];
extern const uint32_t qjsc_frontmatter_size;

// pre-compiled JS bytecode for jamfile:marked
extern const uint8_t qjsc_marked[];
extern const uint32_t qjsc_marked_size;



// pre-compiled JS bytecode for jamfile:linkedom
extern const uint8_t qjsc_linkedom[];
extern const uint32_t qjsc_linkedom_size;

static JSCFunctionListEntry argv0;

// Initially from quickjs-lib.c, but include fallback to llamafile_open
uint8_t *jama_load_file(JSContext *ctx, size_t *pbuf_len, const char *filename)
{
    struct llamafile *f;
    uint8_t *buf;
    size_t buf_len;
    long lret;

    f = llamafile_open_xxx(filename, "rb");
    if (!f)
        return NULL;
    if (!llamafile_seek(f, 0, SEEK_END))
        goto fail;
    lret = llamafile_tell(f);
    if (lret < 0)
        goto fail;
    /* XXX: on Linux, ftell() return LONG_MAX for directories */
    if (lret == LONG_MAX) {
        errno = EISDIR;
        goto fail;
    }
    buf_len = lret;
    if (!llamafile_seek(f, 0, SEEK_SET))
        goto fail;
    if (ctx)
        buf = js_malloc(ctx, buf_len + 1);
    else
        buf = malloc(buf_len + 1);
    if (!buf)
        goto fail;
    if (llamafile_read(f, buf, buf_len) != buf_len) {
        errno = EIO;
        if (ctx)
            js_free(ctx, buf);
        else
            free(buf);
    fail:
        llamafile_close(f);
        return NULL;
    }
    buf[buf_len] = '\0';
    llamafile_close(f);
    *pbuf_len = buf_len;
    return buf;
}

static int eval_buf(JSContext *ctx, const void *buf, int buf_len,
                    const char *filename, int eval_flags)
{
    JSValue val;
    int ret;

    if ((eval_flags & JS_EVAL_TYPE_MASK) == JS_EVAL_TYPE_MODULE) {
        /* for the modules, we compile then run to be able to set
           import.meta */
        val = JS_Eval(ctx, buf, buf_len, filename,
                      eval_flags | JS_EVAL_FLAG_COMPILE_ONLY);
        if (!JS_IsException(val)) {
            js_module_set_import_meta(ctx, val, TRUE, TRUE);
            val = JS_EvalFunction(ctx, val);
        }
        val = js_std_await(ctx, val);
    } else {
        val = JS_Eval(ctx, buf, buf_len, filename, eval_flags);
    }
    if (JS_IsException(val)) {
        js_std_dump_error(ctx);
        ret = -1;
    } else {
        ret = 0;
    }
    JS_FreeValue(ctx, val);
    return ret;
}

static int eval_file(JSContext *ctx, const char *filename, int module)
{
    uint8_t *buf;
    int ret, eval_flags;
    size_t buf_len;

    buf = jama_load_file(ctx, &buf_len, filename);
    if (!buf) {
        perror(filename);
        exit(1);
    }

    if (module < 0) {
        module = (has_suffix(filename, ".mjs") ||
                  JS_DetectModule((const char *)buf, buf_len));
    }
    if (module)
        eval_flags = JS_EVAL_TYPE_MODULE;
    else
        eval_flags = JS_EVAL_TYPE_GLOBAL;
    ret = eval_buf(ctx, buf, buf_len, filename, eval_flags);
    js_free(ctx, buf);
    return ret;
}

static JSValue js_gc(JSContext *ctx, JSValue this_val,
                     int argc, JSValue *argv)
{
    JS_RunGC(JS_GetRuntime(ctx));
    return JS_UNDEFINED;
}

static const JSCFunctionListEntry global_obj[] = {
    JS_CFUNC_DEF("gc", 0, js_gc),
};


void jamfile_define_jamfile_global(JSContext *ctx, int argc, char ** argv) {
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue Jamfile = JS_NewObject(ctx);
  
  JS_SetPropertyStr(ctx, Jamfile, "version", JS_NewString(ctx, JAMFILE_VERSION));
  
  JSValue args = JS_NewArray(ctx);
  for(int i = 0; i < argc; i++) {
      JS_SetPropertyUint32(ctx, args, i, JS_NewString(ctx, argv[i]));
  }
  JS_SetPropertyStr(ctx, Jamfile, "args", args);

  JS_SetPropertyStr(ctx, global, "Jamfile", Jamfile);
  JS_FreeValue(ctx, global);
}

static JSContext *jamfile_context_new(JSRuntime *rt)
{
    JSContext *ctx;
    ctx = JS_NewContext(rt);
    if (!ctx)
        return NULL;
    
    js_init_module_sqlite(ctx, "jamfile:sqlite");
    js_init_module_llamafile(ctx, "jamfile:llamafile", JAMFILE_DEFAULT_EMBEDDING_MODEL, JAMFILE_DEFAULT_COMPLETION_MODEL);
    js_init_module_std(ctx, "qjs:std");
    js_init_module_os(ctx, "qjs:os");
    js_init_module_bjson(ctx, "qjs:bjson");

    JSValue global = JS_GetGlobalObject(ctx);

    JSValue console = JS_NewObject(ctx);
    JS_SetPropertyStr(ctx, console, "log",
                      JS_NewCFunction(ctx, js_print, "log", 1));
    JS_SetPropertyStr(ctx, global, "console", console);

    JS_SetPropertyFunctionList(ctx, global, global_obj, countof(global_obj));
    JS_SetPropertyFunctionList(ctx, global, &argv0, 1);
    JS_FreeValue(ctx, global);

    return ctx;
}

JSModuleDef *jama_module_loader(JSContext *ctx,
                              const char *module_name, void *opaque)
{
  struct {
      char * name;
      uint8_t *bytecode;
      uint32_t bytecode_size;
    } jama_builtin_modules[] = {
      {"jamfile:assert", (uint8_t *) qjsc_assert,  qjsc_assert_size  },
      {"jamfile:colors", (uint8_t *) qjsc_colors,  qjsc_colors_size  },
      {"jamfile:cli",    (uint8_t *) qjsc_cli,     qjsc_cli_size     },
      {"jamfile:fmt",    (uint8_t *) qjsc_fmt,     qjsc_fmt_size     },
      {"jamfile:zod",    (uint8_t *) qjsc_zod,     qjsc_zod_size     },
      {"jamfile:yaml",    (uint8_t *) qjsc_yaml,     qjsc_yaml_size     },
      {"jamfile:toml",        (uint8_t *) qjsc_toml,        qjsc_toml_size        },
      {"jamfile:frontmatter", (uint8_t *) qjsc_frontmatter, qjsc_frontmatter_size },
      {"jamfile:marked",      (uint8_t *) qjsc_marked,      qjsc_marked_size      },
      {"jamfile:linkedom",    (uint8_t *) qjsc_linkedom,    qjsc_linkedom_size    },
    };
    JSModuleDef *m;

    size_t buf_len;
    uint8_t *buf;
    JSValue func_val;
    for(int i = 0; i < countof(jama_builtin_modules); i++) {
      if(strncmp(module_name, jama_builtin_modules[i].name, strlen(module_name)) == 0) {
        JSValue obj = JS_ReadObject(ctx, jama_builtin_modules[i].bytecode, jama_builtin_modules[i].bytecode_size, JS_READ_OBJ_BYTECODE);

        assert(!JS_IsException(obj));
        assert(JS_VALUE_GET_TAG(obj) == JS_TAG_MODULE);

        JSModuleDef *m = JS_VALUE_GET_PTR(obj);
        JS_FreeValue(ctx, obj);
        return m;
      }
    }

    buf = jama_load_file(ctx, &buf_len, module_name);
    if (!buf) {
        JS_ThrowReferenceError(ctx, "could not load module filename '%s'",
                                module_name);
        return NULL;
    }

    if(sqlite3_strlike("%.gbnf", module_name, 0)==0
        || sqlite3_strlike("%.sql", module_name, 0)==0
        || sqlite3_strlike("%.txt", module_name, 0)==0) {

        sqlite3_str * s = sqlite3_str_new(NULL);
        sqlite3_str_appendall(s, "export default `");
        for(int i = 0; i < buf_len; i++) {
            // TODO also escape ${}"
            if(buf[i] == '`') {
                sqlite3_str_appendall(s, "\\`");
            }
            else {
                sqlite3_str_appendchar(s, 1, buf[i]);
            }
        }
        sqlite3_str_appendall(s, "`;");
        int moduleLength = sqlite3_str_length(s);
        char * module = sqlite3_str_finish(s);
        assert(module);
        func_val = JS_Eval(ctx, module, moduleLength, module_name, JS_EVAL_TYPE_MODULE | JS_EVAL_FLAG_COMPILE_ONLY);
        js_free(ctx, buf);
        sqlite3_free((void *) module);
    }
    else {
        func_val = JS_Eval(ctx, (char *)buf, buf_len, module_name,
                        JS_EVAL_TYPE_MODULE | JS_EVAL_FLAG_COMPILE_ONLY);
        js_free(ctx, buf);
    }

    
    if (JS_IsException(func_val))
        return NULL;
    /* XXX: could propagate the exception */
    js_module_set_import_meta(ctx, func_val, TRUE, FALSE);
    /* the module is already referenced, so we must free it */
    m = JS_VALUE_GET_PTR(func_val);
    JS_FreeValue(ctx, func_val);

    return m;
}

#define PROG_NAME "jamfile"

void help(void)
{
    printf("Jamfile version %s, QuickJS-ng version %s\n"
           "usage: " PROG_NAME " [options] [file [args]]\n"
           "-h  --help         list options\n"
           "-e  --eval EXPR    evaluate EXPR\n"
           "-i  --interactive  go to interactive mode\n"
           "-m  --module       load as ES6 module (default=autodetect)\n"
           "    --script       load as ES6 script (default=autodetect)\n"
           "    --memory-limit n       limit the memory usage to 'n' Kbytes\n"
           "    --stack-size n         limit the stack size to 'n' Kbytes\n"
           "    --unhandled-rejection  dump unhandled promise rejections\n"
           "-q  --quit         just instantiate the interpreter and quit\n", JAMFILE_VERSION, JS_GetVersion());
    exit(1);
}

int cmd_run(int argc, char ** argv) {
    JSRuntime *rt;
    JSContext *ctx;
    JSValue ret;
    int optind;
    char *expr = NULL;
    int interactive = 0;
    int module = -1;
    int dump_unhandled_promise_rejection = 0;
    int64_t memory_limit = -1;
    int64_t stack_size = -1;

    argv0 = (JSCFunctionListEntry)JS_PROP_STRING_DEF("argv0", argv[0],
                                                     JS_PROP_C_W_E);

    optind = 1;
    while (optind < argc && *argv[optind] == '-') {
        char *arg = argv[optind] + 1;
        const char *longopt = "";
        char *opt_arg = NULL;
        /* a single - is not an option, it also stops argument scanning */
        if (!*arg)
            break;
        optind++;
        if (*arg == '-') {
            longopt = arg + 1;
            opt_arg = strchr(longopt, '=');
            if (opt_arg)
                *opt_arg++ = '\0';
            arg += strlen(arg);
            /* -- stops argument scanning */
            if (!*longopt)
                break;
        }
        for (; *arg || *longopt; longopt = "") {
            char opt = *arg;
            if (opt) {
                arg++;
                if (!opt_arg && *arg)
                    opt_arg = arg;
            }
            if (opt == 'h' || opt == '?' || !strcmp(longopt, "help")) {
                help();
                continue;
            }
            if (opt == 'e' || !strcmp(longopt, "eval")) {
                if (!opt_arg) {
                    if (optind >= argc) {
                        fprintf(stderr, "jamfile: missing expression for -e\n");
                        exit(2);
                    }
                    opt_arg = argv[optind++];
                }
                expr = opt_arg;
                break;
            }
            if (opt == 'i' || !strcmp(longopt, "interactive")) {
                interactive++;
                continue;
            }
            if (opt == 'm' || !strcmp(longopt, "module")) {
                module = 1;
                continue;
            }
            if (!strcmp(longopt, "script")) {
                module = 0;
                continue;
            }
            if (!strcmp(longopt, "unhandled-rejection")) {
                dump_unhandled_promise_rejection = 1;
                continue;
            }
            if (opt) {
                fprintf(stderr, "jamfile: unknown option '-%c'\n", opt);
            } else {
                fprintf(stderr, "jamfile: unknown option '--%s'\n", longopt);
            }
            help();
        }
    }


    rt = JS_NewRuntime();

    if (!rt) {
        fprintf(stderr, "jamfile: cannot allocate JS runtime\n");
        exit(2);
    }

    JS_SetRuntimeOpaque(rt, (void *) 0x420);

    js_std_set_worker_new_context_func(jamfile_context_new);
    js_std_init_handlers(rt);
    ctx = jamfile_context_new(rt);
    if (!ctx) {
        fprintf(stderr, "jamfile: cannot allocate JS context\n");
        exit(2);
    }

    // -1 and +1 to read past the js file arg
    jamfile_define_jamfile_global(ctx, argc - optind - 1, argv + optind + 1);

    JS_SetModuleLoaderFunc(rt, NULL, jama_module_loader, NULL);

    if (dump_unhandled_promise_rejection) {
        JS_SetHostPromiseRejectionTracker(rt, js_std_promise_rejection_tracker,
                                          NULL);
    }

    //js_std_add_helpers(ctx, argc - optind, argv + optind);

    if (expr) {
        if (eval_buf(ctx, expr, strlen(expr), "<cmdline>", 0))
            goto fail;
    } else
    if (optind >= argc) {
        /* interactive mode */
        interactive = 1;
    } else {
        const char *filename;
        filename = argv[optind];
        if (eval_file(ctx, filename, module))
            goto fail;
    }
    if (interactive) {
        js_std_eval_binary(ctx, qjsc_repl, qjsc_repl_size, 0);
    }
    ret = js_std_loop(ctx);
    if (!JS_IsUndefined(ret)) {
        js_std_dump_error1(ctx, ret);
        goto fail;
    }

    js_std_free_handlers(rt);
    JS_FreeContext(ctx);
    JS_FreeRuntime(rt);


    return 0;
 fail:
    js_std_free_handlers(rt);
    JS_FreeContext(ctx);
    JS_FreeRuntime(rt);
    return 1;
}

int main(int argc, char **argv) {
    int rc;
    sqlite3 *db;
    sqlite3_stmt *stmt;

    FLAG_log_disable = 1;
    argc = cosmo_args("/zip/.args", &argv);
    FLAGS_READY = 1;

    char *modelPath = NULL;
    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];
        if (sqlite3_stricmp(arg, "--default-embedding-model") == 0) {
            assert(++i <= argc);
            JAMFILE_DEFAULT_EMBEDDING_MODEL = argv[i];
        }else if (sqlite3_stricmp(arg, "--default-completion-model") == 0) {
            assert(++i <= argc);
            JAMFILE_DEFAULT_COMPLETION_MODEL = argv[i];
        } else if (sqlite3_stricmp(arg, "--version") == 0 || sqlite3_stricmp(arg, "-v") == 0) {
            fprintf(stderr,"jamfile %s\n",JAMFILE_VERSION);
            return 0;
        } else if (sqlite3_stricmp(arg, "--help") == 0 || sqlite3_stricmp(arg, "-h") == 0) {
            llamafile_help("/zip/jamfile/jamfile.1.asc");
            fprintf(stderr, "Usage: jamfile [ run | types ]\n");
            return 0;
        } else if(sqlite3_stricmp(arg, "run") == 0) {
          return cmd_run(argc-i, argv+i);
        } else if(sqlite3_stricmp(arg, "types") == 0) {
          fprintf(stdout, "%.*s", jamfile_jamfile_d_ts_len, jamfile_jamfile_d_ts);
          return 0;
        }
        else {
            printf("Unknown arg %s\n", arg);
            return 1;
        }
    }

    fprintf(stderr, "Usage: jamfile [run | types]\n");
    return 0;
}
