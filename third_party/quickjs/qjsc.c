/*
 * QuickJS command line compiler
 *
 * Copyright (c) 2018-2021 Fabrice Bellard
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
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#if defined(_MSC_VER)
#include "getopt_compat.h"
#else
#include <unistd.h>
#endif
#include <errno.h>

#include "cutils.h"
#include "quickjs-libc.h"

typedef enum {
    OUTPUT_C,
    OUTPUT_C_MAIN,
    OUTPUT_RAW,
} OutputTypeEnum;

typedef struct {
    char *name;
    char *short_name;
    int flags;
} namelist_entry_t;

typedef struct namelist_t {
    namelist_entry_t *array;
    int count;
    int size;
} namelist_t;

static namelist_t cname_list;
static namelist_t cmodule_list;
static namelist_t init_module_list;
static OutputTypeEnum output_type;
static FILE *outfile;
static const char *c_ident_prefix = "qjsc_";
static int strip;

void namelist_add(namelist_t *lp, const char *name, const char *short_name,
                  int flags)
{
    namelist_entry_t *e;
    if (lp->count == lp->size) {
        size_t newsize = lp->size + (lp->size >> 1) + 4;
        namelist_entry_t *a =
            realloc(lp->array, sizeof(lp->array[0]) * newsize);
        /* XXX: check for realloc failure */
        lp->array = a;
        lp->size = newsize;
    }
    e =  &lp->array[lp->count++];
    e->name = strdup(name);
    if (short_name)
        e->short_name = strdup(short_name);
    else
        e->short_name = NULL;
    e->flags = flags;
}

void namelist_free(namelist_t *lp)
{
    while (lp->count > 0) {
        namelist_entry_t *e = &lp->array[--lp->count];
        free(e->name);
        free(e->short_name);
    }
    free(lp->array);
    lp->array = NULL;
    lp->size = 0;
}

namelist_entry_t *namelist_find(namelist_t *lp, const char *name)
{
    int i;
    for(i = 0; i < lp->count; i++) {
        namelist_entry_t *e = &lp->array[i];
        if (!strcmp(e->name, name))
            return e;
    }
    return NULL;
}

static void get_c_name(char *buf, size_t buf_size, const char *file)
{
    const char *p, *r;
    size_t len, i;
    int c;
    char *q;

    p = strrchr(file, '/');
    if (!p)
        p = file;
    else
        p++;
    r = strrchr(p, '.');
    if (!r)
        len = strlen(p);
    else
        len = r - p;
    pstrcpy(buf, buf_size, c_ident_prefix);
    q = buf + strlen(buf);
    for(i = 0; i < len; i++) {
        c = p[i];
        if (!((c >= '0' && c <= '9') ||
              (c >= 'A' && c <= 'Z') ||
              (c >= 'a' && c <= 'z'))) {
            c = '_';
        }
        if ((q - buf) < buf_size - 1)
            *q++ = c;
    }
    *q = '\0';
}

static void dump_hex(FILE *f, const uint8_t *buf, size_t len)
{
    size_t i, col;
    col = 0;
    for(i = 0; i < len; i++) {
        fprintf(f, " 0x%02x,", buf[i]);
        if (++col == 8) {
            fprintf(f, "\n");
            col = 0;
        }
    }
    if (col != 0)
        fprintf(f, "\n");
}

static void output_object_code(JSContext *ctx,
                               FILE *fo, JSValue obj, const char *c_name,
                               BOOL load_only)
{
    uint8_t *out_buf;
    size_t out_buf_len;
    int flags = JS_WRITE_OBJ_BYTECODE;

    if (strip) {
        flags |= JS_WRITE_OBJ_STRIP_SOURCE;
        if (strip > 1)
            flags |= JS_WRITE_OBJ_STRIP_DEBUG;
    }

    out_buf = JS_WriteObject(ctx, &out_buf_len, obj, flags);
    if (!out_buf) {
        js_std_dump_error(ctx);
        exit(1);
    }

    namelist_add(&cname_list, c_name, NULL, load_only);

    if (output_type == OUTPUT_RAW) {
        fwrite(out_buf, 1, out_buf_len, fo);
    } else {
        fprintf(fo, "const uint32_t %s_size = %u;\n\n",
                c_name, (unsigned int)out_buf_len);
        fprintf(fo, "const uint8_t %s[%u] = {\n",
                c_name, (unsigned int)out_buf_len);
        dump_hex(fo, out_buf, out_buf_len);
        fprintf(fo, "};\n\n");
    }

    js_free(ctx, out_buf);
}

static int js_module_dummy_init(JSContext *ctx, JSModuleDef *m)
{
    /* should never be called when compiling JS code */
    abort();
    return -1; // pacify compiler
}

static void find_unique_cname(char *cname, size_t cname_size)
{
    char cname1[1024];
    int suffix_num;
    size_t len, max_len;
    assert(cname_size >= 32);
    /* find a C name not matching an existing module C name by
       adding a numeric suffix */
    len = strlen(cname);
    max_len = cname_size - 16;
    if (len > max_len)
        cname[max_len] = '\0';
    suffix_num = 1;
    for(;;) {
        snprintf(cname1, sizeof(cname1), "%s_%d", cname, suffix_num);
        if (!namelist_find(&cname_list, cname1))
            break;
        suffix_num++;
    }
    pstrcpy(cname, cname_size, cname1);
}

JSModuleDef *jsc_module_loader(JSContext *ctx,
                              const char *module_name, void *opaque)
{
    JSModuleDef *m;
    namelist_entry_t *e;

    /* check if it is a declared C or system module */
    e = namelist_find(&cmodule_list, module_name);
    if (e) {
        /* add in the static init module list */
        namelist_add(&init_module_list, e->name, e->short_name, 0);
        /* create a dummy module */
        m = JS_NewCModule(ctx, module_name, js_module_dummy_init);
    } else if (has_suffix(module_name, ".so")) {
        JS_ThrowReferenceError(ctx, "%s: dynamically linking to shared libraries not supported",
        module_name);
        return NULL;
    } else {
        size_t buf_len;
        uint8_t *buf;
        JSValue func_val;
        char cname[1024];

        buf = js_load_file(ctx, &buf_len, module_name);
        if (!buf) {
            JS_ThrowReferenceError(ctx, "could not load module filename '%s'",
                                   module_name);
            return NULL;
        }

        /* compile the module */
        func_val = JS_Eval(ctx, (char *)buf, buf_len, module_name,
                           JS_EVAL_TYPE_MODULE | JS_EVAL_FLAG_COMPILE_ONLY);
        js_free(ctx, buf);
        if (JS_IsException(func_val))
            return NULL;
        get_c_name(cname, sizeof(cname), module_name);
        if (namelist_find(&cname_list, cname)) {
            find_unique_cname(cname, sizeof(cname));
        }
        output_object_code(ctx, outfile, func_val, cname, TRUE);

        /* the module is already referenced, so we must free it */
        m = JS_VALUE_GET_PTR(func_val);
        JS_FreeValue(ctx, func_val);
    }
    return m;
}

static void compile_file(JSContext *ctx, FILE *fo,
                         const char *filename,
                         const char *script_name,
                         const char *c_name1,
                         int module)
{
    uint8_t *buf;
    char c_name[1024];
    int eval_flags;
    JSValue obj;
    size_t buf_len;

    buf = js_load_file(ctx, &buf_len, filename);
    if (!buf) {
        fprintf(stderr, "Could not load '%s'\n", filename);
        exit(1);
    }
    eval_flags = JS_EVAL_FLAG_COMPILE_ONLY;
    if (module < 0) {
        module = (has_suffix(filename, ".mjs") ||
                  JS_DetectModule((const char *)buf, buf_len));
    }
    if (module)
        eval_flags |= JS_EVAL_TYPE_MODULE;
    else
        eval_flags |= JS_EVAL_TYPE_GLOBAL;
    obj = JS_Eval(ctx, (const char *)buf, buf_len, script_name ? script_name : filename, eval_flags);
    if (JS_IsException(obj)) {
        js_std_dump_error(ctx);
        exit(1);
    }
    js_free(ctx, buf);
    if (c_name1) {
        pstrcpy(c_name, sizeof(c_name), c_name1);
    } else {
        get_c_name(c_name, sizeof(c_name), filename);
    }
    output_object_code(ctx, fo, obj, c_name, FALSE);
    JS_FreeValue(ctx, obj);
}

static const char main_c_template1[] =
    "int main(int argc, char **argv)\n"
    "{\n"
    "  JSRuntime *rt;\n"
    "  JSContext *ctx;\n"
    "  rt = JS_NewRuntime();\n"
    "  js_std_set_worker_new_context_func(JS_NewCustomContext);\n"
    "  js_std_init_handlers(rt);\n"
    ;

static const char main_c_template2[] =
    "  js_std_loop(ctx);\n"
    "  JS_FreeContext(ctx);\n"
    "  js_std_free_handlers(rt);\n"
    "  JS_FreeRuntime(rt);\n"
    "  return 0;\n"
    "}\n";

#define PROG_NAME "qjsc"

void help(void)
{
    printf("QuickJS-ng Compiler version %s\n"
           "usage: " PROG_NAME " [options] [files]\n"
           "\n"
           "options are:\n"
           "-b          output raw bytecode instead of C code\n"
           "-e          output main() and bytecode in a C file\n"
           "-o output   set the output filename\n"
           "-n script_name    set the script name (as used in stack traces)\n"
           "-N cname    set the C name of the generated data\n"
           "-m          compile as Javascript module (default=autodetect)\n"
           "-D module_name         compile a dynamically loaded module or worker\n"
           "-M module_name[,cname] add initialization code for an external C module\n"
           "-p prefix   set the prefix of the generated C names\n"
           "-s          strip the source code, specify twice to also strip debug info\n"
           "-S n        set the maximum stack size to 'n' bytes (default=%d)\n",
           JS_GetVersion(),
           JS_DEFAULT_STACK_SIZE);
    exit(1);
}

int main(int argc, char **argv)
{
    int c, i, verbose;
    const char *out_filename, *cname, *script_name;
    char cfilename[1024];
    FILE *fo;
    JSRuntime *rt;
    JSContext *ctx;
    int module;
    size_t stack_size;
    namelist_t dynamic_module_list;

    out_filename = NULL;
    script_name = NULL;
    output_type = OUTPUT_C;
    cname = NULL;
    module = -1;
    verbose = 0;
    strip = 0;
    stack_size = 0;
    memset(&dynamic_module_list, 0, sizeof(dynamic_module_list));

    
    /* add system modules */
    namelist_add(&cmodule_list, "qjs:std", "std", 0);
    namelist_add(&cmodule_list, "qjs:os", "os", 0);
    namelist_add(&cmodule_list, "qjs:bjson", "bjson", 0);
    namelist_add(&cmodule_list, "std", "std", 0);
    namelist_add(&cmodule_list, "os", "os", 0);
    namelist_add(&cmodule_list, "bjson", "bjson", 0);

    for(;;) {
        c = getopt(argc, argv, "ho:N:mn:bxesvM:p:S:D:");
        if (c == -1)
            break;
        switch(c) {
        case 'h':
            help();
        case 'b':
            output_type = OUTPUT_RAW;
            break;
        case 'o':
            out_filename = optarg;
            break;
        case 'e':
            output_type = OUTPUT_C_MAIN;
            break;
        case 'n':
            script_name = optarg;
            break;
        case 'N':
            cname = optarg;
            break;
        case 'm':
            module = 1;
            break;
        case 'M':
            {
                char *p;
                char path[1024];
                char cname[1024];
                pstrcpy(path, sizeof(path), optarg);
                p = strchr(path, ',');
                if (p) {
                    *p = '\0';
                    pstrcpy(cname, sizeof(cname), p + 1);
                } else {
                    get_c_name(cname, sizeof(cname), path);
                }
                namelist_add(&cmodule_list, path, cname, 0);
            }
            break;
        case 'D':
            namelist_add(&dynamic_module_list, optarg, NULL, 0);
            break;
        case 's':
            strip++;
            break;
        case 'v':
            verbose++;
            break;
        case 'p':
            c_ident_prefix = optarg;
            break;
        case 'S':
            stack_size = (size_t)strtod(optarg, NULL);
            break;
        default:
            break;
        }
    }

    if (optind >= argc)
        help();

    if (!out_filename)
        out_filename = "out.c";

    pstrcpy(cfilename, sizeof(cfilename), out_filename);

    if (output_type == OUTPUT_RAW)
        fo = fopen(cfilename, "wb");
    else
        fo = fopen(cfilename, "w");

    if (!fo) {
        perror(cfilename);
        exit(1);
    }
    outfile = fo;

    rt = JS_NewRuntime();
    ctx = JS_NewContext(rt);

    /* loader for ES6 modules */
    JS_SetModuleLoaderFunc(rt, NULL, jsc_module_loader, NULL);

    if (output_type != OUTPUT_RAW) {
        fprintf(fo, "/* File generated automatically by the QuickJS-ng compiler. */\n"
                "\n"
                );
    }

    if (output_type == OUTPUT_C_MAIN) {
        fprintf(fo, "#include \"quickjs-libc.h\"\n"
                "\n"
                );
    } else if (output_type == OUTPUT_C) {
        fprintf(fo, "#include <inttypes.h>\n"
                "\n"
                );
    }

    for(i = optind; i < argc; i++) {
        const char *filename = argv[i];
        compile_file(ctx, fo, filename, script_name, cname, module);
        cname = NULL;
    }

    for(i = 0; i < dynamic_module_list.count; i++) {
        if (!jsc_module_loader(ctx, dynamic_module_list.array[i].name, NULL)) {
            fprintf(stderr, "Could not load dynamic module '%s'\n",
                    dynamic_module_list.array[i].name);
            exit(1);
        }
    }

    if (output_type == OUTPUT_C_MAIN) {
        fprintf(fo,
                "static JSContext *JS_NewCustomContext(JSRuntime *rt)\n"
                "{\n"
                "  JSContext *ctx = JS_NewContext(rt);\n"
                "  if (!ctx)\n"
                "    return NULL;\n");
        /* add the precompiled modules (XXX: could modify the module
           loader instead) */
        for(i = 0; i < init_module_list.count; i++) {
            namelist_entry_t *e = &init_module_list.array[i];
            /* initialize the static C modules */

            fprintf(fo,
                    "  {\n"
                    "    extern JSModuleDef *js_init_module_%s(JSContext *ctx, const char *name);\n"
                    "    js_init_module_%s(ctx, \"%s\");\n"
                    "  }\n",
                    e->short_name, e->short_name, e->name);
        }
        for(i = 0; i < cname_list.count; i++) {
            namelist_entry_t *e = &cname_list.array[i];
            if (e->flags) {
                fprintf(fo, "  js_std_eval_binary(ctx, %s, %s_size, 1);\n",
                        e->name, e->name);
            }
        }
        fprintf(fo,
                "  return ctx;\n"
                "}\n\n");

        fputs(main_c_template1, fo);

        if (stack_size != 0) {
            fprintf(fo, "  JS_SetMaxStackSize(rt, %u);\n",
                    (unsigned int)stack_size);
        }

        /* add the module loader */
        fprintf(fo, "  JS_SetModuleLoaderFunc(rt, NULL, js_module_loader, NULL);\n");

        fprintf(fo,
                "  ctx = JS_NewCustomContext(rt);\n"
                "  js_std_add_helpers(ctx, argc, argv);\n");

        for(i = 0; i < cname_list.count; i++) {
            namelist_entry_t *e = &cname_list.array[i];
            if (!e->flags) {
                fprintf(fo, "  js_std_eval_binary(ctx, %s, %s_size, 0);\n",
                        e->name, e->name);
            }
        }
        fputs(main_c_template2, fo);
    }

    JS_FreeContext(ctx);
    JS_FreeRuntime(rt);

    fclose(fo);

    namelist_free(&cname_list);
    namelist_free(&cmodule_list);
    namelist_free(&init_module_list);
    return 0;
}
