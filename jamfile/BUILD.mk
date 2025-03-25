#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_JAMFILE

LLAMA_CPP_JAMFILE_FILES := $(wildcard jamfile/*)
LLAMA_CPP_JAMFILE_HDRS = $(filter %.h,$(LLAMA_CPP_JAMFILE_FILES))
LLAMA_CPP_JAMFILE_SRCS_C = $(filter %.c,$(LLAMA_CPP_JAMFILE_FILES))
LLAMA_CPP_JAMFILE_SRCS_CPP = $(filter %.cpp,$(LLAMA_CPP_JAMFILE_FILES))
LLAMA_CPP_JAMFILE_SRCS = $(LLAMA_CPP_JAMFILE_SRCS_C) $(LLAMA_CPP_JAMFILE_SRCS_CPP)

LLAMA_CPP_JAMFILE_OBJS = \
	$(LLAMA_CPP_JAMFILE_SRCS_C:%.c=o/$(MODE)/%.o) \
	$(LLAMA_CPP_JAMFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.o)


#o/$(MODE)/jamfile/jamfile.a: $(LLAMA_CPP_JAMFILE_SRCS_C)

### o/$(MODE)/jamfile/sqlite-vec.o: jamfile/sqlite-vec.c
### o/$(MODE)/jamfile/sqlite-vec.a: o/$(MODE)/jamfile/sqlite-vec.o
### 
### o/$(MODE)/jamfile/sqlite-csv.o: jamfile/sqlite-csv.c
### o/$(MODE)/jamfile/sqlite-csv.a: o/$(MODE)/jamfile/sqlite-csv.o
### 
### 
o/$(MODE)/jamfile/quickjs-llamafile-completion.o: jamfile/quickjs-llamafile-completion.cpp
o/$(MODE)/jamfile/quickjs-llamafile-completion.a: o/$(MODE)/jamfile/quickjs-llamafile-completion.o

o/$(MODE)/jamfile/quickjs-llamafile.o: jamfile/quickjs-llamafile.c jamfile/quickjs-llamafile-completion.cpp #o/$(MODE)/jamfile/quickjs-llamafile-completion.o
o/$(MODE)/jamfile/quickjs-llamafile.a: o/$(MODE)/jamfile/quickjs-llamafile.o o/$(MODE)/jamfile/quickjs-llamafile-completion.a
### 
### o/$(MODE)/jamfile/sqlite-lines.o: jamfile/sqlite-lines.c
### o/$(MODE)/jamfile/sqlite-lines.a: o/$(MODE)/jamfile/sqlite-lines.o
### 
### o/$(MODE)/jamfile/sqlite-lembed.o: jamfile/sqlite-lembed.c
### o/$(MODE)/jamfile/sqlite-lembed.a: o/$(MODE)/jamfile/sqlite-lembed.o o/$(MODE)/llama.cpp/llama.cpp.a


# jamfile:assert bytecode
o/$(MODE)/jamfile/assert.gen.c: jamfile/js_builtins/assert.js o/$(mode)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -o $@ $<
o/$(MODE)/jamfile/assert.gen.o: o/$(MODE)/jamfile/assert.gen.c

# jamfile:fmt bytecode
o/$(MODE)/jamfile/fmt.gen.c: jamfile/js_builtins/fmt.js o/$(mode)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -o $@ $<
o/$(MODE)/jamfile/fmt.gen.o: o/$(MODE)/jamfile/fmt.gen.c

# jamfile:cli bytecode
o/$(MODE)/jamfile/cli.gen.c: jamfile/js_builtins/cli.js o/$(MODE)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -o $@ $<
o/$(MODE)/jamfile/cli.gen.o: o/$(MODE)/jamfile/cli.gen.c

# jamfile:colors bytecode
o/$(MODE)/jamfile/colors.gen.c: jamfile/js_builtins/colors.js o/$(MODE)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -o $@ $<
o/$(MODE)/jamfile/colors.gen.o: o/$(MODE)/jamfile/colors.gen.c

# jamfile:zod bytecode
o/$(MODE)/jamfile/zod.gen.c: jamfile/js_builtins/zod.js o/$(mode)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -o $@ $<
o/$(MODE)/jamfile/zod.gen.o: o/$(MODE)/jamfile/zod.gen.c

# jamfile:yaml bytecode
o/$(MODE)/jamfile/yaml.gen.c: jamfile/js_builtins/yaml.js o/$(mode)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -o $@ $<
o/$(MODE)/jamfile/yaml.gen.o: o/$(MODE)/jamfile/yaml.gen.c


# jamfile:toml bytecode
o/$(MODE)/jamfile/toml.gen.c: jamfile/js_builtins/toml.js o/$(mode)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -o $@ $<
o/$(MODE)/jamfile/toml.gen.o: o/$(MODE)/jamfile/toml.gen.c


# jamfile:frontmatter bytecode
o/$(MODE)/jamfile/frontmatter.gen.c: jamfile/js_builtins/frontmatter.js o/$(mode)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -M jamfile:toml -M jamfile:yaml -o $@ $<
o/$(MODE)/jamfile/frontmatter.gen.o: o/$(MODE)/jamfile/frontmatter.gen.c

# jamfile:marked bytecode
o/$(MODE)/jamfile/marked.gen.c: jamfile/js_builtins/marked.js o/$(mode)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -o $@ $<
o/$(MODE)/jamfile/marked.gen.o: o/$(MODE)/jamfile/marked.gen.c

# jamfile:linkedom bytecode
o/$(MODE)/jamfile/linkedom.gen.c: jamfile/js_builtins/linkedom.js o/$(mode)/third_party/quickjs/qjsc
	o/$(mode)/third_party/quickjs/qjsc -m -o $@ $<
o/$(MODE)/jamfile/linkedom.gen.o: o/$(MODE)/jamfile/linkedom.gen.c

# include the raw contents of the jamfile.d.ts into jamfile for the `jamfile types` command
o/$(MODE)/jamfile/jamfile-types.c: jamfile/jamfile.d.ts
	xxd --include $< > $@
o/$(MODE)/jamfile/jamfile-types.o: o/$(MODE)/jamfile/jamfile-types.c

o/$(MODE)/jamfile/jamfile.a: \
	o/$(MODE)/jamfile/jamfile-types.o \
	o/$(MODE)/jamfile/cli.gen.o		\
	o/$(MODE)/jamfile/fmt.gen.o		\
	o/$(MODE)/jamfile/zod.gen.o		\
	o/$(MODE)/jamfile/assert.gen.o		\
	o/$(MODE)/jamfile/colors.gen.o		\
	o/$(MODE)/jamfile/toml.gen.o		\
	o/$(MODE)/jamfile/yaml.gen.o		\
	o/$(MODE)/jamfile/frontmatter.gen.o		\
	o/$(MODE)/jamfile/marked.gen.o		\
	o/$(MODE)/jamfile/linkedom.gen.o		\
	o/$(MODE)/jamfile/quickjs-sqlite.o		\
	o/$(MODE)/jamfile/quickjs-llamafile.o		\
	o/$(MODE)/jamfile/quickjs-llamafile-completion.o		\
	o/$(MODE)/third_party/quickjs/repl.o		\
	o/$(MODE)/third_party/quickjs/cutils.o	\
	o/$(MODE)/third_party/quickjs/libbf.o	\
	o/$(MODE)/third_party/quickjs/quickjs.o		\
	o/$(MODE)/third_party/quickjs/libregexp.o		\
	o/$(MODE)/third_party/quickjs/libunicode.o		\
	o/$(MODE)/third_party/quickjs/quickjs-libc.o


o/$(MODE)/jamfile/jamfile:					\
		o/$(MODE)/jamfile/jamfile.a \
		o/$(MODE)/jamfile/jamfile.1.asc.zip.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a \
		o/$(MODE)/third_party/sqlite/sqlite3.a \
		o/$(MODE)/jamfile/quickjs-sqlite.a \
		o/$(MODE)/jamfile/quickjs-llamafile.a \
		o/$(MODE)/third_party/sqlite/sqlite-csv.a	\
		o/$(MODE)/third_party/sqlite/sqlite-vec.a	

$(LLAMA_CPP_JAMFILE_OBJS): private CCFLAGS += -DSQLITE_CORE

.PHONY: o/$(MODE)/jamfile
o/$(MODE)/jamfile:						\
		o/$(MODE)/jamfile/jamfile

$(LLAMA_CPP_JAMFILE_OBJS): llama.cpp/BUILD.mk jamfile/BUILD.mk
.PHONY: jamfile-test-js

jamfile-test-js: o/$(MODE)/jamfile/jamfile jamfile/tests/js/test.js
	$< run jamfile/tests/js/test.js