#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_EMBEDFILE

LLAMA_CPP_EMBEDFILE_FILES := $(wildcard embedfile/*)
LLAMA_CPP_EMBEDFILE_HDRS = $(filter %.h,$(LLAMA_CPP_EMBEDFILE_FILES))
LLAMA_CPP_EMBEDFILE_SRCS_C = $(filter %.c,$(LLAMA_CPP_EMBEDFILE_FILES))
LLAMA_CPP_EMBEDFILE_SRCS_CPP = $(filter %.cpp,$(LLAMA_CPP_EMBEDFILE_FILES))
LLAMA_CPP_EMBEDFILE_SRCS = $(LLAMA_CPP_EMBEDFILE_SRCS_C) $(LLAMA_CPP_EMBEDFILE_SRCS_CPP)

LLAMA_CPP_EMBEDFILE_OBJS = \
	$(LLAMA_CPP_EMBEDFILE_SRCS_C:%.c=o/$(MODE)/%.o) \
	$(LLAMA_CPP_EMBEDFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.o)


o/$(MODE)/embedfile/embedfile.a: $(LLAMA_CPP_EMBEDFILE_SRCS_C)

o/$(MODE)/embedfile/sqlite-vec.o: embedfile/sqlite-vec.c
o/$(MODE)/embedfile/sqlite-vec.a: o/$(MODE)/embedfile/sqlite-vec.o

o/$(MODE)/embedfile/sqlite-csv.o: embedfile/sqlite-csv.c
o/$(MODE)/embedfile/sqlite-csv.a: o/$(MODE)/embedfile/sqlite-csv.o

o/$(MODE)/embedfile/sqlite-lines.o: embedfile/sqlite-lines.c
o/$(MODE)/embedfile/sqlite-lines.a: o/$(MODE)/embedfile/sqlite-lines.o

o/$(MODE)/embedfile/sqlite-lembed.o: embedfile/sqlite-lembed.c
o/$(MODE)/embedfile/sqlite-lembed.a: o/$(MODE)/embedfile/sqlite-lembed.o o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/embedfile/shell.o: embedfile/shell.c
o/$(MODE)/embedfile/shell.a: o/$(MODE)/embedfile/shell.o

#o/$(MODE)/embedfile/embedfile.a: $(LLAMA_CPP_EMBEDFILE_OBJS)

o/$(MODE)/embedfile/shell.o: private CFLAGS += \
	-DSQLITE_ENABLE_STMT_SCANSTATUS

o/$(MODE)/embedfile/embedfile:					\
		o/$(MODE)/embedfile/shell.a \
		o/$(MODE)/embedfile/embedfile.o			\
		o/$(MODE)/embedfile/embedfile.1.asc.zip.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a \
		o/$(MODE)/third_party/sqlite/sqlite3.a \
		o/$(MODE)/embedfile/sqlite-csv.a \
		o/$(MODE)/embedfile/sqlite-vec.a \
		o/$(MODE)/embedfile/sqlite-lines.a \
		o/$(MODE)/embedfile/sqlite-lembed.a

$(LLAMA_CPP_EMBEDFILE_OBJS): private CCFLAGS += -DSQLITE_CORE

.PHONY: o/$(MODE)/embedfile
o/$(MODE)/embedfile:						\
		o/$(MODE)/embedfile/embedfile

$(LLAMA_CPP_EMBEDFILE_OBJS): llama.cpp/BUILD.mk embedfile/BUILD.mk
