#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_EMBEDFILE

LLAMA_CPP_EMBEDFILE_FILES := $(wildcard llama.cpp/embedfile/*)
LLAMA_CPP_EMBEDFILE_HDRS = $(filter %.h,$(LLAMA_CPP_EMBEDFILE_FILES))
LLAMA_CPP_EMBEDFILE_SRCS_C = $(filter %.c,$(LLAMA_CPP_EMBEDFILE_FILES))
LLAMA_CPP_EMBEDFILE_SRCS_CPP = $(filter %.cpp,$(LLAMA_CPP_EMBEDFILE_FILES))
LLAMA_CPP_EMBEDFILE_SRCS = $(LLAMA_CPP_EMBEDFILE_SRCS_C) $(LLAMA_CPP_EMBEDFILE_SRCS_CPP)

LLAMA_CPP_EMBEDFILE_OBJS = \
	$(LLAMA_CPP_EMBEDFILE_SRCS_C:%.c=o/$(MODE)/%.o) \
	$(LLAMA_CPP_EMBEDFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.o)


o/$(MODE)/llama.cpp/embedfile/embedfile.a: $(LLAMA_CPP_EMBEDFILE_SRCS_C)

o/$(MODE)/llama.cpp/embedfile/sqlite3.o: llama.cpp/embedfile/sqlite3.c
o/$(MODE)/llama.cpp/embedfile/sqlite3.a: o/$(MODE)/llama.cpp/embedfile/sqlite3.o

o/$(MODE)/llama.cpp/embedfile/sqlite-vec.o: llama.cpp/embedfile/sqlite-vec.c
o/$(MODE)/llama.cpp/embedfile/sqlite-vec.a: o/$(MODE)/llama.cpp/embedfile/sqlite-vec.o

o/$(MODE)/llama.cpp/embedfile/sqlite-csv.o: llama.cpp/embedfile/sqlite-csv.c
o/$(MODE)/llama.cpp/embedfile/sqlite-csv.a: o/$(MODE)/llama.cpp/embedfile/sqlite-csv.o

o/$(MODE)/llama.cpp/embedfile/sqlite-lines.o: llama.cpp/embedfile/sqlite-lines.c
o/$(MODE)/llama.cpp/embedfile/sqlite-lines.a: o/$(MODE)/llama.cpp/embedfile/sqlite-lines.o

o/$(MODE)/llama.cpp/embedfile/sqlite-lembed.o: llama.cpp/embedfile/sqlite-lembed.c
o/$(MODE)/llama.cpp/embedfile/sqlite-lembed.a: o/$(MODE)/llama.cpp/embedfile/sqlite-lembed.o o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/embedfile/shell.o: llama.cpp/embedfile/shell.c
o/$(MODE)/llama.cpp/embedfile/shell.a: o/$(MODE)/llama.cpp/embedfile/shell.o

#o/$(MODE)/llama.cpp/embedfile/embedfile.a: $(LLAMA_CPP_EMBEDFILE_OBJS)

#o/$(MODE)/llama.cpp/embedfile/sqlite3.o: private COPTS += -O3

o/$(MODE)/llama.cpp/embedfile/embedfile:					\
		o/$(MODE)/llama.cpp/embedfile/shell.a \
		o/$(MODE)/llama.cpp/embedfile/embedfile.o			\
		o/$(MODE)/llama.cpp/embedfile/embedfile.1.asc.zip.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a \
		o/$(MODE)/llama.cpp/embedfile/sqlite3.a \
		o/$(MODE)/llama.cpp/embedfile/sqlite-csv.a \
		o/$(MODE)/llama.cpp/embedfile/sqlite-vec.a \
		o/$(MODE)/llama.cpp/embedfile/sqlite-lines.a \
		o/$(MODE)/llama.cpp/embedfile/sqlite-lembed.a

$(LLAMA_CPP_EMBEDFILE_OBJS): private CCFLAGS += -DSQLITE_CORE

.PHONY: o/$(MODE)/llama.cpp/embedfile
o/$(MODE)/llama.cpp/embedfile:						\
		o/$(MODE)/llama.cpp/embedfile/embedfile

$(LLAMA_CPP_EMBEDFILE_OBJS): llama.cpp/BUILD.mk llama.cpp/embedfile/BUILD.mk
