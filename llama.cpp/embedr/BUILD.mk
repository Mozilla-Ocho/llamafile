#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_EMBEDR

LLAMA_CPP_EMBEDR_FILES := $(wildcard llama.cpp/embedr/*)
LLAMA_CPP_EMBEDR_HDRS = $(filter %.h,$(LLAMA_CPP_EMBEDR_FILES))
LLAMA_CPP_EMBEDR_SRCS_C = $(filter %.c,$(LLAMA_CPP_EMBEDR_FILES))
LLAMA_CPP_EMBEDR_SRCS_CPP = $(filter %.cpp,$(LLAMA_CPP_EMBEDR_FILES))
LLAMA_CPP_EMBEDR_SRCS = $(LLAMA_CPP_EMBEDR_SRCS_C) $(LLAMA_CPP_EMBEDR_SRCS_CPP)

LLAMA_CPP_EMBEDR_OBJS = \
	$(LLAMA_CPP_EMBEDR_SRCS_C:%.c=o/$(MODE)/%.o) \
	$(LLAMA_CPP_EMBEDR_SRCS_CPP:%.cpp=o/$(MODE)/%.o)


o/$(MODE)/llama.cpp/embedr/embedr.a: $(LLAMA_CPP_EMBEDR_SRCS_C)

o/$(MODE)/llama.cpp/embedr/sqlite3.o: llama.cpp/embedr/sqlite3.c
o/$(MODE)/llama.cpp/embedr/sqlite3.a: o/$(MODE)/llama.cpp/embedr/sqlite3.o

o/$(MODE)/llama.cpp/embedr/sqlite-vec.o: llama.cpp/embedr/sqlite-vec.c
o/$(MODE)/llama.cpp/embedr/sqlite-vec.a: o/$(MODE)/llama.cpp/embedr/sqlite-vec.o

o/$(MODE)/llama.cpp/embedr/shell.o: llama.cpp/embedr/shell.c
o/$(MODE)/llama.cpp/embedr/shell.a: o/$(MODE)/llama.cpp/embedr/shell.o

#o/$(MODE)/llama.cpp/embedr/embedr.a: $(LLAMA_CPP_EMBEDR_OBJS)

#o/$(MODE)/llama.cpp/embedr/sqlite3.o: private COPTS += -O3

o/$(MODE)/llama.cpp/embedr/embedr:					\
		o/$(MODE)/llama.cpp/embedr/shell.a \
		o/$(MODE)/llama.cpp/embedr/embedr.o			\
		o/$(MODE)/llama.cpp/embedr/embedr.1.asc.zip.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a \
		o/$(MODE)/llama.cpp/embedr/sqlite3.a \
		o/$(MODE)/llama.cpp/embedr/sqlite-vec.a

$(LLAMA_CPP_EMBEDR_OBJS): private CCFLAGS += -DSQLITE_CORE

.PHONY: o/$(MODE)/llama.cpp/embedr
o/$(MODE)/llama.cpp/embedr:						\
		o/$(MODE)/llama.cpp/embedr/embedr

$(LLAMA_CPP_EMBEDR_OBJS): llama.cpp/BUILD.mk llama.cpp/embedr/BUILD.mk
