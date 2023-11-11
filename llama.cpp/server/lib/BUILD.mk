#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PKGS += LLAMA_CPP_SERVER_LIB

LLAMA_CPP_SERVER_LIB_FILES := $(wildcard llama.cpp/server/lib/*)
LLAMA_CPP_SERVER_LIB_HDRS = $(filter %.h,$(LLAMA_CPP_SERVER_LIB_FILES))
LLAMA_CPP_SERVER_LIB_SRCS = $(filter %.cc,$(LLAMA_CPP_SERVER_LIB_FILES))
LLAMA_CPP_SERVER_LIB_OBJS = $(LLAMA_CPP_SERVER_LIB_SRCS:%.cc=o/$(MODE)/%.o)

.PHONY: tool/args/args.h

o/$(MODE)/llama.cpp/server/lib/lib.a: $(LLAMA_CPP_SERVER_LIB_OBJS)

.PHONY: o/$(MODE)/llama.cpp/server/lib
o/$(MODE)/llama.cpp/server/lib: 				\
		o/$(MODE)/llama.cpp/server/lib/lib.a
