#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PKGS += LLAMA_CPP_SERVER

LLAMA_CPP_SERVER_FILES := $(wildcard llama.cpp/server/*)
LLAMA_CPP_SERVER_HDRS = $(filter %.h,$(LLAMA_CPP_SERVER_FILES))
LLAMA_CPP_SERVER_SRCS = $(filter %.cc,$(LLAMA_CPP_SERVER_FILES))

.PHONY: tool/args/args.h

o/$(MODE)/llama.cpp/server/server:				\
		o/$(MODE)/llama.cpp/server/server.o		\
		o/$(MODE)/llama.cpp/llava/llava.a		\
		o/$(MODE)/llama.cpp/llama.cpp.a

.PHONY: o/$(MODE)/llama.cpp/server
o/$(MODE)/llama.cpp/server:					\
		o/$(MODE)/llama.cpp/server/server
