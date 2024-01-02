#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_MAIN

LLAMA_CPP_MAIN_FILES := $(wildcard llama.cpp/main/*)
LLAMA_CPP_MAIN_HDRS = $(filter %.h,$(LLAMA_CPP_MAIN_FILES))
LLAMA_CPP_MAIN_SRCS = $(filter %.cpp,$(LLAMA_CPP_MAIN_FILES))
LLAMA_CPP_MAIN_OBJS = $(LLAMA_CPP_MAIN_SRCS:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/llama.cpp/main/main:					\
		o/$(MODE)/llama.cpp/main/main.o			\
		o/$(MODE)/llama.cpp/server/server.a		\
		o/$(MODE)/llama.cpp/llava/llava.a		\
		o/$(MODE)/llama.cpp/llama.cpp.a			\
		o/$(MODE)/llama.cpp/main/main.1.asc.zip.o	\
		$(LLAMA_CPP_SERVER_ASSETS:%=o/$(MODE)/%.zip.o)

llama.cpp/main/main.1.asc: llama.cpp/main/main.1
	-man $< >$@.tmp && mv -f $@.tmp $@
	@rm -f $@.tmp

.PHONY: o/$(MODE)/llama.cpp/main
o/$(MODE)/llama.cpp/main:					\
		o/$(MODE)/llama.cpp/main/main
