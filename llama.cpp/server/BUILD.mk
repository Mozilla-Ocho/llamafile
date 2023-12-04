#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PKGS += LLAMA_CPP_SERVER

LLAMA_CPP_SERVER_FILES := $(wildcard llama.cpp/server/*)
LLAMA_CPP_SERVER_ASSETS := $(wildcard llama.cpp/server/public/*)
LLAMA_CPP_SERVER_HDRS = $(filter %.h,$(LLAMA_CPP_SERVER_FILES))
LLAMA_CPP_SERVER_SRCS = $(filter %.cpp,$(LLAMA_CPP_SERVER_FILES))

o/$(MODE)/llama.cpp/server/server.o: private CXXFLAGS += -O1

o/$(MODE)/llama.cpp/server/server:				\
		o/$(MODE)/llama.cpp/server/server.o		\
		o/$(MODE)/llama.cpp/llava/llava.a		\
		o/$(MODE)/llama.cpp/llama.cpp.a			\
		$(LLAMA_CPP_SERVER_ASSETS:%=o/$(MODE)/%.zip.o)

.PHONY: o/$(MODE)/llama.cpp/server
o/$(MODE)/llama.cpp/server:					\
		o/$(MODE)/llama.cpp/server/server
