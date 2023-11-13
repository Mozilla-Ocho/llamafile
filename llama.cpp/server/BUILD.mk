#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PKGS += LLAMA_CPP_SERVER

LLAMA_CPP_SERVER_FILES := $(wildcard llama.cpp/server/*)
LLAMA_CPP_SERVER_HDRS = $(filter %.h,$(LLAMA_CPP_SERVER_FILES))
LLAMA_CPP_SERVER_SRCS = $(filter %.cc,$(LLAMA_CPP_SERVER_FILES))

LLAMA_CPP_SERVER_ASSETS :=					\
	$(wildcard llama.cpp/server/public/*)			\
	llama.cpp/server/.args

o/$(MODE)/llama.cpp/server/server:				\
		o/$(MODE)/llama.cpp/server/server.o		\
		o/$(MODE)/llama.cpp/server/lib/lib.a		\
		o/$(MODE)/llama.cpp/llava/llava.a		\
		o/$(MODE)/llama.cpp/llama.cpp.a			\
		$(LLAMA_CPP_SERVER_ASSETS:%=o/$(MODE)/%.zip.o)

# strip llama.cpp/server/ directory prefix off name in zip archive
o/$(MODE)/llama.cpp/server/.args.zip.o: private ZIPOBJ_FLAGS += -B

include llama.cpp/server/lib/BUILD.mk

.PHONY: o/$(MODE)/llama.cpp/server
o/$(MODE)/llama.cpp/server:					\
		o/$(MODE)/llama.cpp/server/server		\
		o/$(MODE)/llama.cpp/server/lib
