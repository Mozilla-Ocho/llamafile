#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_PERPLEXITY

LLAMA_CPP_PERPLEXITY_FILES := $(wildcard llama.cpp/perplexity/*)
LLAMA_CPP_PERPLEXITY_HDRS = $(filter %.h,$(LLAMA_CPP_PERPLEXITY_FILES))
LLAMA_CPP_PERPLEXITY_SRCS = $(filter %.cpp,$(LLAMA_CPP_PERPLEXITY_FILES))
LLAMA_CPP_PERPLEXITY_OBJS = $(LLAMA_CPP_PERPLEXITY_SRCS:%.cpp=o/$(MODE)/%.o)

.PHONY: o/$(MODE)/llama.cpp/perplexity
o/$(MODE)/llama.cpp/perplexity:						\
		o/$(MODE)/llama.cpp/perplexity/perplexity

o/$(MODE)/llama.cpp/perplexity/perplexity:				\
		o/$(MODE)/llama.cpp/perplexity/perplexity.o		\
		o/$(MODE)/llama.cpp/perplexity/perplexity.1.asc.zip.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a
