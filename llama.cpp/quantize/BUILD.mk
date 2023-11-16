#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PKGS += LLAMA_CPP_QUANTIZE

LLAMA_CPP_QUANTIZE_FILES := $(wildcard llama.cpp/quantize/*)
LLAMA_CPP_QUANTIZE_HDRS = $(filter %.h,$(LLAMA_CPP_QUANTIZE_FILES))
LLAMA_CPP_QUANTIZE_SRCS = $(filter %.cpp,$(LLAMA_CPP_QUANTIZE_FILES))
LLAMA_CPP_QUANTIZE_OBJS = $(LLAMA_CPP_QUANTIZE_SRCS:%.cpp=o/$(MODE)/%.o)

.PHONY: o/$(MODE)/llama.cpp/quantize
o/$(MODE)/llama.cpp/quantize:					\
		o/$(MODE)/llama.cpp/quantize/quantize

o/$(MODE)/llama.cpp/quantize/quantize:				\
		o/$(MODE)/llama.cpp/quantize/quantize.o		\
		o/$(MODE)/llama.cpp/llama.cpp.a
