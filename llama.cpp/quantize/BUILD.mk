#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_QUANTIZE

LLAMA_CPP_QUANTIZE_FILES := $(wildcard llama.cpp/quantize/*)
LLAMA_CPP_QUANTIZE_HDRS = $(filter %.h,$(LLAMA_CPP_QUANTIZE_FILES))
LLAMA_CPP_QUANTIZE_SRCS = $(filter %.cpp,$(LLAMA_CPP_QUANTIZE_FILES))
LLAMA_CPP_QUANTIZE_OBJS = $(LLAMA_CPP_QUANTIZE_SRCS:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/llama.cpp/quantize/quantize:					\
		o/$(MODE)/llama.cpp/quantize/quantize.o			\
		o/$(MODE)/llama.cpp/quantize/quantize.1.asc.zip.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a

.PHONY: o/$(MODE)/llama.cpp/quantize
o/$(MODE)/llama.cpp/quantize:						\
		o/$(MODE)/llama.cpp/quantize/quantize
