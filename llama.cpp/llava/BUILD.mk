#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_LLAVA

LLAMA_CPP_LLAVA_FILES := $(wildcard llama.cpp/llava/*)
LLAMA_CPP_LLAVA_HDRS = $(filter %.h,$(LLAMA_CPP_LLAVA_FILES))
LLAMA_CPP_LLAVA_SRCS = $(filter %.cpp,$(LLAMA_CPP_LLAVA_FILES))
LLAMA_CPP_LLAVA_OBJS = $(LLAMA_CPP_LLAVA_SRCS:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/llama.cpp/llava/llava.a:					\
		$(LLAMA_CPP_LLAVA_OBJS)

o/$(MODE)/llama.cpp/llava/llava-quantize:				\
		o/$(MODE)/llama.cpp/llava/llava-quantize.o		\
		o/$(MODE)/llama.cpp/llava/llava-quantize.1.asc.zip.o	\
		o/$(MODE)/llama.cpp/llava/llava.a			\
		o/$(MODE)/llama.cpp/llama.cpp.a				\
		o/$(MODE)/third_party/stb/stb.a

$(LLAMA_CPP_LLAVA_OBJS): llama.cpp/llava/BUILD.mk

.PHONY: o/$(MODE)/llama.cpp/llava
o/$(MODE)/llama.cpp/llava:						\
		o/$(MODE)/llama.cpp/llava/llava.a			\
		o/$(MODE)/llama.cpp/llava/llava-quantize
