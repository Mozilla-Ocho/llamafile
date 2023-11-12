#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PKGS += LLAMA_CPP

LLAMA_CPP_FILES := $(wildcard llama.cpp/*.*)
LLAMA_CPP_HDRS = $(filter %.h,$(LLAMA_CPP_FILES))
LLAMA_CPP_SRCS_C = $(filter %.c,$(LLAMA_CPP_FILES))
LLAMA_CPP_SRCS_CC = $(filter %.cc,$(LLAMA_CPP_FILES))
LLAMA_CPP_SRCS = $(LLAMA_CPP_SRCS_C) $(LLAMA_CPP_SRCS_CC)

LLAMA_CPP_OBJS =					\
	$(LLAMA_CPP_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(LLAMA_CPP_SRCS_CC:%.cc=o/$(MODE)/%.o)		\
	$(LLAMA_CPP_FILES:%=o/$(MODE)/%.zip.o)

o/$(MODE)/llama.cpp/llama.cpp.a: $(LLAMA_CPP_OBJS)

include llama.cpp/main/BUILD.mk
include llama.cpp/llava/BUILD.mk
include llama.cpp/server/BUILD.mk
include llama.cpp/quantize/BUILD.mk
include llama.cpp/perplexity/BUILD.mk

.PHONY: o/$(MODE)/llama.cpp
o/$(MODE)/llama.cpp: 					\
		o/$(MODE)/llama.cpp/main		\
		o/$(MODE)/llama.cpp/llava		\
		o/$(MODE)/llama.cpp/server		\
		o/$(MODE)/llama.cpp/quantize		\
		o/$(MODE)/llama.cpp/perplexity
