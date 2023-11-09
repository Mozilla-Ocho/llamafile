#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PKGS += LLAMA_CPP

LLAMA_CPP = $(LLAMA_CPP_A)
LLAMA_CPP_A = o/$(MODE)/llama.cpp/llama.cpp.a
LLAMA_CPP_FILES = $(wildcard llama.cpp/*)
LLAMA_CPP_HDRS = $(filter %.h,$(LLAMA_CPP_FILES))
LLAMA_CPP_SRCS_C = $(filter %.c,$(LLAMA_CPP_FILES))
LLAMA_CPP_SRCS_CC = $(filter %.cc,$(LLAMA_CPP_FILES))

LLAMA_CPP_SRCS =					\
	$(LLAMA_CPP_SRCS_C)				\
	$(LLAMA_CPP_SRCS_CC)

LLAMA_CPP_OBJS =					\
	$(LLAMA_CPP_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(LLAMA_CPP_SRCS_CC:%.cc=o/$(MODE)/%.o)

LLAMA_CPP_BINS =					\
	o/$(MODE)/llama.cpp/main			\
	o/$(MODE)/llama.cpp/quantize			\
	o/$(MODE)/llama.cpp/perplexity

.PHONY: o/$(MODE)/llama.cpp
o/$(MODE)/llama.cpp: $(LLAMA_CPP_BINS)
o/$(MODE)/llama.cpp/llama.cpp.a: $(LLAMA_CPP_OBJS)

o/$(MODE)/llama.cpp/main:				\
		o/$(MODE)/llama.cpp/main.o		\
		o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/quantize:				\
		o/$(MODE)/llama.cpp/quantize.o		\
		o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/perplexity:				\
		o/$(MODE)/llama.cpp/perplexity.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a

.PHONY: tool/args/args.h
