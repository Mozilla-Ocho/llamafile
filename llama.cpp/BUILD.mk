#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP

LLAMA_CPP_FILES := $(wildcard llama.cpp/*.*)
LLAMA_CPP_HDRS = $(filter %.h,$(LLAMA_CPP_FILES))
LLAMA_CPP_INCS = $(filter %.inc,$(LLAMA_CPP_FILES))
LLAMA_CPP_SRCS_C = $(filter %.c,$(LLAMA_CPP_FILES))
LLAMA_CPP_SRCS_CPP = $(filter %.cpp,$(LLAMA_CPP_FILES))
LLAMA_CPP_SRCS = $(LLAMA_CPP_SRCS_C) $(LLAMA_CPP_SRCS_CPP)

LLAMA_CPP_OBJS =					\
	$(LLAMAFILE_OBJS)				\
	$(LLAMA_CPP_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(LLAMA_CPP_SRCS_CPP:%.cpp=o/$(MODE)/%.o)	\
	$(LLAMA_CPP_FILES:%=o/$(MODE)/%.zip.o)

o/$(MODE)/llama.cpp/llama.cpp.a: $(LLAMA_CPP_OBJS)

include llama.cpp/llava/BUILD.mk
include llama.cpp/server/BUILD.mk
include llama.cpp/main/BUILD.mk
include llama.cpp/imatrix/BUILD.mk
include llama.cpp/quantize/BUILD.mk
include llama.cpp/perplexity/BUILD.mk

$(LLAMA_CPP_OBJS): private CCFLAGS += -DGGML_MULTIPLATFORM -DGGML_USE_LLAMAFILE

o/$(MODE)/llama.cpp/ggml-alloc.o			\
o/$(MODE)/llama.cpp/ggml-backend.o			\
o/$(MODE)/llama.cpp/grammar-parser.o			\
o/$(MODE)/llama.cpp/json-schema-to-grammar.o		\
o/$(MODE)/llama.cpp/llama.o				\
o/$(MODE)/llama.cpp/stb_image.o				\
o/$(MODE)/llama.cpp/unicode.o				\
o/$(MODE)/llama.cpp/sampling.o				\
o/$(MODE)/llama.cpp/ggml-alloc.o			\
o/$(MODE)/llama.cpp/common.o: private			\
		CCFLAGS += -Os

$(LLAMA_CPP_OBJS): llama.cpp/BUILD.mk

.PHONY: o/$(MODE)/llama.cpp
o/$(MODE)/llama.cpp: 					\
		o/$(MODE)/llama.cpp/main		\
		o/$(MODE)/llama.cpp/llava		\
		o/$(MODE)/llama.cpp/server		\
		o/$(MODE)/llama.cpp/imatrix		\
		o/$(MODE)/llama.cpp/quantize		\
		o/$(MODE)/llama.cpp/perplexity
