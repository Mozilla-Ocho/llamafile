#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_IMATRIX

LLAMA_CPP_IMATRIX_FILES := $(wildcard llama.cpp/imatrix/*)
LLAMA_CPP_IMATRIX_HDRS = $(filter %.h,$(LLAMA_CPP_IMATRIX_FILES))
LLAMA_CPP_IMATRIX_SRCS = $(filter %.cpp,$(LLAMA_CPP_IMATRIX_FILES))
LLAMA_CPP_IMATRIX_OBJS = $(LLAMA_CPP_IMATRIX_SRCS:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/llama.cpp/imatrix/imatrix:					\
		o/$(MODE)/llama.cpp/imatrix/imatrix.o			\
		o/$(MODE)/llama.cpp/imatrix/imatrix.1.asc.zip.o		\
		o/$(MODE)/llama.cpp/llama.cpp.a

.PHONY: o/$(MODE)/llama.cpp/imatrix
o/$(MODE)/llama.cpp/imatrix:						\
		o/$(MODE)/llama.cpp/imatrix/imatrix
