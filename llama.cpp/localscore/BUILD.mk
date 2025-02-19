#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_LOCALSCORE

LLAMA_CPP_LOCALSCORE_FILES := $(wildcard llama.cpp/localscore/*)
LLAMA_CPP_LOCALSCORE_HDRS = $(filter %.h,$(LLAMA_CPP_LOCALSCORE_FILES))
LLAMA_CPP_LOCALSCORE_SRCS = $(filter-out %/main.cpp,$(filter %.cpp,$(LLAMA_CPP_LOCALSCORE_FILES)))
LLAMA_CPP_LOCALSCORE_OBJS = $(LLAMA_CPP_LOCALSCORE_SRCS:%.cpp=o/$(MODE)/%.o)

MAIN_SRC = llama.cpp/localscore/main.cpp
MAIN_OBJ = $(MAIN_SRC:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/llama.cpp/localscore/localscore.a:				\
		$(LLAMA_CPP_LOCALSCORE_OBJS)

o/$(MODE)/llama.cpp/localscore/localscore:				\
		$(MAIN_OBJ)						\
		o/$(MODE)/llama.cpp/localscore/localscore.a		\
		o/$(MODE)/llama.cpp/llama.cpp.a				\
		o/$(MODE)/third_party/mbedtls/mbedtls.a

$(LLAMA_CPP_LOCALSCORE_OBJS) $(MAIN_OBJ): llama.cpp/localscore/BUILD.mk

.PHONY: o/$(MODE)/llama.cpp/localscore
o/$(MODE)/llama.cpp/localscore:							\
		o/$(MODE)/llama.cpp/localscore/localscore
