#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_LLAMA-BENCH

LLAMA_CPP_LLAMA_BENCH_FILES := $(wildcard llama.cpp/llama-bench/*)
LLAMA_CPP_LLAMA_BENCH_SRCS = $(filter %.cpp,$(LLAMA_CPP_LLAMA_BENCH_FILES))
LLAMA_CPP_LLAMA_BENCH_OBJS = $(LLAMA_CPP_LLAMA_BENCH_SRCS:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/llama.cpp/llama-bench/llama-bench:				\
		o/$(MODE)/llama.cpp/llama-bench/llama-bench.o		\
		o/$(MODE)/llama.cpp/llama.cpp.a

$(LLAMA_CPP_LLAMA_BENCH_OBJS): llama.cpp/llama-bench/BUILD.mk

.PHONY: o/$(MODE)/llama.cpp/llama-bench
o/$(MODE)/llama.cpp/llama-bench:					\
		o/$(MODE)/llama.cpp/llama-bench/llama-bench
