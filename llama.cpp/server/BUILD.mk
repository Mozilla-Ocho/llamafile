#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP_SERVER

LLAMA_CPP_SERVER_FILES := $(wildcard llama.cpp/server/*)
LLAMA_CPP_SERVER_ASSETS := $(wildcard llama.cpp/server/public/*)
LLAMA_CPP_SERVER_HDRS = $(filter %.h,$(LLAMA_CPP_SERVER_FILES))
LLAMA_CPP_SERVER_SRCS = $(filter %.cpp,$(LLAMA_CPP_SERVER_FILES))
LLAMA_CPP_SERVER_OBJS = $(LLAMA_CPP_SERVER_SRCS:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/llama.cpp/server/impl.o: private CXXFLAGS += -O1

o/$(MODE)/llama.cpp/server/server.a:				\
		$(LLAMA_CPP_SERVER_OBJS)

o/$(MODE)/llama.cpp/server/server.o: private			\
		CCFLAGS += -Os

$(LLAMA_CPP_SERVER_OBJS): llama.cpp/server/BUILD.mk

.PHONY: o/$(MODE)/llama.cpp/server
o/$(MODE)/llama.cpp/server:					\
		o/$(MODE)/llama.cpp/server/server.a
