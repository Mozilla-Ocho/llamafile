#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PKGS += LLAMA_CPP_SERVER

LLAMA_CPP_SERVER_FILES := $(wildcard llama.cpp/server/*)
LLAMA_CPP_SERVER_HDRS = $(filter %.h,$(LLAMA_CPP_SERVER_FILES))
LLAMA_CPP_SERVER_SRCS = $(filter %.cc,$(LLAMA_CPP_SERVER_FILES))

LLAMA_CPP_SERVER_ASSETS :=					\
	$(wildcard llama.cpp/server/public/*)			\
	llama.cpp/server/.args

LLAMA_CPP_SERVER_LINKARGS =					\
	o/$(MODE)/llama.cpp/server/server.o			\
	o/$(MODE)/llama.cpp/llava/llava.a			\
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/server/server:				\
		$(LLAMA_CPP_SERVER_LINKARGS)			\
		$(LLAMA_CPP_SERVER_ASSETS)
	$(LINK.o) $(LLAMA_CPP_SERVER_LINKARGS) $(LOADLIBES) $(LDLIBS) -o $@.com
	cd llama.cpp/server && zip -q ../../$@.com $(LLAMA_CPP_SERVER_ASSETS:llama.cpp/server/%=%)
	mv -f $@.com $@

.PHONY: o/$(MODE)/llama.cpp/server
o/$(MODE)/llama.cpp/server:					\
		o/$(MODE)/llama.cpp/server/server
