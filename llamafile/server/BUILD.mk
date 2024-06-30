#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMAFILE_SERVER

LLAMAFILE_SERVER_FILES := $(wildcard llamafile/server/*)
LLAMAFILE_SERVER_HDRS = $(filter %.h,$(LLAMAFILE_SERVER_FILES))
LLAMAFILE_SERVER_INCS = $(filter %.inc,$(LLAMAFILE_SERVER_FILES))
LLAMAFILE_SERVER_SRCS = $(filter %.cpp,$(LLAMAFILE_SERVER_FILES))
LLAMAFILE_SERVER_OBJS = $(LLAMAFILE_SERVER_SRCS:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/llamafile/server/server.a:				\
		$(filter-out %_test.o,$(LLAMAFILE_SERVER_OBJS))

o/$(MODE)/llamafile/server/main:				\
		o/$(MODE)/llamafile/server/main.o		\
		o/$(MODE)/llamafile/server/server.a		\
		o/$(MODE)/llama.cpp/llama.cpp.a			\
		o/$(MODE)/llama.cpp/llava/llava.a		\
		o/$(MODE)/double-conversion/double-conversion.a	\
		o/$(MODE)/stb/stb.a				\

$(LLAMAFILE_SERVER_OBJS): llamafile/server/BUILD.mk
$(LLAMAFILE_SERVER_OBJS): private CCFLAGS += -O

o/$(MODE)/llamafile/server/json_test:				\
		o/$(MODE)/llamafile/server/json_test.o		\
		o/$(MODE)/llamafile/server/json.o		\
		o/$(MODE)/double-conversion/double-conversion.a	\

.PHONY: o/$(MODE)/llamafile/server
o/$(MODE)/llamafile/server:					\
		o/$(MODE)/llamafile/server/main			\
		o/$(MODE)/llamafile/server/json_test.runs	\
