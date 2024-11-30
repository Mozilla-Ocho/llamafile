#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMAFILE_SERVER

LLAMAFILE_SERVER_FILES := $(wildcard llamafile/server/*)
LLAMAFILE_SERVER_HDRS = $(filter %.h,$(LLAMAFILE_SERVER_FILES))
LLAMAFILE_SERVER_INCS = $(filter %.inc,$(LLAMAFILE_SERVER_FILES))
LLAMAFILE_SERVER_SRCS = $(filter %.cpp,$(LLAMAFILE_SERVER_FILES))
LLAMAFILE_SERVER_OBJS = $(LLAMAFILE_SERVER_SRCS:%.cpp=o/$(MODE)/%.o)
LLAMAFILE_SERVER_ASSETS = $(wildcard llamafile/server/www/*)

$(LLAMAFILE_SERVER_OBJS): private CCFLAGS += -g

o/$(MODE)/llamafile/server/server.a:						\
		$(filter-out %_test.o,$(LLAMAFILE_SERVER_OBJS))

o/$(MODE)/llamafile/server/main:						\
		o/$(MODE)/llamafile/server/main.o				\
		o/$(MODE)/llamafile/server/main.1.asc.zip.o			\
		o/$(MODE)/llamafile/server/server.a				\
		o/$(MODE)/llama.cpp/llama.cpp.a					\
		o/$(MODE)/llama.cpp/llava/llava.a				\
		o/$(MODE)/third_party/double-conversion/double-conversion.a	\
		o/$(MODE)/third_party/stb/stb.a					\
		o/$(MODE)/third_party/sqlite/sqlite3.a				\
		$(LLAMAFILE_SERVER_ASSETS:%=o/$(MODE)/%.zip.o)			\

# turn /zip/llamafile/server/www/...
# into /zip/www/...
$(LLAMAFILE_SERVER_ASSETS:%=o/$(MODE)/%.zip.o): private ZIPOBJ_FLAGS += -C2

$(LLAMAFILE_SERVER_OBJS): llamafile/server/BUILD.mk

o/$(MODE)/llamafile/server/atom_test:						\
		o/$(MODE)/llamafile/server/atom_test.o				\
		o/$(MODE)/llamafile/server/atom.o				\
		o/$(MODE)/llamafile/server/image.o				\

o/$(MODE)/llamafile/server/image_test:						\
		o/$(MODE)/llamafile/server/image_test.o				\
		o/$(MODE)/llamafile/server/image.o				\

o/$(MODE)/llamafile/server/fastjson_test:					\
		o/$(MODE)/llamafile/server/fastjson_test.o			\
		o/$(MODE)/llamafile/server/fastjson.o				\
		o/$(MODE)/double-conversion/double-conversion.a			\

o/$(MODE)/llamafile/server/tokenbucket_test:					\
		o/$(MODE)/llamafile/server/tokenbucket_test.o			\
		o/$(MODE)/llamafile/server/tokenbucket.o			\
		o/$(MODE)/llamafile/server/log.o				\
		o/$(MODE)/llama.cpp/llama.cpp.a					\

.PHONY: o/$(MODE)/llamafile/server
o/$(MODE)/llamafile/server:							\
		o/$(MODE)/llamafile/server/main					\
		o/$(MODE)/llamafile/server/atom_test.runs			\
		o/$(MODE)/llamafile/server/fastjson_test.runs			\
		o/$(MODE)/llamafile/server/image_test.runs			\
		o/$(MODE)/llamafile/server/tokenbucket_test.runs		\
