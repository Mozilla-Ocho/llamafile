#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMAFILE

LLAMAFILE_FILES := $(wildcard llamafile/*.*)
LLAMAFILE_HDRS = $(filter %.h,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS = $(filter %.c,$(LLAMAFILE_FILES))
LLAMAFILE_DOCS = $(filter %.1,$(LLAMAFILE_FILES))

LLAMAFILE_OBJS =					\
	$(LLAMAFILE_SRCS:%.c=o/$(MODE)/%.o)		\
	$(LLAMAFILE_FILES:%=o/$(MODE)/%.zip.o)

o/$(MODE)/llamafile/zipalign:				\
		o/$(MODE)/llamafile/zipalign.o		\
		o/$(MODE)/llamafile/help.o		\
		o/$(MODE)/llamafile/has.o		\
		o/$(MODE)/llamafile/zipalign.1.asc.zip.o

o/$(MODE)/llamafile/zipcheck:				\
		o/$(MODE)/llamafile/zipcheck.o		\
		o/$(MODE)/llamafile/zip.o

.PHONY: o/$(MODE)/llamafile
o/$(MODE)/llamafile:					\
		$(LLAMAFILE_OBJS)			\
		o/$(MODE)/llamafile/zipalign		\
		o/$(MODE)/llamafile/zipcheck		\
		o/$(MODE)/llamafile/addnl
