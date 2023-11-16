#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PKGS += LLAMAFILE

LLAMAFILE_FILES := $(wildcard llamafile/*.*)
LLAMAFILE_HDRS = $(filter %.h,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS = $(filter %.c,$(LLAMAFILE_FILES))

LLAMAFILE_OBJS =					\
	$(LLAMAFILE_SRCS:%.c=o/$(MODE)/%.o)		\
	$(LLAMAFILE_FILES:%=o/$(MODE)/%.zip.o)

.PHONY: o/$(MODE)/llamafile
o/$(MODE)/llamafile: $(LLAMAFILE_OBJS)
