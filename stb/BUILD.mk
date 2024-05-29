#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += STB

STB_FILES := $(wildcard stb/*)
STB_HDRS = $(filter %.h,$(STB_FILES))
STB_INCS = $(filter %.inc,$(STB_FILES))
STB_SRCS = $(filter %.c,$(STB_FILES))
STB_OBJS = $(STB_SRCS:%.c=o/$(MODE)/%.o)

o/$(MODE)/stb/stb.a: $(STB_OBJS)

$(STB_OBJS): stb/BUILD.mk

.PHONY: o/$(MODE)/stb
o/$(MODE)/stb: o/$(MODE)/stb/stb.a
