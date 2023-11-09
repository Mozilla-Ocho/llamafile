#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

PREFIX = fatcosmo

AR = $(PREFIX)ar
CC = $(PREFIX)cc
CXX = $(PREFIX)c++

ARFLAGS = rcsD
CCFLAGS = -O3
CPPFLAGS = -iquote.

TMPDIR = o//tmp
IGNORE := $(shell mkdir -p $(TMPDIR))

ifneq ($(m),)
ifeq ($(MODE),)
MODE := $(m)
endif
endif

LC_ALL = C
SOURCE_DATE_EPOCH = 0
export TMPDIR
export LC_ALL
export SOURCE_DATE_EPOCH

.PHONY: all
all: o/$(MODE)/

.PHONY: o/$(MODE)/
o/$(MODE)/: o/$(MODE)/llama.cpp

.PHONY: clean
clean:; rm -rf o
