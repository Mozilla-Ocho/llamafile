#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

SHELL = /bin/sh
MAKEFLAGS += --no-builtin-rules

.SUFFIXES:
.DELETE_ON_ERROR:
.FEATURES: output-sync

include build/config.mk
include build/rules.mk

include llama.cpp/BUILD.mk

# the root package is `o//` by default
# building a package also builds its sub-packages
.PHONY: o/$(MODE)/
o/$(MODE)/: o/$(MODE)/llama.cpp

include build/deps.mk
include build/tags.mk
