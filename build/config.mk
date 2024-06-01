#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PREFIX = /usr/local
COSMOCC = .cosmocc/3.3.10
TOOLCHAIN = $(COSMOCC)/bin/cosmo

AR = $(TOOLCHAIN)ar
CC = $(TOOLCHAIN)cc
CXX = $(TOOLCHAIN)c++
ZIPOBJ = $(COSMOCC)/bin/zipobj
MKDEPS = $(COSMOCC)/bin/mkdeps
INSTALL = install

ARFLAGS = rcsD
CCFLAGS = -g -O3 -fexceptions -fsignaling-nans
CPPFLAGS_ = -iquote. -mcosmo -DGGML_MULTIPLATFORM -Wno-attributes -DLLAMAFILE_DEBUG
TARGET_ARCH = -Xx86_64-mavx -Xx86_64-mtune=znver4

TMPDIR = o//tmp
IGNORE := $(shell mkdir -p $(TMPDIR))
ARCH := $(shell uname -m)

# apple still distributes a 17 year old version of gnu make
ifeq ($(MAKE_VERSION), 3.81)
$(error please use bin/make from cosmocc.zip rather than old xcode make)
endif

# let `make m=foo` be shorthand for `make MODE=foo`
ifneq ($(m),)
ifeq ($(MODE),)
MODE := $(m)
endif
endif

# make build more deterministic
LC_ALL = C.UTF-8
SOURCE_DATE_EPOCH = 0
export MODE
export TMPDIR
export LC_ALL
export SOURCE_DATE_EPOCH

# `make` runs `make all` by default
.PHONY: all
all: o/$(MODE)/

.PHONY: clean
clean:; rm -rf o

.PHONY: distclean
distclean:; rm -rf o .cosmocc

.cosmocc/3.3.10:
	build/download-cosmocc.sh $@ 3.3.10 00d61c1215667314f66e288c8285bae38cc6137fca083e5bba6c74e3a52439de
