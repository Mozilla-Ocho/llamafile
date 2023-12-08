#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PREFIX = /usr/local
COSMOCC = cosmocc/3.1.3
TOOLCHAIN = $(COSMOCC)/bin/cosmo

AR = $(TOOLCHAIN)ar
CC = $(TOOLCHAIN)cc
CXX = $(TOOLCHAIN)c++
ZIPOBJ = $(COSMOCC)/bin/zipobj
MKDEPS = $(COSMOCC)/bin/mkdeps
INSTALL = install

ARFLAGS = rcsD
CCFLAGS = -g -O3
TARGET_ARCH = -Xx86_64-mssse3
CPPFLAGS += -iquote. -mcosmo

TMPDIR = o//tmp
IGNORE := $(shell mkdir -p $(TMPDIR))

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
LC_ALL = C
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
distclean:; rm -rf o cosmocc

cosmocc/3.1.3:
	build/download-cosmocc.sh $@ 3.1.3 7470f05ef28f1941eb655c0359de08118023ba75767c2c47b398569a16397504
