#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

SHELL = /bin/sh
MAKEFLAGS += --no-builtin-rules

.SUFFIXES:
.DELETE_ON_ERROR:
.FEATURES: output-sync

include build/config.mk
include build/rules.mk

include llamafile/BUILD.mk
include llama.cpp/BUILD.mk

# the root package is `o//` by default
# building a package also builds its sub-packages
.PHONY: o/$(MODE)/
o/$(MODE)/: o/$(MODE)/llama.cpp o/$(MODE)/llamafile

# for installing to `make PREFIX=/usr/local`
.PHONY: install
install:	o/$(MODE)/llamafile/zipalign				\
		o/$(MODE)/llama.cpp/main/main				\
		o/$(MODE)/llama.cpp/server/server			\
		o/$(MODE)/llama.cpp/quantize/quantize			\
		o/$(MODE)/llama.cpp/perplexity/perplexity		\
		o/$(MODE)/llama.cpp/llava/llava-cli			\
		o/$(MODE)/llama.cpp/llava/llava-quantize
	$(INSTALL) o/$(MODE)/llamafile/zipalign $(PREFIX)/bin/zipalign
	$(INSTALL) o/$(MODE)/llama.cpp/main/main $(PREFIX)/bin/llamafile
	$(INSTALL) o/$(MODE)/llama.cpp/server/server $(PREFIX)/bin/llamafile-server
	$(INSTALL) o/$(MODE)/llama.cpp/quantize/quantize $(PREFIX)/bin/llamafile-quantize
	$(INSTALL) o/$(MODE)/llama.cpp/perplexity/perplexity $(PREFIX)/bin/llamafile-perplexity
	$(INSTALL) o/$(MODE)/llama.cpp/llava/llava-cli $(PREFIX)/bin/llamafile-llava-cli
	$(INSTALL) o/$(MODE)/llama.cpp/llava/llava-quantize $(PREFIX)/bin/llamafile-llava-quantize

include build/deps.mk
include build/tags.mk
