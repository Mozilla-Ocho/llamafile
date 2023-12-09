#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

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
install:	llamafile/zipalign.1					\
		llama.cpp/main/main.1					\
		llama.cpp/server/server.1				\
		llama.cpp/quantize/quantize.1				\
		llama.cpp/perplexity/perplexity.1			\
		llama.cpp/llava/llava-quantize.1			\
		o/$(MODE)/llamafile/zipalign				\
		o/$(MODE)/llama.cpp/main/main				\
		o/$(MODE)/llama.cpp/server/server			\
		o/$(MODE)/llama.cpp/quantize/quantize			\
		o/$(MODE)/llama.cpp/perplexity/perplexity		\
		o/$(MODE)/llama.cpp/llava/llava-quantize
	mkdir -p $(PREFIX)/bin
	$(INSTALL) o/$(MODE)/llamafile/zipalign $(PREFIX)/bin/zipalign
	$(INSTALL) o/$(MODE)/llama.cpp/main/main $(PREFIX)/bin/llamafile
	$(INSTALL) o/$(MODE)/llama.cpp/server/server $(PREFIX)/bin/llamafile-server
	$(INSTALL) o/$(MODE)/llama.cpp/quantize/quantize $(PREFIX)/bin/llamafile-quantize
	$(INSTALL) o/$(MODE)/llama.cpp/perplexity/perplexity $(PREFIX)/bin/llamafile-perplexity
	$(INSTALL) o/$(MODE)/llama.cpp/llava/llava-quantize $(PREFIX)/bin/llava-quantize
	mkdir -p $(PREFIX)/share/man/man1
	$(INSTALL) -m 0644 llamafile/zipalign.1 $(PREFIX)/share/man/man1/zipalign.1
	$(INSTALL) -m 0644 llama.cpp/main/main.1 $(PREFIX)/share/man/man1/llamafile.1
	$(INSTALL) -m 0644 llama.cpp/server/server.1 $(PREFIX)/share/man/man1/llamafile-server.1
	$(INSTALL) -m 0644 llama.cpp/quantize/quantize.1 $(PREFIX)/share/man/man1/llamafile-quantize.1
	$(INSTALL) -m 0644 llama.cpp/perplexity/perplexity.1 $(PREFIX)/share/man/man1/llamafile-perplexity.1
	$(INSTALL) -m 0644 llama.cpp/llava/llava-quantize.1 $(PREFIX)/share/man/man1/llava-quantize.1

include build/deps.mk
include build/tags.mk
