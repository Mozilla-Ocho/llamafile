#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += THIRD_PARTY_QUICKJS

THIRD_PARTY_QUICKJS_SRCS =					\
	third_party/quickjs/qjs.c				\
	third_party/quickjs/repl.c				\
	third_party/quickjs/libbf.c				\
	third_party/quickjs/cutils.c				\
	third_party/quickjs/quickjs.c				\
	third_party/quickjs/libregexp.c				\
	third_party/quickjs/libunicode.c				\
	third_party/quickjs/quickjs-libc.c				\

THIRD_PARTY_QUICKJS_HDRS =					\
	third_party/quickjs/qjs.h				\
	third_party/quickjs/list.h				\
	third_party/quickjs/libbf.h				\
	third_party/quickjs/cutils.h				\
	third_party/quickjs/quickjs.h				\
	third_party/quickjs/libregexp.h				\
	third_party/quickjs/libunicode.h				\
	third_party/quickjs/dirent_compat.h				\
	third_party/quickjs/quickjs-libc.h				\
	third_party/quickjs/quickjs-atom.h				\
	third_party/quickjs/quickjs-opcode.h				\
	third_party/quickjs/libregexp-opcode.h				\
	third_party/quickjs/quickjs-c-atomics.h				\
	third_party/quickjs/libunicode-table.h				\


o//third_party/quickjs/qjs.a \
o/$(MODE)/third_party/quickjs/qjs:				\
		o/$(MODE)/third_party/quickjs/qjs.o		\
		o/$(MODE)/third_party/quickjs/repl.o		\
		o/$(MODE)/third_party/quickjs/cutils.o	\
		o/$(MODE)/third_party/quickjs/libbf.o	\
		o/$(MODE)/third_party/quickjs/quickjs.o		\
		o/$(MODE)/third_party/quickjs/libregexp.o		\
		o/$(MODE)/third_party/quickjs/libunicode.o		\
		o/$(MODE)/third_party/quickjs/quickjs-libc.o		\

o//third_party/quickjs/qjsc.a \
o/$(MODE)/third_party/quickjs/qjsc:				\
		o/$(MODE)/third_party/quickjs/qjsc.o		\
		o/$(MODE)/third_party/quickjs/repl.o		\
		o/$(MODE)/third_party/quickjs/cutils.o	\
		o/$(MODE)/third_party/quickjs/libbf.o	\
		o/$(MODE)/third_party/quickjs/quickjs.o		\
		o/$(MODE)/third_party/quickjs/libregexp.o		\
		o/$(MODE)/third_party/quickjs/libunicode.o		\
		o/$(MODE)/third_party/quickjs/quickjs-libc.o		\

o//third_party/quickjs/quickjs.a: o//third_party/quickjs/quickjs.o

o/$(MODE)/third_party/quickjs/qjs.o				\
o/$(MODE)/third_party/quickjs/list.o				\
o/$(MODE)/third_party/quickjs/libbf.o				\
o/$(MODE)/third_party/quickjs/cutils.o				\
o/$(MODE)/third_party/quickjs/quickjs.o				\
o/$(MODE)/third_party/quickjs/libregexp.o				\
o/$(MODE)/third_party/quickjs/libunicode.o				\
o/$(MODE)/third_party/quickjs/quickjs-libc.o:				\
		private CFLAGS +=				\
			-mgcc					\

o/$(MODE)/third_party/quickjs/qjs.o				\
o/$(MODE)/third_party/quickjs/list.o				\
o/$(MODE)/third_party/quickjs/libbf.o				\
o/$(MODE)/third_party/quickjs/cutils.o				\
o/$(MODE)/third_party/quickjs/quickjs.o				\
o/$(MODE)/third_party/quickjs/libregexp.o				\
o/$(MODE)/third_party/quickjs/libunicode.o				\
o/$(MODE)/third_party/quickjs/quickjs-libc.o:				\
		third_party/quickjs/BUILD.mk



.PHONY: o/$(MODE)/third_party/quickjs
o/$(MODE)/third_party/quickjs:					\
		o/$(MODE)/third_party/sqlite/qjs		\
		o/$(MODE)/third_party/sqlite/qjsc		\
		o/$(MODE)/third_party/sqlite/quickjs.a		\
