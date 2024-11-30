#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += THIRD_PARTY_SQLITE

THIRD_PARTY_SQLITE_SRCS =					\
	third_party/sqlite/sqlite3.c				\
	third_party/sqlite/shell.c				\

THIRD_PARTY_SQLITE_HDRS =					\
	third_party/sqlite/sqlite3.h				\

o/$(MODE)/third_party/sqlite/sqlite3.a:				\
		o/$(MODE)/third_party/sqlite/sqlite3.o		\

o/$(MODE)/third_party/sqlite/shell:				\
		o/$(MODE)/third_party/sqlite/shell.o		\
		o/$(MODE)/third_party/sqlite/sqlite3.o		\

o/$(MODE)/third_party/sqlite/shell.o				\
o/$(MODE)/third_party/sqlite/sqlite3.o:				\
		private CFLAGS +=				\
			-mgcc					\
			-DSQLITE_CORE				\
			-DSQLITE_OS_UNIX			\
			-DHAVE_USLEEP				\
			-DHAVE_READLINK				\
			-DHAVE_FCHOWN				\
			-DHAVE_LOCALTIME_R			\
			-DHAVE_LSTAT				\
			-DHAVE_GMTIME_R				\
			-DHAVE_FDATASYNC			\
			-DHAVE_STRCHRNUL			\
			-DHAVE_LOCALTIME_R			\
			-DHAVE_MALLOC_USABLE_SIZE		\
			-DSQLITE_HAVE_C99_MATH_FUNCS		\
			-DSQLITE_ENABLE_STMT_SCANSTATUS		\
			-DSQLITE_ENABLE_FTS5			\
			-DSQLITE_ENABLE_RTREE			\
			-DSQLITE_SOUNDEX			\
			-DSQLITE_ENABLE_GEOPOLY			\
			-DSQLITE_ENABLE_MATH_FUNCTIONS		\
			-USQLITE_ENABLE_FTS3			\
			-DSQLITE_ENABLE_FTS5			\
			-DSQLITE_ENABLE_DBSTAT_VTAB		\
			-DSQLITE_ENABLE_DBPAGE_VTAB		\
			-DSQLITE_ENABLE_STMTVTAB		\
			-DSQLITE_ENABLE_BYTECODE_VTAB		\
			-DSQLITE_ENABLE_EXPLAIN_COMMENTS	\
			-DSQLITE_HAVE_ZLIB			\
			-DSQLITE_INTROSPECTION_PRAGMAS		\
			-DSQLITE_ENABLE_UNKNOWN_SQL_FUNCTION	\
			-DSQLITE_ENABLE_STMT_SCANSTATUS		\
			-DSQLITE_DQS=0				\

o/$(MODE)/third_party/sqlite/shell.o				\
o/$(MODE)/third_party/sqlite/sqlite3.o:				\
		third_party/sqlite/BUILD.mk

.PHONY: o/$(MODE)/third_party/sqlite
o/$(MODE)/third_party/sqlite:					\
		o/$(MODE)/third_party/sqlite/shell		\
		o/$(MODE)/third_party/sqlite/sqlite3.a		\
