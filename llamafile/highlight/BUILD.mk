#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMAFILE_HIGHLIGHT

LLAMAFILE_HIGHLIGHT_FILES := $(wildcard llamafile/highlight/*)
LLAMAFILE_HIGHLIGHT_HDRS = $(filter %.h,$(LLAMAFILE_HIGHLIGHT_FILES))
LLAMAFILE_HIGHLIGHT_INCS = $(filter %.inc,$(LLAMAFILE_HIGHLIGHT_FILES))
LLAMAFILE_HIGHLIGHT_SRCS_C = $(filter %.c,$(LLAMAFILE_HIGHLIGHT_FILES))
LLAMAFILE_HIGHLIGHT_SRCS_CPP = $(filter %.cpp,$(LLAMAFILE_HIGHLIGHT_FILES))
LLAMAFILE_HIGHLIGHT_SRCS_GPERF = $(filter %.gperf,$(LLAMAFILE_HIGHLIGHT_FILES))
LLAMAFILE_HIGHLIGHT_SRCS_GPERF_C = $(LLAMAFILE_HIGHLIGHT_SRCS_GPERF:%.gperf=o/$(MODE)/%.c)

LLAMAFILE_HIGHLIGHT_SRCS =							\
	$(LLAMAFILE_HIGHLIGHT_SRCS_C)						\
	$(LLAMAFILE_HIGHLIGHT_SRCS_CPP)						\
	$(LLAMAFILE_HIGHLIGHT_SRCS_GPERF)					\

LLAMAFILE_HIGHLIGHT_OBJS =							\
	$(LLAMAFILE_HIGHLIGHT_SRCS_C:%.c=o/$(MODE)/%.o)				\
	$(LLAMAFILE_HIGHLIGHT_SRCS_CPP:%.cpp=o/$(MODE)/%.o)			\
	$(LLAMAFILE_HIGHLIGHT_SRCS_GPERF_C:%.c=%.o)				\

o/$(MODE)/llamafile/highlight/highlight.a: $(LLAMAFILE_HIGHLIGHT_OBJS)

$(LLAMAFILE_HIGHLIGHT_OBJS): llamafile/highlight/BUILD.mk

o/$(MODE)/llamafile/highlight/highlight_test:					\
		o/$(MODE)/llamafile/highlight/highlight_test.o			\
		o/$(MODE)/llamafile/highlight/highlight.a			\

o/$(MODE)/llamafile/highlight/highlight_c_test:					\
		o/$(MODE)/llamafile/highlight/highlight_c_test.o		\
		o/$(MODE)/llamafile/highlight/highlight_c.o			\
		o/$(MODE)/llamafile/highlight/is_keyword_c.o			\
		o/$(MODE)/llamafile/highlight/is_keyword_c_constant.o		\
		o/$(MODE)/llamafile/highlight/is_keyword_c_type.o		\
		o/$(MODE)/llamafile/highlight/is_keyword_c_pod.o		\
		o/$(MODE)/llamafile/highlight/is_keyword_cpp.o			\

o/$(MODE)/llamafile/highlight/highlight_python_test:				\
		o/$(MODE)/llamafile/highlight/highlight_python_test.o		\
		o/$(MODE)/llamafile/highlight/highlight_python.o		\
		o/$(MODE)/llamafile/highlight/is_keyword_python.o		\
		o/$(MODE)/llamafile/highlight/is_keyword_python_builtin.o	\
		o/$(MODE)/llamafile/highlight/is_keyword_python_constant.o	\

.PHONY: o/$(MODE)/llamafile/highlight
o/$(MODE)/llamafile/highlight:							\
		$(LLAMAFILE_HIGHLIGHT_SRCS_GPERF_C)				\
		o/$(MODE)/llamafile/highlight/highlight.a			\
		o/$(MODE)/llamafile/highlight/highlight_c_test.runs		\
		o/$(MODE)/llamafile/highlight/highlight_python_test.runs	\
		o/$(MODE)/llamafile/highlight/highlight_test.runs		\
