#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMAFILE

LLAMAFILE_FILES := $(wildcard llamafile/*.*)
LLAMAFILE_HDRS = $(filter %.h,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS_C = $(filter %.c,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS_CU = $(filter %.cu,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS_CPP = $(filter %.cpp,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS = $(LLAMAFILE_SRCS_C) $(LLAMAFILE_SRCS_CPP) $(LLAMAFILE_SRCS_CU)
LLAMAFILE_DOCS = $(filter %.1,$(LLAMAFILE_FILES))

LLAMAFILE_OBJS =					\
	$(LLAMAFILE_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(LLAMAFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.o)	\
	$(LLAMAFILE_FILES:%=o/$(MODE)/%.zip.o)

o/$(MODE)/llamafile/zipalign:				\
		o/$(MODE)/llamafile/zipalign.o		\
		o/$(MODE)/llamafile/help.o		\
		o/$(MODE)/llamafile/has.o		\
		o/$(MODE)/llamafile/zipalign.1.asc.zip.o

o/$(MODE)/llamafile/zipcheck:				\
		o/$(MODE)/llamafile/zipcheck.o		\
		o/$(MODE)/llamafile/zip.o

o/$(MODE)/llamafile/simple:				\
		o/$(MODE)/llamafile/simple.o		\
		o/$(MODE)/llama.cpp/llava/llava.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a

.PHONY: o/$(MODE)/llamafile
o/$(MODE)/llamafile:					\
		$(LLAMAFILE_OBJS)			\
		o/$(MODE)/llamafile/simple		\
		o/$(MODE)/llamafile/zipalign		\
		o/$(MODE)/llamafile/zipcheck		\
		o/$(MODE)/llamafile/addnl

################################################################################
# testing

o/$(MODE)/llamafile/%.o: llamafile/%.cu llamafile/BUILD.mk
	@mkdir -p $(@D)
	build/cudacc -fPIE -g -O3 -march=native -c -o $@ $<

o/$(MODE)/llamafile/tinyblas_test:			\
		o/$(MODE)/llamafile/tinyblas_test.o	\
		o/$(MODE)/llamafile/tinyblas.o		\
		o/$(MODE)/llamafile/tester.o
	build/cudacc -g -o $@ $^ -lcublas

o/$(MODE)/llamafile/cudaprops:				\
		o/$(MODE)/llamafile/cudaprops.o		\
		o/$(MODE)/llamafile/tester.o
	build/cudacc -g -o $@ $^ -lcublas

o/$(MODE)/llamafile/pick_a_warp_kernel: private LDFLAGS += -fopenmp
o/$(MODE)/llamafile/pick_a_warp_kernel.o: private CFLAGS += -fopenmp

.PHONY: o/$(MODE)/llamafile/check
o/$(MODE)/llamafile/check:				\
		o/$(MODE)/llamafile/tinyblas_test.runs
