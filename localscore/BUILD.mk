#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LOCALSCORE

LOCALSCORE_FILES := $(wildcard localscore/*.*)
LOCALSCORE_INCS = $(filter %.inc,$(LOCALSCORE_FILES))
LOCALSCORE_SRCS_C = $(filter %.c,$(LOCALSCORE_FILES))
LOCALSCORE_SRCS_CPP = $(filter %.cpp,$(LOCALSCORE_FILES))
LOCALSCORE_SRCS = $(LOCALSCORE_SRCS_C) $(LOCALSCORE_SRCS_CPP)

LOCALSCORE_HDRS =					\
	$(filter %.h,$(LOCALSCORE_FILES))		\
	$(filter %.hpp,$(LOCALSCORE_FILES))

LOCALSCORE_OBJS =					\
	$(LOCALSCORE_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(LOCALSCORE_SRCS_CPP:%.cpp=o/$(MODE)/%.o)

# Build the library
o/$(MODE)/localscore/localscore.a: 		\
		$(LOCALSCORE_OBJS)				\

# Any specific compiler flags needed
$(LOCALSCORE_OBJS): private				\
		CXXFLAGS +=				\
			-frtti				\
			-Wno-deprecated-declarations

# Main executable with mbedtls dependency
o/$(MODE)/localscore/localscore:				\
		o/$(MODE)/localscore/main.o		\
		o/$(MODE)/localscore/localscore.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\
		o/$(MODE)/third_party/mbedtls/mbedtls.a	\

$(LOCALSCORE_OBJS): localscore/BUILD.mk

.PHONY: o/$(MODE)/localscore
o/$(MODE)/localscore:					\
		o/$(MODE)/localscore/localscore		\
