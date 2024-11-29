#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += STABLE_DIFFUSION_CPP

STABLE_DIFFUSION_CPP_FILES := $(wildcard stable-diffusion.cpp/*.*)
STABLE_DIFFUSION_CPP_HDRS = $(filter %.h,$(STABLE_DIFFUSION_CPP_FILES))	\
			    $(filter %.hpp,$(STABLE_DIFFUSION_CPP_FILES))
STABLE_DIFFUSION_CPP_INCS = $(filter %.inc,$(STABLE_DIFFUSION_CPP_FILES))
STABLE_DIFFUSION_CPP_SRCS_C = $(filter %.c,$(STABLE_DIFFUSION_CPP_FILES))
STABLE_DIFFUSION_CPP_SRCS_CPP = $(filter %.cpp,$(STABLE_DIFFUSION_CPP_FILES))
STABLE_DIFFUSION_CPP_SRCS = $(STABLE_DIFFUSION_CPP_SRCS_C) $(STABLE_DIFFUSION_CPP_SRCS_CPP)

STABLE_DIFFUSION_CPP_OBJS =						\
	$(STABLE_DIFFUSION_CPP_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(STABLE_DIFFUSION_CPP_SRCS_CPP:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/stable-diffusion.cpp/stable-diffusion.cpp.a: $(STABLE_DIFFUSION_CPP_OBJS)

$(STABLE_DIFFUSION_CPP_OBJS): private					\
		CCFLAGS +=						\
			-DGGML_MULTIPLATFORM

$(STABLE_DIFFUSION_CPP_OBJS): private					\
		CXXFLAGS +=						\
			-frtti						\
			-Wno-deprecated-declarations

o/$(MODE)/stable-diffusion.cpp/main:					\
		o/$(MODE)/stable-diffusion.cpp/main.o			\
		o/$(MODE)/stable-diffusion.cpp/stable-diffusion.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a				\
		o/$(MODE)/third_party/stb/stb.a

$(STABLE_DIFFUSION_CPP_OBJS): stable-diffusion.cpp/BUILD.mk

.PHONY: o/$(MODE)/stable-diffusion.cpp
o/$(MODE)/stable-diffusion.cpp:						\
		o/$(MODE)/stable-diffusion.cpp/main
