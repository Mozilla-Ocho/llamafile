#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += THIRD_PARTY_MBEDTLS

THIRD_PARTY_MBEDTLS_ARTIFACTS += THIRD_PARTY_MBEDTLS_A
THIRD_PARTY_MBEDTLS = $(THIRD_PARTY_MBEDTLS_A_DEPS) $(THIRD_PARTY_MBEDTLS_A)
THIRD_PARTY_MBEDTLS_A = o/$(MODE)/third_party/mbedtls/mbedtls.a
THIRD_PARTY_MBEDTLS_A_FILES := $(wildcard third_party/mbedtls/*)
THIRD_PARTY_MBEDTLS_A_INCS = $(filter %.inc,$(THIRD_PARTY_MBEDTLS_A_FILES))
THIRD_PARTY_MBEDTLS_A_HDRS = $(filter %.h,$(THIRD_PARTY_MBEDTLS_A_FILES))
THIRD_PARTY_MBEDTLS_A_SRCS = $(filter %.c,$(THIRD_PARTY_MBEDTLS_A_FILES))
THIRD_PARTY_MBEDTLS_A_CERTS := $(wildcard third_party/mbedtls/sslroot/*.pem)

THIRD_PARTY_MBEDTLS_A_OBJS =						\
	$(THIRD_PARTY_MBEDTLS_A_SRCS:%.c=o/$(MODE)/%.o)			\
	$(THIRD_PARTY_MBEDTLS_A_CERTS:%=o/$(MODE)/%.zip.o)		\

THIRD_PARTY_MBEDTLS_A_DEPS :=						\
	$(call uniq,$(foreach x,$(THIRD_PARTY_MBEDTLS_A_DIRECTDEPS),$($(x))))

$(THIRD_PARTY_MBEDTLS_A):						\
		third_party/mbedtls/					\
		$(THIRD_PARTY_MBEDTLS_A_OBJS)

$(THIRD_PARTY_MBEDTLS_A_OBJS): private					\
			CFLAGS +=					\
				-fdata-sections				\
				-ffunction-sections			\
				-mgcc

o/$(MODE)/third_party/mbedtls/everest.o: private			\
			CFLAGS +=					\
				-O3

o/$(MODE)/third_party/mbedtls/bigmul4.o					\
o/$(MODE)/third_party/mbedtls/bigmul6.o: private			\
			CFLAGS +=					\
				-O2

o/$(MODE)/third_party/mbedtls/shiftright-avx.o: private			\
			CFLAGS +=					\
				-O3 -Xx86_64-mavx

o/$(MODE)/third_party/mbedtls/zeroize.o: private			\
			CFLAGS +=					\
				-O3					\
				-fomit-frame-pointer			\
				-foptimize-sibling-calls

THIRD_PARTY_MBEDTLS_LIBS = $(foreach x,$(THIRD_PARTY_MBEDTLS_ARTIFACTS),$($(x)))
THIRD_PARTY_MBEDTLS_SRCS = $(foreach x,$(THIRD_PARTY_MBEDTLS_ARTIFACTS),$($(x)_SRCS))
THIRD_PARTY_MBEDTLS_HDRS = $(foreach x,$(THIRD_PARTY_MBEDTLS_ARTIFACTS),$($(x)_HDRS))
THIRD_PARTY_MBEDTLS_INCS = $(foreach x,$(THIRD_PARTY_MBEDTLS_ARTIFACTS),$($(x)_INCS))
THIRD_PARTY_MBEDTLS_CHECKS = $(foreach x,$(THIRD_PARTY_MBEDTLS_ARTIFACTS),$($(x)_CHECKS))
THIRD_PARTY_MBEDTLS_OBJS = $(foreach x,$(THIRD_PARTY_MBEDTLS_ARTIFACTS),$($(x)_OBJS))
$(THIRD_PARTY_MBEDTLS_A_OBJS): third_party/mbedtls/BUILD.mk

.PHONY: o/$(MODE)/third_party/mbedtls
o/$(MODE)/third_party/mbedtls:						\
		$(THIRD_PARTY_MBEDTLS_CHECKS)				\
		o/$(MODE)/third_party/mbedtls/mbedtls.a			\
