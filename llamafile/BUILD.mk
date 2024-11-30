#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMAFILE

LLAMAFILE_FILES := $(wildcard llamafile/*.*)
LLAMAFILE_HDRS = $(filter %.h,$(LLAMAFILE_FILES))
LLAMAFILE_INCS = $(filter %.inc,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS_C = $(filter %.c,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS_CU = $(filter %.cu,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS_CPP = $(filter %.cpp,$(LLAMAFILE_FILES))
LLAMAFILE_SRCS = $(LLAMAFILE_SRCS_C) $(LLAMAFILE_SRCS_CPP) $(LLAMAFILE_SRCS_CU)
LLAMAFILE_DOCS = $(filter %.1,$(LLAMAFILE_FILES))

LLAMAFILE_OBJS :=					\
	$(LLAMAFILE_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(LLAMAFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.o)	\
	$(LLAMAFILE_FILES:%=o/$(MODE)/%.zip.o)		\

$(LLAMAFILE_OBJS): private CCFLAGS += -g

# this executable defines its own malloc(), free(), etc.
# therefore we want to avoid it going inside the .a file
LLAMAFILE_OBJS := $(filter-out o/$(MODE)/llamafile/zipalign.o,$(LLAMAFILE_OBJS))

include llamafile/highlight/BUILD.mk
include llamafile/server/BUILD.mk

o/$(MODE)/llamafile/zipalign:				\
		o/$(MODE)/llamafile/zipalign.o		\
		o/$(MODE)/llamafile/help.o		\
		o/$(MODE)/llamafile/has.o		\
		o/$(MODE)/llamafile/zipalign.1.asc.zip.o

o/$(MODE)/llamafile/zipcheck:				\
		o/$(MODE)/llamafile/zipcheck.o		\
		o/$(MODE)/llamafile/zip.o		\

o/$(MODE)/llamafile/simple:				\
		o/$(MODE)/llamafile/simple.o		\
		o/$(MODE)/llama.cpp/llama.cpp.a		\

o/$(MODE)/llamafile/tokenize:				\
		o/$(MODE)/llamafile/tokenize.o		\
		o/$(MODE)/llama.cpp/llama.cpp.a		\

o/$(MODE)/llamafile/curl:					\
		o/$(MODE)/llamafile/curl.o			\
		o/$(MODE)/llama.cpp/llama.cpp.a			\
		o/$(MODE)/third_party/mbedtls/mbedtls.a		\

.PHONY: o/$(MODE)/llamafile
o/$(MODE)/llamafile:						\
		$(LLAMAFILE_OBJS)				\
		o/$(MODE)/llamafile/server			\
		o/$(MODE)/llamafile/simple			\
		o/$(MODE)/llamafile/zipalign			\
		o/$(MODE)/llamafile/zipcheck			\
		o/$(MODE)/llamafile/tokenize			\
		o/$(MODE)/llamafile/addnl			\
		o/$(MODE)/llamafile/high			\
		o/$(MODE)/llamafile/datauri_test.runs		\
		o/$(MODE)/llamafile/parse_cidr_test.runs	\
		o/$(MODE)/llamafile/pool_cancel_test.runs	\
		o/$(MODE)/llamafile/pool_test.runs		\
		o/$(MODE)/llamafile/json_test.runs		\
		o/$(MODE)/llamafile/thread_test.runs		\
		o/$(MODE)/llamafile/vmathf_test.runs		\

################################################################################
# microarchitectures
#
#### Intel CPU Line
#
# - 2006 core           X64 SSE4.1 (only on 45nm variety) (-march=core2)
# - 2008 nehalem        SSE4.2 VT-x VT-d RDTSCP POPCNT (-march=nehalem)
# - 2010 westmere       CLMUL AES (-march=westmere)
# - 2011 sandybridge    AVX TXT (-march=sandybridge)
# - 2012 ivybridge      F16C MOVBE FSGSBASE (-march=ivybridge)
# - 2013 haswell        AVX2 TSX BMI1 BMI2 FMA (-march=haswell)
# - 2014 broadwell      RDSEED ADX PREFETCHW (-march=broadwell)
# - 2015 skylake        SGX ADX MPX AVX-512[xeon-only] (-march=skylake / -march=skylake-avx512)
# - 2018 cannonlake     SHA (-march=cannonlake)
# - 2019 cascadelake    VNNI
# - 2021 alderlake      efficiency cores
#
#### AMD CPU Line
#
# - 2003 k8             SSE SSE2 (-march=k8)
# - 2005 k8 (Venus)     SSE3 (-march=k8-sse3)
# - 2008 barcelona      SSE4a?! (-march=barcelona)
# - 2011 bulldozer      SSSE3 SSE4.1 SSE4.2 CLMUL AVX AES FMA4?! (-march=bdver1)
# - 2011 piledriver     BMI1 FMA (-march=bdver2)
# - 2015 excavator      AVX2 BMI2 MOVBE (-march=bdver4)
# - 2017 ryzen          F16C SHA ADX (-march=znver1)
# - 2023 zen4           AVX512F AVX512VL AVX512VNNI AVX512BF16 (-march=znver4)
#
#### ARM Revisions
#
# - armv8.0-a           raspberry pi 4
# - armv8.2-a           raspberry pi 5
# - armv8.5-a           apple m1
# - armv8.6-a           apple m2
#
#### ARM Features
#
# - HWCAP_CRC32         +crc32    (e.g. m1, rpi4)  __ARM_FEATRUE_CRC32
# - HWCAP_FPHP          +fp16     (e.g. m1, rpi5)  __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
# - HWCAP_ASIMDHP       +fp16     (e.g. m1, rpi5)  __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
# - HWCAP_ASIMDDP       +dotprod  (e.g. m1, rpi5)  __ARM_FEATURE_DOTPROD
#

o/$(MODE)/llamafile/iqk_mul_mat_amd_avx2.o: private TARGET_ARCH += -Xx86_64-mtune=skylake -Xx86_64-mavx -Xx86_64-mavx2 -Xx86_64-mfma -Xx86_64-mf16c
o/$(MODE)/llamafile/iqk_mul_mat_amd_zen4.o: private TARGET_ARCH += -Xx86_64-mtune=skylake -Xx86_64-mavx -Xx86_64-mavx2 -Xx86_64-mfma -Xx86_64-mf16c -Xx86_64-mavx512f -Xx86_64-mavx512vl -Xx86_64-mavx512vnni -Xx86_64-mavx512bw -Xx86_64-mavx512dq
o/$(MODE)/llamafile/iqk_mul_mat_arm82.o: private TARGET_ARCH += -Xaarch64-march=armv8.2-a+dotprod+fp16
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx.o: private TARGET_ARCH += -Xx86_64-mtune=sandybridge -Xx86_64-mavx -Xx86_64-mf16c
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx.o: private TARGET_ARCH += -Xx86_64-mtune=sandybridge -Xx86_64-mavx -Xx86_64-mf16c
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_fma.o: private TARGET_ARCH += -Xx86_64-mtune=bdver2 -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_fma.o: private TARGET_ARCH += -Xx86_64-mtune=bdver2 -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx2.o: private TARGET_ARCH += -Xx86_64-mtune=skylake -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mfma
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx2.o: private TARGET_ARCH += -Xx86_64-mtune=skylake -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mfma
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avxvnni.o: private TARGET_ARCH += -Xx86_64-mtune=alderlake -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mavxvnni
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avxvnni.o: private TARGET_ARCH += -Xx86_64-mtune=alderlake -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mavxvnni
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx512f.o: private TARGET_ARCH += -Xx86_64-mtune=cannonlake -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mavx512f
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx512f.o: private TARGET_ARCH += -Xx86_64-mtune=cannonlake -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mavx512f
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_zen4.o: private TARGET_ARCH += -Xx86_64-mtune=znver4 -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mavx512f -Xx86_64-mavx512vl -Xx86_64-mavx512vnni -Xx86_64-mavx512bf16
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_zen4.o: private TARGET_ARCH += -Xx86_64-mtune=znver4 -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mavx512f -Xx86_64-mavx512vl -Xx86_64-mavx512vnni -Xx86_64-mavx512bf16
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_arm82.o: private TARGET_ARCH += -Xaarch64-march=armv8.2-a+dotprod+fp16
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_arm82.o: private TARGET_ARCH += -Xaarch64-march=armv8.2-a+dotprod+fp16

o/$(MODE)/llamafile/sgemm.o: private CXXFLAGS += -Os

o/$(MODE)/llamafile/sgemm_matmul_test.o			\
o/$(MODE)/llamafile/sgemm_sss_test.o			\
o/$(MODE)/llamafile/sgemm_vecdot_test.o			\
o/$(MODE)/llamafile/iqk_mul_mat_amd_avx2.o		\
o/$(MODE)/llamafile/iqk_mul_mat_amd_zen4.o		\
o/$(MODE)/llamafile/iqk_mul_mat_arm82.o			\
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx2.o	\
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx512f.o	\
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx.o	\
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avxvnni.o	\
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_fma.o	\
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_zen4.o	\
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_arm80.o		\
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_arm82.o		\
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx2.o	\
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx512f.o	\
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx.o	\
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avxvnni.o	\
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_fma.o	\
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_zen4.o	\
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_arm80.o		\
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_arm82.o:		\
		private CCFLAGS += -O3 -fopenmp -mgcc

################################################################################
# testing

o/$(MODE)/llamafile/json_test:						\
		o/$(MODE)/llamafile/json_test.o				\
		o/$(MODE)/llamafile/json.o				\
		o/$(MODE)/llamafile/hextoint.o				\
		o/$(MODE)/double-conversion/double-conversion.a		\

o/$(MODE)/llamafile/vmathf_test:			\
		o/$(MODE)/llamafile/vmathf_test.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\

o/$(MODE)/llamafile/parse_cidr_test:			\
		o/$(MODE)/llamafile/parse_cidr_test.o	\
		o/$(MODE)/llamafile/parse_cidr.o	\
		o/$(MODE)/llamafile/parse_ip.o		\

o/$(MODE)/llamafile/pool_test:				\
		o/$(MODE)/llamafile/pool_test.o		\
		o/$(MODE)/llamafile/crash.o		\
		o/$(MODE)/llamafile/pool.o		\

o/$(MODE)/llamafile/datauri_test:			\
		o/$(MODE)/llamafile/datauri_test.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\
		o/$(MODE)/third_party/stb/stb.a		\

o/$(MODE)/llamafile/high:					\
		o/$(MODE)/llamafile/high.o			\
		o/$(MODE)/llamafile/highlight/highlight.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a			\

o/$(MODE)/llamafile/hex2xterm:				\
		o/$(MODE)/llamafile/hex2xterm.o		\
		o/$(MODE)/llamafile/xterm.o		\

o/$(MODE)/llamafile/pool_cancel_test:			\
		o/$(MODE)/llamafile/pool_cancel_test.o	\
		o/$(MODE)/llamafile/crash.o		\
		o/$(MODE)/llamafile/pool.o		\

o/$(MODE)/llamafile/thread_test:			\
		o/$(MODE)/llamafile/thread_test.o	\
		o/$(MODE)/llamafile/crash.o		\
		o/$(MODE)/llamafile/dll3.o		\

o/$(MODE)/llamafile/sgemm_sss_test: private LDFLAGS += -fopenmp
o/$(MODE)/llamafile/sgemm_sss_test.o: private CCFLAGS += -fopenmp
o/$(MODE)/llamafile/sgemm_matmul_test: private LDFLAGS += -fopenmp
o/$(MODE)/llamafile/sgemm_matmul_test.o: private CCFLAGS += -fopenmp

o/$(MODE)/llamafile/sgemm_sss_test:			\
		o/$(MODE)/llamafile/sgemm_sss_test.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llamafile/sgemm_matmul_test:			\
		o/$(MODE)/llamafile/sgemm_matmul_test.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llamafile/sgemm_vecdot_test:			\
		o/$(MODE)/llamafile/sgemm_vecdot_test.o	\
		o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llamafile/sgemm_vecdot_test:			\
		private LDFLAGS += -fopenmp

o/$(MODE)/llamafile/%.o: llamafile/%.cu llamafile/BUILD.mk
	@mkdir -p $(@D)
	build/cudacc -fPIE -g -O3 -march=native -ffast-math --use_fast_math -c -o $@ $<

o/$(MODE)/llamafile/tinyblas_test:			\
		o/$(MODE)/llamafile/tinyblas_test.o	\
		o/$(MODE)/llamafile/tinyblas.o		\
		o/$(MODE)/llamafile/tester.o
	build/cudacc -g -o $@ $^ -lcublas

o/$(MODE)/llamafile/compcap:				\
		o/$(MODE)/llamafile/compcap.o
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
