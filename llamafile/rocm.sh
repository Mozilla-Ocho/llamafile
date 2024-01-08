#!/bin/sh
# Compiles distributable DLL for NVIDIA GPU support
#
# The artifact will only depend on KERNEL32.DLL and NVCUDA.DLL.
# NVCUDA DLLs are provided by the installation of the windows GPU
# driver on a Windows system that has a CUDA-capable GPU installed.

hipcc \
  -O3 \
  -fPIC \
  -shared \
  -DNDEBUG \
  -march=native \
  -mtune=native \
  -use_fast_math \
  -DGGML_BUILD=1 \
  -DGGML_SHARED=1 \
  -Wno-unused-result \
  -DGGML_CUDA_DMMV_X=32 \
  -DGGML_CUDA_MMV_Y=1 \
  -DGGML_USE_HIPBLAS \
  -DK_QUANTS_PER_ITERATION=2 \
  -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
  -o ggml-rocm.so \
  ggml-cuda.cu \
  -lhipblas \
  -lrocblas
