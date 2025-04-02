#!/bin/sh

TMP=$(mktemp -d) || exit

cp llama.cpp/ggml-cuda.cu \
   llama.cpp/ggml-cuda.h \
   llama.cpp/ggml-impl.h \
   llama.cpp/ggml-alloc.h \
   llama.cpp/ggml-common.h \
   llama.cpp/ggml-backend.h \
   llama.cpp/ggml-backend-impl.h \
   llama.cpp/ggml.h \
   llamafile/tinyblas.h \
   llamafile/tinyblas.cu \
   llamafile/llamafile.h \
   llamafile/rocm.bat \
   llamafile/rocm.sh \
   llamafile/cuda.bat \
   llamafile/cuda.sh \
   "$TMP" || exit

cd "$TMP"

/usr/local/cuda/bin/nvcc \
  --shared \
  --use_fast_math \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  --forward-unknown-to-host-compiler \
  --compiler-options "-fPIC -O2" \
  -DNDEBUG \
  -DGGML_BUILD=1 \
  -DGGML_SHARED=1 \
  -DGGML_CUDA_MMV_Y=1 \
  -DGGML_MULTIPLATFORM \
  -DGGML_CUDA_DMMV_X=32 \
  -DK_QUANTS_PER_ITERATION=2 \
  -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
  -DGGML_MINIMIZE_CODE_SIZE \
  -DGGML_USE_CUBLAS \
  -o ~/ggml-cuda.localscore.so \
  ggml-cuda.cu \
  -lcublas \
  -lcuda
