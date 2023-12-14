#!/bin/sh

/usr/local/cuda/bin/nvcc \
    -arch=all \
    --shared \
    --forward-unknown-to-host-compiler \
    -use_fast_math \
    --compiler-options "-fPIC -O3 -march=native -mtune=native" \
    -DNDEBUG \
    -DGGML_BUILD=1 \
    -DGGML_SHARED=1 \
    -DGGML_CUDA_DMMV_X=32 \
    -DGGML_CUDA_MMV_Y=1 \
    -DK_QUANTS_PER_ITERATION=2 \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DGGML_USE_TINYBLAS \
    -o ggml-cuda.so \
    ggml-cuda.cu
