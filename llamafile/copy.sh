#!/bin/sh
set -x
HOST=${1:?HOST}

ssh $HOST mkdir -p lfbuild
scp llama.cpp/ggml-cuda.cu \
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
    $HOST:lfbuild/
