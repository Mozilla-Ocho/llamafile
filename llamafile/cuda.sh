#!/bin/sh
set -x
HOST=${1:?HOST}

ssh $HOST mkdir -p lfbuild
scp llama.cpp/ggml-cuda.cu \
    llama.cpp/ggml-cuda.h \
    llama.cpp/tinyblas.cu \
    llama.cpp/tinyblas.h \
    llama.cpp/ggml.h \
    llamafile/llamafile.h \
    llamafile/cuda.bat \
    $HOST:lfbuild/

# TODO(jart): make windows tools able to do automation somehow
# ssh $HOST 'cd lfbuild; C:/WINDOWS/system32/cmd.exe /k cuda.bat'
# mkdir -p o//llama.cpp
# scp $HOST:lfbuild/ggml-cuda.dll o//llama.cpp/
