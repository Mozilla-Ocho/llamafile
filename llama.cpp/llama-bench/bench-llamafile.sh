#!/bin/sh
cd ~/llamafile
make -j16 o//llama.cpp/llama-bench/llama-bench || exit
o//llama.cpp/llama-bench/llama-bench \
  /weights/TinyLlama-1.1B-Chat-v1.0.*.gguf \
  /weights/mixtral-8x7b-instruct-v0.1.*.gguf \
  "$@"
