#!/bin/sh
# prints number of llamafile_sgemm kernels

echo $(( $(aarch64-unknown-cosmo-nm \
             -C o//llama.cpp/main/main.aarch64.elf |
             grep GEMMER.*::gemm |
             wc -l) +
         $(x86_64-unknown-cosmo-nm \
             -C o//llama.cpp/main/main.com.dbg |
             grep GEMMER.*::gemm |
             wc -l) ))
