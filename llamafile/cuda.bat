:: Compiles distributable DLL for NVIDIA GPU support
::
:: The artifact will only depend on KERNEL32.DLL and NVCUDA.DLL.
:: NVCUDA DLLs are provided by the installation of the windows GPU
:: driver on a Windows system that has a CUDA-capable GPU installed.

nvcc --shared ^
     -arch=all-major ^
     --forward-unknown-to-host-compiler ^
     -Xcompiler="/nologo /EHsc /O2 /GR /MT" ^
     -DNDEBUG ^
     -DGGML_BUILD=1 ^
     -DGGML_SHARED=1 ^
     -DGGML_CUDA_MMV_Y=1 ^
     -DGGML_MULTIPLATFORM ^
     -DGGML_CUDA_DMMV_X=32 ^
     -DK_QUANTS_PER_ITERATION=2 ^
     -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 ^
     -DGGML_MINIMIZE_CODE_SIZE ^
     -DGGML_USE_TINYBLAS ^
     -o ggml-cuda.dll ^
     ggml-cuda.cu ^
     -lcuda
