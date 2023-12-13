call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

nvcc -arch=all ^
     --shared ^
     --forward-unknown-to-host-compiler ^
     -Xcompiler="/nologo /EHsc /O2 /GR /MT" ^
     -use_fast_math ^
     -DNDEBUG ^
     -DGGML_BUILD=1 ^
     -DGGML_SHARED=1 ^
     -DGGML_CUDA_DMMV_X=32 ^
     -DGGML_CUDA_MMV_Y=1 ^
     -DK_QUANTS_PER_ITERATION=2 ^
     -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 ^
     -DGGML_USE_TINYBLAS ^
     -o ggml-cuda.dll ^
     ggml-cuda.cu
