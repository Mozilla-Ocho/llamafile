:: Compiles distributable DLL for AMD GPU support
::
:: The following microarchitectures are supported:
::
::   - gfx1010 c. 2019
::   - gfx1012 c. 2019
::   - gfx906  c. 2020
::   - gfx1032 c. 2021
::   - gfx1030 c. 2022
::   - gfx1031 c. 2022
::   - gfx1100 c. 2022
::   - gfx1101 (unreleased)
::   - gfx1102 (unreleased)
::   - gfx1103 (unreleased)
::
:: The ROCm SDK won't need to be installed on the user's machine.
:: There will be a dependency on AMDHIP64.DLL, but unlike hipBLAS
:: and rocBLAS, that DLL comes with the AMD graphics driver.
::
:: TODO(jart): How do we get this to not depend on VCRUNTIME140?

mkdir build\release 2>nul

for %%f in (
   llama.cpp\ggml-cuda.cu
   llama.cpp\ggml-cuda.h
   llama.cpp\ggml-impl.h
   llama.cpp\ggml-alloc.h
   llama.cpp\ggml-common.h
   llama.cpp\ggml-backend.h
   llama.cpp\ggml-backend-impl.h
   llama.cpp\ggml.h
   llamafile\tinyblas.h
   llamafile\tinyblas.cu
   llamafile\llamafile.h
) do copy %%f build\release

cd build\release

"%HIP_PATH_57%\bin\clang++.exe" ^
  -fuse-ld=lld ^
  -shared ^
  -nostartfiles ^
  -nostdlib ^
  -DGGML_BUILD=1 ^
  -DGGML_SHARED=1 ^
  -Wno-ignored-attributes ^
  -DGGML_CUDA_DMMV_X=32 ^
  -DGGML_CUDA_MMV_Y=1 ^
  -DGGML_USE_HIPBLAS ^
  -DGGML_USE_TINYBLAS ^
  -DGGML_MINIMIZE_CODE_SIZE ^
  -DK_QUANTS_PER_ITERATION=2 ^
  -D_CRT_SECURE_NO_WARNINGS ^
  -D_XOPEN_SOURCE=600 ^
  -D__HIP_PLATFORM_AMD__=1 ^
  -D__HIP_PLATFORM_HCC__=1 ^
  -isystem "%HIP_PATH_57%\include" ^
  -O2 ^
  -DNDEBUG ^
  -D_DLL ^
  -D_MT ^
  -Xclang --dependent-lib=msvcrt ^
  -std=gnu++14 ^
  -mllvm -amdgpu-early-inline-all=true ^
  -mllvm -amdgpu-function-calls=false ^
  -x hip ^
  --hip-link ^
  --offload-arch=gfx1010,gfx1012,gfx906,gfx1030,gfx1031,gfx1032,gfx1100,gfx1101,gfx1102,gfx1103 ^
  -o ggml-rocm.dll ^
  ggml-cuda.cu ^
  "-l%HIP_PATH_57%\lib\amdhip64.lib" ^
  -lkernel32
