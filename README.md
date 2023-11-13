# llamafile

llamafile lets you distribute LLMs as a single file.

Our goal is to make the "build once anywhere, run anywhere" dream come
true for AI developers. We're doing that by combining llama.cpp with
Cosmopolitan Libc into one framework that lets you build apps for LLMs
as a single-file artifact that runs locally on most PCs and servers.

First, your llamafiles can run on multiple CPU microarchitectures. We
added runtime dispatching to llama.cpp that lets new Intel systems use
modern CPU features without trading away support for older computers.

Secondly, your llamafiles can run on multiple CPU architectures. We do
that by concatenating AMD64 and ARM64 builds with a shell script that
launches the appropriate one. Our file format is compatible with WIN32
and most UNIX shells. It's also able to be easily converted (by either
you or your users) to the platform-native format, whenever required.

Thirdly, your llamafiles can run on six OSes. You'll only need to build
your code once, using a Linux-style toolchain. The GCC-based compiler we
provide is itself an Actually Portable Executable, so you can build your
software for all six OSes from the comfort of whichever one you prefer
most for development.

Lastly, the weights for your LLM can be embedded within your llamafile.
We added support for PKZIP to the GGML library. This lets uncompressed
weights be mapped directly into memory, similar to a self-extracting
archive. It enables quantized weights distributed online to be prefixed
with a compatible version of the llama.cpp software, thereby ensuring
its originally observed behaviors can be reproduced indefinitely.

## Build Instructions

First, you need the cosmocc toolchain, which is a fat portable binary
version of GCC. Here's how you can download the latest release and add
it to your path.

```sh
mkdir -p cosmocc
cd cosmocc
curl https://cosmo.zip/pub/cosmocc/cosmocc.zip >cosmocc.zip
unzip cosmocc.zip
cd ..
export PATH="$PWD/cosmocc/bin:$PATH"
```

You can now build the llamafile repository by running make:

```sh
make -j8
```

Here's an example of how to generate code for a libc function using the
llama.cpp command line interface using WizardCoder weights from Hugging
Face.

```sh
make -j8 o//llama.cpp/main/main
o//llama.cpp/main/main \
  -m ~/weights/wizardcoder-python-13b-v1.0.Q8_0.gguf \
  --temp 0 \
  -r $'```\n' \
  -p $'```c\nvoid *memcpy_sse2(char *dst, const char *src, size_t size) {\n'
```

Here's an example of how to run the HTTP server. This example includes
its own `.args` file which, by default, assumes
`llava-v1.5-7b-Q8_0.gguf` and `llava-v1.5-7b-mmproj-Q8_0.gguf` have been
added to either (1) the current directory, or (2) placed inside the
PKZIP file structure of the executable. It's important that weights be
stored without compression, so they can be mapped directly from the
executable into memory, without delay.

```sh
make -j8 o//llama.cpp/server/server
zip -0 o//llama.cpp/server/server \
  llava-v1.5-7b-Q8_0.gguf \
  llava-v1.5-7b-mmproj-Q8_0.gguf
o//llama.cpp/server/server
```

The above command will launch a browser tab on your personal computer to
display a web interface that lets you chat with your LLM and upload
images to it.
