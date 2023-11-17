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

## Binary Instructions

You can download and run `llamafile` from the release page in your
terminal or command prompt.

```sh
curl https://foo.example/llamafile >llamafile
chmod +x llamafile
./llamafile
```

On MacOS Arm64 you need to have xcode installed for llamafile to be able
to bootstrap itself.

On Windows, you may need to rename `llamafile` to `llamafile.exe` in
order for it to run.

If you use zsh and have trouble running llamafile, try saying `sh -c
./llamafile`. This is due to a bug that was fixed in zsh 5.9+. The same
is the case for Python `subprocess`, old versions of Fish, etc.

On Linux `binfmt_misc` has been known to cause problems. You can fix
that by installing the actually portable executable interpreter.

```sh
sudo wget -O /usr/bin/ape https://cosmo.zip/pub/cosmos/bin/ape-$(uname -m).elf
sudo sh -c "echo ':APE:M::MZqFpD::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
sudo sh -c "echo ':APE-jart:M::jartsr::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
```

### GPU Support

On Apple Arm64, everything should just work if xcode is installed.

On Linux, Nvidia cuBLAS GPU support will be compiled on the fly if (1)
you have the `cc` compiler installed, (2) you pass the `--n-gpu-layers
35` flag (or whatever value is appropriate) to enable GPU, and (3) the
CUDA developer toolkit is installed on your machine and the `nvcc`
compiler is on your path.

On Windows, that usually means you need to open up the MSVC x64 native
command prompt and run llamafile there, for the first invocation, so it
can build a DLl with native GPU support. After that, `$CUDA_PATH/bin`
still usually needs to be on the `$PATH` so the GGML DLL can find its
other CUDA dependencies.

In the event that GPU support couldn't be compiled and dynamically
linked on the fly for any reason, llamafile will fall back to CPU
inference.

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
its own [`.args` file](llama.cpp/server/.args) which, by default,
assumes `llava-v1.5-7b-Q8_0.gguf` and `llava-v1.5-7b-mmproj-Q8_0.gguf`
have been added to either (1) the current directory, or (2) placed
inside the PKZIP file structure of the executable. It's important that
you add large files to your zip executable archive using a tool we wrote
called `zipalign`. Our command goes 10x faster than the `zip` command.
It also inserts the weights without compression so they're aligned on a
page size boundary. That way, the metal GPU is able to map your weights
directly from the ZIP archive.

```sh
make -j8
o//llamafile/zipalign -j \
  o//llama.cpp/server/server \
  ~/weights/llava-v1.5-7b-Q8_0.gguf \
  ~/weights/llava-v1.5-7b-mmproj-Q8_0.gguf
o//llama.cpp/server/server
```

The above command will launch a browser tab on your personal computer to
display a web interface that lets you chat with your LLM and upload
images to it.
