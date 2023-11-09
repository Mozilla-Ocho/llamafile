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
