# llamafile

![llamafile-250](llamafile/llamafile.png)

**llamafile lets you distribute and run LLMs with a single file. ([announcement blog post](https://hacks.mozilla.org/2023/11/introducing-llamafile/))**

Our goal is to make open source large language models much more 
accessible to both developers and end users. We're doing that by 
combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one 
framework that collapses all the complexity of LLMs down to 
a single-file executable (called a "llamafile") that runs
locally on most computers, with no installation.

## Quickstart

The easiest way to try it for yourself is to download our example 
llamafile for the [LLaVA](https://llava-vl.github.io/) model (license: [LLaMA 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/), 
[OpenAI](https://openai.com/policies/terms-of-use)). LLaVA is a new LLM that can do more 
than just chat; you can also upload images and ask it questions 
about them. With llamafile, this all happens locally; no data 
ever leaves your computer.

1. Download [llava-v1.5-7b-q4-server.llamafile](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-q4-server.llamafile?download=true) (3.97 GB).

2. Open your computer's terminal.

3. If you're using macOS, Linux, or BSD, you'll need to grant permission 
for your computer to execute this new file. (You only need to do this 
once.)

```sh
chmod +x llava-v1.5-7b-q4-server.llamafile
```

4. If you're on Windows, rename the file by adding ".exe" on the end.

5. Run the llamafile. e.g.:

```sh
./llava-v1.5-7b-q4-server.llamafile
```

6. Your browser should open automatically and display a chat interface. 
(If it doesn't, just open your browser and point it at https://localhost:8080.)

7. When you're done chatting, return to your terminal and hit 
```Control-C``` to shut down llamafile.

**Having trouble? See the "Gotchas" section below.**

## Other example llamafiles

We also provide example llamafiles for two other models, so you can 
easily try out llamafile with different kinds of LLMs.

| Model | License | Command-line llamafile | Server llamafile |
| --- | --- | --- | --- |
| Mistral-7B-Instruct | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) | [mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile (4.07 GB)](https://huggingface.co/jartine/mistral-7b.llamafile/resolve/main/mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile?download=true) | [mistral-7b-instruct-v0.1-Q4_K_M-server.llamafile (4.07 GB)](https://huggingface.co/jartine/mistral-7b.llamafile/resolve/main/mistral-7b-instruct-v0.1-Q4_K_M-server.llamafile?download=true) |
| LLaVA 1.5 | [LLaMA 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) | (Not provided because this model's features are best utilized via the web UI) | **[llava-v1.5-7b-q4-server.llamafile (3.97 GB)](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-q4-server.llamafile?download=true)** |
| WizardCoder-Python-13B | [LLaMA 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) | [wizardcoder-python-13b-main.llamafile (7.33 GB)](https://huggingface.co/jartine/wizardcoder-13b-python/resolve/main/wizardcoder-python-13b-main.llamafile?download=true) | [wizardcoder-python-13b-server.llamafile (7.33GB)](https://huggingface.co/jartine/wizardcoder-13b-python/resolve/main/wizardcoder-python-13b-server.llamafile?download=true) |

"Server llamafiles" work just like the LLaVA example above: you simply run 
them from your terminal and then access the chat UI in your web browser at 
https://localhost:8080. 

"Command-line llamafiles" run entirely inside your terminal and 
operate just like  llama.cpp's "main" function. This means you 
have to provide some command-line parameters, just like with 
llama.cpp.

Here is an example for the Mistral command-line llamafile:

```sh
./mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile --temp 0.7 -r '\n' -p '### Instruction: Write a story about llamas\n### Response:\n'
```

And here is an example for WizardCoder-Python command-line llamafile:

```sh
./wizardcoder-python-13b-main.llamafile --temp 0 -r '\n' -p '\nvoid *memcpy_sse2(char *dst, const char *src, size_t size) {\n'
```

As before, macOS, Linux, and BSD users will need to use the "chmod" 
command to grant execution permissions to the file before running 
these llamafiles for the first time.

Unfortunately, Windows users cannot make use of these example llamafiles 
because Windows has a maximum executable file size of 4GB, and all of 
these examples exceed that size. (The LLaVA llamafile works on Windows because it is 30MB shy of the size limit.) But don't lose heart: llamafile allows 
you to use external weights; this is described later in this document.

**Having trouble? See the "Gotchas" section below.**

## How llamafile works

A llamafile is an executable LLM that you can run on your own 
computer. It contains the weights for a given open source LLM, as well 
as everything needed to actually run that model on your computer. 
There's nothing to install or configure (with a few caveats, discussed 
in subsequent sections of this document).

This is all accomplished by combining llama.cpp with Cosmopolitan Libc, 
which provides some useful capabilities:

1. llamafiles can run on multiple CPU microarchitectures. We
added runtime dispatching to llama.cpp that lets new Intel systems use
modern CPU features without trading away support for older computers.

2. llamafiles can run on multiple CPU architectures. We do
that by concatenating AMD64 and ARM64 builds with a shell script that
launches the appropriate one. Our file format is compatible with WIN32
and most UNIX shells. It's also able to be easily converted (by either
you or your users) to the platform-native format, whenever required.

3. llamafiles can run on six OSes (macOS, Windows, Linux,
FreeBSD, OpenBSD, and NetBSD). If you make your own llama files, you'll 
only need to build your code once, using a Linux-style toolchain. The 
GCC-based compiler we provide is itself an Actually Portable Executable, 
so you can build your software for all six OSes from the comfort of 
whichever one you prefer most for development.

4. The weights for an LLM can be embedded within the llamafile.
We added support for PKZIP to the GGML library. This lets uncompressed
weights be mapped directly into memory, similar to a self-extracting
archive. It enables quantized weights distributed online to be prefixed
with a compatible version of the llama.cpp software, thereby ensuring
its originally observed behaviors can be reproduced indefinitely.

5. Finally, with the tools included in this project you can create your 
*own* llamafiles, using any compatible model weights you want. You can 
then distribute these llamafiles to other people, who can easily make 
use of them regardless of what kind of computer they have.

## Using llamafile with external weights

Even though our example llamafiles have the weights built-in, you don't 
*have* to use llamafile that way. Instead, you can download *just* the 
llamafile software (without any weights included) from our releases page. 
You can then use it alongside any external weights you may have on hand. 
External weights are particularly useful for Windows users because they 
enable you to work around Windows' 4GB executable file size limit. 

For Windows users, here's an example for the Mistral LLM:

```sh
curl -o llamafile.exe https://github.com/Mozilla-Ocho/llamafile/releases/download/0.2.1/llamafile-server-0.2.1
curl -o mistral.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
.\llamafile.exe -m mistral.gguf
```

Here's the same example, but for macOS, Linux, and BSD users:

```sh
curl -L https://github.com/Mozilla-Ocho/llamafile/releases/download/0.2.1/llamafile-server-0.2.1 >llamafile
curl -L https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf >mistral.gguf
chmod +x llamafile
./llamafile -m mistral.gguf
```



## Gotchas

On macOS with Apple Silicon you need to have Xcode installed for
llamafile to be able to bootstrap itself.

If you use zsh and have trouble running llamafile, try saying `sh -c
./llamafile`. This is due to a bug that was fixed in zsh 5.9+. The same
is the case for Python `subprocess`, old versions of Fish, etc.

On some Linux systems, you might get errors relating to `run-detectors`
or WINE. This is due to `binfmt_misc` registrations. You can fix that by
adding an additional registration for the APE file format llamafile
uses:

```sh
sudo wget -O /usr/bin/ape https://cosmo.zip/pub/cosmos/bin/ape-$(uname -m).elf
sudo chmod +x /usr/bin/ape
sudo sh -c "echo ':APE:M::MZqFpD::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
sudo sh -c "echo ':APE-jart:M::jartsr::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
```

As mentioned above, on Windows you may need to rename your llamafile by 
adding `.exe` to the filename. 

Also as mentioned above, Windows also has a maximum file size limit of 4GB
for executables. The LLaVA server executable above is just 30MB shy of
that limit, so it'll work on Windows, but with larger models like
WizardCoder 13B, you need to store the weights in a separate file. An 
example is provided above; see "Using llamafile with external weights."

On WSL, it's recommended that the WIN32 interop feature be disabled:

```sh
sudo sh -c "echo -1 > /proc/sys/fs/binfmt_misc/WSLInterop"
```

On any platform, if your llamafile process is immediately killed, check
if you have CrowdStrike and then ask to be whitelisted.

## GPU support

On Apple Silicon, everything should just work if Xcode is installed.

On Linux, Nvidia cuBLAS GPU support will be compiled on the fly if (1)
you have the `cc` compiler installed, (2) you pass the `--n-gpu-layers
35` flag (or whatever value is appropriate) to enable GPU, and (3) the
CUDA developer toolkit is installed on your machine and the `nvcc`
compiler is on your path.

On Windows, that usually means you need to open up the MSVC x64 native
command prompt and run llamafile there, for the first invocation, so it
can build a DLL with native GPU support. After that, `$CUDA_PATH/bin`
still usually needs to be on the `$PATH` so the GGML DLL can find its
other CUDA dependencies.

In the event that GPU support couldn't be compiled and dynamically
linked on the fly for any reason, llamafile will fall back to CPU
inference.

## Source instructions

Here's how to build llamafile from source. First, you need the cosmocc
toolchain, which is a fat portable binary version of GCC. Here's how you
can download the latest release and add it to your path.

```sh
mkdir -p cosmocc
cd cosmocc
curl -L https://github.com/jart/cosmopolitan/releases/download/3.1.3/cosmocc-3.1.3.zip >cosmocc.zip
unzip cosmocc.zip
cd ..
export PATH="$PWD/cosmocc/bin:$PATH"
```

You can now build the llamafile repository by running make:

```sh
make -j8
```

Here's an example of how to generate code for a libc function using the
llama.cpp command line interface, utilizing WizardCoder-Python-13B weights:

```sh
make -j8 o//llama.cpp/main/main
o//llama.cpp/main/main \
  -m ~/weights/wizardcoder-python-13b-v1.0.Q8_0.gguf \
  --temp 0 \
  -r $'```\n' \
  -p $'```c\nvoid *memcpy_sse2(char *dst, const char *src, size_t size) {\n'
```

Here's a similar example that instead utilizes Mistral-7B-Instruct weights:

```sh
make -j8 o//llama.cpp/main/main
o//llama.cpp/main/main \
  -m ~/weights/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  --temp 0.7 \
  -r $'\n' \
  -p $'### Instruction: Write a story about llamas\n### Response:\n'
```

Here's an example of how to run llama.cpp's built-in HTTP server in such
a way that the weights are embedded inside the executable. This example
uses LLaVA v1.5-7B, a multimodal LLM that works with llama.cpp's 
recently-added support for image inputs.

```sh
make -j8

o//llamafile/zipalign -j0 \
  o//llama.cpp/server/server \
  ~/weights/llava-v1.5-7b-Q8_0.gguf \
  ~/weights/llava-v1.5-7b-mmproj-Q8_0.gguf

o//llama.cpp/server/server \
  -m llava-v1.5-7b-Q8_0.gguf \
  --mmproj llava-v1.5-7b-mmproj-Q8_0.gguf \
  --host 0.0.0.0
```

The above command will launch a browser tab on your personal computer to
display a web interface. It lets you chat with your LLM and upload
images to it.

If you want to be able to just say:

```
./server
```

...and have it run the web server without having to specify arguments (for
the paths you already know are in there), then you can add a special
`.args` to the zip archive, which specifies the default arguments. In
this case, we're going to try our luck with the normal `zip` command,
which requires we temporarily rename the file. First, let's create the
arguments file:

```
cat <<EOF >.args
-m
llava-v1.5-7b-Q8_0.gguf
--mmproj
llava-v1.5-7b-mmproj-Q8_0.gguf
--host
0.0.0.0
...
EOF
```

As we can see above, there's one argument per line. The `...` argument
optionally specifies where any additional CLI arguments passed by the
user are to be inserted. Next, we'll add the argument file to the
executable:

```
mv o//llama.cpp/server/server server.com
zip server.com .args
mv server.com server
./server
```

Congratulations. You've just made your own LLM executable that's easy to
share with your friends.

## Security

llamafile adds pledge() and SECCOMP sandboxing to llama.cpp. This is
enabled by default. It can be turned off by passing the `--unsecure`
flag. Sandboxing is currently only supported on Linux and OpenBSD; on
other platforms it'll simply log a warning.

Our approach to security has these benefits:

1. After it starts up, your HTTP server isn't able to access the
   filesystem at all. This is good, since it means if someone discovers
   a bug in the llama.cpp server, then it's much less likely they'll be
   able to access sensitive information on your machine or make changes
   to its configuration. On Linux, we're able to sandbox things even
   further; the only networking related system call the HTTP server will
   allowed to use after starting up, is accept(). That further limits an
   attacker's ability to exfiltrate information, in the event that your
   HTTP server is compromised.

2. The main CLI command won't be able to access the network at all. This
   is enforced by the operating system kernel. It also won't be able to
   write to the file system. This keeps your computer safe in the event
   that a bug is ever discovered in the the GGUF file format that lets
   an attacker craft malicious weights files and post them online. The
   only exception to this rule is if you pass the `--prompt-cache` flag
   without also specifying `--prompt-cache-ro`. In that case, security
   currently needs to be weakened to allow `cpath` and `wpath` access,
   but network access will remain forbidden.

Therefore your llamafile is able to protect itself against the outside
world, but that doesn't mean you're protected from llamafile. Sandboxing
is self-imposed. If you obtained your llamafile from an untrusted source
then its author could have simply modified it to not do that. In that
case, you can run the untrusted llamafile inside another sandbox, such
as a virtual machine, to make sure it behaves how you expect.

## zipalign documentation

```
SYNOPSIS

  o//llamafile/zipalign ZIP FILE...

DESCRIPTION

  Adds aligned uncompressed files to PKZIP archive

  This tool is designed to concatenate gigabytes of LLM weights to an
  executable. This command goes 10x faster than `zip -j0`. Unlike zip
  you are not required to use the .com file extension for it to work.
  But most importantly, this tool has a flag that lets you insert zip
  files that are aligned on a specific boundary. The result is things
  like GPUs that have specific memory alignment requirements will now
  be able to perform math directly on the zip file's mmap()'d weights

FLAGS

  -h        help
  -N        nondeterministic mode
  -a INT    alignment (default 65536)
  -j        strip directory components
  -0        store uncompressed (currently default)
```

## Technical details

Here is a succinct overview of the tricks we used to create the fattest
executable format ever. The long story short is llamafile is a shell
script that launches itself and runs inference on embedded weights in
milliseconds without needing to be copied or installed. What makes that
possible is mmap(). Both the llama.cpp executable and the weights are
concatenated onto the shell script. A tiny loader program is then
extracted by the shell script, which maps the executable into memory.
The llama.cpp executable then opens the shell script again as a file,
and calls mmap() again to pull the weights into memory and make them
directly accessible to both the CPU and GPU.

### ZIP weights embedding

The trick to embedding weights inside llama.cpp executables is to ensure
the local file is aligned on a page size boundary. That way, assuming
the zip file is uncompressed, once it's mmap()'d into memory we can pass
pointers directly to GPUs like Apple Metal, which require that data be
page size aligned. Since no existing ZIP archiving tool has an alignment
flag, we had to write about [400 lines of code](llamafile/zipalign.c) to
insert the ZIP files ourselves. However, once there, every existing ZIP
program should be able to read them, provided they support ZIP64. This
makes the weights much more easily accessible than they otherwise would
have been, had we invented our own file format for concatenated files.

### Microarchitectural portability

On Intel and AMD microprocessors, llama.cpp spends most of its time in
the matmul quants, which are usually written thrice for SSSE3, AVX, and
AVX2. llamafile pulls each of these functions out into a separate file
that can be `#include`ed multiple times, with varying
`__attribute__((__target__("arch")))` function attributes. Then, a
wrapper function is added which uses Cosmopolitan's `X86_HAVE(FOO)`
feature to runtime dispatch to the appropriate implementation.

### Architecture portability

llamafile solves architecture portability by building llama.cpp twice:
once for AMD64 and again for ARM64. It then wraps them with a shell
script which has an MZ prefix. On Windows, it'll run as a native binary.
On Linux, it'll extract a small 8kb executable called [APE
Loader](https://github.com/jart/cosmopolitan/blob/master/ape/loader.c)
to `${TMPDIR:-${HOME:-.}}/.ape` that'll map the binary portions of the
shell script into memory. It's possible to avoid this process by running
the
[`assimilate`](https://github.com/jart/cosmopolitan/blob/master/tool/build/assimilate.c)
program that comes included with the `cosmocc` compiler. What the
`assimilate` program does is turn the shell script executable into
the host platform's native executable format. This guarantees a fallback
path exists for traditional release processes when it's needed.

### GPU support

Cosmopolitan Libc uses static linking, since that's the only way to get
the same executable to run on six OSes. This presents a challenge for
llama.cpp, because it's not possible to statically link GPU support. The
way we solve that is by checking if a compiler is installed on the host
system. For Apple, that would be Xcode, and for other platforms, that
would be `nvcc`. llama.cpp has a single file implementation of each GPU
module, named `ggml-metal.m` (Objective C) and `ggml-cuda.cu` (Nvidia
C). llamafile embeds those source files within the zip archive and asks
the platform compiler to build them at runtime, targeting the native GPU
microarchitecture. If it works, then it's linked with platform C library
dlopen() implementation. See [llamafile/cuda.c](llamafile/cuda.c) and
[llamafile/metal.c](llamafile/metal.c).

In order to use the platform-specific dlopen() function, we need to ask
the platform-specific compiler to build a small executable that exposes
these interfaces. On ELF platforms, Cosmopolitan Libc maps this helper
executable into memory along with the platform's ELF interpreter. The
platform C library then takes care of linking all the GPU libraries, and
then runs the helper program which longjmp()'s back into Cosmopolitan.
The executable program is now in a weird hybrid state where two separate
C libraries exist which have different ABIs. For example, thread local
storage works differently on each operating system, and programs will
crash if the TLS register doesn't point to the appropriate memory. The
way Cosmopolitan Libc solves that is by JITing a trampoline around each
dlsym() import, which blocks signals using `sigprocmask()` and changes
the TLS register using `arch_prctl()`. Under normal circumstances,
aspecting each function call with four additional system calls would be
prohibitively expensive, but for llama.cpp that cost is infinitesimal
compared to the amount of compute used for LLM inference. Our technique
has no noticeable slowdown. The major tradeoff is that, right now, you
can't pass callback pointers to the dlopen()'d module. Only one such
function needed to be removed from the llama.cpp codebase, which was an
API intended for customizing logging. In the future, Cosmoplitan will
just trampoline signal handlers and code morph the TLS instructions to
avoid these tradeoffs entirely. See
[cosmopolitan/dlopen.c](https://github.com/jart/cosmopolitan/blob/master/libc/dlopen/dlopen.c)
for further details.

## A note about models

The example llamafiles provided above should not be interpreted as 
endorsements or recommendations of specific models, licenses, or data 
sets on the part of Mozilla.

## Licensing

While the llamafile project is Apache 2.0-licensed, our changes
to llama.cpp are licensed under MIT (just like the llama.cpp project
itself) so as to remain compatible and upstreamable in the future,
should that be desired.

The llamafile logo on this page was generated with the assistance of DALL·E 3.
