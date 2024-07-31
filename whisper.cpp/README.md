# whisperfile

whisperfile lets you turn .wav files into .txt files,  
using a single file.

## Getting Started

You can use the tiny model. Here's JFK talking.

```
wget -O whisper-tiny.en-q5_1.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin
make -j o//whisper.cpp
o//whisper.cpp/main -m whisper-tiny.en-q5_1.bin -f whisper.cpp/jfk.wav --no-prints
```

Here's how you turn an Edgar Allen Poe MP3 into a WAV file.

```
sudo apt install sox libsox-fmt-all
wget https://archive.org/download/raven/raven_poe_64kb.mp3
sox raven_poe_64kb.mp3 -r 16k raven_poe_64kb.wav
```

Then you can use the large model, which is the best.

```
wget -O whisper-large-v3.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
o//whisper.cpp/main -m whisper-large-v3.bin -f raven_poe_64kb.wav --no-prints
```

## GPU Support

Pass the `--gpu auto` flag to use GPU mode. This is currently
experimental. It appears to be working with `whisper-tiny.en-q5_1.bin`
but isn't reliable yet for the F16 models.
