# Getting Started with LLaMAfiler

LLaMAfiler is a versatile tool that allows you to serve embeddings from
a wide range of models. While it's compatible with various architectures
like Mistral and TinyLLaMA, optimal performance is achieved with models
specifically designed for embeddings. This guide will walk you through
the process of setting up and using LLaMAfiler.

## Step 1: Download an Embedding Model

For this guide, we'll use the `all-MiniLM-L6-v2` model, which is
optimized for embedding tasks. Download it using the following command:

```bash
wget https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.F16.gguf
```

## Step 2: Install LLaMAfiler

Follow these steps to install the LLaMAfiler software:

```bash
git clone https://github.com/mozilla-ocho/llamafile
cd llamafile
make -j
sudo make install
```

## Step 3: Verify Installation

After installation, you can verify that LLaMAfiler is correctly installed by checking its version:

```bash
$ llamafiler --version
llamafiler v0.8.9
```

## Step 4: Launch the Server

To start the LLaMAfiler server on all IPv4 interfaces at port 8080, use the following command:

```bash
$ llamafiler -m all-MiniLM-L6-v2.F16.gguf -l 0.0.0.0:8080
```

You should see output similar to this:

```
2024-07-19T14:51:54.990853 llamafile/server/listen.cpp:33 server listen http://127.0.0.1:8080
2024-07-19T14:51:54.990859 llamafile/server/listen.cpp:33 server listen http://10.10.10.129:8080
```

## Step 5: Send Requests

You can now send requests to your LLaMAfiler server using curl. The API is compatible with the original llama.cpp server. Here's an example of how to generate an embedding for the word "orange":

```bash
curl -s -X POST -H "Content-Type: application/json" \
     --data '{"content": "orange"}' \
     http://localhost:8080/embedding
```

This command will return a JSON object containing the embedding vector for the word "orange".

## Additional Notes

- You can customize the port and interface by modifying the `-l` parameter in the server launch command.
- LLaMAfiler supports various model architectures. Experiment with different models to find the best fit for your use case.
- For production environments, consider setting up proper security measures, such as HTTPS and authentication.

Happy embedding with LLaMAfiler!

## See Also

- [LLaMAfiler Documentation Index](index.md)
- [LLaMAfiler Endpoints Reference](endpoints.md)
- [LLaMAfiler Technical Details](technical_details.md)
