/// <reference types="../../jamfile.d.ts"/>

import { test } from "./test.js";
import { assert, assertEquals, assertThrows } from "jamfile:assert";

import { CompletionModel, TextEmbeddingModel } from "jamfile:llamafile";
import * as llamafile from "jamfile:llamafile";

function testLlamafile() {
  assertEquals(Object.getOwnPropertyNames(llamafile), [
    "CompletionModel",
    "TextEmbeddingModel",
  ]);
}

function testLlamafileTextEmbeddingModel() {
  assertEquals(
    Object.getOwnPropertyNames(TextEmbeddingModel.prototype).sort(),
    ["constructor", "detokenize", "dimensions", "embed", "tokenize"],
  );

  const model = new TextEmbeddingModel("models/mxbai-embed-xsmall-v1-f16.gguf");

  assertEquals(model.dimensions, 384);

  assertThrows(
    () => {
      model.dimensions = 4;
    },
    Error,
    "no setter for property",
  );

  const embedding = model.embed("hello!");

  assert(embedding instanceof Float32Array);
  assertEquals(embedding.length, 384);
  assertEquals(
    Array.from(embedding.slice(0, 4)),
    [
      -0.05826485529541969,
      0.04374322667717934,
      0.03084852732717991,
      0.047234565019607544,
    ],
  );

  const tokens = model.tokenize("hello!");

  assert(tokens instanceof Int32Array);
  assertEquals(tokens.length, 4);
  assertEquals(
    Array.from(tokens),
    [101, 7592, 999, 102],
  );

  const detokenized = model.detokenize(tokens);
  // TODO is this what we expect?
  assertEquals(detokenized, "[CLS] hello![SEP]]");
}

function testLlamafileCompletionModel() {
  assertEquals(
    Object.getOwnPropertyNames(CompletionModel.prototype).sort(),
    ["complete", "constructor", "description", "detokenize", "metadata", "nContext", "nLayer", "nParams", "size", "tokenize"],
  );

  const model = new CompletionModel("models/TinyLLama-v0.1-5M-F16.gguf");
  assertEquals(model.description, 'llama ?B F16');
  assertEquals(model.size, BigInt(9244928));
  assertEquals(model.nParams, BigInt(4621376));
  assertEquals(model.nContext, 512);
  assertEquals(model.nLayer, 8);
  assertEquals(
    model.metadata,
    new Map([
      ["tokenizer.ggml.padding_token_id", "0"],
      ["tokenizer.ggml.unknown_token_id", "0"],
      ["tokenizer.ggml.eos_token_id", "2"],
      ["tokenizer.ggml.bos_token_id", "1"],
      ["tokenizer.ggml.model", "llama"],
      ["llama.attention.head_count_kv", "16"],
      ["general.file_type", "1"],
      ["llama.feed_forward_length", "256"],
      ["llama.vocab_size", "32000"],
      ["llama.rope.dimension_count", "4"],
      [
        "general.description",
        "This gguf is ported from a first version of Maykeye attempt at recreating roneneldan/TinyStories-1M but using Llama architecture",
      ],
      ["llama.attention.layer_norm_rms_epsilon", "0.000001"],
      ["general.architecture", "llama"],
      ["general.author", "mofosyne"],
      ["general.version", "v0.1"],
      ["llama.embedding_length", "64"],
      ["llama.block_count", "8"],
      ["general.url", "https://huggingface.co/mofosyne/TinyLLama-v0-llamafile"],
      [
        "general.source.huggingface.repository",
        "https://huggingface.co/Maykeye/TinyLLama-v0",
      ],
      ["general.source.url", "https://huggingface.co/Maykeye/TinyLLama-v0"],
      ["llama.attention.head_count", "16"],
      ["llama.context_length", "2048"],
      ["general.name", "TinyLLama"],
    ]),
  );
  assertEquals(typeof model.complete("lol"), "string");
  assertEquals(model.tokenize("lol"), new Int32Array([1, 26953]));
  assertEquals(model.detokenize(new Int32Array([1, 26953])), " lol");
}

export default function () {
  test("jamfile:llamafile", testLlamafile);
  test("jamfile:llamafile TextEmbeddingModel", testLlamafileTextEmbeddingModel);
  test("jamfile:llamafile CompletionModel", testLlamafileCompletionModel);
}
