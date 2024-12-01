/// <reference types="../../jamfile.d.ts"/>

import { test } from "./test.js";
import { assert, assertEquals, assertThrows } from "jamfile:assert";

import { TextEmbeddingModel } from "jamfile:llamafile";

function testLlamafile() {
  assertEquals(
    Object.getOwnPropertyNames(TextEmbeddingModel.prototype).sort(),
    ["constructor", "detokenize", "dimensions", "embed", "tokenize"],
  );

  const model = new TextEmbeddingModel("models/mxbai-embed-xsmall-v1-f16.gguf");

  assertEquals(model.dimensions, 384);

  assertThrows(() => {
    model.dimensions = 4;
  }, Error, "no setter for property");

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

export default function () {
  test("jamfile:llamafile", testLlamafile);
}
