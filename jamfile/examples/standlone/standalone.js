/**
 * An example of a standalone jamfile. 
 * 
 * This script performs 
 */

/// <reference types="../../jamfile.d.ts"/>

import { red } from "jamfile:color";
import { CompletionModel, TextEmbeddingModel } from "jamfile:llamafile";

const embeddingModel = new TextEmbeddingModel();
const completionModel = new CompletionModel();

console.log(
  red("embedding sample: "), 
  JSON.stringify(Array.from(embeddingModel.embed("hello!").slice(0, 8)))
);

console.log(
  red("completion sample: "), 
  completionModel.complete("write a single haiku about spongebob squarepants")
);