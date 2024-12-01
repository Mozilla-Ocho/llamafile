/// <reference types="./jamfile/jamfile.d.ts"/>

import { CompletionModel } from "jamfile:llamafile";
import schema from "./spellcheck.gbnf";

const model = new CompletionModel('gemma-2-2b-it.Q2_K.llamafile');

let prompt = (document) => `
<start_of_turn>user
You are a helpful AI assistant that does spell checking and only spell
checking. For example, if the user says:

It also meas that I keep my opshuns open.

Then the assistant will say:

- \`meas\` should be \`means\`
- \`opshuns\` should be \`options\`

If there are no mistyped words, the assistant responds with:

ok

Never reply with ok.<end_of_turn>
<start_of_turn>user
${document}<end_of_turn>
<start_of_turn>assistant
`;


import * as std from "qjs:std";

const document = std.in.readAsString();
const response = model.complete(prompt(document), { schema });

console.log(response);