# `Jamfile` Documentation

Jamfile is an experimental JavaScript runtime for llamafile, for working with large language models and embeddings models.

This page goes over a quick overview of all that Jamfile has to offer. Additionally, check out these other documentation pages for more examples and goal-oriented guides.

- [Jamfile JavaScript Reference](./reference.md)
- [Jamfile for generating and querying embeddings](./embeddings.md)
- [Jamfile for structured output generation](./structured-outputs.md)
- [Compiling Jamfile programs into a "jam"](./jam.md)

## Jamfile isn't a complete JavaScript runtime yet!

Jamfile isn't like other JavaScript runtimes like Node.js, Deno, or Bun. 
It's scope is much smaller, there isn't plans for Node.js compatability, and many "standard" features like `fetch()`, `Temporal`, or other browser APIs are not supported yet. 

However, if you want to have a lighweight scripting engine for working with LLM's across platforms, then Jamfile is great!


## Getting Started

### Working with LLM's

You can work with large language models that are in the [GGUF file format](https://huggingface.co/docs/hub/en/gguf) with Jamfile like so:


```js
// llm-sample.js
import { CompletionModel } from "jamfile:llamafile";

const model = new CompletionModel("./Llama-3.2-1B-Instruct-Q4_0.gguf");
const prompt = "The meaning of life is:";
const response = model.complete(prompt);
console.log(prompt, response);
```


```bash
jamfile run llm-sample.js
The meaning of life is:  
...a complex and multifaceted concept that has been debated and explored by philosophers, scientists, theologians, and anyone who has ever set out to understand the mysteries of existence.
```

You can swap out the the `path` to any GGUF file of any supported LLM. 

The `.complete()` function also supports structured output generation, to force an LLM to respond in a specified schema. For example, you can have a model only respond in "math" with the [GBNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md):

```js
// gbnf-sample.js
import { CompletionModel } from "jamfile:llamafile";

const model = new CompletionModel("./Llama-3.2-1B-Instruct-Q4_0.gguf");

const prompt = `
Convert this transcription of a math problem into a numerical expression:

Seven hundred and thirty four minus sixty-five divided by fourty five. 
`;

// source: https://github.com/ggerganov/llama.cpp/blob/master/grammars/arithmetic.gbnf
const schema = `root  ::= (expr "=" ws term "\\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \\t\\n]*\`'`;

const response = model.complete(prompt, {schema});
console.log(prompt, response);
```

Or if you want the LLM to respond with an exact JSON Schema, use the [builtin `jamfile:zod` module](./reference.md#jamfile_zod):


```js
// zod-sample.js
import { z, zodToJsonSchema } from "jamfile:zod";
import { CompletionModel } from "jamfile:llamafile";

const Country = z.object({
  name: z.string(),
  capital: z.string(),
  languages: z.array(z.string()),
});

const prompt = `Describe the country of Canada in JSON format:`;
const schema = zodToJsonSchema(Country);

const model = new CompletionModel("./Llama-3.2-1B-Instruct-Q4_0.gguf");
const response = model.complete(prompt, { schema });

console.log(response);
```


```
> jamfile run zod-sample.js 
{"name": "Canada", "capital": "Ottawa", "languages": ["English", "French"]}
```
### Working with embeddings

Jamfile can also generate embeddings with [`TextEmbeddingModel`](./reference.md#TextEmbeddingModel).

```js
import { TextEmbeddingModel } from "jamfile:llamafile";

const model = new TextEmbeddingModel("./mxbai-embed-xsmall-v1-f16.gguf");
const embedding = model.embed("hello!");
console.log(
  `${embedding.length} dimensions, preview: ${embedding.slice(0, 4)}`,
);
```


```
jamfile run embedding-sample.js                 
384 dimensions, preview: -0.05826485529541969,0.04374322667717934,0.03084852732717991,0.047234565019607544
```


Use the [`jamfile:sqlite`](./reference.md#jamfile_sqlite) module to store and query these embeddings with [`sqlite-vec`](https://github.com/asg017/sqlite-vec).

```js
import { TextEmbeddingModel } from "jamfile:llamafile";
import { Database } from "jamfile:sqlite";

// Random NPR news headlines from 2024-01-17
const documents = [
  "The Supreme Court upholds a TikTok ban, threatening the app's existence in the U.S.",
  "CNN settles lawsuit after $5 million defamation verdict",
  "Immigrants drive Nebraska's economy. Trump's mass deportations pledge is a threat",
  "President-elect Donald Trump moves his inauguration indoors, citing frigid temperatures",
];

const model = new TextEmbeddingModel("./mxbai-embed-xsmall-v1-f16.gguf");
const db = new Database(":memory:");

db.execute(`
  CREATE VIRTUAL TABLE vec_documents USING vec0(
    contents_embedding FLOAT[384] distance_metric=cosine,
    +contents TEXT,
  );
`);

for (const document of documents) {
  const embedding = model.embed(document);
  db.execute(
    "INSERT INTO vec_documents(contents_embedding, contents) VALUES(?, ?)",
    [embedding, document],
  );
}

// KNN query the documents on 'social media'
const results = db.queryAll(
  `
    SELECT
      contents,
      distance
    FROM vec_documents
    WHERE contents_embedding MATCH ?
      AND k = 3;
  `,
  [model.embed("social media")],
);

for (const row of results) {
  console.log(row["contents"], row["distance"]);
}
```

```
jamfile run embedding-sample.js
The Supreme Court upholds a TikTok ban, threatening the app's existence in the U.S. 0.8698796033859253
CNN settles lawsuit after $5 million defamation verdict 0.9893361926078796
President-elect Donald Trump moves his inauguration indoors, citing frigid temperatures 0.9990869164466858
```

### Create CLI scripts

- `Jamfile.args` and `jamfile:cli`
- json/yaml/toml for config, or `config.js`

### "Compile" a script to a standalone executable

- `zipalign` a simple JS script
- include embedding, completion model
- `esbuild` for more

## `Jamfile` Runtime

### Built-In JavaScript Modules

| Module                                                      | Description                                          |
| ----------------------------------------------------------- | ---------------------------------------------------- |
| [`jamfile:assert`](./reference.md#jamfile_assert)           | Assertion functions for testing and verification     |
| [`jamfile:colors`](./reference.md#jamfile_colors)           | Color utilites for terminal interfaces               |
| [`jamfile:fmt`](./reference.md#jamfile_fmt)                 | Format durations or bytes                            |
| [`jamfile:frontmatter`](./reference.md#jamfile_frontmatter) | Parse frontmatter from JSON, YAML, or TOML documents |
| [`jamfile:cli`](./reference.md#jamfile_cli)                 | Command line parsing utilities                       |
| [`jamfile:llamafile`](./reference.md#jamfile_llamafile)     | Embeddings and completion models                     |
| [`jamfile:linkedom`](./reference.md#jamfile_linkedom)       | HTML and XML document parsing and generating         |
| [`jamfile:marked`](./reference.md#jamfile_marked)           | Markdown document parsing and generating             |
| [`jamfile:sqlite`](./reference.md#jamfile_sqlite)           | SQLite database client                               |
| [`jamfile:toml`](./reference.md#jamfile_toml)               | Parse and generate TOML documents                    |
| [`jamfile:yaml`](./reference.md#jamfile_yaml)               | Parse and generate YAML documents                    |
| [`jamfile:zod`](./reference.md#jamfile_zod)                 | Schema validation                                    |

### "Importing" `.sql`, `gbnf`, or `.txt` files

In addition to importing local JavaScript files, you can directly import SQL,
[GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md),
or plain text files as strings.

For example, to import a `schema.sql` file that contains SQL table schema, you can do so like so:

```js
import SCHEMA from "./schema.sql";
import { Database } from "jamfile:sqlite";

const db = new Database();

db.executeScript(SCHEMA);
db.execute("INSERT INTO my_table(created_at) VALUES(?)", [new Date()]);
```

Where `schema.sql` contains:

```sql
-- schema.sql
CREATE TABLE IF NOT EXISTS my_table(
  created_at datetime
);
```

This also works for
[GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)
files, which are used to define formal grammars to constraint the output of an
LLM.

```js
import schema from "./spellcheck.gbnf";
import { CompletionModel } from "jamfile:llamafile";

const model = new CompletionModel("gemma-2-2b-it.Q2_K.llamafile");

const response = model.complete("You are a ...", { schema });
console.log(response);
```

See [Structured Outputs in `Jamfile`](./structured-outputs.md) for more details.

You can also import plain `.txt` files as strings, for things like system
prompts or other templates.

```js
import PROMPT from "./SYSTEM_PROMPT.txt";
import { CompletionModel } from "jamfile:llamafile";

const model = new CompletionModel("gemma-2-2b-it.Q2_K.llamafile");

const response = model.complete(PROMPT);
console.log(response);
```

## The `jamfile` CLI

### `jamfile run`

```bash
```

#### The `--default-text-embedding-model` flag

#### The `--default-completion-model` flag

### `jamfile types`

The `types` command will print out the contents of `jamfile.d.ts`, a TypeScript declaration file that documents every available JavaScript API in Jamfile's [standard library](./reference.md#built-in-javascript-modules). You can redirect the stream into a `jamfile.d.ts` file like so:

```bash
jamfile types > jamfile.d.ts
```

Then in your JavaScript code, you can "reference" this declartion file with the ["types" triple slash directive](https://www.typescriptlang.org/docs/handbook/triple-slash-directives.html#-reference-types-):

```js
/// <reference types="./jamfile.d.ts"/>
import { TextEmbeddingModel } from "jamfile:llamafile";
const model = new TextEmbeddingModel("./mxbai-embed-xsmall-v1-f16.gguf");
model.embed('contents...');
```

In most IDEs, the inclusion of `jamfile.d.ts` will allow for autocomplete on Jamfile's JavaScript import statements, function names, parameter types, and more. Think of this as a "no-dependency" way to get autocomplete in your Jamfile scripts, no TypeScript or build step needed.
