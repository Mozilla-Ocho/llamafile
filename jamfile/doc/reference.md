# Jamfile JavaScript Reference

A complete reference to all the JavaScript APIs available in the `Jamfile`
runtime.

## Built-In JavaScript Modules

| Module                                        | Description                                          | Source                                                      |
| --------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------- |
| [`jamfile:assert`](#jamfile_assert)           | Asserttion functions for testing and verification    | [`jsr:@std/assert`](https://jsr.io/@std/assert)             |
| [`jamfile:colors`](#jamfile_colors)           | Color utilites for terminal interfaces               | [`jsr:@std/colors`](https://jsr.io/@std/colors)             |
| [`jamfile:fmt`](#jamfile_fmt)                 | Format durations or bytes                            | [`jsr:@std/fmt`](https://jsr.io/@std/fmt)                   |
| [`jamfile:frontmatter`](#jamfile_frontmatter) | Parse frontmatter from JSON, YAML, or TOML documents | [`jsr:@std/frontmatter`](https://jsr.io/@std/frontmatter)   |
| [`jamfile:cli`](#jamfile_cli)                 | Command line parsing utilities                       | [`jsr:@std/cli`](https://jsr.io/@std/cli)                   |
| [`jamfile:llamafile`](#jamfile_llamafile)     | Embeddings and completion models                     | Custom built                                                |
| [`jamfile:linkedom`](#jamfile_linkedom)       | HTML and XML document parsing and generating         | [`npm:linkedom`](https://github.com/WebReflection/linkedom) |
| [`jamfile:marked`](#jamfile_marked)           | Markdown document parsing and generating             | [`npm:marked`](https://github.com/markedjs/marked)          |
| [`jamfile:sqlite`](#jamfile_sqlite)           | SQLite database client                               | Custom built                                                |
| [`jamfile:toml`](#jamfile_toml)               | Parse and generate TOML documents                    | [`jsr:@std/toml`](https://jsr.io/@std/toml)                 |
| [`jamfile:yaml`](#jamfile_yaml)               | Parse and generate YAML documents                    | [`jsr:@std/yaml`](https://jsr.io/@std/yaml)                 |
| [`jamfile:zod`](#jamfile_zod)                 | Schema validation                                    | [`npm:zod`](https://zod.dev/)                               |

<h3 name="jamfile_colors"><code>jamfile:colors</code></h3>

Utilities for stylizing text on the command-line with colors, bolding, italics,
and more. Re-packaged from
[`jsr:@std/fmt/colors`](https://jsr.io/@std/fmt/doc/colors).

```js
import { bgBlue, bold, red } from "jamafile:colors";

console.log(bgBlue(red(bold("Hello, World!"))));
```

<h3 name="jamfile_cli"><code>jamfile:cli</code></h3>

Utilies for parsing command lnke arguments. Re-packaged from
[`jsr:std/cli`](https://jsr.io/@std/cli), only the `parseArgs()` function for
now.

```js
import { parseArgs } from "jamfile:cli";
import { assertEquals } from "jamfile:assert";

const args = parseArgs(["--foo", "--bar=baz", "./quux.txt"]);
assertEquals(args, { "_": ["./quux.txt"], "foo": true, "bar": "baz" });
```

<h3 name="jamfile_zod"><code>jamfile:zod</code></h3>

The [zod library](https://zod.dev/) re-packaged into `jamfile` directly, no
`npm` needed. Also has
[`zodtoJsonSchema()`](https://github.com/StefanTerdell/zod-to-json-schema)
bundled in.

```js
import { z, zodToJsonSchema } from "jamfile:zod";

const Country = z.object({
  name: z.string(),
  capital: z.string(),
  languages: z.array(z.string()),
});

console.log(zodToJsonSchema(Country));
```

<h3 name="jamfile_fmt"><code>jamfile:fmt</code></h3>

Utilites for representing time durations and bytes into human-readable formats.
Re-packaged subset of [`jsr:@std/fmt`](https://jsr.io/@std/fmt), specifically
[`jsr:@std/fmt/bytes`](https://jsr.io/@std/fmt/doc/bytes) as `formatBytes()` and
[`jsr:@std/fmt/duration`](https://jsr.io/@std/fmt/doc/duration) as
`formatDuration()`.

```js
import { formatBytes, formatDuration } from "jamfile:fmt";
import { assertEquals } from "jamfile:assert";

assertEquals(formatBytes(3012), "3.01 kB");

assertEquals(
  formatDuration((60 * 1000) + (1000 * 32) + 4),
  "1m 32s 4ms",
);
```

`formatDuration()` has been modified from the original source, where the 2nd
parameter `options` has `ignoreZero=true` by default.

<h3 name="jamfile_assert"><code>jamfile:assert</code></h3>

Utilies for making assertions and verifying runtime data. Re-packaged from
[`jsr:@std/assert`](https://jsr.io/@std/assert).

```js
import { assert, assertArrayIncludes, assertEquals } from "jamfile:assert";

assert(true);
assertEquals(1 + 1, 2);
assertArrayIncludes([1, 2, 3], [2]);
```

<h3 name="jamfile_yaml"><code>jamfile:yaml</code></h3>
`

Utilies for parsing and stringifying YAML documents. Re-packaged from
[`jsr:@std/yaml`](https://jsr.io/@std/yaml).

```js
import { stringify } from "jamfile:yaml";
import { assertEquals } from "jamfile:assert";

assertEquals(stringify({ name: "alex" }), "name: alex\n");
```

<h3 name="jamfile_toml"><code>jamfile:toml</code></h3>

Utilies for parsing and stringifying YAML documents. Re-packaged from
[`jsr:@std/toml`](https://jsr.io/@std/toml).

```js
import { stringify } from "jamfile:toml";
import { assertEquals } from "jamfile:assert";

function testToml() {
  assertEquals(stringify({ name: "alex" }), 'name = "alex"\n');
}
```

<h3 name="jamfile_frontmatter"><code>jamfile:frontmatter</code></h3>

Utilies for parsing and testing JSON, YAML, or TOML frontmatter data from files,
like markdown. Re-packaged from
[`jsr:@std/front-matter`](https://jsr.io/@std/front-matter).

```js
import { extractToml } from "jamfile:frontmatter";
import { assertEquals } from "jamfile:assert";

assertEquals(
  extractToml('---toml\nname = "alex"\n---\n# markdown'),
  {
    frontMatter: 'name = "alex"',
    body: "# markdown",
    attrs: { name: "alex" },
  },
);
```

<h3 name="jamfile_marked"><code>jamfile:marked</code></h3>

The [`marked`](https://github.com/markedjs/marked) NPM package re-packaged for
`jamfile`. A good low-level primitive for parsing or generating markdown
documents.

```js
import { marked } from "jamfile:marked";
import { assertEquals } from "jamfile:assert";

assertEquals(marked.parse("# hello"), "<h1>hello</h1>\n");
```

<h3 name="jamfile_linkedom"><code>jamfile:linkedom</code></h3>

The [`linkedom`](https://github.com/WebReflection/linkedom) NPM package
re-packaged for `jamfile`. A good low-level primitive for parsing or generating
HTML or XML documents.

```js
import { parseHTML } from "jamfile:linkedom";
import { assertEquals } from "jamfile:assert";

const { document } = parseHTML('<div id="lol" class="Alex">yo</div>');

assertEquals(
  Array.from(document.querySelector("div").classList),
  ["Alex"],
);
```

<h3 name="jamfile_sqlite"><code>jamfile:sqlite</code></h3>

A custom SQLite JavaScript driver for the Jamfile runtime.

```js
import { Database } from "jamfile:sqlite";
import { assertEquals } from "jamfile:assert";

const db = new Database(":memory:");

db.executeScript(`
  CREATE TABLE items(id, title);

  INSERT INTO items 
    SELECT 
      key, 
      value 
  FROM json_each('["gemma", "llama", "phi"]');
`);

assertEquals(
  db.queryValue("select title from items where id = ?", [1]),
  "llama",
);
```

#### `SQLITE_VERSION`

A string representing the SQLite version bundled into Jamfile.

```js
import { SQLITE_VERSION } from "jamfile:sqlite";
import { assertEquals } from "jamfile:assert";

assertEquals(SQLITE_VERSION, "3.47.1");
```

#### `escapeIdentifier(identifier)`

```js
// TODO
```

#### `Database`

A JavaScript class that maintains a connection to a SQLite database.

```js
import { Database } from "jamfile:sqlite";
const db = new Database();
```

##### `new Database(path, [options])`

The constructor for the `Database` class will create a new connection to a
SQLite database on-disk or in-memory.

```js
```

##### `.queryAll(sql [, params])`

```js
```

##### `.queryRow(sql [, params])`

```js
```

##### `.queryValue(sql [, params])`

```js
```

##### `.execute(sql [, params])`

```js
```

##### `.executeScript(sql)`

```js
```

<!-- UNDOCUMENTED: updateHook() -->

<h3 name="jamfile_llamafile"><code>jamfile:llamafile</code></h3>

```js
```

#### `TextEmbeddingModel`

```js
```

##### `new TextEmbeddingModel()`

```js
```

##### `.tokenize()`

```js
```

##### `.embed()`

```js
```

##### `.detokenize()`

```js
```

#### `CompletionModel`

##### `new CompletionModel()`

##### `.complete(input [, options])`

##### `.tokenize(input)`

##### `.detokenize(input)`

### `qjs:std` and `qjs:os`

## The `Jamfile` Global

The `Jamfile` global variable is always available in `jamfile` scripts.

#### `Jamfile.version`

A string of the current "version" of Jamfile, may change in the future.

```js
console.log("Jamfile version: ", Jamfile.version);
```

#### `Jamfile.args`

A string array of the command-line arguments provided in the script run.

```js
// my-script.js
console.log(Jamfile.args);
```

```bash
jamfile run my-script.js gemma llama phi
gemma,llama,phi
```

Do note that `Jamfile.args` does not contain the "script" name as the first
argument. The first `Jamfile.args[0]` value is the first CLI argument provided
after the script filename.

Consider using `parseArgs()` inside [`jamfile:cli`](#jamfilecli) to parse these
CLI arguments in a friendly way.
