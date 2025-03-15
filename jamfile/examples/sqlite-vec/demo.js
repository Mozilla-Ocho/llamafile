import {open} from "qjs:std";
import {TextEmbeddingModel} from "jamfile:llamafile";
import {Database} from "jamfile:sqlite";
import {formatDuration} from "jamfile:fmt";
import {green} from "jamfile:colors";

const README = open('README.md', 'r').readAsString();

const db = new Database();
const model = new TextEmbeddingModel('dist/.models/mxbai-embed-xsmall-v1-f16.gguf');

db.execute(`
  CREATE VIRTUAL TABLE vec_chunks USING vec0(
    +contents TEXT,
    contents_embedding float[384] distance_metric=cosine
  );`);


function* chunks(arr, n) {
  for (let i = 0; i < arr.length; i += n) {
    yield arr.slice(i, i + n);
  }
}

const tokens = model.tokenize(README);

const t0 = Date.now();
const c = Array.from(chunks(tokens, 64));
for(const chunk of c) {
  const chunk_embedding = model.embed(chunk);
  db.execute(
    'INSERT INTO vec_chunks(contents, contents_embedding) VALUES (?, ?);',
    [model.detokenize(chunk), chunk_embedding]
  );
}
const duration = Date.now() - t0;

console.log(`${formatDuration(duration)} for ${c.length} embeddings (${c.length / (duration / 1000)}/second)`)

function search(query) {
  const rows = db.queryAll(
    `SELECT
      rowid, 
      contents 
    FROM vec_chunks 
    WHERE contents_embedding MATCH ? 
      AND k = 10
    `, 
    [model.embed(query)]
  );

  console.log(green(query));
  
  for (const {rowid, contents} of rows) {
    console.log(contents);
  }
}

search('linux CLI Tools ');
search('money supporting ecosystem ');
