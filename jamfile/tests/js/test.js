/// <reference types="../../jamfile.d.ts"/>

import { bold, green, red } from "jamfile:colors";
import { printf } from "qjs:std";

import testGlobals from './test-globals.js';
import testImports from './test-imports.js';
import testFmt from "./test-fmt.js";
import testZod from "./test-zod.js";
import testSqlite from "./test-sqlite.js";
import testSqliteVec from "./test-sqlite-vec.js";
import testLlamafile from "./test-llamafile.js";
import testYaml from "./test-yaml.js";
import testToml from "./test-toml.js";
import testFrontmatter from "./test-frontmatter.js";
import testMarked from "./test-marked.js";
import testLinkedom from "./test-linkedom.js";

export function test(name, fn) {
  try {
    printf(`Testing ${name}… `);
    fn();
    printf("✔︎\n");
  } catch (error) {
    printf(red("✗\n"));
    throw error;
  }
}

testGlobals();
testImports();

testFmt();
testZod();
testSqlite();
testLlamafile();
testSqliteVec();
testYaml();
testToml();
testFrontmatter();
testMarked();
testLinkedom();

console.log(green("✔︎"), bold("All tests passed!"));
