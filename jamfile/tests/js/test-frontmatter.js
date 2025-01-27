/// <reference types="../../jamfile.d.ts"/>

import { test } from "./test.js";
import { assertEquals } from "jamfile:assert";
import { extractToml } from "jamfile:frontmatter";


function testFrontmatter() {
  assertEquals(
    extractToml('---toml\nname = "alex"\n---\n# markdown'),
    {
      frontMatter: 'name = "alex"',
      body: "# markdown",
      attrs: {name: "alex"}
    }
  );
}

export default function () {
  test("jamfile:frontmatter", testFrontmatter);
}
