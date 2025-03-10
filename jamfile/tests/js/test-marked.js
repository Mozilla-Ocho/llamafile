/// <reference types="../../jamfile.d.ts"/>

import { test } from "./test.js";
import { assertEquals } from "jamfile:assert";
import {marked} from "jamfile:marked";


function testMarked() {
  assertEquals(marked.parse('# hello'), '<h1>hello</h1>\n');
}

export default function () {
  test("jamfile:marked", testMarked);
}
