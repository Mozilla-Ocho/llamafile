/// <reference types="../../jamfile.d.ts"/>

import { test } from "./test.js";
import { assertEquals } from "jamfile:assert";
import { stringify } from "jamfile:yaml";


function testYaml() {
  assertEquals(stringify({name: 'alex'}), 'name: alex\n');
}

export default function () {
  test("jamfile:yaml", testYaml);
}
